//! The fused Metal llama-layer kernel on the production decode path (tickets #87,
//! #110).
//!
//! `src/metal/fused_layer.rs` carries the resident chain: `MetalWeights` (fused
//! RMSNorm + QKV + RoPE GEMVs), `ResidentLayer` (the same prologue plus the
//! O-projection, FFN-norm and SwiGLU-MLP weights) and the GEMV/encode entries.
//! Since ticket #87 the prologue is what produces Q/K/V in production. This
//! module is the caller, and since ticket #110 it calls the **whole decoder layer
//! sweep** on one command buffer per step: per layer, the fused prologue, a blit
//! that publishes the fresh token's K/V into the paged-attention pool copy,
//! PagedAttention v2 (through the #109 encode seam), the O projection + residual
//! add, and the SwiGLU MLP + residual add — committed once at the end of the
//! step. That pays the debt `docs/adr/0002-fused-decode-scope.md` recorded
//! against #87 (one buffer per layer prologue plus separately-submitted
//! attention and tail). The final norm and lm_head stay on the tensor path, per
//! that ADR's scope.
//!
//! Three rules keep the wiring honest:
//!
//! * the geometry gate ([`fused_layer_step_supported`]) runs *before* anything is
//!   uploaded, so a model the kernels cannot represent keeps the pre-existing
//!   path — a fallback, never an error and never a silent wrong-shape dispatch;
//! * the prologue reads the model's own rotary tables (`rope_cos_sin`), so the
//!   fused RoPE cannot drift from `rope_apply`;
//! * the step never creates a command buffer inside the layer sweep — the single
//!   submission is the property `Model::metal_decode_command_buffers` measures.

use crate::error::EngineError;
use crate::metal::context::MetalContext;
use crate::metal::fused_layer::{
    FullLayerWeights, FusedLayerDimensions, FusedLayerWeights, FusedQkvTriple, ResidentLayer,
    read_f32, write_f32,
};
use crate::metal::paged_attention::encode_paged_attention_v2;
use crate::model::PagedAttnContext;
use crate::model::paged_attn::{AttnShape, PagedAttnSeam};
use candle_core::{Device, Tensor};
use metal::Buffer;

/// One model's fused decode seam, built the first time a dense Metal decode step runs.
pub(crate) struct FusedDecodeSeam {
    ctx: MetalContext,
    dims: FusedLayerDimensions,
    layers: Vec<ResidentLayer>,
    /// Largest position the resident rotary tables cover.
    max_pos: usize,
    steps: u64,
    /// The step's residual stream, `[hidden_dim]`: the host stages the embedding
    /// row before the step, layer 0's prologue reads it, and every residual add
    /// (inside the step's single command buffer) updates it in place.
    hidden: Buffer,
}

/// What a completed one-command-buffer step hands back to the caller (ticket #110):
/// the final residual-stream row — the input of the tensor-path final norm and
/// lm_head — plus each layer's K/V rows for the KV-pool writeback.
pub(crate) struct StepRows {
    pub hidden: Vec<f32>,
    pub k: Vec<Vec<f32>>,
    pub v: Vec<Vec<f32>>,
}

impl FusedDecodeSeam {
    /// Builds the seam, or `Ok(None)` when this model is outside the kernel geometry.
    ///
    /// The geometry check comes first so a model the kernels cannot serve uploads
    /// nothing: it keeps the pre-existing path and no number changes.
    pub(crate) fn build(model: &super::Model) -> Result<Option<Self>, EngineError> {
        let dims = FusedLayerDimensions {
            hidden_dim: model.config.hidden_size,
            num_q_heads: model.config.num_attention_heads,
            num_kv_heads: model.config.num_key_value_heads,
            head_dim: model.config.head_dim(),
        };
        let max_pos = model.config.max_seq_len.saturating_sub(1);
        let table_len = (max_pos + 1) * (dims.head_dim / 2).max(1);
        let intermediate = model.config.intermediate_size;
        if !fused_layer_step_supported(&dims, table_len, intermediate) {
            return Ok(None);
        }

        // The rotary tables are the model's own, read back once, so the fused RoPE and
        // `rope_apply` cannot drift apart.
        let positions: Vec<usize> = (0..=max_pos).collect();
        let (cos, sin) = model.rope_cos_sin(&positions)?;
        let cos = host_row(&cos)?;
        let sin = host_row(&sin)?;

        let ctx = MetalContext::new()?;
        let mut layers = Vec::with_capacity(model.weights.layers.len());
        for (index, layer) in model.weights.layers.iter().enumerate() {
            let weights = read_layer_weights(model, index, layer, &dims)?;
            layers.push(ResidentLayer::upload(
                &ctx,
                &dims,
                intermediate,
                &weights,
                &cos,
                &sin,
            )?);
        }

        Ok(Some(Self {
            hidden: ctx.alloc_buffer::<f32>(dims.hidden_dim),
            ctx,
            dims,
            layers,
            max_pos,
            steps: 0,
        }))
    }

    /// Largest position the resident rotary tables cover.
    pub(crate) fn max_pos(&self) -> usize {
        self.max_pos
    }

    /// Decode steps' layer sweeps this seam has served (one increment per layer,
    /// whether the layer ran through [`Self::decode_step`] or the one-buffer step).
    pub(crate) fn steps(&self) -> u64 {
        self.steps
    }

    /// Command buffers this seam submitted: exactly one per one-buffer decode step
    /// since ticket #110 — the layer sweep chains into a single submission.
    pub(crate) fn command_buffers(&self) -> usize {
        self.ctx.command_buffers_created()
    }

    /// Runs one layer's fused RMSNorm + QKV + RoPE prologue and returns the host
    /// Q/K/V rows. The per-layer submit strategy the paths that cannot chain a
    /// whole step keep (hybrid FFN offload, a position past the resident tables).
    ///
    /// The rows are already RoPE-rotated in the rotate-half layout `rope_apply`
    /// produces, so the caller must not rotate them a second time.
    pub(crate) fn prologue(
        &mut self,
        layer: usize,
        hidden: &[f32],
        pos: usize,
        eps: f32,
    ) -> Result<FusedQkvTriple, EngineError> {
        if pos > self.max_pos {
            return Err(EngineError::Metal(format!(
                "fused decode: pos {pos} exceeds the resident rotary tables (max {})",
                self.max_pos
            )));
        }
        // Direct field lookup, so the `steps` counter (a different field) can be
        // bumped while the returned layer reference is still held.
        let resident = self.layers.get(layer).ok_or_else(|| {
            EngineError::Metal(format!(
                "fused decode: layer {layer} outside the {} resident layers",
                self.layers.len()
            ))
        })?;
        self.steps += 1;
        resident
            .as_metal_weights()
            .decode_step(&self.ctx, hidden, pos, eps)
    }

    /// One whole-layer-sweep decode step on a **single command buffer** (ticket
    /// #110): per layer, fused prologue → blit the fresh K/V row into the paged
    /// pool copy → PagedAttention v2 → O projection + residual add → SwiGLU MLP +
    /// residual add; committed once and waited once at the end of the sweep.
    ///
    /// `paged` is the step's block geometry (one sequence: `slots` holds the row
    /// this step's token writes), `embedding_row` the input hidden row, and
    /// `residual_scale` the Granite `residual_scale` multiplier (1.0 on plain
    /// llama) — the same factor the tensor path applies via `scale_residual`.
    ///
    /// Returns the final residual row plus every layer's K/V rows; the caller
    /// writes the K/V back into the KV pool (the pool itself is updated after the
    /// step, since the GPU published this step's rows into the seam's device copy
    /// directly) and runs the final norm + lm_head on the tensor path.
    #[allow(clippy::too_many_arguments)] // step surface: one per seam input
    pub(crate) fn run_step(
        &mut self,
        paged_seam: &mut PagedAttnSeam,
        kv: &crate::kv_cache::KvStorage,
        paged: &PagedAttnContext,
        embedding_row: &[f32],
        pos: usize,
        eps: f32,
        residual_scale: f32,
        shape: &AttnShape,
    ) -> Result<StepRows, EngineError> {
        if pos > self.max_pos {
            return Err(EngineError::Metal(format!(
                "fused decode: pos {pos} exceeds the resident rotary tables (max {})",
                self.max_pos
            )));
        }
        if paged.slots.len() != 1 {
            return Err(EngineError::Metal(format!(
                "fused decode: the one-command-buffer step serves one sequence, got {}",
                paged.slots.len()
            )));
        }
        if embedding_row.len() != self.dims.hidden_dim {
            return Err(EngineError::Metal(format!(
                "fused decode: embedding row holds {} floats, expected {}",
                embedding_row.len(),
                self.dims.hidden_dim
            )));
        }
        let slot = paged.slots[0];
        let rows = kv.num_blocks * kv.block_size;
        let row_elems = kv.n_kv_heads * kv.head_dim;

        write_f32(&self.hidden, embedding_row);
        // The single submission the whole step spends (Model::metal_decode_command_buffers
        // counts it through this context).
        let command_buffer = self.ctx.new_command_buffer();

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // 1. Fused RMSNorm + QKV + RoPE, reading the residual stream in place.
            let encoder = command_buffer.new_compute_command_encoder();
            layer.encode_prologue(&self.ctx, encoder, &self.hidden, pos, eps)?;
            encoder.end_encoding();

            // 2. Publish the fresh K/V row into the pool copy attention reads. The
            // blit runs after prepare's (host-side) refresh — a whole-pool upload
            // would replace the buffer — and before the attention dispatch;
            // encoders on one command buffer execute in encode order.
            let prepared = paged_seam.prepare_encode(
                layer.q_out(),
                kv,
                layer_idx,
                &paged.block_tables,
                paged.max_blocks,
                &paged.context_lens,
                &paged.slots,
                &paged.recycled_blocks,
                shape,
            )?;
            let blit = command_buffer.new_blit_command_encoder();
            paged_seam.blit_slot_row(
                blit,
                layer_idx,
                slot,
                layer.k_out(),
                layer.v_out(),
                rows,
                row_elems,
            )?;
            blit.end_encoding();

            // 3. PagedAttention v2 through the #109 encode seam: same kernels, same
            // buffers, the step's command buffer instead of a self-committed one.
            encode_paged_attention_v2(
                paged_seam.ctx(),
                command_buffer,
                layer.q_out(),
                &prepared.k,
                &prepared.v,
                &prepared.tables,
                &prepared.lens,
                &prepared.out,
                &prepared.params,
            )?;

            // 4. O projection + residual add, SwiGLU MLP + residual add.
            layer.encode_tail(
                &self.ctx,
                command_buffer,
                &prepared.out,
                &self.hidden,
                residual_scale,
                eps,
                &self.dims,
            )?;
            self.steps += 1;
        }

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(StepRows {
            hidden: read_f32(&self.hidden, self.dims.hidden_dim),
            k: self.layers.iter().map(ResidentLayer::read_k).collect(),
            v: self.layers.iter().map(ResidentLayer::read_v).collect(),
        })
    }
}

/// Host-side gate for the one-command-buffer decode step (tickets #87 / #110).
///
/// [`FusedLayerDimensions::validate`] covers the prologue's float4 lane reads and
/// RoPE pairs; [`fused_prologue_supported`] adds the row-pair mapping. This gate
/// adds what the chained tail needs: an intermediate size the down-projection
/// GEMV can read in float4 lanes and the gate/up GEMVs can cover in row pairs,
/// and a query dimension the O-projection GEMV can read in float4 lanes.
pub fn fused_layer_step_supported(
    dims: &FusedLayerDimensions,
    table_len: usize,
    intermediate: usize,
) -> bool {
    fused_prologue_supported(dims, table_len)
        && intermediate > 0
        && intermediate.is_multiple_of(4)
        && dims.q_dim().is_multiple_of(4)
}

/// Host-side gate for the fused prologue (ticket #87).
///
/// `true` means the kernel can index every row it is given: `FusedLayerDimensions::validate`
/// covers the float4 lane reads (`hidden_dim % 4`) and the RoPE pair (`head_dim` even), and
/// the two extra clauses cover what the kernel's row-pair mapping assumes — a projection
/// whose rows are whole RoPE heads, and an even row count (one SIMD group owns a row pair).
pub fn fused_prologue_supported(dims: &FusedLayerDimensions, table_len: usize) -> bool {
    dims.validate().is_ok()
        && dims.q_dim().is_multiple_of(2)
        && dims.kv_dim().is_multiple_of(2)
        && dims.q_dim().is_multiple_of(dims.head_dim)
        && dims.kv_dim().is_multiple_of(dims.head_dim)
        && table_len >= dims.head_dim / 2
}

/// Host tensor into a flat host row, wherever it lives.
fn host_row(tensor: &Tensor) -> Result<Vec<f32>, EngineError> {
    Ok(tensor
        .to_device(&Device::Cpu)?
        .contiguous()?
        .flatten_all()?
        .to_vec1::<f32>()?)
}

/// Copies one layer's dense weights to the host, applying the GGUF Q/K row
/// permutation so the fused RoPE sees rotate-half rows — the same permutation
/// `project_qk` applies to the projection output.
///
/// Returned by value: the buffers are uploaded once and then dropped, so an owned
/// [`FullLayerWeights`] is the whole story — a wrapper plus a cloning `view()` would
/// copy every weight twice at build for nothing.
fn read_layer_weights(
    model: &super::Model,
    index: usize,
    layer: &crate::weights::LayerWeights,
    dims: &FusedLayerDimensions,
) -> Result<FullLayerWeights, EngineError> {
    let hidden = dims.hidden_dim;
    let (q_perm, k_perm) = match &model.weights.qk_perm {
        Some(p) => (Some(&p.q), Some(&p.k)),
        None => (None, None),
    };
    Ok(FullLayerWeights {
        prologue: FusedLayerWeights {
            norm_weight: host_row(&layer.attn_norm)?,
            w_q: permute_rows(dense_rows(&layer.attn_q, "attn_q", index)?, hidden, q_perm)?,
            w_k: permute_rows(dense_rows(&layer.attn_k, "attn_k", index)?, hidden, k_perm)?,
            w_v: dense_rows(&layer.attn_v, "attn_v", index)?,
        },
        w_o: dense_rows(&layer.attn_output, "attn_output", index)?,
        ffn_norm: host_row(&layer.ffn_norm)?,
        w_gate: dense_rows(&layer.ffn_gate, "ffn_gate", index)?,
        w_up: dense_rows(&layer.ffn_up, "ffn_up", index)?,
        w_down: dense_rows(&layer.ffn_down, "ffn_down", index)?,
    })
}

/// One dense weight matrix as a flat host row-major `[out, hidden]` buffer.
///
/// The fused chain reads dense F32 rows only: a quantized projection has no dense tensor to
/// upload, so it is an `Err` here rather than a silent whole-table dequantization at load.
fn dense_rows(
    weight: &crate::weights::LinearWeight,
    name: &str,
    index: usize,
) -> Result<Vec<f32>, EngineError> {
    let tensor = weight.as_dense_f32().ok_or_else(|| {
        EngineError::Metal(format!(
            "fused decode: layer {index} {name} is quantized; the fused chain needs dense F32"
        ))
    })?;
    host_row(tensor)
}

/// Reorders `[out, hidden]` rows by `perm`, or returns the rows untouched.
fn permute_rows(
    data: Vec<f32>,
    hidden: usize,
    perm: Option<&Tensor>,
) -> Result<Vec<f32>, EngineError> {
    let Some(perm) = perm else {
        return Ok(data);
    };
    let index: Vec<u32> = perm
        .to_device(&Device::Cpu)?
        .contiguous()?
        .flatten_all()?
        .to_vec1::<u32>()?;
    let mut out = Vec::with_capacity(data.len());
    for &src in &index {
        let src = src as usize;
        out.extend_from_slice(&data[src * hidden..(src + 1) * hidden]);
    }
    Ok(out)
}
