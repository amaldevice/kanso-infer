//! The fused Metal llama-layer kernel on the production decode path (ticket #87).
//!
//! `src/metal/fused_layer.rs` already carries the tested chain — `MetalWeights` (resident
//! weights, one command buffer per step of RMSNorm + QKV + RoPE), `encode_fused_qkv_gemv`,
//! `encode_linear_gemv`, `gpu_chained_pipeline_single_encoder` and the discrete baseline.
//! Until now nothing outside `tests/` called it. This module is the caller: a dense Metal
//! decode step runs its per-layer QKV prologue through [`MetalWeights::decode_step`], so
//! the fused kernel is what actually produces Q/K/V in production, and attention keeps
//! riding the #86 paged-attention seam instead of growing a second Metal path.
//!
//! What the fused kernel covers is the prologue only — RMSNorm, the three projections and
//! RoPE. Attention, the output projection and the MLP stay where #86 and the tensor path
//! put them. That split is a recorded debt, not an oversight: folding them into the same
//! command buffer needs a residual-add kernel and an encoder-shaped attention helper, i.e.
//! new kernels, which spec #82 puts out of scope for this ticket.
//!
//! Two rules keep the wiring honest:
//!
//! * the geometry gate ([`fused_prologue_supported`]) runs *before* anything is uploaded,
//!   so a model the kernels cannot represent keeps the pre-existing path — a fallback,
//!   never an error and never a silent wrong-shape dispatch;
//! * the prologue reads the model's own rotary tables (`rope_cos_sin`), so the fused RoPE
//!   cannot drift from `rope_apply`.

use crate::error::EngineError;
use crate::metal::context::MetalContext;
use crate::metal::fused_layer::{
    FusedLayerDimensions, FusedLayerWeights, FusedQkvTriple, MetalWeights,
};
use candle_core::{Device, Tensor};

/// One model's fused prologue seam, built the first time a dense Metal decode step runs.
pub(crate) struct FusedDecodeSeam {
    ctx: MetalContext,
    /// One resident [`MetalWeights`] per model layer: weights uploaded once at build, so a
    /// decode step neither re-uploads nor re-allocates (ticket #49's contract).
    layers: Vec<MetalWeights>,
    /// Largest position the resident rotary tables cover.
    max_pos: usize,
    steps: u64,
}

impl FusedDecodeSeam {
    /// Builds the seam, or `Ok(None)` when this model is outside the kernel geometry.
    ///
    /// The geometry check comes first so a model the kernels cannot serve uploads nothing:
    /// it keeps the pre-existing path and no number changes.
    pub(crate) fn build(model: &super::Model) -> Result<Option<Self>, EngineError> {
        let dims = FusedLayerDimensions {
            hidden_dim: model.config.hidden_size,
            num_q_heads: model.config.num_attention_heads,
            num_kv_heads: model.config.num_key_value_heads,
            head_dim: model.config.head_dim(),
        };
        let max_pos = model.config.max_seq_len.saturating_sub(1);
        let table_len = (max_pos + 1) * (dims.head_dim / 2).max(1);
        if !fused_prologue_supported(&dims, table_len) {
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
            let weights = LayerHostWeights::read(model, index, layer, &dims)?;
            layers.push(MetalWeights::upload(
                &ctx,
                &dims,
                &weights.view(),
                &cos,
                &sin,
            )?);
        }

        Ok(Some(Self {
            ctx,
            layers,
            max_pos,
            steps: 0,
        }))
    }

    /// Largest position the resident rotary tables cover.
    pub(crate) fn max_pos(&self) -> usize {
        self.max_pos
    }

    /// Decode steps this seam has served.
    pub(crate) fn steps(&self) -> u64 {
        self.steps
    }

    /// Command buffers this seam submitted. One per layer prologue: the fused kernel is a
    /// single submission for the whole RMSNorm + QKV + RoPE chain, not one per projection.
    pub(crate) fn command_buffers(&self) -> usize {
        self.ctx.command_buffers_created()
    }

    /// Runs one layer's fused RMSNorm + QKV + RoPE prologue and returns the host Q/K/V rows.
    ///
    /// The rows are already RoPE-rotated in the rotate-half layout `rope_apply` produces,
    /// so the caller must not rotate them a second time.
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
        let resident = self.layers.get(layer).ok_or_else(|| {
            EngineError::Metal(format!(
                "fused decode: layer {layer} outside the {} resident layers",
                self.layers.len()
            ))
        })?;
        self.steps += 1;
        resident.decode_step(&self.ctx, hidden, pos, eps)
    }
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

/// Reads a tensor into a flat host row, wherever it lives.
fn host_row(tensor: &Tensor) -> Result<Vec<f32>, EngineError> {
    Ok(tensor
        .to_device(&Device::Cpu)?
        .contiguous()?
        .flatten_all()?
        .to_vec1::<f32>()?)
}

/// Owned host copies of one layer's prologue weights.
struct LayerHostWeights {
    norm_weight: Vec<f32>,
    w_q: Vec<f32>,
    w_k: Vec<f32>,
    w_v: Vec<f32>,
}

impl LayerHostWeights {
    /// Copies one layer's dense prologue weights to the host, applying the GGUF Q/K row
    /// permutation so the fused RoPE sees rotate-half rows — the same permutation
    /// `project_qk` applies to the projection output.
    fn read(
        model: &super::Model,
        index: usize,
        layer: &crate::weights::LayerWeights,
        dims: &FusedLayerDimensions,
    ) -> Result<Self, EngineError> {
        let hidden = dims.hidden_dim;
        let (q_perm, k_perm) = match &model.weights.qk_perm {
            Some(p) => (Some(&p.q), Some(&p.k)),
            None => (None, None),
        };
        Ok(Self {
            norm_weight: host_row(&layer.attn_norm)?,
            w_q: permute_rows(dense_rows(&layer.attn_q, "attn_q", index)?, hidden, q_perm)?,
            w_k: permute_rows(dense_rows(&layer.attn_k, "attn_k", index)?, hidden, k_perm)?,
            w_v: dense_rows(&layer.attn_v, "attn_v", index)?,
        })
    }

    fn view(&self) -> FusedLayerWeights {
        FusedLayerWeights {
            norm_weight: self.norm_weight.clone(),
            w_q: self.w_q.clone(),
            w_k: self.w_k.clone(),
            w_v: self.w_v.clone(),
        }
    }
}

/// One dense weight matrix as a flat host row-major `[out, hidden]` buffer.
///
/// The fused chain reads dense F32 rows only: a quantized projection has no dense tensor to
/// upload, so it is an `Err` here rather than a silent whole-table dequantization at load.
/// The caller treats that `Err` as "keep the pre-existing path".
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
