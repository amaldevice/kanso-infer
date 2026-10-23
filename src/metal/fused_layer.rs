use crate::error::EngineError;
use crate::metal::context::MetalContext;
use crate::metal::ops::{
    encode_residual_add, encode_silu_gate_multiply, gpu_rmsnorm_buffer, gpu_rope_buffer,
};
use metal::{Buffer, MTLSize, ResourceRef};

/// Threads per threadgroup required by the kernels in `fused_llama_layer.metal`
/// (mirrors `FUSED_THREADS_PER_TG` in the shader; the kernels drop out on mismatch).
pub const FUSED_THREADS_PER_TG: usize = 256;
/// SIMD groups (32 lanes each) per threadgroup.
pub const FUSED_SIMD_GROUPS: usize = FUSED_THREADS_PER_TG / 32;
/// Output rows one GEMV threadgroup covers: one row pair per SIMD group.
pub const GEMV_ROWS_PER_THREADGROUP: usize = FUSED_SIMD_GROUPS * 2;

/// Dimensions for a compact transformer layer (e.g. ~350M models).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FusedLayerDimensions {
    pub hidden_dim: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl FusedLayerDimensions {
    #[inline]
    pub fn q_dim(&self) -> usize {
        self.num_q_heads * self.head_dim
    }

    #[inline]
    pub fn kv_dim(&self) -> usize {
        self.num_kv_heads * self.head_dim
    }

    /// Host-side dimension contract of the fused kernels (ticket #50): anything the
    /// kernels could index out of range fails here instead of on the GPU. There is no
    /// hidden-dim ceiling - the fixed `shared_norm[2048]` threadgroup array that capped
    /// this at 2048 is gone.
    pub fn validate(&self) -> Result<(), EngineError> {
        if self.num_q_heads == 0 || self.num_kv_heads == 0 {
            return Err(EngineError::Metal(format!(
                "head counts must be non-zero, got num_q_heads={} num_kv_heads={}",
                self.num_q_heads, self.num_kv_heads
            )));
        }
        if self.head_dim == 0 || !self.head_dim.is_multiple_of(2) {
            return Err(EngineError::Metal(format!(
                "head_dim must be a non-zero even count (RoPE rotates two halves), got {}",
                self.head_dim
            )));
        }
        if self.hidden_dim == 0 || !self.hidden_dim.is_multiple_of(4) {
            return Err(EngineError::Metal(format!(
                "hidden_dim must be a non-zero multiple of 4 (GEMV lanes read float4 chunks), got {}",
                self.hidden_dim
            )));
        }
        Ok(())
    }
}

/// Threadgroups and threads per threadgroup for a GEMV over `out_dim` rows.
/// One SIMD group computes one row pair, so a threadgroup covers
/// [`GEMV_ROWS_PER_THREADGROUP`] rows (ticket #50: the single-threadgroup dispatch is
/// gone).
pub fn gemv_threadgroup_geometry(out_dim: usize) -> (u64, u64) {
    let units = out_dim.div_ceil(2) as u64;
    (
        units.div_ceil(FUSED_SIMD_GROUPS as u64),
        FUSED_THREADS_PER_TG as u64,
    )
}

/// Slice-shape contract of the host weights: the kernels index these buffers
/// blindly from `dims`, so a slice of the wrong length is an out-of-bounds GPU read
/// or write.
fn validate_weight_shapes(
    dims: &FusedLayerDimensions,
    weights: &FusedLayerWeights,
) -> Result<(), EngineError> {
    dims.validate()?;

    let q_dim = dims.q_dim();
    let kv_dim = dims.kv_dim();
    let expect = |name: &str, got: usize, want: usize| -> Result<(), EngineError> {
        if got == want {
            Ok(())
        } else {
            Err(EngineError::Metal(format!(
                "{name} has {got} elements, expected {want} for hidden_dim {} \
                 (q_dim {q_dim}, kv_dim {kv_dim})",
                dims.hidden_dim
            )))
        }
    };

    expect("norm_weight", weights.norm_weight.len(), dims.hidden_dim)?;
    expect("w_q", weights.w_q.len(), q_dim * dims.hidden_dim)?;
    expect("w_k", weights.w_k.len(), kv_dim * dims.hidden_dim)?;
    expect("w_v", weights.w_v.len(), kv_dim * dims.hidden_dim)?;
    Ok(())
}

/// Slice-shape contract of the layer entry points, which take host slices for the
/// step input and the rotary tables on every call.
fn validate_layer_inputs(
    dims: &FusedLayerDimensions,
    weights: &FusedLayerWeights,
    hidden_in: &[f32],
    cos_table: &[f32],
    sin_table: &[f32],
    pos: usize,
) -> Result<(), EngineError> {
    validate_weight_shapes(dims, weights)?;

    let q_dim = dims.q_dim();
    let kv_dim = dims.kv_dim();
    if hidden_in.len() != dims.hidden_dim {
        return Err(EngineError::Metal(format!(
            "hidden_in has {} elements, expected {} for hidden_dim {} \
             (q_dim {q_dim}, kv_dim {kv_dim})",
            hidden_in.len(),
            dims.hidden_dim,
            dims.hidden_dim
        )));
    }

    let table_len = (pos + 1) * (dims.head_dim / 2);
    for (name, table) in [("cos_table", cos_table), ("sin_table", sin_table)] {
        if table.len() < table_len {
            return Err(EngineError::Metal(format!(
                "{name} has {} elements, needs at least {table_len} to cover pos {pos}",
                table.len()
            )));
        }
    }

    Ok(())
}

/// Read `len` floats back out of a shared (zero-copy) device buffer.
pub(crate) fn read_f32(buf: &Buffer, len: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; len];
    unsafe {
        std::ptr::copy_nonoverlapping(buf.contents() as *const f32, out.as_mut_ptr(), len);
    }
    out
}

/// Stage host floats into a shared device buffer without reallocating it.
pub(crate) fn write_f32(buf: &Buffer, data: &[f32]) {
    debug_assert!(buf.length() as usize >= std::mem::size_of_val(data));
    unsafe {
        std::ptr::copy_nonoverlapping(data.as_ptr(), buf.contents() as *mut f32, data.len());
    }
}

/// Q, K, V output triple returned by fused/discrete layer helpers.
pub type FusedQkvTriple = (Vec<f32>, Vec<f32>, Vec<f32>);

/// Host-side weights for RMSNorm and QKV linear projections.
#[derive(Debug, Clone)]
pub struct FusedLayerWeights {
    pub norm_weight: Vec<f32>,
    pub w_q: Vec<f32>,
    pub w_k: Vec<f32>,
    pub w_v: Vec<f32>,
}

/// Device-resident weights, tables and scratch for one layer's prologue (ticket #49).
///
/// Upload once at load, then call [`MetalWeights::decode_step`] per token: the step
/// allocates nothing, re-uploads nothing, and submits exactly one command buffer for
/// the whole fused RMSNorm + QKV + RoPE chain.
///
/// Production entry (since #87, widened by #110): the one-command-buffer decode
/// step in `src/model/fused_decode.rs` holds these inside a [`ResidentLayer`] and
/// drives the prologue through [`Self::encode_prologue`] on the step's command
/// buffer. `decode_step` — create, commit, wait — is the per-layer submit
/// strategy the paths that cannot chain a whole step keep, and the harness entry
/// `tests/test_single_command_buffer.rs` asserts.
pub struct MetalWeights {
    dims: FusedLayerDimensions,
    norm_weight: Buffer,
    w_q: Buffer,
    w_k: Buffer,
    w_v: Buffer,
    cos: Buffer,
    sin: Buffer,
    /// Elements per rotary table, used to bound the `pos` a step may ask for.
    table_len: usize,
    in_buf: Buffer,
    q_out: Buffer,
    k_out: Buffer,
    v_out: Buffer,
    weight_uploads: usize,
}

impl MetalWeights {
    /// Upload the host weights and rotary tables into device buffers once.
    pub fn upload(
        ctx: &MetalContext,
        dims: &FusedLayerDimensions,
        weights: &FusedLayerWeights,
        cos_table: &[f32],
        sin_table: &[f32],
    ) -> Result<Self, EngineError> {
        validate_weight_shapes(dims, weights)?;
        if cos_table.len() != sin_table.len() {
            return Err(EngineError::Metal(format!(
                "cos/sin tables must have equal length, got cos={} sin={}",
                cos_table.len(),
                sin_table.len()
            )));
        }
        if cos_table.len() < dims.head_dim / 2 {
            return Err(EngineError::Metal(format!(
                "rotary tables hold {} elements, need at least {} for head_dim {}",
                cos_table.len(),
                dims.head_dim / 2,
                dims.head_dim
            )));
        }

        // Weight and table uploads are counted so a test can prove they happen once
        // at load rather than per decode step.
        let mut weight_uploads = 0usize;
        let mut upload = |data: &[f32]| -> Buffer {
            weight_uploads += 1;
            ctx.alloc_buffer_with_data(data)
        };
        let norm_weight = upload(&weights.norm_weight);
        let w_q = upload(&weights.w_q);
        let w_k = upload(&weights.w_k);
        let w_v = upload(&weights.w_v);
        let cos = upload(cos_table);
        let sin = upload(sin_table);

        Ok(Self {
            dims: *dims,
            norm_weight,
            w_q,
            w_k,
            w_v,
            cos,
            sin,
            table_len: cos_table.len(),
            in_buf: ctx.alloc_buffer::<f32>(dims.hidden_dim),
            q_out: ctx.alloc_buffer::<f32>(dims.q_dim()),
            k_out: ctx.alloc_buffer::<f32>(dims.kv_dim()),
            v_out: ctx.alloc_buffer::<f32>(dims.kv_dim()),
            weight_uploads,
        })
    }

    /// Buffer uploads performed at load (weights and rotary tables).
    pub fn weight_uploads(&self) -> usize {
        self.weight_uploads
    }

    /// One layer decode step: stage the input row, encode RMSNorm + QKV projection +
    /// RoPE into a single command buffer, and return Q, K, V.
    pub fn decode_step(
        &self,
        ctx: &MetalContext,
        hidden_in: &[f32],
        pos: usize,
        eps: f32,
    ) -> Result<FusedQkvTriple, EngineError> {
        if hidden_in.len() != self.dims.hidden_dim {
            return Err(EngineError::Metal(format!(
                "hidden_in has {} elements, expected {} for hidden_dim {}",
                hidden_in.len(),
                self.dims.hidden_dim,
                self.dims.hidden_dim
            )));
        }
        let needed = (pos + 1) * (self.dims.head_dim / 2);
        if needed > self.table_len {
            return Err(EngineError::Metal(format!(
                "rotary tables hold {} elements, pos {pos} needs {needed}",
                self.table_len
            )));
        }

        let q_dim = self.dims.q_dim();
        let kv_dim = self.dims.kv_dim();
        write_f32(&self.in_buf, hidden_in);

        let command_buffer = ctx.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        self.encode_prologue(ctx, encoder, &self.in_buf, pos, eps)?;
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok((
            read_f32(&self.q_out, q_dim),
            read_f32(&self.k_out, kv_dim),
            read_f32(&self.v_out, kv_dim),
        ))
    }

    /// Encodes the fused RMSNorm + QKV + RoPE chain into a caller-owned compute
    /// encoder (ticket #110). The three dispatches write disjoint outputs and only
    /// read shared inputs, so one encoder covers them — the same contract as
    /// [`Self::decode_step`], minus the command buffer. `in_buf` may be the
    /// caller's residual-stream buffer instead of this layer's staging input.
    pub fn encode_prologue(
        &self,
        ctx: &MetalContext,
        encoder: &metal::ComputeCommandEncoderRef,
        in_buf: &Buffer,
        pos: usize,
        eps: f32,
    ) -> Result<(), EngineError> {
        let pipeline = ctx.pipelines.get("fused_rmsnorm_qkv_rope").ok_or_else(|| {
            EngineError::Metal("Pipeline 'fused_rmsnorm_qkv_rope' not found in cache".into())
        })?;

        let q_dim = self.dims.q_dim();
        let kv_dim = self.dims.kv_dim();
        // Q and K are RoPE-rotated on the fly (row pairs are rotation pairs), V is not.
        encode_fused_qkv_gemv(
            encoder,
            pipeline,
            in_buf,
            &self.norm_weight,
            &self.w_q,
            &self.cos,
            &self.sin,
            &self.q_out,
            &self.dims,
            q_dim,
            pos,
            eps,
            true,
        );
        encode_fused_qkv_gemv(
            encoder,
            pipeline,
            in_buf,
            &self.norm_weight,
            &self.w_k,
            &self.cos,
            &self.sin,
            &self.k_out,
            &self.dims,
            kv_dim,
            pos,
            eps,
            true,
        );
        encode_fused_qkv_gemv(
            encoder,
            pipeline,
            in_buf,
            &self.norm_weight,
            &self.w_v,
            &self.cos,
            &self.sin,
            &self.v_out,
            &self.dims,
            kv_dim,
            pos,
            eps,
            false,
        );
        Ok(())
    }

    /// This layer's fused RoPE output row (the query the step's attention reads).
    pub fn q_out(&self) -> &Buffer {
        &self.q_out
    }

    /// This layer's roped K output row, staged for the pool copy.
    pub fn k_out(&self) -> &Buffer {
        &self.k_out
    }

    /// This layer's V output row, staged for the pool copy.
    pub fn v_out(&self) -> &Buffer {
        &self.v_out
    }

    /// The resident rotary tables, bindable for rope-off fused GEMV dispatches
    /// (the kernel only indexes them when `apply_rope` is set).
    pub fn rope_tables(&self) -> (&Buffer, &Buffer) {
        (&self.cos, &self.sin)
    }
}

/// Host-side weights for one whole decoder layer on the one-command-buffer step
/// (ticket #110): the prologue plus the O projection, the FFN norm and the SwiGLU
/// MLP. Dense F32 rows only — the same contract as [`FusedLayerWeights`], so a
/// quantized projection keeps the pre-existing path.
#[derive(Debug, Clone)]
pub struct FullLayerWeights {
    pub prologue: FusedLayerWeights,
    /// `[hidden, q_dim]` attention output projection.
    pub w_o: Vec<f32>,
    /// FFN RMSNorm weight, `[hidden]`.
    pub ffn_norm: Vec<f32>,
    /// `[intermediate, hidden]` gate projection.
    pub w_gate: Vec<f32>,
    /// `[intermediate, hidden]` up projection.
    pub w_up: Vec<f32>,
    /// `[hidden, intermediate]` down projection.
    pub w_down: Vec<f32>,
}

/// Slice-shape contract of the full-layer weights (ticket #110).
fn validate_full_layer_shapes(
    dims: &FusedLayerDimensions,
    intermediate: usize,
    weights: &FullLayerWeights,
) -> Result<(), EngineError> {
    validate_weight_shapes(dims, &weights.prologue)?;
    let hidden = dims.hidden_dim;
    let expect = |name: &str, got: usize, want: usize| -> Result<(), EngineError> {
        if got == want {
            Ok(())
        } else {
            Err(EngineError::Metal(format!(
                "{name} has {got} elements, expected {want} (hidden_dim {hidden}, \
                 intermediate {intermediate})"
            )))
        }
    };
    if intermediate == 0 || !intermediate.is_multiple_of(4) {
        return Err(EngineError::Metal(format!(
            "intermediate_dim must be a non-zero multiple of 4 (down GEMV reads float4 \
             lanes, gate/up cover row pairs), got {intermediate}"
        )));
    }
    if !dims.q_dim().is_multiple_of(4) {
        return Err(EngineError::Metal(format!(
            "q_dim {} must be a multiple of 4 (the O-projection GEMV reads float4 lanes)",
            dims.q_dim()
        )));
    }
    expect("w_o", weights.w_o.len(), hidden * dims.q_dim())?;
    expect("ffn_norm", weights.ffn_norm.len(), hidden)?;
    expect("w_gate", weights.w_gate.len(), intermediate * hidden)?;
    expect("w_up", weights.w_up.len(), intermediate * hidden)?;
    expect("w_down", weights.w_down.len(), hidden * intermediate)?;
    Ok(())
}

/// Device-resident full layer plus the scratch the one-command-buffer step needs
/// (ticket #110): the prologue's [`MetalWeights`] plus the O/FFN weights and the
/// per-layer intermediate buffers. Uploaded once at build; a step allocates
/// nothing and encodes into the step's command buffer.
pub struct ResidentLayer {
    prologue: MetalWeights,
    intermediate: usize,
    w_o: Buffer,
    ffn_norm: Buffer,
    w_gate: Buffer,
    w_up: Buffer,
    w_down: Buffer,
    o_out: Buffer,
    gate_out: Buffer,
    up_out: Buffer,
    silu_out: Buffer,
    down_out: Buffer,
}

impl ResidentLayer {
    /// Uploads the whole layer's weights into device buffers once.
    pub fn upload(
        ctx: &MetalContext,
        dims: &FusedLayerDimensions,
        intermediate: usize,
        weights: &FullLayerWeights,
        cos_table: &[f32],
        sin_table: &[f32],
    ) -> Result<Self, EngineError> {
        validate_full_layer_shapes(dims, intermediate, weights)?;
        let prologue = MetalWeights::upload(ctx, dims, &weights.prologue, cos_table, sin_table)?;
        Ok(Self {
            prologue,
            intermediate,
            w_o: ctx.alloc_buffer_with_data(&weights.w_o),
            ffn_norm: ctx.alloc_buffer_with_data(&weights.ffn_norm),
            w_gate: ctx.alloc_buffer_with_data(&weights.w_gate),
            w_up: ctx.alloc_buffer_with_data(&weights.w_up),
            w_down: ctx.alloc_buffer_with_data(&weights.w_down),
            o_out: ctx.alloc_buffer::<f32>(dims.hidden_dim),
            gate_out: ctx.alloc_buffer::<f32>(intermediate),
            up_out: ctx.alloc_buffer::<f32>(intermediate),
            silu_out: ctx.alloc_buffer::<f32>(intermediate),
            down_out: ctx.alloc_buffer::<f32>(dims.hidden_dim),
        })
    }

    /// Encodes the layer's fused RMSNorm + QKV + RoPE prologue into a caller-owned
    /// command buffer, reading the caller's residual-stream buffer as input.
    pub fn encode_prologue(
        &self,
        ctx: &MetalContext,
        encoder: &metal::ComputeCommandEncoderRef,
        hidden: &Buffer,
        pos: usize,
        eps: f32,
    ) -> Result<(), EngineError> {
        self.prologue
            .encode_prologue(ctx, encoder, hidden, pos, eps)
    }

    /// This layer's fused query row (RoPE-applied), bound straight into the step's
    /// paged-attention dispatch — no host round trip.
    pub fn q_out(&self) -> &Buffer {
        self.prologue.q_out()
    }

    /// This layer's roped K output row, still on the device for the pool copy.
    pub fn k_out(&self) -> &Buffer {
        self.prologue.k_out()
    }

    /// This layer's V output row, still on the device for the pool copy.
    pub fn v_out(&self) -> &Buffer {
        self.prologue.v_out()
    }

    /// Reads the layer's K row back to the host after the step's command buffer
    /// completed, for the KV pool writeback.
    pub fn read_k(&self) -> Vec<f32> {
        read_f32(self.prologue.k_out(), self.prologue_dims().kv_dim())
    }

    /// Reads the layer's V row back to the host after the step's command buffer
    /// completed, for the KV pool writeback.
    pub fn read_v(&self) -> Vec<f32> {
        read_f32(self.prologue.v_out(), self.prologue_dims().kv_dim())
    }

    fn prologue_dims(&self) -> &FusedLayerDimensions {
        // MetalWeights keeps its own dims copy; upload validated it against the
        // dims passed here, so the geometry the scratch was sized for matches.
        &self.prologue.dims
    }

    /// The prologue's resident weights — the per-layer submit strategy's entry
    /// (`decode_step`), kept for the paths that cannot chain a whole step.
    pub fn as_metal_weights(&self) -> &MetalWeights {
        &self.prologue
    }

    /// Encodes the layer tail after attention: O projection + residual add, then
    /// the SwiGLU MLP (fused-norm gate/up GEMVs, SiLU·gate, down GEMV) + residual
    /// add, into the caller's command buffer (ticket #110).
    ///
    /// Every dependent dispatch gets its own compute encoder: dispatches within a
    /// single encoder are not guaranteed to be ordered, while encoders on one
    /// command buffer are. `residual_scale` is the Granite `residual_scale`
    /// multiplier (1.0 on plain llama) — the same factor the tensor path applies
    /// via `Model::scale_residual`.
    pub fn encode_tail(
        &self,
        ctx: &MetalContext,
        command_buffer: &metal::CommandBufferRef,
        attn_out: &Buffer,
        hidden: &Buffer,
        residual_scale: f32,
        eps: f32,
        dims: &FusedLayerDimensions,
    ) -> Result<(), EngineError> {
        let gemv = ctx
            .pipelines
            .get("linear_gemv")
            .ok_or_else(|| EngineError::Metal("Pipeline 'linear_gemv' not found".into()))?;
        let residual = ctx
            .pipelines
            .get("residual_add")
            .ok_or_else(|| EngineError::Metal("Pipeline 'residual_add' not found".into()))?;
        let fused = ctx.pipelines.get("fused_rmsnorm_qkv_rope").ok_or_else(|| {
            EngineError::Metal("Pipeline 'fused_rmsnorm_qkv_rope' not found".into())
        })?;
        let silu = ctx
            .pipelines
            .get("silu_gate")
            .ok_or_else(|| EngineError::Metal("Pipeline 'silu_gate' not found".into()))?;

        let hidden_dim = dims.hidden_dim;
        let intermediate = self.intermediate;

        // O projection: [q_dim] -> [hidden].
        let encoder = command_buffer.new_compute_command_encoder();
        encode_linear_gemv(
            encoder,
            gemv,
            attn_out,
            &self.w_o,
            &self.o_out,
            dims.q_dim(),
            hidden_dim,
        );
        encoder.end_encoding();

        let encoder = command_buffer.new_compute_command_encoder();
        encode_residual_add(
            encoder,
            residual,
            &self.o_out,
            hidden,
            hidden_dim,
            residual_scale,
        );
        encoder.end_encoding();

        // Gate/up: the fused kernel normalises with the FFN norm on the fly; RoPE
        // off means its output rows are plain adjacent pairs, so any even out_dim
        // (the intermediate) is representable.
        let (cos, sin) = self.prologue.rope_tables();
        let encoder = command_buffer.new_compute_command_encoder();
        encode_fused_qkv_gemv(
            encoder,
            fused,
            hidden,
            &self.ffn_norm,
            &self.w_gate,
            cos,
            sin,
            &self.gate_out,
            dims,
            intermediate,
            0,
            eps,
            false,
        );
        encode_fused_qkv_gemv(
            encoder,
            fused,
            hidden,
            &self.ffn_norm,
            &self.w_up,
            cos,
            sin,
            &self.up_out,
            dims,
            intermediate,
            0,
            eps,
            false,
        );
        encoder.end_encoding();

        let encoder = command_buffer.new_compute_command_encoder();
        encode_silu_gate_multiply(
            encoder,
            silu,
            &self.gate_out,
            &self.up_out,
            &self.silu_out,
            intermediate,
        );
        encoder.end_encoding();

        let encoder = command_buffer.new_compute_command_encoder();
        encode_linear_gemv(
            encoder,
            gemv,
            &self.silu_out,
            &self.w_down,
            &self.down_out,
            intermediate,
            hidden_dim,
        );
        encoder.end_encoding();

        let encoder = command_buffer.new_compute_command_encoder();
        encode_residual_add(
            encoder,
            residual,
            &self.down_out,
            hidden,
            hidden_dim,
            residual_scale,
        );
        encoder.end_encoding();

        Ok(())
    }
}

/// Dispatches the fused RMSNorm + RoPE compute kernel in threadgroup memory.
pub fn gpu_fused_rmsnorm_rope_buffer(
    ctx: &MetalContext,
    in_buf: &Buffer,
    weight_buf: &Buffer,
    cos_buf: &Buffer,
    sin_buf: &Buffer,
    out_buf: &Buffer,
    dim: usize,
    head_dim: usize,
    pos: usize,
    eps: f32,
) -> Result<(), EngineError> {
    // Host-side contract of the kernel (ticket #50): the rotation loop indexes the row
    // by whole heads, so dim must be a whole number of even head_dim rows.
    if dim == 0 || !dim.is_multiple_of(2) {
        return Err(EngineError::Metal(format!(
            "fused rmsnorm+rope dim must be a non-zero even count, got {dim}"
        )));
    }
    if head_dim == 0 || !head_dim.is_multiple_of(2) || !dim.is_multiple_of(head_dim) {
        return Err(EngineError::Metal(format!(
            "fused rmsnorm+rope needs a non-zero even head_dim dividing dim, got head_dim={head_dim} dim={dim}"
        )));
    }

    let pipeline = ctx.pipelines.get("fused_rmsnorm_rope").ok_or_else(|| {
        EngineError::Metal("Pipeline 'fused_rmsnorm_rope' not found in cache".into())
    })?;

    let command_buffer = ctx.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);

    encoder.set_buffer(0, Some(in_buf), 0);
    encoder.set_buffer(1, Some(weight_buf), 0);
    encoder.set_buffer(2, Some(cos_buf), 0);
    encoder.set_buffer(3, Some(sin_buf), 0);
    encoder.set_buffer(4, Some(out_buf), 0);

    let dim_u32 = dim as u32;
    let head_dim_u32 = head_dim as u32;
    let pos_u32 = pos as u32;

    encoder.set_bytes(
        5,
        std::mem::size_of::<u32>() as u64,
        &dim_u32 as *const _ as *const _,
    );
    encoder.set_bytes(
        6,
        std::mem::size_of::<u32>() as u64,
        &head_dim_u32 as *const _ as *const _,
    );
    encoder.set_bytes(
        7,
        std::mem::size_of::<u32>() as u64,
        &pos_u32 as *const _ as *const _,
    );
    encoder.set_bytes(
        8,
        std::mem::size_of::<f32>() as u64,
        &eps as *const _ as *const _,
    );

    let threads = FUSED_THREADS_PER_TG as u64;

    // Single threadgroup: the kernel partitions its rotation loop over `tid`
    // only (no `tg` index), so one threadgroup already covers the whole row
    // and extra threadgroups would recompute it redundantly for zero speedup.
    // (The GEMV kernels below *are* partitioned across TGs, one row pair per
    // SIMD group; this RoPE-only helper is not on the fused decode path.)
    let threadgroup_count = MTLSize {
        width: 1,
        height: 1,
        depth: 1,
    };
    let threadgroup_size = MTLSize {
        width: threads,
        height: 1,
        depth: 1,
    };

    encoder.dispatch_thread_groups(threadgroup_count, threadgroup_size);
    encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    Ok(())
}

/// Fused RMSNorm + RoPE execution helper with host slices.
pub fn gpu_fused_rmsnorm_rope(
    ctx: &MetalContext,
    input: &[f32],
    weight: &[f32],
    cos_table: &[f32],
    sin_table: &[f32],
    head_dim: usize,
    pos: usize,
    eps: f32,
) -> Result<Vec<f32>, EngineError> {
    assert_eq!(
        input.len(),
        weight.len(),
        "input and weight must have matching length"
    );
    let dim = input.len();
    let table_len = (pos + 1) * (head_dim / 2);
    if cos_table.len() < table_len || sin_table.len() < table_len {
        return Err(EngineError::Metal(format!(
            "rotary tables must cover pos {pos}: need at least {table_len} elements, \
             got cos={} sin={}",
            cos_table.len(),
            sin_table.len()
        )));
    }

    let in_buf = ctx.alloc_buffer_with_data(input);
    let weight_buf = ctx.alloc_buffer_with_data(weight);
    let cos_buf = ctx.alloc_buffer_with_data(cos_table);
    let sin_buf = ctx.alloc_buffer_with_data(sin_table);
    let out_buf = ctx.alloc_buffer::<f32>(dim);

    gpu_fused_rmsnorm_rope_buffer(
        ctx,
        &in_buf,
        &weight_buf,
        &cos_buf,
        &sin_buf,
        &out_buf,
        dim,
        head_dim,
        pos,
        eps,
    )?;

    let mut out = vec![0.0f32; dim];
    unsafe {
        let ptr = out_buf.contents() as *const f32;
        std::ptr::copy_nonoverlapping(ptr, out.as_mut_ptr(), dim);
    }
    Ok(out)
}

/// Dispatches a linear GEMV kernel for a single input vector.
///
/// The `linear_gemv_kernel` covers row pairs (`units = out_dim / 2`) and reads
/// `float4` lanes (`n4 = in_dim / 4`): an odd `out_dim` would silently drop the
/// last row and a non-multiple-of-4 `in_dim` would silently drop the tail, so
/// both fail here instead of on the GPU.
pub fn encode_linear_gemv(
    encoder: &metal::ComputeCommandEncoderRef,
    pipeline: &metal::ComputePipelineState,
    in_vec_buf: &Buffer,
    weight_buf: &Buffer,
    out_vec_buf: &Buffer,
    in_dim: usize,
    out_dim: usize,
) {
    assert!(
        out_dim > 0 && out_dim.is_multiple_of(2),
        "encode_linear_gemv: out_dim must be a non-zero even count (kernel covers row pairs), got {out_dim}"
    );
    assert!(
        in_dim > 0 && in_dim.is_multiple_of(4),
        "encode_linear_gemv: in_dim must be a non-zero multiple of 4 (kernel reads float4 lanes), got {in_dim}"
    );
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(in_vec_buf), 0);
    encoder.set_buffer(1, Some(weight_buf), 0);
    encoder.set_buffer(2, Some(out_vec_buf), 0);

    let in_dim_u32 = in_dim as u32;
    let out_dim_u32 = out_dim as u32;
    encoder.set_bytes(
        3,
        std::mem::size_of::<u32>() as u64,
        &in_dim_u32 as *const _ as *const _,
    );
    encoder.set_bytes(
        4,
        std::mem::size_of::<u32>() as u64,
        &out_dim_u32 as *const _ as *const _,
    );

    let (threadgroups, threads_per_group) = gemv_threadgroup_geometry(out_dim);
    let threadgroups = MTLSize {
        width: threadgroups,
        height: 1,
        depth: 1,
    };
    let group_size = MTLSize {
        width: threads_per_group,
        height: 1,
        depth: 1,
    };

    encoder.dispatch_thread_groups(threadgroups, group_size);
}

/// Encodes one fused RMSNorm + GEMV (+ RoPE) dispatch for a single weight matrix.
/// Output rows are covered by `gemv_threadgroup_geometry`, one row pair per SIMD group
/// (ticket #50: the projection used to run on a single threadgroup).
/// Public so callers can chain several projections into one command encoder, which is
/// also how the GEMV bandwidth test isolates GPU time from host buffer upload.
pub fn encode_fused_qkv_gemv(
    encoder: &metal::ComputeCommandEncoderRef,
    pipeline: &metal::ComputePipelineState,
    in_buf: &Buffer,
    norm_w_buf: &Buffer,
    w_buf: &Buffer,
    cos_buf: &Buffer,
    sin_buf: &Buffer,
    out_buf: &Buffer,
    dims: &FusedLayerDimensions,
    out_dim: usize,
    pos: usize,
    eps: f32,
    apply_rope: bool,
) {
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(in_buf), 0);
    encoder.set_buffer(1, Some(norm_w_buf), 0);
    encoder.set_buffer(2, Some(w_buf), 0);
    encoder.set_buffer(3, Some(cos_buf), 0);
    encoder.set_buffer(4, Some(sin_buf), 0);
    encoder.set_buffer(5, Some(out_buf), 0);

    let hidden_dim_u32 = dims.hidden_dim as u32;
    let out_dim_u32 = out_dim as u32;
    let head_dim_u32 = dims.head_dim as u32;
    let pos_u32 = pos as u32;
    let apply_rope_u32 = u32::from(apply_rope);

    encoder.set_bytes(
        6,
        std::mem::size_of::<u32>() as u64,
        &hidden_dim_u32 as *const _ as *const _,
    );
    encoder.set_bytes(
        7,
        std::mem::size_of::<u32>() as u64,
        &out_dim_u32 as *const _ as *const _,
    );
    encoder.set_bytes(
        8,
        std::mem::size_of::<u32>() as u64,
        &head_dim_u32 as *const _ as *const _,
    );
    encoder.set_bytes(
        9,
        std::mem::size_of::<u32>() as u64,
        &pos_u32 as *const _ as *const _,
    );
    encoder.set_bytes(
        10,
        std::mem::size_of::<f32>() as u64,
        &eps as *const _ as *const _,
    );
    encoder.set_bytes(
        11,
        std::mem::size_of::<u32>() as u64,
        &apply_rope_u32 as *const _ as *const _,
    );

    let (threadgroups, threads_per_group) = gemv_threadgroup_geometry(out_dim);
    encoder.dispatch_thread_groups(
        MTLSize {
            width: threadgroups,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: threads_per_group,
            height: 1,
            depth: 1,
        },
    );
}

/// Slice-based single-encoder layer entry point: uploads the weights, runs one
/// decode step and returns Q, K, V.
///
/// Convenient for tests and one-shot calls; a decode loop should hold a
/// [`MetalWeights`] from load time instead, so weights are not re-uploaded per step.
pub fn gpu_fused_layer_single_encoder(
    ctx: &MetalContext,
    hidden_in: &[f32],
    weights: &FusedLayerWeights,
    dims: &FusedLayerDimensions,
    cos_table: &[f32],
    sin_table: &[f32],
    pos: usize,
    eps: f32,
) -> Result<FusedQkvTriple, EngineError> {
    validate_layer_inputs(dims, weights, hidden_in, cos_table, sin_table, pos)?;
    MetalWeights::upload(ctx, dims, weights, cos_table, sin_table)?
        .decode_step(ctx, hidden_in, pos, eps)
}

/// Discrete multi-encoder execution that executes RMSNorm, QKV Linear GEMVs, and RoPE
/// using separate command buffers / command encoders and CPU commit-and-wait stalls.
/// Used as ground truth numerical and performance baseline.
pub fn gpu_discrete_layer_multi_encoder(
    ctx: &MetalContext,
    hidden_in: &[f32],
    weights: &FusedLayerWeights,
    dims: &FusedLayerDimensions,
    cos_table: &[f32],
    sin_table: &[f32],
    pos: usize,
    eps: f32,
) -> Result<FusedQkvTriple, EngineError> {
    validate_layer_inputs(dims, weights, hidden_in, cos_table, sin_table, pos)?;

    let q_dim = dims.q_dim();
    let kv_dim = dims.kv_dim();

    let in_buf = ctx.alloc_buffer_with_data(hidden_in);
    let norm_w_buf = ctx.alloc_buffer_with_data(&weights.norm_weight);
    let normed_buf = ctx.alloc_buffer::<f32>(dims.hidden_dim);

    let w_q_buf = ctx.alloc_buffer_with_data(&weights.w_q);
    let w_k_buf = ctx.alloc_buffer_with_data(&weights.w_k);
    let w_v_buf = ctx.alloc_buffer_with_data(&weights.w_v);

    let q_buf = ctx.alloc_buffer::<f32>(q_dim);
    let k_buf = ctx.alloc_buffer::<f32>(kv_dim);
    let v_buf = ctx.alloc_buffer::<f32>(kv_dim);

    let cos_buf = ctx.alloc_buffer_with_data(cos_table);
    let sin_buf = ctx.alloc_buffer_with_data(sin_table);

    // 1. Pass 1: RMSNorm in separate command buffer
    gpu_rmsnorm_buffer(
        ctx,
        &in_buf,
        &norm_w_buf,
        &normed_buf,
        1,
        dims.hidden_dim,
        eps,
    )?;

    // 2. Pass 2: Linear GEMV projections in separate command buffer
    let gemv_pipeline = ctx
        .pipelines
        .get("linear_gemv")
        .ok_or_else(|| EngineError::Metal("Pipeline 'linear_gemv' not found in cache".into()))?;

    let cmd2 = ctx.new_command_buffer();
    let enc2 = cmd2.new_compute_command_encoder();
    encode_linear_gemv(
        enc2,
        gemv_pipeline,
        &normed_buf,
        &w_q_buf,
        &q_buf,
        dims.hidden_dim,
        q_dim,
    );
    encode_linear_gemv(
        enc2,
        gemv_pipeline,
        &normed_buf,
        &w_k_buf,
        &k_buf,
        dims.hidden_dim,
        kv_dim,
    );
    encode_linear_gemv(
        enc2,
        gemv_pipeline,
        &normed_buf,
        &w_v_buf,
        &v_buf,
        dims.hidden_dim,
        kv_dim,
    );
    enc2.end_encoding();
    cmd2.commit();
    cmd2.wait_until_completed();

    // 3. Pass 3: RoPE on Q and K in separate command buffer
    gpu_rope_buffer(ctx, &q_buf, &cos_buf, &sin_buf, q_dim, dims.head_dim, pos)?;
    gpu_rope_buffer(ctx, &k_buf, &cos_buf, &sin_buf, kv_dim, dims.head_dim, pos)?;

    let mut q_out = vec![0.0f32; q_dim];
    let mut k_out = vec![0.0f32; kv_dim];
    let mut v_out = vec![0.0f32; kv_dim];

    unsafe {
        let ptr_q = q_buf.contents() as *const f32;
        let ptr_k = k_buf.contents() as *const f32;
        let ptr_v = v_buf.contents() as *const f32;
        std::ptr::copy_nonoverlapping(ptr_q, q_out.as_mut_ptr(), q_dim);
        std::ptr::copy_nonoverlapping(ptr_k, k_out.as_mut_ptr(), kv_dim);
        std::ptr::copy_nonoverlapping(ptr_v, v_out.as_mut_ptr(), kv_dim);
    }

    Ok((q_out, k_out, v_out))
}

/// Pipeline encoder that chains multi-step kernels within a single command encoder
/// using Metal memory barriers (`memory_barrier_with_resources`).
pub fn gpu_chained_pipeline_single_encoder(
    ctx: &MetalContext,
    hidden_in: &[f32],
    weights: &FusedLayerWeights,
    dims: &FusedLayerDimensions,
    cos_table: &[f32],
    sin_table: &[f32],
    pos: usize,
    eps: f32,
) -> Result<FusedQkvTriple, EngineError> {
    validate_layer_inputs(dims, weights, hidden_in, cos_table, sin_table, pos)?;

    let q_dim = dims.q_dim();
    let kv_dim = dims.kv_dim();

    let in_buf = ctx.alloc_buffer_with_data(hidden_in);
    let norm_w_buf = ctx.alloc_buffer_with_data(&weights.norm_weight);
    let normed_buf = ctx.alloc_buffer::<f32>(dims.hidden_dim);

    let w_q_buf = ctx.alloc_buffer_with_data(&weights.w_q);
    let w_k_buf = ctx.alloc_buffer_with_data(&weights.w_k);
    let w_v_buf = ctx.alloc_buffer_with_data(&weights.w_v);

    let q_buf = ctx.alloc_buffer::<f32>(q_dim);
    let k_buf = ctx.alloc_buffer::<f32>(kv_dim);
    let v_buf = ctx.alloc_buffer::<f32>(kv_dim);

    let cos_buf = ctx.alloc_buffer_with_data(cos_table);
    let sin_buf = ctx.alloc_buffer_with_data(sin_table);

    let rmsnorm_pipeline = ctx
        .pipelines
        .get("rmsnorm")
        .ok_or_else(|| EngineError::Metal("Pipeline 'rmsnorm' not found in cache".into()))?;
    let gemv_pipeline = ctx
        .pipelines
        .get("linear_gemv")
        .ok_or_else(|| EngineError::Metal("Pipeline 'linear_gemv' not found in cache".into()))?;
    let rope_pipeline = ctx
        .pipelines
        .get("rope")
        .ok_or_else(|| EngineError::Metal("Pipeline 'rope' not found in cache".into()))?;

    let command_buffer = ctx.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    // 1. RMSNorm
    encoder.set_compute_pipeline_state(rmsnorm_pipeline);
    encoder.set_buffer(0, Some(&in_buf), 0);
    encoder.set_buffer(1, Some(&norm_w_buf), 0);
    encoder.set_buffer(2, Some(&normed_buf), 0);
    let dim_u32 = dims.hidden_dim as u32;
    encoder.set_bytes(
        3,
        std::mem::size_of::<u32>() as u64,
        &dim_u32 as *const _ as *const _,
    );
    encoder.set_bytes(
        4,
        std::mem::size_of::<f32>() as u64,
        &eps as *const _ as *const _,
    );
    let threads = (dims.hidden_dim.next_power_of_two() as u64).clamp(32, 256);
    encoder.dispatch_thread_groups(
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        },
    );

    // Internal memory barrier on normed_buf before GEMV reads it
    let normed_res: &ResourceRef = &normed_buf;
    encoder.memory_barrier_with_resources(&[normed_res]);

    // 2. QKV Linear GEMVs
    encode_linear_gemv(
        encoder,
        gemv_pipeline,
        &normed_buf,
        &w_q_buf,
        &q_buf,
        dims.hidden_dim,
        q_dim,
    );
    encode_linear_gemv(
        encoder,
        gemv_pipeline,
        &normed_buf,
        &w_k_buf,
        &k_buf,
        dims.hidden_dim,
        kv_dim,
    );
    encode_linear_gemv(
        encoder,
        gemv_pipeline,
        &normed_buf,
        &w_v_buf,
        &v_buf,
        dims.hidden_dim,
        kv_dim,
    );

    // Internal memory barrier on Q and K buffers before RoPE rotates them
    let q_res: &ResourceRef = &q_buf;
    let k_res: &ResourceRef = &k_buf;
    encoder.memory_barrier_with_resources(&[q_res, k_res]);

    // 3. RoPE on Q
    encoder.set_compute_pipeline_state(rope_pipeline);
    encoder.set_buffer(0, Some(&q_buf), 0);
    encoder.set_buffer(1, Some(&cos_buf), 0);
    encoder.set_buffer(2, Some(&sin_buf), 0);
    let head_dim_u32 = dims.head_dim as u32;
    let pos_u32 = pos as u32;
    encoder.set_bytes(
        3,
        std::mem::size_of::<u32>() as u64,
        &head_dim_u32 as *const _ as *const _,
    );
    encoder.set_bytes(
        4,
        std::mem::size_of::<u32>() as u64,
        &pos_u32 as *const _ as *const _,
    );
    let total_q_threads = (q_dim / 2) as u64;
    let q_group_width = total_q_threads.clamp(1, 256);
    encoder.dispatch_threads(
        MTLSize {
            width: total_q_threads,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: q_group_width,
            height: 1,
            depth: 1,
        },
    );

    // RoPE on K
    encoder.set_buffer(0, Some(&k_buf), 0);
    let total_k_threads = (kv_dim / 2) as u64;
    let k_group_width = total_k_threads.clamp(1, 256);
    encoder.dispatch_threads(
        MTLSize {
            width: total_k_threads,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: k_group_width,
            height: 1,
            depth: 1,
        },
    );

    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    let mut q_out = vec![0.0f32; q_dim];
    let mut k_out = vec![0.0f32; kv_dim];
    let mut v_out = vec![0.0f32; kv_dim];

    unsafe {
        let ptr_q = q_buf.contents() as *const f32;
        let ptr_k = k_buf.contents() as *const f32;
        let ptr_v = v_buf.contents() as *const f32;
        std::ptr::copy_nonoverlapping(ptr_q, q_out.as_mut_ptr(), q_dim);
        std::ptr::copy_nonoverlapping(ptr_k, k_out.as_mut_ptr(), kv_dim);
        std::ptr::copy_nonoverlapping(ptr_v, v_out.as_mut_ptr(), kv_dim);
    }

    Ok((q_out, k_out, v_out))
}
