use crate::error::EngineError;
use crate::metal::context::MetalContext;
use crate::metal::ops::{gpu_rmsnorm_buffer, gpu_rope_buffer};
use metal::{Buffer, MTLSize, ResourceRef};

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

/// Dispatches the fused RMSNorm + RoPE compute kernel in threadgroup memory.
#[allow(clippy::too_many_arguments)] // #57: public API; regrouping params out of scope
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
    let pipeline = ctx.pipelines.get("fused_rmsnorm_rope").ok_or_else(|| {
        EngineError::Metal("Pipeline 'fused_rmsnorm_rope' not found in cache".into())
    })?;

    let command_buffer = ctx.queue.new_command_buffer();
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

    let threads = (dim.next_power_of_two() as u64).clamp(32, 256);
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
#[allow(clippy::too_many_arguments)] // #57: public API; regrouping params out of scope
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
pub fn encode_linear_gemv(
    encoder: &metal::ComputeCommandEncoderRef,
    pipeline: &metal::ComputePipelineState,
    in_vec_buf: &Buffer,
    weight_buf: &Buffer,
    out_vec_buf: &Buffer,
    in_dim: usize,
    out_dim: usize,
) {
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

    let threads_per_group = (out_dim as u64).clamp(32, 256);
    let threadgroups = MTLSize {
        width: (out_dim as u64).div_ceil(threads_per_group),
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

/// Single-encoder deep Metal fusion pipeline for a compact transformer layer.
/// Fuses RMSNorm, QKV Linear projection, and RoPE within a single MTLComputeCommandEncoder.
/// Internal memory barriers and threadgroup memory eliminate multi-pass CPU command buffer commit stalls.
#[allow(clippy::too_many_arguments)] // #57: public API; regrouping params out of scope
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
    let q_dim = dims.q_dim();
    let kv_dim = dims.kv_dim();

    let in_buf = ctx.alloc_buffer_with_data(hidden_in);
    let norm_w_buf = ctx.alloc_buffer_with_data(&weights.norm_weight);
    let w_q_buf = ctx.alloc_buffer_with_data(&weights.w_q);
    let w_k_buf = ctx.alloc_buffer_with_data(&weights.w_k);
    let w_v_buf = ctx.alloc_buffer_with_data(&weights.w_v);
    let cos_buf = ctx.alloc_buffer_with_data(cos_table);
    let sin_buf = ctx.alloc_buffer_with_data(sin_table);

    let q_out_buf = ctx.alloc_buffer::<f32>(q_dim);
    let k_out_buf = ctx.alloc_buffer::<f32>(kv_dim);
    let v_out_buf = ctx.alloc_buffer::<f32>(kv_dim);

    let pipeline = ctx.pipelines.get("fused_rmsnorm_qkv_rope").ok_or_else(|| {
        EngineError::Metal("Pipeline 'fused_rmsnorm_qkv_rope' not found in cache".into())
    })?;

    let command_buffer = ctx.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);

    encoder.set_buffer(0, Some(&in_buf), 0);
    encoder.set_buffer(1, Some(&norm_w_buf), 0);
    encoder.set_buffer(2, Some(&w_q_buf), 0);
    encoder.set_buffer(3, Some(&w_k_buf), 0);
    encoder.set_buffer(4, Some(&w_v_buf), 0);
    encoder.set_buffer(5, Some(&cos_buf), 0);
    encoder.set_buffer(6, Some(&sin_buf), 0);
    encoder.set_buffer(7, Some(&q_out_buf), 0);
    encoder.set_buffer(8, Some(&k_out_buf), 0);
    encoder.set_buffer(9, Some(&v_out_buf), 0);

    let hidden_dim_u32 = dims.hidden_dim as u32;
    let q_dim_u32 = q_dim as u32;
    let kv_dim_u32 = kv_dim as u32;
    let head_dim_u32 = dims.head_dim as u32;
    let pos_u32 = pos as u32;

    encoder.set_bytes(
        10,
        std::mem::size_of::<u32>() as u64,
        &hidden_dim_u32 as *const _ as *const _,
    );
    encoder.set_bytes(
        11,
        std::mem::size_of::<u32>() as u64,
        &q_dim_u32 as *const _ as *const _,
    );
    encoder.set_bytes(
        12,
        std::mem::size_of::<u32>() as u64,
        &kv_dim_u32 as *const _ as *const _,
    );
    encoder.set_bytes(
        13,
        std::mem::size_of::<u32>() as u64,
        &head_dim_u32 as *const _ as *const _,
    );
    encoder.set_bytes(
        14,
        std::mem::size_of::<u32>() as u64,
        &pos_u32 as *const _ as *const _,
    );
    encoder.set_bytes(
        15,
        std::mem::size_of::<f32>() as u64,
        &eps as *const _ as *const _,
    );

    let threads = (dims.hidden_dim.next_power_of_two() as u64).clamp(32, 256);
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

    let mut q_out = vec![0.0f32; q_dim];
    let mut k_out = vec![0.0f32; kv_dim];
    let mut v_out = vec![0.0f32; kv_dim];

    unsafe {
        let ptr_q = q_out_buf.contents() as *const f32;
        let ptr_k = k_out_buf.contents() as *const f32;
        let ptr_v = v_out_buf.contents() as *const f32;
        std::ptr::copy_nonoverlapping(ptr_q, q_out.as_mut_ptr(), q_dim);
        std::ptr::copy_nonoverlapping(ptr_k, k_out.as_mut_ptr(), kv_dim);
        std::ptr::copy_nonoverlapping(ptr_v, v_out.as_mut_ptr(), kv_dim);
    }

    Ok((q_out, k_out, v_out))
}

/// Discrete multi-encoder execution that executes RMSNorm, QKV Linear GEMVs, and RoPE
/// using separate command buffers / command encoders and CPU commit-and-wait stalls.
/// Used as ground truth numerical and performance baseline.
#[allow(clippy::too_many_arguments)] // #57: public API; regrouping params out of scope
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

    let cmd2 = ctx.queue.new_command_buffer();
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
#[allow(clippy::too_many_arguments)] // #57: public API; regrouping params out of scope
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

    let command_buffer = ctx.queue.new_command_buffer();
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
