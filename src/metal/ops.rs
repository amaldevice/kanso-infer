use metal::{Buffer, MTLSize};
use crate::error::EngineError;
use crate::metal::context::MetalContext;

pub fn cpu_rmsnorm(input: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let dim = input.len();
    let sum_sq: f32 = input.iter().map(|&x| x * x).sum();
    let mean_sq = sum_sq / (dim as f32);
    let rms = (mean_sq + eps).sqrt();

    input
        .iter()
        .zip(weight.iter())
        .map(|(&x, &w)| (x / rms) * w)
        .collect()
}

pub fn cpu_silu_gate_multiply(gate: &[f32], up: &[f32]) -> Vec<f32> {
    gate.iter()
        .zip(up.iter())
        .map(|(&g, &u)| {
            let silu = g / (1.0 + (-g).exp());
            silu * u
        })
        .collect()
}

pub fn cpu_rope(
    q_k: &mut [f32],
    cos_table: &[f32],
    sin_table: &[f32],
    head_dim: usize,
    pos: usize,
) {
    let half_dim = head_dim / 2;
    let total_elements = q_k.len();
    assert_eq!(total_elements % head_dim, 0, "total elements must be divisible by head_dim");
    let num_heads = total_elements / head_dim;

    for h in 0..num_heads {
        let head_offset = h * head_dim;
        for idx in 0..half_dim {
            let i0 = head_offset + idx;
            let i1 = head_offset + idx + half_dim;
            let x0 = q_k[i0];
            let x1 = q_k[i1];
            let cos_val = cos_table[pos * half_dim + idx];
            let sin_val = sin_table[pos * half_dim + idx];
            q_k[i0] = x0 * cos_val - x1 * sin_val;
            q_k[i1] = x0 * sin_val + x1 * cos_val;
        }
    }
}

pub fn gpu_rmsnorm_buffer(
    ctx: &MetalContext,
    in_buf: &Buffer,
    weight_buf: &Buffer,
    out_buf: &Buffer,
    num_rows: usize,
    dim: usize,
    eps: f32,
) -> Result<(), EngineError> {
    let pipeline = ctx.pipelines.get("rmsnorm").ok_or_else(|| {
        EngineError::Metal("Pipeline 'rmsnorm' not found in cache".into())
    })?;

    let command_buffer = ctx.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);

    encoder.set_buffer(0, Some(in_buf), 0);
    encoder.set_buffer(1, Some(weight_buf), 0);
    encoder.set_buffer(2, Some(out_buf), 0);

    let dim_u32 = dim as u32;
    encoder.set_bytes(3, std::mem::size_of::<u32>() as u64, &dim_u32 as *const _ as *const _);
    encoder.set_bytes(4, std::mem::size_of::<f32>() as u64, &eps as *const _ as *const _);

    let threads = (dim.next_power_of_two() as u64).clamp(32, 256);
    let threadgroup_count = MTLSize {
        width: num_rows as u64,
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

pub fn gpu_rmsnorm(
    ctx: &MetalContext,
    input: &[f32],
    weight: &[f32],
    eps: f32,
) -> Result<Vec<f32>, EngineError> {
    let dim = weight.len();
    assert_eq!(input.len() % dim, 0, "input len must be multiple of dim");
    let num_rows = input.len() / dim;

    let in_buf = ctx.alloc_buffer_with_data(input);
    let weight_buf = ctx.alloc_buffer_with_data(weight);
    let out_buf = ctx.alloc_buffer::<f32>(input.len());

    gpu_rmsnorm_buffer(ctx, &in_buf, &weight_buf, &out_buf, num_rows, dim, eps)?;

    let mut out = vec![0.0f32; input.len()];
    unsafe {
        let ptr = out_buf.contents() as *const f32;
        std::ptr::copy_nonoverlapping(ptr, out.as_mut_ptr(), input.len());
    }
    Ok(out)
}

pub fn gpu_rope_buffer(
    ctx: &MetalContext,
    q_k_buf: &Buffer,
    cos_buf: &Buffer,
    sin_buf: &Buffer,
    total_elements: usize,
    head_dim: usize,
    pos: usize,
) -> Result<(), EngineError> {
    let pipeline = ctx.pipelines.get("rope").ok_or_else(|| {
        EngineError::Metal("Pipeline 'rope' not found in cache".into())
    })?;

    let command_buffer = ctx.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);

    encoder.set_buffer(0, Some(q_k_buf), 0);
    encoder.set_buffer(1, Some(cos_buf), 0);
    encoder.set_buffer(2, Some(sin_buf), 0);

    let head_dim_u32 = head_dim as u32;
    let pos_u32 = pos as u32;
    encoder.set_bytes(3, std::mem::size_of::<u32>() as u64, &head_dim_u32 as *const _ as *const _);
    encoder.set_bytes(4, std::mem::size_of::<u32>() as u64, &pos_u32 as *const _ as *const _);

    let total_threads = (total_elements / 2) as u64;
    let threadgroup_width = (total_threads.min(256)).max(1);
    let threads_per_grid = MTLSize {
        width: total_threads,
        height: 1,
        depth: 1,
    };
    let threads_per_threadgroup = MTLSize {
        width: threadgroup_width,
        height: 1,
        depth: 1,
    };

    encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
    encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    Ok(())
}

pub fn gpu_rope(
    ctx: &MetalContext,
    q_k: &mut [f32],
    cos_table: &[f32],
    sin_table: &[f32],
    head_dim: usize,
    pos: usize,
) -> Result<(), EngineError> {
    let q_k_buf = ctx.alloc_buffer_with_data(q_k);
    let cos_buf = ctx.alloc_buffer_with_data(cos_table);
    let sin_buf = ctx.alloc_buffer_with_data(sin_table);

    gpu_rope_buffer(ctx, &q_k_buf, &cos_buf, &sin_buf, q_k.len(), head_dim, pos)?;

    unsafe {
        let ptr = q_k_buf.contents() as *const f32;
        std::ptr::copy_nonoverlapping(ptr, q_k.as_mut_ptr(), q_k.len());
    }
    Ok(())
}

pub fn gpu_silu_gate_multiply_buffer(
    ctx: &MetalContext,
    gate_buf: &Buffer,
    up_buf: &Buffer,
    out_buf: &Buffer,
    len: usize,
) -> Result<(), EngineError> {
    let pipeline = ctx.pipelines.get("silu_gate").ok_or_else(|| {
        EngineError::Metal("Pipeline 'silu_gate' not found in cache".into())
    })?;

    let command_buffer = ctx.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);

    encoder.set_buffer(0, Some(gate_buf), 0);
    encoder.set_buffer(1, Some(up_buf), 0);
    encoder.set_buffer(2, Some(out_buf), 0);

    let total_threads = len as u64;
    let threadgroup_width = (total_threads.min(256)).max(1);
    let threads_per_grid = MTLSize {
        width: total_threads,
        height: 1,
        depth: 1,
    };
    let threads_per_threadgroup = MTLSize {
        width: threadgroup_width,
        height: 1,
        depth: 1,
    };

    encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
    encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    Ok(())
}

pub fn gpu_silu_gate_multiply(
    ctx: &MetalContext,
    gate: &[f32],
    up: &[f32],
) -> Result<Vec<f32>, EngineError> {
    assert_eq!(gate.len(), up.len(), "gate and up must have same length");
    let len = gate.len();

    let gate_buf = ctx.alloc_buffer_with_data(gate);
    let up_buf = ctx.alloc_buffer_with_data(up);
    let out_buf = ctx.alloc_buffer::<f32>(len);

    gpu_silu_gate_multiply_buffer(ctx, &gate_buf, &up_buf, &out_buf, len)?;

    let mut out = vec![0.0f32; len];
    unsafe {
        let ptr = out_buf.contents() as *const f32;
        std::ptr::copy_nonoverlapping(ptr, out.as_mut_ptr(), len);
    }
    Ok(out)
}
