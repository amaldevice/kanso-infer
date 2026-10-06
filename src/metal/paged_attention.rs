use metal::{Buffer, MTLSize};
use crate::error::EngineError;
use crate::metal::context::MetalContext;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PagedAttentionParams {
    pub num_seqs: u32,
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub block_size: u32,
    pub max_num_blocks_per_seq: u32,
    pub scale: f32,
}

/// Reference CPU implementation of paged attention with GQA and non-contiguous physical blocks.
pub fn cpu_paged_attention(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    block_tables: &[u32],
    context_lens: &[u32],
    params: &PagedAttentionParams,
) -> Vec<f32> {
    let num_seqs = params.num_seqs as usize;
    let num_heads = params.num_heads as usize;
    let num_kv_heads = params.num_kv_heads as usize;
    let head_dim = params.head_dim as usize;
    let block_size = params.block_size as usize;
    let max_num_blocks = params.max_num_blocks_per_seq as usize;
    let scale = params.scale;

    assert!(num_kv_heads > 0, "num_kv_heads must be > 0");
    assert_eq!(
        num_heads % num_kv_heads,
        0,
        "num_heads must be divisible by num_kv_heads for GQA"
    );
    let kv_group_size = num_heads / num_kv_heads;

    let total_elements = num_seqs * num_heads * head_dim;
    let mut out = vec![0.0f32; total_elements];

    for s in 0..num_seqs {
        let ctx_len = context_lens[s] as usize;
        if ctx_len == 0 {
            continue;
        }

        for h in 0..num_heads {
            let kv_h = h / kv_group_size;
            let q_start = (s * num_heads + h) * head_dim;
            let q_vec = &q[q_start..q_start + head_dim];

            // 1. Compute dot-product attention scores
            let mut scores = Vec::with_capacity(ctx_len);
            for t in 0..ctx_len {
                let b_idx = t / block_size;
                let b_offset = t % block_size;
                let phys_block = block_tables[s * max_num_blocks + b_idx] as usize;

                let k_offset = phys_block * (block_size * num_kv_heads * head_dim)
                    + b_offset * (num_kv_heads * head_dim)
                    + kv_h * head_dim;

                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q_vec[d] * k_cache[k_offset + d];
                }
                scores.push(dot * scale);
            }

            // 2. Softmax
            let mut max_score = f32::NEG_INFINITY;
            for &sc in &scores {
                if sc > max_score {
                    max_score = sc;
                }
            }

            let mut sum_exp = 0.0f32;
            let mut exp_scores = Vec::with_capacity(ctx_len);
            for &sc in &scores {
                let ex = (sc - max_score).exp();
                exp_scores.push(ex);
                sum_exp += ex;
            }

            let inv_sum = if sum_exp > 0.0 { 1.0 / sum_exp } else { 0.0 };

            // 3. Weighted sum of V
            let out_start = (s * num_heads + h) * head_dim;
            for t in 0..ctx_len {
                let weight = exp_scores[t] * inv_sum;
                let b_idx = t / block_size;
                let b_offset = t % block_size;
                let phys_block = block_tables[s * max_num_blocks + b_idx] as usize;

                let v_offset = phys_block * (block_size * num_kv_heads * head_dim)
                    + b_offset * (num_kv_heads * head_dim)
                    + kv_h * head_dim;

                for d in 0..head_dim {
                    out[out_start + d] += weight * v_cache[v_offset + d];
                }
            }
        }
    }

    out
}

/// Dispatches fused PagedAttention v2 Metal kernel with online softmax in threadgroup memory.
pub fn gpu_paged_attention_v2_buffer(
    ctx: &MetalContext,
    q_buf: &Buffer,
    k_cache_buf: &Buffer,
    v_cache_buf: &Buffer,
    block_tables_buf: &Buffer,
    context_lens_buf: &Buffer,
    out_buf: &Buffer,
    params: &PagedAttentionParams,
) -> Result<(), EngineError> {
    let pipeline = ctx
        .pipelines
        .get("paged_attention_v2")
        .ok_or_else(|| EngineError::Metal("Pipeline paged_attention_v2 not found".into()))?;

    let command_buffer = ctx.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);

    encoder.set_buffer(0, Some(q_buf), 0);
    encoder.set_buffer(1, Some(k_cache_buf), 0);
    encoder.set_buffer(2, Some(v_cache_buf), 0);
    encoder.set_buffer(3, Some(block_tables_buf), 0);
    encoder.set_buffer(4, Some(context_lens_buf), 0);
    encoder.set_buffer(5, Some(out_buf), 0);

    encoder.set_bytes(
        6,
        std::mem::size_of::<PagedAttentionParams>() as u64,
        params as *const _ as *const _,
    );

    let threadgroup_count = MTLSize {
        width: params.num_heads as u64,
        height: params.num_seqs as u64,
        depth: 1,
    };
    let threadgroup_size = MTLSize {
        width: 32,
        height: 1,
        depth: 1,
    };

    encoder.dispatch_thread_groups(threadgroup_count, threadgroup_size);
    encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    Ok(())
}

/// High-level function that allocates zero-copy unified memory buffers and dispatches paged attention.
pub fn gpu_paged_attention_v2(
    ctx: &MetalContext,
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    block_tables: &[u32],
    context_lens: &[u32],
    params: &PagedAttentionParams,
) -> Result<Vec<f32>, EngineError> {
    let num_seqs = params.num_seqs as usize;
    let num_heads = params.num_heads as usize;
    let head_dim = params.head_dim as usize;
    let total_out_elements = num_seqs * num_heads * head_dim;

    let q_buf = ctx.alloc_buffer_with_data(q);
    let k_cache_buf = ctx.alloc_buffer_with_data(k_cache);
    let v_cache_buf = ctx.alloc_buffer_with_data(v_cache);
    let block_tables_buf = ctx.alloc_buffer_with_data(block_tables);
    let context_lens_buf = ctx.alloc_buffer_with_data(context_lens);
    let out_buf = ctx.alloc_buffer::<f32>(total_out_elements);

    gpu_paged_attention_v2_buffer(
        ctx,
        &q_buf,
        &k_cache_buf,
        &v_cache_buf,
        &block_tables_buf,
        &context_lens_buf,
        &out_buf,
        params,
    )?;

    let mut out = vec![0.0f32; total_out_elements];
    unsafe {
        let ptr = out_buf.contents() as *const f32;
        std::ptr::copy_nonoverlapping(ptr, out.as_mut_ptr(), total_out_elements);
    }

    Ok(out)
}
