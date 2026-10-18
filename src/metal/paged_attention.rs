use crate::error::EngineError;
use crate::metal::context::MetalContext;
use metal::{Buffer, MTLSize};

/// Tokens handled per context partition by the stage-1 kernel (vLLM's PARTITION_SIZE).
const PARTITION_SIZE: u32 = 256;
/// SIMD_WIDTH * MAX_ITEMS in kernels/paged_attention_v2.metal.
const MAX_HEAD_DIM: u32 = 256;
/// MAX_GROUP in kernels/paged_attention_v2.metal.
const MAX_KV_GROUP_SIZE: u32 = 8;

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

/// Context partitions the stage-1 grid is launched with: one per `PARTITION_SIZE`
/// tokens the largest context of this batch can address. Idle partitions cost one
/// empty SIMD group each and contribute nothing to the merge.
pub fn paged_attention_num_partitions(params: &PagedAttentionParams) -> u32 {
    let capacity = params
        .max_num_blocks_per_seq
        .saturating_mul(params.block_size);
    capacity.div_ceil(PARTITION_SIZE).max(1)
}

/// Host-side bounds: anything the kernel cannot represent must fail here instead of
/// silently reading past the end of its per-lane registers (ticket #51).
///
/// Public since ticket #86 so the decode path can reject an unrepresentable shape
/// *before* it uploads a KV pool it would only throw away.
pub fn validate_paged_attention_params(params: &PagedAttentionParams) -> Result<(), EngineError> {
    let num_partitions = paged_attention_num_partitions(params);
    let invalid = |msg: String| Err(EngineError::Metal(msg));

    if params.num_seqs == 0 || params.num_heads == 0 || params.num_kv_heads == 0 {
        return invalid(format!("paged attention: empty shape {params:?}"));
    }
    if !params.num_heads.is_multiple_of(params.num_kv_heads) {
        return invalid(format!(
            "paged attention: num_heads {} not divisible by num_kv_heads {}",
            params.num_heads, params.num_kv_heads
        ));
    }
    let group_size = params.num_heads / params.num_kv_heads;
    if group_size > MAX_KV_GROUP_SIZE {
        return invalid(format!(
            "paged attention: GQA group size {group_size} exceeds kernel limit {MAX_KV_GROUP_SIZE}"
        ));
    }
    if params.head_dim == 0 || params.head_dim > MAX_HEAD_DIM {
        return invalid(format!(
            "paged attention: head_dim {} outside kernel range 1..={MAX_HEAD_DIM}",
            params.head_dim
        ));
    }
    if params.block_size == 0 || params.max_num_blocks_per_seq == 0 {
        return invalid(format!(
            "paged attention: block_size {} / max_num_blocks_per_seq {} must be non-zero",
            params.block_size, params.max_num_blocks_per_seq
        ));
    }
    // The kernel indexes partials with 32-bit element offsets.
    let partial_elements = params.num_seqs as u64
        * params.num_heads as u64
        * num_partitions as u64
        * params.head_dim as u64;
    if partial_elements > (1 << 31) {
        return invalid(format!(
            "paged attention: {partial_elements} partial elements exceed 32-bit kernel indexing"
        ));
    }
    Ok(())
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
            for (t, &ex) in exp_scores.iter().enumerate().take(ctx_len) {
                let weight = ex * inv_sum;
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

/// Dispatches PagedAttention v2: stage 1 splits the context over partitions, stage 2
/// merges them. Partition scratch is allocated here because the caller-facing buffers
/// are only q/k/v/tables/lens/out; both passes share one command buffer.
#[allow(clippy::too_many_arguments)] // #57: public API; regrouping params out of scope
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
    let stage1 = ctx
        .pipelines
        .get("paged_attention_v2")
        .ok_or_else(|| EngineError::Metal("Pipeline paged_attention_v2 not found".into()))?;
    let reduce = ctx
        .pipelines
        .get("paged_attention_v2_reduce")
        .ok_or_else(|| EngineError::Metal("Pipeline paged_attention_v2_reduce not found".into()))?;

    let num_partitions = paged_attention_num_partitions(params);
    validate_paged_attention_params(params)?;

    let num_seqs = params.num_seqs as usize;
    let num_heads = params.num_heads as usize;
    let head_dim = params.head_dim as usize;
    let slots = num_seqs * num_heads * num_partitions as usize;

    let partial_out_buf = ctx.alloc_buffer::<f32>(slots * head_dim);
    // float2 per (seq, head, partition): running max m and running denominator d.
    let partial_stats_buf = ctx.alloc_buffer::<[f32; 2]>(slots);

    let command_buffer = ctx.new_command_buffer();

    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(stage1);
    encoder.set_buffer(0, Some(q_buf), 0);
    encoder.set_buffer(1, Some(k_cache_buf), 0);
    encoder.set_buffer(2, Some(v_cache_buf), 0);
    encoder.set_buffer(3, Some(block_tables_buf), 0);
    encoder.set_buffer(4, Some(context_lens_buf), 0);
    encoder.set_buffer(5, Some(&partial_out_buf), 0);
    encoder.set_buffer(6, Some(&partial_stats_buf), 0);
    encoder.set_bytes(
        7,
        std::mem::size_of::<PagedAttentionParams>() as u64,
        params as *const _ as *const _,
    );
    encoder.set_bytes(
        8,
        std::mem::size_of::<u32>() as u64,
        &num_partitions as *const u32 as *const _,
    );
    encoder.set_bytes(
        9,
        std::mem::size_of::<u32>() as u64,
        &PARTITION_SIZE as *const u32 as *const _,
    );
    encoder.dispatch_thread_groups(
        MTLSize {
            width: num_partitions as u64,
            height: params.num_seqs as u64,
            depth: params.num_kv_heads as u64,
        },
        MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();

    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(reduce);
    encoder.set_buffer(0, Some(&partial_out_buf), 0);
    encoder.set_buffer(1, Some(&partial_stats_buf), 0);
    encoder.set_buffer(2, Some(out_buf), 0);
    encoder.set_bytes(
        3,
        std::mem::size_of::<PagedAttentionParams>() as u64,
        params as *const _ as *const _,
    );
    encoder.set_bytes(
        4,
        std::mem::size_of::<u32>() as u64,
        &num_partitions as *const u32 as *const _,
    );
    encoder.dispatch_thread_groups(
        MTLSize {
            width: num_heads as u64,
            height: params.num_seqs as u64,
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );
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

    // Contexts longer than the block table can address would read outside the block pool.
    let capacity = params.max_num_blocks_per_seq * params.block_size;
    if let Some(&len) = context_lens.iter().find(|&&len| len > capacity) {
        return Err(EngineError::Metal(format!(
            "paged attention: context_len {len} exceeds block table capacity {capacity}"
        )));
    }
    // An empty context leaves every partition idle (all m = -inf), and the merge
    // then computes exp(-inf - -inf) = NaN. Fail loudly instead of emitting NaN.
    if context_lens.contains(&0) {
        return Err(EngineError::Metal(
            "paged attention: context_len must be non-zero (empty context merges to NaN)".into(),
        ));
    }

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
