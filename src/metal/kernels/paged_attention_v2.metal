#include <metal_stdlib>
using namespace metal;

constant uint TILE_SIZE = 32;
constant uint MAX_HEAD_DIM = 512;
constant uint MAX_ITEMS_PER_THREAD = 16; // 16 * 32 = 512

struct PagedAttentionParams {
    uint num_seqs;
    uint num_heads;
    uint num_kv_heads;
    uint head_dim;
    uint block_size;
    uint max_num_blocks_per_seq;
    float scale;
};

kernel void paged_attention_v2_kernel(
    device const float*                q            [[buffer(0)]],
    device const float*                k_cache      [[buffer(1)]],
    device const float*                v_cache      [[buffer(2)]],
    device const uint*                 block_tables [[buffer(3)]],
    device const uint*                 context_lens [[buffer(4)]],
    device float*                      out          [[buffer(5)]],
    constant PagedAttentionParams&     params       [[buffer(6)]],
    uint2                              tg_pos       [[threadgroup_position_in_grid]],
    uint2                              t_pos        [[thread_position_in_threadgroup]]
) {
    uint tid = t_pos.x;
    uint head_idx = tg_pos.x;
    uint seq_idx = tg_pos.y;

    if (seq_idx >= params.num_seqs || head_idx >= params.num_heads) {
        return;
    }

    uint context_len = context_lens[seq_idx];
    uint kv_group_size = params.num_heads / params.num_kv_heads;
    uint kv_head_idx = head_idx / kv_group_size;

    // Threadgroup shared memory for query vector, online softmax statistics, and exponential weights
    threadgroup float s_q[MAX_HEAD_DIM];
    threadgroup float s_scores[TILE_SIZE];
    threadgroup float s_exp[TILE_SIZE]; // exp(s - m)
    threadgroup float m_prev;
    threadgroup float d_prev;
    threadgroup float alpha;

    // Initialize online softmax state
    if (tid == 0) {
        m_prev = -1e30f;
        d_prev = 0.0f;
        alpha = 1.0f;
    }

    // Cooperatively load Q into threadgroup memory
    uint q_offset = (seq_idx * params.num_heads + head_idx) * params.head_dim;
    for (uint d = tid; d < params.head_dim && d < MAX_HEAD_DIM; d += 32) {
        s_q[d] = q[q_offset + d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Thread-local register accumulator for output dimensions
    float acc[MAX_ITEMS_PER_THREAD];
    for (uint i = 0; i < MAX_ITEMS_PER_THREAD; i++) {
        acc[i] = 0.0f;
    }

    // Process historical context tokens in tiles of TILE_SIZE (32)
    for (uint tile_start = 0; tile_start < context_len; tile_start += TILE_SIZE) {
        uint tile_tokens = min(TILE_SIZE, context_len - tile_start);

        // 1. Tiled matrix multiplication: S_t = (Q . K_t) * scale
        if (tid < tile_tokens) {
            uint token_idx = tile_start + tid;
            uint b_idx = token_idx / params.block_size;
            uint b_offset = token_idx % params.block_size;
            uint phys_block = block_tables[seq_idx * params.max_num_blocks_per_seq + b_idx];

            uint k_offset = phys_block * (params.block_size * params.num_kv_heads * params.head_dim)
                          + b_offset * (params.num_kv_heads * params.head_dim)
                          + kv_head_idx * params.head_dim;

            float dot = 0.0f;
            for (uint d = 0; d < params.head_dim; d++) {
                dot += s_q[d] * k_cache[k_offset + d];
            }
            s_scores[tid] = dot * params.scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 2. Online softmax update (m_prev, d_prev, exp(s - m)) in threadgroup memory
        if (tid == 0) {
            float local_m = -1e30f;
            for (uint t = 0; t < tile_tokens; t++) {
                local_m = max(local_m, s_scores[t]);
            }

            float m_new = max(m_prev, local_m);
            float a = (m_prev <= -1e29f) ? 0.0f : exp(m_prev - m_new);
            alpha = a;

            float local_d = 0.0f;
            for (uint t = 0; t < tile_tokens; t++) {
                float ex = exp(s_scores[t] - m_new);
                s_exp[t] = ex; // exp(s - m)
                local_d += ex;
            }

            d_prev = d_prev * a + local_d;
            m_prev = m_new;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 3. Accumulate V using threadgroup exp(s - m) weights directly into register accumulator
        float cur_alpha = alpha;
        for (uint item = 0; item < MAX_ITEMS_PER_THREAD; item++) {
            uint d = tid + item * 32;
            if (d < params.head_dim) {
                float v_sum = 0.0f;
                for (uint t = 0; t < tile_tokens; t++) {
                    uint token_idx = tile_start + t;
                    uint b_idx = token_idx / params.block_size;
                    uint b_offset = token_idx % params.block_size;
                    uint phys_block = block_tables[seq_idx * params.max_num_blocks_per_seq + b_idx];

                    uint v_offset = phys_block * (params.block_size * params.num_kv_heads * params.head_dim)
                                  + b_offset * (params.num_kv_heads * params.head_dim)
                                  + kv_head_idx * params.head_dim
                                  + d;

                    v_sum += s_exp[t] * v_cache[v_offset];
                }
                acc[item] = acc[item] * cur_alpha + v_sum;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // 4. Normalize and write final context activations to global memory out buffer
    float final_d = d_prev;
    for (uint item = 0; item < MAX_ITEMS_PER_THREAD; item++) {
        uint d = tid + item * 32;
        if (d < params.head_dim) {
            uint out_idx = (seq_idx * params.num_heads + head_idx) * params.head_dim + d;
            out[out_idx] = (final_d > 0.0f) ? (acc[item] / final_d) : 0.0f;
        }
    }
}
