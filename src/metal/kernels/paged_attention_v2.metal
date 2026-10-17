#include <metal_stdlib>
using namespace metal;

// Ticket #51: PagedAttention v2 = split the context across partitions (kernel 1)
// and merge the partitions (kernel 2), the layout vLLM uses. One SIMD group
// (32 lanes) owns one (partition, seq, kv_head) triple: lanes split the QK dot
// product of a head, the online softmax keeps per-head (m, d) in registers, and
// every query head sharing a KV head is served by the same group so the KV head
// is read once per token instead of once per query head.
//
// MAX_ITEMS and MAX_GROUP are mirrored by MAX_HEAD_DIM / MAX_KV_GROUP_SIZE in
// src/metal/paged_attention.rs, which rejects anything that does not fit.
constant uint SIMD_WIDTH = 32;
// One lane owns q[lane + i*SIMD_WIDTH] -> head_dim up to SIMD_WIDTH * MAX_ITEMS.
constant uint MAX_ITEMS = 8;
// Query heads per KV head (GQA group) unrolled into registers per SIMD group.
constant uint MAX_GROUP = 8;
constant float NEG_INF = -1e30f;

struct PagedAttentionParams {
    uint num_seqs;
    uint num_heads;
    uint num_kv_heads;
    uint head_dim;
    uint block_size;
    uint max_num_blocks_per_seq;
    float scale;
};

// Stage 1: online-softmax attention over one context partition of one (seq, kv_head).
// Emits the unnormalized accumulator plus the (m, d) statistics the merge kernel needs.
kernel void paged_attention_v2_kernel(
    device const float*            q             [[buffer(0)]],
    device const float*            k_cache       [[buffer(1)]],
    device const float*            v_cache       [[buffer(2)]],
    device const uint*             block_tables  [[buffer(3)]],
    device const uint*             context_lens  [[buffer(4)]],
    device float*                  partial_out   [[buffer(5)]],
    device float2*                 partial_stats [[buffer(6)]],
    constant PagedAttentionParams& params        [[buffer(7)]],
    constant uint&                 num_partitions [[buffer(8)]],
    constant uint&                 partition_size [[buffer(9)]],
    uint3 tg_pos [[threadgroup_position_in_grid]], // (partition, seq, kv_head)
    uint lane [[thread_index_in_simdgroup]]
) {
    uint part_idx = tg_pos.x;
    uint seq_idx = tg_pos.y;
    uint kv_head = tg_pos.z;

    if (seq_idx >= params.num_seqs || kv_head >= params.num_kv_heads) {
        return;
    }

    uint head_dim = params.head_dim;
    uint group_size = params.num_heads / params.num_kv_heads;

    float q_lane[MAX_GROUP][MAX_ITEMS];
    float acc[MAX_GROUP][MAX_ITEMS];
    float m_g[MAX_GROUP];
    float d_g[MAX_GROUP];

    for (uint g = 0; g < MAX_GROUP; g++) {
        m_g[g] = NEG_INF;
        d_g[g] = 0.0f;
        for (uint i = 0; i < MAX_ITEMS; i++) {
            q_lane[g][i] = 0.0f;
            acc[g][i] = 0.0f;
        }
    }

    // This lane's slice of every query head in the group, loaded once.
    for (uint g = 0; g < MAX_GROUP; g++) {
        if (g < group_size) {
            uint h = kv_head * group_size + g;
            device const float* q_head = q + (seq_idx * params.num_heads + h) * head_dim;
            for (uint i = 0; i < MAX_ITEMS; i++) {
                uint d = lane + i * SIMD_WIDTH;
                if (d < head_dim) {
                    q_lane[g][i] = q_head[d];
                }
            }
        }
    }

    // Token window of this partition, clamped to what the block table can address.
    uint capacity = params.max_num_blocks_per_seq * params.block_size;
    uint limit = min(context_lens[seq_idx], capacity);
    uint t_start = min(part_idx * partition_size, limit);
    uint t_end = min(t_start + partition_size, limit);

    uint kv_head_dim = params.num_kv_heads * head_dim;
    uint block_stride = params.block_size * kv_head_dim;
    uint head_offset = kv_head * head_dim;
    device const uint* seq_tables = block_tables + seq_idx * params.max_num_blocks_per_seq;

    for (uint t = t_start; t < t_end; t++) {
        uint phys = seq_tables[t / params.block_size];
        uint kv_base = phys * block_stride + (t % params.block_size) * kv_head_dim + head_offset;

        // One K/V head per token, shared by every query head in the group.
        float k_lane[MAX_ITEMS];
        float v_lane[MAX_ITEMS];
        for (uint i = 0; i < MAX_ITEMS; i++) {
            uint d = lane + i * SIMD_WIDTH;
            bool in_range = d < head_dim;
            k_lane[i] = in_range ? k_cache[kv_base + d] : 0.0f;
            v_lane[i] = in_range ? v_cache[kv_base + d] : 0.0f;
        }

        for (uint g = 0; g < MAX_GROUP; g++) {
            if (g < group_size) {
                float dot = 0.0f;
                for (uint i = 0; i < MAX_ITEMS; i++) {
                    dot += q_lane[g][i] * k_lane[i];
                }
                float score = simd_sum(dot) * params.scale;

                // simd_sum broadcasts the score, so m/d are identical in every lane;
                // only the accumulator stays per lane.
                float m_new = max(m_g[g], score);
                float alpha = exp(m_g[g] - m_new); // 0 for the first token of the group
                float w = exp(score - m_new);
                d_g[g] = d_g[g] * alpha + w;
                for (uint i = 0; i < MAX_ITEMS; i++) {
                    acc[g][i] = acc[g][i] * alpha + w * v_lane[i];
                }
                m_g[g] = m_new;
            }
        }
    }

    // Every (seq, head, partition) slot is written, including idle partitions,
    // so the merge kernel never reads uninitialized memory.
    for (uint g = 0; g < MAX_GROUP; g++) {
        if (g < group_size) {
            uint h = kv_head * group_size + g;
            uint slot = (seq_idx * params.num_heads + h) * num_partitions + part_idx;
            device float* dst = partial_out + slot * head_dim;
            for (uint i = 0; i < MAX_ITEMS; i++) {
                uint d = lane + i * SIMD_WIDTH;
                if (d < head_dim) {
                    dst[d] = acc[g][i];
                }
            }
            if (lane == 0) {
                partial_stats[slot] = float2(m_g[g], d_g[g]);
            }
        }
    }
}

// Stage 2: merge partitions for one (seq, head). out = sum_p acc_p * exp(m_p - M)
// over sum_p d_p * exp(m_p - M), with M the max over partitions.
kernel void paged_attention_v2_reduce_kernel(
    device const float*            partial_out   [[buffer(0)]],
    device const float2*           partial_stats [[buffer(1)]],
    device float*                  out           [[buffer(2)]],
    constant PagedAttentionParams& params        [[buffer(3)]],
    constant uint&                 num_partitions [[buffer(4)]],
    uint2 tg_pos [[threadgroup_position_in_grid]], // (head, seq)
    uint lane [[thread_index_in_simdgroup]]
) {
    uint head_idx = tg_pos.x;
    uint seq_idx = tg_pos.y;

    if (seq_idx >= params.num_seqs || head_idx >= params.num_heads) {
        return;
    }

    uint head_dim = params.head_dim;
    uint base = (seq_idx * params.num_heads + head_idx) * num_partitions;

    float m_max = NEG_INF;
    for (uint p = 0; p < num_partitions; p++) {
        m_max = max(m_max, partial_stats[base + p].x);
    }

    float denom = 0.0f;
    float acc[MAX_ITEMS];
    for (uint i = 0; i < MAX_ITEMS; i++) {
        acc[i] = 0.0f;
    }

    for (uint p = 0; p < num_partitions; p++) {
        float2 stat = partial_stats[base + p];
        float w = exp(stat.x - m_max); // 0 for idle partitions (m = -inf), d = 0 too
        denom += w * stat.y;
        device const float* src = partial_out + (base + p) * head_dim;
        for (uint i = 0; i < MAX_ITEMS; i++) {
            uint d = lane + i * SIMD_WIDTH;
            if (d < head_dim) {
                acc[i] += w * src[d];
            }
        }
    }

    float inv_denom = denom > 0.0f ? 1.0f / denom : 0.0f;
    device float* dst = out + (seq_idx * params.num_heads + head_idx) * head_dim;
    for (uint i = 0; i < MAX_ITEMS; i++) {
        uint d = lane + i * SIMD_WIDTH;
        if (d < head_dim) {
            dst[d] = acc[i] * inv_denom;
        }
    }
}
