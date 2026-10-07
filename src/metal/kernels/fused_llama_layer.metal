#include <metal_stdlib>
using namespace metal;

/// Fused RMSNorm and RoPE kernel executing entirely in threadgroup memory.
/// Replaces multi-pass global memory round-trips and CPU command buffer commit stalls.
kernel void fused_rmsnorm_rope_kernel(
    device const float* in        [[buffer(0)]],
    device const float* weight    [[buffer(1)]],
    device const float* cos_table [[buffer(2)]],
    device const float* sin_table [[buffer(3)]],
    device float*       out       [[buffer(4)]],
    constant uint&      dim       [[buffer(5)]],
    constant uint&      head_dim  [[buffer(6)]],
    constant uint&      pos       [[buffer(7)]],
    constant float&     eps       [[buffer(8)]],
    uint                tid       [[thread_position_in_threadgroup]],
    uint                bid       [[threadgroup_position_in_grid]],
    uint                threads   [[threads_per_threadgroup]]
) {
    threadgroup float sdata[1024];
    threadgroup float shared_norm[2048];

    device const float* row_in = in + bid * dim;
    device float* row_out = out + bid * dim;

    // 1. RMSNorm: Sum of squares computation
    float sum_sq = 0.0f;
    for (uint i = tid; i < dim; i += threads) {
        float val = row_in[i];
        sum_sq += val * val;
    }
    sdata[tid] = sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction across threadgroup with barriers
    for (uint s = threads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid < 32) {
        simdgroup_barrier(mem_flags::mem_none);
    }

    float mean_sq = sdata[0] / float(dim);
    float rms = rsqrt(mean_sq + eps);

    // 2. Normalize and cache in threadgroup shared memory
    for (uint i = tid; i < dim; i += threads) {
        shared_norm[i] = row_in[i] * rms * weight[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 3. In-threadgroup RoPE application
    uint half_dim = head_dim / 2;
    uint total_rotations = dim / 2;

    for (uint rot_idx = tid; rot_idx < total_rotations; rot_idx += threads) {
        uint idx = rot_idx % half_dim;
        uint head_offset = (rot_idx / half_dim) * head_dim;

        uint i0 = head_offset + idx;
        uint i1 = head_offset + idx + half_dim;

        float x0 = shared_norm[i0];
        float x1 = shared_norm[i1];

        float cos_val = cos_table[pos * half_dim + idx];
        float sin_val = sin_table[pos * half_dim + idx];

        row_out[i0] = x0 * cos_val - x1 * sin_val;
        row_out[i1] = x0 * sin_val + x1 * cos_val;
    }
}

/// Fused RMSNorm, QKV Linear GEMV, and RoPE kernel for compact ~350M models.
/// Fuses normalization, projections, and rotary positional embeddings in a single GPU pass.
kernel void fused_rmsnorm_qkv_rope_kernel(
    device const float* hidden_in   [[buffer(0)]],
    device const float* norm_weight [[buffer(1)]],
    device const float* w_q         [[buffer(2)]],
    device const float* w_k         [[buffer(3)]],
    device const float* w_v         [[buffer(4)]],
    device const float* cos_table   [[buffer(5)]],
    device const float* sin_table   [[buffer(6)]],
    device float*       q_out       [[buffer(7)]],
    device float*       k_out       [[buffer(8)]],
    device float*       v_out       [[buffer(9)]],
    constant uint&      hidden_dim  [[buffer(10)]],
    constant uint&      q_dim       [[buffer(11)]],
    constant uint&      kv_dim      [[buffer(12)]],
    constant uint&      head_dim    [[buffer(13)]],
    constant uint&      pos         [[buffer(14)]],
    constant float&     eps         [[buffer(15)]],
    uint                tid         [[thread_position_in_threadgroup]],
    uint                threads     [[threads_per_threadgroup]]
) {
    threadgroup float sdata[1024];
    threadgroup float shared_norm[2048];

    // 1. RMSNorm: Sum of squares in threadgroup
    float sum_sq = 0.0f;
    for (uint i = tid; i < hidden_dim; i += threads) {
        float val = hidden_in[i];
        sum_sq += val * val;
    }
    sdata[tid] = sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = threads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid < 32) {
        simdgroup_barrier(mem_flags::mem_none);
    }

    float mean_sq = sdata[0] / float(hidden_dim);
    float rms = rsqrt(mean_sq + eps);

    for (uint i = tid; i < hidden_dim; i += threads) {
        shared_norm[i] = hidden_in[i] * rms * norm_weight[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2. GEMV for V: v_out[row] = dot(shared_norm, w_v[row])
    for (uint row = tid; row < kv_dim; row += threads) {
        device const float* w_row = w_v + row * hidden_dim;
        float dot_v = 0.0f;
        for (uint d = 0; d < hidden_dim; ++d) {
            dot_v += shared_norm[d] * w_row[d];
        }
        v_out[row] = dot_v;
    }

    // 3. GEMV for Q + RoPE in registers
    uint half_dim = head_dim / 2;
    uint total_q_rotations = q_dim / 2;
    for (uint rot = tid; rot < total_q_rotations; rot += threads) {
        uint idx = rot % half_dim;
        uint head_offset = (rot / half_dim) * head_dim;
        uint r0 = head_offset + idx;
        uint r1 = head_offset + idx + half_dim;

        device const float* w_r0 = w_q + r0 * hidden_dim;
        device const float* w_r1 = w_q + r1 * hidden_dim;

        float q0 = 0.0f;
        float q1 = 0.0f;
        for (uint d = 0; d < hidden_dim; ++d) {
            float n_d = shared_norm[d];
            q0 += n_d * w_r0[d];
            q1 += n_d * w_r1[d];
        }

        float cos_val = cos_table[pos * half_dim + idx];
        float sin_val = sin_table[pos * half_dim + idx];

        q_out[r0] = q0 * cos_val - q1 * sin_val;
        q_out[r1] = q0 * sin_val + q1 * cos_val;
    }

    // 4. GEMV for K + RoPE in registers
    uint total_k_rotations = kv_dim / 2;
    for (uint rot = tid; rot < total_k_rotations; rot += threads) {
        uint idx = rot % half_dim;
        uint head_offset = (rot / half_dim) * head_dim;
        uint r0 = head_offset + idx;
        uint r1 = head_offset + idx + half_dim;

        device const float* w_r0 = w_k + r0 * hidden_dim;
        device const float* w_r1 = w_k + r1 * hidden_dim;

        float k0 = 0.0f;
        float k1 = 0.0f;
        for (uint d = 0; d < hidden_dim; ++d) {
            float n_d = shared_norm[d];
            k0 += n_d * w_r0[d];
            k1 += n_d * w_r1[d];
        }

        float cos_val = cos_table[pos * half_dim + idx];
        float sin_val = sin_table[pos * half_dim + idx];

        k_out[r0] = k0 * cos_val - k1 * sin_val;
        k_out[r1] = k0 * sin_val + k1 * cos_val;
    }
}

/// Generic linear GEMV projection kernel for single-token vector-matrix multiplication.
kernel void linear_gemv_kernel(
    device const float* in_vec  [[buffer(0)]],
    device const float* weight  [[buffer(1)]],
    device float*       out_vec [[buffer(2)]],
    constant uint&      in_dim  [[buffer(3)]],
    constant uint&      out_dim [[buffer(4)]],
    uint                gid     [[thread_position_in_grid]]
) {
    if (gid >= out_dim) return;

    device const float* w_row = weight + gid * in_dim;
    float sum = 0.0f;
    for (uint i = 0; i < in_dim; ++i) {
        sum += in_vec[i] * w_row[i];
    }
    out_vec[gid] = sum;
}
