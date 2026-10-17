#include <metal_stdlib>
using namespace metal;

/// Threads per threadgroup for every kernel in this file. Must match
/// `FUSED_THREADS_PER_TG` on the host (fused_layer.rs); the kernels return early
/// otherwise instead of reading stale threadgroup scratch slots.
constant uint FUSED_THREADS_PER_TG = 256;
constant uint FUSED_SIMD_GROUPS = FUSED_THREADS_PER_TG / 32;

constant uint FUSED_MAX_THREADS_PER_TG = 1024;
constant uint FUSED_MAX_SIMD_GROUPS = FUSED_MAX_THREADS_PER_TG / 32;

static_assert(FUSED_THREADS_PER_TG % 32 == 0 && FUSED_THREADS_PER_TG <= FUSED_MAX_THREADS_PER_TG, "threadgroup size must be a whole number of SIMD groups");

/// RMSNorm scale of one row, reduced cooperatively by the threadgroup.
/// `simd_sum` + a single threadgroup stage replaces the old power-of-two-only
/// `for (s = threads / 2; ...)` tree, so any thread count stays correct.
static inline float rms_norm_scale(
    device const float* row,
    uint dim,
    float eps,
    threadgroup float* partial,
    uint tid,
    uint threads,
    uint sg,
    uint lane)
{
    uint simd_groups = threads / 32;

    float sum_sq = 0.0f;
    for (uint i = tid; i < dim; i += threads) {
        float v = row[i];
        sum_sq += v * v;
    }

    sum_sq = simd_sum(sum_sq);
    if (lane == 0) {
        partial[sg] = sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0) {
        float total = (lane < simd_groups) ? partial[lane] : 0.0f;
        total = simd_sum(total);
        if (lane == 0) {
            partial[0] = total;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    return rsqrt(partial[0] / float(dim) + eps);
}

/// One SIMD group computes the dot product of two output rows of `w` with the input
/// vector; lane `l` accumulates the float4 chunks `l, l + 32, ...` of both rows, so
/// every load is a full-width coalesced access (the old one-thread-per-row loop
/// strided by `dim * 4` bytes). `norm_weight4`/`rms` are applied to the input on the
/// fly, which keeps the normalised row out of threadgroup memory (the fixed
/// staging array there is what #50 removed).
static inline void gemv_two_rows(
    device const float4* x4,
    device const float4* norm_weight4,
    float rms,
    uint n4,
    device const float4* w0,
    device const float4* w1,
    uint lane,
    thread float& a0,
    thread float& a1)
{
    a0 = 0.0f;
    a1 = 0.0f;
    for (uint i = lane; i < n4; i += 32) {
        float4 xv = x4[i];
        if (norm_weight4 != nullptr) {
            xv = xv * rms * norm_weight4[i];
        }
        a0 += dot(xv, w0[i]);
        a1 += dot(xv, w1[i]);
    }
    a0 = simd_sum(a0);
    a1 = simd_sum(a1);
}

/// Fused RMSNorm and RoPE kernel for a single row, partitioned over its
/// `dim / 2` rotations across many threadgroups. Every threadgroup recomputes the
/// cheap RMS scalar of the row rather than materialising the normalised row.
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
    uint                sg        [[simdgroup_index_in_threadgroup]],
    uint                lane      [[thread_index_in_simdgroup]],
    uint                threads   [[threads_per_threadgroup]]
) {
    threadgroup float partial[FUSED_MAX_SIMD_GROUPS];

    if (threads != FUSED_THREADS_PER_TG) {
        return;
    }

    float rms = rms_norm_scale(in, dim, eps, partial, tid, threads, sg, lane);

    uint half_dim = head_dim / 2;
    uint total_rotations = dim / 2;

    for (uint rot = tid; rot < total_rotations; rot += threads) {
        uint idx = rot % half_dim;
        uint head_offset = (rot / half_dim) * head_dim;

        uint i0 = head_offset + idx;
        uint i1 = head_offset + idx + half_dim;

        float x0 = in[i0] * rms * weight[i0];
        float x1 = in[i1] * rms * weight[i1];

        float cos_val = cos_table[pos * half_dim + idx];
        float sin_val = sin_table[pos * half_dim + idx];

        out[i0] = x0 * cos_val - x1 * sin_val;
        out[i1] = x0 * sin_val + x1 * cos_val;
    }
}

/// Fused RMSNorm, linear GEMV and (optionally) RoPE for one projection matrix.
/// Work unit = one SIMD group = one output row pair: for Q/K the pair is the RoPE
/// rotation pair (idx, idx + head_dim / 2), for V it is two adjacent rows, so a
/// threadgroup covers GEMV_ROWS_PER_THREADGROUP (= FUSED_SIMD_GROUPS * 2) output rows.
kernel void fused_rmsnorm_qkv_rope_kernel(
    device const float* hidden_in   [[buffer(0)]],
    device const float* norm_weight [[buffer(1)]],
    device const float* w           [[buffer(2)]],
    device const float* cos_table   [[buffer(3)]],
    device const float* sin_table   [[buffer(4)]],
    device float*       out         [[buffer(5)]],
    constant uint&      hidden_dim  [[buffer(6)]],
    constant uint&      out_dim     [[buffer(7)]],
    constant uint&      head_dim    [[buffer(8)]],
    constant uint&      pos         [[buffer(9)]],
    constant float&     eps         [[buffer(10)]],
    constant uint&      apply_rope  [[buffer(11)]],
    uint                tg          [[threadgroup_position_in_grid]],
    uint                tid         [[thread_position_in_threadgroup]],
    uint                sg          [[simdgroup_index_in_threadgroup]],
    uint                lane        [[thread_index_in_simdgroup]],
    uint                threads     [[threads_per_threadgroup]]
) {
    threadgroup float partial[FUSED_MAX_SIMD_GROUPS];

    if (threads != FUSED_THREADS_PER_TG) {
        return;
    }

    float rms = rms_norm_scale(hidden_in, hidden_dim, eps, partial, tid, threads, sg, lane);

    uint units = out_dim / 2;
    uint half_dim = head_dim / 2;
    uint n4 = hidden_dim / 4;
    device const float4* x4 = (device const float4*)hidden_in;
    device const float4* norm_weight4 = (device const float4*)norm_weight;

    uint unit = tg * FUSED_SIMD_GROUPS + sg;
    if (unit >= units) {
        return;
    }

    uint idx = 0;
    uint r0;
    uint r1;
    if (apply_rope != 0) {
        idx = unit % half_dim;
        uint head_offset = (unit / half_dim) * head_dim;
        r0 = head_offset + idx;
        r1 = r0 + half_dim;
    } else {
        r0 = unit * 2;
        r1 = r0 + 1;
    }

    device const float4* w0 = (device const float4*)(w + r0 * hidden_dim);
    device const float4* w1 = (device const float4*)(w + r1 * hidden_dim);

    float a0;
    float a1;
    gemv_two_rows(x4, norm_weight4, rms, n4, w0, w1, lane, a0, a1);

    if (lane == 0) {
        if (apply_rope != 0) {
            float cos_val = cos_table[pos * half_dim + idx];
            float sin_val = sin_table[pos * half_dim + idx];
            out[r0] = a0 * cos_val - a1 * sin_val;
            out[r1] = a0 * sin_val + a1 * cos_val;
        } else {
            out[r0] = a0;
            out[r1] = a1;
        }
    }
}

/// Generic linear GEMV for a single already-normalised input vector. Same
/// SIMD-group-per-row-pair, float4 lane access and `simd_sum` reduction as the
/// fused GEMV above.
kernel void linear_gemv_kernel(
    device const float* in_vec  [[buffer(0)]],
    device const float* weight  [[buffer(1)]],
    device float*       out_vec [[buffer(2)]],
    constant uint&      in_dim  [[buffer(3)]],
    constant uint&      out_dim [[buffer(4)]],
    uint                tg      [[threadgroup_position_in_grid]],
    uint                sg      [[simdgroup_index_in_threadgroup]],
    uint                lane    [[thread_index_in_simdgroup]],
    uint                threads [[threads_per_threadgroup]]
) {
    if (threads != FUSED_THREADS_PER_TG) {
        return;
    }

    uint units = out_dim / 2;
    uint unit = tg * FUSED_SIMD_GROUPS + sg;
    if (unit >= units) {
        return;
    }

    uint n4 = in_dim / 4;
    uint r0 = unit * 2;
    device const float4* x4 = (device const float4*)in_vec;
    device const float4* w0 = (device const float4*)(weight + r0 * in_dim);
    device const float4* w1 = (device const float4*)(weight + (r0 + 1) * in_dim);

    float a0;
    float a1;
    gemv_two_rows(x4, nullptr, 1.0f, n4, w0, w1, lane, a0, a1);

    if (lane == 0) {
        out_vec[r0] = a0;
        out_vec[r0 + 1] = a1;
    }
}
