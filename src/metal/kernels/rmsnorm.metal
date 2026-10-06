#include <metal_stdlib>
using namespace metal;

kernel void rmsnorm_kernel(
    device const float* in       [[buffer(0)]],
    device const float* weight   [[buffer(1)]],
    device float*       out      [[buffer(2)]],
    constant uint&      dim      [[buffer(3)]],
    constant float&     eps      [[buffer(4)]],
    uint                tid      [[thread_position_in_threadgroup]],
    uint                bid      [[threadgroup_position_in_grid]],
    uint                threads  [[threads_per_threadgroup]]
) {
    device const float* row_in = in + bid * dim;
    device float* row_out = out + bid * dim;

    threadgroup float sdata[256];
    float sum_sq = 0.0f;

    for (uint i = tid; i < dim; i += threads) {
        float val = row_in[i];
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

    float mean_sq = sdata[0] / float(dim);
    float rms = rsqrt(mean_sq + eps);

    for (uint i = tid; i < dim; i += threads) {
        row_out[i] = row_in[i] * rms * weight[i];
    }
}
