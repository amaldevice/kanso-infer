#include <metal_stdlib>
using namespace metal;

kernel void rope_kernel(
    device float*       q_k       [[buffer(0)]],
    device const float* cos_table [[buffer(1)]],
    device const float* sin_table [[buffer(2)]],
    constant uint&      head_dim  [[buffer(3)]],
    constant uint&      pos       [[buffer(4)]],
    uint                gid       [[thread_position_in_grid]]
) {
    uint half_dim = head_dim / 2;
    uint idx = gid % half_dim;
    uint head_offset = (gid / half_dim) * head_dim;

    uint i0 = head_offset + idx;
    uint i1 = head_offset + idx + half_dim;

    float x0 = q_k[i0];
    float x1 = q_k[i1];

    float cos_val = cos_table[pos * half_dim + idx];
    float sin_val = sin_table[pos * half_dim + idx];

    q_k[i0] = x0 * cos_val - x1 * sin_val;
    q_k[i1] = x0 * sin_val + x1 * cos_val;
}
