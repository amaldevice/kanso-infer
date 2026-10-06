#include <metal_stdlib>
using namespace metal;

kernel void silu_gate_kernel(
    device const float* gate [[buffer(0)]],
    device const float* up   [[buffer(1)]],
    device float*       out  [[buffer(2)]],
    uint                gid  [[thread_position_in_grid]]
) {
    float g = gate[gid];
    float u = up[gid];
    // silu(g) = g / (1 + exp(-g))
    float silu_val = g / (1.0f + exp(-g));
    out[gid] = silu_val * u;
}
