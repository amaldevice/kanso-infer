#include <metal_stdlib>
using namespace metal;

/// Residual add of a projection output into the residual stream (ticket #110):
/// hidden[gid] += proj[gid] * scale. `scale` is the Granite `residual_scale`
/// multiplier (1.0 on plain llama) — the same factor the tensor path applies via
/// `Model::scale_residual` before the add. In-place on `hidden`, one thread per
/// element, so each thread reads and writes only its own lane.
kernel void residual_add_kernel(
    device const float* proj   [[buffer(0)]],
    device float*       hidden [[buffer(1)]],
    constant float&     scale  [[buffer(2)]],
    uint                gid    [[thread_position_in_grid]]
) {
    hidden[gid] = hidden[gid] + proj[gid] * scale;
}
