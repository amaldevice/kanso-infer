#include <metal_stdlib>
using namespace metal;

kernel void paged_kv_gather_kernel(
    device const float* paged_pool     [[buffer(0)]],
    device const uint*  block_table    [[buffer(1)]],
    device float*       out_contiguous [[buffer(2)]],
    constant uint&      block_size     [[buffer(3)]],
    constant uint&      head_dim       [[buffer(4)]],
    uint                gid            [[thread_position_in_grid]]
) {
    uint token_pos = gid / head_dim;
    uint dim_idx = gid % head_dim;

    uint block_idx = token_pos / block_size;
    uint block_offset = token_pos % block_size;
    uint physical_block = block_table[block_idx];

    uint pool_offset = physical_block * (block_size * head_dim) + block_offset * head_dim + dim_idx;
    out_contiguous[gid] = paged_pool[pool_offset];
}
