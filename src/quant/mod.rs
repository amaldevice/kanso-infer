//! Hand-rolled Q8_0/Q4_K block kernels (NEON + scalar reference).
//!
//! Reference/benchmark path (review F11): production quantized matmul runs off
//! candle's `QMatMul`, not these kernels. They stay because
//! `tests/test_neon_simd_gemv.rs` pins NEON-vs-scalar parity through them —
//! delete that test first if these ever go.

pub mod neon;
pub mod q4_k;
pub mod q8_0;

pub use neon::{neon_dot_q4_k, neon_dot_q8_0, neon_dot_q8_0_q8_0, neon_gemv_q4_k, neon_gemv_q8_0};
pub use q4_k::{
    BlockQ4_K, K_SCALE_SIZE, QK_K, get_scale_min_k4, scalar_dot_q4_k, scalar_gemv_q4_k,
};
pub use q8_0::{
    BlockQ8_0, QK8_0, quantize_row_q8_0, scalar_dot_q8_0, scalar_dot_q8_0_q8_0, scalar_gemv_q8_0,
};
