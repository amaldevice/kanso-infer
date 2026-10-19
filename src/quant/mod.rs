//! Hand-rolled Q8_0/Q4_K block kernels (NEON + scalar reference).
//!
//! Reference/benchmark path only (review F11, ticket #89): production quantized
//! matmul runs off candle's `QMatMul`, not these kernels, and `--backend neon`
//! refuses to start rather than silently running that same CPU path. The packed
//! layouts do not match, so these kernels cannot read the production
//! `LinearWeight` blocks in place: here `BlockQ8_0` is `{ d: f32, qs: [i8; 32] }`
//! (36 B) while candle stores `{ d: f16, qs: [i8; 32] }` (34 B), and `BlockQ4_K`
//! keeps `d`/`dmin` as f32 where candle keeps f16 — different block stride and
//! different scale width. Repacking every weight per token to bridge that would
//! cost more bandwidth than the kernel saves, so wiring would need a kernel that
//! consumes candle's f16-scale blocks directly (not done).
//!
//! They stay because `tests/test_neon_simd_gemv.rs` pins NEON-vs-scalar parity
//! through them and the perplexity bench measures them on synthesized blocks;
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
