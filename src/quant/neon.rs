#![allow(unused_imports, unsafe_op_in_unsafe_fn)]

use super::q4_k::{get_scale_min_k4, scalar_dot_q4_k, scalar_gemv_q4_k, BlockQ4_K, QK_K};
use super::q8_0::{
    scalar_dot_q8_0, scalar_dot_q8_0_q8_0, scalar_gemv_q8_0, BlockQ8_0, QK8_0,
};

/// Dot product between a Q8_0 block and an unquantized f32 slice.
/// Uses ARM NEON vfmaq_f32 on aarch64, or falls back to scalar reference.
pub fn neon_dot_q8_0(w: &BlockQ8_0, x: &[f32; QK8_0]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        neon_dot_q8_0_arch(w, x)
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        scalar_dot_q8_0(w, x)
    }
}

/// Dot product between two Q8_0 blocks.
/// Uses ARM NEON vdotq_s32 on aarch64, or falls back to scalar reference.
pub fn neon_dot_q8_0_q8_0(w: &BlockQ8_0, x: &BlockQ8_0) -> f32 {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        neon_dot_q8_0_q8_0_arch(w, x)
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        scalar_dot_q8_0_q8_0(w, x)
    }
}

/// Dot product between a Q4_K block and an unquantized f32 slice.
/// Uses ARM NEON vfmaq_f32 on aarch64, or falls back to scalar reference.
pub fn neon_dot_q4_k(w: &BlockQ4_K, x: &[f32; QK_K]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        neon_dot_q4_k_arch(w, x)
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        scalar_dot_q4_k(w, x)
    }
}

/// Quantized matrix-vector multiplication for Q8_0 weights: out = W * x
pub fn neon_gemv_q8_0(
    weights: &[BlockQ8_0],
    x: &[f32],
    out: &mut [f32],
    rows: usize,
    cols: usize,
) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        neon_gemv_q8_0_arch(weights, x, out, rows, cols);
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        scalar_gemv_q8_0(weights, x, out, rows, cols);
    }
}

/// Quantized matrix-vector multiplication for Q4_K weights: out = W * x
pub fn neon_gemv_q4_k(
    weights: &[BlockQ4_K],
    x: &[f32],
    out: &mut [f32],
    rows: usize,
    cols: usize,
) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        neon_gemv_q4_k_arch(weights, x, out, rows, cols);
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        scalar_gemv_q4_k(weights, x, out, rows, cols);
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn neon_dot_q8_0_arch(w: &BlockQ8_0, x: &[f32; QK8_0]) -> f32 {
    use std::arch::aarch64::*;
    let w_ptr = w.qs.as_ptr();
    let x_ptr = x.as_ptr();

    let raw0 = vld1q_s8(w_ptr);
    let raw1 = vld1q_s8(w_ptr.add(16));

    let s16_0 = vmovl_s8(vget_low_s8(raw0));
    let s16_1 = vmovl_s8(vget_high_s8(raw0));
    let f32_0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(s16_0)));
    let f32_1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(s16_0)));
    let f32_2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(s16_1)));
    let f32_3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(s16_1)));

    let mut acc = vdupq_n_f32(0.0);
    acc = vfmaq_f32(acc, f32_0, vld1q_f32(x_ptr));
    acc = vfmaq_f32(acc, f32_1, vld1q_f32(x_ptr.add(4)));
    acc = vfmaq_f32(acc, f32_2, vld1q_f32(x_ptr.add(8)));
    acc = vfmaq_f32(acc, f32_3, vld1q_f32(x_ptr.add(12)));

    let s16_2 = vmovl_s8(vget_low_s8(raw1));
    let s16_3 = vmovl_s8(vget_high_s8(raw1));
    let f32_4 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(s16_2)));
    let f32_5 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(s16_2)));
    let f32_6 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(s16_3)));
    let f32_7 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(s16_3)));

    acc = vfmaq_f32(acc, f32_4, vld1q_f32(x_ptr.add(16)));
    acc = vfmaq_f32(acc, f32_5, vld1q_f32(x_ptr.add(20)));
    acc = vfmaq_f32(acc, f32_6, vld1q_f32(x_ptr.add(24)));
    acc = vfmaq_f32(acc, f32_7, vld1q_f32(x_ptr.add(28)));

    vaddvq_f32(acc) * w.d
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn neon_dot_q8_0_q8_0_arch(w: &BlockQ8_0, x: &BlockQ8_0) -> f32 {
    use std::arch::aarch64::*;
    let w_ptr = w.qs.as_ptr();
    let x_ptr = x.qs.as_ptr();

    let w0 = vld1q_s8(w_ptr);
    let w1 = vld1q_s8(w_ptr.add(16));
    let x0 = vld1q_s8(x_ptr);
    let x1 = vld1q_s8(x_ptr.add(16));

    let zero = vdupq_n_s32(0);
    let p0 = vdotq_s32(zero, w0, x0);
    let p1 = vdotq_s32(p0, w1, x1);

    let sum_i = vaddvq_s32(p1);
    (w.d * x.d) * (sum_i as f32)
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn neon_dot_q4_k_arch(w: &BlockQ4_K, x: &[f32; QK_K]) -> f32 {
    use std::arch::aarch64::*;
    let mut is = 0;
    let mut acc = vdupq_n_f32(0.0);
    let mask_low = vdupq_n_u8(0x0F);

    for j in (0..QK_K).step_by(64) {
        let q_ptr = w.qs.as_ptr().add(j / 2);
        let (sc1, m1) = get_scale_min_k4(is, &w.scales);
        let d1 = w.d * (sc1 as f32);
        let min1 = w.dmin * (m1 as f32);

        let (sc2, m2) = get_scale_min_k4(is + 1, &w.scales);
        let d2 = w.d * (sc2 as f32);
        let min2 = w.dmin * (m2 as f32);

        let vd1 = vdupq_n_f32(d1);
        let vmin1 = vdupq_n_f32(min1);
        let vd2 = vdupq_n_f32(d2);
        let vmin2 = vdupq_n_f32(min2);

        let raw0 = vld1q_u8(q_ptr);
        let raw1 = vld1q_u8(q_ptr.add(16));

        let low0 = vandq_u8(raw0, mask_low);
        let low1 = vandq_u8(raw1, mask_low);
        let hi0 = vshrq_n_u8(raw0, 4);
        let hi1 = vshrq_n_u8(raw1, 4);

        // Process low nibbles (first 32 elements: j..j+32)
        let x_ptr1 = x.as_ptr().add(j);
        let u16_0 = vmovl_u8(vget_low_u8(low0));
        let u16_1 = vmovl_u8(vget_high_u8(low0));
        let w0 = vsubq_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(u16_0))), vd1), vmin1);
        let w1 = vsubq_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(u16_0))), vd1), vmin1);
        let w2 = vsubq_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(u16_1))), vd1), vmin1);
        let w3 = vsubq_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(u16_1))), vd1), vmin1);
        acc = vfmaq_f32(acc, w0, vld1q_f32(x_ptr1));
        acc = vfmaq_f32(acc, w1, vld1q_f32(x_ptr1.add(4)));
        acc = vfmaq_f32(acc, w2, vld1q_f32(x_ptr1.add(8)));
        acc = vfmaq_f32(acc, w3, vld1q_f32(x_ptr1.add(12)));

        let u16_2 = vmovl_u8(vget_low_u8(low1));
        let u16_3 = vmovl_u8(vget_high_u8(low1));
        let w4 = vsubq_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(u16_2))), vd1), vmin1);
        let w5 = vsubq_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(u16_2))), vd1), vmin1);
        let w6 = vsubq_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(u16_3))), vd1), vmin1);
        let w7 = vsubq_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(u16_3))), vd1), vmin1);
        acc = vfmaq_f32(acc, w4, vld1q_f32(x_ptr1.add(16)));
        acc = vfmaq_f32(acc, w5, vld1q_f32(x_ptr1.add(20)));
        acc = vfmaq_f32(acc, w6, vld1q_f32(x_ptr1.add(24)));
        acc = vfmaq_f32(acc, w7, vld1q_f32(x_ptr1.add(28)));

        // Process high nibbles (next 32 elements: j+32..j+64)
        let x_ptr2 = x.as_ptr().add(j + 32);
        let u16_4 = vmovl_u8(vget_low_u8(hi0));
        let u16_5 = vmovl_u8(vget_high_u8(hi0));
        let w8 = vsubq_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(u16_4))), vd2), vmin2);
        let w9 = vsubq_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(u16_4))), vd2), vmin2);
        let w10 = vsubq_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(u16_5))), vd2), vmin2);
        let w11 = vsubq_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(u16_5))), vd2), vmin2);
        acc = vfmaq_f32(acc, w8, vld1q_f32(x_ptr2));
        acc = vfmaq_f32(acc, w9, vld1q_f32(x_ptr2.add(4)));
        acc = vfmaq_f32(acc, w10, vld1q_f32(x_ptr2.add(8)));
        acc = vfmaq_f32(acc, w11, vld1q_f32(x_ptr2.add(12)));

        let u16_6 = vmovl_u8(vget_low_u8(hi1));
        let u16_7 = vmovl_u8(vget_high_u8(hi1));
        let w12 = vsubq_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(u16_6))), vd2), vmin2);
        let w13 = vsubq_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(u16_6))), vd2), vmin2);
        let w14 = vsubq_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(u16_7))), vd2), vmin2);
        let w15 = vsubq_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(u16_7))), vd2), vmin2);
        acc = vfmaq_f32(acc, w12, vld1q_f32(x_ptr2.add(16)));
        acc = vfmaq_f32(acc, w13, vld1q_f32(x_ptr2.add(20)));
        acc = vfmaq_f32(acc, w14, vld1q_f32(x_ptr2.add(24)));
        acc = vfmaq_f32(acc, w15, vld1q_f32(x_ptr2.add(28)));

        is += 2;
    }

    vaddvq_f32(acc)
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn neon_gemv_q8_0_arch(
    weights: &[BlockQ8_0],
    x: &[f32],
    out: &mut [f32],
    rows: usize,
    cols: usize,
) {
    debug_assert_eq!(cols % QK8_0, 0);
    let blocks_per_row = cols / QK8_0;
    debug_assert_eq!(weights.len(), rows * blocks_per_row);
    debug_assert_eq!(x.len(), cols);
    debug_assert_eq!(out.len(), rows);

    for r in 0..rows {
        let row_weights = &weights[r * blocks_per_row..(r + 1) * blocks_per_row];
        let mut row_sum = 0.0f32;
        for (b_idx, block) in row_weights.iter().enumerate() {
            let x_chunk: &[f32; QK8_0] = x[b_idx * QK8_0..(b_idx + 1) * QK8_0]
                .try_into()
                .expect("chunk size 32");
            row_sum += neon_dot_q8_0_arch(block, x_chunk);
        }
        out[r] = row_sum;
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn neon_gemv_q4_k_arch(
    weights: &[BlockQ4_K],
    x: &[f32],
    out: &mut [f32],
    rows: usize,
    cols: usize,
) {
    debug_assert_eq!(cols % QK_K, 0);
    let blocks_per_row = cols / QK_K;
    debug_assert_eq!(weights.len(), rows * blocks_per_row);
    debug_assert_eq!(x.len(), cols);
    debug_assert_eq!(out.len(), rows);

    for r in 0..rows {
        let row_weights = &weights[r * blocks_per_row..(r + 1) * blocks_per_row];
        let mut row_sum = 0.0f32;
        for (b_idx, block) in row_weights.iter().enumerate() {
            let x_chunk: &[f32; QK_K] = x[b_idx * QK_K..(b_idx + 1) * QK_K]
                .try_into()
                .expect("chunk size 256");
            row_sum += neon_dot_q4_k_arch(block, x_chunk);
        }
        out[r] = row_sum;
    }
}
