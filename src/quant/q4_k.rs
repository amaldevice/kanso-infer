pub const QK_K: usize = 256;
pub const K_SCALE_SIZE: usize = 12;

/// Quantized 4-bit block with 256 elements, 12-byte scales/mins, and 32-bit float super-block scales.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlockQ4_K {
    pub d: f32,
    pub dmin: f32,
    pub scales: [u8; K_SCALE_SIZE],
    pub qs: [u8; QK_K / 2], // 128 bytes = 256 4-bit values
}

/// Extracts (scale, min) pair for sub-block j in 0..8 from packed 12-byte scales.
#[inline(always)]
pub fn get_scale_min_k4(j: usize, q: &[u8; K_SCALE_SIZE]) -> (u8, u8) {
    if j < 4 {
        let d = q[j] & 63;
        let m = q[j + 4] & 63;
        (d, m)
    } else {
        let d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (d, m)
    }
}

impl BlockQ4_K {
    /// Dequantize block into 256 32-bit floats.
    pub fn dequantize(&self, out: &mut [f32; QK_K]) {
        let mut is = 0;
        let mut ys_index = 0;

        for j in (0..QK_K).step_by(64) {
            let q_sub = &self.qs[j / 2..j / 2 + 32];
            let (sc1, m1) = get_scale_min_k4(is, &self.scales);
            let d1 = self.d * (sc1 as f32);
            let min1 = self.dmin * (m1 as f32);

            let (sc2, m2) = get_scale_min_k4(is + 1, &self.scales);
            let d2 = self.d * (sc2 as f32);
            let min2 = self.dmin * (m2 as f32);

            for &b in q_sub {
                out[ys_index] = d1 * ((b & 0xF) as f32) - min1;
                ys_index += 1;
            }
            for &b in q_sub {
                out[ys_index] = d2 * ((b >> 4) as f32) - min2;
                ys_index += 1;
            }
            is += 2;
        }
    }
}

/// Scalar reference dot product between a Q4_K block and an unquantized f32 slice.
pub fn scalar_dot_q4_k(w: &BlockQ4_K, x: &[f32; QK_K]) -> f32 {
    let mut dequant = [0.0f32; QK_K];
    w.dequantize(&mut dequant);
    let mut sum = 0.0f32;
    for i in 0..QK_K {
        sum += dequant[i] * x[i];
    }
    sum
}

/// Scalar reference GEMV: out = W * x for Q4_K weights
pub fn scalar_gemv_q4_k(
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
        let mut sum = 0.0f32;
        for (b_idx, block) in row_weights.iter().enumerate() {
            let x_chunk: &[f32; QK_K] = x[b_idx * QK_K..(b_idx + 1) * QK_K]
                .try_into()
                .expect("chunk size 256");
            sum += scalar_dot_q4_k(block, x_chunk);
        }
        out[r] = sum;
    }
}
