pub const QK8_0: usize = 32;

/// Quantized 8-bit block with 32 elements and a 32-bit float scale.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlockQ8_0 {
    pub d: f32,
    pub qs: [i8; QK8_0],
}

impl BlockQ8_0 {
    /// Quantize a slice of 32 32-bit floats into a Q8_0 block.
    pub fn from_f32_slice(slice: &[f32]) -> Self {
        debug_assert_eq!(slice.len(), QK8_0);
        let mut amax = 0.0f32;
        for &x in slice {
            let ax = x.abs();
            if ax > amax {
                amax = ax;
            }
        }

        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let mut qs = [0i8; QK8_0];
        for i in 0..QK8_0 {
            let v = (slice[i] * id).round().clamp(-128.0, 127.0) as i8;
            qs[i] = v;
        }

        Self { d, qs }
    }

    /// Dequantize block into 32 32-bit floats.
    pub fn dequantize(&self, out: &mut [f32; QK8_0]) {
        for (o, &q) in out.iter_mut().zip(self.qs.iter()) {
            *o = self.d * (q as f32);
        }
    }
}

/// Quantizes a row of floats into a slice of Q8_0 blocks.
pub fn quantize_row_q8_0(xs: &[f32], blocks: &mut [BlockQ8_0]) {
    debug_assert_eq!(xs.len(), blocks.len() * QK8_0);
    for (chunk, b) in xs.as_chunks::<QK8_0>().0.iter().zip(blocks.iter_mut()) {
        *b = BlockQ8_0::from_f32_slice(chunk);
    }
}

/// Scalar reference dot product between a Q8_0 block and an unquantized f32 slice.
pub fn scalar_dot_q8_0(w: &BlockQ8_0, x: &[f32; QK8_0]) -> f32 {
    let mut sum = 0.0f32;
    for (&q, &v) in w.qs.iter().zip(x.iter()) {
        sum += (q as f32) * v;
    }
    w.d * sum
}

/// Scalar reference dot product between two Q8_0 blocks.
pub fn scalar_dot_q8_0_q8_0(w: &BlockQ8_0, x: &BlockQ8_0) -> f32 {
    let mut sum_i = 0i32;
    for i in 0..QK8_0 {
        sum_i += (w.qs[i] as i32) * (x.qs[i] as i32);
    }
    w.d * x.d * (sum_i as f32)
}

/// Scalar reference GEMV: out = W * x
pub fn scalar_gemv_q8_0(
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
        let mut sum = 0.0f32;
        for (b_idx, block) in row_weights.iter().enumerate() {
            let x_chunk: &[f32; QK8_0] = x[b_idx * QK8_0..(b_idx + 1) * QK8_0]
                .try_into()
                .expect("chunk size 32");
            sum += scalar_dot_q8_0(block, x_chunk);
        }
        out[r] = sum;
    }
}
