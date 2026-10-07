use kanso_infer::quant::{
    neon_dot_q4_k, neon_dot_q8_0, neon_dot_q8_0_q8_0, neon_gemv_q4_k, neon_gemv_q8_0,
    quantize_row_q8_0, scalar_dot_q4_k, scalar_dot_q8_0, scalar_dot_q8_0_q8_0,
    scalar_gemv_q4_k, scalar_gemv_q8_0, BlockQ4_K, BlockQ8_0, QK8_0, QK_K,
};
use std::time::Instant;

#[test]
fn test_q8_0_block_dot_product_parity() {
    let mut qs = [0i8; QK8_0];
    for (i, q) in qs.iter_mut().enumerate() {
        *q = ((i as i32 * 7) % 255 - 128) as i8;
    }
    let w = BlockQ8_0 { d: 0.025, qs };

    let mut x = [0.0f32; QK8_0];
    for (i, val) in x.iter_mut().enumerate() {
        *val = ((i as f32 * 0.13).sin()) * 2.0;
    }

    let scalar_res = scalar_dot_q8_0(&w, &x);
    let neon_res = neon_dot_q8_0(&w, &x);

    assert!(
        (neon_res - scalar_res).abs() < 1e-4,
        "Q8_0 dot mismatch: neon={}, scalar={}, diff={}",
        neon_res,
        scalar_res,
        (neon_res - scalar_res).abs()
    );

    // Also test Q8_0 x Q8_0 dot product
    let x_block = BlockQ8_0::from_f32_slice(&x);
    let scalar_q8_res = scalar_dot_q8_0_q8_0(&w, &x_block);
    let neon_q8_res = neon_dot_q8_0_q8_0(&w, &x_block);

    assert!(
        (neon_q8_res - scalar_q8_res).abs() < 1e-4,
        "Q8_0 x Q8_0 dot mismatch: neon={}, scalar={}, diff={}",
        neon_q8_res,
        scalar_q8_res,
        (neon_q8_res - scalar_q8_res).abs()
    );
}

#[test]
fn test_quantize_row_q8_0() {
    let mut xs = vec![0.0f32; 64];
    for (i, x) in xs.iter_mut().enumerate() {
        *x = (i as f32) * 0.1 - 3.2;
    }
    let mut blocks = vec![BlockQ8_0 { d: 0.0, qs: [0; QK8_0] }; 2];
    quantize_row_q8_0(&xs, &mut blocks);

    assert!(blocks[0].d > 0.0);
    assert!(blocks[1].d > 0.0);

    let mut dequant = [0.0f32; QK8_0];
    blocks[0].dequantize(&mut dequant);
    for i in 0..QK8_0 {
        assert!((dequant[i] - xs[i]).abs() < blocks[0].d);
    }
}

#[test]
fn test_q4_k_block_dot_product_parity() {
    let mut qs = [0u8; QK_K / 2];
    for (i, q) in qs.iter_mut().enumerate() {
        *q = ((i * 17 + 5) & 0xFF) as u8;
    }

    let scales = [12, 24, 35, 48, 15, 20, 25, 30, 40, 50, 10, 5];
    let w = BlockQ4_K {
        d: 0.05,
        dmin: 0.02,
        scales,
        qs,
    };

    let mut x = [0.0f32; QK_K];
    for (i, val) in x.iter_mut().enumerate() {
        *val = ((i as f32 * 0.07).cos()) * 1.5;
    }

    let scalar_res = scalar_dot_q4_k(&w, &x);
    let neon_res = neon_dot_q4_k(&w, &x);

    assert!(
        (neon_res - scalar_res).abs() < 1e-3,
        "Q4_K dot mismatch: neon={}, scalar={}, diff={}",
        neon_res,
        scalar_res,
        (neon_res - scalar_res).abs()
    );
}

#[test]
fn test_neon_gemv_q8_0_parity() {
    let rows = 64;
    let cols = 128;
    let blocks_per_row = cols / QK8_0;

    let mut weights = Vec::with_capacity(rows * blocks_per_row);
    for r in 0..rows {
        for b in 0..blocks_per_row {
            let mut qs = [0i8; QK8_0];
            for (i, q) in qs.iter_mut().enumerate() {
                *q = (((r * 31 + b * 17 + i * 5) % 255) as i32 - 128) as i8;
            }
            weights.push(BlockQ8_0 {
                d: 0.01 + (r as f32 * 0.001),
                qs,
            });
        }
    }

    let mut x = vec![0.0f32; cols];
    for (i, val) in x.iter_mut().enumerate() {
        *val = ((i as f32 * 0.1).sin()) * 1.2;
    }

    let mut scalar_out = vec![0.0f32; rows];
    let mut neon_out = vec![0.0f32; rows];

    scalar_gemv_q8_0(&weights, &x, &mut scalar_out, rows, cols);
    neon_gemv_q8_0(&weights, &x, &mut neon_out, rows, cols);

    for r in 0..rows {
        assert!(
            (neon_out[r] - scalar_out[r]).abs() < 1e-3,
            "GEMV Q8_0 mismatch at row {}: neon={}, scalar={}",
            r,
            neon_out[r],
            scalar_out[r]
        );
    }
}

#[test]
fn test_neon_gemv_q4_k_parity() {
    let rows = 32;
    let cols = 256;
    let blocks_per_row = cols / QK_K;

    let mut weights = Vec::with_capacity(rows * blocks_per_row);
    for r in 0..rows {
        for _ in 0..blocks_per_row {
            let mut qs = [0u8; QK_K / 2];
            for (i, q) in qs.iter_mut().enumerate() {
                *q = ((r * 13 + i * 7) & 0xFF) as u8;
            }
            let scales = [10, 20, 30, 40, 15, 25, 35, 45, 5, 12, 18, 22];
            weights.push(BlockQ4_K {
                d: 0.02 + (r as f32 * 0.001),
                dmin: 0.01,
                scales,
                qs,
            });
        }
    }

    let mut x = vec![0.0f32; cols];
    for (i, val) in x.iter_mut().enumerate() {
        *val = ((i as f32 * 0.05).cos()) * 2.0;
    }

    let mut scalar_out = vec![0.0f32; rows];
    let mut neon_out = vec![0.0f32; rows];

    scalar_gemv_q4_k(&weights, &x, &mut scalar_out, rows, cols);
    neon_gemv_q4_k(&weights, &x, &mut neon_out, rows, cols);

    for r in 0..rows {
        assert!(
            (neon_out[r] - scalar_out[r]).abs() < 1e-3,
            "GEMV Q4_K mismatch at row {}: neon={}, scalar={}",
            r,
            neon_out[r],
            scalar_out[r]
        );
    }
}

#[test]
fn test_single_token_decode_gemv_speedup_over_fp32() {
    // Hidden dimension representative of small/medium model layer (e.g. 1024 x 1024)
    let rows = 1024;
    let cols = 1024;
    let blocks_per_row = cols / QK8_0;

    // Unquantized FP32 weights baseline
    let mut fp32_weights = vec![0.0f32; rows * cols];
    let mut q8_weights = Vec::with_capacity(rows * blocks_per_row);

    for r in 0..rows {
        for b in 0..blocks_per_row {
            let mut qs = [0i8; QK8_0];
            let d = 0.015f32;
            for i in 0..QK8_0 {
                let q_val = (((r * 17 + b * 11 + i * 3) % 255) as i32 - 128) as i8;
                qs[i] = q_val;
                fp32_weights[r * cols + b * QK8_0 + i] = d * (q_val as f32);
            }
            q8_weights.push(BlockQ8_0 { d, qs });
        }
    }

    let mut x = vec![0.0f32; cols];
    for (i, val) in x.iter_mut().enumerate() {
        *val = ((i as f32 * 0.02).sin()) * 1.0;
    }

    let mut fp32_out = vec![0.0f32; rows];
    let mut neon_out = vec![0.0f32; rows];

    // Warm up
    for _ in 0..5 {
        for r in 0..rows {
            let row = &fp32_weights[r * cols..(r + 1) * cols];
            let mut sum = 0.0f32;
            for c in 0..cols {
                sum += row[c] * x[c];
            }
            fp32_out[r] = sum;
        }
        neon_gemv_q8_0(&q8_weights, &x, &mut neon_out, rows, cols);
    }

    // Benchmark FP32 baseline
    let iters = 50;
    let t0 = Instant::now();
    for _ in 0..iters {
        for r in 0..rows {
            let row = &fp32_weights[r * cols..(r + 1) * cols];
            let mut sum = 0.0f32;
            for c in 0..cols {
                sum += row[c] * x[c];
            }
            fp32_out[r] = sum;
        }
    }
    let fp32_duration = t0.elapsed();

    // Benchmark NEON GEMV
    let t1 = Instant::now();
    for _ in 0..iters {
        neon_gemv_q8_0(&q8_weights, &x, &mut neon_out, rows, cols);
    }
    let neon_duration = t1.elapsed();

    let speedup = fp32_duration.as_secs_f64() / neon_duration.as_secs_f64();
    println!(
        "FP32 latency: {:?}, NEON Q8_0 latency: {:?}, Speedup: {:.2}x",
        fp32_duration / iters,
        neon_duration / iters,
        speedup
    );

    // Verify output numerical consistency
    for r in 0..rows {
        assert!(
            (neon_out[r] - fp32_out[r]).abs() < 1e-2,
            "Row {} mismatch: neon={}, fp32={}",
            r,
            neon_out[r],
            fp32_out[r]
        );
    }

    // Verify significant speedup over FP32 baseline (at least 1.5x in unoptimized debug build)
    assert!(
        speedup >= 1.5,
        "Expected significant speedup over FP32 baseline (>= 1.5x), got {:.2}x",
        speedup
    );
}
