use candle_core::{Device, Tensor};
use kanso_infer::accelerate::amx::{amx_gemm, amx_gemv, amx_matmul, has_accelerate_support};
use kanso_infer::bench::create_engine_for_model_profile;
use kanso_infer::model::Model;
use std::time::Instant;

fn cpu_reference_gemm(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    // A is [M, K], B is [K, N], C is [M, N] in row-major
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

fn cpu_reference_gemv(m: usize, k: usize, a: &[f32], x: &[f32], y: &mut [f32]) {
    // A is [M, K], x is [K], y is [M]
    for i in 0..m {
        let mut sum = 0.0f32;
        for p in 0..k {
            sum += a[i * k + p] * x[p];
        }
        y[i] = sum;
    }
}

#[test]
fn test_amx_gemm_parity_against_cpu_reference() {
    let m = 8;
    let n = 16;
    let k = 32;

    let mut a = vec![0.0f32; m * k];
    for (i, val) in a.iter_mut().enumerate() {
        *val = ((i as f32 * 0.17).sin()) * 1.5;
    }

    let mut b = vec![0.0f32; k * n];
    for (i, val) in b.iter_mut().enumerate() {
        *val = ((i as f32 * 0.23).cos()) * 2.0;
    }

    let mut c_cpu = vec![0.0f32; m * n];
    cpu_reference_gemm(m, n, k, &a, &b, &mut c_cpu);

    let mut c_amx = vec![0.0f32; m * n];
    amx_gemm(m, n, k, &a, &b, &mut c_amx);

    for idx in 0..(m * n) {
        let diff = (c_amx[idx] - c_cpu[idx]).abs();
        assert!(
            diff <= 1e-4,
            "AMX GEMM mismatch at index {}: amx={}, cpu={}, diff={}",
            idx,
            c_amx[idx],
            c_cpu[idx],
            diff
        );
    }
}

#[test]
fn test_amx_gemm_single_token_m_equals_1_dispatch_sgemv() {
    let m = 1;
    let n = 32;
    let k = 64;

    let mut a = vec![0.0f32; m * k];
    for (i, val) in a.iter_mut().enumerate() {
        *val = ((i as f32 * 0.17).sin()) * 1.5;
    }

    let mut b = vec![0.0f32; k * n];
    for (i, val) in b.iter_mut().enumerate() {
        *val = ((i as f32 * 0.23).cos()) * 2.0;
    }

    let mut c_cpu = vec![0.0f32; m * n];
    cpu_reference_gemm(m, n, k, &a, &b, &mut c_cpu);

    let mut c_amx = vec![0.0f32; m * n];
    amx_gemm(m, n, k, &a, &b, &mut c_amx);

    for idx in 0..(m * n) {
        let diff = (c_amx[idx] - c_cpu[idx]).abs();
        assert!(
            diff <= 1e-4,
            "AMX GEMM M=1 mismatch at index {}: amx={}, cpu={}, diff={}",
            idx,
            c_amx[idx],
            c_cpu[idx],
            diff
        );
    }
}

#[test]
fn test_amx_gemv_parity_against_cpu_reference() {
    let m = 64;
    let k = 128;

    let mut a = vec![0.0f32; m * k];
    for (i, val) in a.iter_mut().enumerate() {
        *val = ((i as f32 * 0.31).sin()) * 0.8;
    }

    let mut x = vec![0.0f32; k];
    for (i, val) in x.iter_mut().enumerate() {
        *val = ((i as f32 * 0.19).cos()) * 1.2;
    }

    let mut y_cpu = vec![0.0f32; m];
    cpu_reference_gemv(m, k, &a, &x, &mut y_cpu);

    let mut y_amx = vec![0.0f32; m];
    amx_gemv(m, k, &a, &x, &mut y_amx);

    for idx in 0..m {
        let diff = (y_amx[idx] - y_cpu[idx]).abs();
        assert!(
            diff <= 1e-4,
            "AMX GEMV mismatch at index {}: amx={}, cpu={}, diff={}",
            idx,
            y_amx[idx],
            y_cpu[idx],
            diff
        );
    }
}

#[test]
fn test_model_forward_amx_backend_support() {
    let mut model = Model::create_mock_tiny(64, 4, 2, 2, 256).expect("model creation");
    model.set_backend(kanso_infer::Backend::Amx);
    assert_eq!(model.backend(), kanso_infer::Backend::Amx);

    let prompt = vec![1u32, 5, 42];
    let logits_amx = model.forward_single_prompt(&prompt).expect("amx forward");
    assert_eq!(logits_amx.dims(), &[1, 256]);

    // Check parity with default/cpu
    let mut cpu_model = Model::create_mock_tiny(64, 4, 2, 2, 256).expect("model creation");
    cpu_model.set_backend(kanso_infer::Backend::Cpu);
    let logits_cpu = cpu_model
        .forward_single_prompt(&prompt)
        .expect("cpu forward");

    let vals_amx = logits_amx.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let vals_cpu = logits_cpu.flatten_all().unwrap().to_vec1::<f32>().unwrap();

    for (idx, (&a, &c)) in vals_amx.iter().zip(vals_cpu.iter()).enumerate() {
        let diff = (a - c).abs();
        assert!(
            diff <= 1e-4,
            "Model forward mismatch at index {}: amx={}, cpu={}, diff={}",
            idx,
            a,
            c,
            diff
        );
    }
}

/// The decode path feeds `amx_matmul` a `.t()` weight view; the matmul now consumes the
/// weight's own storage via `CblasTrans` instead of copying it, so parity with the
/// transposed-view reference must hold for non-square shapes.
#[test]
fn test_amx_matmul_transposed_weight_matches_candle() {
    let (m, k, n) = (3usize, 37usize, 11usize);
    let mut x = vec![0.0f32; m * k];
    for (i, v) in x.iter_mut().enumerate() {
        *v = ((i as f32) * 0.31).sin();
    }
    let mut w = vec![0.0f32; n * k];
    for (i, v) in w.iter_mut().enumerate() {
        *v = ((i as f32) * 0.19).cos() * 0.5;
    }

    let x = Tensor::from_vec(x, (m, k), &Device::Cpu).unwrap();
    let w = Tensor::from_vec(w, (n, k), &Device::Cpu).unwrap();
    let w_t = w.t().unwrap();

    let expected = x.matmul(&w_t).unwrap().flatten_all().unwrap();
    let expected = expected.to_vec1::<f32>().unwrap();
    let got = amx_matmul(&x, &w_t).unwrap().flatten_all().unwrap();
    let got = got.to_vec1::<f32>().unwrap();

    assert_eq!(got.len(), m * n);
    for (idx, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - e).abs() <= 1e-4 * e.abs().max(1.0),
            "transposed-weight mismatch at {idx}: amx={g} candle={e}"
        );
    }
}

/// Batched attention scores: `[B, M, K] x [B, K, N]` where the right operand is a
/// per-batch transpose view of a `[B, N, K]` block.
#[test]
fn test_amx_matmul_batched_transposed_matches_candle() {
    let (batch, m, k, n) = (2usize, 4usize, 9usize, 5usize);
    let mut q = vec![0.0f32; batch * m * k];
    for (i, v) in q.iter_mut().enumerate() {
        *v = ((i as f32) * 0.13).sin();
    }
    let mut kk = vec![0.0f32; batch * n * k];
    for (i, v) in kk.iter_mut().enumerate() {
        *v = ((i as f32) * 0.27).cos();
    }

    let q = Tensor::from_vec(q, (batch, m, k), &Device::Cpu).unwrap();
    let kk = Tensor::from_vec(kk, (batch, n, k), &Device::Cpu).unwrap();
    let k_t = kk.transpose(1, 2).unwrap();

    let expected = q.matmul(&k_t).unwrap().flatten_all().unwrap();
    let expected = expected.to_vec1::<f32>().unwrap();
    let got = amx_matmul(&q, &k_t).unwrap().flatten_all().unwrap();
    let got = got.to_vec1::<f32>().unwrap();

    assert_eq!(got.len(), batch * m * n);
    for (idx, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - e).abs() <= 1e-4 * e.abs().max(1.0),
            "batched mismatch at {idx}: amx={g} candle={e}"
        );
    }
}

#[test]
fn test_bench_engine_creation_with_amx_backend() {
    let engine =
        create_engine_for_model_profile("smollm2", kanso_infer::Backend::Amx, 2, 64, 16, 4)
            .expect("engine creation with amx backend");
    assert_eq!(engine.model.backend(), kanso_infer::Backend::Amx);
}

#[test]
fn test_amx_throughput_speedup_over_neon_large_matrices() {
    if !has_accelerate_support() {
        eprintln!("Skipping AMX throughput test on non-macOS environment");
        return;
    }

    // Dimensions M, N, K >= 960 (matching SmolLM2 hidden_size 960)
    let m = 960;
    let n = 960;
    let k = 960;

    let a = vec![0.5f32; m * k];
    let b = vec![0.25f32; k * n];
    let mut c_amx = vec![0.0f32; m * n];

    // Warmup AMX
    amx_gemm(m, n, k, &a, &b, &mut c_amx);

    // Benchmark AMX 2D GEMM (5 iterations)
    let iters = 5;
    let t_amx = Instant::now();
    for _ in 0..iters {
        amx_gemm(m, n, k, &a, &b, &mut c_amx);
    }
    let dur_amx = t_amx.elapsed().as_secs_f64();
    let amx_throughput_gflops =
        (2.0 * m as f64 * n as f64 * k as f64 * iters as f64 / dur_amx) / 1e9;

    // Benchmark 1D NEON SIMD vector operations on a subset of rows to extrapolate full matrix time
    let test_rows = 30;
    let mut c_neon_sub = vec![0.0f32; test_rows * n];
    let t_neon = Instant::now();
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        for i in 0..test_rows {
            let a_row = &a[i * k..(i + 1) * k];
            for j in 0..n {
                let mut sum_vec = vdupq_n_f32(0.0);
                for p in 0..(k / 4) {
                    let a_v = vld1q_f32(a_row.as_ptr().add(p * 4));
                    let b_vals = [
                        b[(p * 4) * n + j],
                        b[(p * 4 + 1) * n + j],
                        b[(p * 4 + 2) * n + j],
                        b[(p * 4 + 3) * n + j],
                    ];
                    let b_v = vld1q_f32(b_vals.as_ptr());
                    sum_vec = vfmaq_f32(sum_vec, a_v, b_v);
                }
                c_neon_sub[i * n + j] = vaddvq_f32(sum_vec);
            }
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for i in 0..test_rows {
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += a[i * k + p] * b[p * n + j];
                }
                c_neon_sub[i * n + j] = sum;
            }
        }
    }
    let neon_sample_dur = t_neon.elapsed().as_secs_f64();
    let neon_extrapolated_dur_per_iter = neon_sample_dur * (m as f64 / test_rows as f64);
    let neon_throughput_gflops =
        (2.0 * m as f64 * n as f64 * k as f64 / neon_extrapolated_dur_per_iter) / 1e9;

    let speedup = amx_throughput_gflops / neon_throughput_gflops;
    println!(
        "Large GEMM ({m}x{n}x{k}): AMX={:.2} GFLOPS, 1D NEON={:.2} GFLOPS, Speedup={:.2}x",
        amx_throughput_gflops, neon_throughput_gflops, speedup
    );

    // AMX 2D GEMM achieves measurable throughput improvement over 1D NEON
    assert!(
        speedup > 1.0,
        "AMX throughput ({:.2} GFLOPS) must exceed 1D NEON ({:.2} GFLOPS)",
        amx_throughput_gflops,
        neon_throughput_gflops
    );
}
