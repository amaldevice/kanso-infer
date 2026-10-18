#[cfg(target_os = "macos")]
mod fused_metal_layer_tests {
    use kanso_infer::config::ModelConfig;
    use kanso_infer::metal::context::MetalContext;
    use kanso_infer::metal::fused_layer::{
        FUSED_THREADS_PER_TG, FusedLayerDimensions, FusedLayerWeights, GEMV_ROWS_PER_THREADGROUP,
        encode_fused_qkv_gemv, gemv_threadgroup_geometry, gpu_chained_pipeline_single_encoder,
        gpu_discrete_layer_multi_encoder, gpu_fused_layer_single_encoder, gpu_fused_rmsnorm_rope,
    };
    use kanso_infer::metal::ops::{cpu_rmsnorm, cpu_rope, gpu_rmsnorm, gpu_rope};
    use kanso_infer::metal::shaders::get_all_shaders;
    use std::time::Instant;

    #[test]
    fn test_fused_shader_embedding_and_pipeline_cache() {
        let shaders = get_all_shaders();
        assert!(
            shaders.contains_key("fused_llama_layer.metal"),
            "fused_llama_layer.metal must be registered in shader map"
        );
        let source = shaders["fused_llama_layer.metal"];
        assert!(source.contains("fused_rmsnorm_rope_kernel"));
        assert!(source.contains("fused_rmsnorm_qkv_rope_kernel"));
        assert!(source.contains("linear_gemv_kernel"));
        assert!(source.contains("threadgroup_barrier"));
        // #50 replaced the cargo-cult `simdgroup_barrier(mem_flags::mem_none)` inside a
        // divergent branch with real SIMD reductions.
        assert!(source.contains("simd_sum"));

        let ctx = MetalContext::new().expect("MetalContext initialization failed");
        assert!(ctx.pipelines.contains("fused_rmsnorm_rope"));
        assert!(ctx.pipelines.contains("fused_rmsnorm_qkv_rope"));
        assert!(ctx.pipelines.contains("linear_gemv"));
    }

    #[test]
    fn test_fused_rmsnorm_rope_parity() {
        let ctx = MetalContext::new().expect("MetalContext initialization failed");

        let head_dim = 64;
        let num_heads = 4;
        let dim = num_heads * head_dim; // 256
        let pos = 5;
        let eps = 1e-5f32;

        let input: Vec<f32> = (0..dim).map(|i| ((i as f32) * 0.05).sin() + 1.0).collect();
        let norm_weight: Vec<f32> = (0..dim).map(|i| 1.0 + (i as f32) * 0.001).collect();

        let half_dim = head_dim / 2;
        let table_len = (pos + 1) * half_dim;
        let cos_table: Vec<f32> = (0..table_len).map(|i| ((i as f32) * 0.02).cos()).collect();
        let sin_table: Vec<f32> = (0..table_len).map(|i| ((i as f32) * 0.02).sin()).collect();

        // 1. Reference discrete CPU execution: RMSNorm then RoPE
        let mut cpu_expected = cpu_rmsnorm(&input, &norm_weight, eps);
        cpu_rope(&mut cpu_expected, &cos_table, &sin_table, head_dim, pos);

        // 2. Reference discrete GPU execution: gpu_rmsnorm then gpu_rope
        let mut discrete_gpu = gpu_rmsnorm(&ctx, &input, &norm_weight, eps)
            .expect("discrete gpu_rmsnorm should succeed");
        gpu_rope(
            &ctx,
            &mut discrete_gpu,
            &cos_table,
            &sin_table,
            head_dim,
            pos,
        )
        .expect("discrete gpu_rope should succeed");

        // 3. Fused GPU execution: fused RMSNorm + RoPE in threadgroup memory
        let fused_gpu = gpu_fused_rmsnorm_rope(
            &ctx,
            &input,
            &norm_weight,
            &cos_table,
            &sin_table,
            head_dim,
            pos,
            eps,
        )
        .expect("gpu_fused_rmsnorm_rope should succeed");

        assert_eq!(fused_gpu.len(), dim);
        for i in 0..dim {
            let diff_cpu = (fused_gpu[i] - cpu_expected[i]).abs();
            let diff_discrete = (fused_gpu[i] - discrete_gpu[i]).abs();
            assert!(
                diff_cpu <= 1e-4,
                "Fused vs CPU mismatch at {i}: fused={}, cpu={}, diff={}",
                fused_gpu[i],
                cpu_expected[i],
                diff_cpu
            );
            assert!(
                diff_discrete <= 1e-4,
                "Fused vs Discrete GPU mismatch at {i}: fused={}, discrete={}, diff={}",
                fused_gpu[i],
                discrete_gpu[i],
                diff_discrete
            );
        }
    }

    #[test]
    fn test_fused_layer_single_command_encoder_parity() {
        let ctx = MetalContext::new().expect("MetalContext initialization failed");

        let dims = FusedLayerDimensions {
            hidden_dim: 256,
            num_q_heads: 4,
            num_kv_heads: 2,
            head_dim: 64,
        };
        let pos = 2;
        let eps = 1e-5f32;

        let q_dim = dims.q_dim();
        let kv_dim = dims.kv_dim();

        let hidden_in: Vec<f32> = (0..dims.hidden_dim)
            .map(|i| ((i as f32) * 0.03).cos())
            .collect();
        let norm_weight: Vec<f32> = vec![1.05f32; dims.hidden_dim];

        let w_q: Vec<f32> = (0..q_dim * dims.hidden_dim)
            .map(|i| ((i as f32) * 0.001).sin())
            .collect();
        let w_k: Vec<f32> = (0..kv_dim * dims.hidden_dim)
            .map(|i| ((i as f32) * 0.0012).cos())
            .collect();
        let w_v: Vec<f32> = (0..kv_dim * dims.hidden_dim)
            .map(|i| ((i as f32) * 0.0015).sin())
            .collect();

        let half_dim = dims.head_dim / 2;
        let table_len = (pos + 1) * half_dim;
        let cos_table: Vec<f32> = (0..table_len).map(|i| ((i as f32) * 0.04).cos()).collect();
        let sin_table: Vec<f32> = (0..table_len).map(|i| ((i as f32) * 0.04).sin()).collect();

        let weights = FusedLayerWeights {
            norm_weight,
            w_q,
            w_k,
            w_v,
        };

        // 1. Discrete multi-encoder execution (3 command encoders / buffer commits)
        let (discrete_q, discrete_k, discrete_v) = gpu_discrete_layer_multi_encoder(
            &ctx, &hidden_in, &weights, &dims, &cos_table, &sin_table, pos, eps,
        )
        .expect("discrete multi-encoder layer execution should succeed");

        // 2. Fused single-encoder execution (fused kernel combining RMSNorm + QKV + RoPE)
        let (fused_q, fused_k, fused_v) = gpu_fused_layer_single_encoder(
            &ctx, &hidden_in, &weights, &dims, &cos_table, &sin_table, pos, eps,
        )
        .expect("fused single-encoder layer execution should succeed");

        // 3. Chained single-encoder execution with internal memory barriers
        let (chained_q, chained_k, chained_v) = gpu_chained_pipeline_single_encoder(
            &ctx, &hidden_in, &weights, &dims, &cos_table, &sin_table, pos, eps,
        )
        .expect("chained single-encoder layer execution should succeed");

        assert_eq!(fused_q.len(), q_dim);
        assert_eq!(fused_k.len(), kv_dim);
        assert_eq!(fused_v.len(), kv_dim);

        for i in 0..q_dim {
            let diff_fused = (fused_q[i] - discrete_q[i]).abs();
            let diff_chained = (chained_q[i] - discrete_q[i]).abs();
            assert!(
                diff_fused <= 1e-4,
                "Q fused vs discrete mismatch at {i}: fused={}, discrete={}, diff={}",
                fused_q[i],
                discrete_q[i],
                diff_fused
            );
            assert!(
                diff_chained <= 1e-4,
                "Q chained vs discrete mismatch at {i}: chained={}, discrete={}, diff={}",
                chained_q[i],
                discrete_q[i],
                diff_chained
            );
        }

        for i in 0..kv_dim {
            let diff_fused_k = (fused_k[i] - discrete_k[i]).abs();
            let diff_chained_k = (chained_k[i] - discrete_k[i]).abs();
            assert!(
                diff_fused_k <= 1e-4,
                "K fused vs discrete mismatch at {i}: fused={}, discrete={}, diff={}",
                fused_k[i],
                discrete_k[i],
                diff_fused_k
            );
            assert!(
                diff_chained_k <= 1e-4,
                "K chained vs discrete mismatch at {i}: chained={}, discrete={}, diff={}",
                chained_k[i],
                discrete_k[i],
                diff_chained_k
            );

            let diff_fused_v = (fused_v[i] - discrete_v[i]).abs();
            let diff_chained_v = (chained_v[i] - discrete_v[i]).abs();
            assert!(
                diff_fused_v <= 1e-4,
                "V fused vs discrete mismatch at {i}: fused={}, discrete={}, diff={}",
                fused_v[i],
                discrete_v[i],
                diff_fused_v
            );
            assert!(
                diff_chained_v <= 1e-4,
                "V chained vs discrete mismatch at {i}: chained={}, discrete={}, diff={}",
                chained_v[i],
                discrete_v[i],
                diff_chained_v
            );
        }
    }

    #[test]
    fn test_granite_350m_fused_layer_parity() {
        let ctx = MetalContext::new().expect("MetalContext initialization failed");

        let model_cfg = ModelConfig::granite_4_0_350m_dense();
        let dims = FusedLayerDimensions {
            hidden_dim: model_cfg.hidden_size,           // 1024
            num_q_heads: model_cfg.num_attention_heads,  // 16
            num_kv_heads: model_cfg.num_key_value_heads, // 4
            head_dim: model_cfg.head_dim(),              // 64
        };
        let pos = 7;
        let eps = model_cfg.rms_norm_eps as f32;

        let q_dim = dims.q_dim();
        let kv_dim = dims.kv_dim();

        let hidden_in: Vec<f32> = (0..dims.hidden_dim)
            .map(|i| ((i as f32) * 0.01).sin() + 0.5)
            .collect();
        let weights = FusedLayerWeights {
            norm_weight: vec![1.0f32; dims.hidden_dim],
            w_q: (0..q_dim * dims.hidden_dim)
                .map(|i| (i as f32 * 0.0001).cos())
                .collect(),
            w_k: (0..kv_dim * dims.hidden_dim)
                .map(|i| (i as f32 * 0.0001).sin())
                .collect(),
            w_v: (0..kv_dim * dims.hidden_dim)
                .map(|i| (i as f32 * 0.0002).cos())
                .collect(),
        };

        let half_dim = dims.head_dim / 2;
        let table_len = (pos + 1) * half_dim;
        let cos_table: Vec<f32> = (0..table_len).map(|i| (i as f32 * 0.05).cos()).collect();
        let sin_table: Vec<f32> = (0..table_len).map(|i| (i as f32 * 0.05).sin()).collect();

        let (discrete_q, discrete_k, discrete_v) = gpu_discrete_layer_multi_encoder(
            &ctx, &hidden_in, &weights, &dims, &cos_table, &sin_table, pos, eps,
        )
        .unwrap();

        let (fused_q, fused_k, fused_v) = gpu_fused_layer_single_encoder(
            &ctx, &hidden_in, &weights, &dims, &cos_table, &sin_table, pos, eps,
        )
        .unwrap();

        for i in 0..q_dim {
            assert!((fused_q[i] - discrete_q[i]).abs() <= 1e-4);
        }
        for i in 0..kv_dim {
            assert!((fused_k[i] - discrete_k[i]).abs() <= 1e-4);
            assert!((fused_v[i] - discrete_v[i]).abs() <= 1e-4);
        }
    }

    /// Ticket #50: the fused QKV kernel staged the normalised hidden state in a fixed
    /// `shared_norm[2048]` threadgroup array, so `hidden_dim` 2048 only just fit and 4096
    /// (Llama-3-8B) wrote past it and silently corrupted Q/K/V. These are the dims the
    /// ticket names.
    #[test]
    fn test_fused_layer_parity_across_hidden_dims() {
        let ctx = MetalContext::new().expect("MetalContext initialization failed");

        // (hidden_dim, num_q_heads, num_kv_heads, head_dim)
        let configs: [(usize, usize, usize, usize); 3] = [
            (960, 15, 5, 64),   // SmolLM2-360M
            (2048, 16, 4, 128), // exactly the old threadgroup array bound
            (4096, 32, 8, 128), // Llama-3-8B: silently corrupted before #50
        ];

        for (hidden_dim, num_q_heads, num_kv_heads, head_dim) in configs {
            let dims = FusedLayerDimensions {
                hidden_dim,
                num_q_heads,
                num_kv_heads,
                head_dim,
            };
            let pos = 3;
            let eps = 1e-5f32;

            let weights = FusedLayerWeights {
                norm_weight: (0..hidden_dim)
                    .map(|i| 1.0 + (i as f32 * 0.001).cos() * 0.1)
                    .collect(),
                w_q: (0..dims.q_dim() * hidden_dim)
                    .map(|i| (i as f32 * 0.0001).sin() * 0.05)
                    .collect(),
                w_k: (0..dims.kv_dim() * hidden_dim)
                    .map(|i| (i as f32 * 0.00012).cos() * 0.05)
                    .collect(),
                w_v: (0..dims.kv_dim() * hidden_dim)
                    .map(|i| (i as f32 * 0.00015).sin() * 0.05)
                    .collect(),
            };
            let hidden_in: Vec<f32> = (0..hidden_dim)
                .map(|i| (i as f32 * 0.017).sin() * 0.6)
                .collect();

            let half_dim = head_dim / 2;
            let table_len = (pos + 1) * half_dim;
            let cos_table: Vec<f32> = (0..table_len).map(|i| (i as f32 * 0.03).cos()).collect();
            let sin_table: Vec<f32> = (0..table_len).map(|i| (i as f32 * 0.03).sin()).collect();

            let (discrete_q, discrete_k, discrete_v) = gpu_discrete_layer_multi_encoder(
                &ctx, &hidden_in, &weights, &dims, &cos_table, &sin_table, pos, eps,
            )
            .expect("discrete multi-encoder layer execution should succeed");

            let (fused_q, fused_k, fused_v) = gpu_fused_layer_single_encoder(
                &ctx, &hidden_in, &weights, &dims, &cos_table, &sin_table, pos, eps,
            )
            .expect("fused single-encoder layer execution should succeed");

            let check = |label: &str, fused: &[f32], discrete: &[f32]| {
                assert_eq!(
                    fused.len(),
                    discrete.len(),
                    "{label} len at hidden {hidden_dim}"
                );
                for (i, (f, d)) in fused.iter().zip(discrete.iter()).enumerate() {
                    let tol = 1e-4 * (1.0 + d.abs());
                    assert!(
                        (f - d).abs() <= tol,
                        "{label}[{i}] mismatch at hidden_dim={hidden_dim} head_dim={head_dim}: \
                         fused={f}, discrete={d}, tol={tol}"
                    );
                }
            };
            check("q", &fused_q, &discrete_q);
            check("k", &fused_k, &discrete_k);
            check("v", &fused_v, &discrete_v);
        }
    }

    /// Ticket #50: dimensions the kernels cannot represent must be rejected on the host
    /// instead of silently reading or writing out of bounds on the GPU.
    #[test]
    fn test_fused_layer_rejects_invalid_dimensions_and_weight_shapes() {
        let ctx = MetalContext::new().expect("MetalContext initialization failed");

        let base = FusedLayerDimensions {
            hidden_dim: 256,
            num_q_heads: 4,
            num_kv_heads: 2,
            head_dim: 64,
        };
        let pos = 2usize;
        let eps = 1e-5f32;
        let half_dim = base.head_dim / 2;
        let cos_table: Vec<f32> = vec![1.0; (pos + 1) * half_dim];
        let sin_table: Vec<f32> = vec![0.0; (pos + 1) * half_dim];
        let hidden_in = vec![0.5f32; base.hidden_dim];

        let weights_for = |dims: &FusedLayerDimensions| FusedLayerWeights {
            norm_weight: vec![1.0f32; dims.hidden_dim],
            w_q: vec![0.01f32; dims.q_dim() * dims.hidden_dim],
            w_k: vec![0.01f32; dims.kv_dim() * dims.hidden_dim],
            w_v: vec![0.01f32; dims.kv_dim() * dims.hidden_dim],
        };

        // Lanes read float4 chunks of the hidden state, so hidden_dim % 4 == 0.
        let unaligned = FusedLayerDimensions {
            hidden_dim: 250,
            ..base
        };
        let unaligned_weights = weights_for(&unaligned);
        assert!(
            gpu_fused_layer_single_encoder(
                &ctx,
                &vec![0.5f32; 250],
                &unaligned_weights,
                &unaligned,
                &cos_table,
                &sin_table,
                pos,
                eps
            )
            .is_err(),
            "hidden_dim % 4 != 0 must be rejected"
        );

        // RoPE rotates head_dim in two halves, so head_dim must be even.
        let odd_head = FusedLayerDimensions {
            head_dim: 63,
            ..base
        };
        assert!(
            gpu_fused_layer_single_encoder(
                &ctx,
                &hidden_in,
                &weights_for(&odd_head),
                &odd_head,
                &cos_table,
                &sin_table,
                pos,
                eps
            )
            .is_err(),
            "odd head_dim must be rejected"
        );

        // Weight slices must match the declared dims; the GPU indexes them blindly.
        let mut short_q = weights_for(&base);
        short_q.w_q.pop();
        assert!(
            gpu_fused_layer_single_encoder(
                &ctx, &hidden_in, &short_q, &base, &cos_table, &sin_table, pos, eps
            )
            .is_err(),
            "a w_q shorter than q_dim * hidden_dim must be rejected (out-of-bounds GPU read)"
        );

        let mut oversized_q = weights_for(&base);
        oversized_q.w_q.push(0.0);
        assert!(
            gpu_fused_layer_single_encoder(
                &ctx,
                &hidden_in,
                &oversized_q,
                &base,
                &cos_table,
                &sin_table,
                pos,
                eps
            )
            .is_err(),
            "w_q length != q_dim * hidden_dim must be rejected"
        );

        let mut short_norm = weights_for(&base);
        short_norm.norm_weight.pop();
        assert!(
            gpu_fused_layer_single_encoder(
                &ctx,
                &hidden_in,
                &short_norm,
                &base,
                &cos_table,
                &sin_table,
                pos,
                eps
            )
            .is_err(),
            "norm_weight length != hidden_dim must be rejected"
        );

        // The rotary table has to cover the requested position.
        assert!(
            gpu_fused_layer_single_encoder(
                &ctx,
                &hidden_in,
                &weights_for(&base),
                &base,
                &cos_table[..half_dim],
                &sin_table,
                pos,
                eps
            )
            .is_err(),
            "cos_table shorter than (pos + 1) * head_dim / 2 must be rejected"
        );
    }

    /// Ticket #50: the QKV GEMV must not run on a single threadgroup with a fixed
    /// threadgroup staging vector any more.
    #[test]
    fn test_fused_gemv_kernel_uses_multigroup_simd_pattern() {
        let shaders = get_all_shaders();
        let src = shaders["fused_llama_layer.metal"];

        assert!(
            src.contains("simd_sum"),
            "GEMV must reduce per-lane dot products with SIMD ops (#50)"
        );
        assert!(
            src.contains("float4"),
            "GEMV lanes must read float4 chunks of rows and the input vector (#50)"
        );
        assert!(
            !src.contains("shared_norm"),
            "fixed threadgroup staging vector must be gone, it overflowed at hidden_dim 4096 (#50)"
        );
    }

    /// Ticket #50: the projection must be spread over many threadgroups, not the single
    /// one the old dispatch used (host-side arithmetic of the grid actually dispatched).
    #[test]
    fn test_fused_gemv_dispatch_geometry_is_multi_threadgroup() {
        for out_dim in [320usize, 960, 1024, 2048, 4096] {
            let (threadgroups, width) = gemv_threadgroup_geometry(out_dim);
            assert_eq!(width, FUSED_THREADS_PER_TG as u64);
            assert!(
                threadgroups > 1,
                "out_dim {out_dim} must be partitioned over several threadgroups"
            );
            let covered = threadgroups * GEMV_ROWS_PER_THREADGROUP as u64;
            assert!(
                covered >= out_dim as u64,
                "out_dim {out_dim}: {threadgroups} threadgroups only cover {covered} rows"
            );
            assert!(
                covered - (GEMV_ROWS_PER_THREADGROUP as u64) < out_dim as u64,
                "out_dim {out_dim}: {threadgroups} threadgroups overshoot the grid"
            );
        }
    }

    /// Ticket #50 acceptance: weight-streaming bandwidth of the fused QKV GEMV.
    /// Weights are uploaded once and many iterations are chained into a single command
    /// buffer, so the number reflects kernel throughput rather than the per-call buffer
    /// upload the host still does (that is #49). The >= 60% of device peak claim itself
    /// needs Instruments on the target Mac.
    #[test]
    fn test_fused_qkv_gemv_bandwidth_960_2048() {
        let ctx = MetalContext::new().expect("MetalContext initialization failed");

        for (hidden_dim, num_q_heads, num_kv_heads, head_dim) in
            [(960usize, 15usize, 3usize, 64usize), (2048, 16, 4, 128)]
        {
            let dims = FusedLayerDimensions {
                hidden_dim,
                num_q_heads,
                num_kv_heads,
                head_dim,
            };
            let pos = 1usize;
            let eps = 1e-5f32;
            let half_dim = head_dim / 2;
            let q_dim = dims.q_dim();
            let kv_dim = dims.kv_dim();

            let in_buf = ctx.alloc_buffer_with_data(&vec![0.5f32; hidden_dim]);
            let norm_w_buf = ctx.alloc_buffer_with_data(&vec![1.0f32; hidden_dim]);
            let w_q_buf = ctx.alloc_buffer_with_data(&vec![0.01f32; q_dim * hidden_dim]);
            let w_k_buf = ctx.alloc_buffer_with_data(&vec![0.01f32; kv_dim * hidden_dim]);
            let w_v_buf = ctx.alloc_buffer_with_data(&vec![0.01f32; kv_dim * hidden_dim]);
            let cos_buf = ctx.alloc_buffer_with_data(&vec![1.0f32; (pos + 1) * half_dim]);
            let sin_buf = ctx.alloc_buffer_with_data(&vec![0.0f32; (pos + 1) * half_dim]);
            let q_out_buf = ctx.alloc_buffer::<f32>(q_dim);
            let k_out_buf = ctx.alloc_buffer::<f32>(kv_dim);
            let v_out_buf = ctx.alloc_buffer::<f32>(kv_dim);

            let pipeline = ctx
                .pipelines
                .get("fused_rmsnorm_qkv_rope")
                .expect("pipeline 'fused_rmsnorm_qkv_rope' missing");

            let weight_bytes =
                ((q_dim + 2 * kv_dim) * hidden_dim * std::mem::size_of::<f32>()) as f64;
            let iters: u64 = 50;

            // Warmup outside the measurement.
            for _ in 0..3 {
                let _ = gpu_fused_layer_single_encoder(
                    &ctx,
                    &vec![0.5f32; hidden_dim],
                    &FusedLayerWeights {
                        norm_weight: vec![1.0f32; hidden_dim],
                        w_q: vec![0.01f32; q_dim * hidden_dim],
                        w_k: vec![0.01f32; kv_dim * hidden_dim],
                        w_v: vec![0.01f32; kv_dim * hidden_dim],
                    },
                    &dims,
                    &vec![1.0f32; (pos + 1) * half_dim],
                    &vec![0.0f32; (pos + 1) * half_dim],
                    pos,
                    eps,
                )
                .unwrap();
            }

            let command_buffer = ctx.queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            for _ in 0..iters {
                encode_fused_qkv_gemv(
                    encoder,
                    pipeline,
                    &in_buf,
                    &norm_w_buf,
                    &w_q_buf,
                    &cos_buf,
                    &sin_buf,
                    &q_out_buf,
                    &dims,
                    q_dim,
                    pos,
                    eps,
                    true,
                );
                encode_fused_qkv_gemv(
                    encoder,
                    pipeline,
                    &in_buf,
                    &norm_w_buf,
                    &w_k_buf,
                    &cos_buf,
                    &sin_buf,
                    &k_out_buf,
                    &dims,
                    kv_dim,
                    pos,
                    eps,
                    true,
                );
                encode_fused_qkv_gemv(
                    encoder,
                    pipeline,
                    &in_buf,
                    &norm_w_buf,
                    &w_v_buf,
                    &cos_buf,
                    &sin_buf,
                    &v_out_buf,
                    &dims,
                    kv_dim,
                    pos,
                    eps,
                    false,
                );
                // Serialise iterations: without this Metal may overlap them and the
                // measured rate would not be a sustained weight-streaming rate.
                let written: [&metal::ResourceRef; 3] = [&q_out_buf, &k_out_buf, &v_out_buf];
                encoder.memory_barrier_with_resources(&written);
            }
            encoder.end_encoding();

            let t0 = Instant::now();
            command_buffer.commit();
            command_buffer.wait_until_completed();
            let elapsed = t0.elapsed();

            let gbps = weight_bytes * iters as f64 / elapsed.as_secs_f64() / 1e9;
            println!(
                "Fused QKV GEMV hidden_dim={hidden_dim}: {:.1} GB/s weight streaming \
                 ({:.2} MB of weights x {iters} iters in {elapsed:?})",
                gbps,
                weight_bytes / 1e6
            );
            assert!(
                gbps > 20.0,
                "hidden_dim={hidden_dim}: fused QKV GEMV only streamed {gbps:.1} GB/s, \
                 the multi-threadgroup GEMV should be far above this"
            );
        }
    }

    #[test]
    fn test_fused_layer_decode_latency_overhead_reduction() {
        let ctx = MetalContext::new().expect("MetalContext initialization failed");

        // Use compact ~350M model configuration (SmolLM2-360M dimensions)
        let model_cfg = ModelConfig::smollm2_360m();
        let dims = FusedLayerDimensions {
            hidden_dim: model_cfg.hidden_size,           // 960
            num_q_heads: model_cfg.num_attention_heads,  // 15
            num_kv_heads: model_cfg.num_key_value_heads, // 5
            head_dim: model_cfg.head_dim(),              // 64
        };
        let pos = 12;
        let eps = model_cfg.rms_norm_eps as f32;

        let q_dim = dims.q_dim();
        let kv_dim = dims.kv_dim();

        let hidden_in: Vec<f32> = vec![0.5f32; dims.hidden_dim];
        let weights = FusedLayerWeights {
            norm_weight: vec![1.0f32; dims.hidden_dim],
            w_q: vec![0.01f32; q_dim * dims.hidden_dim],
            w_k: vec![0.01f32; kv_dim * dims.hidden_dim],
            w_v: vec![0.01f32; kv_dim * dims.hidden_dim],
        };

        let half_dim = dims.head_dim / 2;
        let table_len = (pos + 1) * half_dim;
        let cos_table: Vec<f32> = vec![1.0f32; table_len];
        let sin_table: Vec<f32> = vec![0.0f32; table_len];

        // Warmup both paths
        let _ = gpu_discrete_layer_multi_encoder(
            &ctx, &hidden_in, &weights, &dims, &cos_table, &sin_table, pos, eps,
        )
        .unwrap();
        let _ = gpu_fused_layer_single_encoder(
            &ctx, &hidden_in, &weights, &dims, &cos_table, &sin_table, pos, eps,
        )
        .unwrap();

        let iters = 20;

        // Measure discrete multi-encoder latency
        let t0 = Instant::now();
        for _ in 0..iters {
            let _ = gpu_discrete_layer_multi_encoder(
                &ctx, &hidden_in, &weights, &dims, &cos_table, &sin_table, pos, eps,
            )
            .unwrap();
        }
        let discrete_time = t0.elapsed();

        // Measure single-encoder fused latency
        let t1 = Instant::now();
        for _ in 0..iters {
            let _ = gpu_fused_layer_single_encoder(
                &ctx, &hidden_in, &weights, &dims, &cos_table, &sin_table, pos, eps,
            )
            .unwrap();
        }
        let fused_time = t1.elapsed();

        println!(
            "Latency over {} iterations for ~350M layer (dim={}): Discrete multi-encoder: {:?}, Fused single-encoder: {:?}",
            iters, dims.hidden_dim, discrete_time, fused_time
        );

        // Fused single-encoder must demonstrate reduction in GPU command encoder / commit overhead
        assert!(
            fused_time <= discrete_time,
            "Fused single-encoder ({:?}) should be faster than discrete multi-encoder ({:?})",
            fused_time,
            discrete_time
        );
    }
}
