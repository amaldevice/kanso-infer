#[cfg(target_os = "macos")]
mod fused_metal_layer_tests {
    use kanso_infer::config::ModelConfig;
    use kanso_infer::metal::context::MetalContext;
    use kanso_infer::metal::fused_layer::{
        FusedLayerDimensions, FusedLayerWeights, gpu_chained_pipeline_single_encoder,
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
        assert!(source.contains("simdgroup_barrier"));

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

    #[test]
    fn test_fused_layer_decode_latency_overhead_reduction() {
        let ctx = MetalContext::new().expect("MetalContext initialization failed");

        // Use compact ~350M model configuration (SmolLM2-360M dimensions)
        let model_cfg = ModelConfig::smollm2_360m();
        let dims = FusedLayerDimensions {
            hidden_dim: model_cfg.hidden_size,           // 960
            num_q_heads: model_cfg.num_attention_heads,  // 15
            num_kv_heads: model_cfg.num_key_value_heads, // 3
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
