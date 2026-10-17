//! Ticket #49: host-side Metal must submit one command buffer per decode step,
//! keep weights resident on the device, and never compile shaders from source at
//! startup when an AOT `.metallib` was produced by `build.rs`.
//!
//! The counters asserted here are the trace hooks the ticket asks for:
//! - `MetalContext::command_buffers_created` counts every command buffer the host
//!   submits (a decode step must add exactly one).
//! - `MetalContext::buffers_allocated` plus `MetalWeights::weight_uploads` prove
//!   weights are uploaded once at load and reused (no per-step upload/alloc).
//! - `MetalPipelineCache::source_compiles` / `aot_libraries` prove startup either
//!   loads an AOT library or compiles each shader file at most once.

#[cfg(target_os = "macos")]
mod single_command_buffer_tests {
    use kanso_infer::metal::context::MetalContext;
    use kanso_infer::metal::fused_layer::{
        FusedLayerDimensions, FusedLayerWeights, MetalWeights, gpu_discrete_layer_multi_encoder,
    };
    use kanso_infer::metal::shaders::get_all_shaders;

    /// Weights and tables a resident layer wrapper must upload once: norm_weight,
    /// w_q, w_k, w_v, cos, sin.
    const UPLOADED_WEIGHT_BUFFERS: usize = 6;

    struct Harness {
        ctx: MetalContext,
        dims: FusedLayerDimensions,
        weights: FusedLayerWeights,
        cos_table: Vec<f32>,
        sin_table: Vec<f32>,
        hidden: Vec<f32>,
        pos: usize,
        eps: f32,
    }

    impl Harness {
        fn new() -> Self {
            let dims = FusedLayerDimensions {
                hidden_dim: 256,
                num_q_heads: 4,
                num_kv_heads: 2,
                head_dim: 64,
            };
            let pos = 3;
            let eps = 1e-5f32;
            let q_dim = dims.q_dim();
            let kv_dim = dims.kv_dim();
            let hidden_dim = dims.hidden_dim;

            let weights = FusedLayerWeights {
                norm_weight: vec![1.05f32; hidden_dim],
                w_q: (0..q_dim * hidden_dim)
                    .map(|i| ((i as f32) * 0.001).sin())
                    .collect(),
                w_k: (0..kv_dim * hidden_dim)
                    .map(|i| ((i as f32) * 0.0012).cos())
                    .collect(),
                w_v: (0..kv_dim * hidden_dim)
                    .map(|i| ((i as f32) * 0.0015).sin())
                    .collect(),
            };

            let half_dim = dims.head_dim / 2;
            let table_len = (pos + 1) * half_dim;
            let cos_table: Vec<f32> = (0..table_len).map(|i| ((i as f32) * 0.04).cos()).collect();
            let sin_table: Vec<f32> = (0..table_len).map(|i| ((i as f32) * 0.04).sin()).collect();
            let hidden: Vec<f32> = (0..hidden_dim).map(|i| ((i as f32) * 0.03).cos()).collect();

            Self {
                ctx: MetalContext::new().expect("MetalContext initialization failed"),
                dims,
                weights,
                cos_table,
                sin_table,
                hidden,
                pos,
                eps,
            }
        }

        fn upload(&self) -> MetalWeights {
            MetalWeights::upload(
                &self.ctx,
                &self.dims,
                &self.weights,
                &self.cos_table,
                &self.sin_table,
            )
            .expect("resident weight upload should succeed")
        }
    }

    /// Acceptance: one decode step == one command buffer, one wait.
    #[test]
    fn test_decode_step_submits_exactly_one_command_buffer() {
        let h = Harness::new();
        let resident = h.upload();

        let before = h.ctx.command_buffers_created();
        resident
            .decode_step(&h.ctx, &h.hidden, h.pos, h.eps)
            .expect("resident decode step should succeed");
        assert_eq!(
            h.ctx.command_buffers_created() - before,
            1,
            "a decode step must submit exactly one command buffer, not one per kernel"
        );

        // Further steps stay at one buffer each (and re-read the same resident weights).
        for _ in 0..3 {
            resident
                .decode_step(&h.ctx, &h.hidden, h.pos, h.eps)
                .expect("resident decode step should succeed");
        }
        assert_eq!(h.ctx.command_buffers_created() - before, 4);
    }

    /// Acceptance: weights uploaded once at load, zero uploads/allocations per step.
    #[test]
    fn test_weights_are_uploaded_once_and_stay_resident() {
        let h = Harness::new();

        let buffers_before = h.ctx.buffers_allocated();
        let resident = h.upload();
        assert_eq!(
            resident.weight_uploads(),
            UPLOADED_WEIGHT_BUFFERS,
            "loading must upload norm_weight, w_q, w_k, w_v, cos and sin exactly once"
        );
        let buffers_after_upload = h.ctx.buffers_allocated();
        assert!(
            buffers_after_upload > buffers_before,
            "loading must allocate device buffers for the resident weights"
        );

        for _ in 0..3 {
            resident
                .decode_step(&h.ctx, &h.hidden, h.pos, h.eps)
                .expect("resident decode step should succeed");
        }
        assert_eq!(
            resident.weight_uploads(),
            UPLOADED_WEIGHT_BUFFERS,
            "decode steps must not re-upload weights"
        );
        assert_eq!(
            h.ctx.buffers_allocated(),
            buffers_after_upload,
            "decode steps must not allocate buffers (input and outputs are resident)"
        );
    }

    /// The resident path must stay numerically identical to the discrete baseline.
    #[test]
    fn test_resident_decode_step_matches_discrete_baseline() {
        let h = Harness::new();
        let resident = h.upload();

        let (resident_q, resident_k, resident_v) = resident
            .decode_step(&h.ctx, &h.hidden, h.pos, h.eps)
            .expect("resident decode step should succeed");
        let (base_q, base_k, base_v) = gpu_discrete_layer_multi_encoder(
            &h.ctx,
            &h.hidden,
            &h.weights,
            &h.dims,
            &h.cos_table,
            &h.sin_table,
            h.pos,
            h.eps,
        )
        .expect("discrete baseline should succeed");

        assert_eq!(resident_q.len(), h.dims.q_dim());
        assert_eq!(resident_k.len(), h.dims.kv_dim());
        assert_eq!(resident_v.len(), h.dims.kv_dim());

        for (name, resident, base) in [
            ("q", &resident_q, &base_q),
            ("k", &resident_k, &base_k),
            ("v", &resident_v, &base_v),
        ] {
            for (i, (&got, &want)) in resident.iter().zip(base.iter()).enumerate() {
                assert!(
                    (got - want).abs() <= 1e-4,
                    "{name} mismatch at {i}: resident={got} baseline={want}"
                );
            }
        }
    }

    /// Acceptance: startup never compiles a shader file more than once - and never
    /// compiles at all when the AOT `.metallib` is available.
    #[test]
    fn test_startup_does_not_recompile_shader_sources() {
        let unique_files = get_all_shaders().len();
        let ctx = MetalContext::new().expect("MetalContext initialization failed");

        let from_source = ctx.pipelines.source_compiles();
        let from_data = ctx.pipelines.aot_libraries();

        assert_eq!(
            from_source + from_data,
            unique_files,
            "every shader file must yield exactly one library"
        );

        if from_data == 0 {
            eprintln!(
                "no AOT metallib (build.rs found no Metal compiler); \
                 asserting compile-once-per-file fallback: {from_source} compiles / {unique_files} files"
            );
            assert_eq!(
                from_source, unique_files,
                "fallback startup must compile each shader file exactly once \
                 (fused_llama_layer.metal used to be compiled 3x for its 3 kernels)"
            );
        } else {
            assert_eq!(
                from_source, 0,
                "with AOT metallibs available startup must not call new_library_with_source"
            );
            assert_eq!(from_data, unique_files);
        }
    }
}
