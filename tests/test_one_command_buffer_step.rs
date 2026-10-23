//! Ticket #110: a dense single-token Metal decode step runs the **whole decoder
//! layer sweep on one command buffer** — fused RMSNorm + QKV + RoPE prologue,
//! PagedAttention v2, O-projection + residual add, SwiGLU MLP + residual add, for
//! every layer — committed once per step, instead of one buffer per layer prologue
//! plus separately-submitted attention and tensor-path tail.
//!
//! This closes the debt `docs/adr/0002-fused-decode-scope.md` recorded against
//! #87. What this pins, all through `Model::forward_batch` (the entry the engine
//! decodes through) plus the kernel-seam level for the new kernel:
//! 1. the per-step command-buffer metric — the first that measures the per-step
//!    *total* across the decode seams — moves by exactly 1 per decode step;
//! 2. the one-buffer step's logits stay within `1e-3` relative of the discrete
//!    reference path at ctx 128 (one partition) and 512 (two partitions);
//! 3. a hidden size outside the kernel geometry (66, not a multiple of 4) keeps
//!    the fallback: identical logits, zero fused calls, and per-layer submissions
//!    rather than the single-buffer step;
//! 4. the new residual-add kernel matches the tensor path (`hidden + proj * scale`)
//!    including a Granite-style `residual_scale` multiplier.
//!
//! Mock weights are the fixed-seed `create_mock_tiny` draws and pool contents are
//! a seeded pure function, never `ones` (Fase 0 convention).

#[cfg(target_os = "macos")]
mod common;

#[cfg(target_os = "macos")]
mod one_command_buffer_step {
    use candle_core::{Device, Tensor};
    use kanso_infer::hardware::Backend;
    use kanso_infer::kv_cache::{BlockTable, KvStorageKind};
    use kanso_infer::model::Model;

    const HIDDEN: usize = 64;
    const Q_HEADS: usize = 8;
    const KV_HEADS: usize = 2; // GQA group size 4
    const LAYERS: usize = 2;
    const VOCAB: usize = 64;
    const BLOCK_SIZE: usize = 16;
    const CONTEXTS: [usize; 2] = [128, 512];

    fn mock(device: &Device, backend: Backend) -> Model {
        let mut model = Model::create_mock_tiny(HIDDEN, Q_HEADS, KV_HEADS, LAYERS, VOCAB)
            .expect("mock model (GQA group 4)");
        if !device.is_cpu() {
            model
                .to_device(device)
                .expect("move mock weights to device");
        }
        model.set_backend(backend);
        model
    }

    /// Synthetic pool contents: a pure function of (layer, slot, head, dim), so the
    /// reference and the Metal run attend the exact same K/V.
    fn pool_value(layer: usize, slot: usize, head: usize, dim: usize) -> f32 {
        let x = (layer * 131 + slot * 17 + head * 7 + dim * 3) as f32;
        (x * 0.037).sin() * 0.5
    }

    fn filled_kv(model: &Model, ctx: usize) -> (KvStorageKind, BlockTable) {
        let head_dim = model.config.head_dim();
        let kv_heads = model.config.num_key_value_heads;
        let blocks = ctx.div_ceil(BLOCK_SIZE);
        let mut kv = KvStorageKind::f32_new(
            model.config.num_hidden_layers,
            blocks,
            BLOCK_SIZE,
            kv_heads,
            head_dim,
            &model.device,
        )
        .expect("kv storage");
        for layer in 0..model.config.num_hidden_layers {
            let mut k = vec![0.0f32; blocks * BLOCK_SIZE * kv_heads * head_dim];
            let mut v = vec![0.0f32; blocks * BLOCK_SIZE * kv_heads * head_dim];
            for slot in 0..blocks * BLOCK_SIZE {
                for head in 0..kv_heads {
                    for dim in 0..head_dim {
                        let i = (slot * kv_heads + head) * head_dim + dim;
                        k[i] = pool_value(layer, slot, head, dim);
                        v[i] = -pool_value(layer + 1, slot, head, dim);
                    }
                }
            }
            let shape = (blocks, BLOCK_SIZE, kv_heads, head_dim);
            let pool = kv.f32_pool_mut().expect("F32 pool");
            pool.kv_k[layer] = Tensor::from_vec(k, shape, &model.device).expect("k pool");
            pool.kv_v[layer] = Tensor::from_vec(v, shape, &model.device).expect("v pool");
        }
        let mut table = BlockTable::new();
        for b in 0..blocks {
            table.push_block(b);
        }
        table.set_num_tokens(ctx);
        (kv, table)
    }

    /// One single-sequence decode step through `forward_batch` — the engine entry.
    fn batch_logits(model: &Model, ctx: usize) -> Vec<f32> {
        let (mut kv, table) = filled_kv(model, ctx);
        let pos = ctx - 1;
        let slot = table.blocks()[pos / BLOCK_SIZE] * BLOCK_SIZE + pos % BLOCK_SIZE;
        model
            .forward_batch(
                &[7],
                &[pos],
                &[slot],
                &mut kv,
                &[table.blocks().to_vec()],
                &[ctx],
                &[1],
            )
            .expect("batched decode step")
            .flatten_all()
            .expect("flatten")
            .to_vec1::<f32>()
            .expect("logits")
    }

    fn worst_relative(reference: &[f32], actual: &[f32]) -> f32 {
        assert_eq!(reference.len(), actual.len(), "logits length");
        reference
            .iter()
            .zip(actual.iter())
            .map(|(r, a)| (r - a).abs() / (1.0 + r.abs()))
            .fold(0.0f32, f32::max)
    }

    /// Acceptance #2: one command buffer per step, asserted from the production
    /// entry (`Model::forward_batch`) through the per-step total metric, not from
    /// kernel isolation.
    #[test]
    fn metal_decode_step_spends_one_command_buffer() {
        let Some(dev) = crate::common::metal_device_or_skip() else {
            return;
        };
        let model = mock(&dev, Backend::Metal);

        // Warm-up: the first step builds the resident weights and both seams.
        let _ = batch_logits(&model, 128);
        assert!(
            model.metal_decode_command_buffers() > 0,
            "a Metal decode step must submit command buffers through the decode seams"
        );

        let buffers_before = model.metal_decode_command_buffers();
        let calls_before = model.fused_layer_calls();
        for _ in 0..3 {
            let _ = batch_logits(&model, 128);
        }
        assert_eq!(
            model.metal_decode_command_buffers() - buffers_before,
            3,
            "every decode step must spend exactly ONE command buffer for the whole \
             layer sweep (prologue + attention + O + MLP chained), not one per layer \
             prologue plus separately-submitted attention and tail"
        );
        assert_eq!(
            model.fused_layer_calls() - calls_before,
            (3 * LAYERS) as u64,
            "each step must still run the fused chain once per layer — one submission \
             covering the layer, not a skip of the layer"
        );
    }

    /// Acceptance #3: the one-buffer step's logits stay within tolerance of the
    /// discrete reference path — and the step it measures is provably the
    /// single-buffer one (its metric moves by exactly 1).
    ///
    /// `max_seq_len` is widened past the mock default (128) so the seam's resident
    /// rotary tables cover ctx 512 too — without that, pos 511 would exceed the
    /// tables and the step would rightly fall back to the per-layer loop.
    #[test]
    fn one_buffer_step_logits_match_discrete_reference() {
        let Some(dev) = crate::common::metal_device_or_skip() else {
            return;
        };
        let mut metal_model = mock(&dev, Backend::Metal);
        metal_model.config.max_seq_len = 1024;
        let mut cpu_model = mock(&Device::Cpu, Backend::Cpu);
        cpu_model.config.max_seq_len = 1024;

        for ctx in CONTEXTS {
            let reference = batch_logits(&cpu_model, ctx);
            let before = metal_model.metal_decode_command_buffers();
            let actual = batch_logits(&metal_model, ctx);
            assert_eq!(
                metal_model.metal_decode_command_buffers() - before,
                1,
                "ctx {ctx}: the parity run must go through the one-command-buffer step"
            );
            let worst = worst_relative(&reference, &actual);
            assert!(
                worst <= 1e-3,
                "ctx {ctx}: one-buffer Metal logits drift {worst} from the reference path"
            );
        }
    }

    /// Acceptance #4: a hidden size the kernels cannot represent (66 % 4 != 0)
    /// keeps the pre-existing path — identical logits, zero fused calls — and its
    /// per-step submissions stay per-layer (attention wrapper), proving the
    /// single-buffer step did not run half-way.
    #[test]
    fn non_representable_geometry_keeps_the_fallback() {
        let Some(dev) = crate::common::metal_device_or_skip() else {
            return;
        };
        let falls_back = {
            let mut model =
                Model::create_mock_tiny(66, 3, 1, LAYERS, VOCAB).expect("mock hidden 66");
            model.to_device(&dev).expect("move weights");
            model.set_backend(Backend::Metal);
            model
        };
        assert_eq!(falls_back.config.hidden_size, 66);

        let reference = {
            let mut cpu = Model::create_mock_tiny(66, 3, 1, LAYERS, VOCAB).expect("cpu mock");
            cpu.set_backend(Backend::Cpu);
            batch_logits(&cpu, 128)
        };
        let actual = batch_logits(&falls_back, 128);
        assert_eq!(
            falls_back.fused_layer_calls(),
            0,
            "hidden_dim 66 is outside the kernel geometry: the discrete path must run, \
             not a crash or a silent wrong-shape dispatch"
        );
        let worst = worst_relative(&reference, &actual);
        assert!(
            worst <= 1e-3,
            "fallback logits drift {worst} from the reference path"
        );

        let before = falls_back.metal_decode_command_buffers();
        let _ = batch_logits(&falls_back, 128);
        assert_eq!(
            falls_back.metal_decode_command_buffers() - before,
            LAYERS,
            "the fallback keeps one submission per layer through the paged-attention \
             wrapper; the single-buffer step must not run for this geometry"
        );
    }

    /// Acceptance #1: the residual-add kernel (`hidden += proj * scale`) matches the
    /// tensor path, including a Granite-style `residual_scale` multiplier other
    /// than 1 — the scale the kernel receives must be the one `Model::scale_residual`
    /// applies (`t * s`, f32).
    #[test]
    fn residual_add_kernel_matches_tensor_path() {
        use kanso_infer::metal::context::MetalContext;
        use kanso_infer::metal::ops::gpu_residual_add;
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let ctx = MetalContext::new().expect("MetalContext initialization failed");
        let mut rng = StdRng::seed_from_u64(0x110);

        for scale in [1.0f32, 0.125f32] {
            let hidden: Vec<f32> = (0..512).map(|_| rng.random::<f32>() * 4.0 - 2.0).collect();
            let proj: Vec<f32> = (0..512).map(|_| rng.random::<f32>() * 4.0 - 2.0).collect();

            let got = gpu_residual_add(&ctx, &hidden, &proj, scale)
                .expect("residual-add dispatch should succeed");

            // Tensor-path reference: the residual `hidden + scale_residual(proj)`
            // the per-op decode path computes.
            let want: Vec<f32> = hidden
                .iter()
                .zip(proj.iter())
                .map(|(&h, &p)| h + p * scale)
                .collect();

            assert_eq!(got.len(), want.len(), "scale {scale}: output length");
            for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
                assert!(
                    (g - w).abs() <= 1e-5,
                    "scale {scale}: residual-add mismatch at {i}: kernel={g} tensor={w}"
                );
            }
        }
    }
}
