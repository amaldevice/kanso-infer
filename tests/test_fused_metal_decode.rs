//! Ticket #87: the fused Metal llama-layer kernel must be the production decode path
//! for dense models, not a kernel-only claim.
//!
//! Four things this pins down, all through `Model::forward_batch` (the entry the engine
//! decodes through) rather than through the kernel test harness:
//! 1. a Metal decode step really runs the fused kernel (observable counter), while the CPU
//!    backend leaves that counter at zero;
//! 2. its logits stay within tolerance of the CPU reference path;
//! 3. the fused prologue is one command buffer per layer on the production path, not one
//!    per op — the property `tests/test_single_command_buffer.rs` proves for the kernel and
//!    this file proves for the decode loop that uses it;
//! 4. dimensions outside the kernel geometry fall back to the per-op path instead of
//!    erroring or changing the numbers.
//!
//! What is *not* asserted here, because it is not reachable with the kernels that already
//! exist: one command buffer for the **whole** step (attention + O + MLP in the same
//! submission as the prologue). `src/metal/fused_layer.rs` covers the QKV prologue only,
//! and the only existing attention entry point owns its command buffer. That gap is
//! recorded as debt in `docs/adr/0002-fused-decode-scope.md`, not papered over with a new
//! kernel.

#[cfg(target_os = "macos")]
mod common;

#[cfg(target_os = "macos")]
mod fused_metal_decode {
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

    /// Synthetic pool contents: a pure function of (layer, slot, head, dim), so the CPU
    /// reference and the Metal run attend the exact same K/V.
    fn pool_value(layer: usize, slot: usize, head: usize, dim: usize) -> f32 {
        let x = (layer * 131 + slot * 17 + head * 7 + dim * 3) as f32;
        (x * 0.037).sin() * 0.5
    }

    fn filled_kv(model: &Model, ctx: usize) -> (KvStorageKind, BlockTable) {
        let head_dim = model.config.head_dim();
        // KV heads come from the model, not the module constant: the fallback case builds a
        // model with a different GQA shape and still has to fill a matching pool.
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
        for layer in 0..LAYERS {
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

    #[test]
    fn metal_decode_step_runs_fused_layer_path() {
        let Some(dev) = crate::common::metal_device_or_skip() else {
            return;
        };
        let metal_model = mock(&dev, Backend::Metal);
        let cpu_model = mock(&Device::Cpu, Backend::Cpu);

        let _ = batch_logits(&metal_model, 128);
        assert!(
            metal_model.fused_layer_calls() >= LAYERS as u64,
            "a Metal decode step must run the fused llama-layer path once per layer \
             (got {})",
            metal_model.fused_layer_calls()
        );

        let _ = batch_logits(&cpu_model, 128);
        assert_eq!(
            cpu_model.fused_layer_calls(),
            0,
            "the CPU backend must keep the per-op path"
        );
    }

    #[test]
    fn metal_fused_decode_logits_match_reference() {
        let Some(dev) = crate::common::metal_device_or_skip() else {
            return;
        };
        let metal_model = mock(&dev, Backend::Metal);
        let cpu_model = mock(&Device::Cpu, Backend::Cpu);

        for ctx in CONTEXTS {
            let reference = batch_logits(&cpu_model, ctx);
            let actual = batch_logits(&metal_model, ctx);
            let worst = worst_relative(&reference, &actual);
            assert!(
                worst <= 1e-3,
                "ctx {ctx}: fused Metal logits drift {worst} from the reference path"
            );
        }
    }

    /// Acceptance #4, scoped to what is actually observable: the fused **prologue**
    /// is one submission per layer, read from the production entry point rather than
    /// from the kernel harness.
    ///
    /// Before #87 the prologue cost one command buffer per op (RMSNorm, three
    /// projections, two RoPE passes). The fused kernel encodes the whole prologue into
    /// one buffer and waits once, so a step's *prologue* submissions must equal the
    /// layer count.
    ///
    /// This is deliberately NOT a claim about command buffers per decode step.
    /// `fused_prologue_command_buffers` reads the fused seam's own `MetalContext`;
    /// the paged-attention seam has a second context and candle submits the tensor
    /// path on the device queue, so the per-step total is structurally invisible from
    /// here. ADR-0002 records the one-buffer-per-step goal as unpaid debt; the number
    /// below must not be cited as evidence for it.
    #[test]
    fn metal_fused_decode_submits_one_command_buffer_per_layer_prologue() {
        let Some(dev) = crate::common::metal_device_or_skip() else {
            return;
        };
        let model = mock(&dev, Backend::Metal);

        // Warm-up step: the first step builds the resident weights.
        let _ = batch_logits(&model, 128);
        assert!(
            model.fused_prologue_command_buffers() > 0,
            "the fused prologue must submit command buffers through its own Metal context"
        );

        let before = model.fused_prologue_command_buffers();
        let _ = batch_logits(&model, 128);
        assert_eq!(
            model.fused_prologue_command_buffers() - before,
            LAYERS,
            "the fused prologue must be one command buffer per layer (RMSNorm + QKV + RoPE \
             chained), not one per op — this says nothing about the per-op tail"
        );
    }

    /// Acceptance #3: a hidden size the GEMV/float4 geometry cannot cover (66 % 4 != 0)
    /// must fall back to the pre-existing path, with the same logits and no error.
    #[test]
    fn metal_fused_decode_falls_back_outside_kernel_geometry() {
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
    }
}
