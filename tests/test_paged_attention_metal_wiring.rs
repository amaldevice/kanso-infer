//! Ticket #86: PagedAttention v2 must be the Metal decode path, not a kernel-only claim.
//!
//! Three things this pins down:
//! 1. a Metal decode step really dispatches the kernel (observable counter), while the
//!    CPU backend keeps the strided-view path (counter stays zero);
//! 2. its logits stay within tolerance of the reference path at ctx 128 / 512 / 2048,
//!    with GQA group size 4;
//! 3. kernel parameters the shader cannot represent (head_dim > 256, GQA group > 8)
//!    surface as `Err` instead of a silent fallback to the copying path.

#[cfg(target_os = "macos")]
mod metal_paged_attention_wiring {
    use candle_core::{Device, Tensor};
    use kanso_infer::hardware::Backend;
    use kanso_infer::kv_cache::{BlockTable, KvStorage};
    use kanso_infer::model::Model;

    const HIDDEN: usize = 64;
    const Q_HEADS: usize = 8;
    const KV_HEADS: usize = 2; // GQA group size 4
    const LAYERS: usize = 2;
    const VOCAB: usize = 64;
    const BLOCK_SIZE: usize = 16;
    const CONTEXTS: [usize; 3] = [128, 512, 2048];

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

    fn filled_kv(model: &Model, ctx: usize) -> (KvStorage, BlockTable) {
        let head_dim = model.config.head_dim();
        let blocks = ctx.div_ceil(BLOCK_SIZE);
        let mut kv = KvStorage::new(
            model.config.num_hidden_layers,
            blocks,
            BLOCK_SIZE,
            KV_HEADS,
            head_dim,
            &model.device,
        )
        .expect("kv storage");
        for layer in 0..LAYERS {
            let mut k = vec![0.0f32; blocks * BLOCK_SIZE * KV_HEADS * head_dim];
            let mut v = vec![0.0f32; blocks * BLOCK_SIZE * KV_HEADS * head_dim];
            for slot in 0..blocks * BLOCK_SIZE {
                for head in 0..KV_HEADS {
                    for dim in 0..head_dim {
                        let i = (slot * KV_HEADS + head) * head_dim + dim;
                        k[i] = pool_value(layer, slot, head, dim);
                        v[i] = -pool_value(layer + 1, slot, head, dim);
                    }
                }
            }
            let shape = (blocks, BLOCK_SIZE, KV_HEADS, head_dim);
            kv.kv_k[layer] = Tensor::from_vec(k, shape, &model.device).expect("k pool");
            kv.kv_v[layer] = Tensor::from_vec(v, shape, &model.device).expect("v pool");
        }
        let mut table = BlockTable::new();
        for b in 0..blocks {
            table.push_block(b);
        }
        table.set_num_tokens(ctx);
        (kv, table)
    }

    fn decode_logits(model: &Model, ctx: usize) -> Tensor {
        let (mut kv, table) = filled_kv(model, ctx);
        model
            .forward_decode_step(7, ctx - 1, &table, &mut kv)
            .expect("decode step")
    }

    #[test]
    fn metal_decode_dispatches_paged_attention_kernel() {
        let dev = Device::new_metal(0).expect("Metal device");
        let metal_model = mock(&dev, Backend::Metal);
        let cpu_model = mock(&Device::Cpu, Backend::Cpu);

        let _ = decode_logits(&metal_model, 128);
        assert!(
            metal_model.paged_attention_calls() > 0,
            "Metal decode must dispatch gpu_paged_attention_v2, not the copying matmul path"
        );

        let _ = decode_logits(&cpu_model, 128);
        assert_eq!(
            cpu_model.paged_attention_calls(),
            0,
            "CPU backend must keep the strided-view path"
        );
    }

    /// `engine.step` reaches attention through `forward_batch`, not
    /// `forward_decode_step`, so that is the entry the kernel has to serve.
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

    #[test]
    fn metal_batched_decode_dispatches_kernel_and_keeps_parity() {
        let dev = Device::new_metal(0).expect("Metal device");
        let metal_model = mock(&dev, Backend::Metal);
        let cpu_model = mock(&Device::Cpu, Backend::Cpu);

        let reference = batch_logits(&cpu_model, 128);
        assert_eq!(
            cpu_model.paged_attention_calls(),
            0,
            "the batched CPU path must not dispatch the Metal kernel"
        );

        let actual = batch_logits(&metal_model, 128);
        assert!(
            metal_model.paged_attention_calls() >= LAYERS as u64,
            "forward_batch on Metal must dispatch the kernel once per layer (got {})",
            metal_model.paged_attention_calls()
        );
        assert_eq!(reference.len(), actual.len(), "logits length");
        let mut worst = 0.0f32;
        for (r, a) in reference.iter().zip(actual.iter()) {
            let diff = (r - a).abs() / (1.0 + r.abs());
            if diff > worst {
                worst = diff;
            }
        }
        assert!(
            worst <= 1e-3,
            "batched Metal paged-attention logits drift {worst} from the reference path"
        );
    }

    #[test]
    fn metal_decode_logits_match_reference_across_contexts() {
        let dev = Device::new_metal(0).expect("Metal device");
        let metal_model = mock(&dev, Backend::Metal);
        let cpu_model = mock(&Device::Cpu, Backend::Cpu);

        for ctx in CONTEXTS {
            let reference = decode_logits(&cpu_model, ctx)
                .flatten_all()
                .expect("flatten")
                .to_vec1::<f32>()
                .expect("reference logits");
            let actual = decode_logits(&metal_model, ctx)
                .flatten_all()
                .expect("flatten")
                .to_vec1::<f32>()
                .expect("metal logits");

            assert_eq!(reference.len(), actual.len(), "ctx {ctx}: logits length");
            let mut worst = 0.0f32;
            for (r, a) in reference.iter().zip(actual.iter()) {
                let diff = (r - a).abs() / (1.0 + r.abs());
                if diff > worst {
                    worst = diff;
                }
            }
            assert!(
                worst <= 1e-3,
                "ctx {ctx}: Metal paged-attention logits drift {worst} from the reference path"
            );
        }
    }

    #[test]
    fn metal_decode_rejects_unrepresentable_kernel_params() {
        let dev = Device::new_metal(0).expect("Metal device");

        // head_dim 512 exceeds the kernel's MAX_HEAD_DIM (256).
        let mut wide =
            Model::create_mock_tiny(1024, 2, 1, LAYERS, VOCAB).expect("mock head_dim 512");
        wide.to_device(&dev).expect("move weights");
        wide.set_backend(Backend::Metal);
        assert_eq!(wide.config.head_dim(), 512);
        let (mut kv, table) = filled_kv(&wide, 32);
        assert!(
            wide.forward_decode_step(7, 31, &table, &mut kv).is_err(),
            "head_dim 512 is not representable by the kernel: must be Err, not a silent fallback"
        );

        // GQA group 16 exceeds the kernel's MAX_GROUP (8).
        let mut grouped =
            Model::create_mock_tiny(1024, 16, 1, LAYERS, VOCAB).expect("mock GQA group 16");
        grouped.to_device(&dev).expect("move weights");
        grouped.set_backend(Backend::Metal);
        let (mut kv, table) = filled_kv(&grouped, 32);
        assert!(
            grouped.forward_decode_step(7, 31, &table, &mut kv).is_err(),
            "GQA group 16 is not representable by the kernel: must be Err, not a silent fallback"
        );
    }
}
