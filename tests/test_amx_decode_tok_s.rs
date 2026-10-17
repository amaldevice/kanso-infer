//! Ticket #52 acceptance: decode tok/s with `--backend amx` must beat `--backend cpu`.
//!
//! No weight file is available offline, so this drives the real decode path
//! (`Model::forward_decode_step`) on a mock-weight model with SmolLM2-360M dimensions.
//! Weights are random but every per-token matmul has the real `[out, in]` size, which is
//! what the ticket measured. With a real GGUF the same comparison comes out of
//! `kanso bench --model smollm2-360m.gguf --backend {amx,cpu}`.

use std::time::Instant;

use kanso_infer::kv_cache::{BlockTable, KvStorage};
use kanso_infer::model::Model;

/// SmolLM2-360M: hidden 960, 15 heads / 5 kv heads, 32 layers, vocab 49152.
const HIDDEN: usize = 960;
const Q_HEADS: usize = 15;
const KV_HEADS: usize = 5;
const LAYERS: usize = 32;
const VOCAB: usize = 49_152;
const STEPS: usize = 32;

fn decode_tok_s(model: &Model, steps: usize) -> f64 {
    let mut kv_storage = KvStorage::new(
        model.config.num_hidden_layers,
        steps + 2,
        model.config.block_size,
        model.config.num_key_value_heads,
        model.config.head_dim(),
        &model.device,
    )
    .expect("kv storage");
    let mut block_table = BlockTable::new();
    for b in 0..(steps + 2).div_ceil(model.config.block_size) {
        block_table.push_block(b);
    }

    // Warmup step (first call allocates scratch / lazily initializes Accelerate).
    let _ = model
        .forward_decode_step(1, 0, &block_table, &mut kv_storage)
        .expect("warmup decode step");
    block_table.set_num_tokens(1);

    let start = Instant::now();
    for pos in 1..=steps {
        let _ = model
            .forward_decode_step(7, pos, &block_table, &mut kv_storage)
            .expect("decode step");
        block_table.set_num_tokens(pos + 1);
    }
    let elapsed = start.elapsed().as_secs_f64();
    steps as f64 / elapsed
}

#[test]
fn test_decode_tok_s_amx_beats_cpu_at_smollm2_dimensions() {
    if !kanso_infer::accelerate::amx::has_accelerate_support() {
        eprintln!("SKIP (non-macOS): amx is a scalar fallback here, cannot beat cpu");
        return;
    }
    let mut model = Model::create_mock_tiny(HIDDEN, Q_HEADS, KV_HEADS, LAYERS, VOCAB)
        .expect("mock model at SmolLM2-360M dimensions");

    // Median of 3 samples each: single wall-clock samples are flaky under load.
    let mut amx = Vec::with_capacity(3);
    let mut cpu = Vec::with_capacity(3);
    for _ in 0..3 {
        model.set_backend(kanso_infer::Backend::Amx);
        amx.push(decode_tok_s(&model, STEPS));
        model.set_backend(kanso_infer::Backend::Cpu);
        cpu.push(decode_tok_s(&model, STEPS));
    }
    amx.sort_by(|a, b| a.partial_cmp(b).unwrap());
    cpu.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let (amx_tok_s, cpu_tok_s) = (amx[1], cpu[1]);

    println!("decode tok/s @ SmolLM2-360M dims: amx={amx_tok_s:.2}, cpu={cpu_tok_s:.2}");
    assert!(
        amx_tok_s > cpu_tok_s,
        "amx decode ({amx_tok_s:.2} tok/s) must beat cpu ({cpu_tok_s:.2} tok/s)"
    );
}
