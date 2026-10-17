//! Ticket #48 / spec #61 story 3: one decode token at context 2048 must not cost more
//! than 1.3x one at context 128.
//!
//! No weight file is available offline, so this drives the real decode path
//! (`Model::forward_decode_step`) on a mock-weight model with SmolLM2-360M dimensions —
//! the same shape `tests/test_amx_decode_tok_s.rs` measures. The pool is left zeroed:
//! this is a latency test, and the KV *bytes* are what the old gather-copy paid for.
//!
//! Every backend the engine can pick on this platform is measured, because the copy the
//! ticket is about happens one layer down, inside each backend's GEMM.

use std::time::Instant;

use kanso_infer::kv_cache::{BlockTable, KvStorage};
use kanso_infer::model::Model;

/// SmolLM2-360M: hidden 960, 15 heads / 5 kv heads, 32 layers, vocab 49152.
const HIDDEN: usize = 960;
const Q_HEADS: usize = 15;
const KV_HEADS: usize = 5;
const LAYERS: usize = 32;
const VOCAB: usize = 49_152;

const SHORT_CTX: usize = 128;
const LONG_CTX: usize = 2048;
const REPS: usize = 5;

/// Median wall time of a single decode step with `ctx` tokens already cached.
fn decode_step_ms(model: &Model, ctx: usize, reps: usize) -> f64 {
    let block_size = model.config.block_size;
    let blocks = ctx.div_ceil(block_size);
    let mut kv_storage = KvStorage::new(
        model.config.num_hidden_layers,
        blocks,
        block_size,
        model.config.num_key_value_heads,
        model.config.head_dim(),
        &model.device,
    )
    .expect("kv storage");

    let mut block_table = BlockTable::new();
    for b in 0..blocks {
        block_table.push_block(b);
    }
    block_table.set_num_tokens(ctx);
    let pos = ctx - 1;

    // Warmup: rope table, Accelerate init, allocator arenas.
    for _ in 0..2 {
        let _ = model
            .forward_decode_step(1, pos, &block_table, &mut kv_storage)
            .expect("warmup decode step");
    }

    let mut times = Vec::with_capacity(reps);
    for _ in 0..reps {
        let start = Instant::now();
        let _ = model
            .forward_decode_step(7, pos, &block_table, &mut kv_storage)
            .expect("decode step");
        times.push(start.elapsed().as_secs_f64() * 1e3);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times[times.len() / 2]
}

#[test]
fn decode_latency_does_not_scale_with_context() {
    let mut model = Model::create_mock_tiny(HIDDEN, Q_HEADS, KV_HEADS, LAYERS, VOCAB)
        .expect("mock model at SmolLM2-360M dimensions");

    let mut backends = vec![kanso_infer::Backend::Cpu];
    // Default backend (metal on macOS) is what the engine runs unless told otherwise.
    if !matches!(kanso_infer::Backend::default(), kanso_infer::Backend::Cpu) {
        backends.push(kanso_infer::Backend::default());
    }
    if kanso_infer::accelerate::amx::has_accelerate_support() {
        backends.push(kanso_infer::Backend::Amx);
    }

    for backend in backends {
        model.set_backend(backend);
        let short = decode_step_ms(&model, SHORT_CTX, REPS);
        let long = decode_step_ms(&model, LONG_CTX, REPS);
        let ratio = long / short;
        println!(
            "#48 [{backend}] decode step: ctx{SHORT_CTX} {short:.2}ms, ctx{LONG_CTX} {long:.2}ms, ratio {ratio:.3} (limit 1.3)"
        );
        assert!(
            ratio <= 1.3,
            "[{backend}] decode at ctx{LONG_CTX} ({long:.2}ms) must stay within 1.3x of ctx{SHORT_CTX} ({short:.2}ms), got {ratio:.3}x"
        );
    }
}
