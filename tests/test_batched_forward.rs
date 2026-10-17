//! Ticket #45 acceptance: continuous batching must be ONE batched forward per
//! `step()`, not a per-sequence loop over batch-1 forwards.

use kanso_infer::bench::run_synthetic_benchmark;
use kanso_infer::engine::Engine;
use kanso_infer::model::Model;

/// Every sequence in a step — decode tokens and prefill chunks alike — must ride
/// exactly one model forward. The per-sequence loop this ticket replaces shows up
/// here as a forward count that grows with the batch size.
#[test]
fn test_one_forward_call_per_step_regardless_of_batch() {
    let model = Model::create_mock_tiny(16, 4, 2, 2, 50).unwrap();
    let mut engine = Engine::new(model, /* max_running */ 4, /* blocks */ 40, /* block_size */ 4);
    for i in 0..4u32 {
        engine.add_request(vec![i, i + 1, i + 2, i + 3], 8);
    }

    // Step 1: all four requests prefill (each a 4-token chunk) in the same step.
    let before = engine.model.forward_calls();
    let out = engine.step().unwrap();
    assert_eq!(out.len(), 4, "all four prefills must complete in step 1");
    assert_eq!(
        engine.model.forward_calls() - before,
        1,
        "four prefill chunks must share one forward"
    );

    // Step 2: all four requests decode one token each.
    let before = engine.model.forward_calls();
    let out = engine.step().unwrap();
    assert_eq!(out.len(), 4, "four sequences decode in step 2");
    assert_eq!(
        engine.model.forward_calls() - before,
        1,
        "four decode tokens must share one forward"
    );
}

/// SmolLM2-360M shapes with the layer count trimmed so the test still finishes:
/// every linear keeps its real `[out, in]` size, and decode cost is dominated by
/// reading those weights — which is exactly what batching amortizes.
const HIDDEN: usize = 960;
const Q_HEADS: usize = 15;
const KV_HEADS: usize = 3;
const LAYERS: usize = 8;
const VOCAB: usize = 49_152;
const PROMPT: usize = 8;
const GEN: usize = 16;

/// Best of two samples: a single wall-clock sample is flaky under load.
fn best_tok_s(batch: usize) -> f64 {
    let mut model = Model::create_mock_tiny(HIDDEN, Q_HEADS, KV_HEADS, LAYERS, VOCAB)
        .expect("mock model at SmolLM2-360M shapes");
    model.set_backend(kanso_infer::Backend::Cpu);
    let mut engine = Engine::new(model, batch, 8 * 8, 16);
    let mut best = 0.0f64;
    for _ in 0..2 {
        let report = run_synthetic_benchmark(&mut engine, batch, PROMPT, GEN, false)
            .expect("synthetic benchmark");
        best = best.max(report.total_tokens_per_sec);
    }
    best
}

#[test]
fn test_batch8_throughput_at_least_4x_batch1() {
    let single = best_tok_s(1);
    let batched = best_tok_s(8);
    println!(
        "total tok/s @ SmolLM2-360M shapes ({LAYERS} layers, {PROMPT} prompt + {GEN} gen): \
         batch-1={single:.2}, batch-8={batched:.2}, ratio={:.2}x",
        batched / single
    );
    assert!(
        batched >= 4.0 * single,
        "batch-8 totalled {batched:.2} tok/s vs batch-1 {single:.2} tok/s \
         ({:.2}x): batching is not amortizing the weight read",
        batched / single
    );
}
