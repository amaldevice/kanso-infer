//! Ticket #106: a recycled KV block must not leak the previous request's contents.
//!
//! Request 1 frees its blocks without erasing them; request 2 reallocates the same
//! physical blocks and writes only `[0..ctx_len)`, so attention used to read the stale
//! tail of block 1. Same prompt twice, same engine, greedy: the tokens must match.
//! Seeded random mock weights (lesson #57: an `ones` mock is blind to KV corruption).

use kanso_infer::engine::Engine;
use kanso_infer::model::Model;

fn engine() -> Engine {
    let model = Model::create_mock_tiny(16, 2, 1, 1, 32).expect("mock model");
    Engine::new_with_budget(
        model,
        /* max_running */ 1,
        /* total_blocks */ 8,
        /* block_size */ 4,
        usize::MAX,
    )
    .expect("engine")
}

/// Runs one greedy request to completion and returns the tokens it generated.
fn generated(engine: &mut Engine, prompt: &[u32], max_tokens: usize) -> Vec<u32> {
    let id = engine.add_request(prompt.to_vec(), max_tokens);
    let mut steps = 0;
    while engine.has_unfinished_requests() {
        engine.step().expect("step failed");
        steps += 1;
        assert!(steps < 50, "engine did not drain in 50 steps");
    }
    let tokens = engine
        .sequences
        .iter()
        .find(|s| s.id == id)
        .expect("request id not found")
        .generated_tokens
        .clone();
    engine.drain_finished();
    tokens
}

#[test]
fn test_same_prompt_twice_sequential_is_identical() {
    let mut engine = engine();
    let prompt = vec![1, 2, 3];

    let first = generated(&mut engine, &prompt, 3);
    let second = generated(&mut engine, &prompt, 3);

    assert_eq!(first.len(), 3, "first request must emit max_tokens tokens");
    assert_eq!(
        first, second,
        "ticket #106: the same prompt must decode identically twice in a row; \
         the second request read stale KV from the reused block"
    );
}
