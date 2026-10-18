//! Ticket #56: engine robustness under KV pressure (preemption) + EOS retirement.
//! Seeded random mock weights (lesson #57: an `ones` mock is blind to KV corruption).

use kanso_infer::engine::Engine;
use kanso_infer::model::Model;
use std::collections::HashMap;

fn mock() -> Model {
    Model::create_mock_tiny(16, 4, 2, 2, 50).unwrap()
}

/// Runs the engine until it drains, collecting emitted tokens per request.
/// Panics if a step errors — under block pressure the engine must preempt, not
/// abort the whole batch.
fn run_to_completion(engine: &mut Engine, step_cap: usize) -> HashMap<u64, Vec<u32>> {
    let mut emitted: HashMap<u64, Vec<u32>> = HashMap::new();
    let mut steps = 0;
    while engine.has_unfinished_requests() {
        let outputs = engine
            .step()
            .expect("#56: KV pressure must preempt, not error out the whole batch");
        for out in outputs {
            emitted
                .entry(out.request_id)
                .or_default()
                .push(out.token_id);
        }
        steps += 1;
        assert!(
            steps < step_cap,
            "#56: engine livelocked under block pressure"
        );
    }
    emitted
}

/// 8 blocks × 4 tokens = 32 KV slots. Four requests of 12 prompt + 8 generated
/// tokens need 5 blocks each — 20 blocks in total, two resident requests already
/// exceed the pool — so the engine must preempt (recompute) the newest request
/// instead of silently skipping block allocation, which aborted the whole batch.
#[test]
fn test_small_block_pool_four_long_requests_all_finish() {
    let mut engine = Engine::new_with_budget(mock(), 2, 8, 4, usize::MAX).expect("engine");
    let ids: Vec<u64> = (0..4)
        .map(|i| {
            let prompt: Vec<u32> = (0..12).map(|t| ((i * 9 + t) % 40) as u32).collect();
            engine.add_request(prompt, 8)
        })
        .collect();

    let emitted = run_to_completion(&mut engine, 500);

    for id in &ids {
        assert_eq!(
            emitted.get(id).map(Vec::len),
            Some(8),
            "#56: request {} did not emit its full max_tokens",
            id
        );
    }
    assert_eq!(engine.running_count(), 0, "no request left running");
    assert_eq!(
        engine.free_blocks_count(),
        8,
        "#56: every block must be back in the pool once all requests finish"
    );
}

/// EOS must be recognised by the engine: the sequence stops short of `max_tokens`
/// and its KV blocks return to the pool in the *same* step.
#[test]
fn test_eos_stops_sequence_and_frees_blocks_in_same_step() {
    let prompt: Vec<u32> = vec![1, 2, 3, 4, 5];

    // Discover the deterministic greedy token this prompt produces.
    let mut probe = Engine::new_with_budget(mock(), 1, 8, 4, usize::MAX).expect("engine");
    let probe_id = probe.add_request(prompt.clone(), 32);
    let first = probe
        .step()
        .expect("probe step failed")
        .iter()
        .find(|o| o.request_id == probe_id)
        .expect("probe emitted no token")
        .token_id;
    assert!(
        !probe.sequences[0].is_finished(),
        "probe has no stop token configured and must keep running"
    );

    let mut engine = Engine::new_with_budget(mock(), 1, 8, 4, usize::MAX).expect("engine");
    let id = engine.add_request_with_stops(prompt, 32, vec![first]);
    let outputs = engine.step().expect("step failed");
    let out = outputs
        .iter()
        .find(|o| o.request_id == id)
        .expect("no output for request");

    assert_eq!(out.token_id, first);
    assert!(out.is_finished, "#56: EOS token must finish the sequence");
    assert_eq!(
        engine.sequences[0].generated_tokens,
        vec![first],
        "#56: engine must not generate past EOS"
    );
    assert!(
        !engine.has_unfinished_requests(),
        "#56: EOS request must be retired, not left Running holding blocks"
    );
    assert_eq!(
        engine.free_blocks_count(),
        8,
        "#56: EOS must return blocks to the pool in the same step"
    );
}

/// Ticket #61 F1: a client that hung up must not buy `max_tokens` more steps of engine
/// work — `cancel_request` retires the sequence and returns its blocks to the pool.
#[test]
fn test_cancel_request_retires_sequence_and_frees_blocks() {
    let mut engine = Engine::new_with_budget(mock(), 2, 16, 4, usize::MAX).expect("engine");
    let keep = engine.add_request(vec![1, 2, 3, 4], 8);
    let doomed = engine.add_request(vec![5, 6, 7, 8], 64);

    let first = engine.step().expect("prefill step");
    assert_eq!(first.len(), 2, "both requests prefill in one step");
    assert!(engine.free_blocks_count() < 16, "sequences hold KV blocks");

    assert!(engine.cancel_request(doomed), "live request must cancel");
    assert!(
        !engine.sequences.iter().any(|s| s.id == doomed),
        "a cancelled request leaves the sequence vector"
    );

    // The survivor keeps decoding; the cancelled one never appears again.
    let outputs = engine.step().expect("decode step");
    assert!(
        outputs.iter().all(|o| o.request_id == keep),
        "cancelled request must not emit: {outputs:?}"
    );

    // Draining to the end returns every remaining block, so the cancel freed the doomed
    // request's share rather than leaking it.
    while engine.has_unfinished_requests() {
        engine.step().expect("decode step");
        engine.drain_finished();
    }
    assert_eq!(engine.free_blocks_count(), 16, "blocks leaked after cancel");
    assert!(
        !engine.cancel_request(doomed),
        "cancelling twice is a no-op"
    );
}
