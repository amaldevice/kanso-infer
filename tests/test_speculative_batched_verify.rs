//! Ticket #90 acceptance: the speculative verifier must ride `forward_batch`, so a
//! speculative step counts against `Model::forward_batch_calls` like any other step.
//!
//! Before this ticket the verifier ran `forward_verify_tokens` on its own, which left the
//! counter blind to speculative traffic and made "one batched forward per step" a claim
//! that only held for non-speculative serving.

use kanso_infer::engine::Engine;
use kanso_infer::model::Model;

#[test]
fn speculative_step_moves_the_forward_batch_counter() {
    let model = Model::create_mock_tiny(16, 4, 2, 2, 50).unwrap();
    let mut engine = Engine::new(model, 1, 20, 4).expect("engine");
    let req_id = engine.add_request(vec![1, 2, 3], 10);

    // Prefill: one ordinary batched forward.
    let before = engine.model.forward_batch_calls();
    engine.step().expect("prefill");
    assert_eq!(
        engine.model.forward_batch_calls() - before,
        1,
        "prefill rides forward_batch"
    );

    // A speculative step over one candidate must add exactly one batched forward: the
    // candidate block is verified in that same forward, not in a separate one.
    let before = engine.model.forward_batch_calls();
    let result = engine
        .speculative_step(req_id, &[7])
        .expect("speculative step");
    assert_eq!(
        engine.model.forward_batch_calls() - before,
        1,
        "a speculative step must add exactly one forward_batch call, got {}",
        engine.model.forward_batch_calls() - before
    );
    assert_eq!(
        result.committed_tokens.len(),
        result.num_accepted + 1,
        "verification result must survive the move onto forward_batch"
    );
}
