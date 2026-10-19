//! Ticket #90 acceptance: the speculative verifier must ride `forward_batch`, so a
//! speculative step counts against `Model::forward_batch_calls` like any other step.
//!
//! Before this ticket the verifier ran `forward_verify_tokens` on its own, which left the
//! counter blind to speculative traffic and made "one batched forward per step" a claim
//! that only held for non-speculative serving.

use kanso_infer::engine::Engine;
use kanso_infer::kv_cache::BlockTable;
use kanso_infer::model::Model;
use kanso_infer::sampler::greedy_token;

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

/// AC #3: the batched verifier must reach the same conclusion as the decode path on the
/// same prefix. `forward_decode_step` at the candidate position gives the target's argmax;
/// a verifier that drifted would reject a candidate the target itself would have emitted.
#[test]
fn batched_verification_agrees_with_the_decode_path() {
    let model = Model::create_mock_tiny(16, 4, 2, 2, 50).unwrap();
    let mut engine = Engine::new(model, 1, 20, 4).expect("engine");
    let req_id = engine.add_request(vec![1, 2, 3], 10);
    engine.step().expect("prefill");

    let expected = {
        let seq = &engine.sequences[0];
        let last_tok = *seq.generated_tokens.last().unwrap();
        let pos = seq.prompt_tokens.len() + seq.generated_tokens.len() - 1;
        let table = BlockTable::with_blocks(
            seq.block_table.blocks().to_vec(),
            seq.block_table.num_tokens(),
        );
        let logits = engine
            .model
            .forward_decode_step(last_tok, pos, &table, &mut engine.kv_storage)
            .expect("decode reference");
        greedy_token(&logits.flatten_all().unwrap().to_vec1::<f32>().unwrap())
    };

    let result = engine
        .speculative_step(req_id, &[expected])
        .expect("speculative step");
    assert_eq!(result.num_accepted, 1, "decode-path token must be accepted");
    assert_eq!(result.accepted_tokens, vec![expected]);
}
