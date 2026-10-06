use kanso_infer::engine::Engine;
use kanso_infer::kv_cache::{BlockPool, BlockTable};
use kanso_infer::model::Model;
use kanso_infer::speculative::{SpeculativeEngine, SpeculativeVerifier, VerificationResult};

#[test]
fn test_draft_candidate_verification_acceptance_and_rejection() {
    let model = Model::create_mock_tiny(16, 4, 2, 2, 50).unwrap();
    let mut engine = Engine::new(model, 1, 20, 4);

    let prompt = vec![1, 2, 3];
    let req_id = engine.add_request(prompt.clone(), 10);

    // Step 1: prefill prompt
    let out = engine.step().expect("prefill failed");
    assert_eq!(out.len(), 1);
    let first_token = out[0].token_id;

    // Check what the target model would predict next in normal autoregressive mode
    let target_next_1 = {
        let seq = &engine.sequences[0];
        let pos = seq.prompt_tokens.len() + seq.generated_tokens.len() - 1;
        let last_tok = *seq.generated_tokens.last().unwrap();
        let logits = engine
            .model
            .forward_decode_step(last_tok, pos, &seq.block_table, &mut engine.kv_storage)
            .unwrap();
        kanso_infer::sampler::sample_token(&logits, 0.0, 1.0, 42).unwrap()
    };

    // Case A: Perfect draft match (all K accepted)
    // We draft target_next_1 as candidate 1.
    // If target accepts candidate 1, it will also evaluate candidate 1 and produce bonus token.
    let result = engine
        .speculative_step(req_id, &[target_next_1])
        .expect("speculative step failed");

    assert_eq!(result.num_accepted, 1);
    assert_eq!(result.num_rejected, 0);
    assert_eq!(result.accepted_tokens, vec![target_next_1]);
    assert_eq!(result.committed_tokens.len(), 2); // candidate 1 + bonus token
    // More than 1 token emitted per target step (> 1.0 acceptance rate)
    assert!(result.num_accepted as f32 >= 1.0);

    // Case B: Deliberate rejection
    // Propose an invalid candidate within vocab range that does not match target_next_1
    let bad_token = (target_next_1 + 1) % 50;
    let bad_candidates = vec![bad_token];
    let result_rejected = engine
        .speculative_step(req_id, &bad_candidates)
        .expect("speculative step with bad candidate failed");

    assert_eq!(result_rejected.num_accepted, 0);
    assert_eq!(result_rejected.num_rejected, 1);
    assert!(result_rejected.accepted_tokens.is_empty());
    assert_eq!(result_rejected.committed_tokens.len(), 1);
    assert_ne!(result_rejected.emitted_token, bad_token);
}

#[test]
fn test_atomic_rollback_and_zero_memory_leak() {
    let mut pool = BlockPool::new(10);
    let mut table = BlockTable::new();
    let block_size = 2;

    // Initial state: 2 tokens (occupies 1 block)
    let b0 = pool.allocate().unwrap();
    table.push_block(b0);
    table.set_num_tokens(2);
    let free_before = pool.free_blocks_count();
    assert_eq!(free_before, 9);

    // Speculate 4 more tokens -> total 6 tokens (needs 3 blocks, allocate 2 more blocks)
    let b1 = pool.allocate().unwrap();
    let b2 = pool.allocate().unwrap();
    table.push_block(b1);
    table.push_block(b2);
    table.set_num_tokens(6);
    assert_eq!(pool.free_blocks_count(), 7);

    // Rollback to 2 tokens: unverified blocks (b1, b2) must be recycled atomically
    let recycled = table.rollback_blocks(2, block_size, &mut pool);
    assert_eq!(recycled, vec![b2, b1]);
    assert_eq!(table.num_tokens(), 2);
    assert_eq!(table.blocks(), &[b0]);
    assert_eq!(pool.free_blocks_count(), free_before);

    // Test rollback_blocks directly on BlockPool
    let extra_blocks = vec![pool.allocate().unwrap(), pool.allocate().unwrap()];
    assert_eq!(pool.free_blocks_count(), free_before - 2);
    pool.rollback_blocks(&extra_blocks);
    assert_eq!(pool.free_blocks_count(), free_before);

    // Repeated speculation and rejection cycles on Engine
    let model = Model::create_mock_tiny(16, 4, 2, 2, 50).unwrap();
    let mut engine = Engine::new(model, 1, 15, 2); // small block_size=2 to force frequent allocations
    let req_id = engine.add_request(vec![1, 2], 20);
    engine.step().unwrap();

    let baseline_free = engine.free_blocks_count();
    for _ in 0..10 {
        // Repeatedly propose candidates that get rejected
        let _ = engine.speculative_step(req_id, &[49, 48, 47]).unwrap();
    }
    // All speculative allocations from rejections must be cleanly recycled
    // The only blocks retained are those for legitimately committed tokens
    let seq = &engine.sequences[0];
    let expected_needed = (seq.block_table.num_tokens() + 2 - 1) / 2;
    assert_eq!(seq.block_table.blocks().len(), expected_needed);
    assert_eq!(engine.free_blocks_count(), 15 - expected_needed);
}

#[test]
fn test_greedy_token_parity_with_autoregressive_generation() {
    let prompt = vec![5, 12, 19, 23];
    let num_tokens_to_generate = 8;

    // 1. Standard autoregressive generation
    let mut autoregressive_tokens = Vec::new();
    {
        let model = Model::create_mock_tiny(16, 4, 2, 2, 50).unwrap();
        let mut engine = Engine::new(model, 1, 30, 4);
        engine.add_request(prompt.clone(), num_tokens_to_generate);

        while engine.has_unfinished_requests() {
            let out = engine.step().unwrap();
            for item in out {
                autoregressive_tokens.push(item.token_id);
            }
        }
    }

    // 2. Speculative decoding generation with SpeculativeEngine
    let mut speculative_tokens = Vec::new();
    {
        let target_model = Model::create_mock_tiny(16, 4, 2, 2, 50).unwrap();
        let draft_model = Model::create_mock_tiny(16, 4, 2, 2, 50).unwrap();
        let mut spec_engine = SpeculativeEngine::new(
            target_model,
            Some(draft_model),
            /* max_running */ 1,
            /* total_blocks */ 30,
            /* block_size */ 4,
            /* gamma */ 3,
        );

        let id = spec_engine.add_request(prompt.clone(), num_tokens_to_generate);
        while spec_engine.has_unfinished_requests() {
            let out = spec_engine.step().unwrap();
            for item in out {
                if item.request_id == id {
                    speculative_tokens.push(item.token_id);
                }
            }
        }
    }

    assert_eq!(
        autoregressive_tokens, speculative_tokens,
        "Speculative decoding output must have 100% token parity with standard autoregressive generation"
    );
}

#[test]
fn test_speculative_multi_candidate_acceptance_rate_greater_than_one() {
    let model = Model::create_mock_tiny(16, 4, 2, 2, 50).unwrap();
    let mut engine = Engine::new(model, 1, 30, 4);

    let prompt = vec![2, 4, 6];
    let req_id = engine.add_request(prompt, 15);
    let out = engine.step().unwrap();
    assert_eq!(out.len(), 1);

    // Compute the next 3 ground-truth autoregressive tokens
    let mut current_seq = [engine.sequences[0].prompt_tokens.as_slice(), engine.sequences[0].generated_tokens.as_slice()].concat();
    let mut true_candidates = Vec::new();
    for _ in 0..3 {
        let logits = engine.model.forward_single_prompt(&current_seq).unwrap();
        let tok = kanso_infer::sampler::sample_token(&logits, 0.0, 1.0, 42).unwrap();
        current_seq.push(tok);
        true_candidates.push(tok);
    }

    // Propose all 3 ground truth tokens as candidates (K = 3)
    let result = engine
        .speculative_step(req_id, &true_candidates)
        .expect("speculative multi-token step failed");

    // On matched distributions:
    // All 3 tokens accepted + 1 bonus token emitted = 4 committed tokens in 1 target verification pass!
    assert_eq!(result.num_accepted, 3);
    assert_eq!(result.num_rejected, 0);
    assert_eq!(result.accepted_tokens, true_candidates);
    assert_eq!(result.committed_tokens.len(), 4);
    // Acceptance rate is 3.0 accepted tokens per target step (> 1.0!)
    assert!(result.num_accepted as f64 > 1.0);
}

#[test]
fn test_partial_candidate_acceptance_and_kv_rollback() {
    let model = Model::create_mock_tiny(16, 4, 2, 2, 50).unwrap();
    // Small block_size=2 so adding candidates triggers block allocation
    let mut engine = Engine::new(model, 1, 30, 2);

    let prompt = vec![1, 3];
    let req_id = engine.add_request(prompt, 15);
    engine.step().unwrap(); // prefill

    let current_seq = [engine.sequences[0].prompt_tokens.as_slice(), engine.sequences[0].generated_tokens.as_slice()].concat();
    let logits = engine.model.forward_single_prompt(&current_seq).unwrap();
    let true_first = kanso_infer::sampler::sample_token(&logits, 0.0, 1.0, 42).unwrap();
    let bad_second = (true_first + 1) % 50;

    // Candidate 1 matches, candidate 2 intentionally mismatches
    let candidates = vec![true_first, bad_second];
    let result = engine
        .speculative_step(req_id, &candidates)
        .expect("speculative step failed");

    // Candidate 1 accepted, candidate 2 rejected
    assert_eq!(result.num_accepted, 1);
    assert_eq!(result.num_rejected, 1);
    assert_eq!(result.accepted_tokens, vec![true_first]);
    assert_eq!(result.committed_tokens.len(), 2); // true_first + target correction

    // Memory integrity check
    let seq = &engine.sequences[0];
    let expected_blocks = (seq.block_table.num_tokens() + 2 - 1) / 2;
    assert_eq!(seq.block_table.blocks().len(), expected_blocks);
    assert_eq!(engine.free_blocks_count(), 30 - expected_blocks);
}

#[test]
fn test_speculative_parity_with_adversarial_draft_candidates() {
    let prompt = vec![3, 7, 11];
    let num_tokens = 6;

    // Ground truth autoregressive decode
    let mut ar_tokens = Vec::new();
    {
        let model = Model::create_mock_tiny(16, 4, 2, 2, 50).unwrap();
        let mut engine = Engine::new(model, 1, 30, 4);
        engine.add_request(prompt.clone(), num_tokens);
        while engine.has_unfinished_requests() {
            for out in engine.step().unwrap() {
                ar_tokens.push(out.token_id);
            }
        }
    }

    // Speculative decoding where draft model has completely different weights (adversarial)
    let mut spec_tokens = Vec::new();
    {
        let target_model = Model::create_mock_tiny(16, 4, 2, 2, 50).unwrap();
        // Draft model with hidden_size 16, but different layer count
        let draft_model = Model::create_mock_tiny(16, 4, 2, 1, 50).unwrap();
        let mut spec_engine = SpeculativeEngine::new(
            target_model,
            Some(draft_model),
            1,
            30,
            4,
            /* gamma */ 2,
        );
        let id = spec_engine.add_request(prompt.clone(), num_tokens);
        while spec_engine.has_unfinished_requests() {
            for out in spec_engine.step().unwrap() {
                if out.request_id == id {
                    spec_tokens.push(out.token_id);
                }
            }
        }
    }

    assert_eq!(
        ar_tokens, spec_tokens,
        "Even with mismatched draft model, speculative decoding must produce 100% token parity"
    );
}
