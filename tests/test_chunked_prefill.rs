use kanso_infer::engine::Engine;
use kanso_infer::model::Model;
use std::time::Instant;

#[test]
fn test_chunked_prefill_admission_and_budget_enforcement() {
    let model = Model::create_mock_tiny(16, 4, 2, 2, 50).unwrap();
    // token_budget = 4, block_size = 4, total_blocks = 30
    let mut engine = Engine::new_with_budget(model, 2, 30, 4, 4);
    assert_eq!(engine.token_budget, 4);

    // Prompt of 10 tokens: will need ceil(10/4) = 3 steps (chunk 4, chunk 4, chunk 2)
    let prompt: Vec<u32> = (0..10).collect();
    let id = engine.add_request(prompt, 2);

    // Step 1: Chunk 0..4 prefilled, 0 tokens generated
    let out1 = engine.step().expect("step 1 failed");
    assert!(
        out1.is_empty(),
        "Partial prefill chunk must not emit output"
    );
    assert_eq!(engine.sequences[0].prefilled_tokens, 4);

    // Step 2: Chunk 4..8 prefilled, 0 tokens generated
    let out2 = engine.step().expect("step 2 failed");
    assert!(
        out2.is_empty(),
        "Partial prefill chunk must not emit output"
    );
    assert_eq!(engine.sequences[0].prefilled_tokens, 8);

    // Step 3: Chunk 8..10 prefilled (prompt finished), 1st decode token generated
    let out3 = engine.step().expect("step 3 failed");
    assert_eq!(
        out3.len(),
        1,
        "Final prefill chunk emits first generated token"
    );
    assert_eq!(out3[0].request_id, id);
    assert!(!out3[0].is_finished);
    assert_eq!(engine.sequences[0].prefilled_tokens, 10);
    assert_eq!(engine.sequences[0].generated_tokens.len(), 1);

    // Step 4: 2nd decode token generated, sequence finishes
    let out4 = engine.step().expect("step 4 failed");
    assert_eq!(out4.len(), 1);
    assert_eq!(out4[0].request_id, id);
    assert!(out4[0].is_finished);
    assert_eq!(engine.sequences[0].generated_tokens.len(), 2);
}

#[test]
fn test_co_scheduling_active_decode_and_prefill_chunks() {
    let model = Model::create_mock_tiny(16, 4, 2, 2, 50).unwrap();
    // token_budget = 5, max_running = 2
    let mut engine = Engine::new_with_budget(model, 2, 30, 4, 5);

    // Request 1: prompt len 2, gen len 4. Admitted in step 1.
    let id1 = engine.add_request(vec![1, 2], 4);

    // Step 1: Req 1 prefills 2 tokens and generates token 1
    let out1 = engine.step().expect("step 1 failed");
    assert_eq!(out1.len(), 1);
    assert_eq!(out1[0].request_id, id1);

    // Req 2 arrives: long prompt of 8 tokens
    let id2 = engine.add_request(vec![10, 11, 12, 13, 14, 15, 16, 17], 2);

    // Step 2:
    // - Req 1 is active decode: uses 1 token from budget. Emits token 2.
    // - Remaining budget is 5 - 1 = 4 tokens.
    // - Req 2 prefills chunk 0..4 (4 tokens). No output for Req 2 yet.
    // Active decode MUST NOT be stalled!
    let out2 = engine.step().expect("step 2 failed");
    assert_eq!(out2.len(), 1, "Only Req 1 decode token should be emitted");
    assert_eq!(out2[0].request_id, id1);
    assert_eq!(engine.sequences[1].prefilled_tokens, 4);

    // Step 3:
    // - Req 1 is active decode: uses 1 token from budget. Emits token 3.
    // - Remaining budget is 4 tokens.
    // - Req 2 needs 4 tokens (4..8): completes prefill and emits token 1!
    // Both active decode and finished prefill emit tokens in the same iteration!
    let out3 = engine.step().expect("step 3 failed");
    assert_eq!(
        out3.len(),
        2,
        "Both Req 1 decode and Req 2 first token emitted"
    );
    let req_ids: Vec<u64> = out3.iter().map(|o| o.request_id).collect();
    assert!(req_ids.contains(&id1));
    assert!(req_ids.contains(&id2));
    assert_eq!(engine.sequences[1].prefilled_tokens, 8);
}

#[test]
fn test_prefix_cache_lookup_reused_before_first_chunk() {
    let model = Model::create_mock_tiny(16, 4, 2, 2, 50).unwrap();
    // block_size = 4. Prompt of 8 tokens = exactly 2 blocks.
    let mut engine = Engine::new_with_budget(model, 2, 30, 4, 16);

    let prefix_prompt: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let _id1 = engine.add_request(prefix_prompt.clone(), 1);

    // Run Req 1 to completion so its blocks are stored in prefix cache
    while engine.has_unfinished_requests() {
        engine.step().expect("step failed");
    }

    // Now request 2 arrives with shared prefix + 2 new tokens
    let mut prompt2 = prefix_prompt.clone();
    prompt2.extend_from_slice(&[9, 10]);
    let id2 = engine.add_request(prompt2, 1);

    // Before first chunk is scheduled, prefix cache lookup should reuse the 2 cached blocks
    let out = engine.step().expect("step failed");
    // Seq 2 only needed to prefill the 2 remaining tokens (9, 10), then emits its token
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].request_id, id2);
    let seq2 = &engine.sequences[1];
    assert_eq!(seq2.prefilled_tokens, 10);
    // Sequence 2's block table must contain the reused prefix blocks
    assert!(seq2.block_table.blocks().len() >= 2);
}

#[test]
fn test_streaming_benchmark_2048_prompt_no_itl_spike() {
    let model = Model::create_mock_tiny(16, 4, 2, 2, 50).unwrap();
    // token_budget = 256 tokens per step, max_running = 4, total_blocks = 600, block_size = 16
    let mut engine = Engine::new_with_budget(model, 4, 600, 16, 256);

    // Active generating request 1
    let id_active = engine.add_request(vec![1, 2, 3, 4], 30);

    // Run 2 steps so id_active is actively decoding
    engine.step().expect("warmup step 1 failed");
    engine.step().expect("warmup step 2 failed");

    // Concurrent arrival of 2048-token prompt
    let huge_prompt: Vec<u32> = (0..2048).map(|i| (i % 45) as u32).collect();
    let id_huge = engine.add_request(huge_prompt, 2);

    let mut active_req_intervals = Vec::new();
    let mut last_step_emitted = engine.current_step;
    let mut huge_emitted_count = 0;

    // Track active request generation across consecutive steps while 2048 prompt is chunked
    while engine.has_unfinished_requests() && engine.current_step < 50 {
        let step_start = Instant::now();
        let outputs = engine.step().expect("step failed");
        let step_elapsed_ms = step_start.elapsed().as_secs_f64() * 1000.0;

        let emitted_for_active = outputs.iter().any(|o| o.request_id == id_active);
        if emitted_for_active {
            let step_diff = engine.current_step - last_step_emitted;
            // Active generating request MUST NOT stall: exactly 1 step between tokens!
            assert_eq!(
                step_diff, 1,
                "Active request stalled! Inter-token step diff was {}",
                step_diff
            );
            active_req_intervals.push(step_elapsed_ms);
            last_step_emitted = engine.current_step;
        }

        if outputs.iter().any(|o| o.request_id == id_huge) {
            huge_emitted_count += 1;
        }
    }

    // Verify active request was never stalled across all its generation steps
    assert!(
        active_req_intervals.len() >= 10,
        "Active request should have produced multiple tokens during chunking"
    );

    // Calculate p99 ITL and mean ITL to verify no latency spikes
    active_req_intervals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean_itl: f64 =
        active_req_intervals.iter().sum::<f64>() / active_req_intervals.len() as f64;
    let p99_idx = ((active_req_intervals.len() as f64) * 0.99).ceil() as usize - 1;
    let p99_itl = active_req_intervals[p99_idx.min(active_req_intervals.len() - 1)];

    // ITL p99 must be stably bounded (no catastrophic 40x-100x spike from monolithic 2048 prefill)
    assert!(
        p99_itl < mean_itl * 15.0 + 100.0,
        "p99 ITL latency spiked: mean={:.2}ms, p99={:.2}ms",
        mean_itl,
        p99_itl
    );
    assert!(
        huge_emitted_count >= 1,
        "2048-token prompt must be admitted and emit within 50 steps"
    );
}
