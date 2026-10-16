use kanso_infer::bench::{
    ShareGptTurn, format_chatml_prompt, parse_sharegpt_dataset, run_trace_benchmark,
    tokenize_chatml,
};
use kanso_infer::engine::Engine;
use kanso_infer::model::Model;

#[test]
fn test_sharegpt_conversation_parsing() {
    let json_data = r#"[
        {
            "id": "conv_001",
            "conversations": [
                {"from": "human", "value": "Hello, how are you today?"},
                {"from": "gpt", "value": "I am doing well, thank you! How can I assist you?"},
                {"from": "human", "value": "Explain quantum computing briefly."},
                {"from": "gpt", "value": "Quantum computing uses qubits to perform complex calculations."}
            ]
        }
    ]"#;

    let trace = parse_sharegpt_dataset(json_data).expect("Failed to parse ShareGPT dataset");
    assert_eq!(
        trace.len(),
        2,
        "2 human turns should produce 2 TraceEntry requests"
    );

    // Both entries must have non-zero lengths and tokens
    assert!(trace[0].prompt_len > 0);
    assert!(trace[0].gen_len > 0);
    assert!(trace[0].prompt_tokens.is_some());
    assert_eq!(
        trace[0].prompt_tokens.as_ref().unwrap().len(),
        trace[0].prompt_len
    );

    assert!(
        trace[1].prompt_len > trace[0].prompt_len,
        "Turn 2 prompt must include Turn 1 history"
    );
    assert!(trace[1].gen_len > 0);
    assert!(trace[1].prompt_tokens.is_some());

    // Arrival steps must be staggered sequentially
    assert!(
        trace[1].arrival_step > trace[0].arrival_step,
        "Turn 2 arrival step must be greater than Turn 1 arrival step"
    );
}

#[test]
fn test_chatml_formatting_and_token_stream() {
    let turns = vec![
        ShareGptTurn {
            from: "system".to_string(),
            value: "You are a helpful coding assistant.".to_string(),
        },
        ShareGptTurn {
            from: "human".to_string(),
            value: "Write hello world in Rust.".to_string(),
        },
    ];

    let formatted = format_chatml_prompt(&turns);
    assert!(
        formatted.contains("<|im_start|>system\nYou are a helpful coding assistant.<|im_end|>\n")
    );
    assert!(formatted.contains("<|im_start|>user\nWrite hello world in Rust.<|im_end|>\n"));
    assert!(formatted.ends_with("<|im_start|>assistant\n"));

    let tokens = tokenize_chatml(&formatted);
    assert_eq!(tokens.len(), formatted.len());
    for &tok in &tokens {
        assert!(tok < 40, "Token ID must be within mock vocab range");
    }
}

#[test]
fn test_sharegpt_prefix_cache_hit_rate_above_60_percent() {
    // Dataset with shared system prompt across multi-turn conversation
    let json_data = r#"[
        {
            "id": "conv_shared_1",
            "conversations": [
                {"from": "system", "value": "A comprehensive system prompt providing general instructions and guidelines for mathematical analysis and continuous batching evaluation under paged memory architectures."},
                {"from": "human", "value": "First inquiry on systems architecture."},
                {"from": "gpt", "value": "Architecture involves paged memory tables."},
                {"from": "human", "value": "Follow up question on memory block reclamation."},
                {"from": "gpt", "value": "Blocks are reclaimed via reference counting."},
                {"from": "human", "value": "Third question on paged memory fragmentation."},
                {"from": "gpt", "value": "Paged attention eliminates internal and external memory fragmentation entirely."}
            ]
        },
        {
            "id": "conv_shared_2",
            "conversations": [
                {"from": "system", "value": "A comprehensive system prompt providing general instructions and guidelines for mathematical analysis and continuous batching evaluation under paged memory architectures."},
                {"from": "human", "value": "Second inquiry on prefix caching behavior."},
                {"from": "gpt", "value": "Prefix caching avoids redundant prompt KV computation."},
                {"from": "human", "value": "Follow up on cache hit rates."},
                {"from": "gpt", "value": "Hit rates exceed sixty percent on shared prefixes."},
                {"from": "human", "value": "Follow up on continuous batching scheduling."},
                {"from": "gpt", "value": "Continuous batching schedules iteration-level execution across requests."}
            ]
        }
    ]"#;

    let trace = parse_sharegpt_dataset(json_data).expect("Failed to parse ShareGPT dataset");
    assert_eq!(trace.len(), 6);

    let _max_tokens = trace
        .iter()
        .map(|t| t.prompt_len + t.gen_len)
        .max()
        .unwrap_or(256);
    let model = Model::create_mock_tiny(64, 4, 2, 2, 256).expect("Model creation failed");
    let mut engine = Engine::new(model, 16, 256, 16);

    let report =
        run_trace_benchmark(&mut engine, &trace, false).expect("Trace benchmark execution failed");

    assert_eq!(report.num_requests, 6);
    assert!(
        report.prefix_hit_rate >= 0.60,
        "Prefix cache hit rate must be >= 60%, got: {:.2}%",
        report.prefix_hit_rate * 100.0
    );
}

#[test]
fn test_invalid_json_handling() {
    let bad_json = r#"{"this is": "not valid sharegpt json format"#;
    let res = parse_sharegpt_dataset(bad_json);
    assert!(res.is_err(), "Invalid JSON must return EngineError");
}
