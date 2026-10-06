use kanso_infer::bench::{
    evaluate_run, fnv1a_64, parse_trace_file, run_synthetic_benchmark, run_trace_benchmark,
    BenchmarkReport, TraceEntry,
};
use kanso_infer::engine::Engine;
use kanso_infer::model::Model;
use std::fs;

#[test]
fn test_fnv1a_64_checksum_vectors() {
    // 1. Empty token slice must match FNV offset basis 0xcbf29ce484222325
    assert_eq!(fnv1a_64(&[]), 0xcbf29ce484222325);

    // 2. Known single-token and multi-token hashes (wrapping mul 0x100000001b3)
    let hash_single = fnv1a_64(&[42]);
    assert_eq!(hash_single, 0xaf63a74c8601927d);

    let hash_multi = fnv1a_64(&[1, 2, 3]);
    assert_eq!(hash_multi, 0xd0aa6218672cf5ab);

    // 3. Order-sensitivity
    let hash_permuted = fnv1a_64(&[3, 2, 1]);
    assert_ne!(hash_multi, hash_permuted);

    // 4. Determinism
    assert_eq!(fnv1a_64(&[1, 2, 3]), fnv1a_64(&[1, 2, 3]));
}

#[test]
fn test_fnv1a_64_parity_verification() {
    // Reference greedy token stream simulation
    let cpu_tokens = vec![12u32, 45, 99, 1024, 7];
    let neon_tokens = vec![12u32, 45, 99, 1024, 7];
    let metal_tokens = vec![12u32, 45, 99, 1024, 7];
    let divergent_tokens = vec![12u32, 45, 99, 1025, 7];

    let cpu_hash = fnv1a_64(&cpu_tokens);
    let neon_hash = fnv1a_64(&neon_tokens);
    let metal_hash = fnv1a_64(&metal_tokens);
    let div_hash = fnv1a_64(&divergent_tokens);

    assert_eq!(cpu_hash, neon_hash, "NEON tokens must match CPU reference hash");
    assert_eq!(cpu_hash, metal_hash, "Metal tokens must match CPU reference hash");
    assert_ne!(cpu_hash, div_hash, "Divergent tokens must produce different hash");
}

#[test]
fn test_jsonl_trace_parsing() {
    let tmp_path = std::env::temp_dir().join("kanso_test_trace.jsonl");
    let content = r#"{"prompt_len": 128, "gen_len": 64, "arrival_step": 0}
{"prompt_len": 256, "gen_len": 32, "arrival_step": 5}
{"prompt_len": 64, "gen_len": 16}
"#;
    fs::write(&tmp_path, content).expect("failed to write test trace file");

    let entries = parse_trace_file(tmp_path.to_str().unwrap()).expect("trace parsing failed");
    let _ = fs::remove_file(&tmp_path);

    assert_eq!(entries.len(), 3);
    assert_eq!(
        entries[0],
        TraceEntry {
            prompt_len: 128,
            gen_len: 64,
            arrival_step: 0,
        }
    );
    assert_eq!(
        entries[1],
        TraceEntry {
            prompt_len: 256,
            gen_len: 32,
            arrival_step: 5,
        }
    );
    // Missing arrival_step must default to 0
    assert_eq!(
        entries[2],
        TraceEntry {
            prompt_len: 64,
            gen_len: 16,
            arrival_step: 0,
        }
    );
}

#[test]
fn test_jsonl_trace_parsing_invalid() {
    let tmp_path = std::env::temp_dir().join("kanso_test_trace_invalid.jsonl");
    let content = "not valid json\n";
    fs::write(&tmp_path, content).unwrap();

    let res = parse_trace_file(tmp_path.to_str().unwrap());
    let _ = fs::remove_file(&tmp_path);
    assert!(res.is_err(), "Invalid JSON must return an EngineError");

    let non_existent = parse_trace_file("/non/existent/path/trace.jsonl");
    assert!(non_existent.is_err(), "Non-existent path must return an EngineError");
}

#[test]
fn test_batch_sweep_with_warmup_exclusion_and_metrics() {
    let model = Model::create_mock_tiny(16, 4, 2, 2, 50).unwrap();
    let mut engine = Engine::new(model, 4, 60, 4);

    let batch_size = 2;
    let prompt_len = 2;
    let gen_len = 3;

    let report: BenchmarkReport = run_synthetic_benchmark(
        &mut engine,
        batch_size,
        prompt_len,
        gen_len,
        /* warmup */ true,
    )
    .expect("synthetic benchmark failed");

    // Warmup must be excluded from measured totals
    assert_eq!(report.num_requests, 2);
    assert_eq!(report.total_tokens, 6);
    assert!(report.total_tokens_per_sec > 0.0);
    assert!(report.tokens_per_sec_per_request > 0.0);

    // Timing and parity metrics
    assert!(report.mean_ttft_ms >= 0.0);
    assert!(report.max_ttft_ms >= report.mean_ttft_ms);
    assert!(report.decode_ms_per_step > 0.0);
    assert_ne!(report.checksum, 0);

    // Repeat run should match deterministic checksum
    let model2 = Model::create_mock_tiny(16, 4, 2, 2, 50).unwrap();
    let mut engine2 = Engine::new(model2, 4, 60, 4);
    let report2 = run_synthetic_benchmark(&mut engine2, batch_size, prompt_len, gen_len, true).unwrap();
    assert_eq!(report.checksum, report2.checksum);
}

#[test]
fn test_trace_benchmark_execution() {
    let model = Model::create_mock_tiny(16, 4, 2, 2, 50).unwrap();
    let mut engine = Engine::new(model, 4, 60, 4);

    let trace = vec![
        TraceEntry {
            prompt_len: 2,
            gen_len: 3,
            arrival_step: 0,
        },
        TraceEntry {
            prompt_len: 2,
            gen_len: 2,
            arrival_step: 1,
        },
    ];

    let report = run_trace_benchmark(&mut engine, &trace, /* warmup */ true)
        .expect("trace benchmark failed");

    assert_eq!(report.num_requests, 2);
    assert_eq!(report.total_tokens, 5);
    assert!(report.total_tokens_per_sec > 0.0);
    assert!(report.mean_ttft_ms >= 0.0);
    assert!(report.decode_ms_per_step > 0.0);
    assert_ne!(report.checksum, 0);
}
