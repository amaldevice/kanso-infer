use kanso_infer::bench::BenchmarkReport;
use kanso_infer::speculative::VerificationResult;
use kanso_infer::speculative::profiler::{SpeculativeProfiler, SpeculativeReport};
use std::time::Duration;

#[test]
fn test_speculative_profiler_initial_state() {
    let profiler = SpeculativeProfiler::new();
    assert_eq!(profiler.total_proposed, 0);
    assert_eq!(profiler.total_accepted, 0);
    assert_eq!(profiler.total_target_tokens, 0);
    assert_eq!(profiler.total_steps, 0);
    assert_eq!(profiler.rollback_events, 0);
    assert_eq!(profiler.total_duration, Duration::ZERO);
    assert_eq!(profiler.acceptance_rate(), 0.0);
    assert_eq!(profiler.alpha(), 0.0);
    assert_eq!(profiler.average_accepted_tokens_per_step(), 0.0);
    assert_eq!(profiler.mean_accepted_per_step(), 0.0);
    assert_eq!(profiler.tau_bar(), 0.0);
    assert_eq!(profiler.speedup_ratio(Duration::from_millis(100)), 0.0);
}

#[test]
fn test_speculative_profiler_step_recording_and_metrics() {
    let mut profiler = SpeculativeProfiler::new();

    // Step 1: 4 proposed, 4 accepted, no rollback
    profiler.record_step(4, 4, Duration::from_millis(10), false);
    // Step 2: 4 proposed, 2 accepted, rollback occurred
    profiler.record_step(4, 2, Duration::from_millis(12), true);
    // Step 3: 4 proposed, 0 accepted, rollback occurred
    profiler.record_step(4, 0, Duration::from_millis(8), true);

    assert_eq!(profiler.total_proposed, 12);
    assert_eq!(profiler.total_accepted, 6);
    assert_eq!(profiler.total_target_tokens, 3);
    assert_eq!(profiler.total_steps, 3);
    assert_eq!(profiler.rollback_events, 2);
    assert_eq!(profiler.total_duration, Duration::from_millis(30));

    // alpha = 6 / 12 = 0.5
    let expected_alpha = 6.0 / 12.0;
    assert!((profiler.acceptance_rate() - expected_alpha).abs() < 1e-6);
    assert!((profiler.alpha() - expected_alpha).abs() < 1e-6);

    // tau_bar = (6 + 3) / 3 = 3.0
    let expected_tau = (6.0 + 3.0) / 3.0;
    assert!((profiler.average_accepted_tokens_per_step() - expected_tau).abs() < 1e-6);
    assert!((profiler.mean_accepted_per_step() - expected_tau).abs() < 1e-6);
    assert!((profiler.tau_bar() - expected_tau).abs() < 1e-6);

    // Speedup ratio S = T_ar / T_spec = 60ms / 30ms = 2.0
    let ar_duration = Duration::from_millis(60);
    let speedup = profiler.speedup_ratio(ar_duration);
    assert!((speedup - 2.0).abs() < 1e-6);

    // Also support speedup with f64 seconds
    let speedup_secs = profiler.speedup_ratio_secs(0.060);
    assert!((speedup_secs - 2.0).abs() < 1e-6);
}

#[test]
fn test_speculative_profiler_record_verification_result() {
    let mut profiler = SpeculativeProfiler::new();

    // Perfect verification result
    let v_perfect = VerificationResult {
        accepted_tokens: vec![101, 102, 103],
        emitted_token: 104,
        num_accepted: 3,
        num_rejected: 0,
        committed_tokens: vec![101, 102, 103, 104],
        rolled_back_blocks: vec![],
    };
    profiler.record_verification(3, &v_perfect, Duration::from_millis(15));

    assert_eq!(profiler.total_proposed, 3);
    assert_eq!(profiler.total_accepted, 3);
    assert_eq!(profiler.rollback_events, 0);

    // Partial rejection with rollback
    let v_partial = VerificationResult {
        accepted_tokens: vec![201],
        emitted_token: 205,
        num_accepted: 1,
        num_rejected: 2,
        committed_tokens: vec![201, 205],
        rolled_back_blocks: vec![7],
    };
    profiler.record_verification(3, &v_partial, Duration::from_millis(15));

    assert_eq!(profiler.total_proposed, 6);
    assert_eq!(profiler.total_accepted, 4);
    assert_eq!(profiler.total_target_tokens, 2);
    assert_eq!(profiler.rollback_events, 1);
    assert_eq!(profiler.total_steps, 2);
    assert!((profiler.acceptance_rate() - (4.0 / 6.0)).abs() < 1e-6);
    assert!((profiler.average_accepted_tokens_per_step() - 3.0).abs() < 1e-6);
    assert!((profiler.mean_accepted_per_step() - 3.0).abs() < 1e-6);
}

#[test]
fn test_speculative_profiler_json_export_and_report() {
    let mut profiler = SpeculativeProfiler::new();
    profiler.record_step(5, 3, Duration::from_millis(25), true);
    profiler.record_step(5, 5, Duration::from_millis(25), false);

    let report: SpeculativeReport = profiler.report();
    assert_eq!(report.total_proposed, 10);
    assert_eq!(report.total_accepted, 8);
    assert_eq!(report.total_target_tokens, 2);
    assert_eq!(report.total_steps, 2);
    assert_eq!(report.rollback_events, 1);
    assert!((report.acceptance_rate - 0.8).abs() < 1e-6);
    assert!((report.mean_accepted_per_step - 5.0).abs() < 1e-6);
    assert!((report.average_accepted_tokens_per_step() - 5.0).abs() < 1e-6);
    assert!((report.total_duration_secs - 0.050).abs() < 1e-6);

    // JSON export
    let json_str = profiler
        .to_json()
        .expect("failed to serialize profiler to JSON");
    assert!(json_str.contains("\"total_proposed\":10"));
    assert!(json_str.contains("\"total_accepted\":8"));
    assert!(json_str.contains("\"total_target_tokens\":2"));
    assert!(json_str.contains("\"rollback_events\":1"));
    assert!(json_str.contains("\"acceptance_rate\":0.8"));
    assert!(json_str.contains("\"mean_accepted_per_step\":5.0"));

    // Deserialization round-trip
    let deserialized: SpeculativeReport =
        serde_json::from_str(&json_str).expect("failed to deserialize SpeculativeReport");
    assert_eq!(deserialized.total_proposed, 10);
    assert_eq!(deserialized.total_accepted, 8);
    assert_eq!(deserialized.total_target_tokens, 2);
    assert_eq!(deserialized.rollback_events, 1);
    // JSON with speedup
    let json_with_speedup = profiler
        .to_json_with_speedup(Duration::from_millis(100))
        .expect("failed to serialize with speedup");
    assert!(json_with_speedup.contains("\"speedup_ratio\":2.0"));

    // Formatted report string contains key metrics
    let formatted = profiler.format_report(Some(Duration::from_millis(100)));
    assert!(formatted.contains("N_prop"));
    assert!(formatted.contains("N_acc"));
    assert!(formatted.contains("alpha"));
    assert!(formatted.contains("tau_bar"));
    assert!(formatted.contains("Speedup Ratio"));
}

#[test]
fn test_speculative_profiler_simulated_acceptance_rejection_sequences() {
    // 1. Perfect sequence: K=4, 10 steps, all accepted
    let mut perfect_profiler = SpeculativeProfiler::new();
    for _ in 0..10 {
        perfect_profiler.record_step(4, 4, Duration::from_millis(5), false);
    }
    assert_eq!(perfect_profiler.total_proposed, 40);
    assert_eq!(perfect_profiler.total_accepted, 40);
    assert_eq!(perfect_profiler.total_target_tokens, 10);
    assert_eq!(perfect_profiler.rollback_events, 0);
    assert!((perfect_profiler.acceptance_rate() - 1.0).abs() < 1e-6);
    assert!((perfect_profiler.average_accepted_tokens_per_step() - 5.0).abs() < 1e-6);
    assert!((perfect_profiler.mean_accepted_per_step() - 5.0).abs() < 1e-6);

    // 2. Complete rejection sequence: K=4, 10 steps, 0 accepted
    let mut reject_profiler = SpeculativeProfiler::new();
    for _ in 0..10 {
        reject_profiler.record_step(4, 0, Duration::from_millis(5), true);
    }
    assert_eq!(reject_profiler.total_proposed, 40);
    assert_eq!(reject_profiler.total_accepted, 0);
    assert_eq!(reject_profiler.total_target_tokens, 10);
    assert_eq!(reject_profiler.rollback_events, 10);
    assert_eq!(reject_profiler.acceptance_rate(), 0.0);
    assert!((reject_profiler.average_accepted_tokens_per_step() - 1.0).abs() < 1e-6);
    assert!((reject_profiler.mean_accepted_per_step() - 1.0).abs() < 1e-6);

    // 3. Reset functionality
    perfect_profiler.reset();
    assert_eq!(perfect_profiler.total_proposed, 0);
    assert_eq!(perfect_profiler.total_target_tokens, 0);
    assert_eq!(perfect_profiler.total_steps, 0);
}

#[test]
fn test_benchmark_report_speculative_integration() {
    let mut profiler = SpeculativeProfiler::new();
    profiler.record_step(4, 3, Duration::from_millis(20), true);

    let spec_report = profiler.report_with_speedup(Duration::from_millis(40));

    let bench_report = BenchmarkReport {
        total_tokens: 100,
        total_duration_secs: 0.5,
        num_requests: 1,
        total_tokens_per_sec: 200.0,
        tokens_per_sec_per_request: 200.0,
        avg_ms_per_step: 5.0,
        mean_ttft_ms: 10.0,
        max_ttft_ms: 10.0,
        ttft_p50_ms: 10.0,
        ttft_p90_ms: 10.0,
        itl_mean_ms: 5.0,
        itl_p50_ms: 5.0,
        itl_p90_ms: 5.0,
        itl_p99_ms: 5.0,
        prefill_tokens_per_sec: 500.0,
        decode_ms_per_step: 4.0,
        checksum: 0x12345678,
        prefix_hit_rate: 0.0,
        bytes_read_per_step: 0,
        peak_bandwidth_gb_s: 0.0,
        peak_bandwidth_source: String::new(),
        speculative: None,
    }
    .with_speculative(spec_report);

    assert!(bench_report.speculative.is_some());
    let spec = bench_report.speculative.as_ref().unwrap();
    assert_eq!(spec.total_proposed, 4);
    assert_eq!(spec.total_accepted, 3);
    assert_eq!(spec.rollback_events, 1);
    assert_eq!(spec.speedup_ratio, Some(2.0));

    let json = bench_report
        .to_json()
        .expect("failed to serialize BenchmarkReport");
    assert!(json.contains("\"speculative\":"));
    assert!(json.contains("\"speedup_ratio\":2.0"));
}

#[test]
fn test_speculative_engine_profiler_automatic_tracking() {
    use kanso_infer::model::Model;
    use kanso_infer::speculative::SpeculativeEngine;

    let target_model = Model::create_mock_tiny(16, 4, 2, 2, 50).unwrap();
    let draft_model = Model::create_mock_tiny(16, 4, 2, 2, 50).unwrap();
    let mut spec_engine = SpeculativeEngine::new(
        target_model,
        Some(draft_model),
        1,
        30,
        4,
        3, // gamma = 3
    );

    let prompt = vec![1, 2, 3];
    spec_engine.add_request(prompt, 6);

    while spec_engine.has_unfinished_requests() {
        let _ = spec_engine.step().unwrap();
    }

    // Verify the engine's built-in profiler recorded steps and metrics automatically
    let profiler = &spec_engine.profiler;
    assert!(
        profiler.total_proposed > 0,
        "Proposed tokens must be recorded"
    );
    assert!(profiler.total_steps > 0, "Steps must be recorded");
    assert!(
        profiler.total_duration > Duration::ZERO,
        "Duration must be positive"
    );
    assert!(profiler.acceptance_rate() >= 0.0 && profiler.acceptance_rate() <= 1.0);
    assert!(profiler.mean_accepted_per_step() >= 0.0);

    // Check that report generation works
    let report = profiler.report();
    assert_eq!(report.total_proposed, profiler.total_proposed);
    assert_eq!(report.total_accepted, profiler.total_accepted);
    assert_eq!(report.total_steps, profiler.total_steps);
}
