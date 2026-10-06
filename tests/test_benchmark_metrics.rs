#[test]
fn test_bench_metric_calculation() {
    let report = kanso_infer::bench::evaluate_run(
        /* total_tokens */ 128,
        /* total_duration_secs */ 2.0,
        /* num_requests */ 4,
    );

    assert_eq!(report.total_tokens, 128);
    assert_eq!(report.num_requests, 4);
    assert!((report.total_tokens_per_sec - 64.0).abs() < 1e-4);
    assert!((report.tokens_per_sec_per_request - 16.0).abs() < 1e-4);
}

#[test]
fn test_bench_itl_metrics_calculation() {
    let ttfts = vec![10.0, 15.0];
    let itls = vec![5.0, 10.0, 15.0, 20.0];
    let report = kanso_infer::bench::evaluate_run_with_timings(
        100,
        1.0,
        2,
        &ttfts,
        &itls,
        10,
        0.1,
        20,
        0.5,
        12345,
    );

    assert_eq!(report.itl_mean_ms, 12.5);
    assert_eq!(report.itl_p99_ms, 20.0);
}
#[test]
fn test_run_synthetic_sweep_execution() {
    let model = kanso_infer::model::Model::create_mock_tiny(16, 4, 2, 2, 50).unwrap();
    let mut engine = kanso_infer::engine::Engine::new(model, 4, 30, 4);

    let report = kanso_infer::bench::run_synthetic_sweep(
        &mut engine,
        /* batch_size */ 2,
        /* prompt_len */ 2,
        /* gen_len */ 3,
    ).expect("sweep failed");

    assert_eq!(report.num_requests, 2);
    assert_eq!(report.total_tokens, 6);
    assert!(report.total_tokens_per_sec > 0.0);
    assert!(report.itl_mean_ms >= 0.0);
    assert!(report.itl_p99_ms >= report.itl_mean_ms);
}
