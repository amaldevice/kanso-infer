use std::sync::Arc;
use parking_lot::Mutex;
use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use tower::ServiceExt;
use kanso_infer::bench::traffic::{
    calculate_percentiles, generate_poisson_schedule, run_poisson_concurrency_sweep,
    CONCURRENCY_SWEEP_RATES,
};
use kanso_infer::engine::Engine;
use kanso_infer::model::Model;
use kanso_infer::server::create_app_with_queue_limit;

#[test]
fn test_poisson_inter_arrival_times_convergence() {
    let num_requests = 1000;
    let rate_lambda = 10.0;
    let seed = 42;

    let schedule = generate_poisson_schedule(num_requests, rate_lambda, seed);
    assert_eq!(schedule.len(), num_requests);

    // Verify monotonic arrival timestamps
    for i in 1..schedule.len() {
        assert!(
            schedule[i] > schedule[i - 1],
            "Timestamps must be monotonically increasing: {} <= {}",
            schedule[i],
            schedule[i - 1]
        );
    }

    // Inter-arrival times: delta_t_0 = schedule[0], delta_t_i = schedule[i] - schedule[i - 1]
    let mut inter_arrivals = Vec::with_capacity(num_requests);
    inter_arrivals.push(schedule[0]);
    for i in 1..schedule.len() {
        inter_arrivals.push(schedule[i] - schedule[i - 1]);
    }

    let mean_inter_arrival = inter_arrivals.iter().sum::<f64>() / (num_requests as f64);
    let expected_mean = 1.0 / rate_lambda;
    let relative_error = (mean_inter_arrival - expected_mean).abs() / expected_mean;

    // Must converge to 1/lambda within 5% statistical tolerance for N=1000
    assert!(
        relative_error < 0.05,
        "Mean inter-arrival time {:.4} did not converge to expected {:.4} (relative error: {:.2}%)",
        mean_inter_arrival,
        expected_mean,
        relative_error * 100.0
    );

    // Test across concurrency sweep lambdas: {2, 5, 20, 40}
    for &rate in &[2.0, 5.0, 20.0, 40.0] {
        let sched = generate_poisson_schedule(1000, rate, seed);
        let mean = sched[sched.len() - 1] / 1000.0;
        let exp = 1.0 / rate;
        let err = (mean - exp).abs() / exp;
        assert!(
            err < 0.05,
            "Rate {} mean {:.4} did not converge to expected {:.4} (err: {:.2}%)",
            rate,
            mean,
            exp,
            err * 100.0
        );
    }
}

#[test]
fn test_poisson_schedule_reproducibility_and_edges() {
    // Deterministic reproducibility with same seed
    let s1 = generate_poisson_schedule(50, 5.0, 12345);
    let s2 = generate_poisson_schedule(50, 5.0, 12345);
    assert_eq!(s1, s2);

    // Different seed produces different schedule
    let s3 = generate_poisson_schedule(50, 5.0, 54321);
    assert_ne!(s1, s3);

    // Edge cases
    assert!(generate_poisson_schedule(0, 10.0, 42).is_empty());
    assert!(generate_poisson_schedule(50, 0.0, 42).is_empty());
    assert!(generate_poisson_schedule(50, -5.0, 42).is_empty());
}

#[test]
fn test_nearest_rank_percentiles_calculation() {
    // Exact nearest rank on 1..=100
    let values: Vec<f64> = (1..=100).map(|v| v as f64).collect();
    let (p50, p90, p99) = calculate_percentiles(&values);
    assert_eq!(p50, 50.0);
    assert_eq!(p90, 90.0);
    assert_eq!(p99, 99.0);

    // Order independence (reversed input)
    let mut reversed = values.clone();
    reversed.reverse();
    let (rev_p50, rev_p90, rev_p99) = calculate_percentiles(&reversed);
    assert_eq!(rev_p50, 50.0);
    assert_eq!(rev_p90, 90.0);
    assert_eq!(rev_p99, 99.0);

    // Edge cases
    assert_eq!(calculate_percentiles(&[]), (0.0, 0.0, 0.0));
    assert_eq!(calculate_percentiles(&[42.0]), (42.0, 42.0, 42.0));

    // Small slice: [10.0, 20.0, 30.0, 40.0]
    // N=4: p50 rank = ceil(4*0.50) = 2 -> 20.0
    //      p90 rank = ceil(4*0.90) = 4 -> 40.0
    //      p99 rank = ceil(4*0.99) = 4 -> 40.0
    let small = [10.0, 20.0, 30.0, 40.0];
    let (sp50, sp90, sp99) = calculate_percentiles(&small);
    assert_eq!(sp50, 20.0);
    assert_eq!(sp90, 40.0);
    assert_eq!(sp99, 40.0);
}
#[test]
fn test_ttft_and_itl_latency_percentiles() {
    // Simulated TTFT latency samples (ms)
    let ttft_samples = vec![12.5, 14.2, 15.0, 15.8, 16.3, 18.0, 22.4, 25.1, 35.0, 48.2];
    let (ttft_p50, ttft_p90, ttft_p99) = calculate_percentiles(&ttft_samples);
    assert!(ttft_p50 <= ttft_p90);
    assert!(ttft_p90 <= ttft_p99);
    assert_eq!(ttft_p50, 16.3);
    assert_eq!(ttft_p90, 35.0);
    assert_eq!(ttft_p99, 48.2);

    // Simulated ITL latency samples (ms)
    let itl_samples = vec![4.1, 4.2, 4.3, 4.5, 4.8, 5.0, 5.2, 5.5, 6.8, 9.4];
    let (itl_p50, itl_p90, itl_p99) = calculate_percentiles(&itl_samples);
    assert!(itl_p50 <= itl_p90);
    assert!(itl_p90 <= itl_p99);
    assert_eq!(itl_p50, 4.8);
    assert_eq!(itl_p90, 6.8);
    assert_eq!(itl_p99, 9.4);

    // Test direct export from kanso_infer::bench
    let sched = kanso_infer::bench::generate_poisson_schedule(10, 5.0, 99);
    assert_eq!(sched.len(), 10);
    let p = kanso_infer::bench::calculate_percentiles(&[1.0, 2.0, 3.0]);
    assert_eq!(p.0, 2.0);
}


#[test]
fn test_concurrency_sweep_rates_and_execution() {
    assert_eq!(CONCURRENCY_SWEEP_RATES, [2.0, 5.0, 10.0, 20.0, 40.0]);

    let model = Model::create_mock_tiny(16, 4, 2, 2, 50).unwrap();
    let mut engine = Engine::new(model, 4, 40, 4);

    let sweep_results = run_poisson_concurrency_sweep(
        &mut engine,
        &CONCURRENCY_SWEEP_RATES,
        /* requests_per_rate */ 4,
        /* prompt_len */ 2,
        /* gen_len */ 3,
        /* seed */ 42,
    )
    .expect("sweep execution failed");

    assert_eq!(sweep_results.len(), 5);
    for (i, point) in sweep_results.iter().enumerate() {
        assert_eq!(point.lambda, CONCURRENCY_SWEEP_RATES[i]);
        assert_eq!(point.num_requests, 4);
        assert!(point.total_tokens > 0);
        assert!(point.throughput_tokens_per_sec > 0.0);
        assert!(point.mean_ttft_ms >= 0.0);
        assert!(point.itl_mean_ms >= 0.0);
    }
}

#[tokio::test]
async fn test_graceful_http_429_backpressure_under_burst() {
    let model = Model::create_mock_tiny(16, 4, 2, 2, 50).unwrap();
    // Engine with max_running = 2 and small waiting queue limit of 1
    let engine = Engine::new(model, 2, 20, 4);
    let shared_engine = Arc::new(Mutex::new(engine));

    // Server with queue limit = 1: max 1 waiting request allowed, subsequent get 429
    let app = create_app_with_queue_limit(shared_engine, Some(1));

    let make_request = || {
        let payload = serde_json::to_vec(&serde_json::json!({
            "model": "smollm2-360m",
            "messages": [{"role": "user", "content": "burst test"}],
            "max_tokens": 4,
            "stream": false
        }))
        .unwrap();

        Request::builder()
            .uri("/v1/chat/completions")
            .method("POST")
            .header("content-type", "application/json")
            .body(Body::from(payload))
            .unwrap()
    };

    // Send a concurrent burst of 8 requests to overwhelm the engine queue
    let mut handles = Vec::new();
    for _ in 0..8 {
        let app_clone = app.clone();
        let req = make_request();
        handles.push(tokio::spawn(async move {
            app_clone.oneshot(req).await.unwrap()
        }));
    }

    let mut ok_count = 0;
    let mut rate_limited_count = 0;

    for handle in handles {
        let resp = handle.await.unwrap();
        match resp.status() {
            StatusCode::OK => ok_count += 1,
            StatusCode::TOO_MANY_REQUESTS => rate_limited_count += 1,
            other => panic!("Unexpected status code: {}", other),
        }
    }

    // At least one request should succeed, and at least one must be rejected with 429 backpressure
    assert!(ok_count > 0, "Expected at least one request to succeed");
    assert!(
        rate_limited_count > 0,
        "Expected HTTP 429 backpressure when arrival rate exceeds capacity, but got 0 (ok: {})",
        ok_count
    );
}
