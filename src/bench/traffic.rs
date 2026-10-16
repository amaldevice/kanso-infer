use crate::bench::{BenchmarkReport, TraceEntry, run_benchmark_workload};
use crate::engine::Engine;
use crate::error::EngineError;
use rand::{Rng, SeedableRng};

/// Standard concurrency sweep arrival rates (requests per second): {2, 5, 10, 20, 40}.
pub const CONCURRENCY_SWEEP_RATES: [f64; 5] = [2.0, 5.0, 10.0, 20.0, 40.0];

/// Alias for concurrency sweep rate parameters (lambda).
pub const CONCURRENCY_SWEEP_LAMBDAS: [f64; 5] = CONCURRENCY_SWEEP_RATES;

/// Generates request arrival schedules with rate parameter `rate_lambda` (requests/sec)
/// using exponential random sampling following a homogeneous Poisson process:
/// Δt = -ln(U) / λ, where U ~ Uniform(0, 1].
///
/// Returns cumulative arrival timestamps in seconds starting from t = 0.
pub fn generate_poisson_schedule(num_requests: usize, rate_lambda: f64, seed: u64) -> Vec<f64> {
    if num_requests == 0 || rate_lambda <= 0.0 {
        return Vec::new();
    }
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut schedule = Vec::with_capacity(num_requests);
    let mut current_time = 0.0;

    for _ in 0..num_requests {
        let u = (1.0 - rng.random::<f64>()).clamp(f64::MIN_POSITIVE, 1.0);
        let delta_t = -u.ln() / rate_lambda;
        current_time += delta_t;
        schedule.push(current_time);
    }

    schedule
}

/// Computes nearest-rank percentiles (p50, p90, p99) for a slice of values.
/// Returns (p50, p90, p99).
pub fn calculate_percentiles(values: &[f64]) -> (f64, f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();

    let p50_rank = ((n as f64 * 0.50).ceil() as usize).max(1).min(n);
    let p90_rank = ((n as f64 * 0.90).ceil() as usize).max(1).min(n);
    let p99_rank = ((n as f64 * 0.99).ceil() as usize).max(1).min(n);

    (
        sorted[p50_rank - 1],
        sorted[p90_rank - 1],
        sorted[p99_rank - 1],
    )
}

/// Latency percentiles summary (p50, p90, p99, and mean).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct LatencyPercentiles {
    pub p50: f64,
    pub p90: f64,
    pub p99: f64,
    pub mean: f64,
}

impl LatencyPercentiles {
    pub fn from_values(values: &[f64]) -> Self {
        let (p50, p90, p99) = calculate_percentiles(values);
        let mean = if values.is_empty() {
            0.0
        } else {
            values.iter().sum::<f64>() / (values.len() as f64)
        };
        Self {
            p50,
            p90,
            p99,
            mean,
        }
    }
}

/// Converts continuous Poisson arrival timestamps into discrete `TraceEntry`s
/// based on an assumed time duration per engine step.
pub fn schedule_to_trace(
    schedule: &[f64],
    prompt_len: usize,
    gen_len: usize,
    step_duration_secs: f64,
) -> Vec<TraceEntry> {
    let step_duration = if step_duration_secs > 0.0 {
        step_duration_secs
    } else {
        0.001
    };
    schedule
        .iter()
        .map(|&t| TraceEntry {
            prompt_len,
            gen_len,
            arrival_step: (t / step_duration).floor() as usize,
            prompt_tokens: None,
        })
        .collect()
}

/// Result of evaluating one concurrency sweep point (lambda).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ConcurrencySweepPoint {
    pub lambda: f64,
    pub num_requests: usize,
    pub total_tokens: usize,
    pub throughput_tokens_per_sec: f64,
    pub mean_ttft_ms: f64,
    pub max_ttft_ms: f64,
    pub itl_mean_ms: f64,
    pub itl_p99_ms: f64,
    pub duration_secs: f64,
}

/// Runs a Poisson workload for a given arrival rate (lambda).
pub fn run_poisson_workload(
    engine: &mut Engine,
    rate_lambda: f64,
    num_requests: usize,
    prompt_len: usize,
    gen_len: usize,
    seed: u64,
    step_duration_secs: f64,
) -> Result<BenchmarkReport, EngineError> {
    let schedule = generate_poisson_schedule(num_requests, rate_lambda, seed);
    let trace = schedule_to_trace(&schedule, prompt_len, gen_len, step_duration_secs);
    run_benchmark_workload(engine, &trace, false)
}

/// Runs a concurrency sweep across multiple lambda arrival rates on the given engine.
pub fn run_poisson_concurrency_sweep(
    engine: &mut Engine,
    lambdas: &[f64],
    requests_per_rate: usize,
    prompt_len: usize,
    gen_len: usize,
    seed: u64,
) -> Result<Vec<ConcurrencySweepPoint>, EngineError> {
    let mut points = Vec::new();
    let mut current_seed = seed;

    for &lambda in lambdas {
        let report = run_poisson_workload(
            engine,
            lambda,
            requests_per_rate,
            prompt_len,
            gen_len,
            current_seed,
            0.005,
        )?;
        current_seed = current_seed.wrapping_add(1);

        points.push(ConcurrencySweepPoint {
            lambda,
            num_requests: requests_per_rate,
            total_tokens: report.total_tokens,
            throughput_tokens_per_sec: report.total_tokens_per_sec,
            mean_ttft_ms: report.mean_ttft_ms,
            max_ttft_ms: report.max_ttft_ms,
            itl_mean_ms: report.itl_mean_ms,
            itl_p99_ms: report.itl_p99_ms,
            duration_secs: report.total_duration_secs,
        });
    }

    Ok(points)
}
