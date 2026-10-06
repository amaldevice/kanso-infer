use std::time::Instant;
use crate::engine::Engine;
use crate::error::EngineError;

#[derive(Debug, Clone)]
pub struct BenchmarkReport {
    pub total_tokens: usize,
    pub total_duration_secs: f64,
    pub num_requests: usize,
    pub total_tokens_per_sec: f64,
    pub tokens_per_sec_per_request: f64,
    pub avg_ms_per_step: f64,
}

pub fn evaluate_run(
    total_tokens: usize,
    total_duration_secs: f64,
    num_requests: usize,
) -> BenchmarkReport {
    let safe_duration = if total_duration_secs > 0.0 {
        total_duration_secs
    } else {
        1e-6
    };

    let total_tokens_per_sec = (total_tokens as f64) / safe_duration;
    let tokens_per_sec_per_request = if num_requests > 0 {
        total_tokens_per_sec / (num_requests as f64)
    } else {
        0.0
    };

    BenchmarkReport {
        total_tokens,
        total_duration_secs,
        num_requests,
        total_tokens_per_sec,
        tokens_per_sec_per_request,
        avg_ms_per_step: (safe_duration * 1000.0) / (total_tokens.max(1) as f64),
    }
}

pub fn run_synthetic_sweep(
    engine: &mut Engine,
    batch_size: usize,
    prompt_len: usize,
    gen_len: usize,
) -> Result<BenchmarkReport, EngineError> {
    for i in 0..batch_size {
        let prompt: Vec<u32> = (0..prompt_len).map(|p| ((i * 10 + p) % 30) as u32).collect();
        engine.add_request(prompt, gen_len);
    }

    let start = Instant::now();
    let mut total_generated = 0;

    while engine.has_unfinished_requests() {
        let outputs = engine.step()?;
        total_generated += outputs.len();
    }

    let elapsed = start.elapsed().as_secs_f64();
    Ok(evaluate_run(batch_size * gen_len, elapsed, batch_size))
}
