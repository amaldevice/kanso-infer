use std::collections::HashMap;
use std::time::Instant;
use serde::{Deserialize, Serialize};

use crate::engine::Engine;
use crate::error::EngineError;

/// Order-sensitive 64-bit FNV-1a hash over token sequence.
/// Offset basis: 0xcbf29ce484222325, Prime: 0x100000001b3.
pub fn fnv1a_64(tokens: &[u32]) -> u64 {
    const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;

    let mut hash = FNV_OFFSET_BASIS;
    for &tok in tokens {
        hash = (hash ^ (tok as u64)).wrapping_mul(FNV_PRIME);
    }
    hash
}

/// JSONL trace item describing a synthetic or replayed request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceEntry {
    pub prompt_len: usize,
    pub gen_len: usize,
    #[serde(default)]
    pub arrival_step: usize,
}

/// Parses a JSONL trace file where each line is a `TraceEntry` JSON object.
pub fn parse_trace_file(path: &str) -> Result<Vec<TraceEntry>, EngineError> {
    let content = std::fs::read_to_string(path)?;
    let mut entries = Vec::new();
    for (line_idx, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let entry: TraceEntry = serde_json::from_str(line).map_err(|e| {
            EngineError::Config(format!("Invalid JSONL trace at {}:{}: {}", path, line_idx + 1, e))
        })?;
        entries.push(entry);
    }
    Ok(entries)
}

#[derive(Debug, Clone)]
pub struct BenchmarkReport {
    pub total_tokens: usize,
    pub total_duration_secs: f64,
    pub num_requests: usize,
    pub total_tokens_per_sec: f64,
    pub tokens_per_sec_per_request: f64,
    pub avg_ms_per_step: f64,
    pub mean_ttft_ms: f64,
    pub max_ttft_ms: f64,
    pub itl_mean_ms: f64,
    pub itl_p99_ms: f64,
    pub prefill_tokens_per_sec: f64,
    pub decode_ms_per_step: f64,
    pub checksum: u64,
}

pub fn evaluate_run(
    total_tokens: usize,
    total_duration_secs: f64,
    num_requests: usize,
) -> BenchmarkReport {
    evaluate_run_with_timings(
        total_tokens,
        total_duration_secs,
        num_requests,
        &[],
        &[],
        0,
        0.0,
        0,
        0.0,
        0,
    )
}

pub fn evaluate_run_with_metrics(
    total_tokens: usize,
    total_duration_secs: f64,
    num_requests: usize,
    ttft_ms_records: &[f64],
    prefill_tokens: usize,
    prefill_duration_secs: f64,
    decode_steps: usize,
    decode_duration_secs: f64,
    checksum: u64,
) -> BenchmarkReport {
    evaluate_run_with_timings(
        total_tokens,
        total_duration_secs,
        num_requests,
        ttft_ms_records,
        &[],
        prefill_tokens,
        prefill_duration_secs,
        decode_steps,
        decode_duration_secs,
        checksum,
    )
}

pub fn evaluate_run_with_timings(
    total_tokens: usize,
    total_duration_secs: f64,
    num_requests: usize,
    ttft_ms_records: &[f64],
    itl_ms_records: &[f64],
    prefill_tokens: usize,
    prefill_duration_secs: f64,
    decode_steps: usize,
    decode_duration_secs: f64,
    checksum: u64,
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

    let mean_ttft_ms = if !ttft_ms_records.is_empty() {
        ttft_ms_records.iter().sum::<f64>() / (ttft_ms_records.len() as f64)
    } else {
        0.0
    };
    let max_ttft_ms = ttft_ms_records.iter().copied().fold(0.0f64, f64::max);

    let itl_mean_ms = if !itl_ms_records.is_empty() {
        itl_ms_records.iter().sum::<f64>() / (itl_ms_records.len() as f64)
    } else {
        0.0
    };

    let itl_p99_ms = if !itl_ms_records.is_empty() {
        let mut sorted = itl_ms_records.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let rank = ((sorted.len() as f64 * 0.99).ceil() as usize).max(1).min(sorted.len());
        sorted[rank - 1]
    } else {
        0.0
    };
    let prefill_tokens_per_sec = if prefill_duration_secs > 0.0 {
        (prefill_tokens as f64) / prefill_duration_secs
    } else {
        0.0
    };

    let decode_ms_per_step = if decode_steps > 0 {
        (decode_duration_secs * 1000.0) / (decode_steps as f64)
    } else {
        (safe_duration * 1000.0) / (total_tokens.max(1) as f64)
    };

    BenchmarkReport {
        total_tokens,
        total_duration_secs,
        num_requests,
        total_tokens_per_sec,
        tokens_per_sec_per_request,
        avg_ms_per_step: decode_ms_per_step,
        mean_ttft_ms,
        max_ttft_ms,
        itl_mean_ms,
        itl_p99_ms,
        prefill_tokens_per_sec,
        decode_ms_per_step,
        checksum,
    }
}

fn execute_engine_run(
    engine: &mut Engine,
    requests: &[TraceEntry],
) -> Result<BenchmarkReport, EngineError> {
    let mut order: Vec<usize> = (0..requests.len()).collect();
    order.sort_by_key(|&i| requests[i].arrival_step);

    let mut next_pending = 0;
    let mut step_idx = 0;

    let mut request_start_times: HashMap<u64, Instant> = HashMap::new();
    let mut first_token_times: HashMap<u64, Instant> = HashMap::new();
    let mut last_token_times: HashMap<u64, Instant> = HashMap::new();
    let mut ttft_records: Vec<f64> = Vec::new();
    let mut itl_records: Vec<f64> = Vec::new();
    let mut generated_tokens: Vec<u32> = Vec::new();

    let mut total_prompt_tokens = 0;
    for entry in requests {
        total_prompt_tokens += entry.prompt_len;
    }

    let mut prefill_duration = std::time::Duration::ZERO;
    let mut decode_duration = std::time::Duration::ZERO;
    let mut decode_steps = 0;

    let overall_start = Instant::now();
    loop {
        while next_pending < order.len() && requests[order[next_pending]].arrival_step <= step_idx {
            let req_idx = order[next_pending];
            next_pending += 1;
            let entry = &requests[req_idx];
            let prompt: Vec<u32> = (0..entry.prompt_len)
                .map(|p| ((req_idx * 10 + p) % 30) as u32)
                .collect();
            let req_id = engine.add_request(prompt, entry.gen_len);
            request_start_times.insert(req_id, Instant::now());
        }

        if next_pending >= order.len() && !engine.has_unfinished_requests() {
            break;
        }

        let is_prefilling = engine.sequences.iter().any(|s| {
            s.status == crate::scheduler::SequenceStatus::Running
                && s.prefilled_tokens < s.prompt_tokens.len()
        }) || engine.waiting_count() > 0;

        let step_start = Instant::now();
        let outputs = engine.step()?;
        let step_elapsed = step_start.elapsed();

        if is_prefilling {
            prefill_duration += step_elapsed;
        }

        if !outputs.is_empty() {
            decode_steps += 1;
            decode_duration += step_elapsed;
        }

        for out in outputs {
            generated_tokens.push(out.token_id);
            let now = Instant::now();
            if let std::collections::hash_map::Entry::Vacant(e) = first_token_times.entry(out.request_id) {
                e.insert(now);
                if let Some(start_t) = request_start_times.get(&out.request_id) {
                    let ttft_ms = now.duration_since(*start_t).as_secs_f64() * 1000.0;
                    ttft_records.push(ttft_ms);
                }
            } else if let Some(last_t) = last_token_times.get(&out.request_id) {
                let itl_ms = now.duration_since(*last_t).as_secs_f64() * 1000.0;
                itl_records.push(itl_ms);
            }
            last_token_times.insert(out.request_id, now);
        }

        step_idx += 1;
    }

    let overall_elapsed = overall_start.elapsed().as_secs_f64();
    let checksum = fnv1a_64(&generated_tokens);

    Ok(evaluate_run_with_timings(
        generated_tokens.len(),
        overall_elapsed,
        requests.len(),
        &ttft_records,
        &itl_records,
        total_prompt_tokens,
        prefill_duration.as_secs_f64(),
        decode_steps,
        decode_duration.as_secs_f64(),
        checksum,
    ))
}

pub fn run_benchmark_workload(
    engine: &mut Engine,
    requests: &[TraceEntry],
    warmup: bool,
) -> Result<BenchmarkReport, EngineError> {
    if warmup {
        let _ = execute_engine_run(engine, requests)?;
        engine.sequences.retain(|s| !s.is_finished());
    }
    let report = execute_engine_run(engine, requests)?;
    engine.sequences.retain(|s| !s.is_finished());
    Ok(report)
}

pub fn run_synthetic_benchmark(
    engine: &mut Engine,
    batch_size: usize,
    prompt_len: usize,
    gen_len: usize,
    warmup: bool,
) -> Result<BenchmarkReport, EngineError> {
    let requests: Vec<TraceEntry> = (0..batch_size)
        .map(|_| TraceEntry {
            prompt_len,
            gen_len,
            arrival_step: 0,
        })
        .collect();
    run_benchmark_workload(engine, &requests, warmup)
}

pub fn run_trace_benchmark(
    engine: &mut Engine,
    trace: &[TraceEntry],
    warmup: bool,
) -> Result<BenchmarkReport, EngineError> {
    run_benchmark_workload(engine, trace, warmup)
}
pub fn run_synthetic_sweep(
    engine: &mut Engine,
    batch_size: usize,
    prompt_len: usize,
    gen_len: usize,
) -> Result<BenchmarkReport, EngineError> {
    run_synthetic_benchmark(engine, batch_size, prompt_len, gen_len, false)
}

pub fn create_engine_for_model_profile(
    model_type: &str,
    backend: &str,
    num_requests: usize,
    max_tokens_per_seq: usize,
    block_size: usize,
    max_running: usize,
) -> Result<Engine, EngineError> {
    let config = crate::config::ModelConfig::from_model_type(model_type)?;
    let blocks_per_seq = (max_tokens_per_seq + block_size - 1) / block_size + 1;
    let total_blocks = (blocks_per_seq * num_requests.max(1)).max(128);
    let mut model = crate::model::Model::create_mock_from_config(&config)?;
    model.set_backend(backend);
    let engine = Engine::new(model, max_running, total_blocks, block_size);
    Ok(engine)
}
