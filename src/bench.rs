use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

use crate::engine::Engine;
use crate::error::EngineError;
pub mod traffic;
pub use traffic::*;
pub mod sharegpt;
pub use sharegpt::*;
pub mod perplexity;
pub use perplexity::*;
pub mod probes;
pub use probes::*;
pub mod gate;
pub use gate::*;

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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_tokens: Option<Vec<u32>>,
}

impl TraceEntry {
    pub fn new(prompt_len: usize, gen_len: usize, arrival_step: usize) -> Self {
        Self {
            prompt_len,
            gen_len,
            arrival_step,
            prompt_tokens: None,
        }
    }

    pub fn with_tokens(
        prompt_len: usize,
        gen_len: usize,
        arrival_step: usize,
        tokens: Vec<u32>,
    ) -> Self {
        Self {
            prompt_len,
            gen_len,
            arrival_step,
            prompt_tokens: Some(tokens),
        }
    }
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
            EngineError::Config(format!(
                "Invalid JSONL trace at {}:{}: {}",
                path,
                line_idx + 1,
                e
            ))
        })?;
        entries.push(entry);
    }
    Ok(entries)
}

/// Whether the report's byte telemetry was actually weighed (#91).
///
/// A report whose `bytes_read_per_step` is zero was built from numbers alone:
/// nothing was weighed, so the zero means "not measured yet", not "small".
/// Every report carries this marker, and the gate refuses to pass an unweighed one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TelemetryStatus {
    /// Bytes were weighed from the model that ran.
    Measured,
    /// No weighing happened; the byte fields carry no information.
    /// Also the reading of a report written before the marker existed.
    #[default]
    Unmeasured,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkReport {
    pub total_tokens: usize,
    pub total_duration_secs: f64,
    pub num_requests: usize,
    pub total_tokens_per_sec: f64,
    pub tokens_per_sec_per_request: f64,
    pub avg_ms_per_step: f64,
    pub mean_ttft_ms: f64,
    pub max_ttft_ms: f64,
    pub ttft_p50_ms: f64,
    pub ttft_p90_ms: f64,
    pub itl_mean_ms: f64,
    pub itl_p50_ms: f64,
    pub itl_p90_ms: f64,
    pub itl_p99_ms: f64,
    pub prefill_tokens_per_sec: f64,
    pub decode_ms_per_step: f64,
    /// Per-actual-token decode latency in ms (decode duration / total tokens, #62 F5).
    #[serde(default)]
    pub decode_ms_per_token: f64,
    pub checksum: u64,
    pub prefix_hit_rate: f64,
    /// Resident weight bytes + KV bytes for the run's sequence context, weighed
    /// from the model that actually ran (#53). Zero = predates the field.
    #[serde(default)]
    pub bytes_read_per_step: usize,
    /// Denominator behind the bus-util figure: a measured number, never a literal.
    #[serde(default)]
    pub peak_bandwidth_gb_s: f64,
    /// How `peak_bandwidth_gb_s` was obtained (e.g. the STREAM-triad microbench).
    #[serde(default)]
    pub peak_bandwidth_source: String,
    /// Whether the byte fields above came from a weighing (#91).
    #[serde(default)]
    pub telemetry: TelemetryStatus,
    /// Peak resident set size of the measuring process, straight from
    /// `hardware::get_peak_rss_mb()` (#100). `None` = the report predates the
    /// field, so there is no measurement to judge — a zero would read as "nothing
    /// resident" and wrongly satisfy a `max` budget.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub peak_rss_mb: Option<f64>,
    /// Perplexity on the bundled WikiText-2 slice, attached only by a run that
    /// asked for it (`--eval-ppl`). `None` = never evaluated (#100).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub perplexity_wikitext2: Option<f64>,
    /// Context length this report ran, so a ctx sweep can pair two reports (#100).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_len: Option<usize>,
    /// The per-kernel / functional probes that ran alongside this workload.
    /// Absent when no probe was asked for (#100).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub probes: Option<KernelProbes>,
    /// Which KV pool served the run, attached only by a run that really used
    /// an engine, straight from the pool itself (#111). `None` = no engine ran
    /// (or the report predates the field), so nothing was measured.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv_type: Option<crate::kv_cache::KvCacheType>,
    /// Bytes of KV the pool named in `kv_type` holds per token (per layer).
    /// Same `None`-means-unmeasured rule as `kv_type` (#111).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bytes_per_token: Option<usize>,
    /// How many times smaller that pool is than the F32 pool (`1.0` for F32
    /// itself). Same `None`-means-unmeasured rule as `kv_type` (#111).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ratio_vs_f32: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speculative: Option<crate::speculative::profiler::SpeculativeReport>,
}

/// The numbers that only a dedicated probe can produce (#100).
///
/// Every field is an `Option` on purpose: a probe that did not run is `None`
/// (the gate skips the row), never a `0.0` that would read as a measurement.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct KernelProbes {
    /// GEMV throughput of one Q8_0 kernel over the host's measured peak
    /// bandwidth, in percent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gemv_peak_pct: Option<f64>,
    /// Attention at ctx 4096 / batch 1 over its theoretical floor (2 * ctx *
    /// heads * head_dim * bytes / peak bandwidth).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub paged_attn_ctx4096_b1_over_theoretical: Option<f64>,
    /// CPU-GPU idle gap as a percentage of decode step wall time (Metal only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub idle_gap_pct: Option<f64>,
    /// 1.0 when the SSE probe consumed a whole `chat.completion.chunk` stream
    /// over plain HTTP/1.1, 0.0 when it did not. A functional result, not a rate.
    /// Named for what it measures: the real OpenAI SDK client is #101's job.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sse_stream_consumed: Option<f64>,
}

impl BenchmarkReport {
    pub fn with_speculative(
        mut self,
        speculative: crate::speculative::profiler::SpeculativeReport,
    ) -> Self {
        self.speculative = Some(speculative);
        self
    }

    /// Attaches peak RSS, the figure `peak_rss_mb` rows judge (#100).
    #[must_use]
    pub fn with_peak_rss_mb(mut self, peak_rss_mb: f64) -> Self {
        self.peak_rss_mb = Some(peak_rss_mb);
        self
    }

    /// Attaches the perplexity a `--eval-ppl` run measured (#100).
    #[must_use]
    pub fn with_perplexity_wikitext2(mut self, perplexity: f64) -> Self {
        self.perplexity_wikitext2 = Some(perplexity);
        self
    }

    /// Records the context length this report ran (#100).
    #[must_use]
    pub fn with_context_len(mut self, context_len: usize) -> Self {
        self.context_len = Some(context_len);
        self
    }

    /// Attaches the probe results gathered alongside this workload (#100).
    #[must_use]
    pub fn with_probes(mut self, probes: KernelProbes) -> Self {
        self.probes = Some(probes);
        self
    }

    /// Attaches the KV pool's memory numbers: which pool type served the run,
    /// its bytes per token, and its ratio against the F32 pool (#111). Taken
    /// from the pool itself, so the report cannot disagree with the thing it
    /// measured.
    #[must_use]
    pub fn with_kv_memory(mut self, pool: &crate::kv_cache::KvStorageKind) -> Self {
        self.kv_type = Some(pool.cache_type());
        self.bytes_per_token = Some(pool.memory_bytes_per_token());
        self.ratio_vs_f32 = Some(pool.memory_ratio_vs_f32() as f64);
        self
    }

    /// Attaches weighed bytes and measured peak hardware bandwidth to report (#62 F3).
    #[must_use]
    pub fn with_telemetry(mut self, bytes_read_per_step: usize) -> Self {
        self.bytes_read_per_step = bytes_read_per_step;
        self.telemetry = TelemetryStatus::from_bytes(bytes_read_per_step);
        let (peak, source) = crate::hardware::measure_peak_bandwidth_gb_s();
        self.peak_bandwidth_gb_s = peak;
        self.peak_bandwidth_source = source;
        self
    }

    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// True when nothing was weighed, so the byte fields mean nothing (#91).
    pub fn is_unweighed(&self) -> bool {
        self.telemetry == TelemetryStatus::Unmeasured
    }

    pub fn to_json_pretty(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}
/// A zero byte count is never a measurement: a resident model weighs something.
impl TelemetryStatus {
    pub fn from_bytes(bytes_read_per_step: usize) -> Self {
        if bytes_read_per_step > 0 {
            Self::Measured
        } else {
            Self::Unmeasured
        }
    }
}

/// Numbers-only report: no weighing happens here, so the result is marked
/// [`TelemetryStatus::Unmeasured`] and the gate will not pass it (#91).
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
        0,
    )
}

#[allow(clippy::too_many_arguments)] // #57: public API; regrouping params out of scope
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
    bytes_read_per_step: usize,
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
        bytes_read_per_step,
    )
}

#[allow(clippy::too_many_arguments)] // #57: public API; regrouping params out of scope
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
    bytes_read_per_step: usize,
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
    let (ttft_p50_ms, ttft_p90_ms, _) = traffic::calculate_percentiles(ttft_ms_records);

    let itl_mean_ms = if !itl_ms_records.is_empty() {
        itl_ms_records.iter().sum::<f64>() / (itl_ms_records.len() as f64)
    } else {
        0.0
    };
    let (itl_p50_ms, itl_p90_ms, itl_p99_ms) = traffic::calculate_percentiles(itl_ms_records);
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

    let decode_ms_per_token = if total_tokens > 0 && decode_duration_secs > 0.0 {
        (decode_duration_secs * 1000.0) / (total_tokens as f64)
    } else {
        decode_ms_per_step
    };

    let (peak_bandwidth_gb_s, peak_bandwidth_source) =
        crate::hardware::measure_peak_bandwidth_gb_s();

    BenchmarkReport {
        total_tokens,
        total_duration_secs,
        num_requests,
        total_tokens_per_sec,
        tokens_per_sec_per_request,
        avg_ms_per_step: decode_ms_per_step,
        mean_ttft_ms,
        max_ttft_ms,
        ttft_p50_ms,
        ttft_p90_ms,
        itl_mean_ms,
        itl_p50_ms,
        itl_p90_ms,
        itl_p99_ms,
        prefill_tokens_per_sec,
        decode_ms_per_step,
        decode_ms_per_token,
        checksum,
        prefix_hit_rate: 0.0,
        bytes_read_per_step,
        peak_bandwidth_gb_s,
        peak_bandwidth_source,
        telemetry: TelemetryStatus::from_bytes(bytes_read_per_step),
        peak_rss_mb: Some(crate::hardware::get_peak_rss_mb()),
        perplexity_wikitext2: None,
        context_len: None,
        probes: None,
        kv_type: None,
        bytes_per_token: None,
        ratio_vs_f32: None,
        speculative: None,
    }
}

fn execute_engine_run(
    engine: &mut Engine,
    requests: &[TraceEntry],
) -> Result<BenchmarkReport, EngineError> {
    let mut order: Vec<usize> = (0..requests.len()).collect();
    order.sort_by_key(|&i| requests[i].arrival_step);

    let initial_matched = engine.prefix_matched_tokens;
    let initial_total = engine.total_prompt_tokens_admitted;
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
            let prompt: Vec<u32> = if let Some(tokens) = &entry.prompt_tokens {
                tokens.clone()
            } else {
                (0..entry.prompt_len)
                    .map(|p| ((req_idx * 10 + p) % 30) as u32)
                    .collect()
            };
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
            if let std::collections::hash_map::Entry::Vacant(e) =
                first_token_times.entry(out.request_id)
            {
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

    let total_ctx: usize = requests.iter().map(|t| t.prompt_len + t.gen_len).sum();
    let bytes_read_per_step = engine.model.weights.storage_bytes()
        + engine.model.config.kv_cache_bytes_per_token_f32() * total_ctx;

    let mut report = evaluate_run_with_timings(
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
        bytes_read_per_step,
    );
    let delta_matched = engine.prefix_matched_tokens.saturating_sub(initial_matched);
    let delta_total = engine
        .total_prompt_tokens_admitted
        .saturating_sub(initial_total);
    report.prefix_hit_rate = if delta_total == 0 {
        0.0
    } else {
        (delta_matched as f64 / delta_total as f64).min(1.0)
    };
    // The pool that really served this run (#111): the report carries its own
    // numbers, so the memory saving is measured, not claimed.
    let report = report.with_kv_memory(&engine.kv_storage);
    Ok(report)
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
            prompt_tokens: None,
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
    backend: crate::hardware::Backend,
    num_requests: usize,
    max_tokens_per_seq: usize,
    block_size: usize,
    max_running: usize,
) -> Result<Engine, EngineError> {
    let config = crate::config::ModelConfig::from_model_type(model_type)?;
    let blocks_per_seq = max_tokens_per_seq.div_ceil(block_size) + 1;
    let total_blocks = (blocks_per_seq * num_requests.max(1)).max(128);
    let mut model = crate::model::Model::create_mock_from_config(&config)?;
    model.set_backend(backend);
    Engine::new(model, max_running, total_blocks, block_size)
}
