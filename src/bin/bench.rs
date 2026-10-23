use clap::Parser;
use kanso_infer::Backend;
use kanso_infer::bench::{
    BenchmarkReport, parse_trace_file, run_synthetic_benchmark, run_trace_benchmark,
};
use kanso_infer::engine::Engine;
use kanso_infer::error::EngineError;
use kanso_infer::kv_cache::KvCacheType;
use kanso_infer::model::Model;

#[derive(Parser, Debug)]
#[command(
    name = "kanso-bench",
    about = "kanso-infer multi-engine benchmark suite"
)]
pub struct Args {
    /// Comma-separated list of batch sizes to sweep (e.g. "1,4,8,16,32")
    #[arg(long, default_value = "1,4,8,16,32")]
    pub batch_sizes: String,

    /// Path to JSONL trace file (e.g. ShareGPT trace with prompt_len, gen_len, arrival_step)
    #[arg(long)]
    pub trace_file: Option<String>,
    /// Dataset format: "synthetic" (default), "sharegpt", or "lmsys"
    #[arg(long, default_value = "synthetic")]
    pub dataset: String,

    /// Synthetic prompt token length
    #[arg(long, default_value_t = 128)]
    pub prompt_len: usize,

    /// Synthetic generation token length
    #[arg(long, default_value_t = 64)]
    pub gen_len: usize,

    /// Execution backend: "cpu", "metal", or "amx". "neon" parses but errors: its\n    /// hand-written kernels are not wired to the packed-weight decode path (ticket #89).
    #[arg(long, default_value_t = Backend::default())]
    pub backend: Backend,

    /// KV cache block size
    #[arg(long, default_value_t = 16)]
    pub block_size: usize,

    /// KV cache type: f32 (default), int8 or fp8. The quantized pools are
    /// host-side only: `metal` refuses them at engine construction (ticket #88).
    /// The chosen pool's memory numbers land in the report as `kv_type` /
    /// `bytes_per_token` / `ratio_vs_f32` (ticket #111).
    #[arg(long, default_value_t = KvCacheType::default())]
    pub kv_type: KvCacheType,

    /// Maximum concurrent running sequences
    #[arg(long, default_value_t = 64)]
    pub max_running: usize,

    /// Skip warmup run before measurement
    #[arg(long, default_value_t = false)]
    pub no_warmup: bool,

    /// Path to real GGUF model file (REQUIRED; shadow/mock models are disabled)
    #[arg(short, long)]
    pub model: Option<String>,
    /// Run perplexity evaluation on specified dataset (e.g. "wikitext2")
    #[arg(long)]
    pub eval_ppl: Option<String>,

    /// Sliding window context stride for perplexity evaluation
    #[arg(long, default_value_t = 512)]
    pub ppl_stride: usize,

    /// Path to export benchmark report in JSON format
    #[arg(long)]
    pub json_output: Option<String>,

    /// Path to a gate budget JSON; exits 0 (passed), 1 (violated), 2 (skipped)
    #[arg(long)]
    pub gate: Option<String>,

    /// Evaluate an existing BenchmarkReport JSON against `--gate` instead of running a workload
    #[arg(long, requires = "gate")]
    pub gate_report: Option<String>,

    /// Run the per-kernel / functional probes that fill the F12/F15 rows no
    /// workload figure answers (GEMV share of peak, paged-attention floor,
    /// SSE smoke). Results land in the report under `probes`.
    #[arg(long, default_value_t = false)]
    pub probes: bool,

    /// Comma-separated context lengths to sweep, e.g. "128,2048". Each length
    /// runs the workload once, so `ctx_latency_ratio` has the pair it needs.
    #[arg(long)]
    pub ctx_lengths: Option<String>,

    /// `host:port` of a live `kanso serve` to smoke-test the SSE stream against.
    /// Without it the `sse_stream_consumed` probe does not run, and the gate
    /// skips that row rather than judging a stream nobody opened.
    #[arg(long)]
    pub sse_endpoint: Option<String>,
}

fn print_report(label: &str, report: &BenchmarkReport, backend: Backend) {
    let peak_rss_mb = report.peak_rss_mb;
    let thermal = kanso_infer::hardware::get_thermal_state();

    // Effective bandwidth vs the measured peak carried on the report (#53).
    let (effective_gb_s, util_pct) = kanso_infer::hardware::calculate_bandwidth_utilization(
        report.bytes_read_per_step,
        report.decode_ms_per_step,
        report.peak_bandwidth_gb_s,
    );

    println!(
        "{label:<24} backend={backend:<6} batch={:<4} tok/s(total)={:<9.1} tok/s/req={:<8.2}",
        report.num_requests, report.total_tokens_per_sec, report.tokens_per_sec_per_request,
    );
    println!(
        "{:<24} ttft_ms(p50/p90/mean/max)={:.1}/{:.1}/{:.1}/{:.1} itl_ms(p50/p90/p99)={:.1}/{:.1}/{:.1} prefill_tok/s={:.1} decode_ms/step={:.2} (decode_ms/tok={:.2}) prefix_hit={:.1}% checksum={:016x}",
        "",
        report.ttft_p50_ms,
        report.ttft_p90_ms,
        report.mean_ttft_ms,
        report.max_ttft_ms,
        report.itl_p50_ms,
        report.itl_p90_ms,
        report.itl_p99_ms,
        report.prefill_tokens_per_sec,
        report.decode_ms_per_step,
        report.decode_ms_per_token,
        report.prefix_hit_rate * 100.0,
        report.checksum,
    );
    println!(
        "{:<24} peak_rss={} thermal=\"{}\" bus_est={:.1}GB/s ({:.1}% of {:.1}GB/s {})",
        "",
        match peak_rss_mb {
            Some(mb) => format!("{mb:.1}MB"),
            None => "unmeasured".to_string(),
        },
        thermal.as_str(),
        effective_gb_s,
        util_pct,
        report.peak_bandwidth_gb_s,
        report.peak_bandwidth_source,
    );
    if let Some(spec) = &report.speculative {
        println!("{}", spec.format_report());
    }
    if let Some(ppl) = report.perplexity_wikitext2 {
        println!("{:<24} perplexity(wikitext2)={ppl:.4}", "");
    }
    if let Some(ctx) = report.context_len {
        println!("{:<24} context_len={ctx}", "");
    }
    // The pool that served the run, with its own memory numbers (#111).
    if let (Some(kv_type), Some(bytes), Some(ratio)) =
        (report.kv_type, report.bytes_per_token, report.ratio_vs_f32)
    {
        println!(
            "{:<24} kv={kv_type} bytes_per_token={bytes} ratio_vs_f32={ratio:.2}x",
            ""
        );
    }
    if let Some(probes) = &report.probes {
        println!("{:<24} probes: {probes:?}", "");
    }
}

fn create_engine_for_workload(
    model_path: &std::path::Path,
    backend: Backend,
    num_requests: usize,
    max_tokens_per_seq: usize,
    block_size: usize,
    max_running: usize,
    kv_type: KvCacheType,
) -> Result<Engine, EngineError> {
    let device = backend.device()?;
    println!(
        "Loading model from '{}' on backend '{}' (device: {})...",
        model_path.display(),
        backend,
        kanso_infer::device_label(&device)
    );
    let mut model = Model::load_gguf(model_path, device)?;
    model.set_backend(backend);

    let blocks_per_seq = max_tokens_per_seq.div_ceil(block_size) + 1;
    let total_blocks = (blocks_per_seq * num_requests.max(1)).max(128);

    Engine::new_with_budget_and_kv(
        model,
        max_running,
        total_blocks,
        block_size,
        usize::MAX,
        kv_type.mode(),
    )
}

/// `bench --gate <budget.json> --gate-report <report.json>`: re-checks an
/// existing report without loading a model, then exits with the gate's code.
fn run_gate(
    gate_path: &str,
    report_path: Option<&str>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use std::io::Write;

    let Some(report_path) = report_path else {
        eprintln!("Error: --gate requires --gate-report <report.json>, the evidence to re-check.");
        std::process::exit(1);
    };

    let budget = kanso_infer::bench::GateBudget::from_file(std::path::Path::new(gate_path))?;
    let evidence = std::fs::read_to_string(report_path)
        .map_err(|e| format!("cannot read gate report '{report_path}': {e}"))?;
    let value: serde_json::Value = serde_json::from_str(&evidence)
        .map_err(|e| format!("invalid gate report '{report_path}': {e}"))?;
    // A sweep run writes an array; a single run writes one object.
    let parsed: Vec<BenchmarkReport> = match value {
        serde_json::Value::Array(_) => serde_json::from_value(value)
            .map_err(|e| format!("invalid gate report '{report_path}': {e}"))?,
        single => vec![
            serde_json::from_value(single)
                .map_err(|e| format!("invalid gate report '{report_path}': {e}"))?,
        ],
    };

    let outcome = kanso_infer::bench::evaluate_gate_run(&budget, &parsed)?;
    print!("{}", outcome.format_table());
    let violated = outcome.code() == 1;
    let skipped = outcome.code() == 2;
    println!("evidence: budget JSON {gate_path}");
    println!("evidence: report JSON {report_path}");
    std::io::stdout().flush()?;
    std::process::exit(if violated {
        1
    } else if skipped {
        2
    } else {
        0
    });
}

/// Gate the reports just produced by a run. A violation anywhere outranks a
/// skip; every report gets its own table. The report JSON is always written:
/// to `--json-output` when given, otherwise to a temp file whose path is
/// printed, so the verdict can be re-checked without re-running the workload.
fn gate_reports(
    gate_path: &str,
    labeled: &[(String, &BenchmarkReport)],
    json_output: Option<&str>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use std::io::Write;

    let budget = kanso_infer::bench::GateBudget::from_file(std::path::Path::new(gate_path))?;
    let reports: Vec<BenchmarkReport> = labeled.iter().map(|(_, r)| (*r).clone()).collect();
    let outcome = kanso_infer::bench::evaluate_gate_run(&budget, &reports)?;
    println!(
        "[{}]",
        labeled
            .iter()
            .map(|(label, _)| label.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    );
    print!("{}", outcome.format_table());
    let violated = outcome.code() == 1;
    let skipped = outcome.code() == 2;

    let evidence_path = match json_output {
        Some(path) => std::path::PathBuf::from(path),
        None => {
            let path = std::env::temp_dir().join(format!(
                "kanso-gate-{}-{}.json",
                budget.name,
                std::process::id()
            ));
            let json = if reports.len() == 1 {
                serde_json::to_string_pretty(&reports[0])?
            } else {
                serde_json::to_string_pretty(&reports)?
            };
            std::fs::write(&path, json)?;
            path
        }
    };
    println!("evidence: budget JSON {gate_path}");
    println!("evidence: report JSON {}", evidence_path.display());
    std::io::stdout().flush()?;
    std::process::exit(if violated {
        1
    } else if skipped {
        2
    } else {
        0
    });
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args = Args::parse();
    let warmup = !args.no_warmup;

    if args.gate_report.is_some() {
        let gate_path = args
            .gate
            .as_deref()
            .expect("clap requires --gate alongside --gate-report");
        return run_gate(gate_path, args.gate_report.as_deref());
    }
    let model_path = match &args.model {
        Some(m) => match kanso_infer::weights::resolve_model_path(m) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("{}", e);
                std::process::exit(1);
            }
        },
        None => {
            eprintln!(
                "Error: --model <path> is required. Shadow/mock model execution has been removed.\n\
                 Please provide a path to a real GGUF model file, for example:\n  \
                 bench --model ./models/smollm2-360m.gguf\n\
                 Or download one using:\n  \
                 huggingface-cli download HuggingFaceTB/SmolLM2-360M-Instruct-GGUF smollm2-360m-instruct-q8_0.gguf"
            );
            std::process::exit(1);
        }
    };
    let hw = kanso_infer::hardware::get_hardware_info();
    let thermal = kanso_infer::hardware::get_thermal_state();

    println!("============================================================");
    println!(" kanso-infer multi-engine benchmark suite");
    println!(" Model    : {}", model_path.display());
    println!(
        " Hardware : {} ({} cores, {:.1} GB RAM, Model: {})",
        hw.chip_name, hw.core_count, hw.total_ram_gb, hw.model_identifier
    );
    println!(" Thermal  : {}", thermal.as_str());
    println!(" Backend  : {}, Warmup: {}", args.backend, warmup);
    println!("============================================================");
    // `--eval-ppl` on its own writes the perplexity report and stops. With
    // `--gate` the number is attached to the benchmark reports instead, because
    // the gate reads a `BenchmarkReport`, not a `PerplexityReport` (#100).
    let mut ppl_value: Option<f64> = None;
    if let Some(dataset) = &args.eval_ppl {
        println!("Evaluating perplexity on dataset: {}", dataset);
        // An absent dataset is not an error and not a fixture: it means the run
        // has no perplexity to report, so the gate skips the row (#100).
        match kanso_infer::bench::run_perplexity_eval(
            model_path.to_str().unwrap_or("model"),
            dataset,
            args.ppl_stride,
        ) {
            Ok(report) => {
                let ascii_table = kanso_infer::bench::format_ascii_ppl_report(&report);
                println!("{}", ascii_table);

                if let Some(json_path) = &args.json_output {
                    let json_content = serde_json::to_string_pretty(&report)?;
                    std::fs::write(json_path, json_content)?;
                    println!("Exported JSON perplexity report to {}", json_path);
                }
                if args.gate.is_none() {
                    return Ok(());
                }
                ppl_value = Some(report.baseline_ppl);
            }
            Err(e) => {
                eprintln!("perplexity skipped: {e}");
                if args.gate.is_none() {
                    return Ok(());
                }
            }
        }
    }
    if let Some(trace_path) = &args.trace_file {
        let trace = if args.dataset.to_lowercase() == "sharegpt"
            || args.dataset.to_lowercase() == "lmsys"
        {
            let content = std::fs::read_to_string(trace_path)?;
            kanso_infer::bench::parse_sharegpt_dataset(&content)?
        } else {
            parse_trace_file(trace_path)?
        };
        println!(
            "Running trace replay from '{}' ({} requests)",
            trace_path,
            trace.len()
        );

        let max_tokens = trace
            .iter()
            .map(|t| t.prompt_len + t.gen_len)
            .max()
            .unwrap_or(args.prompt_len + args.gen_len);

        let mut engine = create_engine_for_workload(
            &model_path,
            args.backend,
            trace.len(),
            max_tokens,
            args.block_size,
            args.max_running,
            args.kv_type,
        )?;

        let geometry = kanso_infer::bench::attention_geometry(&engine.model);
        let report = attach_instruments(
            run_trace_benchmark(&mut engine, &trace, warmup)?,
            &args,
            geometry,
            None,
            ppl_value,
        );
        print_report("trace_replay", &report, args.backend);
        if let Some(json_path) = &args.json_output {
            let json_content = serde_json::to_string_pretty(&report)?;
            std::fs::write(json_path, json_content)?;
            println!("Exported JSON trace benchmark report to {}", json_path);
        }
        if let Some(gate_path) = &args.gate {
            gate_reports(
                gate_path,
                &[("trace_replay".to_string(), &report)],
                args.json_output.as_deref(),
            )?;
        }
        return Ok(());
    }

    let batch_sizes: Vec<usize> = args
        .batch_sizes
        .split(',')
        .map(|s| s.trim().parse::<usize>())
        .collect::<Result<_, _>>()?;

    // `--ctx-lengths 128,2048` turns the sweep into a context sweep: one report
    // per context length at the first batch size, which is the pair
    // `ctx_latency_ratio` reads as last/first (#100).
    let ctx_lengths: Option<Vec<usize>> = args
        .ctx_lengths
        .as_deref()
        .map(|list| {
            list.split(',')
                .map(|s| s.trim().parse::<usize>())
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()?;
    let runs: Vec<(usize, usize)> = match &ctx_lengths {
        Some(lengths) => lengths.iter().map(|&ctx| (batch_sizes[0], ctx)).collect(),
        None => batch_sizes
            .iter()
            .map(|&batch| (batch, args.prompt_len))
            .collect(),
    };

    if ctx_lengths.is_some() {
        println!(
            "Running context sweep: gen_len={} block_size={} (prompt_len is the swept context)",
            args.gen_len, args.block_size
        );
    } else {
        println!(
            "Running synthetic sweep: prompt_len={} gen_len={} block_size={}",
            args.prompt_len, args.gen_len, args.block_size
        );
    }

    let mut reports = Vec::new();
    let mut labels = Vec::new();
    for &(batch_size, prompt_len) in &runs {
        let max_tokens = prompt_len + args.gen_len;
        let mut engine = create_engine_for_workload(
            &model_path,
            args.backend,
            batch_size,
            max_tokens,
            args.block_size,
            args.max_running,
            args.kv_type,
        )?;
        let geometry = kanso_infer::bench::attention_geometry(&engine.model);

        let report = attach_instruments(
            run_synthetic_benchmark(&mut engine, batch_size, prompt_len, args.gen_len, warmup)?,
            &args,
            geometry,
            ctx_lengths.as_ref().map(|_| prompt_len),
            ppl_value,
        );

        let label = match &ctx_lengths {
            Some(_) => format!("ctx_len={prompt_len}"),
            None => format!("batch_size={batch_size}"),
        };
        print_report(&label, &report, args.backend);
        reports.push(report);
        labels.push(label);
    }

    if let Some(json_path) = &args.json_output {
        let json_content = if reports.len() == 1 {
            serde_json::to_string_pretty(&reports[0])?
        } else {
            serde_json::to_string_pretty(&reports)?
        };
        std::fs::write(json_path, json_content)?;
        println!("Exported JSON benchmark sweep report to {}", json_path);
    }

    if let Some(gate_path) = &args.gate {
        let labeled: Vec<(String, &BenchmarkReport)> =
            labels.iter().cloned().zip(reports.iter()).collect();
        gate_reports(gate_path, &labeled, args.json_output.as_deref())?;
    }

    Ok(())
}

/// Attaches the numbers no workload figure answers: peak RSS (always), the
/// context length this report ran, the perplexity the `--eval-ppl` pass measured,
/// and — when `--probes` asked for them — the per-kernel / functional probes (#100).
fn attach_instruments(
    report: BenchmarkReport,
    args: &Args,
    geometry: (usize, usize),
    context_len: Option<usize>,
    perplexity: Option<f64>,
) -> BenchmarkReport {
    let mut report = report.with_peak_rss_mb(kanso_infer::hardware::get_peak_rss_mb());
    if let Some(ctx) = context_len {
        report = report.with_context_len(ctx);
    }
    if let Some(ppl) = perplexity {
        report = report.with_perplexity_wikitext2(ppl);
    }
    if args.probes {
        let sse_endpoint = args.sse_endpoint.as_deref().and_then(|endpoint| {
            let (host, port) = endpoint.rsplit_once(':')?;
            Some((host.to_string(), port.parse::<u16>().ok()?))
        });
        let probes = kanso_infer::bench::run_probes(
            report.peak_bandwidth_gb_s,
            geometry.0,
            geometry.1,
            sse_endpoint
                .as_ref()
                .map(|(host, port)| (host.as_str(), *port)),
        );
        report = report.with_probes(probes);
    }
    report
}
