use clap::Parser;
use kanso_infer::Backend;
use kanso_infer::bench::{
    BenchmarkReport, parse_trace_file, run_synthetic_benchmark, run_trace_benchmark,
};
use kanso_infer::engine::Engine;
use kanso_infer::error::EngineError;
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

    /// Execution backend: "cpu", "metal", "neon", or "amx" (unknown values are rejected)
    #[arg(long, default_value_t = Backend::default())]
    pub backend: Backend,

    /// KV cache block size
    #[arg(long, default_value_t = 16)]
    pub block_size: usize,

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
}

fn print_report(label: &str, report: &BenchmarkReport, backend: Backend) {
    let peak_rss_mb = kanso_infer::hardware::get_peak_rss_mb();
    let thermal = kanso_infer::hardware::get_thermal_state();

    // Compute effective bandwidth utilization (assuming ~380 MB weights on a 150 GB/s bus)
    let (effective_gb_s, util_pct) = kanso_infer::hardware::calculate_bandwidth_utilization(
        380_000_000,
        report.decode_ms_per_step,
        150.0,
    );

    println!(
        "{label:<24} backend={backend:<6} batch={:<4} tok/s(total)={:<9.1} tok/s/req={:<8.2}",
        report.num_requests, report.total_tokens_per_sec, report.tokens_per_sec_per_request,
    );
    println!(
        "{:<24} ttft_ms(p50/p90/mean/max)={:.1}/{:.1}/{:.1}/{:.1} itl_ms(p50/p90/p99)={:.1}/{:.1}/{:.1} prefill_tok/s={:.1} decode_ms/step={:.2} prefix_hit={:.1}% checksum={:016x}",
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
        report.prefix_hit_rate * 100.0,
        report.checksum,
    );
    println!(
        "{:<24} peak_rss={:.1}MB thermal=\"{}\" bus_est={:.1}GB/s ({:.1}% bus util)",
        "",
        peak_rss_mb,
        thermal.as_str(),
        effective_gb_s,
        util_pct,
    );
    if let Some(spec) = &report.speculative {
        println!("{}", spec.format_report());
    }
}

fn create_engine_for_workload(
    model_path: &std::path::Path,
    backend: Backend,
    num_requests: usize,
    max_tokens_per_seq: usize,
    block_size: usize,
    max_running: usize,
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

    Ok(Engine::new(model, max_running, total_blocks, block_size))
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args = Args::parse();
    let warmup = !args.no_warmup;

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
    if let Some(dataset) = &args.eval_ppl {
        println!("Evaluating perplexity on dataset: {}", dataset);
        let report = kanso_infer::bench::run_wikitext2_perplexity_eval(
            model_path.to_str().unwrap_or("model"),
            args.ppl_stride,
        )?;
        let ascii_table = kanso_infer::bench::format_ascii_ppl_report(&report);
        println!("{}", ascii_table);

        if let Some(json_path) = &args.json_output {
            let json_content = serde_json::to_string_pretty(&report)?;
            std::fs::write(json_path, json_content)?;
            println!("Exported JSON perplexity report to {}", json_path);
        }
        return Ok(());
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
        )?;

        let report = run_trace_benchmark(&mut engine, &trace, warmup)?;
        print_report("trace_replay", &report, args.backend);
        if let Some(json_path) = &args.json_output {
            let json_content = serde_json::to_string_pretty(&report)?;
            std::fs::write(json_path, json_content)?;
            println!("Exported JSON trace benchmark report to {}", json_path);
        }
        return Ok(());
    }

    let batch_sizes: Vec<usize> = args
        .batch_sizes
        .split(',')
        .map(|s| s.trim().parse::<usize>())
        .collect::<Result<_, _>>()?;

    println!(
        "Running synthetic sweep: prompt_len={} gen_len={} block_size={}",
        args.prompt_len, args.gen_len, args.block_size
    );

    let mut reports = Vec::new();
    for &batch_size in &batch_sizes {
        let max_tokens = args.prompt_len + args.gen_len;
        let mut engine = create_engine_for_workload(
            &model_path,
            args.backend,
            batch_size,
            max_tokens,
            args.block_size,
            args.max_running,
        )?;

        let report = run_synthetic_benchmark(
            &mut engine,
            batch_size,
            args.prompt_len,
            args.gen_len,
            warmup,
        )?;

        let label = format!("batch_size={batch_size}");
        print_report(&label, &report, args.backend);
        reports.push(report);
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

    Ok(())
}
