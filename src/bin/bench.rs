use clap::Parser;
use kanso_infer::bench::{
    parse_trace_file, run_synthetic_benchmark, run_trace_benchmark, BenchmarkReport,
};
use kanso_infer::engine::Engine;
use kanso_infer::error::EngineError;
use kanso_infer::model::Model;

#[derive(Parser, Debug)]
#[command(name = "kanso-bench", about = "kanso-infer multi-engine benchmark suite")]
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

    /// Execution backend: "cpu", "metal", "neon", or "amx"
    #[arg(long, default_value = "cpu")]
    pub backend: String,

    /// KV cache block size
    #[arg(long, default_value_t = 16)]
    pub block_size: usize,

    /// Maximum concurrent running sequences
    #[arg(long, default_value_t = 64)]
    pub max_running: usize,

    /// Skip warmup run before measurement
    #[arg(long, default_value_t = false)]
    pub no_warmup: bool,

    /// Path to model weights (optional; uses mock tiny model if not supplied)
    #[arg(long)]
    pub model: Option<String>,

    /// Model profile type: "mock", "smollm2", "granite-dense", "granite-h", "lfm"
    #[arg(long, default_value = "mock")]
    pub model_type: String,
}

fn print_report(label: &str, report: &BenchmarkReport, backend: &str) {
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
        report.num_requests,
        report.total_tokens_per_sec,
        report.tokens_per_sec_per_request,
    );
    println!(
        "{:<24} ttft_ms(mean/max)={:.1}/{:.1} itl_ms(mean/p99)={:.1}/{:.1} prefill_tok/s={:.1} decode_ms/step={:.2} prefix_hit={:.1}% checksum={:016x}",
        "",
        report.mean_ttft_ms,
        report.max_ttft_ms,
        report.itl_mean_ms,
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
}

fn create_engine_for_workload(
    model_type: &str,
    backend: &str,
    num_requests: usize,
    max_tokens_per_seq: usize,
    block_size: usize,
    max_running: usize,
) -> Result<Engine, EngineError> {
    if model_type.to_lowercase() != "mock" {
        return kanso_infer::bench::create_engine_for_model_profile(
            model_type,
            backend,
            num_requests,
            max_tokens_per_seq,
            block_size,
            max_running,
        );
    }

    // Sizing total blocks to fit all concurrent sequences without premature eviction
    let blocks_per_seq = (max_tokens_per_seq + block_size - 1) / block_size + 1;
    let total_blocks = (blocks_per_seq * num_requests.max(1)).max(128);

    // Initialize model backend
    match backend.to_lowercase().as_str() {
        "cpu" => {
            let mut model = Model::create_mock_tiny(64, 4, 2, 2, 256)?;
            model.set_backend("cpu");
            Ok(Engine::new(model, max_running, total_blocks, block_size))
        }
        "neon" => {
            #[cfg(target_arch = "aarch64")]
            {
                let mut model = Model::create_mock_tiny(64, 4, 2, 2, 256)?;
                model.set_backend("neon");
                Ok(Engine::new(model, max_running, total_blocks, block_size))
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                eprintln!("Warning: NEON backend requested on non-aarch64 architecture, falling back to CPU reference.");
                let mut model = Model::create_mock_tiny(64, 4, 2, 2, 256)?;
                model.set_backend("cpu");
                Ok(Engine::new(model, max_running, total_blocks, block_size))
            }
        }
        "metal" => {
            #[cfg(target_os = "macos")]
            {
                // Verify Metal context availability
                if let Ok(_ctx) = kanso_infer::metal::context::MetalContext::new() {
                    let mut model = Model::create_mock_tiny(64, 4, 2, 2, 256)?;
                    model.set_backend("metal");
                    Ok(Engine::new(model, max_running, total_blocks, block_size))
                } else {
                    eprintln!("Warning: Metal unavailable, falling back to CPU reference.");
                    let mut model = Model::create_mock_tiny(64, 4, 2, 2, 256)?;
                    model.set_backend("cpu");
                    Ok(Engine::new(model, max_running, total_blocks, block_size))
                }
            }
            #[cfg(not(target_os = "macos"))]
            {
                eprintln!("Warning: Metal backend only supported on macOS, falling back to CPU reference.");
                let mut model = Model::create_mock_tiny(64, 4, 2, 2, 256)?;
                model.set_backend("cpu");
                Ok(Engine::new(model, max_running, total_blocks, block_size))
            }
        }
        "amx" => {
            #[cfg(target_os = "macos")]
            {
                let mut model = Model::create_mock_tiny(64, 4, 2, 2, 256)?;
                model.set_backend("amx");
                Ok(Engine::new(model, max_running, total_blocks, block_size))
            }
            #[cfg(not(target_os = "macos"))]
            {
                eprintln!("Warning: AMX backend only supported on macOS, falling back to CPU reference.");
                let mut model = Model::create_mock_tiny(64, 4, 2, 2, 256)?;
                model.set_backend("cpu");
                Ok(Engine::new(model, max_running, total_blocks, block_size))
            }
        }
        other => Err(EngineError::Config(format!(
            "Unsupported backend '{}'. Valid options: cpu, metal, neon, amx",
            other
        ))),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args = Args::parse();
    let warmup = !args.no_warmup;

    let hw = kanso_infer::hardware::get_hardware_info();
    let thermal = kanso_infer::hardware::get_thermal_state();

    println!("============================================================");
    println!(" kanso-infer multi-engine benchmark suite");
    println!(" Hardware : {} ({} cores, {:.1} GB RAM, Model: {})", hw.chip_name, hw.core_count, hw.total_ram_gb, hw.model_identifier);
    println!(" Thermal  : {}", thermal.as_str());
    println!(" Backend  : {}, Model Profile: {}, Warmup: {}", args.backend, args.model_type, warmup);
    println!("============================================================");

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
            &args.model_type,
            &args.backend,
            trace.len(),
            max_tokens,
            args.block_size,
            args.max_running,
        )?;

        let report = run_trace_benchmark(&mut engine, &trace, warmup)?;
        print_report("trace_replay", &report, &args.backend);
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

    for &batch_size in &batch_sizes {
        let max_tokens = args.prompt_len + args.gen_len;
        let mut engine = create_engine_for_workload(
            &args.model_type,
            &args.backend,
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
        print_report(&label, &report, &args.backend);
    }

    Ok(())
}
