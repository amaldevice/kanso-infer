use std::io::{self, Write};
use std::sync::Arc;
use clap::{Parser, Subcommand};
use parking_lot::Mutex;

use kanso_infer::bench::create_engine_for_model_profile;
use kanso_infer::chat::ChatSession;
use kanso_infer::server::create_app;

#[derive(Parser, Debug)]
#[command(name = "kanso", about = "kanso-infer: A minimalist, high-performance LLM inference engine")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Run single or batched completion directly from CLI
    Cli {
        /// Prompt text to generate from
        #[arg(short, long)]
        prompt: Vec<String>,

        /// Path to real GGUF model file (REQUIRED; shadow/mock models are disabled)
        #[arg(short, long)]
        model: String,

        /// Hardware backend: cpu, metal, neon, amx
        #[arg(long, default_value = "cpu")]
        backend: String,

        /// Maximum tokens to generate
        #[arg(long, default_value_t = 32)]
        max_tokens: usize,
    },

    /// Start an interactive multi-turn terminal chat session (REPL)
    Chat {
        /// Path to real GGUF model file (REQUIRED; shadow/mock models are disabled)
        #[arg(short, long)]
        model: String,

        /// Hardware backend: cpu, metal, neon, amx
        #[arg(long, default_value = "cpu")]
        backend: String,

        /// Optional custom system prompt
        #[arg(long, default_value = "You are a helpful and concise assistant.")]
        system: String,

        /// Maximum tokens per response
        #[arg(long, default_value_t = 64)]
        max_tokens: usize,
    },

    /// Launch the OpenAI-compatible HTTP server
    Serve {
        /// HTTP port to bind
        #[arg(short, long, default_value_t = 8000)]
        port: u16,

        /// Path to real GGUF model file (REQUIRED; shadow/mock models are disabled)
        #[arg(short, long)]
        model: String,

        /// Hardware backend: cpu, metal, neon, amx
        #[arg(long, default_value = "cpu")]
        backend: String,
    },

    /// Run performance benchmarks and quality evaluations
    Bench {
        /// Comma-separated list of batch sizes to sweep
        #[arg(long, default_value = "1,4,8,16,32")]
        batch_sizes: String,

        /// Path to JSONL trace file
        #[arg(long)]
        trace_file: Option<String>,

        /// Dataset format: "synthetic", "sharegpt", "lmsys"
        #[arg(long, default_value = "synthetic")]
        dataset: String,

        /// Synthetic prompt token length
        #[arg(long, default_value_t = 128)]
        prompt_len: usize,

        /// Synthetic generation token length
        #[arg(long, default_value_t = 64)]
        gen_len: usize,

        /// Hardware backend: cpu, metal, neon, amx
        #[arg(long, default_value = "cpu")]
        backend: String,

        /// KV cache block size
        #[arg(long, default_value_t = 16)]
        block_size: usize,

        /// Maximum concurrent running sequences
        #[arg(long, default_value_t = 64)]
        max_running: usize,

        /// Skip warmup run before measurement
        #[arg(long, default_value_t = false)]
        no_warmup: bool,

        /// Path to model weights
        #[arg(long)]
        model: Option<String>,

        /// Model profile architecture: "mock", "smollm2", "granite-dense", "granite-h", "lfm"
        #[arg(long, default_value = "mock")]
        model_type: String,

        /// Run perplexity evaluation on specified dataset (e.g. "wikitext2")
        #[arg(long)]
        eval_ppl: Option<String>,

        /// Sliding window context stride for perplexity evaluation
        #[arg(long, default_value_t = 512)]
        ppl_stride: usize,

        /// Path to export report in JSON format
        #[arg(long)]
        json_output: Option<String>,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Cli {
            prompt,
            model,
            backend,
            max_tokens,
        } => {
            if prompt.is_empty() {
                eprintln!("Error: Please provide at least one prompt with --prompt \"...\"");
                return Ok(());
            }

            let resolved_path = match kanso_infer::weights::resolve_model_path(&model) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("{}", e);
                    std::process::exit(1);
                }
            };
            println!("Loading model from '{}' on backend '{}'...", resolved_path.display(), backend);
            let mut model_instance = kanso_infer::model::Model::load_gguf(&resolved_path, candle_core::Device::Cpu)?;
            model_instance.set_backend(&backend);
            let total_blocks = ((128 + max_tokens + 15) / 16) * prompt.len().max(1);
            let mut engine = kanso_infer::engine::Engine::new(model_instance, 8, total_blocks.max(32), 16);

            for p in &prompt {
                // Mock tokenization for prompt
                let tokens: Vec<u32> = p.as_bytes().iter().map(|&b| (b % 40) as u32).collect();
                let req_id = engine.add_request(tokens, max_tokens);

                let mut gen_tokens = Vec::new();
                while engine.has_unfinished_requests() {
                    let outputs = engine.step()?;
                    for out in outputs {
                        if out.request_id == req_id {
                            gen_tokens.push(out.token_id);
                        }
                    }
                }

                println!("\nPrompt: \"{}\"", p);
                println!("Output: Generated {} tokens successfully", gen_tokens.len());
            }
        }

        Commands::Chat {
            model,
            backend,
            system,
            max_tokens,
        } => {
            println!("============================================================");
            println!(" kanso-infer interactive chat session (Model: {}, Backend: {})", model, backend);
            println!(" Type 'exit', 'quit' or Ctrl-D to quit. Type '/reset' to clear history.");
            println!("============================================================");
            let resolved_path = match kanso_infer::weights::resolve_model_path(&model) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("{}", e);
                    std::process::exit(1);
                }
            };
            println!("Loading model from '{}' on backend '{}'...", resolved_path.display(), backend);
            let mut model_instance = kanso_infer::model::Model::load_gguf(&resolved_path, candle_core::Device::Cpu)?;
            model_instance.set_backend(&backend);
            let total_blocks = ((256 + max_tokens + 15) / 16) * 4;
            let engine = kanso_infer::engine::Engine::new(model_instance, 4, total_blocks.max(64), 16);
            let mut session = ChatSession::new(engine, Some(system));
            let stdin = io::stdin();

            loop {
                print!("\nUser > ");
                io::stdout().flush()?;

                let mut line = String::new();
                if stdin.read_line(&mut line)? == 0 {
                    println!("\nGoodbye!");
                    break;
                }

                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                if trimmed == "exit" || trimmed == "quit" {
                    println!("Goodbye!");
                    break;
                }
                if trimmed == "/reset" {
                    session.reset();
                    println!("Conversation history reset.");
                    continue;
                }

                session.add_user_message(trimmed);
                print!("Assistant > ");
                io::stdout().flush()?;

                match session.step_reply(max_tokens) {
                    Ok(reply) => println!("{}", reply),
                    Err(e) => eprintln!("Error generating reply: {}", e),
                }
            }
        }

        Commands::Serve {
            port,
            model,
            backend,
        } => {
            println!("Starting OpenAI-compatible server on http://localhost:{}...", port);
            println!("Model: {}, Backend: {}", model, backend);

            let resolved_path = match kanso_infer::weights::resolve_model_path(&model) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("{}", e);
                    std::process::exit(1);
                }
            };
            println!("Loading model from '{}' on backend '{}'...", resolved_path.display(), backend);
            let mut model_instance = kanso_infer::model::Model::load_gguf(&resolved_path, candle_core::Device::Cpu)?;
            model_instance.set_backend(&backend);
            let engine = kanso_infer::engine::Engine::new(model_instance, 8, 128, 16);

            let shared_engine = Arc::new(Mutex::new(engine));
            let app = create_app(shared_engine);

            let addr = format!("0.0.0.0:{}", port);
            let listener = tokio::net::TcpListener::bind(&addr).await?;
            println!("Listening on {}", addr);
            axum::serve(listener, app).await?;
        }

        Commands::Bench {
            batch_sizes,
            trace_file,
            dataset,
            prompt_len,
            gen_len,
            backend,
            block_size,
            max_running,
            no_warmup,
            model: _,
            model_type,
            eval_ppl,
            ppl_stride,
            json_output,
        } => {
            if let Some(ds) = eval_ppl {
                println!("Evaluating perplexity on dataset: {}", ds);
                let report = kanso_infer::bench::run_wikitext2_perplexity_eval(&model_type, ppl_stride)?;
                let ascii_table = kanso_infer::bench::format_ascii_ppl_report(&report);
                println!("{}", ascii_table);

                if let Some(json_path) = json_output {
                    let json_content = serde_json::to_string_pretty(&report)?;
                    std::fs::write(&json_path, json_content)?;
                    println!("Exported JSON perplexity report to {}", json_path);
                }
            } else if let Some(trace_path) = trace_file {
                let trace = if dataset.to_lowercase() == "sharegpt" || dataset.to_lowercase() == "lmsys" {
                    let content = std::fs::read_to_string(&trace_path)?;
                    kanso_infer::bench::parse_sharegpt_dataset(&content)?
                } else {
                    kanso_infer::bench::parse_trace_file(&trace_path)?
                };
                let max_tokens = trace.iter().map(|t| t.prompt_len + t.gen_len).max().unwrap_or(prompt_len + gen_len);
                let mut engine = create_engine_for_model_profile(&model_type, &backend, trace.len(), max_tokens, block_size, max_running)?;
                let report = kanso_infer::bench::run_trace_benchmark(&mut engine, &trace, !no_warmup)?;
                println!("{:#?}", report);
                if let Some(json_path) = json_output {
                    let json_content = serde_json::to_string_pretty(&report)?;
                    std::fs::write(&json_path, json_content)?;
                    println!("Exported JSON trace benchmark report to {}", json_path);
                }
            } else {
                let batch_sizes_vec: Vec<usize> = batch_sizes
                    .split(',')
                    .map(|s| s.trim().parse::<usize>())
                    .collect::<Result<_, _>>()?;
                let mut reports = Vec::new();
                for &b in &batch_sizes_vec {
                    let max_tokens = prompt_len + gen_len;
                    let mut engine = create_engine_for_model_profile(&model_type, &backend, b, max_tokens, block_size, max_running)?;
                    let report = kanso_infer::bench::run_synthetic_benchmark(&mut engine, b, prompt_len, gen_len, !no_warmup)?;
                    println!("{:#?}", report);
                    reports.push(report);
                }
                if let Some(json_path) = json_output {
                    let json_content = if reports.len() == 1 {
                        serde_json::to_string_pretty(&reports[0])?
                    } else {
                        serde_json::to_string_pretty(&reports)?
                    };
                    std::fs::write(&json_path, json_content)?;
                    println!("Exported JSON benchmark sweep report to {}", json_path);
                }
            }
        }
    }

    Ok(())
}
