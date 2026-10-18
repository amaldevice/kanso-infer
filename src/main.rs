use clap::{Parser, Subcommand};
use std::io::{self, Write};

use kanso_infer::bench::create_engine_for_model_profile;
use kanso_infer::chat::ChatSession;
use kanso_infer::engine::Engine;
use kanso_infer::error::EngineError;
use kanso_infer::model::Model;
use kanso_infer::server::create_app;

/// Builds a session engine, turning a KV request the machine cannot serve into a
/// user-facing message and a non-zero exit instead of a panic (ticket #85).
fn build_engine(
    model: Model,
    max_running: usize,
    block_size: usize,
    context_tokens: usize,
    requests: usize,
    min_blocks: usize,
) -> Engine {
    let per_sequence = context_tokens.div_ceil(block_size);
    let total_blocks = per_sequence
        .checked_mul(requests.max(1))
        .map(|total| total.max(min_blocks));

    let built = match total_blocks {
        Some(blocks) => Engine::new(model, max_running, blocks, block_size),
        None => Err(EngineError::Config(format!(
            "cannot size the KV pool: {context_tokens} context tokens x {requests} requests exceeds the addressable size"
        ))),
    };

    match built {
        Ok(engine) => engine,
        Err(e) => {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    }
}
#[derive(Parser, Debug)]
#[command(
    name = "kanso",
    about = "kanso-infer: A minimalist, high-performance LLM inference engine"
)]
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

        /// Hardware backend: cpu, metal, neon, amx (unknown values are rejected)
        #[arg(long, default_value_t = kanso_infer::Backend::default())]
        backend: kanso_infer::Backend,

        /// Maximum tokens to generate
        #[arg(long, default_value_t = 32)]
        max_tokens: usize,

        /// Path to tokenizer.json (optional, auto-resolved from working directory or HF cache)
        #[arg(long)]
        tokenizer: Option<String>,

        /// Sampling temperature; 0 = greedy (default)
        #[arg(long, default_value_t = 0.0)]
        temperature: f32,

        /// Nucleus sampling top-p
        #[arg(long, default_value_t = 1.0)]
        top_p: f32,

        /// Sampling RNG seed
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },

    /// Start an interactive multi-turn terminal chat session (REPL)
    Chat {
        /// Path to real GGUF model file (REQUIRED; shadow/mock models are disabled)
        #[arg(short, long)]
        model: String,

        /// Hardware backend: cpu, metal, neon, amx (unknown values are rejected)
        #[arg(long, default_value_t = kanso_infer::Backend::default())]
        backend: kanso_infer::Backend,

        /// Optional custom system prompt
        #[arg(long, default_value = "You are a helpful and concise assistant.")]
        system: String,

        /// Maximum tokens per response
        #[arg(long, default_value_t = 64)]
        max_tokens: usize,

        /// Path to tokenizer.json (optional, auto-resolved from working directory or HF cache)
        #[arg(long)]
        tokenizer: Option<String>,
    },

    /// Launch the OpenAI-compatible HTTP server
    Serve {
        /// HTTP port to bind
        #[arg(short, long, default_value_t = 8000)]
        port: u16,

        /// Path to real GGUF model file (REQUIRED; shadow/mock models are disabled)
        #[arg(short, long)]
        model: String,

        /// Hardware backend: cpu, metal, neon, amx (unknown values are rejected)
        #[arg(long, default_value_t = kanso_infer::Backend::default())]
        backend: kanso_infer::Backend,

        /// Path to tokenizer.json (optional, auto-resolved from working directory or HF cache)
        #[arg(long)]
        tokenizer: Option<String>,
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

        /// Hardware backend: cpu, metal, neon, amx (unknown values are rejected)
        #[arg(long, default_value_t = kanso_infer::Backend::default())]
        backend: kanso_infer::Backend,

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
            tokenizer,
            temperature,
            top_p,
            seed,
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
            let device = backend.device()?;
            println!(
                "Loading model from '{}' on backend '{}' (device: {})...",
                resolved_path.display(),
                backend,
                kanso_infer::device_label(&device)
            );
            let mut model_instance = kanso_infer::model::Model::load_gguf(&resolved_path, device)?;
            model_instance.set_backend(backend);
            let context_tokens = 128usize.saturating_add(max_tokens);
            let mut engine = build_engine(model_instance, 8, 16, context_tokens, prompt.len(), 32);

            let tok_wrapper = kanso_infer::tokenizer::TokenizerWrapper::resolve(
                tokenizer.as_deref(),
                Some(&resolved_path),
            )
            .ok();

            for p in &prompt {
                let tokens = if let Some(tok) = &tok_wrapper {
                    tok.encode(p)
                        .unwrap_or_else(|_| p.as_bytes().iter().map(|&b| (b % 40) as u32).collect())
                } else {
                    p.as_bytes().iter().map(|&b| (b % 40) as u32).collect()
                };
                let mut decoder = kanso_infer::tokenizer::StreamDecoder::new();
                let eos_id = tok_wrapper.as_ref().map(|t| t.eos_token_id).unwrap_or(2);
                // #56: EOS retires the sequence in the engine (blocks freed) instead of
                // generating up to max_tokens after the client already stopped reading.
                // F5: the user's sampling knobs ride along on the sequence.
                let req_id = engine.add_request_with_sampling(
                    tokens,
                    max_tokens,
                    vec![eos_id],
                    kanso_infer::scheduler::SamplingParams {
                        temperature,
                        top_p,
                        seed,
                    },
                );

                println!("\nPrompt: \"{}\"", p);
                print!("Assistant > ");
                io::stdout().flush()?;

                let mut done = false;

                while !done && engine.has_unfinished_requests() {
                    let outputs = engine.step()?;
                    for out in outputs {
                        if out.request_id == req_id {
                            if let Some(tok) = &tok_wrapper {
                                let delta = decoder.push(tok, out.token_id)?;
                                let clean = delta.replace("<|im_end|>", "");
                                print!("{}", clean);
                                io::stdout().flush()?;
                                if delta.contains("<|im_end|>") {
                                    done = true;
                                }
                            } else {
                                print!("tok_{} ", out.token_id);
                                io::stdout().flush()?;
                            }

                            if out.token_id == eos_id || out.is_finished {
                                done = true;
                            }
                        }
                    }
                }

                // #56: retire this prompt's sequence before the next one.
                engine.drain_finished();

                if let Some(tok) = &tok_wrapper {
                    let rem = decoder.flush(tok)?;
                    let clean = rem.replace("<|im_end|>", "");
                    print!("{}", clean);
                    io::stdout().flush()?;
                }
                println!();
            }
        }

        Commands::Chat {
            model,
            backend,
            system,
            max_tokens,
            tokenizer,
        } => {
            println!("============================================================");
            println!(
                " kanso-infer interactive chat session (Model: {}, Backend: {})",
                model, backend
            );
            println!(" Type 'exit', 'quit' or Ctrl-D to quit. Type '/reset' to clear history.");
            println!("============================================================");
            let resolved_path = match kanso_infer::weights::resolve_model_path(&model) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("{}", e);
                    std::process::exit(1);
                }
            };
            let device = backend.device()?;
            println!(
                "Loading model from '{}' on backend '{}' (device: {})...",
                resolved_path.display(),
                backend,
                kanso_infer::device_label(&device)
            );
            let mut model_instance = kanso_infer::model::Model::load_gguf(&resolved_path, device)?;
            model_instance.set_backend(backend);
            let context_tokens = 256usize.saturating_add(max_tokens);
            let engine = build_engine(model_instance, 4, 16, context_tokens, 4, 64);
            let tok_wrapper = kanso_infer::tokenizer::TokenizerWrapper::resolve(
                tokenizer.as_deref(),
                Some(&resolved_path),
            )
            .ok();
            let mut session = ChatSession::new_with_tokenizer(engine, Some(system), tok_wrapper);
            let stdin = io::stdin();

            loop {
                #[cfg(target_family = "unix")]
                unsafe {
                    libc::tcflush(libc::STDIN_FILENO, libc::TCIFLUSH);
                }

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

                match session.stream_step_reply(max_tokens, |delta| {
                    print!("{}", delta);
                    let _ = io::stdout().flush();
                }) {
                    Ok(_) => println!(),
                    Err(e) => eprintln!("\nError generating reply: {}", e),
                }
            }
        }

        Commands::Serve {
            port,
            model,
            backend,
            tokenizer,
        } => {
            println!(
                "Starting OpenAI-compatible server on http://localhost:{}...",
                port
            );
            println!("Model: {}, Backend: {}", model, backend);
            let resolved_path = match kanso_infer::weights::resolve_model_path(&model) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("{}", e);
                    std::process::exit(1);
                }
            };
            let device = backend.device()?;
            println!(
                "Loading model from '{}' on backend '{}' (device: {})...",
                resolved_path.display(),
                backend,
                kanso_infer::device_label(&device)
            );
            let mut model_instance = kanso_infer::model::Model::load_gguf(&resolved_path, device)?;
            model_instance.set_backend(backend);
            let engine = build_engine(model_instance, 8, 16, 128, 1, 128);

            // #46: the server tokenizes and detokenizes with the real tokenizer, so a
            // missing one is a startup failure rather than silently wrong text.
            let tokenizer = match kanso_infer::tokenizer::TokenizerWrapper::resolve(
                tokenizer.as_deref(),
                Some(&resolved_path),
            ) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("{}", e);
                    std::process::exit(1);
                }
            };

            let app = create_app(engine, tokenizer);

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
                let report =
                    kanso_infer::bench::run_wikitext2_perplexity_eval(&model_type, ppl_stride)?;
                let ascii_table = kanso_infer::bench::format_ascii_ppl_report(&report);
                println!("{}", ascii_table);

                if let Some(json_path) = json_output {
                    let json_content = serde_json::to_string_pretty(&report)?;
                    std::fs::write(&json_path, json_content)?;
                    println!("Exported JSON perplexity report to {}", json_path);
                }
            } else if let Some(trace_path) = trace_file {
                let trace =
                    if dataset.to_lowercase() == "sharegpt" || dataset.to_lowercase() == "lmsys" {
                        let content = std::fs::read_to_string(&trace_path)?;
                        kanso_infer::bench::parse_sharegpt_dataset(&content)?
                    } else {
                        kanso_infer::bench::parse_trace_file(&trace_path)?
                    };
                let max_tokens = trace
                    .iter()
                    .map(|t| t.prompt_len + t.gen_len)
                    .max()
                    .unwrap_or(prompt_len + gen_len);
                let mut engine = create_engine_for_model_profile(
                    &model_type,
                    backend,
                    trace.len(),
                    max_tokens,
                    block_size,
                    max_running,
                )?;
                let report =
                    kanso_infer::bench::run_trace_benchmark(&mut engine, &trace, !no_warmup)?;
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
                    let mut engine = create_engine_for_model_profile(
                        &model_type,
                        backend,
                        b,
                        max_tokens,
                        block_size,
                        max_running,
                    )?;
                    let report = kanso_infer::bench::run_synthetic_benchmark(
                        &mut engine,
                        b,
                        prompt_len,
                        gen_len,
                        !no_warmup,
                    )?;
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
