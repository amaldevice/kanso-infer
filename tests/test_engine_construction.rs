//! Ticket #85: engine construction reports a KV request it cannot serve as `Err`
//! (readable and actionable), instead of panicking — or silently building a cache
//! that can never hold a token.
//!
//! Seams under test:
//! - S1: the public constructor `Engine::new` / `Engine::new_with_budget`.
//! - The `kanso` binary's startup: `cli` and `serve` must print a user message and
//!   exit non-zero without a Rust panic when the KV geometry is impossible.

use candle_core::quantized::gguf_file::{self, Value};
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{DType, Device, Tensor};
use kanso_infer::engine::Engine;
use kanso_infer::model::Model;
use std::io::BufWriter;
use std::path::{Path, PathBuf};

fn mock() -> Model {
    Model::create_mock_tiny(16, 4, 2, 2, 50).expect("mock model")
}

/// Without blocks there is no KV to allocate: the engine can never admit a request.
#[test]
fn zero_total_blocks_is_rejected() {
    let err = Engine::new(mock(), 4, 0, 4).expect_err("a pool with zero blocks cannot serve");
    let msg = err.to_string();
    assert!(
        msg.contains("KV") && msg.contains('0'),
        "error must name the KV pool and the rejected value, got: {msg}"
    );
}

/// A block holds `block_size` tokens; zero slots per block is not representable.
#[test]
fn zero_block_size_is_rejected() {
    let err = Engine::new(mock(), 4, 8, 0).expect_err("block_size 0 holds no token");
    assert!(
        err.to_string().contains("KV"),
        "error must name the KV storage, got: {err}"
    );
}

/// The element count of the KV shape overflows `usize`. Today candle's shape product
/// panics; it must be reported as a configuration error instead.
#[test]
fn overflowing_total_blocks_is_rejected_not_panicked() {
    let err = Engine::new(mock(), 4, usize::MAX, 4096)
        .expect_err("an element count that overflows usize cannot be allocated");
    assert!(
        err.to_string().contains("KV"),
        "error must name the KV storage, got: {err}"
    );
}

/// A KV pool of ~1 PB cannot be allocated on any real machine (and does not overflow
/// `usize`), so it must be rejected up front rather than attempted.
#[test]
fn kv_allocation_beyond_physical_memory_is_rejected() {
    const BLOCKS: usize = 1_000_000_000_000;
    let err = Engine::new(mock(), 4, BLOCKS, 4)
        .expect_err("a petabyte of KV slots cannot be allocated");
    let msg = err.to_string();
    assert!(
        msg.contains("KV"),
        "error must name the KV storage, got: {msg}"
    );
}

/// Success path unchanged: a sane geometry builds, decodes `max_tokens` tokens and
/// returns every block to the pool.
#[test]
fn valid_geometry_still_builds_and_decodes() {
    let mut engine = Engine::new(mock(), 4, 8, 4).expect("valid geometry");
    engine.add_request(vec![1, 2, 3], 4);

    let mut emitted = 0usize;
    while engine.has_unfinished_requests() {
        emitted += engine.step().expect("decode step").len();
    }

    assert_eq!(emitted, 4, "engine must still emit max_tokens tokens");
    assert_eq!(
        engine.free_blocks_count(),
        8,
        "all blocks must return to the pool"
    );
}

/// AC #85: no `expect`/`unwrap`/`panic!` may hide in the KV allocation path. Audited as
/// source text, the same way the README honesty table is audited: the gate fails
/// loudly when a panic site creeps back in.
#[test]
fn kv_allocation_path_has_no_panic_sites() {
    for (file, signature) in [
        ("src/engine.rs", "pub fn new_with_budget("),
        ("src/kv_cache/storage.rs", "pub fn new("),
    ] {
        let body = function_body(file, signature);
        for panic_site in ["expect(", "unwrap(", "panic!("] {
            assert!(
                !body.contains(panic_site),
                "{file}::{signature} must not contain `{panic_site}` in the KV allocation path"
            );
        }
    }
}

/// The CLI's engine construction must map the failure to a user message and a clean
/// non-zero exit — never a panic. Driven with a GGUF that declares `head_count_kv = 0`.
#[test]
fn cli_reports_kv_failure_and_exits_nonzero() {
    let dir = temp_dir("cli-kv");
    let model = dir.join("kv0.gguf");
    write_synthetic_gguf(&model, 0);

    let output = std::process::Command::new(env!("CARGO_BIN_EXE_kanso"))
        .args([
            "cli",
            "--model",
            model.to_str().unwrap(),
            "--prompt",
            "hi",
            "--max-tokens",
            "4",
            "--backend",
            "cpu",
        ])
        .output()
        .expect("run kanso cli");

    assert_startup_failure(&output, "cli");
}

/// A block budget derived from `--max-tokens` can be impossible (it does not fit in the
/// address space); the CLI must reject it with a message, not an arithmetic panic.
#[test]
fn cli_rejects_impossible_block_budget_without_panicking() {
    let dir = temp_dir("cli-overflow");
    let model = dir.join("tiny.gguf");
    write_synthetic_gguf(&model, 2);

    let output = std::process::Command::new(env!("CARGO_BIN_EXE_kanso"))
        .args([
            "cli",
            "--model",
            model.to_str().unwrap(),
            "--prompt",
            "hi",
            "--max-tokens",
            "18446744073709551615",
            "--backend",
            "cpu",
        ])
        .output()
        .expect("run kanso cli");

    assert_startup_failure(&output, "cli");
}

/// The server startup path constructs the engine before binding; the same failure must
/// be reported and the process must exit non-zero without panicking.
#[test]
fn serve_reports_kv_failure_and_exits_nonzero() {
    let dir = temp_dir("serve-kv");
    let model = dir.join("kv0.gguf");
    write_synthetic_gguf(&model, 0);

    let output = std::process::Command::new(env!("CARGO_BIN_EXE_kanso"))
        .args([
            "serve",
            "--model",
            model.to_str().unwrap(),
            "--port",
            "0",
            "--backend",
            "cpu",
        ])
        .output()
        .expect("run kanso serve");

    assert_startup_failure(&output, "serve");
}

fn assert_startup_failure(output: &std::process::Output, command: &str) {
    assert!(
        !output.status.success(),
        "{command} must exit non-zero, got {:?}",
        output.status
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("KV"),
        "{command} stderr must name the unusable KV configuration, got:\n{stderr}"
    );
    assert!(
        !stderr.contains("panicked"),
        "{command} must not panic, got:\n{stderr}"
    );
}

/// Body of the first function whose definition line contains `signature`, by brace
/// balance (comments containing braces are not expected in these constructors).
fn function_body(file: &str, signature: &str) -> String {
    let source = std::fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(file))
        .unwrap_or_else(|e| panic!("read {file}: {e}"));
    let start = source
        .find(signature)
        .unwrap_or_else(|| panic!("{file} has no `{signature}`"));
    let tail = &source[start..];
    let mut depth = 0i32;
    let mut started = false;
    for (i, c) in tail.char_indices() {
        match c {
            '{' => {
                depth += 1;
                started = true;
            }
            '}' => {
                depth -= 1;
                if started && depth == 0 {
                    return tail[..=i].to_string();
                }
            }
            _ => {}
        }
    }
    panic!("unbalanced braces after `{signature}` in {file}");
}

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("kanso-85-{tag}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

/// Minimal loadable llama GGUF: `head_count_kv = 0` produces KV tensors with no heads,
/// which no KV storage can serve. Shape keys and tensors match the loader's contract.
fn write_synthetic_gguf(path: &Path, kv_heads: usize) {
    use rand::{Rng, SeedableRng};
    const HIDDEN: usize = 64;
    const Q_HEADS: usize = 4;
    const INTER: usize = 128;
    const VOCAB: usize = 256;
    const LAYERS: usize = 2;

    let mut rng = rand::rngs::StdRng::seed_from_u64(0x85_5EED);
    let dev = Device::Cpu;
    let head_dim = HIDDEN / Q_HEADS;
    let rand_tensor = |rng: &mut rand::rngs::StdRng, rows: usize, cols: usize| -> Tensor {
        let data: Vec<f32> = (0..rows * cols)
            .map(|_| (rng.random::<f32>() * 2.0 - 1.0) * 0.05)
            .collect();
        Tensor::from_vec(data, (rows, cols), &dev).unwrap()
    };

    let mut tensors: Vec<(String, QTensor)> = vec![
        (
            "token_embd.weight".to_string(),
            QTensor::quantize(&rand_tensor(&mut rng, VOCAB, HIDDEN), GgmlDType::F32).unwrap(),
        ),
        (
            "output_norm.weight".to_string(),
            QTensor::quantize(
                &Tensor::ones(HIDDEN, DType::F32, &dev).unwrap(),
                GgmlDType::F32,
            )
            .unwrap(),
        ),
    ];
    for i in 0..LAYERS {
        for (name, tensor, is_norm) in [
            (
                format!("blk.{i}.attn_norm.weight"),
                Tensor::ones(HIDDEN, DType::F32, &dev).unwrap(),
                true,
            ),
            (
                format!("blk.{i}.attn_q.weight"),
                rand_tensor(&mut rng, Q_HEADS * head_dim, HIDDEN),
                false,
            ),
            (
                format!("blk.{i}.attn_k.weight"),
                rand_tensor(&mut rng, kv_heads * head_dim, HIDDEN),
                false,
            ),
            (
                format!("blk.{i}.attn_v.weight"),
                rand_tensor(&mut rng, kv_heads * head_dim, HIDDEN),
                false,
            ),
            (
                format!("blk.{i}.attn_output.weight"),
                rand_tensor(&mut rng, HIDDEN, Q_HEADS * head_dim),
                false,
            ),
            (
                format!("blk.{i}.ffn_norm.weight"),
                Tensor::ones(HIDDEN, DType::F32, &dev).unwrap(),
                true,
            ),
            (
                format!("blk.{i}.ffn_gate.weight"),
                rand_tensor(&mut rng, INTER, HIDDEN),
                false,
            ),
            (
                format!("blk.{i}.ffn_up.weight"),
                rand_tensor(&mut rng, INTER, HIDDEN),
                false,
            ),
            (
                format!("blk.{i}.ffn_down.weight"),
                rand_tensor(&mut rng, HIDDEN, INTER),
                false,
            ),
        ] {
            let dtype = if is_norm { GgmlDType::F32 } else { GgmlDType::F32 };
            tensors.push((name, QTensor::quantize(&tensor, dtype).unwrap()));
        }
    }

    let meta = [
        ("general.architecture", Value::String("llama".to_string())),
        ("llama.block_count", Value::U32(LAYERS as u32)),
        ("llama.attention.head_count", Value::U32(Q_HEADS as u32)),
        ("llama.attention.head_count_kv", Value::U32(kv_heads as u32)),
        ("llama.embedding_length", Value::U32(HIDDEN as u32)),
        ("llama.feed_forward_length", Value::U32(INTER as u32)),
        ("llama.rope.freq_base", Value::F32(10000.0)),
        (
            "llama.attention.layer_norm_rms_epsilon",
            Value::F32(1e-5),
        ),
    ];
    let meta_refs: Vec<(&str, &Value)> = meta.iter().map(|(k, v)| (*k, v)).collect();
    let tensor_refs: Vec<(&str, &QTensor)> = tensors.iter().map(|(n, t)| (n.as_str(), t)).collect();

    let file = std::fs::File::create(path).expect("create synthetic gguf");
    gguf_file::write(&mut BufWriter::new(file), &meta_refs, &tensor_refs)
        .expect("write synthetic gguf");
}
