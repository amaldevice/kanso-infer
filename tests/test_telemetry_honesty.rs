//! RED tests for ticket #53 (telemetry honesty).
//!
//! 1. `report_shows_derived_bytes`: `bytes_read_per_step` must equal the real
//!    model's resident weight bytes plus KV bytes for the sequence context —
//!    not the `380_000_000` literal in `src/bin/bench.rs`.
//! 2. `no_magic_peak_constants`: the peak bandwidth behind the bus-util figure
//!    must be a labeled measurement, not the `150.0` literal.
//! 3. `same_seed_same_tokens`: sampling with the same seed replays the same
//!    token (Fase-2 F9 threaded the seed via `StdRng`; this pins it), at both
//!    the sampler and the engine level.
//! 4. `ane_claim_requires_real_dispatch`: `has_ane_support()` must stay false
//!    while `CoreMlAneAdapter` has no Core ML execution path — the old
//!    chip-name string match claimed ANE on every Mac.

use candle_core::quantized::gguf_file::{self, Value};
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{DType, Device, Tensor};
use kanso_infer::bench::run_synthetic_benchmark;
use kanso_infer::engine::Engine;
use kanso_infer::model::Model;
use kanso_infer::sampler::sample_logits;
use kanso_infer::scheduler::SamplingParams;
use rand::{Rng, SeedableRng};
use std::io::BufWriter;
use std::path::Path;
fn tiny_engine() -> (Engine, usize, usize) {
    let model = Model::create_mock_tiny(16, 4, 2, 2, 50).expect("mock model");
    let weights_bytes = model.weights.storage_bytes();
    let kv_per_token = model.config.kv_cache_bytes_per_token_f32();
    let engine = Engine::new(model, 4, 30, 4);
    (engine, weights_bytes, kv_per_token)
}

#[test]
fn report_shows_derived_bytes() {
    let (mut engine, weights_bytes, kv_per_token) = tiny_engine();
    let (prompt_len, gen_len) = (4, 4);
    let batch_size = 2;
    let report = run_synthetic_benchmark(&mut engine, batch_size, prompt_len, gen_len, false)
        .expect("benchmark");
    // #62 F5: KV bytes sum over all active sequences (batch_size * (prompt + gen))
    let total_ctx = batch_size * (prompt_len + gen_len);
    let expected = weights_bytes + kv_per_token * total_ctx;
    assert_eq!(
        report.bytes_read_per_step, expected,
        "bytes_read_per_step must be derived from the weighed model \
         (weights {weights_bytes} + KV {kv_per_token}/tok x {total_ctx} sum-ctx across batch {batch_size}), not a literal",
    );
    assert!(report.decode_ms_per_step > 0.0);
    assert!(report.decode_ms_per_token > 0.0);
}

#[test]
fn no_magic_peak_constants() {
    let (mut engine, _, _) = tiny_engine();
    let report = run_synthetic_benchmark(&mut engine, 2, 4, 4, false).expect("benchmark");
    assert!(
        report.peak_bandwidth_gb_s > 0.0,
        "peak bandwidth must be measured, got {}",
        report.peak_bandwidth_gb_s
    );
    assert!(
        report.peak_bandwidth_source.contains("measured"),
        "peak source must say how it was obtained, got {:?}",
        report.peak_bandwidth_source
    );
}

#[test]
fn same_seed_same_tokens_sampler() {
    let logits = vec![0.5, 2.0, 1.0, 0.1, 1.5, 0.8];
    for seed in [7u64, 42, 12345] {
        let a = sample_logits(&logits, 1.0, 0.95, seed).expect("sample");
        let b = sample_logits(&logits, 1.0, 0.95, seed).expect("sample");
        assert_eq!(a, b, "same seed {seed} must replay the same token");
    }
}

#[test]
fn same_seed_same_tokens_engine() {
    fn run(seed: u64) -> Vec<u32> {
        let model = Model::create_mock_tiny(16, 4, 2, 2, 50).expect("mock model");
        let mut engine = Engine::new(model, 4, 30, 4);
        engine.add_request_with_sampling(
            vec![1, 2, 3],
            5,
            Vec::new(),
            SamplingParams {
                temperature: 1.0,
                top_p: 1.0,
                seed,
            },
        );
        for _ in 0..32 {
            if !engine.has_unfinished_requests() {
                break;
            }
            engine.step().expect("step");
        }
        assert!(
            !engine.has_unfinished_requests(),
            "engine should finish within 32 steps"
        );
        engine.sequences[0].generated_tokens.clone()
    }
    assert_eq!(run(99), run(99), "same seed must generate the same tokens");
}

#[test]
fn ane_claim_requires_real_dispatch() {
    assert!(
        !kanso_infer::ffn_offload::has_ane_support(),
        "no Core ML execution path exists yet; claiming ANE support lies"
    );
}

fn write_synthetic_q8_gguf(path: &Path) {
    const HIDDEN: usize = 64;
    const Q_HEADS: usize = 4;
    const KV_HEADS: usize = 2;
    const INTER: usize = 128;
    const VOCAB: usize = 256;
    const LAYERS: usize = 2;

    let mut rng = rand::rngs::StdRng::seed_from_u64(0x62F8_5EED);
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
            QTensor::quantize(&rand_tensor(&mut rng, VOCAB, HIDDEN), GgmlDType::Q8_0).unwrap(),
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
        for (name, t, is_norm) in [
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
                rand_tensor(&mut rng, KV_HEADS * head_dim, HIDDEN),
                false,
            ),
            (
                format!("blk.{i}.attn_v.weight"),
                rand_tensor(&mut rng, KV_HEADS * head_dim, HIDDEN),
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
            let dtype = if is_norm {
                GgmlDType::F32
            } else {
                GgmlDType::Q8_0
            };
            tensors.push((name, QTensor::quantize(&t, dtype).unwrap()));
        }
    }
    let meta = [
        ("general.architecture", Value::String("llama".to_string())),
        ("llama.block_count", Value::U32(LAYERS as u32)),
        ("llama.attention.head_count", Value::U32(Q_HEADS as u32)),
        ("llama.attention.head_count_kv", Value::U32(KV_HEADS as u32)),
        ("llama.embedding_length", Value::U32(HIDDEN as u32)),
        ("llama.feed_forward_length", Value::U32(INTER as u32)),
        ("llama.rope.freq_base", Value::F32(10000.0)),
        ("llama.attention.layer_norm_rms_epsilon", Value::F32(1e-5)),
    ];
    let meta_refs: Vec<(&str, &Value)> = meta.iter().map(|(k, v)| (*k, v)).collect();
    let tensor_refs: Vec<(&str, &QTensor)> = tensors.iter().map(|(n, t)| (n.as_str(), t)).collect();
    let file = std::fs::File::create(path).expect("create synthetic gguf");
    gguf_file::write(&mut BufWriter::new(file), &meta_refs, &tensor_refs)
        .expect("write synthetic gguf");
}

#[test]
fn report_weighs_real_gguf_quantized_storage() {
    let path = std::env::temp_dir().join("kanso-62-f8-q8.gguf");
    let _ = std::fs::remove_file(&path);
    write_synthetic_q8_gguf(&path);

    let model = Model::load_gguf(&path, Device::Cpu).expect("load synthetic Q8_0 model");
    let storage_bytes = model.weights.storage_bytes();
    let param_count = model.weights.parameter_count();
    let f32_bytes = param_count * std::mem::size_of::<f32>();
    // Packed Q8_0 weights take far less memory than materialized F32
    assert!(
        storage_bytes < f32_bytes / 2,
        "Q8_0 storage {storage_bytes} must be packed (< half of F32 {f32_bytes})"
    );

    let kv_per_token = model.config.kv_cache_bytes_per_token_f32();
    let mut engine = Engine::new(model, 4, 64, 16);
    let (prompt_len, gen_len) = (4, 4);
    let batch_size = 2;
    let report = run_synthetic_benchmark(&mut engine, batch_size, prompt_len, gen_len, false)
        .expect("benchmark");

    let total_ctx = batch_size * (prompt_len + gen_len);
    let expected = storage_bytes + kv_per_token * total_ctx;
    assert_eq!(
        report.bytes_read_per_step, expected,
        "bytes_read_per_step must match real GGUF packed storage + KV cache"
    );
    let _ = std::fs::remove_file(&path);
}
