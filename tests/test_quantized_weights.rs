//! Ticket #47 acceptance: weights must stay in their GGUF storage format
//! (Q8_0 blocks) instead of being dequantized to F32 at load.
//!
//! No SmolLM2 GGUF is downloadable in this environment, so these tests write a
//! **synthetic** GGUF with the real SmolLM2-360M shapes (hidden 960, 15/3 heads,
//! 32 layers, intermediate 2560, vocab 49152) and load it back through the
//! production loader and decode loop. Storage format, block layout, loader and
//! matmul path are all real; only the weight values are random.
//!
//! Criteria exercised here:
//!   - peak RSS after loading Q8_0 weights < 600 MB (issue #47),
//!   - decode tok/s at batch 1 >= 2.5x the F32 (dequantized) baseline,
//!   - PPL of Q8_0 weights within 1% of the same weights unquantized.
//!
//! The last one is a same-engine proxy: the literal criterion ("within 1% of
//! llama.cpp for the same file") needs the real file, i.e.
//! `KANSO_GGUF=/path/SmolLM2-360M-Instruct-Q8_0.gguf` plus a llama.cpp
//! `perplexity` run to compare against; `test_real_gguf_when_available` is the
//! hook for it. The mechanism (quantized blocks, quantized matmul, per-row
//! embedding dequant) is fully covered below.
//!
//! Timing/memory criteria are only meaningful in release builds; in debug builds
//! the heavy SmolLM2-shaped cases short-circuit with a printed note.

use candle_core::quantized::gguf_file::{self, Value};
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{Device, Tensor};
use kanso_infer::bench::perplexity::{calculate_perplexity, get_bundled_wikitext2_tokens};
use kanso_infer::kv_cache::{BlockTable, KvStorage};
use kanso_infer::model::Model;
use kanso_infer::weights::ModelWeights;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Instant;

/// Fixture creation is serialized: two tests must not write the same file at once.
static FIXTURES: Mutex<()> = Mutex::new(());
/// The SmolLM2-shaped cases are serialized as well: each holds a 380 MB quantized
/// model plus a 1.4 GB dense baseline, and running them side by side would make the
/// RSS measurement meaningless.
static HEAVY: Mutex<()> = Mutex::new(());

// ---------------------------------------------------------------------------
// Synthetic GGUF fixture
// ---------------------------------------------------------------------------

/// SmolLM2-360M-Instruct: hidden 960, 15 q heads / 3 kv heads, 32 layers,
/// intermediate 2560, vocab 49152.
const HIDDEN: usize = 960;
const Q_HEADS: usize = 15;
const KV_HEADS: usize = 3;
const INTERMEDIATE: usize = 2560;
const VOCAB: usize = 49_152;
const LAYERS: usize = 32;

/// Small fixture used by the correctness tests (fast in debug builds).
const S_HIDDEN: usize = 64;
const S_Q_HEADS: usize = 4;
const S_KV_HEADS: usize = 2;
const S_INTERMEDIATE: usize = 128;
const S_VOCAB: usize = 256;
const S_LAYERS: usize = 2;

struct Dims {
    hidden: usize,
    q_heads: usize,
    kv_heads: usize,
    intermediate: usize,
    vocab: usize,
    layers: usize,
}

const SMALL: Dims = Dims {
    hidden: S_HIDDEN,
    q_heads: S_Q_HEADS,
    kv_heads: S_KV_HEADS,
    intermediate: S_INTERMEDIATE,
    vocab: S_VOCAB,
    layers: S_LAYERS,
};

const SMOLLM2: Dims = Dims {
    hidden: HIDDEN,
    q_heads: Q_HEADS,
    kv_heads: KV_HEADS,
    intermediate: INTERMEDIATE,
    vocab: VOCAB,
    layers: LAYERS,
};

fn rand_weight(rng: &mut StdRng, rows: usize, cols: usize, scale: f32) -> Tensor {
    let data: Vec<f32> = (0..rows * cols)
        .map(|_| (rng.random::<f32>() * 2.0 - 1.0) * scale)
        .collect();
    Tensor::from_vec(data, (rows, cols), &Device::Cpu).unwrap()
}

/// Writes a GGUF file with random weights at `dims`, stored as `dtype`.
///
/// `scale` keeps mock logits near unit scale, which is what makes a
/// quantization-fidelity PPL comparison meaningful (a trained model's logits are
/// O(1); uncalibrated N(0,1) mock weights are not).
fn write_synthetic_gguf(path: &Path, dims: &Dims, dtype: GgmlDType, scale: f32) {
    let dev = Device::Cpu;
    let head_dim = dims.hidden / dims.q_heads;
    let q_dim = dims.q_heads * head_dim;
    let kv_dim = dims.kv_heads * head_dim;
    let mut rng = StdRng::seed_from_u64(0x47_5EED);

    let mut tensors: Vec<(String, QTensor)> = Vec::new();

    let ones = |n: usize| Tensor::ones(n, candle_core::DType::F32, &dev).unwrap();
    tensors.push((
        "token_embd.weight".to_string(),
        QTensor::quantize(
            &rand_weight(&mut rng, dims.vocab, dims.hidden, scale),
            dtype,
        )
        .unwrap(),
    ));
    tensors.push((
        "output_norm.weight".to_string(),
        QTensor::quantize(&ones(dims.hidden), dtype).unwrap(),
    ));

    for i in 0..dims.layers {
        for (name, t) in [
            (format!("blk.{i}.attn_norm.weight"), ones(dims.hidden)),
            (
                format!("blk.{i}.attn_q.weight"),
                rand_weight(&mut rng, q_dim, dims.hidden, scale),
            ),
            (
                format!("blk.{i}.attn_k.weight"),
                rand_weight(&mut rng, kv_dim, dims.hidden, scale),
            ),
            (
                format!("blk.{i}.attn_v.weight"),
                rand_weight(&mut rng, kv_dim, dims.hidden, scale),
            ),
            (
                format!("blk.{i}.attn_output.weight"),
                rand_weight(&mut rng, dims.hidden, q_dim, scale),
            ),
            (format!("blk.{i}.ffn_norm.weight"), ones(dims.hidden)),
            (
                format!("blk.{i}.ffn_gate.weight"),
                rand_weight(&mut rng, dims.intermediate, dims.hidden, scale),
            ),
            (
                format!("blk.{i}.ffn_up.weight"),
                rand_weight(&mut rng, dims.intermediate, dims.hidden, scale),
            ),
            (
                format!("blk.{i}.ffn_down.weight"),
                rand_weight(&mut rng, dims.hidden, dims.intermediate, scale),
            ),
        ] {
            tensors.push((name, QTensor::quantize(&t, dtype).unwrap()));
        }
    }

    // Deliberately no "output.weight": lm_head stays tied to token_embd, which is
    // what SmolLM2 GGUFs do and what makes the RSS budget interesting.
    let metadata = [
        ("general.architecture", Value::String("llama".to_string())),
        ("llama.block_count", Value::U32(dims.layers as u32)),
        (
            "llama.attention.head_count",
            Value::U32(dims.q_heads as u32),
        ),
        (
            "llama.attention.head_count_kv",
            Value::U32(dims.kv_heads as u32),
        ),
        ("llama.embedding_length", Value::U32(dims.hidden as u32)),
        (
            "llama.feed_forward_length",
            Value::U32(dims.intermediate as u32),
        ),
        ("llama.rope.freq_base", Value::F32(10000.0)),
        ("llama.attention.layer_norm_rms_epsilon", Value::F32(1e-5)),
    ];
    let metadata: Vec<(&str, &Value)> = metadata.iter().map(|(k, v)| (*k, v)).collect();

    let refs: Vec<(&str, &QTensor)> = tensors.iter().map(|(name, t)| (name.as_str(), t)).collect();
    let file = std::fs::File::create(path).expect("create synthetic gguf");
    let mut writer = BufWriter::new(file);
    gguf_file::write(&mut writer, &metadata, &refs).expect("write synthetic gguf");
}

/// Cached fixture path; the file is written once and reused across tests.
fn fixture_path(name: &str, dims: &Dims, dtype: GgmlDType, scale: f32) -> PathBuf {
    let path = std::env::temp_dir().join(format!("kanso-47-{name}-{}L.gguf", dims.layers));
    let _guard = FIXTURES.lock().unwrap();
    if !path.is_file() {
        write_synthetic_gguf(&path, dims, dtype, scale);
    }
    path
}

/// Resident-set size of this process in bytes (macOS `ps`), or `None` if ps fails.
fn rss_bytes() -> Option<u64> {
    let pid = std::process::id().to_string();
    let out = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &pid])
        .output()
        .ok()?;
    let kb: u64 = String::from_utf8_lossy(&out.stdout).trim().parse().ok()?;
    Some(kb * 1024)
}

fn decode_tok_s(model: &Model, steps: usize) -> f64 {
    let mut kv_storage = KvStorage::new(
        model.config.num_hidden_layers,
        steps + 2,
        model.config.block_size,
        model.config.num_key_value_heads,
        model.config.head_dim(),
        &model.device,
    )
    .expect("kv storage");
    let mut block_table = BlockTable::new();
    for b in 0..(steps + 2).div_ceil(model.config.block_size) {
        block_table.push_block(b);
    }

    let _ = model
        .forward_decode_step(1, 0, &block_table, &mut kv_storage)
        .expect("warmup decode");
    block_table.set_num_tokens(1);

    let start = Instant::now();
    for pos in 1..=steps {
        let _ = model
            .forward_decode_step(7, pos, &block_table, &mut kv_storage)
            .expect("decode step");
        block_table.set_num_tokens(pos + 1);
    }
    steps as f64 / start.elapsed().as_secs_f64()
}

// ---------------------------------------------------------------------------
// #47 criterion 1: peak RSS after loading Q8_0 weights
// ---------------------------------------------------------------------------

#[test]
fn test_q8_0_gguf_weights_stay_quantized_and_rss_under_600mb() {
    let _serial = HEAVY.lock().unwrap();
    if cfg!(debug_assertions) {
        eprintln!("SKIP (debug build): SmolLM2-shaped RSS criterion needs --release");
        return;
    }

    let path = fixture_path("q8_0", &SMOLLM2, GgmlDType::Q8_0, 0.05);
    let model = Model::load_gguf(&path, Device::Cpu).expect("load synthetic Q8_0 gguf");

    let params = model.weights.parameter_count();
    let stored = model.weights.storage_bytes();
    let f32_bytes = params * std::mem::size_of::<f32>();
    let rss = rss_bytes().expect("read rss");
    println!(
        "Q8_0 load: params {params}, stored {:.1} MB ({:.2} B/param), f32 would be {:.1} MB, RSS {:.1} MB",
        stored as f64 / 1e6,
        stored as f64 / params as f64,
        f32_bytes as f64 / 1e6,
        rss as f64 / 1e6,
    );

    assert!(
        stored * 3 < f32_bytes,
        "weights must stay quantized: {stored} bytes stored vs {f32_bytes} bytes for F32"
    );
    assert!(
        rss < 600 * 1024 * 1024,
        "RSS after loading SmolLM2-shaped Q8_0 weights must be < 600 MB, got {:.1} MB",
        rss as f64 / 1e6
    );
}

// ---------------------------------------------------------------------------
// #47 criterion 2: decode tok/s >= 2.5x the F32 baseline
// ---------------------------------------------------------------------------

#[test]
fn test_q8_0_decode_tok_s_at_smollm2_dimensions_beats_f32() {
    let _serial = HEAVY.lock().unwrap();
    if cfg!(debug_assertions) {
        eprintln!("SKIP (debug build): decode tok/s criterion needs --release");
        return;
    }

    let path = fixture_path("q8_0", &SMOLLM2, GgmlDType::Q8_0, 0.05);
    let model = Model::load_gguf(&path, Device::Cpu).expect("load synthetic Q8_0 gguf");

    // The old (F32) behaviour: dequantize everything at load.
    let dense_weights = model.weights.to_dense().expect("dense copy");
    let dense_model = Model::new(dense_weights, model.config.clone(), Device::Cpu);

    let q_tok_s = decode_tok_s(&model, 16);
    let f32_tok_s = decode_tok_s(&dense_model, 16);
    println!("decode tok/s @ SmolLM2-360M dims: q8_0={q_tok_s:.2}, f32={f32_tok_s:.2}");

    assert!(
        q_tok_s >= 2.5 * f32_tok_s,
        "Q8_0 decode {q_tok_s:.2} tok/s must be >= 2.5x the F32 baseline {f32_tok_s:.2} tok/s"
    );
}

// ---------------------------------------------------------------------------
// #47 criterion 3: PPL within 1% (same-engine proxy for the llama.cpp check)
// ---------------------------------------------------------------------------

/// PPL of Q8_0-stored weights vs the identical unquantized weights.
///
/// This is the offline proxy for "PPL within 1% of llama.cpp for the same file":
/// it isolates the one variable #47 changes (storage format) on identical values.
#[test]
fn test_q8_0_ppl_within_one_percent_of_unquantized() {
    let scale = 1.0 / (S_HIDDEN as f32).sqrt();
    let f32_path = fixture_path("f32-small", &SMALL, GgmlDType::F32, scale);
    let q8_path = fixture_path("q8_0-small", &SMALL, GgmlDType::Q8_0, scale);

    let f32_model = Model::load_gguf(&f32_path, Device::Cpu).expect("load f32 fixture");
    let q8_model = Model::load_gguf(&q8_path, Device::Cpu).expect("load q8_0 fixture");

    let tokens = get_bundled_wikitext2_tokens();
    let tokens = &tokens[..64.min(tokens.len())];
    let ppl_f32 = calculate_perplexity(&f32_model, tokens, 32).expect("f32 ppl");
    let ppl_q8 = calculate_perplexity(&q8_model, tokens, 32).expect("q8_0 ppl");
    let delta = (ppl_q8 - ppl_f32).abs() / ppl_f32;
    println!(
        "PPL (unquantized) {ppl_f32:.4} vs Q8_0 {ppl_q8:.4}: delta {:.4}%",
        delta * 100.0
    );

    assert!(
        delta < 0.01,
        "Q8_0 PPL {ppl_q8:.4} differs from unquantized {ppl_f32:.4} by {:.3}% (>1%)",
        delta * 100.0
    );
}

/// Logits parity between quantized-resident weights and the dense dequantization
/// of those exact same weights: the matmul path must do real quantized math.
///
/// The quantized kernel also rounds the *activations* to 8 bits (same as
/// llama.cpp's `mul_mat`), so per-logit deviation from a dense matmul is ~1%.
/// What must not move is the prediction: same argmax, and no order-of-magnitude
/// drift (which is what a broken block layout or head permutation would produce).
#[test]
fn test_quantized_matmul_matches_dense_reference_logits() {
    let scale = 1.0 / (S_HIDDEN as f32).sqrt();
    let q8_path = fixture_path("q8_0-small", &SMALL, GgmlDType::Q8_0, scale);
    let model = Model::load_gguf(&q8_path, Device::Cpu).expect("load q8_0 fixture");
    let dense_model = Model::new(
        model.weights.to_dense().expect("dense copy"),
        model.config.clone(),
        Device::Cpu,
    );

    let tokens: Vec<u32> = (0..16).map(|i| (i * 7) % S_VOCAB as u32).collect();
    let q = model.forward_tokens(&tokens).expect("quantized forward");
    let d = dense_model.forward_tokens(&tokens).expect("dense forward");

    let qv = q.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let dv = d.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(qv.len(), dv.len());

    let max_abs = dv.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1.0);
    let max_diff = qv
        .iter()
        .zip(dv.iter())
        .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
    let mean_diff = qv
        .iter()
        .zip(dv.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / qv.len() as f32;
    println!(
        "logits: max|dense| {max_abs:.4}, max|diff| {max_diff:.6} ({:.2}%), mean|diff| {mean_diff:.6}",
        100.0 * max_diff / max_abs
    );
    assert!(
        mean_diff <= 0.01 * max_abs,
        "mean quantized logit deviation {mean_diff} exceeds 1% of the logit scale {max_abs}"
    );
    assert!(
        max_diff <= 0.05 * max_abs,
        "quantized logits drift from the dense reference: {max_diff} vs scale {max_abs}"
    );

    // Prediction invariance: the same weights in the same order must pick the
    // same token, both for the last position of the chunk and over all positions.
    let vocab = S_VOCAB;
    let (_, last) =
        qv[qv.len() - vocab..]
            .iter()
            .enumerate()
            .fold(
                (0.0f32, 0usize),
                |(bm, bi), (i, &v)| {
                    if v > bm { (v, i) } else { (bm, bi) }
                },
            );
    let (_, dense_last) =
        dv[dv.len() - vocab..]
            .iter()
            .enumerate()
            .fold(
                (0.0f32, 0usize),
                |(bm, bi), (i, &v)| {
                    if v > bm { (v, i) } else { (bm, bi) }
                },
            );
    assert_eq!(
        last, dense_last,
        "quantized weights changed the predicted token: {last} vs {dense_last}"
    );
}

/// Embedding lookup must dequantize per row and agree with the dense lookup.
#[test]
fn test_quantized_embedding_lookup_matches_dense() {
    let scale = 1.0 / (S_HIDDEN as f32).sqrt();
    let q8_path = fixture_path("q8_0-small", &SMALL, GgmlDType::Q8_0, scale);
    let model = Model::load_gguf(&q8_path, Device::Cpu).expect("load q8_0 fixture");
    let dense = model.weights.to_dense().expect("dense copy");

    let ids = Tensor::from_slice(&[3u32, 17, 200], (3,), &Device::Cpu).unwrap();
    let q = model
        .weights
        .embed_tokens
        .embed(&ids)
        .expect("quantized embed");
    let d = dense.embed_tokens.embed(&ids).expect("dense embed");
    assert_eq!(q.dims(), d.dims());

    let diff = (q - d)
        .unwrap()
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    println!("embedding max|diff| {diff:.6}");
    assert!(
        diff == 0.0,
        "per-row dequant must reproduce the dense values exactly, diff {diff}"
    );
}

/// The numeric format of a GGUF tensor must survive the loader: an F16 file must
/// not be blown up to F32 at load (issue #47 also asks for an F16 minimum).
#[test]
fn test_f16_gguf_stays_16_bit_and_matches_f32_logits() {
    let scale = 1.0 / (S_HIDDEN as f32).sqrt();
    let path = fixture_path("f16-small", &SMALL, GgmlDType::F16, scale);
    let f32_path = fixture_path("f32-small", &SMALL, GgmlDType::F32, scale);
    let model = Model::load_gguf(&path, Device::Cpu).expect("load f16 fixture");
    let f32_model = Model::load_gguf(&f32_path, Device::Cpu).expect("load f32 fixture");

    let stored = model.weights.storage_bytes();
    let params = model.weights.parameter_count();
    println!(
        "F16 fixture: {stored} bytes for {params} params ({:.2} B/param)",
        stored as f64 / params as f64
    );
    assert!(
        stored <= params * 2,
        "F16 weights must stay 16-bit, got {stored} bytes for {params} params"
    );

    let tokens: Vec<u32> = (0..8).map(|i| (i * 13) % S_VOCAB as u32).collect();
    let a = model.forward_tokens(&tokens).expect("f16 forward");
    let b = f32_model.forward_tokens(&tokens).expect("f32 forward");
    let av = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let bv = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let scale_abs = bv.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let max_diff = av
        .iter()
        .zip(bv.iter())
        .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()));
    println!("f16 vs f32 logits: max|diff| {max_diff:.6} (scale {scale_abs:.4})");
    assert!(
        max_diff <= 1e-2 * scale_abs.max(1.0),
        "F16 forward diverges from F32: {max_diff} vs scale {scale_abs}"
    );
}

/// Cross-check hook for the real thing: when a real SmolLM2 Q8_0 GGUF is present
/// (`KANSO_GGUF=...`), load it and assert it stays quantized at SmolLM2 scale.
#[test]
fn test_real_gguf_when_available() {
    let Ok(path) = std::env::var("KANSO_GGUF") else {
        eprintln!("SKIP: set KANSO_GGUF=/path/to/SmolLM2-360M-Q8_0.gguf to check the real file");
        return;
    };
    let model = Model::load_gguf(&path, Device::Cpu).expect("load real gguf");
    let params = model.weights.parameter_count();
    let stored = model.weights.storage_bytes();
    println!(
        "real GGUF: {params} params, {:.1} MB resident ({:.2} B/param), RSS {:.1} MB",
        stored as f64 / 1e6,
        stored as f64 / params as f64,
        rss_bytes().unwrap_or(0) as f64 / 1e6
    );
    assert!(
        stored * 3 < params * 4,
        "real GGUF weights were expanded to F32-sized storage"
    );
}

/// Metal path: quantized weights must move to the GPU device and still compute
/// (candle dequantizes inside the GEMV kernels, so no F32 expansion).
#[test]
fn test_quantized_weights_run_on_metal() {
    let scale = 1.0 / (S_HIDDEN as f32).sqrt();
    let q8_path = fixture_path("q8_0-small", &SMALL, GgmlDType::Q8_0, scale);
    let cpu_model = Model::load_gguf(&q8_path, Device::Cpu).expect("load q8_0 fixture");

    let Ok(metal) = Device::new_metal(0) else {
        eprintln!("SKIP: no Metal device available");
        return;
    };
    let mut model = Model::new(
        cpu_model.weights.clone(),
        cpu_model.config.clone(),
        Device::Cpu,
    );
    model
        .to_device(&metal)
        .expect("move quantized weights to metal");
    assert!(matches!(model.device, Device::Metal(_)));

    let tokens: Vec<u32> = (0..8).map(|i| (i * 5) % S_VOCAB as u32).collect();
    let metal_logits = model
        .forward_tokens(&tokens)
        .expect("metal forward")
        .to_device(&Device::Cpu)
        .unwrap();
    let cpu_logits = cpu_model.forward_tokens(&tokens).expect("cpu forward");

    let m = metal_logits
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let c = cpu_logits.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let max_abs = c.iter().fold(0.0f32, |a, v| a.max(v.abs()));
    let max_diff = m
        .iter()
        .zip(c.iter())
        .fold(0.0f32, |a, (x, y)| a.max((x - y).abs()));
    println!("metal vs cpu logits: max|diff| {max_diff:.6} (scale {max_abs:.4})");
    assert!(
        max_diff <= 0.05 * max_abs.max(1.0),
        "metal quantized forward diverges from cpu: {max_diff} vs scale {max_abs}"
    );
}

/// Guard against the loader silently dropping back to F32: inspect the type of a
/// loaded projection weight directly.
#[test]
fn test_loaded_projection_is_quantized_not_dense() {
    let scale = 1.0 / (S_HIDDEN as f32).sqrt();
    let q8_path = fixture_path("q8_0-small", &SMALL, GgmlDType::Q8_0, scale);
    let (weights, _config) =
        ModelWeights::load_from_gguf(&q8_path, &Device::Cpu).expect("load weights");
    let layer = &weights.layers[0];
    assert!(
        layer.attn_q.is_quantized(),
        "attn_q must be stored as quantized blocks after load"
    );
    assert!(
        layer.ffn_down.is_quantized(),
        "ffn_down must be stored as quantized blocks after load"
    );
    assert!(weights.embed_tokens.is_quantized());
}
