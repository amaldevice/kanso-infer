//! Ticket #54 (Fase-3 honesty) RED tests: loader arch gate, Granite scales, README audit.
//!
//! All three fail on the pre-fix code:
//! - foreign arch loads as llama instead of erroring,
//! - Granite scale keys are ignored so scaled/unscaled files decode identically,
//! - README has no Feature Status table to audit.

use candle_core::quantized::gguf_file::{self, Value};
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{DType, Device, Tensor};
use kanso_infer::model::Model;
use kanso_infer::weights::ModelWeights;
use std::io::BufWriter;
use std::path::{Path, PathBuf};

const HIDDEN: usize = 64;
const Q_HEADS: usize = 4;
const KV_HEADS: usize = 2;
const INTER: usize = 128;
const VOCAB: usize = 256;
const LAYERS: usize = 2;

fn tiny_tensor(rng: &mut rand::rngs::StdRng, rows: usize, cols: usize) -> Tensor {
    use rand::{Rng, SeedableRng};
    let data: Vec<f32> = (0..rows * cols)
        .map(|_| (rng.random::<f32>() * 2.0 - 1.0) * 0.05)
        .collect();
    Tensor::from_vec(data, (rows, cols), &Device::Cpu).unwrap()
}

/// Minimal synthetic GGUF: `arch` goes in `general.architecture`, shape keys under `prefix`.
fn write_gguf(path: &Path, arch: &str, prefix: &str, extra_meta: &[(String, Value)]) {
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(0x54_5EED);
    let dev = Device::Cpu;
    let head_dim = HIDDEN / Q_HEADS;
    let mut tensors: Vec<(String, QTensor)> = vec![
        (
            "token_embd.weight".to_string(),
            QTensor::quantize(&tiny_tensor(&mut rng, VOCAB, HIDDEN), GgmlDType::F32).unwrap(),
        ),
        (
            "output_norm.weight".to_string(),
            QTensor::quantize(&Tensor::ones(HIDDEN, DType::F32, &dev).unwrap(), GgmlDType::F32)
                .unwrap(),
        ),
    ];
    for i in 0..LAYERS {
        for (name, t) in [
            (
                format!("blk.{i}.attn_norm.weight"),
                Tensor::ones(HIDDEN, DType::F32, &dev).unwrap(),
            ),
            (
                format!("blk.{i}.attn_q.weight"),
                tiny_tensor(&mut rng, Q_HEADS * head_dim, HIDDEN),
            ),
            (
                format!("blk.{i}.attn_k.weight"),
                tiny_tensor(&mut rng, KV_HEADS * head_dim, HIDDEN),
            ),
            (
                format!("blk.{i}.attn_v.weight"),
                tiny_tensor(&mut rng, KV_HEADS * head_dim, HIDDEN),
            ),
            (
                format!("blk.{i}.attn_output.weight"),
                tiny_tensor(&mut rng, HIDDEN, Q_HEADS * head_dim),
            ),
            (
                format!("blk.{i}.ffn_norm.weight"),
                Tensor::ones(HIDDEN, DType::F32, &dev).unwrap(),
            ),
            (
                format!("blk.{i}.ffn_gate.weight"),
                tiny_tensor(&mut rng, INTER, HIDDEN),
            ),
            (
                format!("blk.{i}.ffn_up.weight"),
                tiny_tensor(&mut rng, INTER, HIDDEN),
            ),
            (
                format!("blk.{i}.ffn_down.weight"),
                tiny_tensor(&mut rng, HIDDEN, INTER),
            ),
        ] {
            tensors.push((name, QTensor::quantize(&t, GgmlDType::F32).unwrap()));
        }
    }
    let mut meta: Vec<(String, Value)> = vec![
        ("general.architecture".to_string(), Value::String(arch.to_string())),
        (format!("{prefix}.block_count"), Value::U32(LAYERS as u32)),
        (format!("{prefix}.attention.head_count"), Value::U32(Q_HEADS as u32)),
        (format!("{prefix}.attention.head_count_kv"), Value::U32(KV_HEADS as u32)),
        (format!("{prefix}.embedding_length"), Value::U32(HIDDEN as u32)),
        (format!("{prefix}.feed_forward_length"), Value::U32(INTER as u32)),
        (format!("{prefix}.rope.freq_base"), Value::F32(10000.0)),
        (format!("{prefix}.attention.layer_norm_rms_epsilon"), Value::F32(1e-5)),
    ];
    for (k, v) in extra_meta {
        meta.push((k.clone(), v.clone()));
    }
    let meta_refs: Vec<(&str, &Value)> = meta.iter().map(|(k, v)| (k.as_str(), v)).collect();
    let tensor_refs: Vec<(&str, &QTensor)> =
        tensors.iter().map(|(n, t)| (n.as_str(), t)).collect();
    let file = std::fs::File::create(path).expect("create synthetic gguf");
    gguf_file::write(&mut BufWriter::new(file), &meta_refs, &tensor_refs)
        .expect("write synthetic gguf");
}

fn tmp(name: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("kanso-54-{name}.gguf"));
    let _ = std::fs::remove_file(&p);
    p
}

#[test]
fn loader_rejects_foreign_arch() {
    let path = tmp("foreign");
    write_gguf(&path, "mamba", "llama", &[]);
    let err = match ModelWeights::load_from_gguf(&path, &Device::Cpu) {
        Ok(_) => panic!("foreign arch must not load as llama"),
        Err(e) => e,
    };
    assert!(
        err.to_string().contains("unsupported"),
        "error must say unsupported, got: {err}"
    );
    assert!(err.to_string().contains("mamba"), "error must name the arch, got: {err}");
}

#[test]
fn granite_embedding_scale_changes_logits() {
    let plain = tmp("granite-plain");
    write_gguf(&plain, "granite", "granite", &[]);
    let scaled = tmp("granite-scaled");
    write_gguf(
        &scaled,
        "granite",
        "granite",
        &[("granite.embedding_scale".to_string(), Value::F32(2.0))],
    );
    let load = |p: &Path| {
        let (w, c) = ModelWeights::load_from_gguf(p, &Device::Cpu).expect("load granite");
        Model::new(w, c, Device::Cpu)
    };
    let a = load(&plain).forward_tokens(&[1, 2, 3]).unwrap();
    let b = load(&scaled).forward_tokens(&[1, 2, 3]).unwrap();
    let (va, vb) = (
        a.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        b.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
    );
    let max_diff = va.iter().zip(&vb).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
    assert!(
        max_diff > 1e-3,
        "granite.embedding_scale=2.0 must move logits, max diff {max_diff}"
    );
}

#[test]
fn readme_feature_status_audit() {
    let readme = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/README.md"))
        .expect("read README");
    let section = readme
        .split("## Feature Status")
        .nth(1)
        .expect("README must have a ## Feature Status table")
        .split("## ")
        .next()
        .unwrap();
    let rows: Vec<&str> = section
        .lines()
        .map(str::trim)
        .filter(|l| l.starts_with('|') && !l.contains("---") && !l.starts_with("| Feature"))
        .collect();
    assert!(!rows.is_empty(), "Feature Status table must have rows");
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for row in rows {
        let cells: Vec<&str> = row.split('|').map(str::trim).collect();
        // status cell is index 1, pointer cell is second-to-last (leading/trailing `|` split empty).
        let status = cells.get(2).copied().unwrap_or("");
        let pointer = cells.get(cells.len() - 2).copied().unwrap_or("");
        // Planned rows point at the roadmap instead of code.
        if status.contains("📝") {
            assert!(
                pointer.contains("roadmap"),
                "planned row must point at the roadmap: {row}"
            );
            continue;
        }
        let mut found = false;
        let mut start = 0;
        while let Some(i) = pointer[start..].find("src/").or_else(|| pointer[start..].find("tests/")).or_else(|| pointer[start..].find("docs/")) {
            let abs = start + i;
            let end = pointer[abs..]
                .find(|c: char| c == '`' || c == ' ' || c == ',' || c == ')')
                .map(|e| abs + e)
                .unwrap_or(pointer.len());
            if root.join(&pointer[abs..end]).exists() {
                found = true;
                break;
            }
            start = end;
        }
        assert!(
            found,
            "feature row has no existing exec pointer: {row}"
        );
    }
}
