//! Golden-logits test for #57: full prefill on the fixed-seed random mock.
//! Snapshots top-k token ids + full logits checksum so numeric regressions
//! (masking, RoPE, gather, matmul) fail loudly. Fixture lives in
//! `tests/fixtures/golden_logits.json`; regenerate only deliberately via
//! `UPDATE_GOLDEN=1 cargo test --test test_golden_logits`.

use std::collections::HashMap;
use std::path::PathBuf;

use kanso_infer::model::Model;

const FIXTURE: &str = "tests/fixtures/golden_logits.json";
const TOP_K: usize = 5;
const TOL: f32 = 1e-4;

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(FIXTURE)
}

fn mock() -> Model {
    Model::create_mock_tiny(16, 4, 2, 2, 50).unwrap()
}

fn prompts() -> Vec<Vec<u32>> {
    vec![vec![1, 2, 3, 4], vec![7, 7, 7, 7], vec![0, 49, 25, 13, 42]]
}

fn logits_of(model: &Model, prompt: &[u32]) -> Vec<f32> {
    model
        .forward_single_prompt(prompt)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
}

fn top_k(logits: &[f32], k: usize) -> Vec<u32> {
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    idx.into_iter().take(k).map(|i| i as u32).collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn test_golden_logits_full_prefill() {
    let model = mock();
    let mut current: HashMap<String, Vec<f32>> = HashMap::new();
    for p in prompts() {
        let key = p
            .iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join(",");
        current.insert(key, logits_of(&model, &p));
    }

    let path = fixture_path();
    if std::env::var("UPDATE_GOLDEN").is_ok() {
        let json = serde_json::to_string_pretty(&current).unwrap();
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, json).unwrap();
        return;
    }

    let raw = std::fs::read_to_string(&path)
        .expect("golden fixture missing; run with UPDATE_GOLDEN=1 once");
    let expected: HashMap<String, Vec<f32>> = serde_json::from_str(&raw).unwrap();
    assert_eq!(
        current.len(),
        expected.len(),
        "prompt set changed; regenerate fixture deliberately"
    );

    for (key, cur) in &current {
        let exp = expected
            .get(key)
            .unwrap_or_else(|| panic!("no golden entry for prompt [{key}]"));
        assert_eq!(
            cur.len(),
            exp.len(),
            "vocab size changed for prompt [{key}]"
        );
        let d = max_abs_diff(cur, exp);
        assert!(
            d <= TOL,
            "golden logits drift for prompt [{key}]: max abs diff = {d} (tol {TOL})"
        );
        assert_eq!(
            top_k(cur, TOP_K),
            top_k(exp, TOP_K),
            "top-{TOP_K} changed for prompt [{key}]"
        );
    }
}

#[test]
fn test_mock_weights_are_deterministic() {
    let a = logits_of(&mock(), &[3, 1, 4, 1, 5]);
    let b = logits_of(&mock(), &[3, 1, 4, 1, 5]);
    assert_eq!(a, b, "fixed-seed mock must be bit-deterministic");
}
