//! RED test for #57: proves the all-ones mock is blind.
//! With uniform weights every prompt maps to identical logits, so attention
//! masking bugs are invisible. These tests FAIL on `Tensor::ones` weights
//! and PASS once the mock uses fixed-seed random weights.

use kanso_infer::model::Model;

fn last_token_logits(prompt: &[u32]) -> Vec<f32> {
    let model = Model::create_mock_tiny(16, 4, 2, 2, 50).unwrap();
    model
        .forward_single_prompt(prompt)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn test_mock_distinguishes_token_content() {
    let a = last_token_logits(&[1, 2, 3]);
    let b = last_token_logits(&[7, 8, 9]);
    let d = max_abs_diff(&a, &b);
    assert!(
        d > 1e-6,
        "blind mock: different prompts produced identical logits (max abs diff = {d})"
    );
}

#[test]
fn test_mock_distinguishes_single_token_identity() {
    let a = last_token_logits(&[5]);
    let b = last_token_logits(&[12]);
    let d = max_abs_diff(&a, &b);
    assert!(
        d > 1e-6,
        "blind mock: different tokens produced identical logits (max abs diff = {d})"
    );
}
