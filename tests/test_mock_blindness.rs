//! RED test for #57: proves the all-ones mock is blind.
//! With uniform weights every prompt maps to identical logits, so attention
//! masking bugs are invisible. These tests FAIL on `Tensor::ones` weights
//! and PASS once the mock uses fixed-seed random weights.

use kanso_infer::model::Model;

mod common;
use common::{last_token_logits, max_abs_diff};

fn mock_logits(prompt: &[u32]) -> Vec<f32> {
    let model = Model::create_mock_tiny(16, 4, 2, 2, 50).unwrap();
    last_token_logits(&model, prompt)
}

#[test]
fn test_mock_distinguishes_token_content() {
    let a = mock_logits(&[1, 2, 3]);
    let b = mock_logits(&[7, 8, 9]);
    let d = max_abs_diff(&a, &b);
    assert!(
        d > 1e-6,
        "blind mock: different prompts produced identical logits (max abs diff = {d})"
    );
}

#[test]
fn test_mock_distinguishes_single_token_identity() {
    let a = mock_logits(&[5]);
    let b = mock_logits(&[12]);
    let d = max_abs_diff(&a, &b);
    assert!(
        d > 1e-6,
        "blind mock: different tokens produced identical logits (max abs diff = {d})"
    );
}
