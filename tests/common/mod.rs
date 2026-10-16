//! Shared helpers for #57-era numeric tests (mock is fixed-seed random).

use kanso_infer::model::Model;

/// Last-token logits of a full monolithic forward as a flat vec.
pub fn last_token_logits(model: &Model, prompt: &[u32]) -> Vec<f32> {
    model
        .forward_single_prompt(prompt)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
}

/// Max absolute element-wise difference between two flat logit vecs.
pub fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}
