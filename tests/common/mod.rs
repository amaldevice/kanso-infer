//! Shared helpers for #57-era numeric tests (mock is fixed-seed random).

// Each test binary only pulls in the helpers it uses; the rest are dead code there.
#![allow(dead_code)]

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

/// A Metal device, or `None` (with the reason printed) when this runtime
/// cannot provide one. Non-capability failures still panic.
///
/// Ticket #99: on an older Metal runtime (the `macos-14` CI runner) candle panics
/// while looking up `MTLResidencySetDescriptor`; failing the test there reports a
/// missing class as a regression. Skipping keeps the suite honest: on a host that
/// has the class nothing is skipped, so the assertions still run.
pub fn metal_device_or_skip() -> Option<candle_core::Device> {
    match kanso_infer::hardware::metal_device() {
        Ok(d) => Some(d),
        Err(e) if kanso_infer::hardware::is_metal_runtime_gap(&e) => {
            println!("[ticket #99] skip: {e}");
            None
        }
        Err(e) => panic!("metal device failed for a non-capability reason: {e}"),
    }
}

/// Max absolute element-wise difference between two flat logit vecs.
pub fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}
