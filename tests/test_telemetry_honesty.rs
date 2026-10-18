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

use kanso_infer::bench::run_synthetic_benchmark;
use kanso_infer::engine::Engine;
use kanso_infer::model::Model;
use kanso_infer::sampler::sample_logits;
use kanso_infer::scheduler::SamplingParams;

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
    let report =
        run_synthetic_benchmark(&mut engine, 2, prompt_len, gen_len, false).expect("benchmark");
    let expected = weights_bytes + kv_per_token * (prompt_len + gen_len);
    assert_eq!(
        report.bytes_read_per_step,
        expected,
        "bytes_read_per_step must be derived from the weighed model \
         (weights {weights_bytes} + KV {kv_per_token}/tok x {} ctx), not a literal",
        prompt_len + gen_len,
    );
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
        !kanso_infer::ane::has_ane_support(),
        "no Core ML execution path exists yet; claiming ANE support lies"
    );
}
