//! Ticket #55 acceptance: the draft model runs on its own KV cache instead of re-running
//! the whole prefix for every draft token, so speculation is a wall-clock win.
//!
//! Real SmolLM2 pair (360M target + 135M draft, γ=4, greedy, ctx 512) when
//! `KANSO_SPEC_TARGET_GGUF` / `KANSO_SPEC_DRAFT_GGUF` point at GGUF files. Otherwise a
//! mock proxy of the same shape: 4-layer target, 1-layer draft, shared heads/embeddings,
//! and the target's extra layers neutralized (zero output projections ⇒ identity), so the
//! pair agrees token-for-token.
//!
//! Measured (release, 2026-10-17, SmolLM2-360M/135M-Instruct Q8_0 on this Mac, α=0.884):
//! before #45's batched forward landed on the integration branch the real pair landed at
//! 0.84-0.99x; after merging it the same test reads 1.63-2.00x. The draft-KV fix is the same
//! in both, so the difference is the baseline: #45 moved `Engine::step()` onto the batched
//! path, which by design trades batch-1 latency for batch throughput (#45). Machine load was
//! high during the later runs, so neither number is a trustworthy A/B. Neither gates this
//! ticket — see the cost analysis below, which uses the single-token decode path.
//!
//! On the pre-#45 decode path the round economics do not close: a 5-row target forward costs
//! 43.6ms against 15.7ms for a single-row decode (weight reads are only partially amortized,
//! the rest is per-row compute/dispatch), and the 135M draft costs 8.0ms per committed token,
//! so `verify_row (8.7ms) + draft_row (8.0ms) > target_row (15.7ms)`. Closing that is the
//! parent spec's story 1 (one batched forward per step, amortized weight reads — #45 and the
//! Fase-1 kernels), not this ticket. The proxy below is what gates the fix here.
//!
//! What the proxy does not cover is the real pair's acceptance; it is reported in the same
//! run (set the two env vars) and asserted to stay out of chance level.
//!
//! `speculative_decode_beats_non_spec_at_ctx_512` carries a wall-clock gate (1.3x on the
//! mock proxy), so it is `#[ignore]`d in debug builds where per-op overhead dominates the
//! measurement (ticket #61 F5). The release-only CI job runs it; the debug run reports it as
//! ignored rather than passing under a lowered floor.

use candle_core::{DType, Device, Tensor};
use kanso_infer::engine::Engine;
use kanso_infer::model::Model;
use kanso_infer::speculative::SpeculativeEngine;
use kanso_infer::weights::LinearWeight;
use std::time::{Duration, Instant};

const HIDDEN: usize = 64;
const HEADS: usize = 4;
const KV_HEADS: usize = 2;
const VOCAB: usize = 256;
const TARGET_LAYERS: usize = 4;
const DRAFT_LAYERS: usize = 1;
const BLOCK_SIZE: usize = 16;
/// 512-token prompt + 192 generated tokens = 44 blocks; kept small so the real-model path
/// does not allocate a multi-GB KV pool.
const TOTAL_BLOCKS: usize = 128;
const PROMPT_LEN: usize = 512;
const GENERATE: usize = 192;
/// Tokens generated before the timed window (setup: draft prompt prefill, table builds).
const WARMUP: usize = 32;
const GAMMA: usize = 4;
const MIN_SPEEDUP: f64 = 1.3;

fn mock_pair() -> (Model, Model) {
    let mut target =
        Model::create_mock_tiny(HIDDEN, HEADS, KV_HEADS, TARGET_LAYERS, VOCAB).unwrap();
    let mut draft = Model::create_mock_tiny(HIDDEN, HEADS, KV_HEADS, DRAFT_LAYERS, VOCAB).unwrap();
    // Depth is the only difference between the two: share layer 0, the embedding and the
    // head (the mock RNG draw order depends on depth, so the draft's own copies differ).
    draft.weights.embed_tokens = target.weights.embed_tokens.clone();
    draft.weights.norm = target.weights.norm.clone();
    draft.weights.lm_head = target.weights.lm_head.clone();
    draft.weights.layers[0] = target.weights.layers[0].clone();

    let dev = Device::Cpu;
    let zero = |shape: (usize, usize)| {
        LinearWeight::from_tensor(Tensor::zeros(shape, DType::F32, &dev).unwrap())
    };
    for layer in &mut target.weights.layers[DRAFT_LAYERS..] {
        layer.attn_output = zero((HIDDEN, HIDDEN));
        layer.ffn_down = zero((HIDDEN, HIDDEN * 2));
    }
    (target, draft)
}

fn real_pair() -> Option<(Model, Model)> {
    let target = std::env::var("KANSO_SPEC_TARGET_GGUF").ok()?;
    let draft = std::env::var("KANSO_SPEC_DRAFT_GGUF").ok()?;
    if !std::path::Path::new(&target).is_file() || !std::path::Path::new(&draft).is_file() {
        return None;
    }
    Some((
        Model::load_gguf(&target, Device::Cpu).ok()?,
        Model::load_gguf(&draft, Device::Cpu).ok()?,
    ))
}

/// Steady-state decode wall clock plus the generated token ids. The prefill step and the
/// first `WARMUP` tokens are untimed: they carry the one-off setup that says nothing about
/// decode throughput (the draft's prompt prefill, RoPE table builds, backend warmup).
fn decode_phase(warmup: usize, mut next: impl FnMut() -> Option<Vec<u32>>) -> (Duration, Vec<u32>) {
    let mut tokens = next().unwrap_or_default();
    while tokens.len() < warmup {
        match next() {
            Some(batch) => tokens.extend(batch),
            None => return (Duration::ZERO, tokens),
        }
    }
    let start = Instant::now();
    while let Some(batch) = next() {
        tokens.extend(batch);
    }
    (start.elapsed(), tokens)
}

fn step_tokens(engine: &mut Engine) -> Option<Vec<u32>> {
    engine.has_unfinished_requests().then(|| {
        engine
            .step()
            .expect("engine step failed")
            .into_iter()
            .map(|o| o.token_id)
            .collect()
    })
}

fn last_logits(model: &Model, tokens: &[u32]) -> Vec<f32> {
    model
        .forward_single_prompt(tokens)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
}

fn argmax(v: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best as u32
}

/// The verifier's batched forward and the single-token decode path must agree on the same
/// prefix. The verify path skipped RoPE entirely, so the target disagreed with its own
/// decode step on the same tokens (max |Δlogits| ≈ 30 on this mock — fp noise is ~2e-5),
/// which caps speculative acceptance at chance level no matter how good the draft is.
#[test]
fn verify_path_logits_match_the_decode_path() {
    use candle_core::Device;
    use kanso_infer::kv_cache::{BlockTable, KvStorage};

    let model = Model::create_mock_tiny(HIDDEN, HEADS, KV_HEADS, 2, VOCAB).unwrap();
    let prompt: Vec<u32> = (0..32).map(|i| (i % VOCAB) as u32).collect();
    let fresh = || {
        let kv = KvStorage::new(2, 64, BLOCK_SIZE, KV_HEADS, HIDDEN / HEADS, &Device::Cpu).unwrap();
        let mut table = BlockTable::new();
        for b in 0..8 {
            table.push_block(b);
        }
        (kv, table)
    };

    let (mut kv, table) = fresh();
    let decoded: Vec<Vec<f32>> = prompt
        .iter()
        .enumerate()
        .map(|(i, &t)| {
            model
                .forward_decode_step(t, i, &table, &mut kv)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        })
        .collect();

    let (mut kv2, table2) = fresh();
    let verified = model
        .forward_verify_tokens(&prompt, 0, &table2, &mut kv2)
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();

    for i in [8usize, 16, 24, 31] {
        let drift = decoded[i]
            .iter()
            .zip(&verified[i])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            drift < 1e-3,
            "decode vs verify drift {drift} at position {i} — the verify path must run the \
             same RoPE'd attention as decode"
        );
        assert_eq!(argmax(&decoded[i]), argmax(&verified[i]), "argmax at {i}");
    }
}

#[test]
fn draft_agrees_with_target_on_the_proxy_pair() {
    let (target, draft) = mock_pair();
    let prompt: Vec<u32> = (0..PROMPT_LEN).map(|i| (i % VOCAB) as u32).collect();
    for prefix in [&prompt[..8], &prompt[..]] {
        let (t, d) = (last_logits(&target, prefix), last_logits(&draft, prefix));
        let drift = t
            .iter()
            .zip(&d)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            drift < 1e-3,
            "proxy pair must agree: max |Δlogits| = {drift} at ctx {}",
            prefix.len()
        );
    }
}

/// The 1.3x ticket gate, applied where throughput is meaningful. An unoptimized `cargo test`
/// build spends its time in per-op overhead (candle is built at opt-level 0), so its ratio
/// is noise around break-even, not a benchmark, and the test is `#[ignore]`d there instead of
/// being asserted against a floor a regression could slide under (ticket #61 F5): release
/// measures 1.68x and the release-only CI job gates on it.
#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "release-only speedup gate (ticket #61 F5): run `cargo test --release`"
)]
fn speculative_decode_beats_non_spec_at_ctx_512() {
    let prompt: Vec<u32> = (0..PROMPT_LEN).map(|i| (i % VOCAB) as u32).collect();

    let (target, draft, proxy) = match real_pair() {
        Some((t, d)) => (t, d, false),
        None => {
            let (t, d) = mock_pair();
            (t, d, true)
        }
    };

    // Non-spec baseline first: the real-model pair does not fit in memory twice.
    let target_plain = if proxy {
        mock_pair().0
    } else {
        Model::load_gguf(
            std::env::var("KANSO_SPEC_TARGET_GGUF").unwrap(),
            Device::Cpu,
        )
        .unwrap()
    };
    let (plain_time, plain_tokens) = {
        let mut engine = Engine::new(target_plain, 1, TOTAL_BLOCKS, BLOCK_SIZE).expect("engine");
        engine.add_request(prompt.clone(), GENERATE);
        decode_phase(WARMUP, || step_tokens(&mut engine))
    };

    let (spec_time, spec_tokens, alpha) = {
        let mut engine =
            SpeculativeEngine::new(target, Some(draft), 1, TOTAL_BLOCKS, BLOCK_SIZE, GAMMA)
                .expect("engine");
        engine.add_request(prompt.clone(), GENERATE);
        let (duration, tokens) = decode_phase(WARMUP, || {
            if !engine.has_unfinished_requests() {
                return None;
            }
            Some(
                engine
                    .step()
                    .expect("speculative step failed")
                    .into_iter()
                    .map(|o| o.token_id)
                    .collect(),
            )
        });
        // Every draft block must be back in the draft pool once the request is done.
        assert_eq!(
            engine.draft_free_blocks_count(),
            Some(TOTAL_BLOCKS),
            "draft KV blocks leaked"
        );
        println!(
            "  draft acceptance α={:.3}, τ̄={:.2} tokens/step, rollbacks={}",
            engine.profiler.acceptance_rate(),
            engine.profiler.mean_accepted_per_step(),
            engine.profiler.rollback_count(),
        );
        (duration, tokens, engine.profiler.acceptance_rate())
    };

    let speedup = plain_time.as_secs_f64() / spec_time.as_secs_f64();
    let timed = GENERATE - WARMUP;
    println!(
        "ctx={PROMPT_LEN} γ={GAMMA} greedy: non-spec {:.2} tok/s / spec {:.2} tok/s → {speedup:.2}x \
         ({})",
        timed as f64 / plain_time.as_secs_f64(),
        timed as f64 / spec_time.as_secs_f64(),
        if proxy {
            "mock proxy (no KANSO_SPEC_*_GGUF; 100% acceptance, upper bound)"
        } else {
            "real SmolLM2 pair"
        }
    );

    assert_eq!(spec_tokens.len(), GENERATE);
    assert_eq!(plain_tokens.len(), GENERATE);
    if proxy {
        // Identical weights + greedy speculation ⇒ byte-identical token stream.
        assert_eq!(spec_tokens, plain_tokens);
        assert!(
            speedup > MIN_SPEEDUP,
            "speculative speedup {speedup:.2}x ≤ {MIN_SPEEDUP}x (non-spec {:.1}ms vs spec {:.1}ms)",
            plain_time.as_secs_f64() * 1e3,
            spec_time.as_secs_f64() * 1e3
        );
    } else {
        // A working draft-KV path keeps the draft in distribution; a broken one (stale or
        // rolled-back-wrong cache) collapses α toward chance. The speedup itself is
        // reported, not gated — see the module docs.
        assert!(
            alpha >= 0.5,
            "draft acceptance collapsed to α={alpha:.3}: the draft cache is not tracking the \
             committed prefix"
        );
    }
}

/// The temperature path, end to end: the draft's distributions must be handed to the
/// verifier in the order it proposes them. With a draft that *is* the target, `p == q`, so
/// every proposal is accepted (min(1, p/q) = 1) and τ̄ is the full γ+1 — a misaligned or
/// missing `draft_probs` vector would drop acceptance to chance or error out.
///
/// This covers the aligned case only. The `q ≠ p` case (residual branch firing, output
/// staying on `p`) is driven through the real verifier in
/// `test_speculative_sampling_distribution.rs` (ticket #61 F8).
#[test]
fn temperature_sampling_verifies_draft_distributions() {
    use kanso_infer::scheduler::SamplingParams;

    let (target, draft) = mock_pair();
    let mut engine =
        SpeculativeEngine::new(target, Some(draft), 1, TOTAL_BLOCKS, BLOCK_SIZE, GAMMA)
            .expect("engine");
    engine.add_request_with_sampling(
        (0..64).map(|i| (i % VOCAB) as u32).collect(),
        64,
        Vec::new(),
        SamplingParams {
            temperature: 1.0,
            top_p: 1.0,
            seed: 7,
        },
    );

    let mut generated = 0usize;
    while engine.has_unfinished_requests() {
        generated += engine.step().unwrap().len();
    }

    assert_eq!(generated, 64);
    assert_eq!(engine.profiler.rollback_count(), 0);
    assert_eq!(
        engine.profiler.acceptance_rate(),
        1.0,
        "identical draft/target distributions must accept every proposal"
    );
    assert_eq!(
        engine.draft_free_blocks_count(),
        Some(TOTAL_BLOCKS),
        "draft KV blocks leaked"
    );
}
