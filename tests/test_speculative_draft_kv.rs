//! Ticket #55 acceptance: the draft model runs on its own KV cache instead of re-running
//! the whole prefix for every draft token, so speculation is a wall-clock win.
//!
//! Real SmolLM2 pair (360M target + 135M draft, γ=4, greedy, ctx 512) when
//! `KANSO_SPEC_TARGET_GGUF` / `KANSO_SPEC_DRAFT_GGUF` point at GGUF files. Otherwise a
//! mock proxy of the same shape: 4-layer target, 1-layer draft, shared heads/embeddings,
//! and the target's extra layers neutralized (zero output projections ⇒ identity), so the
//! pair agrees token-for-token. 100% acceptance is optimistic — a real 135M/360M pair
//! agrees ~65% of the time — so the proxy number is an upper bound, not a substitute.

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
const TOTAL_BLOCKS: usize = 512;
const PROMPT_LEN: usize = 512;
const GENERATE: usize = 64;
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

/// Decode-phase wall clock (prefill and the first sampled token are untimed — the same
/// work either way) plus the generated token ids.
fn decode_phase(mut next: impl FnMut() -> Option<Vec<u32>>) -> (Duration, Vec<u32>) {
    let mut tokens = next().unwrap_or_default();
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
        let mut kv =
            KvStorage::new(2, 64, BLOCK_SIZE, KV_HEADS, HIDDEN / HEADS, &Device::Cpu).unwrap();
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

#[test]
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
            &std::env::var("KANSO_SPEC_TARGET_GGUF").unwrap(),
            Device::Cpu,
        )
        .unwrap()
    };
    let (plain_time, plain_tokens) = {
        let mut engine = Engine::new(target_plain, 1, TOTAL_BLOCKS, BLOCK_SIZE);
        engine.add_request(prompt.clone(), GENERATE);
        decode_phase(|| step_tokens(&mut engine))
    };

    let (spec_time, spec_tokens) = {
        let mut engine =
            SpeculativeEngine::new(target, Some(draft), 1, TOTAL_BLOCKS, BLOCK_SIZE, GAMMA);
        engine.add_request(prompt.clone(), GENERATE);
        decode_phase(|| {
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
        })
    };

    let speedup = plain_time.as_secs_f64() / spec_time.as_secs_f64();
    println!(
        "ctx={PROMPT_LEN} γ={GAMMA} greedy: non-spec {:.1}ms / spec {:.1}ms → {:.2}x ({})",
        plain_time.as_secs_f64() * 1e3,
        spec_time.as_secs_f64() * 1e3,
        speedup,
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
    }
    assert!(
        speedup > MIN_SPEEDUP,
        "speculative speedup {speedup:.2}x ≤ {MIN_SPEEDUP}x (non-spec {:.1}ms vs spec {:.1}ms)",
        plain_time.as_secs_f64() * 1e3,
        spec_time.as_secs_f64() * 1e3
    );
}
