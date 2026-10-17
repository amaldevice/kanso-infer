//! Ticket #55: speculative sampling at temperature > 0 must reproduce the plain sampler's
//! token distribution (Leviathan et al. 2023, Alg. 1).
//!
//! Two levels, because they fail differently (ticket #61 F8):
//! - [`speculative_temperature_sampling_matches_plain_distribution`] unit-checks the accept
//!   and residual math itself at high draw counts, without a model in the way.
//! - [`speculative_verifier_stays_on_the_target_distribution_when_q_differs_from_p`] drives
//!   `q ≠ p` through `SpeculativeVerifier::verify` on a real (mock-weight) model, so a
//!   verifier that never reached the residual branch, or wired `p`/`q` backwards, fails.

use candle_core::Device;
use kanso_infer::kv_cache::{BlockPool, BlockTable, KvStorage, PrefixCache};
use kanso_infer::model::Model;
use kanso_infer::sampler::{sample_from_probs, softmax_probs};
use kanso_infer::scheduler::SamplingParams;
use kanso_infer::speculative::sampling::{acceptance_prob, residual_probs};
use kanso_infer::speculative::{SpeculativeVerifier, VerificationResult};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const DRAWS: usize = 40_000;
/// χ²₇ at α = 0.001.
const CRITICAL: f64 = 24.322;
const VOCAB: usize = 8;

/// Deliberately different draft: mass pushed toward the other end of the vocabulary.
const TARGET_LOGITS: [f32; VOCAB] = [2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5];
const DRAFT_LOGITS: [f32; VOCAB] = [0.0, 0.5, 1.0, 1.5, 2.0, 1.0, 2.5, -2.0];

/// The verifier's temperature > 0 math, called directly: draw from the draft, accept with
/// min(1, p/q), else correct with the residual `max(0, p - q)` normalized. This is the
/// unit-level twin of `SpeculativeVerifier::verify` (see the module docs).
fn speculative_draw(p: &[f32], q: &[f32], rng: &mut StdRng) -> u32 {
    let candidate = sample_from_probs(q, rng);
    if rng.random::<f32>() <= acceptance_prob(p, q, candidate) {
        candidate
    } else {
        sample_from_probs(
            &residual_probs(p, q).expect("p and q cover the same vocab"),
            rng,
        )
    }
}

fn histogram(vocab: usize, mut draw: impl FnMut(&mut StdRng) -> u32) -> Vec<usize> {
    let mut rng = StdRng::seed_from_u64(0x5EED_0055);
    let mut hist = vec![0usize; vocab];
    for _ in 0..DRAWS {
        hist[draw(&mut rng) as usize] += 1;
    }
    hist
}

/// χ² of `hist` against `p`, using the histogram's own draw count as the total (so a
/// caller can run any number of draws).
fn chi_square(hist: &[usize], p: &[f32]) -> f64 {
    let total: usize = hist.iter().sum();
    hist.iter()
        .zip(p)
        .map(|(&count, &pi)| {
            let expected = total as f64 * pi as f64;
            (count as f64 - expected).powi(2) / expected
        })
        .sum()
}

#[test]
fn speculative_temperature_sampling_matches_plain_distribution() {
    let p = softmax_probs(&TARGET_LOGITS, 1.0, 1.0).unwrap();
    let q = softmax_probs(&DRAFT_LOGITS, 1.0, 1.0).unwrap();
    assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    assert!((q.iter().sum::<f32>() - 1.0).abs() < 1e-5);

    let residual = residual_probs(&p, &q).unwrap();
    assert!((residual.iter().sum::<f32>() - 1.0).abs() < 1e-5);

    let plain = chi_square(&histogram(VOCAB, |rng| sample_from_probs(&p, rng)), &p);
    assert!(
        plain < CRITICAL,
        "plain sampler is not even χ² on its own distribution: {plain}"
    );

    let speculative = chi_square(&histogram(VOCAB, |rng| speculative_draw(&p, &q, rng)), &p);
    assert!(
        speculative < CRITICAL,
        "speculative sampling deviates from the target distribution: χ² = {speculative} \
         (critical {CRITICAL}); q ≠ p, so the residual path must be doing the work"
    );

    // The draft distribution alone must *not* pass — otherwise the test has no power.
    let draft_only = chi_square(&histogram(VOCAB, |rng| sample_from_probs(&q, rng)), &p);
    assert!(
        draft_only > CRITICAL,
        "control failed: draft-only χ² {draft_only} ≤ {CRITICAL}, distributions too close"
    );
}

#[test]
fn identical_draft_leaves_the_target_distribution_untouched() {
    let p = softmax_probs(&TARGET_LOGITS, 1.0, 1.0).unwrap();
    assert!(acceptance_prob(&p, &p, 3) == 1.0);
    let residual = residual_probs(&p, &p).unwrap();
    for (a, b) in residual.iter().zip(&p) {
        assert!((a - b).abs() < 1e-6);
    }
    assert!(chi_square(&histogram(VOCAB, |rng| speculative_draw(&p, &p, rng)), &p) < CRITICAL);
}

// --- `q ≠ p` through the real verifier (ticket #61 F8) ---------------------------------

/// χ²₁₅ at α = 0.001.
const CRITICAL_16: f64 = 37.697;
const E2E_VOCAB: usize = 16;
const E2E_DRAWS: usize = 800;
/// Small enough that 800 verify rounds finish inside a test; what matters is that it is a
/// real `Model` with real logits, not that it is large.
const E2E_HIDDEN: usize = 16;
const E2E_HEADS: usize = 4;
const E2E_KV_HEADS: usize = 2;
const E2E_LAYERS: usize = 2;
const E2E_BLOCK_SIZE: usize = 8;
const E2E_BLOCKS: usize = 4;
/// The single token the batch is positioned on (`start_pos = 0`).
const E2E_CURRENT: u32 = 5;
/// Sampling temperature for the e2e rounds. The mock model's raw logits are peaked enough
/// that most of the vocabulary would expect well under one draw, which is outside what
/// χ² assumes; at 8.0 every bin expects ≥ 15 of the 800 draws.
const E2E_TEMPERATURE: f32 = 8.0;
/// Per-round seed base: each round is a distinct reproducible request.
const E2E_SEED_BASE: u64 = 0x5EED_0061;

/// One verifier round at the fixed position, on a fresh block pool/table so every round
/// starts from the same state.
fn verify_one_round(
    model: &Model,
    storage: &mut KvStorage,
    sampling: SamplingParams,
    candidate: u32,
    q: &[f32],
) -> VerificationResult {
    let mut pool = BlockPool::new(E2E_BLOCKS);
    let mut cache = PrefixCache::new(E2E_BLOCK_SIZE);
    let mut table = BlockTable::new();
    SpeculativeVerifier::verify(
        model,
        E2E_CURRENT,
        &[candidate],
        &[q.to_vec()],
        sampling,
        0,
        &mut table,
        &mut pool,
        &mut cache,
        storage,
        E2E_BLOCK_SIZE,
    )
    .expect("verify failed")
}

/// The target's own next-token distribution at `temperature`, read through
/// `forward_verify_tokens` — the same forward the verifier's `p` comes from.
fn target_distribution(model: &Model, storage: &mut KvStorage, temperature: f32) -> Vec<f32> {
    let mut pool = BlockPool::new(E2E_BLOCKS);
    let mut table = BlockTable::new();
    table.push_block(pool.allocate().expect("pool has a block"));
    let logits = model
        .forward_verify_tokens(&[E2E_CURRENT], 0, &table, storage)
        .expect("probe forward");
    softmax_probs(
        &logits.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        temperature,
        1.0,
    )
    .unwrap()
}

/// Ticket #61 F8: the residual branch, driven end to end. `q` is `p` rotated half a
/// vocabulary — overlapping enough that the accept branch fires, different enough that the
/// residual must carry the tail. The token the verifier produces for the current position
/// must still follow the target's `p`, which a verifier that skipped the residual (or
/// swapped `p`/`q`) cannot produce.
#[test]
fn speculative_verifier_stays_on_the_target_distribution_when_q_differs_from_p() {
    let model = Model::create_mock_tiny(E2E_HIDDEN, E2E_HEADS, E2E_KV_HEADS, E2E_LAYERS, E2E_VOCAB)
        .unwrap();
    let sampling = SamplingParams {
        temperature: E2E_TEMPERATURE,
        top_p: 1.0,
        seed: 7,
    };
    let mut storage = KvStorage::new(
        E2E_LAYERS,
        E2E_BLOCKS,
        E2E_BLOCK_SIZE,
        E2E_KV_HEADS,
        E2E_HIDDEN / E2E_HEADS,
        &Device::Cpu,
    )
    .unwrap();

    let p = target_distribution(&model, &mut storage, E2E_TEMPERATURE);
    // χ² wants at least ~5 expected draws per bin; with a fixed seed and a fixed model this
    // is deterministic, so the check guards the test's validity, not a flaky threshold.
    let min_expected = p.iter().cloned().fold(f32::MAX, f32::min) * E2E_DRAWS as f32;
    assert!(
        min_expected >= 5.0,
        "chi-square needs >= 5 expected draws per bin, got {min_expected:.2}: p = {p:?}"
    );
    let q: Vec<f32> = (0..E2E_VOCAB)
        .map(|i| p[(i + E2E_VOCAB / 2) % E2E_VOCAB])
        .collect();

    let mut rng = StdRng::seed_from_u64(0x5EED_0061);
    let mut hist = vec![0usize; E2E_VOCAB];
    let mut rejected_rounds = 0usize;
    for round in 0..E2E_DRAWS {
        // A fresh seed per round: one verify call draws its accept test and residual from a
        // single seeded stream, so replaying a request with one seed is reproducible but
        // one round is one draw — the empirical distribution needs many of them.
        let round_sampling = SamplingParams {
            seed: E2E_SEED_BASE + round as u64,
            ..sampling
        };
        let candidate = sample_from_probs(&q, &mut rng) as u32;
        let result = verify_one_round(&model, &mut storage, round_sampling, candidate, &q);
        // `committed_tokens[0]` is the token produced for the position the verifier started
        // at — the accepted candidate or the residual correction. (`emitted_token` is the
        // bonus draw from the *next* position when every candidate was accepted.)
        hist[result.committed_tokens[0] as usize] += 1;
        if result.num_rejected > 0 {
            rejected_rounds += 1;
        }
    }

    // Same inputs and seed: the round replays exactly (ticket #61 F9).
    let replay_a = verify_one_round(&model, &mut storage, sampling, 3, &q);
    let replay_b = verify_one_round(&model, &mut storage, sampling, 3, &q);
    assert_eq!(
        replay_a, replay_b,
        "the same round with the same seed must replay exactly"
    );

    assert!(
        rejected_rounds > 0,
        "the residual branch never fired in {E2E_DRAWS} rounds — q ≠ p, it must"
    );

    let speculative = chi_square(&hist, &p);
    assert!(
        speculative < CRITICAL_16,
        "verifier output deviates from the target distribution: χ² = {speculative} \
         (critical {CRITICAL_16}, {rejected_rounds}/{E2E_DRAWS} rounds rejected)"
    );

    // The draft's own distribution must *not* pass — otherwise this test cannot see a
    // verifier that just returns whatever the draft proposed.
    let draft_only = chi_square(&histogram(E2E_VOCAB, |rng| sample_from_probs(&q, rng)), &p);
    assert!(
        draft_only > CRITICAL_16,
        "control failed: draft-only χ² {draft_only} ≤ {CRITICAL_16}, q too close to p"
    );
}
