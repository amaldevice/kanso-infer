//! Ticket #55: speculative sampling at temperature > 0 must reproduce the plain sampler's
//! token distribution (Leviathan et al. 2023, Alg. 1) — checked with a chi-square test on
//! the accept/residual procedure the verifier runs.

use kanso_infer::sampler::{sample_from_probs, softmax_probs};
use kanso_infer::speculative::sampling::{acceptance_prob, residual_probs};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const DRAWS: usize = 40_000;
/// χ²₇ at α = 0.001.
const CRITICAL: f64 = 24.322;
const VOCAB: usize = 8;

/// Deliberately different draft: mass pushed toward the other end of the vocabulary.
const TARGET_LOGITS: [f32; VOCAB] = [2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5];
const DRAFT_LOGITS: [f32; VOCAB] = [0.0, 0.5, 1.0, 1.5, 2.0, 1.0, 2.5, -2.0];

/// The verifier's temperature > 0 path: draw from the draft, accept with min(1, p/q),
/// else correct with the residual `max(0, p - q)` normalized.
fn speculative_draw(p: &[f32], q: &[f32], rng: &mut StdRng) -> u32 {
    let candidate = sample_from_probs(q, rng);
    if rng.random::<f32>() <= acceptance_prob(p, q, candidate) {
        candidate
    } else {
        sample_from_probs(&residual_probs(p, q), rng)
    }
}

fn histogram(mut draw: impl FnMut(&mut StdRng) -> u32) -> Vec<usize> {
    let mut rng = StdRng::seed_from_u64(0x5EED_0055);
    let mut hist = vec![0usize; VOCAB];
    for _ in 0..DRAWS {
        hist[draw(&mut rng) as usize] += 1;
    }
    hist
}

fn chi_square(hist: &[usize], p: &[f32]) -> f64 {
    hist.iter()
        .zip(p)
        .map(|(&count, &pi)| {
            let expected = DRAWS as f64 * pi as f64;
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

    let residual = residual_probs(&p, &q);
    assert!((residual.iter().sum::<f32>() - 1.0).abs() < 1e-5);

    let plain = chi_square(&histogram(|rng| sample_from_probs(&p, rng)), &p);
    assert!(
        plain < CRITICAL,
        "plain sampler is not even χ² on its own distribution: {plain}"
    );

    let speculative = chi_square(&histogram(|rng| speculative_draw(&p, &q, rng)), &p);
    assert!(
        speculative < CRITICAL,
        "speculative sampling deviates from the target distribution: χ² = {speculative} \
         (critical {CRITICAL}); q ≠ p, so the residual path must be doing the work"
    );

    // The draft distribution alone must *not* pass — otherwise the test has no power.
    let draft_only = chi_square(&histogram(|rng| sample_from_probs(&q, rng)), &p);
    assert!(
        draft_only > CRITICAL,
        "control failed: draft-only χ² {draft_only} ≤ {CRITICAL}, distributions too close"
    );
}

#[test]
fn identical_draft_leaves_the_target_distribution_untouched() {
    let p = softmax_probs(&TARGET_LOGITS, 1.0, 1.0).unwrap();
    assert!(acceptance_prob(&p, &p, 3) == 1.0);
    let residual = residual_probs(&p, &p);
    for (a, b) in residual.iter().zip(&p) {
        assert!((a - b).abs() < 1e-6);
    }
    assert!(chi_square(&histogram(|rng| speculative_draw(&p, &p, rng)), &p) < CRITICAL);
}
