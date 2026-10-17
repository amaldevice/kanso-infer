//! Speculative rejection sampling (#55).
//!
//! Leviathan et al., "Fast Inference from Transformers via Speculative Decoding" (ICML
//! 2023), Algorithm 1: the draft draws a token from `q`, the target accepts it with
//! probability `min(1, p(token)/q(token))`, and a rejected draw is corrected with the
//! residual `max(0, p - q)` normalized. The output distribution is then exactly `p`, for
//! any `q` — which is what makes temperature sampling, not just greedy decoding, exact.

/// Probability that the target accepts `token` proposed by the draft.
///
/// `p` is the target and `q` the draft distribution over the same vocabulary, both already
/// truncated by the request's temperature/top_p (`crate::sampler::softmax_probs`). `q` must
/// be the distribution the draft actually drew `token` from.
pub fn acceptance_prob(p: &[f32], q: &[f32], token: u32) -> f32 {
    let idx = token as usize;
    let (Some(&p_i), Some(&q_i)) = (p.get(idx), q.get(idx)) else {
        return 0.0;
    };
    if q_i <= 0.0 {
        return if p_i > 0.0 { 1.0 } else { 0.0 };
    }
    (p_i / q_i).min(1.0)
}

/// `max(0, p - q)`, renormalized: the distribution a rejected draft token is replaced
/// with. Falls back to `p` when the residual has no mass (a draft that *is* the target).
pub fn residual_probs(p: &[f32], q: &[f32]) -> Vec<f32> {
    let mut out: Vec<f32> = p
        .iter()
        .zip(q)
        .map(|(&p_i, &q_i)| (p_i - q_i).max(0.0))
        .collect();
    let sum: f32 = out.iter().sum();
    if sum <= 0.0 {
        return p.to_vec();
    }
    for x in out.iter_mut() {
        *x /= sum;
    }
    out
}
