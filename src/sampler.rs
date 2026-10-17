use crate::error::EngineError;
use candle_core::Tensor;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// argmax over `logits`: the token every temperature-0 path draws.
pub fn greedy_token(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &val) in logits.iter().enumerate() {
        if val > best_val {
            best_val = val;
            best = i;
        }
    }
    best as u32
}

/// Softmax over `logits / temperature`, truncated to the top-p nucleus and renormalized
/// (probability outside the nucleus is 0). `temperature <= 0` is the greedy one-hot.
///
/// #55: the speculative rejection test needs the distribution itself, not only a draw —
/// draft and target must be compared on the same truncated distribution the plain
/// sampler would have drawn from.
pub fn softmax_probs(
    logits: &[f32],
    temperature: f32,
    top_p: f32,
) -> Result<Vec<f32>, EngineError> {
    if logits.is_empty() {
        return Err(EngineError::Model("Empty logits".to_string()));
    }
    if temperature <= 0.0 {
        let mut one_hot = vec![0.0; logits.len()];
        one_hot[greedy_token(logits) as usize] = 1.0;
        return Ok(one_hot);
    }

    let scaled: Vec<f32> = logits.iter().map(|&x| x / temperature).collect();
    let max_val = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scaled.iter().map(|&x| (x - max_val).exp()).collect();
    let sum_exp: f32 = exps.iter().sum();
    let mut probs: Vec<f32> = exps.iter().map(|&e| e / sum_exp).collect();

    if top_p < 1.0 {
        let mut order: Vec<usize> = (0..probs.len()).collect();
        order.sort_by(|&a, &b| {
            probs[b]
                .partial_cmp(&probs[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut kept = Vec::new();
        let mut cumulative = 0.0;
        for &idx in &order {
            kept.push(idx);
            cumulative += probs[idx];
            if cumulative >= top_p {
                break;
            }
        }
        let kept_sum: f32 = kept.iter().map(|&idx| probs[idx]).sum();
        let mut nucleus = vec![0.0f32; probs.len()];
        for &idx in &kept {
            nucleus[idx] = probs[idx] / kept_sum;
        }
        probs = nucleus;
    }

    Ok(probs)
}

/// Draws one index from a (already normalized) distribution.
pub fn sample_from_probs(probs: &[f32], rng: &mut impl Rng) -> u32 {
    let total: f32 = probs.iter().sum();
    let r: f32 = rng.random_range(0.0..total.max(f32::MIN_POSITIVE));
    let mut acc = 0.0;
    for (idx, &p) in probs.iter().enumerate() {
        acc += p;
        if acc >= r {
            return idx as u32;
        }
    }
    // fp undershoot on the last chunk: fall back to the last index holding mass.
    probs.iter().rposition(|&p| p > 0.0).unwrap_or(0) as u32
}

/// Draws one token from a flat logits row under `temperature`/`top_p`.
///
/// Reproducible by construction (ticket #61 F9): the same row, knobs and `seed` always
/// draw the same token, so a `temperature > 0` request replays exactly.
pub fn sample_logits(
    logits: &[f32],
    temperature: f32,
    top_p: f32,
    seed: u64,
) -> Result<u32, EngineError> {
    if logits.is_empty() {
        return Err(EngineError::Model("Empty logits".to_string()));
    }

    if temperature <= 0.0 {
        return Ok(greedy_token(logits));
    }

    let probs = softmax_probs(logits, temperature, top_p)?;
    Ok(sample_from_probs(&probs, &mut StdRng::seed_from_u64(seed)))
}

pub fn sample_token(
    logits: &Tensor,
    temperature: f32,
    top_p: f32,
    seed: u64,
) -> Result<u32, EngineError> {
    sample_logits(
        &logits.flatten_all()?.to_vec1::<f32>()?,
        temperature,
        top_p,
        seed,
    )
}
