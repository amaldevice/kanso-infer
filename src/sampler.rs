use crate::error::EngineError;
use candle_core::Tensor;
use rand::Rng;

pub fn sample_token(
    logits: &Tensor,
    temperature: f32,
    top_p: f32,
    _seed: u64,
) -> Result<u32, EngineError> {
    let logits_vec = logits.flatten_all()?.to_vec1::<f32>()?;
    if logits_vec.is_empty() {
        return Err(EngineError::Model("Empty logits".to_string()));
    }

    if temperature <= 0.0 {
        // Greedy argmax
        let mut max_idx = 0;
        let mut max_val = f32::NEG_INFINITY;
        for (i, &val) in logits_vec.iter().enumerate() {
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }
        return Ok(max_idx as u32);
    }

    // Softmax with temperature
    let scaled: Vec<f32> = logits_vec.iter().map(|&x| x / temperature).collect();
    let max_val = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scaled.iter().map(|&x| (x - max_val).exp()).collect();
    let sum_exp: f32 = exps.iter().sum();
    let mut probs: Vec<(usize, f32)> = exps
        .iter()
        .enumerate()
        .map(|(i, &e)| (i, e / sum_exp))
        .collect();

    // Sort descending by probability
    probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Top-p truncation
    let mut cumulative = 0.0;
    let mut filtered = Vec::new();
    for (idx, p) in probs {
        filtered.push((idx, p));
        cumulative += p;
        if cumulative >= top_p {
            break;
        }
    }

    // Renormalize and sample
    let filtered_sum: f32 = filtered.iter().map(|(_, p)| p).sum();
    let mut rng = rand::rng();
    let r: f32 = rng.random_range(0.0..filtered_sum);

    let mut acc = 0.0;
    for (idx, p) in filtered {
        acc += p;
        if acc >= r {
            return Ok(idx as u32);
        }
    }

    Ok(0)
}
