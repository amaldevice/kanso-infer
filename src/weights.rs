use candle_core::Tensor;
use crate::error::EngineError;

#[derive(Clone)]
pub struct LayerWeights {
    pub attn_q: Tensor,
    pub attn_k: Tensor,
    pub attn_v: Tensor,
    pub attn_output: Tensor,
    pub attn_norm: Tensor,
    pub ffn_gate: Tensor,
    pub ffn_up: Tensor,
    pub ffn_down: Tensor,
    pub ffn_norm: Tensor,
}

#[derive(Clone)]
pub struct ModelWeights {
    pub embed_tokens: Tensor,
    pub layers: Vec<LayerWeights>,
    pub norm: Tensor,
    pub lm_head: Tensor,
}

/// Permutes Q/K weight matrix from llama.cpp's interleaved layout [x0, y0, x1, y1, ...]
/// to the standard rotate-half layout [x0, x1, ..., y0, y1, ...] used by custom kernels.
pub fn permute_interleaved_to_rotate_half(
    weight: &Tensor,
    num_heads: usize,
    head_dim: usize,
) -> Result<Tensor, EngineError> {
    let dev = weight.device();
    let rows = num_heads * head_dim;
    let _cols = weight.dims()[1];

    let mut permutation = Vec::with_capacity(rows);
    let half = head_dim / 2;

    for h in 0..num_heads {
        let base = h * head_dim;
        // Evens first
        for i in 0..half {
            permutation.push((base + 2 * i) as u32);
        }
        // Odds next
        for i in 0..half {
            permutation.push((base + 2 * i + 1) as u32);
        }
    }

    let idx = Tensor::from_vec(permutation, (rows,), dev)?;
    let permuted = weight.index_select(&idx, 0)?;
    Ok(permuted)
}
