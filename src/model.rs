use candle_core::{DType, Device, IndexOp, Tensor};
use crate::config::{ModelArchitecture, ModelConfig};
use crate::error::EngineError;
use crate::weights::{LayerWeights, ModelWeights};

pub struct Model {
    pub weights: ModelWeights,
    pub config: ModelConfig,
    pub device: Device,
}

impl Model {
    pub fn new(weights: ModelWeights, config: ModelConfig, device: Device) -> Self {
        Self { weights, config, device }
    }

    pub fn create_mock_tiny(
        hidden_size: usize,
        num_attention_heads: usize,
        num_kv_heads: usize,
        num_layers: usize,
        vocab_size: usize,
    ) -> Result<Self, EngineError> {
        let dev = Device::Cpu;
        let head_dim = hidden_size / num_attention_heads;
        let intermediate_size = hidden_size * 2;

        let mut layers = Vec::new();
        for _ in 0..num_layers {
            let layer = LayerWeights {
                attn_q: Tensor::ones((num_attention_heads * head_dim, hidden_size), DType::F32, &dev)?,
                attn_k: Tensor::ones((num_kv_heads * head_dim, hidden_size), DType::F32, &dev)?,
                attn_v: Tensor::ones((num_kv_heads * head_dim, hidden_size), DType::F32, &dev)?,
                attn_output: Tensor::ones((hidden_size, num_attention_heads * head_dim), DType::F32, &dev)?,
                attn_norm: Tensor::ones((hidden_size,), DType::F32, &dev)?,
                ffn_gate: Tensor::ones((intermediate_size, hidden_size), DType::F32, &dev)?,
                ffn_up: Tensor::ones((intermediate_size, hidden_size), DType::F32, &dev)?,
                ffn_down: Tensor::ones((hidden_size, intermediate_size), DType::F32, &dev)?,
                ffn_norm: Tensor::ones((hidden_size,), DType::F32, &dev)?,
            };
            layers.push(layer);
        }

        let weights = ModelWeights {
            embed_tokens: Tensor::ones((vocab_size, hidden_size), DType::F32, &dev)?,
            layers,
            norm: Tensor::ones((hidden_size,), DType::F32, &dev)?,
            lm_head: Tensor::ones((vocab_size, hidden_size), DType::F32, &dev)?,
        };

        let config = ModelConfig {
            hidden_size,
            num_attention_heads,
            num_key_value_heads: num_kv_heads,
            num_hidden_layers: num_layers,
            intermediate_size,
            vocab_size,
            rms_norm_eps: 1e-5,
            block_size: 16,
            max_seq_len: 128,
            architecture: ModelArchitecture::DenseLlama,
            attention_layers: None,
        };

        Ok(Self::new(weights, config, dev))
    }

    fn rmsnorm(&self, x: &Tensor, weight: &Tensor) -> Result<Tensor, EngineError> {
        let sq = x.sqr()?;
        let mean_sq = sq.mean_keepdim(candle_core::D::Minus1)?;
        let rms = (mean_sq + self.config.rms_norm_eps)?.sqrt()?;
        let normalized = x.broadcast_div(&rms)?;
        Ok(normalized.broadcast_mul(weight)?)
    }

    pub fn forward_single_prompt(&self, tokens: &[u32]) -> Result<Tensor, EngineError> {
        let seq_len = tokens.len();
        if seq_len == 0 {
            return Err(EngineError::Model("Empty input tokens".to_string()));
        }

        // Embedding lookup
        let token_ids = Tensor::from_slice(tokens, (seq_len,), &self.device)?;
        let mut hidden = self.weights.embed_tokens.index_select(&token_ids, 0)?;

        let head_dim = self.config.head_dim();
        let num_q_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;

        for layer in &self.weights.layers {
            // Attention Norm
            let normed = self.rmsnorm(&hidden, &layer.attn_norm)?;

            // Linear Q, K, V: [seq_len, hidden_size] @ W^T -> [seq_len, out_features]
            let q = normed.matmul(&layer.attn_q.t()?)?;
            let k = normed.matmul(&layer.attn_k.t()?)?;
            let v = normed.matmul(&layer.attn_v.t()?)?;

            // Reshape for multi-head: [seq_len, num_heads, head_dim] -> [num_heads, seq_len, head_dim]
            let q = q.reshape((seq_len, num_q_heads, head_dim))?.transpose(0, 1)?;
            let k = k.reshape((seq_len, num_kv_heads, head_dim))?.transpose(0, 1)?;
            let v = v.reshape((seq_len, num_kv_heads, head_dim))?.transpose(0, 1)?;

            // Repeat KV heads if GQA (Grouped-Query Attention)
            let (k, v) = if num_q_heads > num_kv_heads {
                let factor = num_q_heads / num_kv_heads;
                let k_rep = k.repeat((factor, 1, 1))?;
                let v_rep = v.repeat((factor, 1, 1))?;
                (k_rep, v_rep)
            } else {
                (k, v)
            };

            // Scaled dot-product attention
            let scale = 1.0 / (head_dim as f64).sqrt();
            let scores = (q.matmul(&k.t()?)? * scale)?;
            let attn_weights = candle_nn::ops::softmax_last_dim(&scores)?;
            let attn_out = attn_weights.matmul(&v)?;

            // Merge heads back: [num_q_heads, seq_len, head_dim] -> [seq_len, hidden_size]
            let attn_out = attn_out.transpose(0, 1)?.reshape((seq_len, num_q_heads * head_dim))?;
            let attn_proj = attn_out.matmul(&layer.attn_output.t()?)?;

            // Residual connection
            hidden = (hidden + attn_proj)?;

            // FFN Norm
            let ffn_normed = self.rmsnorm(&hidden, &layer.ffn_norm)?;

            // SwiGLU: silu(x @ W_gate^T) * (x @ W_up^T) @ W_down^T
            let gate = ffn_normed.matmul(&layer.ffn_gate.t()?)?;
            let up = ffn_normed.matmul(&layer.ffn_up.t()?)?;
            let silu_gate = candle_nn::ops::silu(&gate)?;
            let ffn_hidden = (silu_gate * up)?;
            let ffn_out = ffn_hidden.matmul(&layer.ffn_down.t()?)?;

            // Residual connection
            hidden = (hidden + ffn_out)?;
        }

        // Final RMSNorm
        let final_normed = self.rmsnorm(&hidden, &self.weights.norm)?;

        // Only take the last token embedding for next-token prediction
        let last_hidden = final_normed.i(seq_len - 1)?.unsqueeze(0)?;

        // LM Head projection: [1, hidden_size] @ W_head^T -> [1, vocab_size]
        let logits = last_hidden.matmul(&self.weights.lm_head.t()?)?;
        Ok(logits)
    }
}
