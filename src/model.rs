use candle_core::{DType, Device, IndexOp, Tensor};
use crate::config::{ModelArchitecture, ModelConfig};
use crate::engine::ScratchpadWorkspace;
use crate::error::EngineError;
use crate::kv_cache::{BlockTable, KvStorage};
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

        let causal_mask = if seq_len > 1 {
            let mut mask_data = vec![0.0f32; seq_len * seq_len];
            for i in 0..seq_len {
                for j in 0..seq_len {
                    if j > i {
                        mask_data[i * seq_len + j] = -1e9f32;
                    }
                }
            }
            Some(Tensor::from_vec(mask_data, (1, seq_len, seq_len), &self.device)?)
        } else {
            None
        };

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
            let mut scores = (q.matmul(&k.t()?)? * scale)?;
            if let Some(mask) = &causal_mask {
                scores = scores.broadcast_add(mask)?;
            }
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

    pub fn forward_decode_step(
        &self,
        token: u32,
        pos: usize,
        block_table: &BlockTable,
        kv_storage: &mut KvStorage,
    ) -> Result<Tensor, EngineError> {
        self.forward_decode_step_with_workspace(token, pos, block_table, kv_storage, None)
    }

    pub fn forward_decode_step_with_workspace(
        &self,
        token: u32,
        pos: usize,
        block_table: &BlockTable,
        kv_storage: &mut KvStorage,
        workspace: Option<&mut ScratchpadWorkspace>,
    ) -> Result<Tensor, EngineError> {
        let head_dim = self.config.head_dim();
        let num_q_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;

        // Embedding lookup for single token: [1, hidden_size]
        let token_ids = Tensor::from_slice(&[token], (1,), &self.device)?;
        let mut hidden = self.weights.embed_tokens.index_select(&token_ids, 0)?;

        let block_size = kv_storage.block_size;
        let block_idx = pos / block_size;
        let offset = pos % block_size;

        if block_idx >= block_table.blocks().len() {
            return Err(EngineError::Model(format!(
                "Position {} requires block index {} but block table only has {} blocks",
                pos,
                block_idx,
                block_table.blocks().len()
            )));
        }
        let physical_block = block_table.blocks()[block_idx];

        for (layer_idx, layer) in self.weights.layers.iter().enumerate() {
            // Attention Norm
            let normed = self.rmsnorm(&hidden, &layer.attn_norm)?;

            // Linear Q, K, V for single token
            let q = normed.matmul(&layer.attn_q.t()?)?;
            let k = normed.matmul(&layer.attn_k.t()?)?;
            let v = normed.matmul(&layer.attn_v.t()?)?;

            // Reshape Q: [num_q_heads, 1, head_dim]
            let q = q.reshape((1, num_q_heads, head_dim))?.transpose(0, 1)?;

            // Reshape K, V to store: [num_kv_heads, head_dim]
            let k_token = k.reshape((num_kv_heads, head_dim))?;
            let v_token = v.reshape((num_kv_heads, head_dim))?;

            // Write token K, V directly into physical block in KvStorage
            kv_storage.write_kv_token(layer_idx, physical_block, offset, &k_token, &v_token)?;

            // Gather context directly from physical memory blocks mapped in block table
            let (k_all, v_all) = kv_storage.gather_blocks(
                layer_idx,
                &block_table.blocks()[..=block_idx],
                pos + 1,
            )?;

            // Reshape for attention: [num_kv_heads, pos + 1, head_dim]
            let k_all = k_all.transpose(0, 1)?;
            let v_all = v_all.transpose(0, 1)?;

            // Repeat KV heads if GQA
            let (k_all, v_all) = if num_q_heads > num_kv_heads {
                let factor = num_q_heads / num_kv_heads;
                (k_all.repeat((factor, 1, 1))?, v_all.repeat((factor, 1, 1))?)
            } else {
                (k_all, v_all)
            };

            // Scaled dot-product attention against historical context
            let scale = 1.0 / (head_dim as f64).sqrt();
            let scores = (q.matmul(&k_all.t()?)? * scale)?;
            let attn_weights = candle_nn::ops::softmax_last_dim(&scores)?;
            let attn_out = attn_weights.matmul(&v_all)?;

            // Merge heads back: [1, hidden_size]
            let attn_out = attn_out.transpose(0, 1)?.reshape((1, num_q_heads * head_dim))?;
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

        // LM Head projection: [1, vocab_size]
        let logits = final_normed.matmul(&self.weights.lm_head.t()?)?;

        if let Some(ws) = workspace {
            if let Ok(flat) = logits.flatten_all() {
                if let Ok(slice) = flat.to_vec1::<f32>() {
                    let len = slice.len().min(ws.logits.len());
                    ws.logits[..len].copy_from_slice(&slice[..len]);
                }
            }
        }

        Ok(logits)
    }

    pub fn prefill_prompt(
        &self,
        prompt_tokens: &[u32],
        block_table: &BlockTable,
        kv_storage: &mut KvStorage,
    ) -> Result<Tensor, EngineError> {
        self.prefill_prompt_with_workspace(prompt_tokens, block_table, kv_storage, None)
    }

    pub fn prefill_prompt_with_workspace(
        &self,
        prompt_tokens: &[u32],
        block_table: &BlockTable,
        kv_storage: &mut KvStorage,
        mut workspace: Option<&mut ScratchpadWorkspace>,
    ) -> Result<Tensor, EngineError> {
        if prompt_tokens.is_empty() {
            return Err(EngineError::Model("Empty input tokens".to_string()));
        }
        let mut last_logits = None;
        for (pos, &token) in prompt_tokens.iter().enumerate() {
            let logits = self.forward_decode_step_with_workspace(
                token,
                pos,
                block_table,
                kv_storage,
                workspace.as_deref_mut(),
            )?;
            last_logits = Some(logits);
        }
        Ok(last_logits.unwrap())
    }

    pub fn forward_verify_tokens(
        &self,
        tokens: &[u32],
        start_pos: usize,
        block_table: &BlockTable,
        kv_storage: &mut KvStorage,
    ) -> Result<Tensor, EngineError> {
        self.forward_verify_tokens_with_workspace(tokens, start_pos, block_table, kv_storage, None)
    }

    pub fn forward_verify_tokens_with_workspace(
        &self,
        tokens: &[u32],
        start_pos: usize,
        block_table: &BlockTable,
        kv_storage: &mut KvStorage,
        _workspace: Option<&mut ScratchpadWorkspace>,
    ) -> Result<Tensor, EngineError> {
        let seq_len = tokens.len();
        if seq_len == 0 {
            return Err(EngineError::Model("Empty input tokens".to_string()));
        }

        let head_dim = self.config.head_dim();
        let num_q_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;
        let block_size = kv_storage.block_size;
        let total_tokens = start_pos + seq_len;

        let token_ids = Tensor::from_slice(tokens, (seq_len,), &self.device)?;
        let mut hidden = self.weights.embed_tokens.index_select(&token_ids, 0)?;

        let causal_mask = if seq_len > 1 || total_tokens > 1 {
            let mut mask_data = vec![0.0f32; seq_len * total_tokens];
            for i in 0..seq_len {
                let allowed_key_limit = start_pos + i;
                for j in 0..total_tokens {
                    if j > allowed_key_limit {
                        mask_data[i * total_tokens + j] = -1e9f32;
                    }
                }
            }
            Some(Tensor::from_vec(
                mask_data,
                (1, seq_len, total_tokens),
                &self.device,
            )?)
        } else {
            None
        };

        for (layer_idx, layer) in self.weights.layers.iter().enumerate() {
            let normed = self.rmsnorm(&hidden, &layer.attn_norm)?;

            let q = normed.matmul(&layer.attn_q.t()?)?;
            let k = normed.matmul(&layer.attn_k.t()?)?;
            let v = normed.matmul(&layer.attn_v.t()?)?;

            let q = q.reshape((seq_len, num_q_heads, head_dim))?.transpose(0, 1)?;
            let k_reshaped = k.reshape((seq_len, num_kv_heads, head_dim))?;
            let v_reshaped = v.reshape((seq_len, num_kv_heads, head_dim))?;

            for i in 0..seq_len {
                let pos = start_pos + i;
                let block_idx = pos / block_size;
                let offset = pos % block_size;

                if block_idx >= block_table.blocks().len() {
                    return Err(EngineError::Model(format!(
                        "Position {} requires block index {} but block table only has {} blocks",
                        pos,
                        block_idx,
                        block_table.blocks().len()
                    )));
                }
                let physical_block = block_table.blocks()[block_idx];
                let k_token = k_reshaped.i(i)?;
                let v_token = v_reshaped.i(i)?;
                kv_storage.write_kv_token(layer_idx, physical_block, offset, &k_token, &v_token)?;
            }

            let (k_all, v_all) = kv_storage.gather_blocks(
                layer_idx,
                block_table.blocks(),
                total_tokens,
            )?;

            let k_all = k_all.transpose(0, 1)?;
            let v_all = v_all.transpose(0, 1)?;

            let (k_all, v_all) = if num_q_heads > num_kv_heads {
                let factor = num_q_heads / num_kv_heads;
                (k_all.repeat((factor, 1, 1))?, v_all.repeat((factor, 1, 1))?)
            } else {
                (k_all, v_all)
            };

            let scale = 1.0 / (head_dim as f64).sqrt();
            let mut scores = (q.matmul(&k_all.t()?)? * scale)?;
            if let Some(mask) = &causal_mask {
                scores = scores.broadcast_add(mask)?;
            }
            let attn_weights = candle_nn::ops::softmax_last_dim(&scores)?;
            let attn_out = attn_weights.matmul(&v_all)?;

            let attn_out = attn_out
                .transpose(0, 1)?
                .reshape((seq_len, num_q_heads * head_dim))?;
            let attn_proj = attn_out.matmul(&layer.attn_output.t()?)?;
            hidden = (hidden + attn_proj)?;

            let ffn_normed = self.rmsnorm(&hidden, &layer.ffn_norm)?;
            let gate = ffn_normed.matmul(&layer.ffn_gate.t()?)?;
            let up = ffn_normed.matmul(&layer.ffn_up.t()?)?;
            let silu_gate = candle_nn::ops::silu(&gate)?;
            let ffn_hidden = (silu_gate * up)?;
            let ffn_out = ffn_hidden.matmul(&layer.ffn_down.t()?)?;
            hidden = (hidden + ffn_out)?;
        }

        let final_normed = self.rmsnorm(&hidden, &self.weights.norm)?;
        let logits = final_normed.matmul(&self.weights.lm_head.t()?)?;
        Ok(logits)
    }
}
