use crate::config::{ModelArchitecture, ModelConfig};
use crate::engine::ScratchpadWorkspace;
use crate::error::EngineError;
use crate::kv_cache::{BlockTable, KvStorage};
use crate::weights::{LayerWeights, ModelWeights};
use candle_core::{DType, Device, IndexOp, Tensor};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
pub struct Model {
    pub weights: ModelWeights,
    pub config: ModelConfig,
    pub device: Device,
    pub backend: String,
    pub hybrid_pipeline: Option<crate::ane::HybridExecutionPipeline>,
}

/// Fixed seed for mock-model weights so tests are reproducible.
/// NOTE: candle's CPU RNG cannot be seeded (`Device::set_seed` bails on CPU
/// and `Tensor::randn` draws from a thread RNG), so we draw N(0,1) ourselves
/// via Box-Muller from a seeded `StdRng` and build tensors with `from_vec`.
const MOCK_SEED: u64 = 0x5EED_0057;

/// Builds a tensor filled with standard-normal samples (Box-Muller) from a seeded RNG.
fn mock_randn_tensor(
    rng: &mut StdRng,
    dev: &Device,
    shape: Vec<usize>,
) -> Result<Tensor, EngineError> {
    let n: usize = shape.iter().product();
    Ok(Tensor::from_vec(mock_randn_vec(n, rng), shape, dev)?)
}

/// Draws `n` standard-normal samples (Box-Muller) from a seeded RNG.
fn mock_randn_vec(n: usize, rng: &mut StdRng) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let u1: f64 = rng.random_range(f64::MIN_POSITIVE..1.0);
        let u2: f64 = rng.random_range(0.0..1.0);
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        out.push((r * theta.cos()) as f32);
        if out.len() < n {
            out.push((r * theta.sin()) as f32);
        }
    }
    out
}

impl Model {
    pub fn new(weights: ModelWeights, config: ModelConfig, device: Device) -> Self {
        Self {
            weights,
            config,
            device,
            backend: crate::hardware::DEFAULT_BACKEND.to_string(),
            hybrid_pipeline: None,
        }
    }

    /// Loads a model directly from a real GGUF file.
    /// Fails fast if the file does not exist. Shadow/mock models are disabled.
    pub fn load_gguf<P: AsRef<std::path::Path>>(
        path: P,
        device: Device,
    ) -> Result<Self, EngineError> {
        let path_ref = path.as_ref();
        let resolved = crate::weights::resolve_model_path(path_ref.to_str().unwrap_or(""))?;
        let (weights, config) = ModelWeights::load_from_gguf(&resolved, &device)?;
        Ok(Self::new(weights, config, device))
    }

    pub fn backend(&self) -> &str {
        &self.backend
    }

    pub fn set_backend(&mut self, backend: impl Into<String>) {
        self.backend = backend.into();
    }

    pub fn set_hybrid_pipeline(&mut self, pipeline: crate::ane::HybridExecutionPipeline) {
        self.hybrid_pipeline = Some(pipeline);
    }

    pub fn enable_ane_hybrid(&mut self) {
        let pipeline = crate::ane::HybridExecutionPipeline::from_model(self);
        self.hybrid_pipeline = Some(pipeline);
    }

    pub fn forward_mlp_block(
        &self,
        layer_idx: usize,
        ffn_normed: &Tensor,
        layer: &LayerWeights,
    ) -> Result<Tensor, EngineError> {
        if let Some(pipeline) = &self.hybrid_pipeline {
            pipeline.forward_layer_mlp(layer_idx, ffn_normed)
        } else {
            let gate = self.matmul(ffn_normed, &layer.ffn_gate.t()?)?;
            let up = self.matmul(ffn_normed, &layer.ffn_up.t()?)?;
            let silu_gate = candle_nn::ops::silu(&gate)?;
            let ffn_hidden = (silu_gate * up)?;
            self.matmul(&ffn_hidden, &layer.ffn_down.t()?)
        }
    }

    pub fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor, EngineError> {
        if self.backend.eq_ignore_ascii_case("amx") {
            crate::accelerate::amx::amx_matmul(a, b)
        } else {
            Ok(a.matmul(b)?)
        }
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
        let mut rng = StdRng::seed_from_u64(MOCK_SEED);
        let mut randn = |shape: Vec<usize>| mock_randn_tensor(&mut rng, &dev, shape);

        let mut layers = Vec::new();
        for _ in 0..num_layers {
            let layer = LayerWeights {
                attn_q: randn(vec![num_attention_heads * head_dim, hidden_size])?,
                attn_k: randn(vec![num_kv_heads * head_dim, hidden_size])?,
                attn_v: randn(vec![num_kv_heads * head_dim, hidden_size])?,
                attn_output: randn(vec![hidden_size, num_attention_heads * head_dim])?,
                attn_norm: Tensor::ones((hidden_size,), DType::F32, &dev)?,
                ffn_gate: randn(vec![intermediate_size, hidden_size])?,
                ffn_up: randn(vec![intermediate_size, hidden_size])?,
                ffn_down: randn(vec![hidden_size, intermediate_size])?,
                ffn_norm: Tensor::ones((hidden_size,), DType::F32, &dev)?,
            };
            layers.push(layer);
        }

        let weights = ModelWeights {
            embed_tokens: mock_randn_tensor(&mut rng, &dev, vec![vocab_size, hidden_size])?,
            layers,
            norm: Tensor::ones((hidden_size,), DType::F32, &dev)?,
            lm_head: mock_randn_tensor(&mut rng, &dev, vec![vocab_size, hidden_size])?,
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
            rope_theta: 10000.0,
        };

        Ok(Self::new(weights, config, dev))
    }

    pub fn create_mock_from_config(config: &ModelConfig) -> Result<Self, EngineError> {
        let dev = Device::Cpu;
        let hidden_size = config.hidden_size;
        let num_attention_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim();
        let intermediate_size = config.intermediate_size;
        let num_layers = config.num_hidden_layers.min(2);
        let vocab_size = config.vocab_size.min(1000);
        let mut rng = StdRng::seed_from_u64(MOCK_SEED);
        let mut randn = |shape: Vec<usize>| mock_randn_tensor(&mut rng, &dev, shape);

        let mut layers = Vec::new();
        for _ in 0..num_layers {
            let layer = LayerWeights {
                attn_q: randn(vec![num_attention_heads * head_dim, hidden_size])?,
                attn_k: randn(vec![num_kv_heads * head_dim, hidden_size])?,
                attn_v: randn(vec![num_kv_heads * head_dim, hidden_size])?,
                attn_output: randn(vec![hidden_size, num_attention_heads * head_dim])?,
                attn_norm: Tensor::ones((hidden_size,), DType::F32, &dev)?,
                ffn_gate: randn(vec![intermediate_size, hidden_size])?,
                ffn_up: randn(vec![intermediate_size, hidden_size])?,
                ffn_down: randn(vec![hidden_size, intermediate_size])?,
                ffn_norm: Tensor::ones((hidden_size,), DType::F32, &dev)?,
            };
            layers.push(layer);
        }

        let weights = ModelWeights {
            embed_tokens: mock_randn_tensor(&mut rng, &dev, vec![vocab_size, hidden_size])?,
            layers,
            norm: Tensor::ones((hidden_size,), DType::F32, &dev)?,
            lm_head: mock_randn_tensor(&mut rng, &dev, vec![vocab_size, hidden_size])?,
        };

        let mut cfg = config.clone();
        cfg.num_hidden_layers = num_layers;
        cfg.vocab_size = vocab_size;

        Ok(Self::new(weights, cfg, dev))
    }

    pub fn rmsnorm(&self, x: &Tensor, weight: &Tensor) -> Result<Tensor, EngineError> {
        let sq = x.sqr()?;
        let mean_sq = sq.mean_keepdim(candle_core::D::Minus1)?;
        let rms = (mean_sq + self.config.rms_norm_eps)?.sqrt()?;
        let normalized = x.broadcast_div(&rms)?;
        Ok(normalized.broadcast_mul(weight)?)
    }

    pub fn rope_cos_sin(&self, positions: &[usize]) -> Result<(Tensor, Tensor), EngineError> {
        let head_dim = self.config.head_dim();
        let half = head_dim / 2;
        let theta: Vec<f32> = (0..half)
            .map(|j| {
                self.config
                    .rope_theta
                    .powf(-2.0 * j as f32 / head_dim as f32)
            })
            .collect();
        let theta = Tensor::from_vec(theta, (1, half), &self.device)?;

        let positions_f: Vec<f32> = positions.iter().map(|&p| p as f32).collect();
        let positions = Tensor::from_vec(positions_f, (positions.len(), 1), &self.device)?;

        let angles = positions.matmul(&theta)?;
        Ok((angles.cos()?, angles.sin()?))
    }

    pub fn rope_apply(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor, EngineError> {
        let half = x.dim(candle_core::D::Minus1)? / 2;
        let first = x.narrow(candle_core::D::Minus1, 0, half)?;
        let second = x.narrow(candle_core::D::Minus1, half, half)?;

        let first_rot = (first.broadcast_mul(cos)? - second.broadcast_mul(sin)?)?;
        let second_rot = (first.broadcast_mul(sin)? + second.broadcast_mul(cos)?)?;

        Ok(Tensor::cat(
            &[&first_rot, &second_rot],
            candle_core::D::Minus1,
        )?)
    }

    pub fn gqa_attention(
        &self,
        q: &Tensor,            // [n_heads, q_len, head_dim]
        k: &Tensor,            // [n_kv_heads, ctx_len, head_dim]
        v: &Tensor,            // [n_kv_heads, ctx_len, head_dim]
        mask: Option<&Tensor>, // optional [q_len, ctx_len]
    ) -> Result<Tensor, EngineError> {
        let (n_heads, q_len, head_dim) = q.dims3()?;
        let (n_kv_heads, ctx_len, _) = k.dims3()?;
        let group_size = n_heads / n_kv_heads;

        // Fold Q heads into groups: [n_kv_heads, group_size * q_len, head_dim]
        let q_folded = q
            .contiguous()?
            .reshape((n_kv_heads, group_size * q_len, head_dim))?;

        // Attention scores: [n_kv_heads, group_size * q_len, ctx_len]
        let scale = 1.0 / (head_dim as f64).sqrt();
        let mut scores = (self.matmul(&q_folded, &k.t()?)? * scale)?;

        if let Some(m) = mask {
            let scores_unfolded = scores.reshape((n_kv_heads, group_size, q_len, ctx_len))?;
            let masked = scores_unfolded.broadcast_add(m)?;
            scores = masked.reshape((n_kv_heads, group_size * q_len, ctx_len))?;
        }

        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let attn_out = self.matmul(&probs, v)?; // [n_kv_heads, group_size * q_len, head_dim]

        // Reshape back to [n_heads, q_len, head_dim]
        Ok(attn_out.reshape((n_heads, q_len, head_dim))?)
    }

    pub fn forward_tokens(&self, tokens: &[u32]) -> Result<Tensor, EngineError> {
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
        let positions: Vec<usize> = (0..seq_len).collect();
        let (cos, sin) = self.rope_cos_sin(&positions)?;
        let cos = cos.reshape((seq_len, 1, head_dim / 2))?;
        let sin = sin.reshape((seq_len, 1, head_dim / 2))?;

        let causal_mask = if seq_len > 1 {
            let mut mask_data = vec![0.0f32; seq_len * seq_len];
            for i in 0..seq_len {
                for j in 0..seq_len {
                    if j > i {
                        mask_data[i * seq_len + j] = -1e9f32;
                    }
                }
            }
            Some(Tensor::from_vec(
                mask_data,
                (1, 1, seq_len, seq_len),
                &self.device,
            )?)
        } else {
            None
        };

        for (layer_idx, layer) in self.weights.layers.iter().enumerate() {
            // Attention Norm
            let normed = self.rmsnorm(&hidden, &layer.attn_norm)?;

            // Linear Q, K, V: [seq_len, hidden_size] @ W^T -> [seq_len, out_features]
            let q = self.matmul(&normed, &layer.attn_q.t()?)?;
            let k = self.matmul(&normed, &layer.attn_k.t()?)?;
            let v = self.matmul(&normed, &layer.attn_v.t()?)?;

            let q = q.reshape((seq_len, num_q_heads, head_dim))?;
            let k = k.reshape((seq_len, num_kv_heads, head_dim))?;
            let v = v.reshape((seq_len, num_kv_heads, head_dim))?;

            let q = self.rope_apply(&q, &cos, &sin)?.transpose(0, 1)?;
            let k = self.rope_apply(&k, &cos, &sin)?.transpose(0, 1)?;
            let v = v.transpose(0, 1)?;
            let attn_out = self.gqa_attention(&q, &k, &v, causal_mask.as_ref())?;
            // Merge heads back: [num_q_heads, seq_len, head_dim] -> [seq_len, hidden_size]
            let attn_out = attn_out
                .transpose(0, 1)?
                .reshape((seq_len, num_q_heads * head_dim))?;
            let attn_proj = self.matmul(&attn_out, &layer.attn_output.t()?)?;

            // Residual connection
            hidden = (hidden + attn_proj)?;

            // FFN Norm
            let ffn_normed = self.rmsnorm(&hidden, &layer.ffn_norm)?;

            let ffn_out = self.forward_mlp_block(layer_idx, &ffn_normed, layer)?;
            // Residual connection
            hidden = (hidden + ffn_out)?;
        }

        // Final RMSNorm
        let final_normed = self.rmsnorm(&hidden, &self.weights.norm)?;

        // LM Head projection for all tokens: [seq_len, hidden_size] @ W_head^T -> [seq_len, vocab_size]
        let logits = self.matmul(&final_normed, &self.weights.lm_head.t()?)?;
        Ok(logits)
    }

    pub fn forward_single_prompt(&self, tokens: &[u32]) -> Result<Tensor, EngineError> {
        let logits = self.forward_tokens(tokens)?;
        let seq_len = tokens.len();
        Ok(logits.i(seq_len - 1)?.unsqueeze(0)?)
    }

    /// Batched prefill pass over a prompt token chunk.
    /// Computes full-chunk embeddings and linear projections in a single GEMM pass,
    /// stores Key and Value tokens into physical blocks in `kv_storage`,
    /// and returns logits for the final token in O(1) step time.
    pub fn forward_prefill(
        &self,
        tokens: &[u32],
        start_pos: usize,
        block_table: &BlockTable,
        kv_storage: &mut KvStorage,
    ) -> Result<Tensor, EngineError> {
        let chunk_len = tokens.len();
        if chunk_len == 0 {
            return Err(EngineError::Model("Empty prefill chunk tokens".to_string()));
        }

        let token_ids = Tensor::from_slice(tokens, (chunk_len,), &self.device)?;
        let mut hidden = self.weights.embed_tokens.index_select(&token_ids, 0)?;

        let head_dim = self.config.head_dim();
        let num_q_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;
        let block_size = kv_storage.block_size;

        let positions: Vec<usize> = (start_pos..start_pos + chunk_len).collect();
        let (cos, sin) = self.rope_cos_sin(&positions)?;
        let cos = cos.reshape((chunk_len, 1, head_dim / 2))?;
        let sin = sin.reshape((chunk_len, 1, head_dim / 2))?;

        let total_len = start_pos + chunk_len;
        // Offset causal mask [chunk_len, total_len]: query i at absolute
        // position start_pos + i may attend keys 0..=start_pos+i (#43).
        let causal_mask = if chunk_len > 1 || total_len > 1 {
            let mut mask_data = vec![0.0f32; chunk_len * total_len];
            for i in 0..chunk_len {
                let allowed_key_limit = start_pos + i;
                for j in 0..total_len {
                    if j > allowed_key_limit {
                        mask_data[i * total_len + j] = -1e9f32;
                    }
                }
            }
            Some(Tensor::from_vec(
                mask_data,
                (1, 1, chunk_len, total_len),
                &self.device,
            )?)
        } else {
            None
        };

        for (layer_idx, layer) in self.weights.layers.iter().enumerate() {
            let normed = self.rmsnorm(&hidden, &layer.attn_norm)?;

            let q = self.matmul(&normed, &layer.attn_q.t()?)?;
            let k = self.matmul(&normed, &layer.attn_k.t()?)?;
            let v = self.matmul(&normed, &layer.attn_v.t()?)?;

            let q = q.reshape((chunk_len, num_q_heads, head_dim))?;
            let k = k.reshape((chunk_len, num_kv_heads, head_dim))?;
            let v = v.reshape((chunk_len, num_kv_heads, head_dim))?;

            let q = self.rope_apply(&q, &cos, &sin)?.transpose(0, 1)?;
            let k = self.rope_apply(&k, &cos, &sin)?;

            // Write chunk tokens into physical blocks
            for i in 0..chunk_len {
                let pos = start_pos + i;
                let b_idx = pos / block_size;
                let offset = pos % block_size;
                if b_idx < block_table.blocks().len() {
                    let physical_block = block_table.blocks()[b_idx];
                    let k_i = k.i(i)?.reshape((num_kv_heads, head_dim))?;
                    let v_i = v.i(i)?.reshape((num_kv_heads, head_dim))?;
                    kv_storage.write_kv_token(layer_idx, physical_block, offset, &k_i, &v_i)?;
                }
            }

            // Gather full K/V context 0..start_pos+chunk_len (prior chunks +
            // current chunk just written above) so attention sees the whole
            // prefix, not only the intra-chunk slice (#43).
            let (k_all, v_all) =
                kv_storage.gather_blocks(layer_idx, block_table.blocks(), total_len)?;
            let k_all = k_all.transpose(0, 1)?;
            let v_all = v_all.transpose(0, 1)?;

            let attn_out = self.gqa_attention(&q, &k_all, &v_all, causal_mask.as_ref())?;
            let attn_out = attn_out
                .transpose(0, 1)?
                .reshape((chunk_len, num_q_heads * head_dim))?;
            let attn_proj = self.matmul(&attn_out, &layer.attn_output.t()?)?;

            hidden = (hidden + attn_proj)?;

            let ffn_normed = self.rmsnorm(&hidden, &layer.ffn_norm)?;
            let ffn_out = self.forward_mlp_block(layer_idx, &ffn_normed, layer)?;
            hidden = (hidden + ffn_out)?;
        }

        let final_normed = self.rmsnorm(&hidden, &self.weights.norm)?;
        let last_hidden = final_normed.i(chunk_len - 1)?.unsqueeze(0)?;
        let logits = self.matmul(&last_hidden, &self.weights.lm_head.t()?)?;
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

        let (cos, sin) = self.rope_cos_sin(&[pos])?;
        let cos = cos.reshape((1, 1, head_dim / 2))?;
        let sin = sin.reshape((1, 1, head_dim / 2))?;

        for (layer_idx, layer) in self.weights.layers.iter().enumerate() {
            // Attention Norm
            let normed = self.rmsnorm(&hidden, &layer.attn_norm)?;

            // Linear Q, K, V for single token
            let q = self.matmul(&normed, &layer.attn_q.t()?)?;
            let k = self.matmul(&normed, &layer.attn_k.t()?)?;
            let v = self.matmul(&normed, &layer.attn_v.t()?)?;

            let q = q.reshape((1, num_q_heads, head_dim))?;
            let k = k.reshape((1, num_kv_heads, head_dim))?;
            let v = v.reshape((1, num_kv_heads, head_dim))?;

            let q = self.rope_apply(&q, &cos, &sin)?.transpose(0, 1)?;
            let k = self.rope_apply(&k, &cos, &sin)?;

            let k_token = k.reshape((num_kv_heads, head_dim))?;
            let v_token = v.reshape((num_kv_heads, head_dim))?;

            // Write roped token K, V directly into physical block in KvStorage
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

            let attn_out = self.gqa_attention(&q, &k_all, &v_all, None)?;
            // Merge heads back: [1, hidden_size]
            let attn_out = attn_out
                .transpose(0, 1)?
                .reshape((1, num_q_heads * head_dim))?;
            let attn_proj = self.matmul(&attn_out, &layer.attn_output.t()?)?;

            // Residual connection
            hidden = (hidden + attn_proj)?;

            // FFN Norm
            let ffn_normed = self.rmsnorm(&hidden, &layer.ffn_norm)?;

            let ffn_out = self.forward_mlp_block(layer_idx, &ffn_normed, layer)?;
            // Residual connection
            hidden = (hidden + ffn_out)?;
        }

        // Final RMSNorm
        let final_normed = self.rmsnorm(&hidden, &self.weights.norm)?;

        // LM Head projection: [1, vocab_size]
        let logits = self.matmul(&final_normed, &self.weights.lm_head.t()?)?;

        if let Some(ws) = workspace
            && let Ok(flat) = logits.flatten_all()
            && let Ok(slice) = flat.to_vec1::<f32>()
        {
            let len = slice.len().min(ws.logits.len());
            ws.logits[..len].copy_from_slice(&slice[..len]);
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

            let q = self.matmul(&normed, &layer.attn_q.t()?)?;
            let k = self.matmul(&normed, &layer.attn_k.t()?)?;
            let v = self.matmul(&normed, &layer.attn_v.t()?)?;

            let q = q
                .reshape((seq_len, num_q_heads, head_dim))?
                .transpose(0, 1)?;
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

            let (k_all, v_all) =
                kv_storage.gather_blocks(layer_idx, block_table.blocks(), total_tokens)?;

            let k_all = k_all.transpose(0, 1)?;
            let v_all = v_all.transpose(0, 1)?;

            let (k_all, v_all) = if num_q_heads > num_kv_heads {
                let factor = num_q_heads / num_kv_heads;
                (k_all.repeat((factor, 1, 1))?, v_all.repeat((factor, 1, 1))?)
            } else {
                (k_all, v_all)
            };

            let scale = 1.0 / (head_dim as f64).sqrt();
            let mut scores = (self.matmul(&q, &k_all.t()?)? * scale)?;
            if let Some(mask) = &causal_mask {
                scores = scores.broadcast_add(mask)?;
            }
            let attn_weights = candle_nn::ops::softmax_last_dim(&scores)?;
            let attn_out = self.matmul(&attn_weights, &v_all)?;

            let attn_out = attn_out
                .transpose(0, 1)?
                .reshape((seq_len, num_q_heads * head_dim))?;
            let attn_proj = self.matmul(&attn_out, &layer.attn_output.t()?)?;
            hidden = (hidden + attn_proj)?;

            let ffn_normed = self.rmsnorm(&hidden, &layer.ffn_norm)?;
            let ffn_out = self.forward_mlp_block(layer_idx, &ffn_normed, layer)?;
            hidden = (hidden + ffn_out)?;
        }

        let final_normed = self.rmsnorm(&hidden, &self.weights.norm)?;
        let logits = self.matmul(&final_normed, &self.weights.lm_head.t()?)?;
        Ok(logits)
    }
}
