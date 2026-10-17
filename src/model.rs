use crate::config::{ModelArchitecture, ModelConfig};
use crate::error::EngineError;
use crate::hardware::Backend;
use crate::kv_cache::{BlockTable, KvStorage};
use crate::weights::{LayerWeights, LinearWeight, ModelWeights};
use candle_core::{DType, Device, IndexOp, Tensor};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::cell::RefCell;

/// Precomputed RoPE rows for positions `0..len` (ticket #48): decode indexes this
/// instead of rebuilding the theta vector, the angle matmul and both cos/sin tensors
/// on every step. Column count is `head_dim / 2`.
struct RopeTable {
    len: usize,
    cos: Tensor,
    sin: Tensor,
}

/// Everything a pure-decode batch's attention needs from the block geometry, built once
/// per step instead of once per layer (ticket #61 F11): the padded physical block index
/// per sequence, the `-1e9` padding mask shared by every KV head and layer, and the
/// shapes those indices imply.
struct DecodeLayout {
    /// `[num_seqs * max_blocks]` physical block ids, padded to the batch's longest table.
    indices: Tensor,
    /// `[num_seqs * num_kv_heads, 1, padded_ctx]` additive attention mask.
    mask: Tensor,
    num_seqs: usize,
    padded_ctx: usize,
}

pub struct Model {
    pub weights: ModelWeights,
    pub config: ModelConfig,
    pub device: Device,
    pub backend: crate::hardware::Backend,
    pub hybrid_pipeline: Option<crate::ane::HybridExecutionPipeline>,
    /// #45: batched forward passes issued so far. One per engine step is the whole
    /// point of continuous batching — a per-sequence loop shows up here as a count
    /// that grows with the batch size.
    ///
    /// It counts `forward_batch` only (ticket #61 F6): speculative verification
    /// (`forward_verify_tokens`) and draft decode steps serve the speculative loop, not
    /// the batched serving path, and are deliberately not counted here.
    forward_batch_calls: std::sync::atomic::AtomicU64,
    /// Grown on demand, dropped by [`Model::to_device`] because its rows live on the
    /// device the model was built for.
    rope_table: RefCell<Option<RopeTable>>,
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
            backend: crate::hardware::Backend::default(),
            hybrid_pipeline: None,
            forward_batch_calls: std::sync::atomic::AtomicU64::new(0),
            rope_table: RefCell::new(None),
        }
    }

    /// Batched forwards (`forward_batch`) issued so far (#45 acceptance hook).
    pub fn forward_batch_calls(&self) -> u64 {
        self.forward_batch_calls
            .load(std::sync::atomic::Ordering::Relaxed)
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

    pub fn backend(&self) -> crate::hardware::Backend {
        self.backend
    }

    pub fn set_backend(&mut self, backend: crate::hardware::Backend) {
        self.backend = backend;
    }

    /// Moves every weight tensor plus the device tag onto `device`.
    /// Needed for the honest Metal path: GGUF loads on CPU, then moves whole.
    pub fn to_device(&mut self, device: &Device) -> Result<(), EngineError> {
        // Quantized blocks are rebuilt on the target device, not expanded to F32.
        self.weights = self.weights.to_device(device)?;
        self.device = device.clone();
        self.rope_table.borrow_mut().take();
        Ok(())
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
            let gate = self.linear(&layer.ffn_gate, ffn_normed)?;
            let up = self.linear(&layer.ffn_up, ffn_normed)?;
            let silu_gate = candle_nn::ops::silu(&gate)?;
            let ffn_hidden = (silu_gate * up)?;
            self.linear(&layer.ffn_down, &ffn_hidden)
        }
    }

    /// Projects `x` through a stored weight (`x @ W^T`).
    ///
    /// Quantized weights run off their own block storage; only dense weights take
    /// the AMX GEMM path, which needs materialized F32 (`as_dense_f32` is `None`
    /// for quantized storage, so quantized decode bypasses AMX entirely — the
    /// mock-dense tok/s test does not cover real GGUF weights).
    pub fn linear(&self, w: &LinearWeight, x: &Tensor) -> Result<Tensor, EngineError> {
        if matches!(self.backend, crate::hardware::Backend::Amx)
            && let Some(dense) = w.as_dense_f32()
        {
            return crate::accelerate::amx::amx_matmul(x, &dense.t()?);
        }
        w.forward(x)
    }

    /// Dequantizes an embedding row per token id instead of the whole table.
    pub fn embed(&self, ids: &Tensor) -> Result<Tensor, EngineError> {
        self.weights.embed_tokens.embed(ids)
    }

    /// Q/K/V projections for `normed`, with Q/K put into rotate-half head order.
    ///
    /// The GGUF layout permutation belongs to the weights (`QkPerm`), so every
    /// path that projects Q/K shares this helper and none of them can drift.
    pub fn project_qkv(
        &self,
        layer: &LayerWeights,
        normed: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor), EngineError> {
        let (q_perm, k_perm) = match &self.weights.qk_perm {
            Some(p) => (Some(&p.q), Some(&p.k)),
            None => (None, None),
        };
        let q = self.linear(&layer.attn_q, normed)?;
        let k = self.linear(&layer.attn_k, normed)?;
        let q = match q_perm {
            Some(idx) => q.index_select(idx, 1)?,
            None => q,
        };
        let k = match k_perm {
            Some(idx) => k.index_select(idx, 1)?,
            None => k,
        };
        Ok((q, k, self.linear(&layer.attn_v, normed)?))
    }

    pub fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor, EngineError> {
        if matches!(self.backend, crate::hardware::Backend::Amx) {
            crate::accelerate::amx::amx_matmul(a, b)
        } else if matches!(self.backend, crate::hardware::Backend::Metal) {
            // Candle's MLX GEMM only accepts contiguous (or simply
            // transposed) operands; double-transposed views like
            // `transpose(0,1).t()` from the GQA path are rejected, so
            // normalize first. No-op copy when already contiguous.
            //
            // Only the MLX GEMM needs this: CPU-resident tensors go to candle's
            // CPU matmul, which streams a strided operand without copying it, and
            // forcing contiguity there would copy the whole attention history per
            // layer per token (ticket #48).
            //
            // Exempt from the #48 no-materialization claim (ticket #61 F4): on Metal
            // this *does* copy the `[ctx, n_kv_heads, head_dim]` pool view per layer
            // per token — `gpu_paged_attention_v2` (`src/metal/ops.rs`) is the wired
            // fix, and it has no production caller yet. Metal decode therefore
            // remains bounded-alloc but not copy-free.
            if a.device().is_metal() {
                Ok(a.contiguous()?.matmul(&b.contiguous()?)?)
            } else {
                Ok(a.matmul(b)?)
            }
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
                attn_q: LinearWeight::from_tensor(randn(vec![
                    num_attention_heads * head_dim,
                    hidden_size,
                ])?),
                attn_k: LinearWeight::from_tensor(randn(vec![
                    num_kv_heads * head_dim,
                    hidden_size,
                ])?),
                attn_v: LinearWeight::from_tensor(randn(vec![
                    num_kv_heads * head_dim,
                    hidden_size,
                ])?),
                attn_output: LinearWeight::from_tensor(randn(vec![
                    hidden_size,
                    num_attention_heads * head_dim,
                ])?),
                attn_norm: Tensor::ones((hidden_size,), DType::F32, &dev)?,
                ffn_gate: LinearWeight::from_tensor(randn(vec![intermediate_size, hidden_size])?),
                ffn_up: LinearWeight::from_tensor(randn(vec![intermediate_size, hidden_size])?),
                ffn_down: LinearWeight::from_tensor(randn(vec![hidden_size, intermediate_size])?),
                ffn_norm: Tensor::ones((hidden_size,), DType::F32, &dev)?,
            };
            layers.push(layer);
        }

        let weights = ModelWeights {
            embed_tokens: LinearWeight::from_tensor(mock_randn_tensor(
                &mut rng,
                &dev,
                vec![vocab_size, hidden_size],
            )?),
            layers,
            norm: Tensor::ones((hidden_size,), DType::F32, &dev)?,
            lm_head: LinearWeight::from_tensor(mock_randn_tensor(
                &mut rng,
                &dev,
                vec![vocab_size, hidden_size],
            )?),
            // Synthetic weights are generated directly in rotate-half order.
            qk_perm: None,
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
                attn_q: LinearWeight::from_tensor(randn(vec![
                    num_attention_heads * head_dim,
                    hidden_size,
                ])?),
                attn_k: LinearWeight::from_tensor(randn(vec![
                    num_kv_heads * head_dim,
                    hidden_size,
                ])?),
                attn_v: LinearWeight::from_tensor(randn(vec![
                    num_kv_heads * head_dim,
                    hidden_size,
                ])?),
                attn_output: LinearWeight::from_tensor(randn(vec![
                    hidden_size,
                    num_attention_heads * head_dim,
                ])?),
                attn_norm: Tensor::ones((hidden_size,), DType::F32, &dev)?,
                ffn_gate: LinearWeight::from_tensor(randn(vec![intermediate_size, hidden_size])?),
                ffn_up: LinearWeight::from_tensor(randn(vec![intermediate_size, hidden_size])?),
                ffn_down: LinearWeight::from_tensor(randn(vec![hidden_size, intermediate_size])?),
                ffn_norm: Tensor::ones((hidden_size,), DType::F32, &dev)?,
            };
            layers.push(layer);
        }

        let weights = ModelWeights {
            embed_tokens: LinearWeight::from_tensor(mock_randn_tensor(
                &mut rng,
                &dev,
                vec![vocab_size, hidden_size],
            )?),
            layers,
            norm: Tensor::ones((hidden_size,), DType::F32, &dev)?,
            lm_head: LinearWeight::from_tensor(mock_randn_tensor(
                &mut rng,
                &dev,
                vec![vocab_size, hidden_size],
            )?),
            // Synthetic weights are generated directly in rotate-half order.
            qk_perm: None,
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

    /// RoPE cos/sin for `positions`, read out of a table computed once per position
    /// range (ticket #48) instead of rebuilding theta, the angle matmul and both
    /// cos/sin tensors on every decode step.
    ///
    /// The table grows in 1024-row steps and is dropped by [`Model::to_device`], so it
    /// never outlives its device.
    pub fn rope_cos_sin(&self, positions: &[usize]) -> Result<(Tensor, Tensor), EngineError> {
        let head_dim = self.config.head_dim();
        let max_pos = positions.iter().copied().max().unwrap_or(0);

        {
            let mut table = self.rope_table.borrow_mut();
            if table.as_ref().is_none_or(|t| t.len <= max_pos) {
                let len = (max_pos + 1).next_multiple_of(1024);
                *table = Some(self.build_rope_table(len, head_dim)?);
            }
        }

        let index = Tensor::from_vec(
            positions.iter().map(|&p| p as u32).collect::<Vec<u32>>(),
            (positions.len(),),
            &self.device,
        )?;
        let table = self.rope_table.borrow();
        let table = table.as_ref().ok_or_else(|| {
            // The block above just built it for `max_pos`; missing here means the table was
            // dropped (device move) between the two borrows.
            EngineError::Model("rope table missing after build".to_string())
        })?;
        Ok((
            table.cos.index_select(&index, 0)?,
            table.sin.index_select(&index, 0)?,
        ))
    }

    /// Fills `[len, head_dim / 2]` cos/sin rows with the same math the per-step path
    /// used, so results are unchanged row for row.
    fn build_rope_table(&self, len: usize, head_dim: usize) -> Result<RopeTable, EngineError> {
        let half = head_dim / 2;
        let theta: Vec<f32> = (0..half)
            .map(|j| {
                self.config
                    .rope_theta
                    .powf(-2.0 * j as f32 / head_dim as f32)
            })
            .collect();
        let theta = Tensor::from_vec(theta, (1, half), &self.device)?;

        let positions: Vec<f32> = (0..len).map(|p| p as f32).collect();
        let positions = Tensor::from_vec(positions, (len, 1), &self.device)?;

        let angles = positions.matmul(&theta)?;
        Ok(RopeTable {
            len,
            cos: angles.cos()?,
            sin: angles.sin()?,
        })
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

        let probs = candle_nn::ops::softmax_last_dim(&scores.contiguous()?)?;
        let attn_out = self.matmul(&probs, v)?; // [n_kv_heads, group_size * q_len, head_dim]

        // Reshape back to [n_heads, q_len, head_dim]
        Ok(attn_out.reshape((n_heads, q_len, head_dim))?)
    }

    /// Single-token decode attention over the K/V pool read (ticket #48).
    ///
    /// `k`/`v` are `[ctx, n_kv_heads, head_dim]` — pool views when the block table is
    /// one consecutive run, the fragmented gather otherwise. The AMX backend feeds those
    /// rows straight into Accelerate with the pool's own row stride, so the history is
    /// never copied; every other backend runs the same math through `gqa_attention`
    /// (candle's CPU and metal matmul stream the strided operand without re-copying it).
    /// Returns `[n_heads, 1, head_dim]`.
    ///
    /// Two paths fall outside the #48 claim, both documented rather than papered over
    /// (ticket #61 F4): a Metal tensor reaching `matmul` materializes the history (see
    /// [`Model::matmul`]), and the AMX fallback below copies once when Accelerate rejects
    /// the layout.
    fn decode_attention_paged(
        &self,
        q: &Tensor, // [n_heads, 1, head_dim]
        k: &Tensor, // [ctx, n_kv_heads, head_dim]
        v: &Tensor, // [ctx, n_kv_heads, head_dim]
    ) -> Result<Tensor, EngineError> {
        if matches!(self.backend, Backend::Amx) {
            let (n_heads, q_len, head_dim) = q.dims3()?;
            let (_, n_kv_heads, _) = k.dims3()?;
            let group_size = n_heads / n_kv_heads;
            let q_folded = q
                .contiguous()?
                .reshape((n_kv_heads, group_size * q_len, head_dim))?;
            let scale = 1.0 / (head_dim as f64).sqrt();
            if let Some(out) =
                crate::accelerate::amx::amx_paged_decode_attention(&q_folded, k, v, scale as f32)
            {
                // [n_kv_heads, group * q_len, head_dim] -> [n_heads, q_len, head_dim]
                return Ok(out?.reshape((n_heads, q_len, head_dim))?);
            }
            // Unusable layout (a stride Accelerate cannot read): one contiguous copy of the
            // history, then the generic path. Exempt from the #48 zero-copy claim — it is
            // the rare fallback, not the decode path the ticket measures (ticket #61 F4).
            // ponytail: ctx-sized copy here; a strided fallback GEMM if this ever shows up
            // in a profile.
            let (k_all, v_all) = (
                k.transpose(0, 1)?.contiguous()?,
                v.transpose(0, 1)?.contiguous()?,
            );
            return self.gqa_attention(q, &k_all, &v_all, None);
        }
        let (k_all, v_all) = (k.transpose(0, 1)?, v.transpose(0, 1)?);
        self.gqa_attention(q, &k_all, &v_all, None)
    }

    pub fn forward_tokens(&self, tokens: &[u32]) -> Result<Tensor, EngineError> {
        let seq_len = tokens.len();
        if seq_len == 0 {
            return Err(EngineError::Model("Empty input tokens".to_string()));
        }

        // Embedding lookup
        let token_ids = Tensor::from_slice(tokens, (seq_len,), &self.device)?;
        let mut hidden = self.embed(&token_ids)?;

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
            let (q, k, v) = self.project_qkv(layer, &normed)?;

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
            let attn_proj = self.linear(&layer.attn_output, &attn_out)?;

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
        let logits = self.linear(&self.weights.lm_head, &final_normed)?;
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
        let mut hidden = self.embed(&token_ids)?;

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

            let (q, k, v) = self.project_qkv(layer, &normed)?;

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
            let attn_proj = self.linear(&layer.attn_output, &attn_out)?;

            hidden = (hidden + attn_proj)?;

            let ffn_normed = self.rmsnorm(&hidden, &layer.ffn_norm)?;
            let ffn_out = self.forward_mlp_block(layer_idx, &ffn_normed, layer)?;
            hidden = (hidden + ffn_out)?;
        }

        let final_normed = self.rmsnorm(&hidden, &self.weights.norm)?;
        let last_hidden = final_normed.i(chunk_len - 1)?.unsqueeze(0)?;
        let logits = self.linear(&self.weights.lm_head, &last_hidden)?;
        Ok(logits)
    }

    pub fn forward_decode_step(
        &self,
        token: u32,
        pos: usize,
        block_table: &BlockTable,
        kv_storage: &mut KvStorage,
    ) -> Result<Tensor, EngineError> {
        let head_dim = self.config.head_dim();
        let num_q_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;

        // Embedding lookup for single token: [1, hidden_size]
        let token_ids = Tensor::from_slice(&[token], (1,), &self.device)?;
        let mut hidden = self.embed(&token_ids)?;

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
            let (q, k, v) = self.project_qkv(layer, &normed)?;

            let q = q.reshape((1, num_q_heads, head_dim))?;
            let k = k.reshape((1, num_kv_heads, head_dim))?;
            let v = v.reshape((1, num_kv_heads, head_dim))?;

            let q = self.rope_apply(&q, &cos, &sin)?.transpose(0, 1)?;
            let k = self.rope_apply(&k, &cos, &sin)?;

            let k_token = k.reshape((num_kv_heads, head_dim))?;
            let v_token = v.reshape((num_kv_heads, head_dim))?;

            // Write roped token K, V directly into physical block in KvStorage
            kv_storage.write_kv_token(layer_idx, physical_block, offset, &k_token, &v_token)?;

            // #48: attention reads the history straight out of the block pool through the
            // block table — a view of the pool storage when the table is one consecutive
            // run — instead of re-gathering the whole history into a fresh tensor per
            // layer per token.
            let (k_tokens, v_tokens) = kv_storage.paged_kv_tokens(
                layer_idx,
                &block_table.blocks()[..=block_idx],
                pos + 1,
            )?;

            let attn_out = self.decode_attention_paged(&q, &k_tokens, &v_tokens)?;
            // Merge heads back: [1, hidden_size]
            let attn_out = attn_out
                .transpose(0, 1)?
                .reshape((1, num_q_heads * head_dim))?;
            let attn_proj = self.linear(&layer.attn_output, &attn_out)?;

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
        let logits = self.linear(&self.weights.lm_head, &final_normed)?;

        Ok(logits)
    }

    pub fn forward_verify_tokens(
        &self,
        tokens: &[u32],
        start_pos: usize,
        block_table: &BlockTable,
        kv_storage: &mut KvStorage,
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
        let mut hidden = self.embed(&token_ids)?;

        // #55: this path used to skip RoPE entirely, so its Q/K were unrotated while the
        // K it *wrote* into the pool came from the decode path's rotated rows — the target
        // then disagreed with its own decode step on the same prefix (max |Δlogits| ~30 on
        // a mock model), which capped speculative acceptance at chance level.
        let positions: Vec<usize> = (start_pos..total_tokens).collect();
        let (cos, sin) = self.rope_cos_sin(&positions)?;
        let cos = cos.reshape((seq_len, 1, head_dim / 2))?;
        let sin = sin.reshape((seq_len, 1, head_dim / 2))?;

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

            let (q, k, v) = self.project_qkv(layer, &normed)?;

            let q = q.reshape((seq_len, num_q_heads, head_dim))?;
            let k = k.reshape((seq_len, num_kv_heads, head_dim))?;
            let v = v.reshape((seq_len, num_kv_heads, head_dim))?;

            let q = self.rope_apply(&q, &cos, &sin)?.transpose(0, 1)?;
            let k = self.rope_apply(&k, &cos, &sin)?;

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
                let k_token = k.i(i)?;
                let v_token = v.i(i)?;
                kv_storage.write_kv_token(layer_idx, physical_block, offset, &k_token, &v_token)?;
            }

            // #55: read the history straight out of the block pool (no gather copy, #48)
            // and let `gqa_attention` fold the query heads into the KV groups — the old
            // `k_all.repeat((factor, 1, 1))` materialized K/V × group_size per layer.
            let (k_all, v_all) =
                kv_storage.paged_kv_tokens(layer_idx, block_table.blocks(), total_tokens)?;
            let (k_all, v_all) = (k_all.transpose(0, 1)?, v_all.transpose(0, 1)?);

            let attn_out = self.gqa_attention(&q, &k_all, &v_all, causal_mask.as_ref())?;

            let attn_out = attn_out
                .transpose(0, 1)?
                .reshape((seq_len, num_q_heads * head_dim))?;
            let attn_proj = self.linear(&layer.attn_output, &attn_out)?;
            hidden = (hidden + attn_proj)?;

            let ffn_normed = self.rmsnorm(&hidden, &layer.ffn_norm)?;
            let ffn_out = self.forward_mlp_block(layer_idx, &ffn_normed, layer)?;
            hidden = (hidden + ffn_out)?;
        }

        let final_normed = self.rmsnorm(&hidden, &self.weights.norm)?;
        let logits = self.linear(&self.weights.lm_head, &final_normed)?;
        Ok(logits)
    }

    /// One batched forward for a whole engine step (#45).
    ///
    /// `tokens`, `positions` and `slot_mapping` are the step's new tokens flattened
    /// across every sequence — decode tokens and prefill-chunk tokens in any mix.
    /// `slot_mapping[i] = physical_block * block_size + offset` is where token `i`'s
    /// K/V is scattered in `kv_storage`; `block_tables[s]`/`context_lens[s]`/
    /// `query_lens[s]` describe sequence `s`: its physical block list, the KV length
    /// it attends once this batch is written, and how many of the flattened tokens
    /// are its own (contiguous, ascending positions).
    ///
    /// Every linear runs once over the `[T, hidden]` stack, so the weights are read
    /// once per step rather than once per sequence. Attention reads each sequence's
    /// context straight from its physical blocks — in one padded pass when every
    /// sequence contributes a single decode token, per sequence otherwise.
    /// Returns logits `[num_seqs, vocab]`: one row per sequence's last new token.
    ///
    /// This is the *serving* path's only forward: the "one model forward per step"
    /// property (#45) is a property of `Engine::step` over non-speculative traffic.
    /// Speculative decoding is deliberately outside it (ticket #61 F3) — the verifier runs
    /// `forward_verify_tokens` for the candidate block and the draft spends one
    /// `forward_decode_step` per proposal, neither of which goes through here, which is why
    /// [`Model::forward_batch_calls`] counts this call alone.
    #[allow(clippy::too_many_arguments)] // #45: the issue's flat-batch signature
    pub fn forward_batch(
        &self,
        tokens: &[u32],
        positions: &[usize],
        slot_mapping: &[usize],
        kv_storage: &mut KvStorage,
        block_tables: &[Vec<usize>],
        context_lens: &[usize],
        query_lens: &[usize],
    ) -> Result<Tensor, EngineError> {
        self.forward_batch_calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        let num_seqs = block_tables.len();
        let total: usize = query_lens.iter().sum();
        if num_seqs == 0 {
            return Err(EngineError::Model("forward_batch: empty batch".to_string()));
        }
        if total == 0
            || tokens.len() != total
            || positions.len() != total
            || slot_mapping.len() != total
            || context_lens.len() != num_seqs
            || query_lens.len() != num_seqs
        {
            return Err(EngineError::Model(format!(
                "forward_batch: shape mismatch (tokens {}, positions {}, slots {}, seqs {}, \
                 context_lens {}, query_lens {}, total tokens {})",
                tokens.len(),
                positions.len(),
                slot_mapping.len(),
                num_seqs,
                context_lens.len(),
                query_lens.len(),
                total
            )));
        }

        let head_dim = self.config.head_dim();
        let num_q_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;

        let token_ids = Tensor::from_slice(tokens, (total,), &self.device)?;
        let mut hidden = self.embed(&token_ids)?;

        let (cos, sin) = self.rope_cos_sin(positions)?;
        let cos = cos.reshape((total, 1, head_dim / 2))?;
        let sin = sin.reshape((total, 1, head_dim / 2))?;

        // Each sequence's slice of the flattened batch, plus its offset causal mask
        // (query i of sequence s sits at absolute position `positions[offset + i]`).
        // Pure decode batches attend in one padded pass and need no mask at all.
        let pure_decode = query_lens.iter().all(|&q_len| q_len == 1);
        let mut offsets = Vec::with_capacity(num_seqs);
        let mut masks: Vec<Option<Tensor>> = Vec::with_capacity(num_seqs);
        let mut offset = 0usize;
        for s in 0..num_seqs {
            let q_len = query_lens[s];
            let ctx_len = context_lens[s];
            if q_len == 0 || ctx_len == 0 {
                return Err(EngineError::Model(format!(
                    "forward_batch: sequence {} has query_len {} / context_len {}",
                    s, q_len, ctx_len
                )));
            }
            let start_pos = positions[offset];
            // The mask below is built from `start_pos` alone, so the sequence's flattened
            // tokens must actually sit at `start_pos + i` (ticket #61 F10): a caller that
            // hands over a hole would otherwise attend with the wrong causal limit.
            for i in 0..q_len {
                if positions[offset + i] != start_pos + i {
                    return Err(EngineError::Model(format!(
                        "forward_batch: sequence {} token {} is at position {}, expected {}",
                        s,
                        i,
                        positions[offset + i],
                        start_pos + i
                    )));
                }
            }
            offsets.push(offset);
            masks.push(if !pure_decode && (q_len > 1 || ctx_len > 1) {
                let mut mask_data = vec![0.0f32; q_len * ctx_len];
                for i in 0..q_len {
                    let allowed_key_limit = start_pos + i;
                    for j in 0..ctx_len {
                        if j > allowed_key_limit {
                            mask_data[i * ctx_len + j] = -1e9f32;
                        }
                    }
                }
                Some(Tensor::from_vec(
                    mask_data,
                    (1, 1, q_len, ctx_len),
                    &self.device,
                )?)
            } else {
                None
            });
            offset += q_len;
        }

        // Per-step, not per-layer (ticket #61 F11): the scatter index tensor and the padded
        // decode layout (block indices + `-1e9` mask) depend only on the step's shapes and
        // block tables, so `num_layers` rebuilds of each are pure waste.
        let slot_ids = kv_storage.slot_id_tensor(slot_mapping, &self.device)?;
        let decode_layout = pure_decode
            .then(|| self.decode_layout(kv_storage, block_tables, context_lens))
            .transpose()?;

        for (layer_idx, layer) in self.weights.layers.iter().enumerate() {
            let normed = self.rmsnorm(&hidden, &layer.attn_norm)?;
            let (q, k, v) = self.project_qkv(layer, &normed)?;

            let q = q.reshape((total, num_q_heads, head_dim))?;
            let k = k.reshape((total, num_kv_heads, head_dim))?;
            let v = v.reshape((total, num_kv_heads, head_dim))?;

            let q = self.rope_apply(&q, &cos, &sin)?;
            let k = self.rope_apply(&k, &cos, &sin)?;

            // Scatter this step's K/V into the physical blocks the scheduler mapped.
            kv_storage.write_kv_slots(layer_idx, &slot_ids, &k, &v)?;

            let attn_out = match &decode_layout {
                Some(layout) => {
                    // One query token per sequence: the whole batch attends in a single
                    // padded GEMM instead of one attention pass per sequence (which is
                    // what per-op overhead makes expensive at small batch sizes).
                    // ponytail: each sequence is padded to the longest context in the batch,
                    // so ragged batches waste lanes; a varlen kernel drops the padding.
                    self.batch_decode_attention(&q, kv_storage, layer_idx, layout)?
                }
                None => {
                    // Steps that mix prefill chunks with decode tokens attend per sequence
                    // instead (those steps are latency- not throughput-critical).
                    // ponytail: per-sequence attention costs one gather + a dozen ops per
                    // sequence; a varlen/paged kernel over the whole batch removes that.
                    let mut attn_rows = Vec::with_capacity(num_seqs);
                    for s in 0..num_seqs {
                        let q_s = q.narrow(0, offsets[s], query_lens[s])?.transpose(0, 1)?;
                        let (k_all, v_all) = kv_storage.gather_blocks(
                            layer_idx,
                            &block_tables[s],
                            context_lens[s],
                        )?;
                        let attn = self.gqa_attention(
                            &q_s,
                            &k_all.transpose(0, 1)?,
                            &v_all.transpose(0, 1)?,
                            masks[s].as_ref(),
                        )?;
                        attn_rows.push(
                            attn.transpose(0, 1)?
                                .reshape((query_lens[s], num_q_heads * head_dim))?,
                        );
                    }
                    Tensor::cat(&attn_rows.iter().collect::<Vec<_>>(), 0)?
                }
            };
            let attn_proj = self.linear(&layer.attn_output, &attn_out)?;
            hidden = (hidden + attn_proj)?;

            let ffn_normed = self.rmsnorm(&hidden, &layer.ffn_norm)?;
            let ffn_out = self.forward_mlp_block(layer_idx, &ffn_normed, layer)?;
            hidden = (hidden + ffn_out)?;
        }

        let final_normed = self.rmsnorm(&hidden, &self.weights.norm)?;

        // Only each sequence's last new token needs logits (that is the token the
        // caller samples; partial prefill chunks discard theirs).
        let last_indices: Vec<u32> = (0..num_seqs)
            .map(|s| (offsets[s] + query_lens[s] - 1) as u32)
            .collect();
        let last_indices = Tensor::from_vec(last_indices, (num_seqs,), &self.device)?;
        let last_hidden = final_normed.index_select(&last_indices, 0)?;
        self.linear(&self.weights.lm_head, &last_hidden)
    }

    /// Attention for a whole decode batch in a handful of tensor ops.
    ///
    /// Every sequence contributes exactly one query token (`q` is
    /// `[num_seqs, num_q_heads, head_dim]`, row `s` = sequence `s`). Each sequence's
    /// context is gathered from its own physical blocks and padded to the longest
    /// context in the batch — padding lanes repeat a real block and are masked out
    /// with `-1e9`, so they contribute exactly zero to the softmax.
    /// Returns `[num_seqs, num_q_heads * head_dim]`.
    ///
    /// The layout (`layout`) is the same for every layer and is built once per step in
    /// `forward_batch` (ticket #61 F11); only the K/V it indexes change per layer.
    fn batch_decode_attention(
        &self,
        q: &Tensor,
        kv_storage: &KvStorage,
        layer_idx: usize,
        layout: &DecodeLayout,
    ) -> Result<Tensor, EngineError> {
        let head_dim = self.config.head_dim();
        let num_q_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;
        let group = num_q_heads / num_kv_heads;
        let num_seqs = layout.num_seqs;
        let padded_ctx = layout.padded_ctx;

        // The two GEMMs below are 3D (batch = sequence x KV head), which keeps them on
        // the accelerated path instead of the 4D fallback. Both right-hand sides have to
        // be contiguous: candle's CPU kernel rejects a permuted one outright.
        let k = kv_storage.kv_k[layer_idx]
            .index_select(&layout.indices, 0)?
            .reshape((num_seqs, padded_ctx, num_kv_heads, head_dim))?
            .permute((0, 2, 3, 1))?
            .contiguous()?
            .reshape((num_seqs * num_kv_heads, head_dim, padded_ctx))?;
        let v = kv_storage.kv_v[layer_idx]
            .index_select(&layout.indices, 0)?
            .reshape((num_seqs, padded_ctx, num_kv_heads, head_dim))?
            .permute((0, 2, 1, 3))?
            .contiguous()?
            .reshape((num_seqs * num_kv_heads, padded_ctx, head_dim))?;

        let q = q
            .reshape((num_seqs, num_kv_heads, group, head_dim))?
            .reshape((num_seqs * num_kv_heads, group, head_dim))?;
        let scale = 1.0 / (head_dim as f64).sqrt();
        let scores = (self.matmul(&q, &k)? * scale)?;
        let scores = scores.broadcast_add(&layout.mask)?;
        let probs = candle_nn::ops::softmax_last_dim(&scores.contiguous()?)?;
        let out = self.matmul(&probs, &v)?;

        // out is [num_seqs, num_kv_heads, group, head_dim], which flattens to the
        // query-head order this model uses everywhere (`head = kv_head * group + group_i`).
        Ok(out.reshape((num_seqs, num_q_heads * head_dim))?)
    }

    /// Padded block indices plus the `-1e9` padding mask for a pure-decode batch: what
    /// every sequence's context is read with, and the same for every layer (ticket #61 F11).
    fn decode_layout(
        &self,
        kv_storage: &KvStorage,
        block_tables: &[Vec<usize>],
        context_lens: &[usize],
    ) -> Result<DecodeLayout, EngineError> {
        let num_kv_heads = self.config.num_key_value_heads;
        let num_seqs = block_tables.len();
        let block_size = kv_storage.block_size;

        let max_blocks = context_lens
            .iter()
            .map(|&ctx| ctx.div_ceil(block_size))
            .max()
            .unwrap_or(1);
        let padded_ctx = max_blocks * block_size;

        let mut indices = Vec::with_capacity(num_seqs * max_blocks);
        let mut mask_data = vec![0.0f32; num_seqs * num_kv_heads * padded_ctx];
        for s in 0..num_seqs {
            let blocks = &block_tables[s];
            let needed = context_lens[s].div_ceil(block_size).max(1);
            if blocks.len() < needed {
                return Err(EngineError::Model(format!(
                    "forward_batch: sequence {} attends {} tokens but its block table has \
                     {} of the {} blocks needed",
                    s,
                    context_lens[s],
                    blocks.len(),
                    needed
                )));
            }
            for b in 0..max_blocks {
                indices.push(blocks[b.min(needed - 1)] as u32);
            }
            for g in 0..num_kv_heads {
                let row = (s * num_kv_heads + g) * padded_ctx;
                for j in context_lens[s]..padded_ctx {
                    mask_data[row + j] = -1e9f32;
                }
            }
        }

        Ok(DecodeLayout {
            indices: Tensor::from_vec(indices, (num_seqs * max_blocks,), &self.device)?,
            mask: Tensor::from_vec(
                mask_data,
                (num_seqs * num_kv_heads, 1, padded_ctx),
                &self.device,
            )?,
            num_seqs,
            padded_ctx,
        })
    }
}
