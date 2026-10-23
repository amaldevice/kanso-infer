use crate::config::{ModelArchitecture, ModelConfig};
use crate::error::EngineError;
use crate::hardware::Backend;
use crate::kv_cache::{BlockTable, KvSlotPlan, KvStorageKind, PaddedBatch};
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
    /// The same ids on the host, for the quantized pool's per-block dequantized gather.
    blocks: Vec<u32>,
    /// Blocks per sequence in `indices` / `blocks`.
    max_blocks: usize,
    /// `[num_seqs * num_kv_heads, 1, padded_ctx]` additive attention mask.
    mask: Tensor,
    num_seqs: usize,
    padded_ctx: usize,
}

/// What a decode step needs to attend straight out of the block pool (ticket #86): the
/// block geometry the scheduler already built, plus the pool rows the step wrote. One
/// seam — the Metal paged-attention kernel is its only consumer, so no backend grows a
/// second KV read path.
#[derive(Debug, Clone)]
pub struct PagedAttnContext {
    /// Physical block ids, `[num_seqs * max_blocks]`, padded to the batch's longest table.
    pub block_tables: Vec<u32>,
    /// Blocks per sequence in `block_tables`.
    pub max_blocks: usize,
    /// Context length per sequence.
    pub context_lens: Vec<u32>,
    /// Pool row each sequence's query token was just written to.
    pub slots: Vec<usize>,
    /// Blocks erased since the last decode step (ticket #106): the seam's device copy
    /// still holds their old contents, so it copies these blocks whole from the pool.
    pub recycled_blocks: Vec<usize>,
}

/// The block geometry of a pure-decode batch, padded to the batch's longest table.
///
/// One builder for every consumer: the Metal paged-attention kernel, the padded
/// layout's `index_select` ids and the quantized pool's per-block walk all read the
/// same padded table, so a second copy of this padding could disagree with the first
/// about which block a lane points at. Padding repeats a real block (the attention
/// mask zeroes those lanes out), and a short table is an error rather than a clamp —
/// clamping would attend over a block the scheduler never allocated.
struct PaddedDecodeTables {
    block_size: usize,
    max_blocks: usize,
    /// `[num_seqs * max_blocks]` physical block ids, padded.
    blocks: Vec<u32>,
}

impl PaddedDecodeTables {
    fn new(
        block_size: usize,
        block_tables: &[Vec<usize>],
        context_lens: &[usize],
    ) -> Result<Self, EngineError> {
        if block_tables.len() != context_lens.len() {
            return Err(EngineError::Model(format!(
                "decode batch has {} block tables for {} context lengths",
                block_tables.len(),
                context_lens.len()
            )));
        }
        let max_blocks = context_lens
            .iter()
            .map(|&ctx| ctx.div_ceil(block_size).max(1))
            .max()
            .unwrap_or(1);
        let mut blocks = Vec::with_capacity(block_tables.len() * max_blocks);
        for (s, table) in block_tables.iter().enumerate() {
            let needed = context_lens[s].div_ceil(block_size).max(1);
            if table.len() < needed {
                return Err(EngineError::Model(format!(
                    "forward_batch: sequence {} attends {} tokens but its block table has \
                     {} of the {} blocks needed",
                    s,
                    context_lens[s],
                    table.len(),
                    needed
                )));
            }
            for b in 0..max_blocks {
                blocks.push(table[b.min(needed - 1)] as u32);
            }
        }
        Ok(Self {
            block_size,
            max_blocks,
            blocks,
        })
    }

    fn padded_ctx(&self) -> usize {
        self.max_blocks * self.block_size
    }
}

impl PagedAttnContext {
    /// The context of a pure-decode batch: every sequence's block table padded to the
    /// batch's longest, its context length, and the pool row its token landed in.
    fn decode_batch(
        kv_storage: &KvStorageKind,
        block_tables: &[Vec<usize>],
        context_lens: &[usize],
        slot_mapping: &[usize],
        recycled_blocks: Vec<usize>,
    ) -> Result<Self, EngineError> {
        let tables = PaddedDecodeTables::new(kv_storage.block_size(), block_tables, context_lens)?;
        Ok(Self {
            block_tables: tables.blocks,
            max_blocks: tables.max_blocks,
            context_lens: context_lens.iter().map(|&c| c as u32).collect(),
            slots: slot_mapping.to_vec(),
            recycled_blocks,
        })
    }
}

/// Which logit rows a batched forward keeps.
///
/// The serving path samples one token per sequence; the speculative verifier needs the
/// distribution under every candidate (ticket #90). One flag on the shared body, so the
/// two entry points cannot drift in anything but the row selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RowSelection {
    /// `[num_seqs, vocab]`: each sequence's last new token.
    LastPerSequence,
    /// `[total tokens, vocab]`: every new token.
    EveryToken,
}

/// The fused Metal llama-layer kernel on the production decode path (ticket #87).
mod fused_decode;
/// Device copy of the block pool plus the kernel dispatch that reads it (ticket #86).
mod paged_attn;

use fused_decode::FusedDecodeSeam;
pub(crate) use paged_attn::AttnShape;
use paged_attn::PagedAttnSeam;

pub struct Model {
    pub weights: ModelWeights,
    pub config: ModelConfig,
    pub device: Device,
    pub backend: crate::hardware::Backend,
    pub hybrid_pipeline: Option<crate::ffn_offload::HybridExecutionPipeline>,
    /// #45: batched forward passes issued so far. One per engine step is the whole
    /// point of continuous batching — a per-sequence loop shows up here as a count
    /// that grows with the batch size.
    ///
    /// It counts `forward_batch` only: speculative verification rides `forward_batch`
    /// since ticket #90, so speculative steps are counted too. The draft model's decode
    /// steps are not — they are a different model, on its own KV cache.
    forward_batch_calls: std::sync::atomic::AtomicU64,
    /// Grown on demand, dropped by [`Model::to_device`] because its rows live on the
    /// device the model was built for.
    rope_table: RefCell<Option<RopeTable>>,
    /// Reused decode-path matmul outputs (ticket #93 / spec #82 D16): the AMX path fills
    /// these slots in place instead of returning a fresh `Vec` per projection per layer.
    /// Interior mutability because decode takes `&self`; see
    /// [`crate::accelerate::amx::AmxWorkspace`] for the aliasing contract.
    amx_workspace: RefCell<crate::accelerate::amx::AmxWorkspace>,
    /// #86: Metal decode dispatches the fused PagedAttention v2 kernel. Counts the
    /// dispatches so a test can tell the paged path from the copying `matmul` one.
    paged_attention_calls: std::sync::atomic::AtomicU64,
    /// The single seam that lets a decode step attend over the block pool instead of a
    /// per-layer gather: `Some` only once a Metal decode step has run (ticket #86).
    paged_attn: RefCell<Option<PagedAttnSeam>>,
    /// #87: the fused Metal prologue (RMSNorm + QKV + RoPE) as the decode path. `None`
    /// means this model is outside the kernel geometry, or has not decoded on Metal yet,
    /// and the per-op path runs instead.
    fused_decode: RefCell<Option<FusedDecodeSeam>>,
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
            amx_workspace: RefCell::new(crate::accelerate::amx::AmxWorkspace::default()),
            paged_attention_calls: std::sync::atomic::AtomicU64::new(0),
            paged_attn: RefCell::new(None),
            fused_decode: RefCell::new(None),
        }
    }

    /// Fused PagedAttention v2 dispatches this model has issued (ticket #86). A Metal
    /// decode step must move this; every other backend must leave it at zero.
    pub fn paged_attention_calls(&self) -> u64 {
        self.paged_attention_calls
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Layer prologues this model ran through the fused Metal chain (ticket #87).
    ///
    /// Zero on every non-Metal backend, and zero on Metal for a model whose dimensions fall
    /// outside the kernel geometry — which is what makes "the fused kernel is the production
    /// path" an assertion a test can read instead of a claim.
    pub fn fused_layer_calls(&self) -> u64 {
        self.fused_decode
            .borrow()
            .as_ref()
            .map(FusedDecodeSeam::steps)
            .unwrap_or(0)
    }

    /// Command buffers the Metal decode path created, across both decode seams
    /// (ticket #110).
    ///
    /// The first metric that measures the **per-step total**: since #110 the fused
    /// seam spends exactly one command buffer per decode step (prologue, paged
    /// attention, O + residual and SwiGLU MLP + residual for every layer chained
    /// into it), and the paged-attention seam spends none while riding that
    /// buffer — so a fused step moves this by 1. A step on the fallback loop
    /// instead spends one per layer through the paged-attention wrapper, so it
    /// moves by the layer count. What stays outside the metric on purpose
    /// (ADR 0002 scope): the final norm and lm_head run on the tensor path,
    /// submitted by candle on the device queue.
    pub fn metal_decode_command_buffers(&self) -> usize {
        let fused = self
            .fused_decode
            .borrow()
            .as_ref()
            .map(FusedDecodeSeam::command_buffers)
            .unwrap_or(0);
        let paged = self
            .paged_attn
            .borrow()
            .as_ref()
            .map(PagedAttnSeam::command_buffers)
            .unwrap_or(0);
        fused + paged
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
        self.amx_workspace.borrow_mut().clear();
        // The seams' buffers live on the old device (ticket #86 / #87).
        self.paged_attn.borrow_mut().take();
        self.fused_decode.borrow_mut().take();
        Ok(())
    }

    pub fn set_hybrid_pipeline(&mut self, pipeline: crate::ffn_offload::HybridExecutionPipeline) {
        self.hybrid_pipeline = Some(pipeline);
    }

    pub fn enable_ffn_offload(&mut self) {
        let pipeline = crate::ffn_offload::HybridExecutionPipeline::from_model(self);
        self.hybrid_pipeline = Some(pipeline);
    }

    /// [`Model::forward_mlp_block`] with the three projections written into workspace
    /// slots (ticket #93). The MLP is the widest allocation site of a decode step, so it
    /// benefits most from the reused buffers. Falls back to the plain path when a hybrid
    /// pipeline owns the layer.
    fn forward_mlp_block_into(
        &self,
        layer_idx: usize,
        ffn_normed: &Tensor,
        layer: &LayerWeights,
    ) -> Result<Tensor, EngineError> {
        if self.hybrid_pipeline.is_some() {
            return self.forward_mlp_block(layer_idx, ffn_normed, layer);
        }
        use crate::accelerate::amx::AmxWorkspace;
        let gate = self.linear_into(&layer.ffn_gate, ffn_normed, AmxWorkspace::SLOT_FFN_GATE)?;
        let up = self.linear_into(&layer.ffn_up, ffn_normed, AmxWorkspace::SLOT_FFN_UP)?;
        let silu_gate = candle_nn::ops::silu(&gate)?;
        let ffn_hidden = (silu_gate * up)?;
        self.linear_into(&layer.ffn_down, &ffn_hidden, AmxWorkspace::SLOT_FFN_DOWN)
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

    /// [`Model::linear`] writing into AMX workspace `slot` instead of returning a fresh
    /// output buffer (ticket #93 / spec #82 D16).
    ///
    /// The returned tensor aliases `slot`, so the caller must consume it before the next
    /// `linear_into` on the same slot. Only the AMX backend has an in-place GEMM; every
    /// other backend (and quantized storage, which never reaches AMX) delegates to
    /// [`Model::linear`], so the numbers are the same on every path.
    pub fn linear_into(
        &self,
        w: &LinearWeight,
        x: &Tensor,
        slot: usize,
    ) -> Result<Tensor, EngineError> {
        if matches!(self.backend, crate::hardware::Backend::Amx)
            && let Some(dense) = w.as_dense_f32()
        {
            return self
                .amx_workspace
                .borrow_mut()
                .matmul_weight_into(slot, x, dense);
        }
        w.forward(x)
    }

    /// Dequantizes an embedding row per token id instead of the whole table.
    /// Granite `embedding_scale` multiplies the rows (llama.cpp `llama-graph.cpp`).
    pub fn embed(&self, ids: &Tensor) -> Result<Tensor, EngineError> {
        let e = self.weights.embed_tokens.embed(ids)?;
        match self.config.embedding_scale {
            Some(s) => Ok((e * s as f64)?),
            None => Ok(e),
        }
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
        let (q, k) = self.project_qk(layer, normed, None)?;
        Ok((q, k, self.linear(&layer.attn_v, normed)?))
    }

    /// Q/K projections with Q/K in rotate-half head order. When `slots` is given, the
    /// projections are written into those workspace slots (ticket #93) and the returned
    /// tensors alias them.
    fn project_qk(
        &self,
        layer: &LayerWeights,
        normed: &Tensor,
        slots: Option<(usize, usize)>,
    ) -> Result<(Tensor, Tensor), EngineError> {
        let (q_perm, k_perm) = match &self.weights.qk_perm {
            Some(p) => (Some(&p.q), Some(&p.k)),
            None => (None, None),
        };
        let (q, k) = match slots {
            Some((q_slot, k_slot)) => (
                self.linear_into(&layer.attn_q, normed, q_slot)?,
                self.linear_into(&layer.attn_k, normed, k_slot)?,
            ),
            None => (
                self.linear(&layer.attn_q, normed)?,
                self.linear(&layer.attn_k, normed)?,
            ),
        };
        let q = match q_perm {
            Some(idx) => q.index_select(idx, 1)?,
            None => q,
        };
        let k = match k_perm {
            Some(idx) => k.index_select(idx, 1)?,
            None => k,
        };
        Ok((q, k))
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
            // Metal decode attention no longer reaches here: since ticket #86 the decode
            // path dispatches `gpu_paged_attention_v2`, which reads K/V out of the block
            // pool itself, so the `[ctx, n_kv_heads, head_dim]` history is not copied per
            // layer per token. What is left on Metal is prefill and the GEMMs whose
            // operands are already contiguous.
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
            embedding_scale: None,
            attention_scale: None,
            residual_scale: None,
            logit_scale: None,
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

    /// Granite `residual_scale` scales the attention/FFN outputs before each residual add
    /// (llama.cpp `granite.cpp` scales both `cur` branches).
    fn scale_residual(&self, t: Tensor) -> Result<Tensor, EngineError> {
        match self.config.residual_scale {
            Some(s) => Ok((t * s as f64)?),
            None => Ok(t),
        }
    }

    /// Granite `logit_scale` divides the final logits (llama.cpp `granite.cpp`).
    fn scale_logits(&self, t: Tensor) -> Result<Tensor, EngineError> {
        match self.config.logit_scale {
            Some(s) => Ok((t * (1.0 / s as f64))?),
            None => Ok(t),
        }
    }

    /// Granite `attention_scale` replaces `1/sqrt(head_dim)` when nonzero (llama.cpp `granite.cpp`).
    fn attention_scale(&self, head_dim: usize) -> f64 {
        match self.config.attention_scale {
            Some(s) if s != 0.0 => s as f64,
            _ => 1.0 / (head_dim as f64).sqrt(),
        }
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
        let scale = self.attention_scale(head_dim);
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

    /// Single-token decode attention over the K/V pool (ticket #48 / #86).
    ///
    /// `paged` carries the block geometry the scheduler already built. On Metal the fused
    /// paged-attention kernel walks that table itself, so the history is never
    /// materialized per layer; every other backend reads the same pool as a strided
    /// `[ctx, n_kv_heads, head_dim]` view (one consecutive run) or gather, and runs
    /// `gqa_attention` on it — unchanged behaviour.
    /// Returns `[n_heads, 1, head_dim]`.
    ///
    /// One path falls outside the #48 claim, documented rather than papered over (ticket
    /// #61 F4): the AMX fallback copies once when Accelerate rejects the layout.
    fn decode_attention_paged(
        &self,
        q: &Tensor, // [n_heads, 1, head_dim]
        paged: &PagedAttnContext,
        kv_storage: &KvStorageKind,
        layer_idx: usize,
    ) -> Result<Tensor, EngineError> {
        // #86: Metal reads K/V straight out of the block pool through the kernel instead of
        // letting `matmul` materialize the history per layer per token.
        if matches!(self.backend, Backend::Metal) {
            let batched = q.transpose(0, 1)?.contiguous()?; // [1, n_heads, head_dim]
            let out = self.metal_paged_attention(&batched, kv_storage, layer_idx, paged)?;
            return Ok(out.reshape((q.dim(0)?, 1, self.config.head_dim()))?);
        }

        let ctx_len = paged.context_lens[0] as usize;
        let blocks: Vec<usize> = paged.block_tables[..paged.max_blocks]
            .iter()
            .map(|&b| b as usize)
            .collect();
        let (k, v) = kv_storage.paged_kv_tokens(layer_idx, &blocks, ctx_len)?;
        if matches!(self.backend, Backend::Amx) {
            let (n_heads, q_len, head_dim) = q.dims3()?;
            let (_, n_kv_heads, _) = k.dims3()?;
            let group_size = n_heads / n_kv_heads;
            let q_folded = q
                .contiguous()?
                .reshape((n_kv_heads, group_size * q_len, head_dim))?;
            let scale = self.attention_scale(head_dim);
            // Ticket #93: the three scratch buffers this kernel used to allocate per
            // layer per token come from the workspace instead.
            let in_workspace = self.amx_workspace.borrow_mut().paged_decode_attention_into(
                crate::accelerate::amx::AmxWorkspace::SLOT_PAGED_ATTN,
                &q_folded,
                &k,
                &v,
                scale as f32,
            );
            if let Some(out) = in_workspace {
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

    /// Metal decode attention through `gpu_paged_attention_v2_buffer` (ticket #86): the
    /// kernel walks the block table itself, so the `[ctx, n_kv_heads, head_dim]` history is
    /// never materialized per layer.
    ///
    /// `q` is `[num_seqs, n_heads, head_dim]` — one query token per sequence, which is what
    /// a decode step produces. Returns `[num_seqs, n_heads * head_dim]`.
    ///
    /// Kernel parameters the shader cannot represent (head_dim > 256, GQA group > 8) are
    /// `Err` here — a silent fall back to the copying path would hide them.
    pub fn metal_paged_attention(
        &self,
        q: &Tensor,
        kv_storage: &KvStorageKind,
        layer_idx: usize,
        paged: &PagedAttnContext,
    ) -> Result<Tensor, EngineError> {
        // Ticket #88: the kernel reads an F32 pool buffer; a quantized pool is refused
        // here (one place) instead of being dequantized behind the kernel's back.
        let kv_storage = kv_storage.f32_pool()?;
        let head_dim = self.config.head_dim();
        let shape = AttnShape {
            num_heads: self.config.num_attention_heads,
            num_kv_heads: self.config.num_key_value_heads,
            head_dim,
            scale: self.attention_scale(head_dim) as f32,
        };

        let mut seam = self.paged_attn.borrow_mut();
        if seam.is_none() {
            *seam = Some(PagedAttnSeam::new(
                self.config.num_hidden_layers,
                &self.device,
            )?);
        }
        let seam = seam.as_mut().expect("just built");
        let out = seam.attention(
            q,
            kv_storage,
            layer_idx,
            &paged.block_tables,
            paged.max_blocks,
            &paged.context_lens,
            &paged.slots,
            &paged.recycled_blocks,
            &shape,
        )?;
        self.paged_attention_calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(out)
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
            hidden = (hidden + &self.scale_residual(attn_proj)?)?;

            // FFN Norm
            let ffn_normed = self.rmsnorm(&hidden, &layer.ffn_norm)?;

            let ffn_out = self.forward_mlp_block(layer_idx, &ffn_normed, layer)?;
            // Residual connection
            hidden = (hidden + &self.scale_residual(ffn_out)?)?;
        }

        // Final RMSNorm
        let final_normed = self.rmsnorm(&hidden, &self.weights.norm)?;

        // LM Head projection for all tokens: [seq_len, hidden_size] @ W_head^T -> [seq_len, vocab_size]
        let logits = self.scale_logits(self.linear(&self.weights.lm_head, &final_normed)?)?;
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
        kv_storage: &mut KvStorageKind,
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
        let block_size = kv_storage.block_size();

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

            hidden = (hidden + &self.scale_residual(attn_proj)?)?;

            let ffn_normed = self.rmsnorm(&hidden, &layer.ffn_norm)?;
            let ffn_out = self.forward_mlp_block(layer_idx, &ffn_normed, layer)?;
            hidden = (hidden + &self.scale_residual(ffn_out)?)?;
        }

        let final_normed = self.rmsnorm(&hidden, &self.weights.norm)?;
        let last_hidden = final_normed.i(chunk_len - 1)?.unsqueeze(0)?;
        let logits = self.scale_logits(self.linear(&self.weights.lm_head, &last_hidden)?)?;
        Ok(logits)
    }

    pub fn forward_decode_step(
        &self,
        token: u32,
        pos: usize,
        block_table: &BlockTable,
        kv_storage: &mut KvStorageKind,
    ) -> Result<Tensor, EngineError> {
        let head_dim = self.config.head_dim();
        let num_q_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;

        // Embedding lookup for single token: [1, hidden_size]
        let token_ids = Tensor::from_slice(&[token], (1,), &self.device)?;
        let mut hidden = self.embed(&token_ids)?;

        let block_size = kv_storage.block_size();
        let recycled_blocks = if matches!(self.backend, Backend::Metal) {
            kv_storage.take_recycled_blocks()
        } else {
            Vec::new()
        };
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

            // Linear Q, K, V for single token. Q/K/V outputs land in workspace slots
            // (ticket #93): the AMX GEMM writes them in place instead of returning a
            // fresh `[out]` Vec per projection per layer per token.
            let (q, k) = self.project_qk(
                layer,
                &normed,
                Some((
                    crate::accelerate::amx::AmxWorkspace::SLOT_Q,
                    crate::accelerate::amx::AmxWorkspace::SLOT_K,
                )),
            )?;
            let v = self.linear_into(
                &layer.attn_v,
                &normed,
                crate::accelerate::amx::AmxWorkspace::SLOT_V,
            )?;

            let q = q.reshape((1, num_q_heads, head_dim))?;
            let k = k.reshape((1, num_kv_heads, head_dim))?;
            let v = v.reshape((1, num_kv_heads, head_dim))?;

            let q = self.rope_apply(&q, &cos, &sin)?.transpose(0, 1)?;
            let k = self.rope_apply(&k, &cos, &sin)?;

            let k_token = k.reshape((num_kv_heads, head_dim))?;
            let v_token = v.reshape((num_kv_heads, head_dim))?;

            // Write roped token K, V directly into physical block in KvStorage
            kv_storage.write_kv_token(layer_idx, physical_block, offset, &k_token, &v_token)?;

            // #48/#86: attention reads the history straight out of the block pool through
            // the block table. On Metal the fused paged-attention kernel walks the table
            // itself; other backends read the same table as a pool view.
            let paged = PagedAttnContext {
                block_tables: block_table.blocks()[..=block_idx]
                    .iter()
                    .map(|&b| b as u32)
                    .collect(),
                max_blocks: block_idx + 1,
                context_lens: vec![(pos + 1) as u32],
                slots: vec![crate::kv_cache::slot_for(
                    physical_block,
                    offset,
                    block_size,
                )],
                recycled_blocks: recycled_blocks.clone(),
            };
            let attn_out = self.decode_attention_paged(&q, &paged, kv_storage, layer_idx)?;
            // Merge heads back: [1, hidden_size]
            let attn_out = attn_out
                .transpose(0, 1)?
                .reshape((1, num_q_heads * head_dim))?;
            let attn_proj = self.linear_into(
                &layer.attn_output,
                &attn_out,
                crate::accelerate::amx::AmxWorkspace::SLOT_ATTN_OUT,
            )?;

            // Residual connection
            hidden = (hidden + &self.scale_residual(attn_proj)?)?;

            // FFN Norm
            let ffn_normed = self.rmsnorm(&hidden, &layer.ffn_norm)?;

            let ffn_out = self.forward_mlp_block_into(layer_idx, &ffn_normed, layer)?;
            // Residual connection
            hidden = (hidden + &self.scale_residual(ffn_out)?)?;
        }

        // Final RMSNorm
        let final_normed = self.rmsnorm(&hidden, &self.weights.norm)?;

        // LM Head projection: [1, vocab_size]
        let logits = self.scale_logits(self.linear_into(
            &self.weights.lm_head,
            &final_normed,
            crate::accelerate::amx::AmxWorkspace::SLOT_LOGITS,
        )?)?;

        // The AMX path wrote those logits into a workspace slot, so the tensor aliases a
        // buffer the next decode step overwrites. Every caller in this crate consumes
        // logits before stepping again, but this method is public: hand back an owned copy
        // rather than a slice of the workspace. One allocation against the ~230 the
        // workspace saves per step.
        if matches!(self.backend, Backend::Amx) {
            return Ok(logits.copy()?);
        }
        Ok(logits)
    }

    pub fn forward_verify_tokens(
        &self,
        tokens: &[u32],
        start_pos: usize,
        block_table: &BlockTable,
        kv_storage: &mut KvStorageKind,
    ) -> Result<Tensor, EngineError> {
        let seq_len = tokens.len();
        if seq_len == 0 {
            return Err(EngineError::Model("Empty input tokens".to_string()));
        }

        let head_dim = self.config.head_dim();
        let num_q_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;
        let block_size = kv_storage.block_size();
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
            hidden = (hidden + &self.scale_residual(attn_proj)?)?;

            let ffn_normed = self.rmsnorm(&hidden, &layer.ffn_norm)?;
            let ffn_out = self.forward_mlp_block(layer_idx, &ffn_normed, layer)?;
            hidden = (hidden + &self.scale_residual(ffn_out)?)?;
        }

        let final_normed = self.rmsnorm(&hidden, &self.weights.norm)?;
        let logits = self.scale_logits(self.linear(&self.weights.lm_head, &final_normed)?)?;
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
    /// This is the serving path's only forward, and since ticket #90 the speculative
    /// verifier rides it too: `Engine::speculative_step_sampled` submits the candidate
    /// block `[current, candidates...]` as one sequence of this batch, so "one model
    /// forward per step" (#45) holds for speculative traffic as well and
    /// [`Model::forward_batch_calls`] moves for it.
    #[allow(clippy::too_many_arguments)] // #45: the issue's flat-batch signature
    pub fn forward_batch(
        &self,
        tokens: &[u32],
        positions: &[usize],
        slot_mapping: &[usize],
        kv_storage: &mut KvStorageKind,
        block_tables: &[Vec<usize>],
        context_lens: &[usize],
        query_lens: &[usize],
    ) -> Result<Tensor, EngineError> {
        self.forward_batch_rows(
            tokens,
            positions,
            slot_mapping,
            kv_storage,
            block_tables,
            context_lens,
            query_lens,
            RowSelection::LastPerSequence,
        )
    }

    /// Same batched forward, but returns logits for *every* new token
    /// (`[total tokens, vocab]`) instead of one row per sequence.
    ///
    /// The speculative verifier needs the distribution under each of its candidates, not
    /// just the last one (ticket #90). Counted exactly like [`Model::forward_batch`],
    /// which it delegates to the same implementation as.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_batch_all_rows(
        &self,
        tokens: &[u32],
        positions: &[usize],
        slot_mapping: &[usize],
        kv_storage: &mut KvStorageKind,
        block_tables: &[Vec<usize>],
        context_lens: &[usize],
        query_lens: &[usize],
    ) -> Result<Tensor, EngineError> {
        self.forward_batch_rows(
            tokens,
            positions,
            slot_mapping,
            kv_storage,
            block_tables,
            context_lens,
            query_lens,
            RowSelection::EveryToken,
        )
    }

    /// The one implementation behind [`Model::forward_batch`] and
    /// [`Model::forward_batch_all_rows`]; they differ only in which logit rows survive.
    ///
    /// The `forward_batch_calls` counter lives here rather than in the wrappers, so
    /// "counted exactly once per forward" is a property of the shared body instead of
    /// something both wrappers have to remember.
    #[allow(clippy::too_many_arguments)]
    fn forward_batch_rows(
        &self,
        tokens: &[u32],
        positions: &[usize],
        slot_mapping: &[usize],
        kv_storage: &mut KvStorageKind,
        block_tables: &[Vec<usize>],
        context_lens: &[usize],
        query_lens: &[usize],
        rows: RowSelection,
    ) -> Result<Tensor, EngineError> {
        self.forward_batch_calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let all_rows = rows == RowSelection::EveryToken;
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
        // Not `mut` here: the one-command-buffer branch reads it once and produces
        // the step's final row; only the fallback loop rebinds it per layer.
        let hidden = self.embed(&token_ids)?;

        let (cos, sin) = self.rope_cos_sin(positions)?;
        let cos = cos.reshape((total, 1, head_dim / 2))?;
        let sin = sin.reshape((total, 1, head_dim / 2))?;

        // Each sequence's slice of the flattened batch, plus its offset causal mask
        // (query i of sequence s sits at absolute position `positions[offset + i]`).
        // Pure decode batches attend in one padded pass and need no mask at all.
        let pure_decode = query_lens.iter().all(|&q_len| q_len == 1);
        // #106: the Metal seam reads its own device copy of the pool, so blocks erased
        // since the last decode step must be re-copied whole before it runs. Only drain
        // when the seam will actually dispatch; otherwise the next decode picks them up.
        let recycled_blocks = if matches!(self.backend, Backend::Metal) && pure_decode {
            kv_storage.take_recycled_blocks()
        } else {
            Vec::new()
        };
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
        let slot_plan = KvSlotPlan::new(kv_storage, slot_mapping, &self.device)?;
        let decode_layout = pure_decode
            .then(|| self.decode_layout(kv_storage, block_tables, context_lens))
            .transpose()?;
        // #86: the Metal pure-decode path hands the kernel the same block geometry the
        // padded layout is built from, so it needs no per-layer KV view at all.
        let paged_decode = match (&self.backend, pure_decode) {
            (Backend::Metal, true) => Some(PagedAttnContext::decode_batch(
                kv_storage,
                block_tables,
                context_lens,
                slot_mapping,
                recycled_blocks,
            )?),
            _ => None,
        };

        // #87: a single-token Metal decode runs each layer's RMSNorm + QKV + RoPE through
        // the fused kernel, which is one command buffer for the whole prologue instead of
        // one per op. The seam stays `None` for a model outside the kernel geometry (or a
        // quantized projection), and the per-op prologue below runs unchanged — a
        // fallback, never an error.
        let fused_decode = total == 1 && matches!(self.backend, Backend::Metal);
        let mut fused_seam = self.fused_decode.borrow_mut();
        if fused_decode && fused_seam.is_none() {
            *fused_seam = FusedDecodeSeam::build(self)?;
        }
        let fused_pos = positions[0];

        // #110: the one-command-buffer step. The same conditions the #87 wiring
        // checked, plus the two that make a chained sweep valid: no hybrid FFN
        // offload owning the layer tail, and the step's paged geometry being built
        // (Metal pure decode). Everything else — bigger batches, other backends,
        // off-geometry models, hybrid offload, a position past the resident rotary
        // tables — keeps the loop below, which itself runs the per-layer fused
        // prologue since #87.
        let single_buffer_step = fused_decode
            && fused_seam.is_some()
            && self.hybrid_pipeline.is_none()
            && paged_decode.is_some()
            && fused_seam
                .as_ref()
                .is_some_and(|seam| fused_pos <= seam.max_pos());

        let single_buffer_paged = if single_buffer_step {
            paged_decode.as_ref()
        } else {
            None
        };

        let hidden = if let Some(paged) = single_buffer_paged {
            // The step consumes the embedding row and produces the final residual
            // row on the device; K/V rows come back per layer for the pool writeback
            // (the GPU published them into the seam's device pool copy directly, so
            // the pool itself is written after the single commit-and-wait).
            let embedding_row = hidden.flatten_all()?.to_vec1::<f32>()?;
            let kv_pool = kv_storage.f32_pool()?;
            let head_dim = self.config.head_dim();
            let shape = AttnShape {
                num_heads: self.config.num_attention_heads,
                num_kv_heads: self.config.num_key_value_heads,
                head_dim,
                scale: self.attention_scale(head_dim) as f32,
            };
            let residual_scale = self.config.residual_scale.unwrap_or(1.0);

            let mut paged_seam = self.paged_attn.borrow_mut();
            if paged_seam.is_none() {
                *paged_seam = Some(PagedAttnSeam::new(
                    self.config.num_hidden_layers,
                    &self.device,
                )?);
            }
            let paged_seam = paged_seam.as_mut().expect("just built");
            let seam = fused_seam.as_mut().expect("checked above");
            let rows = seam.run_step(
                paged_seam,
                kv_pool,
                paged,
                &embedding_row,
                fused_pos,
                self.config.rms_norm_eps as f32,
                residual_scale,
                &shape,
            )?;

            // Write the step's K/V back into the pool the engine owns: same rows the
            // per-op path scatters before attention, written after the step's single
            // submission completed.
            let num_kv_heads = self.config.num_key_value_heads;
            for (layer_idx, (k, v)) in rows.k.iter().zip(rows.v.iter()).enumerate() {
                let k = Tensor::from_vec(k.clone(), (1, num_kv_heads, head_dim), &self.device)?;
                let v = Tensor::from_vec(v.clone(), (1, num_kv_heads, head_dim), &self.device)?;
                kv_storage.write_kv_slots(layer_idx, &slot_plan, &k, &v)?;
            }
            self.paged_attention_calls.fetch_add(
                self.config.num_hidden_layers as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            Tensor::from_vec(rows.hidden, (1, self.config.hidden_size), &self.device)?
        } else {
            let mut hidden = hidden;
            for (layer_idx, layer) in self.weights.layers.iter().enumerate() {
                // The prologue reads this layer's input row, which the previous layer just
                // wrote back into `hidden`, so it is read here and not once before the loop.
                let fused = match fused_seam.as_mut() {
                    Some(seam) if fused_decode && fused_pos <= seam.max_pos() => {
                        let row = hidden.flatten_all()?.to_vec1::<f32>()?;
                        Some(seam.prologue(
                            layer_idx,
                            &row,
                            fused_pos,
                            self.config.rms_norm_eps as f32,
                        )?)
                    }
                    _ => None,
                };
                let (q, k, v) = match fused {
                    // The fused kernel already applied RoPE, so the rows below are final.
                    Some((q, k, v)) => (
                        Tensor::from_vec(q, (1, num_q_heads, head_dim), &self.device)?,
                        Tensor::from_vec(k, (1, num_kv_heads, head_dim), &self.device)?,
                        Tensor::from_vec(v, (1, num_kv_heads, head_dim), &self.device)?,
                    ),
                    None => {
                        let normed = self.rmsnorm(&hidden, &layer.attn_norm)?;
                        let (q, k, v) = self.project_qkv(layer, &normed)?;
                        let q = q.reshape((total, num_q_heads, head_dim))?;
                        let k = k.reshape((total, num_kv_heads, head_dim))?;
                        let v = v.reshape((total, num_kv_heads, head_dim))?;
                        (
                            self.rope_apply(&q, &cos, &sin)?,
                            self.rope_apply(&k, &cos, &sin)?,
                            v,
                        )
                    }
                };

                // Scatter this step's K/V into the physical blocks the scheduler mapped.
                kv_storage.write_kv_slots(layer_idx, &slot_plan, &k, &v)?;

                let attn_out = match (&paged_decode, &decode_layout) {
                    (Some(paged), _) => {
                        // #86: the kernel reads K/V out of the block pool itself, so the
                        // history is not gathered, padded or copied per layer.
                        self.metal_paged_attention(&q, kv_storage, layer_idx, paged)?
                    }
                    (None, Some(layout)) => {
                        // One query token per sequence: the whole batch attends in a single
                        // padded GEMM instead of one attention pass per sequence (which is
                        // what per-op overhead makes expensive at small batch sizes).
                        // ponytail: each sequence is padded to the longest context in the batch,
                        // so ragged batches waste lanes; a varlen kernel drops the padding.
                        self.batch_decode_attention(&q, kv_storage, layer_idx, layout)?
                    }
                    (None, None) => {
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
                hidden = (hidden + &self.scale_residual(attn_proj)?)?;

                let ffn_normed = self.rmsnorm(&hidden, &layer.ffn_norm)?;
                let ffn_out = self.forward_mlp_block(layer_idx, &ffn_normed, layer)?;
                hidden = (hidden + &self.scale_residual(ffn_out)?)?;
            }
            hidden
        };

        let final_normed = self.rmsnorm(&hidden, &self.weights.norm)?;

        // The serving path only needs each sequence's last new token (that is the token
        // the caller samples; partial prefill chunks discard theirs). The speculative
        // verifier needs every candidate's row (ticket #90).
        if all_rows {
            return self.scale_logits(self.linear(&self.weights.lm_head, &final_normed)?);
        }
        let last_indices: Vec<u32> = (0..num_seqs)
            .map(|s| (offsets[s] + query_lens[s] - 1) as u32)
            .collect();
        let last_indices = Tensor::from_vec(last_indices, (num_seqs,), &self.device)?;
        let last_hidden = final_normed.index_select(&last_indices, 0)?;
        self.scale_logits(self.linear(&self.weights.lm_head, &last_hidden)?)
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
        kv_storage: &KvStorageKind,
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
        // Ticket #88: the padded history comes from the pool through the enum, so this
        // function has no idea whether it is attending over F32 or dequantized KV.
        let batch = PaddedBatch::new(
            layout.indices.clone(),
            layout.blocks.clone(),
            num_seqs,
            layout.max_blocks,
            padded_ctx,
        );
        let (k_all, v_all) = kv_storage.padded_batch_kv(layer_idx, &batch)?;
        let k = k_all
            .reshape((num_seqs, padded_ctx, num_kv_heads, head_dim))?
            .permute((0, 2, 3, 1))?
            .contiguous()?
            .reshape((num_seqs * num_kv_heads, head_dim, padded_ctx))?;
        let v = v_all
            .reshape((num_seqs, padded_ctx, num_kv_heads, head_dim))?
            .permute((0, 2, 1, 3))?
            .contiguous()?
            .reshape((num_seqs * num_kv_heads, padded_ctx, head_dim))?;

        let q = q
            .reshape((num_seqs, num_kv_heads, group, head_dim))?
            .reshape((num_seqs * num_kv_heads, group, head_dim))?;
        let scale = self.attention_scale(head_dim);
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
        kv_storage: &KvStorageKind,
        block_tables: &[Vec<usize>],
        context_lens: &[usize],
    ) -> Result<DecodeLayout, EngineError> {
        let num_kv_heads = self.config.num_key_value_heads;
        let num_seqs = block_tables.len();
        // Same builder the paged-attention seam uses: one padding rule, one bounds check.
        let tables = PaddedDecodeTables::new(kv_storage.block_size(), block_tables, context_lens)?;
        let max_blocks = tables.max_blocks;
        let padded_ctx = tables.padded_ctx();

        let mut mask_data = vec![0.0f32; num_seqs * num_kv_heads * padded_ctx];
        for (s, &ctx_len) in context_lens.iter().enumerate() {
            for g in 0..num_kv_heads {
                let row = (s * num_kv_heads + g) * padded_ctx;
                for j in ctx_len..padded_ctx {
                    mask_data[row + j] = -1e9f32;
                }
            }
        }

        Ok(DecodeLayout {
            indices: Tensor::from_vec(
                tables.blocks.clone(),
                (num_seqs * max_blocks,),
                &self.device,
            )?,
            blocks: tables.blocks,
            max_blocks,
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
