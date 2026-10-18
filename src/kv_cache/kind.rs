//! One KV pool type for the decode path (ticket #88).
//!
//! The decode path must not branch on *which* storage it is writing into: it writes
//! tokens and slots, reads tokens, gathers context and asks for a paged view, and the
//! pool decides how those are stored. [`KvStorageKind`] is that single seam — the
//! F32 pool ([`KvStorage`]) and the quantized pool ([`QuantizedKvStorage`]) sit behind
//! the same method set, and the choice is made once, in `Engine`'s constructor.
//!
//! Quantization is a host-side (CPU/AMX) feature: the Metal paged-attention kernel
//! reads an F32 pool buffer directly, so a quantized pool would have to be dequantized
//! behind its back. That is refused where the pool is chosen ([`Self::quantized`] is
//! reached from the engine constructor, which rejects it for Metal) rather than papered
//! over with a silent fallback.

use candle_core::{DType, Device, IndexOp, Tensor};

use crate::error::EngineError;
use crate::kv_cache::block_table::BlockTable;
use crate::kv_cache::quantized_storage::{QuantMode, QuantizedKvStorage};
use crate::kv_cache::storage::KvStorage;

/// The block geometry of a pure-decode batch, padded to the batch's longest table
/// (ticket #61 F11 / #88): both pools read it, one as a device index, one as a walk.
pub struct PaddedBatch {
    /// `[num_seqs * max_blocks]` physical block ids on the device (F32 `index_select`).
    pub ids: Tensor,
    /// The same ids on the host (the quantized pool dequantizes block by block).
    pub blocks: Vec<u32>,
    pub num_seqs: usize,
    pub max_blocks: usize,
    /// `max_blocks * block_size`: the padded context every sequence is read as.
    pub padded_ctx: usize,
}

impl PaddedBatch {
    pub fn new(
        ids: Tensor,
        blocks: Vec<u32>,
        num_seqs: usize,
        max_blocks: usize,
        padded_ctx: usize,
    ) -> Self {
        Self {
            ids,
            blocks,
            num_seqs,
            max_blocks,
            padded_ctx,
        }
    }
}

/// Where a step's K/V rows go, built once per step and shared by every layer (#61 F11).
pub struct KvSlotPlan {
    /// `[tokens, n_kv_heads, head_dim]` scatter index for the F32 pool.
    ids: Tensor,
    /// `(block, offset)` per token for the quantized pool.
    targets: Vec<(usize, usize)>,
}

impl KvSlotPlan {
    pub fn new(
        storage: &KvStorageKind,
        slots: &[usize],
        device: &Device,
    ) -> Result<Self, EngineError> {
        let num_slots = storage.num_blocks() * storage.block_size();
        if let Some(&slot) = slots.iter().find(|&&slot| slot >= num_slots) {
            return Err(EngineError::Model(format!(
                "Slot {} out of bounds for {} blocks of size {}",
                slot,
                storage.num_blocks(),
                storage.block_size()
            )));
        }
        let block_size = storage.block_size();
        let targets = slots
            .iter()
            .map(|&slot| (slot / block_size, slot % block_size))
            .collect();
        Ok(Self {
            ids: storage.slot_id_tensor(slots, device)?,
            targets,
        })
    }
}

/// The KV pool the decode path writes and reads, chosen once per engine.
#[derive(Debug, Clone)]
pub enum KvStorageKind {
    /// Full-precision pool; the default and the pre-#88 behaviour.
    F32(KvStorage),
    /// INT8/FP8 pool with per-token-per-head scales.
    Quantized(QuantizedKvStorage),
}

impl KvStorageKind {
    pub fn f32(storage: KvStorage) -> Self {
        Self::F32(storage)
    }

    /// Builds an F32 pool with `KvStorage::new`'s argument list, so call sites that only
    /// want the default pool do not need to know about the enum's variants.
    pub fn f32_new(
        num_layers: usize,
        num_blocks: usize,
        block_size: usize,
        n_kv_heads: usize,
        head_dim: usize,
        device: &Device,
    ) -> Result<Self, EngineError> {
        Ok(Self::F32(KvStorage::new(
            num_layers,
            num_blocks,
            block_size,
            n_kv_heads,
            head_dim,
            device,
        )?))
    }

    pub fn quantized(
        num_layers: usize,
        num_blocks: usize,
        block_size: usize,
        n_kv_heads: usize,
        head_dim: usize,
        mode: QuantMode,
        device: &Device,
    ) -> Result<Self, EngineError> {
        Ok(Self::Quantized(QuantizedKvStorage::new(
            num_layers,
            num_blocks,
            block_size,
            n_kv_heads,
            head_dim,
            mode,
            device,
        )?))
    }

    /// `None` for the F32 pool, the quantization mode otherwise.
    pub fn mode(&self) -> Option<QuantMode> {
        match self {
            Self::F32(_) => None,
            Self::Quantized(q) => Some(q.quant_mode),
        }
    }

    pub fn block_size(&self) -> usize {
        match self {
            Self::F32(kv) => kv.block_size,
            Self::Quantized(q) => q.block_size,
        }
    }

    pub fn num_blocks(&self) -> usize {
        match self {
            Self::F32(kv) => kv.num_blocks,
            Self::Quantized(q) => q.num_blocks,
        }
    }

    pub fn n_kv_heads(&self) -> usize {
        match self {
            Self::F32(kv) => kv.n_kv_heads,
            Self::Quantized(q) => q.n_kv_heads,
        }
    }

    pub fn head_dim(&self) -> usize {
        match self {
            Self::F32(kv) => kv.head_dim,
            Self::Quantized(q) => q.head_dim,
        }
    }

    /// The F32 pool, for the one consumer that needs the raw device buffer: the Metal
    /// paged-attention seam (ticket #86). Everywhere else goes through the enum.
    pub fn f32_pool(&self) -> Result<&KvStorage, EngineError> {
        match self {
            Self::F32(kv) => Ok(kv),
            Self::Quantized(q) => Err(EngineError::Metal(format!(
                "quantized KV ({:?}) requires a dequantized gather; the Metal \
                 paged-attention kernel reads the F32 pool directly",
                q.quant_mode
            ))),
        }
    }

    /// The F32 pool mutably, for tests that seed pool contents directly.
    pub fn f32_pool_mut(&mut self) -> Result<&mut KvStorage, EngineError> {
        match self {
            Self::F32(kv) => Ok(kv),
            Self::Quantized(q) => Err(EngineError::Metal(format!(
                "quantized KV ({:?}) has no F32 pool to hand out",
                q.quant_mode
            ))),
        }
    }

    /// Bytes of KV held per token, which is what quantization buys.
    pub fn memory_bytes_per_token(&self) -> usize {
        match self {
            Self::F32(_) => self.f32_bytes_per_token(),
            Self::Quantized(q) => q.memory_bytes_per_token(),
        }
    }

    /// The F32 pool's bytes per token for this pool's geometry.
    pub fn f32_bytes_per_token(&self) -> usize {
        2 * self.n_kv_heads() * self.head_dim() * std::mem::size_of::<f32>()
    }

    /// How many times smaller this pool is than the F32 pool (`1.0` for F32 itself).
    pub fn memory_ratio_vs_f32(&self) -> f32 {
        self.f32_bytes_per_token() as f32 / self.memory_bytes_per_token() as f32
    }

    /// The `[tokens, n_kv_heads, head_dim]` slot ids `write_kv_slots` scatters with.
    ///
    /// candle's `scatter_set` is element-wise, so each token's slot id repeats across its
    /// head/dim row (ticket #61 F11). Built once per step by [`KvSlotPlan`].
    pub fn slot_id_tensor(
        &self,
        slots: &[usize],
        device: &Device,
    ) -> Result<Tensor, EngineError> {
        let repeats = self.n_kv_heads() * self.head_dim();
        let mut slot_ids: Vec<u32> = Vec::with_capacity(slots.len() * repeats);
        for &slot in slots {
            slot_ids.resize(slot_ids.len() + repeats, slot as u32);
        }
        Ok(Tensor::from_vec(
            slot_ids,
            (slots.len(), self.n_kv_heads(), self.head_dim()),
            device,
        )?)
    }

    /// Writes one token's K and V into physical `block` at `offset`.
    pub fn write_kv_token(
        &mut self,
        layer: usize,
        block: usize,
        offset: usize,
        k: &Tensor,
        v: &Tensor,
    ) -> Result<(), EngineError> {
        match self {
            Self::F32(kv) => kv.write_kv_token(layer, block, offset, k, v),
            Self::Quantized(q) => q.write_quantized_kv_token(layer, block, offset, k, v),
        }
    }

    /// Writes a batch of tokens' K and V (`[tokens, n_kv_heads, head_dim]`).
    pub fn write_kv_slots(
        &mut self,
        layer: usize,
        plan: &KvSlotPlan,
        k: &Tensor,
        v: &Tensor,
    ) -> Result<(), EngineError> {
        match self {
            Self::F32(kv) => kv.write_kv_slots(layer, &plan.ids, k, v),
            Self::Quantized(q) => {
                let (rows, heads, dim) = k.dims3()?;
                if rows != plan.targets.len() {
                    return Err(EngineError::Model(format!(
                        "write_kv_slots: {} K rows for {} slot targets",
                        rows,
                        plan.targets.len()
                    )));
                }
                for (i, &(block, offset)) in plan.targets.iter().enumerate() {
                    let k_row = k.i(i)?.reshape((heads, dim))?;
                    let v_row = v.i(i)?.reshape((heads, dim))?;
                    q.write_quantized_kv_token(layer, block, offset, &k_row, &v_row)?;
                }
                Ok(())
            }
        }
    }

    /// Reads back one token's K and V as `[n_kv_heads, head_dim]`.
    pub fn read_kv_token(
        &self,
        layer: usize,
        block: usize,
        offset: usize,
    ) -> Result<(Tensor, Tensor), EngineError> {
        match self {
            Self::F32(kv) => kv.read_kv_token(layer, block, offset),
            Self::Quantized(q) => q.read_quantized_kv_token(layer, block, offset),
        }
    }

    /// `[num_tokens, n_kv_heads, head_dim]` for a block table, dequantized if needed.
    pub fn gather_blocks(
        &self,
        layer: usize,
        blocks: &[usize],
        num_tokens: usize,
    ) -> Result<(Tensor, Tensor), EngineError> {
        match self {
            Self::F32(kv) => kv.gather_blocks(layer, blocks, num_tokens),
            Self::Quantized(q) => q.gather_dequantized_blocks(layer, blocks, num_tokens),
        }
    }

    pub fn gather_context(
        &self,
        layer: usize,
        block_table: &BlockTable,
    ) -> Result<(Tensor, Tensor), EngineError> {
        self.gather_blocks(layer, block_table.blocks(), block_table.num_tokens())
    }

    /// Decode-path read of `num_tokens` of history (ticket #48): a zero-copy pool view
    /// for a consecutive F32 table, a gather otherwise — and always a gather for a
    /// quantized pool, whose history does not exist as floats until it is dequantized.
    pub fn paged_kv_tokens(
        &self,
        layer: usize,
        blocks: &[usize],
        num_tokens: usize,
    ) -> Result<(Tensor, Tensor), EngineError> {
        match self {
            Self::F32(kv) => kv.paged_kv_tokens(layer, blocks, num_tokens),
            Self::Quantized(q) => q.gather_dequantized_blocks(layer, blocks, num_tokens),
        }
    }

    /// `[num_seqs * padded_ctx, n_kv_heads, head_dim]` K/V for a padded decode batch.
    ///
    /// Padding lanes repeat a real block (the attention mask zeroes them out), so both
    /// pools read only in-bounds blocks — the F32 pool as one `index_select`, the
    /// quantized pool as one dequantized gather per sequence.
    pub fn padded_batch_kv(
        &self,
        layer: usize,
        batch: &PaddedBatch,
    ) -> Result<(Tensor, Tensor), EngineError> {
        let heads = self.n_kv_heads();
        let dim = self.head_dim();
        if batch.num_seqs == 0 || batch.padded_ctx == 0 {
            let shape = (0, heads, dim);
            return Ok((
                Tensor::zeros(shape, DType::F32, batch.ids.device())?,
                Tensor::zeros(shape, DType::F32, batch.ids.device())?,
            ));
        }
        match self {
            Self::F32(kv) => {
                let gather = |pool: &Tensor| -> Result<Tensor, EngineError> {
                    Ok(pool
                        .index_select(&batch.ids, 0)?
                        .reshape((batch.num_seqs * batch.padded_ctx, heads, dim))?
                        .contiguous()?)
                };
                Ok((gather(&kv.kv_k[layer])?, gather(&kv.kv_v[layer])?))
            }
            Self::Quantized(q) => {
                let mut ks = Vec::with_capacity(batch.num_seqs);
                let mut vs = Vec::with_capacity(batch.num_seqs);
                for s in 0..batch.num_seqs {
                    let start = s * batch.max_blocks;
                    let blocks: Vec<usize> = batch.blocks[start..start + batch.max_blocks]
                        .iter()
                        .map(|&b| b as usize)
                        .collect();
                    let (k, v) = q.gather_dequantized_blocks(layer, &blocks, batch.padded_ctx)?;
                    ks.push(k);
                    vs.push(v);
                }
                Ok((Tensor::cat(&ks, 0)?, Tensor::cat(&vs, 0)?))
            }
        }
    }
}
