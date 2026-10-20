use crate::error::EngineError;
use crate::kv_cache::block_table::BlockTable;
use candle_core::{DType, Device, Tensor};

#[derive(Debug, Clone)]
pub struct KvStorage {
    pub kv_k: Vec<Tensor>,
    pub kv_v: Vec<Tensor>,
    pub num_layers: usize,
    pub num_blocks: usize,
    pub block_size: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    /// Blocks erased since the last consumer picked them up (ticket #106). The Metal
    /// paged-attention seam keeps a device copy of this pool, so an erase has to reach
    /// that copy too: the seam refreshes every block listed here from the pool.
    recycled_blocks: Vec<usize>,
}

impl KvStorage {
    pub fn new(
        num_layers: usize,
        num_blocks: usize,
        block_size: usize,
        n_kv_heads: usize,
        head_dim: usize,
        device: &Device,
    ) -> Result<Self, EngineError> {
        // A KV pool the machine cannot hold is a startup configuration error the caller
        // reports (ticket #85), not an overflow or an allocation abort inside candle.
        if num_blocks == 0 {
            return Err(EngineError::Config(format!(
                "cannot allocate KV storage: received {num_blocks} total blocks; the pool must hold at least one block"
            )));
        }
        if block_size == 0 {
            return Err(EngineError::Config(format!(
                "cannot allocate KV storage: block size {block_size} holds no token"
            )));
        }
        if n_kv_heads == 0 || head_dim == 0 {
            return Err(EngineError::Config(format!(
                "cannot allocate KV storage: model declares {n_kv_heads} KV heads of head_dim {head_dim}"
            )));
        }

        let elements = num_blocks
            .checked_mul(block_size)
            .and_then(|n| n.checked_mul(n_kv_heads))
            .and_then(|n| n.checked_mul(head_dim))
            .and_then(|n| n.checked_mul(num_layers))
            .and_then(|n| n.checked_mul(2));
        let bytes = elements
            .and_then(|n| n.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| {
                EngineError::Config(format!(
                    "cannot allocate KV storage: {num_layers} layers x {num_blocks} blocks x {block_size} tokens x {n_kv_heads} KV heads x {head_dim} head_dim overflows the addressable size"
                ))
            })?;
        let host_gb = crate::hardware::get_hardware_info().total_ram_gb;
        let host_bytes = (host_gb * 1024.0 * 1024.0 * 1024.0) as u64;
        if bytes as u64 > host_bytes {
            return Err(EngineError::Config(format!(
                "cannot allocate KV storage: request of {:.1} GiB exceeds the {:.1} GiB of physical memory on this machine",
                bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                host_gb
            )));
        }

        let shape = (num_blocks, block_size, n_kv_heads, head_dim);
        let mut kv_k = Vec::with_capacity(num_layers);
        let mut kv_v = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            kv_k.push(Tensor::zeros(shape, DType::F32, device)?);
            kv_v.push(Tensor::zeros(shape, DType::F32, device)?);
        }
        Ok(Self {
            kv_k,
            kv_v,
            num_layers,
            num_blocks,
            block_size,
            n_kv_heads,
            head_dim,
            recycled_blocks: Vec::new(),
        })
    }

    /// Deep clone cache tensors so modifications do not alias.
    pub fn clone_deep(&self) -> Result<Self, EngineError> {
        let deep = |ts: &[Tensor]| -> Result<Vec<Tensor>, EngineError> {
            ts.iter().map(|t| Ok(t.copy()?)).collect()
        };
        Ok(Self {
            kv_k: deep(&self.kv_k)?,
            kv_v: deep(&self.kv_v)?,
            num_layers: self.num_layers,
            num_blocks: self.num_blocks,
            block_size: self.block_size,
            n_kv_heads: self.n_kv_heads,
            head_dim: self.head_dim,
            recycled_blocks: self.recycled_blocks.clone(),
        })
    }

    /// Writes one token's K and V into physical block at specified offset.
    pub fn write_kv_token(
        &mut self,
        layer: usize,
        block: usize,
        offset: usize,
        k: &Tensor,
        v: &Tensor,
    ) -> Result<(), EngineError> {
        if layer >= self.num_layers {
            return Err(EngineError::Model(format!(
                "Layer index {} out of bounds ({})",
                layer, self.num_layers
            )));
        }
        if block >= self.num_blocks {
            return Err(EngineError::Model(format!(
                "Block index {} out of bounds ({})",
                block, self.num_blocks
            )));
        }
        if offset >= self.block_size {
            return Err(EngineError::Model(format!(
                "Offset {} out of bounds for block size {}",
                offset, self.block_size
            )));
        }

        let run_k = k.reshape((1, 1, self.n_kv_heads, self.head_dim))?;
        let run_v = v.reshape((1, 1, self.n_kv_heads, self.head_dim))?;

        self.kv_k[layer]
            .narrow(0, block, 1)?
            .slice_set(&run_k, 1, offset)?;
        self.kv_v[layer]
            .narrow(0, block, 1)?
            .slice_set(&run_v, 1, offset)?;

        Ok(())
    }

    /// Erases one physical block across all layers (ticket #106).
    ///
    /// A block is erased when it is handed to a new request, so a recycled block cannot
    /// expose the previous request's KV in the `[ctx_len..block_size)` tail that
    /// attention still reads.
    pub fn reset_block(&mut self, block: usize) -> Result<(), EngineError> {
        if block >= self.num_blocks {
            return Err(EngineError::Model(format!(
                "Block index {} out of bounds ({})",
                block, self.num_blocks
            )));
        }
        let zeros = Tensor::zeros(
            (1, self.block_size, self.n_kv_heads, self.head_dim),
            DType::F32,
            self.kv_k[0].device(),
        )?;
        for layer in 0..self.num_layers {
            self.kv_k[layer]
                .narrow(0, block, 1)?
                .slice_set(&zeros, 0, 0)?;
            self.kv_v[layer]
                .narrow(0, block, 1)?
                .slice_set(&zeros, 0, 0)?;
        }
        if !self.recycled_blocks.contains(&block) {
            self.recycled_blocks.push(block);
        }
        Ok(())
    }

    /// Takes the blocks erased since the last call (ticket #106). The Metal
    /// paged-attention seam owns a device copy of this pool, so it must copy these
    /// blocks again before reading them; every other consumer reads the pool directly.
    pub fn take_recycled_blocks(&mut self) -> Vec<usize> {
        std::mem::take(&mut self.recycled_blocks)
    }

    /// The `[len, n_kv_heads, head_dim]` slot ids [`Self::write_kv_slots`] scatters with,
    /// built once per step instead of once per layer (ticket #61 F11).
    ///
    /// candle's `scatter_set` is element-wise (indexes must have the source's shape), so
    /// each token's slot id is repeated across its head/dim row — a few KB of host work per
    /// step against one scatter op per layer per K and per V.
    pub fn slot_id_tensor(&self, slots: &[usize], device: &Device) -> Result<Tensor, EngineError> {
        let num_slots = self.num_blocks * self.block_size;
        if let Some(&slot) = slots.iter().find(|&&slot| slot >= num_slots) {
            return Err(EngineError::Model(format!(
                "Slot {} out of bounds for {} blocks of size {}",
                slot, self.num_blocks, self.block_size
            )));
        }

        let repeats = self.n_kv_heads * self.head_dim;
        let mut slot_ids: Vec<u32> = Vec::with_capacity(slots.len() * repeats);
        for &slot in slots {
            let end = slot_ids.len() + repeats;
            slot_ids.resize(end, slot as u32);
        }
        Ok(Tensor::from_vec(
            slot_ids,
            (slots.len(), self.n_kv_heads, self.head_dim),
            device,
        )?)
    }

    /// Writes a whole batch of tokens' K and V in one scatter each (#45).
    ///
    /// `slot_ids` is [`Self::slot_id_tensor`] for the step's `slots`; `k`/`v` are
    /// `[T, n_kv_heads, head_dim]`. Same result as `T` calls to [`Self::write_kv_token`],
    /// but two tensor ops per layer instead of roughly eight per token — which is what
    /// a decode batch pays for otherwise.
    pub fn write_kv_slots(
        &mut self,
        layer: usize,
        slot_ids: &Tensor,
        k: &Tensor,
        v: &Tensor,
    ) -> Result<(), EngineError> {
        if layer >= self.num_layers {
            return Err(EngineError::Model(format!(
                "Layer index {} out of bounds ({})",
                layer, self.num_layers
            )));
        }
        if k.dims() != [slot_ids.dims()[0], self.n_kv_heads, self.head_dim]
            || v.dims() != [slot_ids.dims()[0], self.n_kv_heads, self.head_dim]
        {
            return Err(EngineError::Model(format!(
                "write_kv_slots: expected [{}, {}, {}] K/V, got {:?} / {:?}",
                slot_ids.dims()[0],
                self.n_kv_heads,
                self.head_dim,
                k.dims(),
                v.dims()
            )));
        }
        let num_slots = self.num_blocks * self.block_size;
        let cache_shape = (num_slots, self.n_kv_heads, self.head_dim);

        self.kv_k[layer]
            .reshape(cache_shape)?
            .scatter_set(slot_ids, k, 0)?;
        self.kv_v[layer]
            .reshape(cache_shape)?
            .scatter_set(slot_ids, v, 0)?;
        Ok(())
    }

    /// Reads back token K and V stored at physical block and offset.
    pub fn read_kv_token(
        &self,
        layer: usize,
        block: usize,
        offset: usize,
    ) -> Result<(Tensor, Tensor), EngineError> {
        if layer >= self.num_layers || block >= self.num_blocks || offset >= self.block_size {
            return Err(EngineError::Model("Indices out of bounds".to_string()));
        }

        let k = self.kv_k[layer]
            .narrow(0, block, 1)?
            .narrow(1, offset, 1)?
            .squeeze(0)?
            .squeeze(0)?;
        let v = self.kv_v[layer]
            .narrow(0, block, 1)?
            .narrow(1, offset, 1)?
            .squeeze(0)?
            .squeeze(0)?;

        Ok((k, v))
    }

    /// Gathers historical context for `layer` using a sequence of physical block IDs and total tokens.
    /// Returns (K, V) tensors of shape `[num_tokens, n_kv_heads, head_dim]`.
    pub fn gather_blocks(
        &self,
        layer: usize,
        blocks: &[usize],
        num_tokens: usize,
    ) -> Result<(Tensor, Tensor), EngineError> {
        if layer >= self.num_layers {
            return Err(EngineError::Model(format!(
                "Layer index {} out of bounds ({})",
                layer, self.num_layers
            )));
        }

        let dev = self.kv_k[layer].device();
        if num_tokens == 0 || blocks.is_empty() {
            let empty_k = Tensor::zeros((0, self.n_kv_heads, self.head_dim), DType::F32, dev)?;
            let empty_v = Tensor::zeros((0, self.n_kv_heads, self.head_dim), DType::F32, dev)?;
            return Ok((empty_k, empty_v));
        }

        let needed_blocks = num_tokens.div_ceil(self.block_size);
        if needed_blocks > blocks.len() {
            return Err(EngineError::Model(format!(
                "gather_blocks: {} tokens need {} blocks, block table has {}",
                num_tokens,
                needed_blocks,
                blocks.len()
            )));
        }
        let active_blocks = &blocks[..needed_blocks];

        let block_idx_u32: Vec<u32> = active_blocks.iter().map(|&b| b as u32).collect();
        let idx_tensor = Tensor::from_vec(block_idx_u32, (active_blocks.len(),), dev)?;

        let gathered_k = self.kv_k[layer].index_select(&idx_tensor, 0)?;
        let gathered_v = self.kv_v[layer].index_select(&idx_tensor, 0)?;

        let flat_k = gathered_k.reshape((
            active_blocks.len() * self.block_size,
            self.n_kv_heads,
            self.head_dim,
        ))?;
        let flat_v = gathered_v.reshape((
            active_blocks.len() * self.block_size,
            self.n_kv_heads,
            self.head_dim,
        ))?;

        let k_ctx = flat_k.narrow(0, 0, num_tokens)?;
        let v_ctx = flat_v.narrow(0, 0, num_tokens)?;

        Ok((k_ctx, v_ctx))
    }

    /// Gathers historical context from `BlockTable` for given layer.
    pub fn gather_context(
        &self,
        layer: usize,
        block_table: &BlockTable,
    ) -> Result<(Tensor, Tensor), EngineError> {
        self.gather_blocks(layer, block_table.blocks(), block_table.num_tokens())
    }

    /// First physical block of `blocks` when the table's used blocks are one consecutive
    /// run of the pool, `None` when the table is fragmented or too short for `num_tokens`
    /// (ticket #61 F10 — a short table must not silently narrow the read to fewer blocks
    /// than the context needs; `paged_kv_tokens` reports it).
    fn contiguous_run(&self, blocks: &[usize], num_tokens: usize) -> Option<usize> {
        let needed = num_tokens.div_ceil(self.block_size);
        if needed > blocks.len() {
            return None;
        }
        let first = *blocks.first()?;
        (0..needed).all(|i| blocks[i] == first + i).then_some(first)
    }

    /// Decode-path read: K/V for `num_tokens` tokens of `layer` as `[num_tokens,
    /// n_kv_heads, head_dim]`, straight out of the block pool.
    ///
    /// A consecutive table (the pool hands out the lowest free block, so a growing
    /// sequence stays contiguous) is a zero-copy *view* of the pool storage — the whole
    /// history is never materialized per layer. Fragmented tables (prefix-cache reuse,
    /// speculation rollback) fall back to one `gather_blocks` copy.
    ///
    /// A table too short for `num_tokens` is an error, not a truncated read (ticket #61 F10).
    pub fn paged_kv_tokens(
        &self,
        layer: usize,
        blocks: &[usize],
        num_tokens: usize,
    ) -> Result<(Tensor, Tensor), EngineError> {
        match self.contiguous_run(blocks, num_tokens) {
            Some(first) => {
                let start = first * self.block_size;
                let view = |pool: &Tensor| -> Result<Tensor, EngineError> {
                    Ok(pool.flatten(0, 1)?.narrow(0, start, num_tokens)?)
                };
                Ok((view(&self.kv_k[layer])?, view(&self.kv_v[layer])?))
            }
            None => self.gather_blocks(layer, blocks, num_tokens),
        }
    }
}
