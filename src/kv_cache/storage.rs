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
        let active_blocks = &blocks[..needed_blocks.min(blocks.len())];

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
}
