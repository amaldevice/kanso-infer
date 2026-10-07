use candle_core::{DType, Device, Tensor};
use crate::error::EngineError;
use crate::kv_cache::block_table::BlockTable;

/// Quantization mode for KV cache storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantMode {
    Int8,
    Fp8,
}

pub type QuantType = QuantMode;

/// Physical quantized Key-Value block storing quantized bytes and per-token/per-head scale factors.
#[derive(Debug, Clone)]
pub struct QuantizedKvBlock {
    pub k_quant: Vec<i8>,   // block_size * n_kv_heads * head_dim
    pub v_quant: Vec<i8>,   // block_size * n_kv_heads * head_dim
    pub k_scales: Vec<f32>, // block_size * n_kv_heads (per-token-per-head scale factor)
    pub v_scales: Vec<f32>, // block_size * n_kv_heads
}

impl QuantizedKvBlock {
    pub fn new(block_size: usize, n_kv_heads: usize, head_dim: usize) -> Self {
        let total_elements = block_size * n_kv_heads * head_dim;
        let total_scales = block_size * n_kv_heads;
        Self {
            k_quant: vec![0i8; total_elements],
            v_quant: vec![0i8; total_elements],
            k_scales: vec![1.0f32; total_scales],
            v_scales: vec![1.0f32; total_scales],
        }
    }

    pub fn reset(&mut self) {
        self.k_quant.fill(0);
        self.v_quant.fill(0);
        self.k_scales.fill(1.0);
        self.v_scales.fill(1.0);
    }
}

/// Quantized Key-Value Cache Storage supporting INT8 and FP8 representations.
#[derive(Debug, Clone)]
pub struct QuantizedKvStorage {
    pub blocks: Vec<Vec<QuantizedKvBlock>>, // [num_layers][num_blocks]
    pub num_layers: usize,
    pub num_blocks: usize,
    pub block_size: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub quant_mode: QuantMode,
    pub device: Device,
}

impl QuantizedKvStorage {
    /// Creates a new `QuantizedKvStorage` instance with the specified quantization mode.
    pub fn new(
        num_layers: usize,
        num_blocks: usize,
        block_size: usize,
        n_kv_heads: usize,
        head_dim: usize,
        quant_mode: QuantMode,
        device: &Device,
    ) -> Result<Self, EngineError> {
        let mut blocks = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            let mut layer_blocks = Vec::with_capacity(num_blocks);
            for _ in 0..num_blocks {
                layer_blocks.push(QuantizedKvBlock::new(block_size, n_kv_heads, head_dim));
            }
            blocks.push(layer_blocks);
        }

        Ok(Self {
            blocks,
            num_layers,
            num_blocks,
            block_size,
            n_kv_heads,
            head_dim,
            quant_mode,
            device: device.clone(),
        })
    }

    /// Convenience constructor for INT8 quantized storage.
    pub fn new_int8(
        num_layers: usize,
        num_blocks: usize,
        block_size: usize,
        n_kv_heads: usize,
        head_dim: usize,
        device: &Device,
    ) -> Result<Self, EngineError> {
        Self::new(
            num_layers,
            num_blocks,
            block_size,
            n_kv_heads,
            head_dim,
            QuantMode::Int8,
            device,
        )
    }

    /// Convenience constructor for FP8 quantized storage.
    pub fn new_fp8(
        num_layers: usize,
        num_blocks: usize,
        block_size: usize,
        n_kv_heads: usize,
        head_dim: usize,
        device: &Device,
    ) -> Result<Self, EngineError> {
        Self::new(
            num_layers,
            num_blocks,
            block_size,
            n_kv_heads,
            head_dim,
            QuantMode::Fp8,
            device,
        )
    }

    /// Deep clone cache so modifications do not alias.
    pub fn clone_deep(&self) -> Result<Self, EngineError> {
        Ok(self.clone())
    }

    /// Resets all data in a physical block across all layers (for pool recycling).
    pub fn reset_block(&mut self, block: usize) {
        if block < self.num_blocks {
            for layer in 0..self.num_layers {
                self.blocks[layer][block].reset();
            }
        }
    }

    /// Quantizes and writes one token's K and V into physical block at specified offset.
    pub fn write_quantized_kv_token(
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

        let expected_len = self.n_kv_heads * self.head_dim;
        let k_flat = k.flatten_all()?.to_vec1::<f32>()?;
        let v_flat = v.flatten_all()?.to_vec1::<f32>()?;

        if k_flat.len() != expected_len || v_flat.len() != expected_len {
            return Err(EngineError::Model(format!(
                "KV token tensor dimensions mismatch: expected {}, got k={}, v={}",
                expected_len,
                k_flat.len(),
                v_flat.len()
            )));
        }

        let quant_mode = self.quant_mode;
        let blk = &mut self.blocks[layer][block];
        let elements_per_token = self.n_kv_heads * self.head_dim;

        for h in 0..self.n_kv_heads {
            let k_slice = &k_flat[h * self.head_dim..(h + 1) * self.head_dim];
            let v_slice = &v_flat[h * self.head_dim..(h + 1) * self.head_dim];

            let scale_idx = offset * self.n_kv_heads + h;
            let dst_offset = offset * elements_per_token + h * self.head_dim;

            let k_out = &mut blk.k_quant[dst_offset..dst_offset + self.head_dim];
            let k_scale = Self::quantize_head_slice(quant_mode, k_slice, k_out);
            blk.k_scales[scale_idx] = k_scale;

            let v_out = &mut blk.v_quant[dst_offset..dst_offset + self.head_dim];
            let v_scale = Self::quantize_head_slice(quant_mode, v_slice, v_out);
            blk.v_scales[scale_idx] = v_scale;
        }

        Ok(())
    }

    /// Alias for `write_quantized_kv_token` for drop-in compatibility with `KvStorage`.
    pub fn write_kv_token(
        &mut self,
        layer: usize,
        block: usize,
        offset: usize,
        k: &Tensor,
        v: &Tensor,
    ) -> Result<(), EngineError> {
        self.write_quantized_kv_token(layer, block, offset, k, v)
    }

    /// Reads and dequantizes token K and V stored at physical block and offset.
    pub fn read_quantized_kv_token(
        &self,
        layer: usize,
        block: usize,
        offset: usize,
    ) -> Result<(Tensor, Tensor), EngineError> {
        if layer >= self.num_layers || block >= self.num_blocks || offset >= self.block_size {
            return Err(EngineError::Model("Indices out of bounds".to_string()));
        }

        let elements_per_token = self.n_kv_heads * self.head_dim;
        let mut out_k = vec![0.0f32; elements_per_token];
        let mut out_v = vec![0.0f32; elements_per_token];
        let blk = &self.blocks[layer][block];

        for h in 0..self.n_kv_heads {
            let scale_idx = offset * self.n_kv_heads + h;
            let k_scale = blk.k_scales[scale_idx];
            let v_scale = blk.v_scales[scale_idx];

            let src_start = offset * elements_per_token + h * self.head_dim;
            let dst_start = h * self.head_dim;

            let k_dst = &mut out_k[dst_start..dst_start + self.head_dim];
            let k_src = &blk.k_quant[src_start..src_start + self.head_dim];
            let v_dst = &mut out_v[dst_start..dst_start + self.head_dim];
            let v_src = &blk.v_quant[src_start..src_start + self.head_dim];

            match self.quant_mode {
                QuantMode::Int8 => {
                    for (dst, &src) in k_dst.iter_mut().zip(k_src.iter()) {
                        *dst = (src as f32) * k_scale;
                    }
                    for (dst, &src) in v_dst.iter_mut().zip(v_src.iter()) {
                        *dst = (src as f32) * v_scale;
                    }
                }
                QuantMode::Fp8 => {
                    for (dst, &src) in k_dst.iter_mut().zip(k_src.iter()) {
                        *dst = Self::fp8_to_f32(src as u8) * k_scale;
                    }
                    for (dst, &src) in v_dst.iter_mut().zip(v_src.iter()) {
                        *dst = Self::fp8_to_f32(src as u8) * v_scale;
                    }
                }
            }
        }

        let k_tensor = Tensor::from_vec(out_k, (self.n_kv_heads, self.head_dim), &self.device)?;
        let v_tensor = Tensor::from_vec(out_v, (self.n_kv_heads, self.head_dim), &self.device)?;
        Ok((k_tensor, v_tensor))
    }

    /// Alias for `read_quantized_kv_token`.
    pub fn read_kv_token(
        &self,
        layer: usize,
        block: usize,
        offset: usize,
    ) -> Result<(Tensor, Tensor), EngineError> {
        self.read_quantized_kv_token(layer, block, offset)
    }

    /// Gathers and dequantizes historical context for `layer` using a sequence of physical block IDs and total tokens.
    /// Returns (K, V) tensors of shape `[num_tokens, n_kv_heads, head_dim]`.
    pub fn gather_dequantized_blocks(
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

        if num_tokens == 0 || blocks.is_empty() {
            let empty_k =
                Tensor::zeros((0, self.n_kv_heads, self.head_dim), DType::F32, &self.device)?;
            let empty_v =
                Tensor::zeros((0, self.n_kv_heads, self.head_dim), DType::F32, &self.device)?;
            return Ok((empty_k, empty_v));
        }

        let elements_per_token = self.n_kv_heads * self.head_dim;
        let mut out_k = vec![0.0f32; num_tokens * elements_per_token];
        let mut out_v = vec![0.0f32; num_tokens * elements_per_token];

        for t in 0..num_tokens {
            let block_idx = t / self.block_size;
            let offset = t % self.block_size;

            if block_idx >= blocks.len() {
                return Err(EngineError::Model(format!(
                    "Token index {} requires block {} but only {} blocks provided",
                    t,
                    block_idx,
                    blocks.len()
                )));
            }
            let physical_block = blocks[block_idx];
            if physical_block >= self.num_blocks {
                return Err(EngineError::Model(format!(
                    "Physical block index {} out of bounds ({})",
                    physical_block, self.num_blocks
                )));
            }

            let blk = &self.blocks[layer][physical_block];
            let token_dst_start = t * elements_per_token;

            for h in 0..self.n_kv_heads {
                let scale_idx = offset * self.n_kv_heads + h;
                let k_scale = blk.k_scales[scale_idx];
                let v_scale = blk.v_scales[scale_idx];

                let src_start = offset * elements_per_token + h * self.head_dim;
                let dst_start = token_dst_start + h * self.head_dim;

                let k_dst = &mut out_k[dst_start..dst_start + self.head_dim];
                let k_src = &blk.k_quant[src_start..src_start + self.head_dim];
                let v_dst = &mut out_v[dst_start..dst_start + self.head_dim];
                let v_src = &blk.v_quant[src_start..src_start + self.head_dim];

                match self.quant_mode {
                    QuantMode::Int8 => {
                        for (dst, &src) in k_dst.iter_mut().zip(k_src.iter()) {
                            *dst = (src as f32) * k_scale;
                        }
                        for (dst, &src) in v_dst.iter_mut().zip(v_src.iter()) {
                            *dst = (src as f32) * v_scale;
                        }
                    }
                    QuantMode::Fp8 => {
                        for (dst, &src) in k_dst.iter_mut().zip(k_src.iter()) {
                            *dst = Self::fp8_to_f32(src as u8) * k_scale;
                        }
                        for (dst, &src) in v_dst.iter_mut().zip(v_src.iter()) {
                            *dst = Self::fp8_to_f32(src as u8) * v_scale;
                        }
                    }
                }
            }
        }

        let k_tensor = Tensor::from_vec(
            out_k,
            (num_tokens, self.n_kv_heads, self.head_dim),
            &self.device,
        )?;
        let v_tensor = Tensor::from_vec(
            out_v,
            (num_tokens, self.n_kv_heads, self.head_dim),
            &self.device,
        )?;
        Ok((k_tensor, v_tensor))
    }

    /// Gathers historical context from `BlockTable` for given layer.
    pub fn gather_dequantized_context(
        &self,
        layer: usize,
        block_table: &BlockTable,
    ) -> Result<(Tensor, Tensor), EngineError> {
        self.gather_dequantized_blocks(layer, block_table.blocks(), block_table.num_tokens())
    }

    /// Alias for `gather_dequantized_blocks`.
    pub fn gather_blocks(
        &self,
        layer: usize,
        blocks: &[usize],
        num_tokens: usize,
    ) -> Result<(Tensor, Tensor), EngineError> {
        self.gather_dequantized_blocks(layer, blocks, num_tokens)
    }

    /// Alias for `gather_dequantized_context`.
    pub fn gather_context(
        &self,
        layer: usize,
        block_table: &BlockTable,
    ) -> Result<(Tensor, Tensor), EngineError> {
        self.gather_dequantized_context(layer, block_table)
    }

    /// Memory consumption in bytes per token stored in quantized cache.
    pub fn memory_bytes_per_token(&self) -> usize {
        let elements = self.n_kv_heads * self.head_dim;
        let quant_bytes = 2 * elements * std::mem::size_of::<i8>();
        let scale_bytes = 2 * self.n_kv_heads * std::mem::size_of::<f32>();
        quant_bytes + scale_bytes
    }

    /// Memory consumption in bytes per token stored in standard FP32 KV cache.
    pub fn fp32_memory_bytes_per_token(&self) -> usize {
        2 * self.n_kv_heads * self.head_dim * std::mem::size_of::<f32>()
    }

    /// Physical memory reduction ratio compared to standard FP32 KV cache.
    pub fn memory_reduction_ratio(&self) -> f32 {
        self.fp32_memory_bytes_per_token() as f32 / self.memory_bytes_per_token() as f32
    }

    /// Total memory allocation in bytes for the entire quantized cache storage.
    pub fn total_memory_bytes(&self) -> usize {
        self.num_layers
            * self.num_blocks
            * (2 * self.block_size * self.n_kv_heads * self.head_dim * std::mem::size_of::<i8>()
                + 2 * self.block_size * self.n_kv_heads * std::mem::size_of::<f32>())
    }

    #[inline(always)]
    fn fp8_to_f32(b: u8) -> f32 {
        static FP8_LOOKUP: std::sync::LazyLock<[f32; 256]> = std::sync::LazyLock::new(|| {
            let mut table = [0.0f32; 256];
            for (i, entry) in table.iter_mut().enumerate() {
                *entry = float8::F8E4M3::from_bits(i as u8).to_f32();
            }
            table
        });
        FP8_LOOKUP[b as usize]
    }

    #[inline]
    fn quantize_head_slice(quant_mode: QuantMode, slice: &[f32], out_quant: &mut [i8]) -> f32 {
        let mut amax = 0.0f32;
        for &x in slice {
            let ax = x.abs();
            if ax > amax {
                amax = ax;
            }
        }

        match quant_mode {
            QuantMode::Int8 => {
                let scale = if amax > 0.0 { amax / 127.0 } else { 1.0 };
                let inv_scale = 1.0 / scale;
                for (i, &x) in slice.iter().enumerate() {
                    out_quant[i] = (x * inv_scale).round().clamp(-127.0, 127.0) as i8;
                }
                scale
            }
            QuantMode::Fp8 => {
                let scale = if amax > 0.0 {
                    let exp = (amax / 64.0).log2().ceil() as i32;
                    2.0f32.powi(exp)
                } else {
                    1.0
                };
                let inv_scale = 1.0 / scale;
                for (i, &x) in slice.iter().enumerate() {
                    let scaled = x * inv_scale;
                    let fp8 = float8::F8E4M3::from_f32(scaled);
                    out_quant[i] = fp8.to_bits() as i8;
                }
                scale
            }
        }
    }
}
