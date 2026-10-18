//! Metal decode attention over the block pool (ticket #86).
//!
//! Decode used to reach attention through [`crate::model::Model::matmul`], which
//! materializes the `[ctx, n_kv_heads, head_dim]` pool view per layer per token because
//! candle's MLX GEMM rejects a double-transposed operand. This module is the single seam
//! that replaces it on Metal: `gpu_paged_attention_v2_buffer` reads K/V straight out of
//! the block pool through the block table, so the history is never copied per layer.
//!
//! The kernel needs `metal::Buffer`s while the pool lives in candle tensors, so the seam
//! owns a device copy: uploaded once per (layer, pool), then refreshed one token row at a
//! time — the only rows a decode step changes.

use crate::error::EngineError;
use crate::kv_cache::KvStorage;
use crate::metal::context::MetalContext;
use crate::metal::paged_attention::{
    PagedAttentionParams, gpu_paged_attention_v2_buffer, validate_paged_attention_params,
};
use candle_core::{Device, Tensor, TensorId};
use metal::Buffer;

/// Attention shapes the kernel needs, taken from the model config once per call.
pub(crate) struct AttnShape {
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub scale: f32,
}

/// Device copy of one layer's K/V pool, flattened to `[num_blocks * block_size,
/// n_kv_heads, head_dim]` — the layout the kernel indexes.
struct LayerPool {
    /// Tensor id of the pool these buffers were uploaded from.
    source: TensorId,
    k: Buffer,
    v: Buffer,
}

pub(crate) struct PagedAttnSeam {
    ctx: MetalContext,
    device: Device,
    /// One entry per model layer; `None` until that layer first decodes.
    pools: Vec<Option<LayerPool>>,
    /// Block table / context lens last uploaded. Every layer of a step reads the same
    /// ones, so a single upload serves the whole layer sweep.
    tables_key: (Vec<u32>, Vec<u32>),
    tables_buf: Option<Buffer>,
    lens_buf: Option<Buffer>,
    /// Q staging and kernel output, kept at the widest size seen so far.
    q_buf: Option<Buffer>,
    q_len: usize,
    out_buf: Option<Buffer>,
    out_len: usize,
}

impl PagedAttnSeam {
    pub(crate) fn new(num_layers: usize, device: &Device) -> Result<Self, EngineError> {
        Ok(Self {
            ctx: MetalContext::new()?,
            device: device.clone(),
            pools: (0..num_layers).map(|_| None).collect(),
            tables_key: (Vec::new(), Vec::new()),
            tables_buf: None,
            lens_buf: None,
            q_buf: None,
            q_len: 0,
            out_buf: None,
            out_len: 0,
        })
    }

    /// Paged attention for one layer of a pure-decode step.
    ///
    /// `q` is `[num_seqs, num_heads, head_dim]` (a decode step is one query token per
    /// sequence), `block_tables` the padded `[num_seqs * max_blocks]` physical block ids,
    /// `context_lens` the length each sequence attends over, and `slots` the pool rows the
    /// step just wrote — the only rows the device copy has to refresh.
    ///
    /// Returns `[num_seqs, num_heads * head_dim]`.
    #[allow(clippy::too_many_arguments)] // kernel surface: one per dispatch argument
    pub(crate) fn attention(
        &mut self,
        q: &Tensor,
        kv: &KvStorage,
        layer: usize,
        block_tables: &[u32],
        max_blocks: usize,
        context_lens: &[u32],
        slots: &[usize],
        shape: &AttnShape,
    ) -> Result<Tensor, EngineError> {
        let num_seqs = context_lens.len();
        if num_seqs == 0 || max_blocks == 0 {
            return Err(EngineError::Metal(
                "metal paged attention: empty batch or block table".into(),
            ));
        }
        if block_tables.len() < num_seqs * max_blocks {
            return Err(EngineError::Metal(format!(
                "metal paged attention: {} block ids for {num_seqs} sequences of {max_blocks} blocks",
                block_tables.len()
            )));
        }
        if slots.len() != num_seqs {
            return Err(EngineError::Metal(format!(
                "metal paged attention: {} slots for {num_seqs} sequences",
                slots.len()
            )));
        }

        let params = PagedAttentionParams {
            num_seqs: num_seqs as u32,
            num_heads: shape.num_heads as u32,
            num_kv_heads: shape.num_kv_heads as u32,
            head_dim: shape.head_dim as u32,
            block_size: kv.block_size as u32,
            max_num_blocks_per_seq: max_blocks as u32,
            scale: shape.scale,
        };
        // Anything the shader cannot represent is an error here, never a quiet fall back
        // to the copying path (ticket #51 / #86).
        validate_paged_attention_params(&params)?;

        self.upload_tables(block_tables, context_lens);

        let q_host = q
            .to_device(&Device::Cpu)?
            .contiguous()?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let out_elems = num_seqs * shape.num_heads * shape.head_dim;
        if q_host.len() != out_elems {
            return Err(EngineError::Metal(format!(
                "metal paged attention: q holds {} floats, expected {out_elems}",
                q_host.len()
            )));
        }
        self.upload_q(&q_host);
        self.reserve_out(out_elems);

        self.refresh_pool(kv, layer, slots)?;

        let pool = self.pools[layer]
            .as_ref()
            .ok_or_else(|| EngineError::Metal("metal paged attention: pool missing".into()))?;
        let tables_buf = self
            .tables_buf
            .as_ref()
            .ok_or_else(|| EngineError::Metal("metal paged attention: tables missing".into()))?;
        let lens_buf = self
            .lens_buf
            .as_ref()
            .ok_or_else(|| EngineError::Metal("metal paged attention: lens missing".into()))?;
        let q_buf = self
            .q_buf
            .as_ref()
            .ok_or_else(|| EngineError::Metal("metal paged attention: q missing".into()))?;
        let out_buf = self
            .out_buf
            .as_ref()
            .ok_or_else(|| EngineError::Metal("metal paged attention: out missing".into()))?;

        gpu_paged_attention_v2_buffer(
            &self.ctx, q_buf, &pool.k, &pool.v, tables_buf, lens_buf, out_buf, &params,
        )?;

        let mut out = vec![0.0f32; out_elems];
        unsafe {
            let ptr = out_buf.contents() as *const f32;
            std::ptr::copy_nonoverlapping(ptr, out.as_mut_ptr(), out_elems);
        }
        Ok(Tensor::from_vec(
            out,
            (num_seqs, shape.num_heads * shape.head_dim),
            &self.device,
        )?)
    }

    /// Uploads the block table and context lens when they change. The engine builds them
    /// once per step and every layer reads the same ones.
    fn upload_tables(&mut self, table: &[u32], lens: &[u32]) {
        if self.tables_key.0 == table && self.tables_key.1 == lens {
            return;
        }
        self.tables_buf = Some(self.ctx.alloc_buffer_with_data(table));
        self.lens_buf = Some(self.ctx.alloc_buffer_with_data(lens));
        self.tables_key = (table.to_vec(), lens.to_vec());
    }

    fn upload_q(&mut self, data: &[f32]) {
        if self.q_len < data.len() || self.q_buf.is_none() {
            self.q_buf = Some(self.ctx.alloc_buffer_with_data(data));
            self.q_len = data.len();
            return;
        }
        if let Some(buf) = &self.q_buf {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    data.as_ptr(),
                    buf.contents() as *mut f32,
                    data.len(),
                );
            }
        }
    }

    fn reserve_out(&mut self, len: usize) {
        if self.out_len < len || self.out_buf.is_none() {
            self.out_buf = Some(self.ctx.alloc_buffer::<f32>(len));
            self.out_len = len;
        }
    }

    /// Brings `layer`'s device copy of the pool up to date: whole-pool upload the first
    /// time this layer is seen (or when the pool tensor is replaced), then one row copy
    /// per slot the step wrote. A rewritten slot — block reuse, speculation rollback — is
    /// just another row copy, so it needs no special case.
    fn refresh_pool(
        &mut self,
        kv: &KvStorage,
        layer: usize,
        slots: &[usize],
    ) -> Result<(), EngineError> {
        let source = kv.kv_k[layer].id();
        let rows = kv.num_blocks * kv.block_size;
        let row_elems = kv.n_kv_heads * kv.head_dim;

        let current = self.pools[layer]
            .as_ref()
            .is_some_and(|pool| pool.source == source);
        if !current {
            let k = self.upload_pool(&kv.kv_k[layer], rows, row_elems)?;
            let v = self.upload_pool(&kv.kv_v[layer], rows, row_elems)?;
            self.pools[layer] = Some(LayerPool { source, k, v });
            return Ok(());
        }

        for &slot in slots {
            if slot >= rows {
                return Err(EngineError::Metal(format!(
                    "metal paged attention: slot {slot} outside the {rows}-row pool"
                )));
            }
            let k_row = self.read_row(&kv.kv_k[layer], slot)?;
            let v_row = self.read_row(&kv.kv_v[layer], slot)?;
            if k_row.len() != row_elems || v_row.len() != row_elems {
                return Err(EngineError::Metal(format!(
                    "metal paged attention: pool row holds {} / {} floats, expected {row_elems}",
                    k_row.len(),
                    v_row.len()
                )));
            }
            let pool = self.pools[layer].as_ref().expect("checked above");
            let offset = slot * row_elems;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    k_row.as_ptr(),
                    (pool.k.contents() as *mut f32).add(offset),
                    row_elems,
                );
                std::ptr::copy_nonoverlapping(
                    v_row.as_ptr(),
                    (pool.v.contents() as *mut f32).add(offset),
                    row_elems,
                );
            }
        }
        Ok(())
    }

    /// One `[n_kv_heads, head_dim]` pool row.
    fn read_row(&self, pool: &Tensor, slot: usize) -> Result<Vec<f32>, EngineError> {
        Ok(pool
            .flatten(0, 1)?
            .narrow(0, slot, 1)?
            .to_device(&Device::Cpu)?
            .contiguous()?
            .flatten_all()?
            .to_vec1::<f32>()?)
    }

    /// Whole-pool upload: `[num_blocks, block_size, n_kv_heads, head_dim]` flattened.
    fn upload_pool(
        &self,
        pool: &Tensor,
        rows: usize,
        row_elems: usize,
    ) -> Result<Buffer, EngineError> {
        let host = pool
            .to_device(&Device::Cpu)?
            .contiguous()?
            .flatten_all()?
            .to_vec1::<f32>()?;
        if host.len() != rows * row_elems {
            return Err(EngineError::Metal(format!(
                "metal paged attention: pool holds {} floats, expected {}",
                host.len(),
                rows * row_elems
            )));
        }
        Ok(self.ctx.alloc_buffer_with_data(&host))
    }
}
