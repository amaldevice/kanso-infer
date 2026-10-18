use crate::error::EngineError;
use candle_core::{CpuStorage, DType, Device, InplaceOp1, Layout, Storage, Tensor};

#[cfg(target_os = "macos")]
#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    pub fn cblas_sgemm(
        order: i32,
        trans_a: i32,
        trans_b: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );

    pub fn cblas_sgemv(
        order: i32,
        trans_a: i32,
        m: i32,
        n: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        x: *const f32,
        inc_x: i32,
        beta: f32,
        y: *mut f32,
        inc_y: i32,
    );
}

pub const CBLAS_ROW_MAJOR: i32 = 101;
pub const CBLAS_COL_MAJOR: i32 = 102;
pub const CBLAS_NO_TRANS: i32 = 111;
pub const CBLAS_TRANS: i32 = 112;

/// Checks whether Apple Accelerate / AMX coprocessor backend is supported on current platform.
pub fn has_accelerate_support() -> bool {
    cfg!(target_os = "macos")
}

/// 2D General Matrix Multiplication: C = A * B
/// A is [M, K], B is [K, N], C is [M, N] in row-major layout.
pub fn amx_gemm(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    assert!(a.len() >= m * k, "A buffer too small");
    assert!(b.len() >= k * n, "B buffer too small");
    assert!(c.len() >= m * n, "C buffer too small");

    #[cfg(target_os = "macos")]
    unsafe {
        if m == 1 {
            cblas_sgemv(
                CBLAS_ROW_MAJOR,
                CBLAS_TRANS,
                k as i32,
                n as i32,
                1.0,
                b.as_ptr(),
                n as i32,
                a.as_ptr(),
                1,
                0.0,
                c.as_mut_ptr(),
                1,
            );
        } else {
            cblas_sgemm(
                CBLAS_ROW_MAJOR,
                CBLAS_NO_TRANS,
                CBLAS_NO_TRANS,
                m as i32,
                n as i32,
                k as i32,
                1.0,
                a.as_ptr(),
                k as i32,
                b.as_ptr(),
                n as i32,
                0.0,
                c.as_mut_ptr(),
                n as i32,
            );
        }
    }

    #[cfg(not(target_os = "macos"))]
    {
        fallback_gemm(m, n, k, a, b, c);
    }
}

/// 2D Matrix Multiplication with transposed B: C = A * B^T
/// A is [M, K], B is [N, K], C is [M, N] in row-major layout.
pub fn amx_gemm_b_trans(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    assert!(a.len() >= m * k, "A buffer too small");
    assert!(b.len() >= n * k, "B buffer too small");
    assert!(c.len() >= m * n, "C buffer too small");

    #[cfg(target_os = "macos")]
    unsafe {
        if m == 1 {
            cblas_sgemv(
                CBLAS_ROW_MAJOR,
                CBLAS_NO_TRANS,
                n as i32,
                k as i32,
                1.0,
                b.as_ptr(),
                k as i32,
                a.as_ptr(),
                1,
                0.0,
                c.as_mut_ptr(),
                1,
            );
        } else {
            cblas_sgemm(
                CBLAS_ROW_MAJOR,
                CBLAS_NO_TRANS,
                CBLAS_TRANS,
                m as i32,
                n as i32,
                k as i32,
                1.0,
                a.as_ptr(),
                k as i32,
                b.as_ptr(),
                k as i32,
                0.0,
                c.as_mut_ptr(),
                n as i32,
            );
        }
    }

    #[cfg(not(target_os = "macos"))]
    {
        fallback_gemm_b_trans(m, n, k, a, b, c);
    }
}

/// C = A * B^T where A is [M, K] with row stride `lda` and B is a contiguous [N, K]
/// block (ticket #48: the paged KV pool hands A rows `n_kv_heads * head_dim` apart).
#[allow(clippy::too_many_arguments)]
pub fn amx_gemm_a_strided_b_trans(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    lda: usize,
    b: &[f32],
    c: &mut [f32],
) {
    assert!(
        a.len() >= m.saturating_sub(1) * lda + k,
        "A buffer too small"
    );
    assert!(b.len() >= n * k, "B buffer too small");
    assert!(c.len() >= m * n, "C buffer too small");

    #[cfg(target_os = "macos")]
    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_TRANS,
            m as i32,
            n as i32,
            k as i32,
            1.0,
            a.as_ptr(),
            lda as i32,
            b.as_ptr(),
            k as i32,
            0.0,
            c.as_mut_ptr(),
            n as i32,
        );
    }

    #[cfg(not(target_os = "macos"))]
    {
        for i in 0..m {
            let a_row = &a[i * lda..i * lda + k];
            for j in 0..n {
                let b_row = &b[j * k..(j + 1) * k];
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += a_row[p] * b_row[p];
                }
                c[i * n + j] = sum;
            }
        }
    }
}

/// C = A * B where A is a contiguous [M, K] block and B is [K, N] with row stride `ldb`
/// (ticket #48: the paged KV pool hands B rows `n_kv_heads * head_dim` apart).
#[allow(clippy::too_many_arguments)]
pub fn amx_gemm_b_strided(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    ldb: usize,
    c: &mut [f32],
) {
    assert!(a.len() >= m * k, "A buffer too small");
    assert!(
        b.len() >= k.saturating_sub(1) * ldb + n,
        "B buffer too small"
    );
    assert!(c.len() >= m * n, "C buffer too small");

    #[cfg(target_os = "macos")]
    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_NO_TRANS,
            m as i32,
            n as i32,
            k as i32,
            1.0,
            a.as_ptr(),
            k as i32,
            b.as_ptr(),
            ldb as i32,
            0.0,
            c.as_mut_ptr(),
            n as i32,
        );
    }

    #[cfg(not(target_os = "macos"))]
    {
        for i in 0..m {
            let a_row = &a[i * k..(i + 1) * k];
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += a_row[p] * b[p * ldb + j];
                }
                c[i * n + j] = sum;
            }
        }
    }
}

/// Matrix-vector multiplication: y = A * x
/// A is [M, K], x is [K], y is [M] in row-major layout.
pub fn amx_gemv(m: usize, k: usize, a: &[f32], x: &[f32], y: &mut [f32]) {
    assert!(a.len() >= m * k, "A buffer too small");
    assert!(x.len() >= k, "x buffer too small");
    assert!(y.len() >= m, "y buffer too small");

    #[cfg(target_os = "macos")]
    unsafe {
        cblas_sgemv(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            m as i32,
            k as i32,
            1.0,
            a.as_ptr(),
            k as i32,
            x.as_ptr(),
            1,
            0.0,
            y.as_mut_ptr(),
            1,
        );
    }

    #[cfg(not(target_os = "macos"))]
    {
        fallback_gemv(m, k, a, x, y);
    }
}

/// Non-macOS fallback 2D GEMM (scalar reference)
pub fn fallback_gemm(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    for i in 0..m {
        let a_row = &a[i * k..(i + 1) * k];
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a_row[p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

/// Non-macOS fallback 2D GEMM with transposed B (scalar reference)
pub fn fallback_gemm_b_trans(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    for i in 0..m {
        let a_row = &a[i * k..(i + 1) * k];
        for j in 0..n {
            let b_row = &b[j * k..(j + 1) * k];
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a_row[p] * b_row[p];
            }
            c[i * n + j] = sum;
        }
    }
}

/// Non-macOS fallback GEMV (scalar reference)
pub fn fallback_gemv(m: usize, k: usize, a: &[f32], x: &[f32], y: &mut [f32]) {
    for i in 0..m {
        let a_row = &a[i * k..(i + 1) * k];
        let mut sum = 0.0f32;
        for p in 0..k {
            sum += a_row[p] * x[p];
        }
        y[i] = sum;
    }
}

/// Fallback copies taken by `get_f32_vec` (the weight-sized path `amx_matmul`
/// avoids on recognized layouts). A layout regression shows up here before it
/// shows up in tok/s. Quantized weights never reach this path at all — see
/// `Model::linear`, which only routes dense F32 through AMX.
pub static AMX_FALLBACK_COPIES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Fallback copy path: materializes a tensor into a contiguous row-major `Vec`.
/// Only reached for layouts the GEMM entry points cannot consume in place.
fn get_f32_vec(tensor: &Tensor) -> Result<Vec<f32>, EngineError> {
    AMX_FALLBACK_COPIES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    eprintln!("amx_matmul: unrecognized layout, taking weight-sized fallback copy");
    Ok(tensor.flatten_all()?.to_vec1::<f32>()?)
}

/// The `rows * cols` f32 elements backing a row-major `[rows, cols]` tensor, borrowed
/// from its own storage. `None` when the layout would need a reshuffle.
fn contiguous_block<'a>(
    data: &'a [f32],
    layout: &Layout,
    rows: usize,
    cols: usize,
) -> Option<&'a [f32]> {
    if !layout.is_contiguous() {
        return None;
    }
    let start = layout.start_offset();
    let end = start.checked_add(rows.checked_mul(cols)?)?;
    data.get(start..end)
}

/// The `rows * cols` elements of a `[rows, cols]` tensor that is the *view* transpose of
/// a contiguous `[cols, rows]` block — what `.t()` on a loaded `[out, in]` weight
/// produces. The elements come back in `[cols, rows]` order for `amx_gemm_b_trans`.
fn transposed_block<'a>(
    data: &'a [f32],
    layout: &Layout,
    rows: usize,
    cols: usize,
) -> Option<&'a [f32]> {
    // Shape [rows, cols] read off a contiguous [cols, rows] block: stride (1, rows).
    if layout.stride() != [1, rows] {
        return None;
    }
    let start = layout.start_offset();
    let end = start.checked_add(rows.checked_mul(cols)?)?;
    data.get(start..end)
}

/// 2D GEMM straight out of the operands' storage. `a` must be contiguous `[m, k]`;
/// `b` may be contiguous `[k, n]` or the transposed view of a contiguous `[n, k]` block.
fn gemm_2d_in_place(
    a: &Tensor,
    b: &Tensor,
    m: usize,
    n: usize,
    k: usize,
) -> Option<Result<Tensor, EngineError>> {
    let (a_guard, a_layout) = a.storage_and_layout();
    let (b_guard, b_layout) = b.storage_and_layout();
    let Storage::Cpu(a_storage) = &*a_guard else {
        return None;
    };
    let Storage::Cpu(b_storage) = &*b_guard else {
        return None;
    };
    let a_data = a_storage.as_slice::<f32>().ok()?;
    let b_data = b_storage.as_slice::<f32>().ok()?;

    let a_block = contiguous_block(a_data, a_layout, m, k)?;
    let mut c = vec![0.0f32; m * n];
    match (
        contiguous_block(b_data, b_layout, k, n),
        transposed_block(b_data, b_layout, k, n),
    ) {
        (Some(b_block), _) => amx_gemm(m, n, k, a_block, b_block, &mut c),
        (None, Some(b_block)) => amx_gemm_b_trans(m, n, k, a_block, b_block, &mut c),
        (None, None) => return None,
    }
    Some(Tensor::from_vec(c, (m, n), &Device::Cpu).map_err(EngineError::from))
}

/// 3D GEMM straight out of the operands' storage, one `[m, k] x [k, n]` block per batch.
/// `a` must be contiguous `[batch, m, k]`; `b` is `[batch, k, n]` either contiguous or
/// the per-batch transpose of a contiguous `[batch, n, k]` block.
fn gemm_3d_in_place(
    a: &Tensor,
    b: &Tensor,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Option<Result<Tensor, EngineError>> {
    let (a_guard, a_layout) = a.storage_and_layout();
    let (b_guard, b_layout) = b.storage_and_layout();
    let Storage::Cpu(a_storage) = &*a_guard else {
        return None;
    };
    let Storage::Cpu(b_storage) = &*b_guard else {
        return None;
    };
    let a_data = a_storage.as_slice::<f32>().ok()?;
    let b_data = b_storage.as_slice::<f32>().ok()?;

    if !a_layout.is_contiguous() {
        return None;
    }
    let b_transposed = if b_layout.is_contiguous() {
        false
    } else if b_layout.stride() == [k * n, 1, k] {
        true
    } else {
        return None;
    };

    let a_start = a_layout.start_offset();
    let b_start = b_layout.start_offset();
    let a_end = a_start.checked_add(batch.checked_mul(m)?.checked_mul(k)?)?;
    let b_end = b_start.checked_add(batch.checked_mul(k)?.checked_mul(n)?)?;
    if a_end > a_data.len() || b_end > b_data.len() {
        return None;
    }

    let mut c = vec![0.0f32; batch * m * n];
    for i in 0..batch {
        let a_block = &a_data[a_start + i * m * k..a_start + (i + 1) * m * k];
        let b_block = &b_data[b_start + i * k * n..b_start + (i + 1) * k * n];
        let c_block = &mut c[i * m * n..(i + 1) * m * n];
        if b_transposed {
            amx_gemm_b_trans(m, n, k, a_block, b_block, c_block);
        } else {
            amx_gemm(m, n, k, a_block, b_block, c_block);
        }
    }
    Some(Tensor::from_vec(c, (batch, m, n), &Device::Cpu).map_err(EngineError::from))
}

/// Paged decode attention for one token (ticket #48): `out[h] = softmax(q[h] K[h]^T *
/// scale) V[h]` over the context, with `q` a contiguous `[n_kv_heads, m, head_dim]`
/// tensor (`m` = GQA group size for decode) and `k`/`v` the `[ctx, n_kv_heads,
/// head_dim]` pool views — or the fragmented gather, which shares their strides. Pool
/// rows go straight into Accelerate with their own leading dimension, so the history
/// is never copied; only the `[n_kv_heads, m, head_dim]` output and two
/// `[ctx, m]`-sized scratch buffers are allocated.
///
/// Returns `None` for non-CPU-F32 or unexpected layouts; the caller falls back to the
/// generic attention path.
///
/// Allocates its own output and scratch buffers; the decode path instead uses
/// [`AmxWorkspace::paged_decode_attention_into`], which reuses them across layers.
#[allow(clippy::too_many_lines)]
pub fn amx_paged_decode_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
) -> Option<Result<Tensor, EngineError>> {
    let (kv_heads, m, head_dim) = q.dims3().ok()?;
    let (ctx, ..) = k.dims3().ok()?;
    let mut out = vec![0.0f32; kv_heads * m * head_dim];
    let mut scores_t = vec![0.0f32; ctx * m];
    let mut probs = vec![0.0f32; m * ctx];
    paged_decode_attention_core(q, k, v, scale, &mut out, &mut scores_t, &mut probs)?;
    Some(Tensor::from_vec(out, (kv_heads, m, head_dim), &Device::Cpu).map_err(EngineError::from))
}

/// The paged decode attention kernel over caller-provided buffers: `out` is
/// `[n_kv_heads * m * head_dim]`, `scores_t` is `[ctx * m]`, `probs` is `[m * ctx]`.
///
/// `None` for non-CPU-F32 or unexpected layouts; `out` is then left untouched.
fn paged_decode_attention_core(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    out: &mut [f32],
    scores_t: &mut [f32],
    probs: &mut [f32],
) -> Option<()> {
    let (kv_heads, m, head_dim) = q.dims3().ok()?;
    let (ctx, kv_k, hd_k) = k.dims3().ok()?;
    let (ctx_v, kv_v, hd_v) = v.dims3().ok()?;
    if kv_k != kv_heads || kv_v != kv_heads || hd_k != head_dim || hd_v != head_dim {
        return None;
    }
    if ctx_v != ctx || ctx == 0 || m == 0 || kv_heads == 0 {
        return None;
    }
    if out.len() < kv_heads * m * head_dim || scores_t.len() < ctx * m || probs.len() < m * ctx {
        return None;
    }
    if q.dtype() != DType::F32 || k.dtype() != DType::F32 || v.dtype() != DType::F32 {
        return None;
    }
    if !matches!(q.device(), Device::Cpu)
        || !matches!(k.device(), Device::Cpu)
        || !matches!(v.device(), Device::Cpu)
    {
        return None;
    }

    let (q_guard, q_layout) = q.storage_and_layout();
    let (k_guard, k_layout) = k.storage_and_layout();
    let (v_guard, v_layout) = v.storage_and_layout();
    let Storage::Cpu(q_storage) = &*q_guard else {
        return None;
    };
    let Storage::Cpu(k_storage) = &*k_guard else {
        return None;
    };
    let Storage::Cpu(v_storage) = &*v_guard else {
        return None;
    };
    let q_data = q_storage.as_slice::<f32>().ok()?;
    let k_data = k_storage.as_slice::<f32>().ok()?;
    let v_data = v_storage.as_slice::<f32>().ok()?;

    if !q_layout.is_contiguous() {
        return None;
    }
    // `[ctx, n_kv_heads, head_dim]` with heads interleaved per token — the pool view
    // and the contiguous fragmented gather share these strides.
    let pool_stride = [kv_heads * head_dim, head_dim, 1];
    if k_layout.stride() != pool_stride || v_layout.stride() != pool_stride {
        return None;
    }

    let q_start = q_layout.start_offset();
    let k_start = k_layout.start_offset();
    let v_start = v_layout.start_offset();
    let q_end = q_start.checked_add(kv_heads * m * head_dim)?;
    let kv_end = ctx.checked_mul(kv_heads)?.checked_mul(head_dim)?;
    if q_end > q_data.len()
        || k_start.checked_add(kv_end)? > k_data.len()
        || v_start.checked_add(kv_end)? > v_data.len()
    {
        return None;
    }

    let row_stride = kv_heads * head_dim;
    let scores_t = &mut scores_t[..ctx * m];
    let probs = &mut probs[..m * ctx];
    for h in 0..kv_heads {
        let q_h = &q_data[q_start + h * m * head_dim..q_start + (h + 1) * m * head_dim];
        let k_h = &k_data[k_start + h * head_dim..];
        let v_h = &v_data[v_start + h * head_dim..];
        let out_h = &mut out[h * m * head_dim..(h + 1) * m * head_dim];

        // scores^T [ctx, m] = K_h [ctx, hd] x Q_h^T [hd, m]
        amx_gemm_a_strided_b_trans(ctx, m, head_dim, k_h, row_stride, q_h, scores_t);

        // Scale + softmax over the ctx axis, one column per query row; the probabilities
        // land in `probs` as a contiguous [m, ctx] block for the GEMV below.
        for g in 0..m {
            let mut max = f32::NEG_INFINITY;
            for t in 0..ctx {
                let s = scores_t[t * m + g] * scale;
                scores_t[t * m + g] = s;
                if s > max {
                    max = s;
                }
            }
            let mut sum = 0.0f32;
            for t in 0..ctx {
                let e = (scores_t[t * m + g] - max).exp();
                scores_t[t * m + g] = e;
                sum += e;
            }
            let inv = 1.0 / sum;
            for t in 0..ctx {
                probs[g * ctx + t] = scores_t[t * m + g] * inv;
            }
        }

        // out_h [m, hd] = P [m, ctx] x V_h [ctx, hd]
        amx_gemm_b_strided(m, head_dim, ctx, probs, v_h, row_stride, out_h);
    }
    Some(())
}

/// Reusable buffers for the AMX decode path (ticket #93 / spec #82 D16).
///
/// `amx_matmul` returns a fresh `[m, n]` `Vec` per call, so a decode step paid seven
/// small allocations per layer per token for its projection outputs plus three more in
/// the paged attention. This workspace keeps one output tensor per slot and fills it in
/// place instead, so a token allocates nothing on those seams.
///
/// **Aliasing contract:** a tensor handed back by a `*_into` call aliases the slot it was
/// written to, and stays valid only until the next `*_into` call on that same slot. Each
/// live intermediate in a forward pass therefore owns a distinct slot; the decode path's
/// slot numbers are fixed by [`AmxWorkspace::SLOT_*`] and never reused concurrently.
#[derive(Default)]
pub struct AmxWorkspace {
    /// One reusable output tensor per slot, grown on demand.
    slots: Vec<Option<Tensor>>,
    /// `[ctx * group]` attention score scratch, reused across layers.
    scores_t: Vec<f32>,
    /// `[group * ctx]` softmax scratch, reused across layers.
    probs: Vec<f32>,
    /// `[kv_heads * group * head_dim]` paged-attention output staging.
    attn_out: Vec<f32>,
}

impl std::fmt::Debug for AmxWorkspace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AmxWorkspace")
            .field("slots", &self.slots.len())
            .finish_non_exhaustive()
    }
}

/// Copies `data` into a tensor's own storage, row-major.
struct WriteF32<'a> {
    data: &'a [f32],
}

impl InplaceOp1 for WriteF32<'_> {
    fn name(&self) -> &'static str {
        "amx-workspace-write"
    }

    fn cpu_fwd(&self, storage: &mut CpuStorage, layout: &Layout) -> candle_core::Result<()> {
        let CpuStorage::F32(dst) = storage else {
            candle_core::bail!("amx workspace slot is not an F32 cpu tensor")
        };
        let Some((start, end)) = layout.contiguous_offsets() else {
            candle_core::bail!("amx workspace slot is not contiguous")
        };
        if self.data.len() != end - start {
            candle_core::bail!(
                "amx workspace write of {} elements into a {}-element slot",
                self.data.len(),
                end - start
            )
        }
        dst[start..end].copy_from_slice(self.data);
        Ok(())
    }
}

/// `C = A * W^T` written straight into a tensor's own storage: `a` is contiguous `[m, k]`,
/// `w` contiguous `[n, k]`, and the slot holds `[m, n]`.
struct GemmWeightInto<'a> {
    a: &'a [f32],
    w: &'a [f32],
    m: usize,
    n: usize,
    k: usize,
}

impl InplaceOp1 for GemmWeightInto<'_> {
    fn name(&self) -> &'static str {
        "amx-workspace-gemm"
    }

    fn cpu_fwd(&self, storage: &mut CpuStorage, layout: &Layout) -> candle_core::Result<()> {
        let CpuStorage::F32(dst) = storage else {
            candle_core::bail!("amx workspace slot is not an F32 cpu tensor")
        };
        let Some((start, end)) = layout.contiguous_offsets() else {
            candle_core::bail!("amx workspace slot is not contiguous")
        };
        let elems = self.m * self.n;
        if end - start != elems {
            candle_core::bail!(
                "amx workspace gemm writes {elems} elements into a {}-element slot",
                end - start
            )
        }
        amx_gemm_b_trans(
            self.m,
            self.n,
            self.k,
            self.a,
            self.w,
            &mut dst[start..start + elems],
        );
        Ok(())
    }
}

impl AmxWorkspace {
    /// Slot numbers the decode path uses. Each live intermediate owns one, so no two
    /// tensors returned to a forward pass ever share storage.
    pub const SLOT_Q: usize = 0;
    pub const SLOT_K: usize = 1;
    pub const SLOT_V: usize = 2;
    pub const SLOT_ATTN_OUT: usize = 3;
    pub const SLOT_FFN_GATE: usize = 4;
    pub const SLOT_FFN_UP: usize = 5;
    pub const SLOT_FFN_DOWN: usize = 6;
    pub const SLOT_LOGITS: usize = 7;
    pub const SLOT_PAGED_ATTN: usize = 8;

    /// Returns slot `slot` sized to `dims`, allocating only when the shape changed.
    fn slot(&mut self, slot: usize, dims: &[usize]) -> Result<Tensor, EngineError> {
        while self.slots.len() <= slot {
            self.slots.push(None);
        }
        if let Some(t) = &self.slots[slot]
            && t.dims() == dims
        {
            return Ok(t.clone());
        }
        let t = Tensor::zeros(dims, DType::F32, &Device::Cpu)?;
        self.slots[slot] = Some(t.clone());
        Ok(t)
    }

    /// Drops every buffer. Called when the model changes device, so a workspace can
    /// never outlive the device its slots were built for.
    pub fn clear(&mut self) {
        self.slots.clear();
        self.scores_t = Vec::new();
        self.probs = Vec::new();
        self.attn_out = Vec::new();
    }

    /// `x @ w^T` written into `slot`: `x` is `[m, k]`, `w` is `[n, k]` (the loaded weight
    /// layout, so no transposed view has to be built per call).
    ///
    /// Returns a tensor aliasing `slot` — see the type-level aliasing contract. Layouts
    /// the in-place GEMM cannot read fall back to [`amx_matmul`] and store its result in
    /// the slot, so callers never see a different answer, only a slower path.
    pub fn matmul_weight_into(
        &mut self,
        slot: usize,
        x: &Tensor,
        w: &Tensor,
    ) -> Result<Tensor, EngineError> {
        if let Some(out) = self.try_matmul_weight_into(slot, x, w)? {
            return Ok(out);
        }
        let out = amx_matmul(x, &w.t()?)?;
        let target = self.slot(slot, out.dims())?;
        target.inplace_op1(&WriteF32 {
            data: &out.flatten_all()?.to_vec1::<f32>()?,
        })?;
        Ok(target)
    }

    /// The in-place path of [`AmxWorkspace::matmul_weight_into`]; `None` when the operand
    /// layouts need a copy the workspace cannot avoid.
    fn try_matmul_weight_into(
        &mut self,
        slot: usize,
        x: &Tensor,
        w: &Tensor,
    ) -> Result<Option<Tensor>, EngineError> {
        if x.dtype() != DType::F32
            || w.dtype() != DType::F32
            || !matches!(x.device(), Device::Cpu)
            || !matches!(w.device(), Device::Cpu)
        {
            return Ok(None);
        }
        let x = x.contiguous()?;
        let (x_dims, w_dims) = (x.dims(), w.dims());
        if x_dims.len() != 2 || w_dims.len() != 2 || x_dims[1] != w_dims[1] {
            return Ok(None);
        }
        let (m, k, n) = (x_dims[0], x_dims[1], w_dims[0]);

        // Both operands are read out of their own storage; the slot is written in place.
        let target = self.slot(slot, &[m, n])?;
        let (x_guard, x_layout) = x.storage_and_layout();
        let (w_guard, w_layout) = w.storage_and_layout();
        let (Storage::Cpu(x_storage), Storage::Cpu(w_storage)) = (&*x_guard, &*w_guard) else {
            return Ok(None);
        };
        let (Ok(x_data), Ok(w_data)) = (x_storage.as_slice::<f32>(), w_storage.as_slice::<f32>())
        else {
            return Ok(None);
        };
        let (Some(a_block), Some(w_block)) = (
            contiguous_block(x_data, x_layout, m, k),
            contiguous_block(w_data, w_layout, n, k),
        ) else {
            return Ok(None);
        };

        target.inplace_op1(&GemmWeightInto {
            a: a_block,
            w: w_block,
            m,
            n,
            k,
        })?;
        Ok(Some(target))
    }

    /// Paged decode attention written into `slot` (ticket #93): the three scratch buffers
    /// [`amx_paged_decode_attention`] allocated per call are reused across layers.
    ///
    /// Returns a tensor aliasing `slot`, or `None` for layouts the kernel cannot read —
    /// the caller then takes its own fallback path.
    pub fn paged_decode_attention_into(
        &mut self,
        slot: usize,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        scale: f32,
    ) -> Option<Result<Tensor, EngineError>> {
        let (kv_heads, group, head_dim) = q.dims3().ok()?;
        let (ctx, ..) = k.dims3().ok()?;

        self.attn_out.clear();
        self.attn_out.resize(kv_heads * group * head_dim, 0.0f32);
        self.scores_t.clear();
        self.scores_t.resize(ctx * group, 0.0f32);
        self.probs.clear();
        self.probs.resize(group * ctx, 0.0f32);

        paged_decode_attention_core(
            q,
            k,
            v,
            scale,
            &mut self.attn_out,
            &mut self.scores_t,
            &mut self.probs,
        )?;

        let target = match self.slot(slot, &[kv_heads, group, head_dim]) {
            Ok(t) => t,
            Err(e) => return Some(Err(e)),
        };
        if let Err(e) = target.inplace_op1(&WriteF32 {
            data: &self.attn_out,
        }) {
            return Some(Err(EngineError::from(e)));
        }
        Some(Ok(target))
    }
}

/// AMX-accelerated matrix multiplication for Candle Tensors.
///
/// Consumes the operands' existing storage: a `.t()` weight view is handed to
/// `amx_gemm_b_trans` (`C = A * W^T` via `CblasTrans`) instead of being materialized, so
/// a decode step allocates only its own `[m, n]` output.
pub fn amx_matmul(a: &Tensor, b: &Tensor) -> Result<Tensor, EngineError> {
    if a.dtype() != DType::F32
        || b.dtype() != DType::F32
        || !matches!(a.device(), Device::Cpu)
        || !matches!(b.device(), Device::Cpu)
    {
        return Ok(a.matmul(b)?);
    }

    // No-op (storage is shared) when the activations are already row-major.
    let a = a.contiguous()?;
    let a_dims = a.dims();
    let b_dims = b.dims();

    // 2D x 2D: [M, K] x [K, N] -> [M, N]
    if a_dims.len() == 2 && b_dims.len() == 2 && a_dims[1] == b_dims[0] {
        let (m, k, n) = (a_dims[0], a_dims[1], b_dims[1]);

        if let Some(out) = gemm_2d_in_place(&a, b, m, n, k) {
            return out;
        }

        let a_data = get_f32_vec(&a)?;
        let b_data = get_f32_vec(b)?;
        let mut c_data = vec![0.0f32; m * n];

        amx_gemm(m, n, k, &a_data, &b_data, &mut c_data);
        return Ok(Tensor::from_vec(c_data, (m, n), &Device::Cpu)?);
    }

    // 3D x 3D: [B, M, K] x [B, K, N] -> [B, M, N]
    if a_dims.len() == 3 && b_dims.len() == 3 && a_dims[0] == b_dims[0] && a_dims[2] == b_dims[1] {
        let batch = a_dims[0];
        let (m, k, n) = (a_dims[1], a_dims[2], b_dims[2]);

        if let Some(out) = gemm_3d_in_place(&a, b, batch, m, n, k) {
            return out;
        }

        let a_data = get_f32_vec(&a)?;
        let b_data = get_f32_vec(b)?;
        let mut c_data = vec![0.0f32; batch * m * n];

        for i in 0..batch {
            let a_off = i * m * k;
            let b_off = i * k * n;
            let c_off = i * m * n;
            amx_gemm(
                m,
                n,
                k,
                &a_data[a_off..a_off + m * k],
                &b_data[b_off..b_off + k * n],
                &mut c_data[c_off..c_off + m * n],
            );
        }
        return Ok(Tensor::from_vec(c_data, (batch, m, n), &Device::Cpu)?);
    }

    // Fallback to candle matmul for any higher dims or broadcasting
    Ok(a.matmul(b)?)
}
