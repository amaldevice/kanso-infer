use crate::error::EngineError;
use candle_core::{DType, Device, Layout, Storage, Tensor};

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
