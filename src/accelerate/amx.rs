use candle_core::{DType, Device, Tensor};
use crate::error::EngineError;

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

/// Helper to extract CPU f32 slice if contiguous, or clone into Vec.
fn get_f32_vec(tensor: &Tensor) -> Result<Vec<f32>, EngineError> {
    Ok(tensor.flatten_all()?.to_vec1::<f32>()?)
}

/// AMX-accelerated matrix multiplication for Candle Tensors.
pub fn amx_matmul(a: &Tensor, b: &Tensor) -> Result<Tensor, EngineError> {
    if a.dtype() != DType::F32
        || b.dtype() != DType::F32
        || !matches!(a.device(), Device::Cpu)
        || !matches!(b.device(), Device::Cpu)
    {
        return Ok(a.matmul(b)?);
    }

    let a_dims = a.dims();
    let b_dims = b.dims();

    // 2D x 2D: [M, K] x [K, N] -> [M, N]
    if a_dims.len() == 2 && b_dims.len() == 2 && a_dims[1] == b_dims[0] {
        let m = a_dims[0];
        let k = a_dims[1];
        let n = b_dims[1];

        let a_data = get_f32_vec(a)?;
        let b_data = get_f32_vec(b)?;
        let mut c_data = vec![0.0f32; m * n];

        amx_gemm(m, n, k, &a_data, &b_data, &mut c_data);
        return Ok(Tensor::from_vec(c_data, (m, n), &Device::Cpu)?);
    }

    // 3D x 3D: [B, M, K] x [B, K, N] -> [B, M, N]
    if a_dims.len() == 3 && b_dims.len() == 3 && a_dims[0] == b_dims[0] && a_dims[2] == b_dims[1] {
        let b_batch = a_dims[0];
        let m = a_dims[1];
        let k = a_dims[2];
        let n = b_dims[2];

        let a_data = get_f32_vec(a)?;
        let b_data = get_f32_vec(b)?;
        let mut c_data = vec![0.0f32; b_batch * m * n];

        for i in 0..b_batch {
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
        return Ok(Tensor::from_vec(c_data, (b_batch, m, n), &Device::Cpu)?);
    }

    // Fallback to candle matmul for any higher dims or broadcasting
    Ok(a.matmul(b)?)
}
