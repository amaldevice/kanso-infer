//! Ticket #52: `amx_matmul` must not materialize weight-sized copies of its inputs.
//!
//! Decode calls `amx_matmul(x, w.t())` for every projection of every layer ([out, in]
//! weights viewed transposed). The old path did `flatten_all().to_vec1()` on both
//! operands, i.e. a full copy of every weight matrix per token. This test counts
//! allocator traffic during a single `amx_matmul` call and asserts none of it is
//! weight-sized, and checks the no-copy path stays numerically exact.
//!
//! The counting global allocator lives in this file's own test binary so its counters
//! are not polluted by other test modules running in parallel.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use candle_core::{Device, Tensor};
use kanso_infer::accelerate::amx::amx_matmul;

static COUNTING: AtomicBool = AtomicBool::new(false);
static BYTES_ALLOCATED: AtomicUsize = AtomicUsize::new(0);

struct CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: delegates to `System` with the same layout.
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() && COUNTING.load(Ordering::Relaxed) {
            BYTES_ALLOCATED.fetch_add(layout.size(), Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr`/`layout` come from a matching `alloc` call.
        unsafe { System.dealloc(ptr, layout) };
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: delegates to `System` with the same pointer and layout.
        let new_ptr = unsafe { System.realloc(ptr, layout, new_size) };
        if !new_ptr.is_null() && COUNTING.load(Ordering::Relaxed) {
            BYTES_ALLOCATED.fetch_add(new_size, Ordering::Relaxed);
        }
        new_ptr
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

/// SmolLM2-360M hidden size, and vocab-sized lm_head width for the widest projection.
const IN: usize = 960;
const OUT: usize = 960;
const VOCAB: usize = 49_152;

fn tensor(data: Vec<f32>, dims: (usize, usize)) -> Tensor {
    Tensor::from_vec(data, dims, &Device::Cpu).expect("tensor")
}

#[test]
fn test_amx_matmul_no_weight_sized_copy_per_token() {
    // One decode step: activations [1, IN] against a transposed weight [IN, OUT].
    let x = tensor(vec![0.125f32; IN], (1, IN));

    let mut w_data = vec![0.0f32; OUT * IN];
    for (i, v) in w_data.iter_mut().enumerate() {
        *v = ((i as f32) * 0.017).sin() * 0.25;
    }
    let w = tensor(w_data, (OUT, IN));
    let w_t = w.t().expect("transposed weight view");

    // Reference computed before the measured window (candle CPU matmul).
    let reference = x.matmul(&w_t).expect("reference matmul");
    let reference = reference.flatten_all().unwrap().to_vec1::<f32>().unwrap();

    // Warm up: first call may lazily initialize Accelerate / cache internal buffers.
    let _ = amx_matmul(&x, &w_t).expect("warmup matmul");

    let weight_bytes = OUT * IN * std::mem::size_of::<f32>();

    BYTES_ALLOCATED.store(0, Ordering::SeqCst);
    COUNTING.store(true, Ordering::SeqCst);
    let out = amx_matmul(&x, &w_t).expect("amx matmul");
    COUNTING.store(false, Ordering::SeqCst);
    let allocated = BYTES_ALLOCATED.load(Ordering::SeqCst);

    let got = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(got.len(), OUT);
    for (idx, (g, r)) in got.iter().zip(reference.iter()).enumerate() {
        assert!(
            (g - r).abs() <= 1e-4 * r.abs().max(1.0),
            "element {idx}: amx={g} reference={r}"
        );
    }

    println!(
        "decode matmul [1x{IN}] x ({OUT}x{IN}).t(): allocated {allocated} bytes, \
         weight is {weight_bytes} bytes (output alone needs {OUT} * 4 bytes)"
    );
    assert!(
        allocated < weight_bytes,
        "amx_matmul allocated {allocated} bytes for a single decode step; that is a \
         weight-sized ({weight_bytes} bytes) copy of the [out, in] weight"
    );

    // Same invariant for the widest projection (lm_head, [vocab, hidden]): the budget
    // stays in the kilobyte range regardless of weight size.
    let logits_ref_len = VOCAB;
    let lm_head = tensor(vec![0.0625f32; VOCAB * IN], (VOCAB, IN));
    let lm_head_t = lm_head.t().unwrap();
    let logits_small = x.matmul(&lm_head_t).unwrap().flatten_all().unwrap();
    let first_ref = logits_small.to_vec1::<f32>().unwrap()[0];
    let _ = amx_matmul(&x, &lm_head_t).expect("warmup lm_head");

    BYTES_ALLOCATED.store(0, Ordering::SeqCst);
    COUNTING.store(true, Ordering::SeqCst);
    let logits = amx_matmul(&x, &lm_head_t).expect("amx lm_head matmul");
    COUNTING.store(false, Ordering::SeqCst);
    let allocated_head = BYTES_ALLOCATED.load(Ordering::SeqCst);

    let logits = logits.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(logits.len(), logits_ref_len);
    assert!(
        (logits[0] - first_ref).abs() <= 1e-4 * first_ref.abs().max(1.0),
        "lm_head element 0: amx={} reference={first_ref}",
        logits[0]
    );
    println!("lm_head matmul [1x{IN}] x ({VOCAB}x{IN}).t(): allocated {allocated_head} bytes");
    assert!(
        allocated_head < VOCAB * IN * std::mem::size_of::<f32>(),
        "amx_matmul allocated {allocated_head} bytes for lm_head; expected only the \
         {VOCAB}-element output"
    );
    assert!(
        allocated_head < VOCAB * std::mem::size_of::<f32>() * 4,
        "amx_matmul allocated {allocated_head} bytes for lm_head, more than 4x the \
         {VOCAB}-element output size"
    );
}

/// 3D prefill path (`[B, M, K] x [B, K, N]`) must also avoid weight-sized
/// copies, including the per-batch-transposed weight view. Also pins the
/// `AMX_FALLBACK_COPIES` counter at zero on the happy path.
#[test]
fn test_amx_matmul_3d_no_weight_sized_copy() {
    use kanso_infer::accelerate::amx::AMX_FALLBACK_COPIES;

    const B: usize = 2;
    const M: usize = 4;
    const K: usize = 64;
    const N: usize = 64;
    let a = Tensor::from_vec(vec![0.1f32; B * M * K], (B, M, K), &Device::Cpu).unwrap();
    let w = Tensor::from_vec(vec![0.05f32; B * N * K], (B, N, K), &Device::Cpu).unwrap();
    // Per-batch transpose view, matching `linear`'s `.t()` weight usage.
    let b = w.transpose(1, 2).unwrap();

    let reference = a.matmul(&b).expect("reference 3d matmul");
    let reference = reference.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let _ = amx_matmul(&a, &b).expect("warmup 3d matmul");

    let weight_bytes = B * K * N * std::mem::size_of::<f32>();
    let before = AMX_FALLBACK_COPIES.load(Ordering::SeqCst);
    BYTES_ALLOCATED.store(0, Ordering::SeqCst);
    COUNTING.store(true, Ordering::SeqCst);
    let out = amx_matmul(&a, &b).expect("amx 3d matmul");
    COUNTING.store(false, Ordering::SeqCst);
    let allocated = BYTES_ALLOCATED.load(Ordering::SeqCst);
    let after = AMX_FALLBACK_COPIES.load(Ordering::SeqCst);

    let got = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(got.len(), reference.len());
    for (idx, (g, r)) in got.iter().zip(reference.iter()).enumerate() {
        assert!(
            (g - r).abs() <= 1e-4 * r.abs().max(1.0),
            "element {idx}: amx={g} reference={r}"
        );
    }
    assert_eq!(
        after, before,
        "3D happy path must not take the get_f32_vec fallback copy"
    );
    println!(
        "3d matmul [{B}x{M}x{K}] x [{B}x{K}x{N}].t(): allocated {allocated} bytes, weight is {weight_bytes} bytes"
    );
    assert!(
        allocated < weight_bytes,
        "amx_matmul allocated {allocated} bytes for a 3D step; that is a weight-sized ({weight_bytes} bytes) copy"
    );
}
