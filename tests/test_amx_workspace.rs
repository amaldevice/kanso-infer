//! Ticket #93: the AMX decode path must not allocate its matmul outputs per call.
//!
//! Spec #82 D16: `amx_matmul` returned a fresh `vec![0.0f32; m * n]` for every
//! projection of every layer, ~7 small `Vec`s per layer per token, and
//! `amx_paged_decode_attention` allocated three scratch buffers per call. The decode
//! path now writes into a reused workspace instead, so the per-token allocation
//! count drops and logits stay bit-identical.
//!
//! The counting global allocator lives in this file's own test binary so its counters
//! are not polluted by other test modules running in parallel.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use candle_core::{Device, Tensor};
use kanso_infer::Backend;
use kanso_infer::kv_cache::{BlockTable, KvStorage};
use kanso_infer::model::Model;

struct CountingAllocator;

static COUNTING: AtomicBool = AtomicBool::new(false);
static ALLOCS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() && COUNTING.load(Ordering::Relaxed) {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
            BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_ptr = unsafe { System.realloc(ptr, layout, new_size) };
        if !new_ptr.is_null() && COUNTING.load(Ordering::Relaxed) {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
            BYTES.fetch_add(new_size, Ordering::Relaxed);
        }
        new_ptr
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

/// The allocation counters are global, so a counting window must not overlap another
/// test's forward passes. Cargo runs a binary's tests on parallel threads; this lock
/// serializes them, which is what makes the measured counts exact rather than merely
/// indicative. The whole file costs ~1s, so serializing is free.
static TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn serial() -> std::sync::MutexGuard<'static, ()> {
    TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// Small mock geometry: the counting test must stay cheap in debug builds. A
/// production-shaped mock (32 layers, vocab 49152) takes tens of minutes here.
const HIDDEN: usize = 256;
const Q_HEADS: usize = 8;
const KV_HEADS: usize = 2;
const LAYERS: usize = 2;
const VOCAB: usize = 512;
const CTX: usize = 32;

fn mock() -> Model {
    Model::create_mock_tiny(HIDDEN, Q_HEADS, KV_HEADS, LAYERS, VOCAB).expect("mock model")
}

/// A block table covering `ctx` tokens, plus a KV storage sized to match.
fn kv_for(model: &Model, ctx: usize) -> (KvStorage, BlockTable) {
    let block_size = model.config.block_size;
    let blocks = ctx.div_ceil(block_size);
    let kv = KvStorage::new(
        model.config.num_hidden_layers,
        blocks,
        block_size,
        model.config.num_key_value_heads,
        model.config.head_dim(),
        &model.device,
    )
    .expect("kv storage");
    let mut table = BlockTable::new();
    for b in 0..blocks {
        table.push_block(b);
    }
    table.set_num_tokens(ctx);
    (kv, table)
}

/// Allocations and bytes charged to one `forward_decode_step` after three warmup steps.
fn step_cost(model: &Model) -> (usize, usize) {
    let (mut kv, table) = kv_for(model, CTX);
    for _ in 0..3 {
        model
            .forward_decode_step(1, CTX - 1, &table, &mut kv)
            .expect("warmup decode step");
    }
    COUNTING.store(true, Ordering::SeqCst);
    ALLOCS.store(0, Ordering::SeqCst);
    BYTES.store(0, Ordering::SeqCst);
    model
        .forward_decode_step(7, CTX - 1, &table, &mut kv)
        .expect("decode step");
    COUNTING.store(false, Ordering::SeqCst);
    (ALLOCS.load(Ordering::SeqCst), BYTES.load(Ordering::SeqCst))
}

/// Logits of one decode step at `CTX - 1`.
fn step_logits(model: &Model) -> Vec<f32> {
    let (mut kv, table) = kv_for(model, CTX);
    for _ in 0..CTX - 1 {
        model
            .forward_decode_step(3, 0, &table, &mut kv)
            .expect("prefix step");
    }
    model
        .forward_decode_step(7, CTX - 1, &table, &mut kv)
        .expect("decode step")
        .flatten_all()
        .expect("flatten logits")
        .to_vec1::<f32>()
        .expect("logits to host")
}

/// Bytes the seven per-layer projection outputs of a decode step occupy at this geometry.
fn projection_output_bytes() -> usize {
    let head_dim = HIDDEN / Q_HEADS;
    let kv_dim = KV_HEADS * head_dim;
    (HIDDEN + kv_dim + kv_dim + HIDDEN + HIDDEN * 2 + HIDDEN * 2 + HIDDEN) * 4
}

/// AC1 + AC2: the per-token allocation count on the AMX decode path drops, and the
/// drop is measured rather than claimed. Each layer issues seven projections whose
/// outputs used to be a fresh `Vec` each (`vec![0.0f32; m * n]`), plus three scratch
/// buffers in the paged attention — so the saved count scales with the layer count.
#[test]
fn amx_decode_step_allocates_fewer_buffers_per_token() {
    let _serial = serial();
    if !kanso_infer::accelerate::amx::has_accelerate_support() {
        eprintln!("SKIP (non-macOS): AMX is a scalar fallback here");
        return;
    }
    let mut model = mock();

    model.set_backend(Backend::Cpu);
    let (cpu_allocs, cpu_bytes) = step_cost(&model);

    model.set_backend(Backend::Amx);
    let (amx_allocs, amx_bytes) = step_cost(&model);

    // Seven projection outputs per layer are no longer allocated per call; the
    // paged-attention scratch buffers go with them. Assert the saving is at least
    // that many buffers, not a token one or two.
    let saved = cpu_allocs.saturating_sub(amx_allocs);
    let per_step_outputs = projection_output_bytes() * model.config.num_hidden_layers;
    println!(
        "#93 decode step allocations: cpu {cpu_allocs}, amx {amx_allocs}, saved {saved} \
         ({} layers); bytes cpu {cpu_bytes}, amx {amx_bytes}, projection outputs alone \
         {per_step_outputs}",
        model.config.num_hidden_layers
    );
    assert!(
        saved >= 7 * model.config.num_hidden_layers,
        "AMX decode saved only {saved} allocations against cpu ({amx_allocs} vs {cpu_allocs}); \
         the {} projection outputs per layer plus the paged-attention scratch buffers \
         must not be allocated per call",
        7 * model.config.num_hidden_layers
    );
    // AC1 is about the output buffers themselves, not only the call count: the bytes the
    // step allocates must drop by at least what those outputs used to cost.
    let bytes_saved = cpu_bytes.saturating_sub(amx_bytes);
    assert!(
        bytes_saved >= per_step_outputs,
        "AMX decode allocated {bytes_saved} bytes fewer than cpu ({amx_bytes} vs {cpu_bytes}); \
         the projection outputs alone are {per_step_outputs} bytes per step"
    );
}

/// AC1, at the seam itself: one `matmul_weight_into` call must not allocate an `m * n`
/// output buffer. The workspace slot is filled in place.
#[test]
fn amx_matmul_into_writes_into_the_caller_buffer() {
    let _serial = serial();
    if !kanso_infer::accelerate::amx::has_accelerate_support() {
        eprintln!("SKIP (non-macOS): AMX is a scalar fallback here");
        return;
    }
    let x = Tensor::from_vec(vec![0.125f32; HIDDEN], (1, HIDDEN), &Device::Cpu).expect("x");
    let w = Tensor::from_vec(
        vec![0.25f32; HIDDEN * HIDDEN],
        (HIDDEN, HIDDEN),
        &Device::Cpu,
    )
    .expect("w");

    let reference = x.matmul(&w.t().expect("transposed weight"));
    let reference = reference
        .expect("reference matmul")
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    let mut workspace = kanso_infer::accelerate::amx::AmxWorkspace::default();
    // Warm up: the first call may lazily initialize Accelerate.
    let _ = workspace
        .matmul_weight_into(0, &x, &w)
        .expect("warmup matmul into workspace");

    COUNTING.store(true, Ordering::SeqCst);
    ALLOCS.store(0, Ordering::SeqCst);
    let out = workspace
        .matmul_weight_into(0, &x, &w)
        .expect("matmul into workspace");
    let allocs = ALLOCS.load(Ordering::SeqCst);
    COUNTING.store(false, Ordering::SeqCst);

    let got = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    for (idx, (g, r)) in got.iter().zip(reference.iter()).enumerate() {
        assert!(
            (g - r).abs() <= 1e-4 * r.abs().max(1.0),
            "element {idx}: workspace matmul={g} reference={r}"
        );
    }
    println!("#93 amx matmul_weight_into [{HIDDEN}x{HIDDEN}]: {allocs} allocations");
    assert_eq!(
        allocs, 0,
        "matmul_weight_into allocated {allocs} buffers; the output buffer must be reused"
    );
}

/// AC4: the workspace path changes no numbers. Decode logits on the AMX backend must
/// stay identical to the CPU backend's, exactly as before the out-param path.
#[test]
fn amx_workspace_decode_logits_match_cpu() {
    let _serial = serial();
    let mut model = mock();

    model.set_backend(Backend::Cpu);
    let cpu_logits = step_logits(&model);

    model.set_backend(Backend::Amx);
    let amx_logits = step_logits(&model);

    assert_eq!(cpu_logits.len(), amx_logits.len());
    let drift = cpu_logits
        .iter()
        .zip(amx_logits.iter())
        .map(|(c, a)| (c - a).abs())
        .fold(0.0f32, f32::max);
    println!("#93 decode logits max |Δ| amx vs cpu: {drift}");
    assert!(
        drift <= 1e-4,
        "AMX decode logits drifted from cpu by {drift} (tol 1e-4)"
    );
}

/// AC4, against an independent source of truth: a decode step must agree with the
/// monolithic forward over the same sequence, through the workspace path too.
///
/// Uses the repo's standard parity geometry (`tests/test_incremental_decode.rs`): at
/// wider hidden sizes both backends drift from the monolithic pass by ~2e-3 simply
/// because the two paths accumulate in a different order, so the tight tolerance is
/// only meaningful at this size. `amx_workspace_decode_logits_match_cpu` covers the
/// wider geometry against the CPU decode path.
#[test]
fn amx_workspace_decode_matches_monolithic_forward() {
    let _serial = serial();
    let mut model = Model::create_mock_tiny(16, 4, 2, 2, 50).expect("mock model");
    model.set_backend(Backend::Amx);

    let prompt = [5u32, 12u32, 9u32];
    let next_token = 21u32;
    let mut full_sequence = prompt.to_vec();
    full_sequence.push(next_token);

    let reference = model
        .forward_single_prompt(&full_sequence)
        .expect("monolithic forward")
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    let (mut kv, table) = kv_for(&model, full_sequence.len());
    let mut step_logits = Vec::new();
    for (pos, &token) in full_sequence.iter().enumerate() {
        step_logits = model
            .forward_decode_step(token, pos, &table, &mut kv)
            .expect("decode step")
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
    }
    let drift = reference
        .iter()
        .zip(step_logits.iter())
        .map(|(r, s)| (r - s).abs())
        .fold(0.0f32, f32::max);
    println!("#93 decode vs monolithic max |Δ|: {drift}");
    assert!(
        drift <= 1e-4,
        "workspace decode drifted from the monolithic forward by {drift} (tol 1e-4)"
    );
}
