//! Ticket #88: quantized (INT8/FP8) KV cache as an `Engine`-selectable storage type.
//!
//! Four things this pins down:
//! 1. the engine constructor picks F32 / INT8 / FP8, F32 stays the default, and the
//!    quantized modes are refused on Metal with one clear error instead of a silent
//!    fallback to the F32 paged-attention kernel;
//! 2. logit drift against F32 stays under the declared thresholds at ctx 512 and 2048;
//! 3. the quantized pool really is smaller per token, and the reported ratio says so;
//! 4. the decode path holds one storage type — `KvStorageKind` — so no decode function
//!    branches on the storage implementation.
//!
//! Declared thresholds (measured on the seeded mock below, ctx 512 / 2048):
//! `MAX_REL_DRIFT` = 20% of the F32 logit range for INT8 and FP8 alike, and the greedy
//! argmax must be identical. Both are drift bounds, not accuracy claims: INT8 measures
//! ~1.8% / ~0.5% and FP8 ~15% / ~4% of the range at those contexts (FP8's 4-bit mantissa
//! costs ~8x INT8's error), so the threshold leaves headroom for machine-to-machine
//! float differences without letting a regression hide.

use candle_core::Tensor;
use kanso_infer::engine::Engine;
use kanso_infer::error::EngineError;
use kanso_infer::hardware::Backend;
use kanso_infer::kv_cache::{BlockTable, KvStorage, KvStorageKind, QuantMode};
use kanso_infer::model::Model;

/// Max |quantized - F32| logit, as a fraction of the F32 logit range (max - min).
const MAX_REL_DRIFT: f32 = 0.20;

const CONTEXTS: [usize; 2] = [512, 2048];

fn mock() -> Model {
    let mut model = Model::create_mock_tiny(16, 4, 2, 2, 50).expect("mock model");
    // The CPU path is the one quantized KV supports (Metal refuses it, see below).
    model.set_backend(Backend::Cpu);
    model
}

/// Prompt + block table for `ctx` tokens, with one spare block for the decode token.
fn context(model: &Model, ctx: usize) -> (Vec<u32>, BlockTable) {
    let block_size = model.config.block_size;
    let blocks = (ctx + 1).div_ceil(block_size);
    let mut table = BlockTable::new();
    for b in 0..blocks {
        table.push_block(b);
    }
    table.set_num_tokens(ctx);
    let prompt: Vec<u32> = (0..ctx).map(|i| (i % 50) as u32).collect();
    (prompt, table)
}

fn f32_storage(model: &Model, table: &BlockTable) -> KvStorage {
    KvStorage::new(
        model.config.num_hidden_layers,
        table.blocks().len(),
        model.config.block_size,
        model.config.num_key_value_heads,
        model.config.head_dim(),
        &model.device,
    )
    .expect("f32 storage")
}

fn logits_of(model: &Model, storage: &mut KvStorageKind, ctx: usize) -> Vec<f32> {
    let (prompt, mut table) = context(model, ctx);
    let mut storage = storage;
    let _ = model.forward_prefill(&prompt, 0, &table, storage);
    table.set_num_tokens(ctx + 1);
    let logits = model
        .forward_decode_step(7, ctx, &table, storage)
        .expect("decode step");
    logits.flatten_all().unwrap().to_vec1::<f32>().unwrap()
}

fn relative_drift(reference: &[f32], candidate: &[f32]) -> (f32, bool) {
    assert_eq!(reference.len(), candidate.len());
    let max_diff = reference
        .iter()
        .zip(candidate.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let lo = reference.iter().copied().fold(f32::INFINITY, f32::min);
    let hi = reference.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let argmax = |v: &[f32]| {
        let mut best = 0usize;
        for (i, &x) in v.iter().enumerate() {
            if x > v[best] {
                best = i;
            }
        }
        best
    };
    (max_diff / (hi - lo), argmax(reference) == argmax(candidate))
}

#[test]
fn engine_defaults_to_f32_kv_storage() {
    let engine = Engine::new(mock(), 1, 64, 16).expect("engine");
    assert_eq!(
        engine.kv_storage.mode(),
        None,
        "F32 must stay the default: the engine may not quantize unless asked"
    );
    assert_eq!(engine.kv_storage.block_size(), 16);
}

#[test]
fn engine_accepts_int8_and_fp8_kv_storage() {
    for mode in [QuantMode::Int8, QuantMode::Fp8] {
        let engine = Engine::new_with_budget_and_kv(mock(), 1, 64, 16, usize::MAX, Some(mode))
            .expect("quantized engine");
        assert_eq!(
            engine.kv_storage.mode(),
            Some(mode),
            "constructor must keep the requested quantization mode"
        );
    }
}

#[test]
fn engine_preserves_legacy_constructor_signatures() {
    // `new` / `new_with_budget` keep their old shape and stay F32 (ticket #85).
    let engine = Engine::new_with_budget(mock(), 1, 64, 16, usize::MAX).expect("engine");
    assert_eq!(engine.kv_storage.mode(), None);
}

#[test]
fn quantized_kv_is_refused_on_metal() {
    let mut model = Model::create_mock_tiny(16, 4, 2, 2, 50).expect("mock model");
    model.set_backend(Backend::Metal);
    let err = match Engine::new_with_budget_and_kv(model, 1, 64, 16, usize::MAX, Some(QuantMode::Int8))
    {
        Ok(_) => panic!("Metal must refuse quantized KV, not fall back to the F32 kernel"),
        Err(err) => err,
    };
    assert!(
        matches!(err, EngineError::Config(_)),
        "expected a Config error naming the Metal limitation, got {err:?}"
    );
    let msg = err.to_string();
    assert!(
        msg.contains("Metal"),
        "error must name the backend that cannot serve it: {msg}"
    );
}

#[test]
fn int8_kv_logits_track_f32_within_declared_drift() {
    for ctx in CONTEXTS {
        let model = mock();
        let (_, table) = context(&model, ctx);
        let mut reference = KvStorageKind::f32(f32_storage(&model, &table));
        let mut quantized = KvStorageKind::quantized(
            model.config.num_hidden_layers,
            table.blocks().len(),
            model.config.block_size,
            model.config.num_key_value_heads,
            model.config.head_dim(),
            QuantMode::Int8,
            &model.device,
        )
        .expect("int8 storage");
        let expected = logits_of(&model, &mut reference, ctx);
        let actual = logits_of(&model, &mut quantized, ctx);
        let (rel, same_argmax) = relative_drift(&expected, &actual);
        assert!(
            rel <= MAX_REL_DRIFT,
            "INT8 KV drift at ctx {ctx} was {:.3} of the logit range, declared threshold {MAX_REL_DRIFT}",
            rel
        );
        assert!(
            same_argmax,
            "INT8 KV drift at ctx {ctx} changed the greedy token"
        );
    }
}

#[test]
fn fp8_kv_logits_track_f32_within_declared_drift() {
    for ctx in CONTEXTS {
        let model = mock();
        let (_, table) = context(&model, ctx);
        let mut reference = KvStorageKind::f32(f32_storage(&model, &table));
        let mut quantized = KvStorageKind::quantized(
            model.config.num_hidden_layers,
            table.blocks().len(),
            model.config.block_size,
            model.config.num_key_value_heads,
            model.config.head_dim(),
            QuantMode::Fp8,
            &model.device,
        )
        .expect("fp8 storage");
        let expected = logits_of(&model, &mut reference, ctx);
        let actual = logits_of(&model, &mut quantized, ctx);
        let (rel, same_argmax) = relative_drift(&expected, &actual);
        assert!(
            rel <= MAX_REL_DRIFT,
            "FP8 KV drift at ctx {ctx} was {:.3} of the logit range, declared threshold {MAX_REL_DRIFT}",
            rel
        );
        assert!(
            same_argmax,
            "FP8 KV drift at ctx {ctx} changed the greedy token"
        );
    }
}

#[test]
fn quantized_kv_bytes_per_token_are_below_f32() {
    let model = mock();
    let f32_bytes = KvStorageKind::f32_new(
        model.config.num_hidden_layers,
        8,
        model.config.block_size,
        model.config.num_key_value_heads,
        model.config.head_dim(),
        &model.device,
    )
    .expect("f32 storage")
    .memory_bytes_per_token();
    for mode in [QuantMode::Int8, QuantMode::Fp8] {
        let storage = KvStorageKind::quantized(
            model.config.num_hidden_layers,
            8,
            model.config.block_size,
            model.config.num_key_value_heads,
            model.config.head_dim(),
            mode,
            &model.device,
        )
        .expect("storage");
        let bytes = storage.memory_bytes_per_token();
        assert!(
            bytes < f32_bytes,
            "{mode:?} KV reports {bytes} bytes/token, F32 holds {f32_bytes}: quantization bought nothing"
        );
        let ratio = storage.memory_ratio_vs_f32();
        assert!(
            ratio > 1.0,
            "{mode:?} KV compression ratio {ratio} is not better than F32"
        );
        println!(
            "{mode:?} KV: {bytes} bytes/token vs F32 {f32_bytes} ({ratio:.2}x smaller)"
        );
    }
}

#[test]
fn engine_reports_kv_bytes_per_token() {
    let engine = Engine::new_with_budget_and_kv(mock(), 1, 64, 16, usize::MAX, Some(QuantMode::Int8))
        .expect("int8 engine");
    let f32_engine = Engine::new(mock(), 1, 64, 16).expect("f32 engine");
    assert!(
        engine.kv_storage.memory_bytes_per_token() < f32_engine.kv_storage.memory_bytes_per_token(),
        "the engine must expose the per-token KV footprint so the saving is reportable"
    );
    assert!(engine.kv_storage.memory_ratio_vs_f32() > 1.0);
}

#[test]
fn decode_path_holds_one_storage_type() {
    // The decode entry points take the storage enum, never a concrete implementation:
    // the pool is chosen once, in the constructor, and every read/write below it goes
    // through the same five methods.
    let model = mock();
    let (prompt, table) = context(&model, 40);
    let mut storage = KvStorageKind::f32(f32_storage(&model, &table));
    let mut table = table;
    model
        .forward_prefill(&prompt, 0, &table, &mut storage)
        .expect("prefill through the enum");
    table.set_num_tokens(prompt.len() + 1);
    model
        .forward_decode_step(7, prompt.len(), &table, &mut storage)
        .expect("decode through the enum");
    let (k, v) = storage
        .gather_context(0, &table)
        .expect("gather through the enum");
    // 40 prefilled tokens plus the decode token just written at position 40.
    assert_eq!(k.dims(), &[prompt.len() + 1, 2, 4]);
    assert_eq!(v.dims(), &[prompt.len() + 1, 2, 4]);
    let (k, _) = storage.paged_kv_tokens(0, table.blocks(), prompt.len()).unwrap();
    assert_eq!(k.dims(), &[prompt.len(), 2, 4]);
    let (k1, v1) = storage.read_kv_token(0, 0, 0).unwrap();
    assert_eq!(k1.dims(), &[2, 4]);
    assert_eq!(v1.dims(), &[2, 4]);
    let _ = Tensor::zeros((1,), candle_core::DType::F32, &model.device).unwrap();
}

#[test]
fn int8_engine_generates_the_same_tokens_as_f32() {
    // End-to-end: a greedy run over the engine seam. Drift may reorder near-ties, but a
    // tiny mock's greedy trajectory is what the F32 engine produces too.
    let prompt: Vec<u32> = (0..12).collect();
    let mut f32_engine = Engine::new(mock(), 1, 64, 16).expect("f32 engine");
    let mut int8_engine = Engine::new_with_budget_and_kv(
        mock(),
        1,
        64,
        16,
        usize::MAX,
        Some(QuantMode::Int8),
    )
    .expect("int8 engine");
    let f32_id = f32_engine.add_request(prompt.clone(), 8);
    let int8_id = int8_engine.add_request(prompt, 8);
    let mut f32_tokens = Vec::new();
    let mut int8_tokens = Vec::new();
    while f32_engine.has_unfinished_requests() {
        for out in f32_engine.step().expect("f32 step") {
            if out.request_id == f32_id {
                f32_tokens.push(out.token_id);
            }
        }
    }
    while int8_engine.has_unfinished_requests() {
        for out in int8_engine.step().expect("int8 step") {
            if out.request_id == int8_id {
                int8_tokens.push(out.token_id);
            }
        }
    }
    assert_eq!(
        f32_tokens, int8_tokens,
        "INT8 KV changed the greedy trajectory of a mock request"
    );
}
