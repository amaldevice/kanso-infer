//! Feature registry (ticket #83): the in-crate source of truth for the README
//! "Feature Status" table.
//!
//! Every public feature claim is one [`FeatureSpec`]: a name, a [`FeatureStatus`],
//! the module that executes it, and — for `Wired` rows — a `wired` marker that
//! names the production symbol. The markers are `fn` pointers initialised inside a
//! `const`, so they are *compile-checked*: deleting the symbol a marker names (or
//! renaming it, or making it private) breaks the build instead of silently
//! invalidating a ✅. `tests/test_feature_registry.rs` keeps the registry and the
//! README table in lockstep, in both directions.
//!
//! What a marker proves is that the symbol **exists and is public**. It does not
//! prove the symbol is *reachable* from `kanso cli` / `chat` / `serve`; that is what
//! the behavioural tests at the engine/server seam are for. This registry stops a
//! label from outliving the code it describes — nothing more, nothing less.

/// Status of a public feature claim, mirroring the README glyphs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureStatus {
    /// ✅ Reachable from `kanso cli` / `chat` / `serve`; must name a production marker.
    Wired,
    /// 🧪 Implemented and tested, but no production caller yet; must not claim a marker.
    KernelOnly,
    /// 📝 Roadmap, not code; must point at the roadmap and claim no marker.
    Planned,
}

/// One row of the public Feature Status table.
#[derive(Debug)]
pub struct FeatureSpec {
    /// Row label, byte-identical to the README `Feature` cell.
    pub name: &'static str,
    /// ✅ / 🧪 / 📝 as declared in the README.
    pub status: FeatureStatus,
    /// Space-separated module paths that execute the feature, or `roadmap` for
    /// planned rows. The README row must reference every path listed here.
    pub exec_pointer: &'static str,
    /// Compile-checked production marker; `Some` exactly when `status` is `Wired`.
    pub wired: Option<fn()>,
}

/// A marker that fails the build when its production symbol disappears.
macro_rules! wired {
    ($($path:path),+ $(,)?) => {
        Some(|| {
            $(
                let _ = $path;
            )+
        })
    };
}

/// Every public feature claim, one entry per README Feature Status row.
pub const FEATURES: &[FeatureSpec] = &[
    FeatureSpec {
        name: "CLI / chat / serve (Axum + SSE, 429 backpressure)",
        status: FeatureStatus::Wired,
        exec_pointer: "src/main.rs src/server/engine_loop.rs",
        wired: wired!(
            crate::server::routes::create_app,
            crate::chat::ChatSession::step_reply
        ),
    },
    FeatureSpec {
        name: "Incremental $O(1)$ decode over paged KV blocks",
        status: FeatureStatus::Wired,
        exec_pointer: "src/model.rs src/kv_cache/storage.rs",
        wired: wired!(
            crate::model::Model::forward_decode_step,
            crate::kv_cache::KvStorage::gather_context
        ),
    },
    FeatureSpec {
        name: "Continuous batching + Sarathi chunked prefill",
        status: FeatureStatus::Wired,
        exec_pointer: "src/engine.rs src/scheduler/",
        wired: wired!(crate::engine::Engine::step, crate::engine::Engine::new),
    },
    FeatureSpec {
        name: "Prefix-cache block reuse",
        status: FeatureStatus::Wired,
        exec_pointer: "src/engine.rs src/kv_cache/prefix_cache.rs",
        wired: wired!(
            crate::engine::Engine::step,
            crate::kv_cache::PrefixCache::find_longest_prefix
        ),
    },
    FeatureSpec {
        name: "Quantized-resident weights (Q8_0/Q4_K/F16 stay packed; per-row embedding dequant)",
        status: FeatureStatus::Wired,
        exec_pointer: "src/weights.rs tests/test_quantized_weights.rs",
        wired: wired!(
            crate::weights::ModelWeights::load_from_gguf,
            crate::weights::LinearWeight::embed
        ),
    },
    FeatureSpec {
        name: "Granite dense multipliers (`embedding_scale`, `attention.scale`, `residual_scale`, `logit_scale`)",
        status: FeatureStatus::Wired,
        exec_pointer: "src/weights.rs src/model.rs src/config.rs",
        wired: wired!(
            crate::weights::ModelWeights::load_from_gguf,
            crate::model::Model::forward_tokens
        ),
    },
    FeatureSpec {
        name: "Loader arch gate (rejects non-llama/non-granite GGUF)",
        status: FeatureStatus::Wired,
        exec_pointer: "src/weights.rs",
        wired: wired!(crate::weights::ModelWeights::load_from_gguf),
    },
    FeatureSpec {
        name: "Speculative verification (argmax + `min(1,p/q)` rejection sampling, draft KV rollback)",
        status: FeatureStatus::Wired,
        exec_pointer: "src/engine.rs src/speculative/verifier.rs",
        wired: wired!(
            crate::engine::Engine::speculative_step_sampled,
            crate::speculative::verifier::SpeculativeVerifier::verify
        ),
    },
    FeatureSpec {
        name: "Metal UMA buffers + startup pipeline cache",
        status: FeatureStatus::Wired,
        exec_pointer: "src/metal/context.rs src/metal/pipeline.rs",
        wired: wired!(
            crate::metal::context::MetalContext::new,
            crate::metal::pipeline::MetalPipelineCache::new
        ),
    },
    FeatureSpec {
        name: "AMX/Accelerate GEMM path (dense weights)",
        status: FeatureStatus::Wired,
        exec_pointer: "src/accelerate/amx.rs src/model.rs",
        wired: wired!(
            crate::accelerate::amx::amx_gemm,
            crate::model::Model::linear
        ),
    },
    FeatureSpec {
        name: "FFN offload seam (per-layer SwiGLU adapters, CPU/AMX execution)",
        status: FeatureStatus::Wired,
        exec_pointer: "src/ffn_offload/adapter.rs src/model.rs",
        wired: wired!(
            crate::ffn_offload::adapter::HybridExecutionPipeline::forward_layer_mlp,
            crate::model::Model::forward_mlp_block
        ),
    },
    FeatureSpec {
        name: "Real Core ML / ANE execution (`.mlpackage`, `MLModel`)",
        status: FeatureStatus::Planned,
        exec_pointer: "roadmap",
        wired: None,
    },
    FeatureSpec {
        name: "Fused PagedAttention v2 Metal kernel",
        status: FeatureStatus::Wired,
        exec_pointer: "src/metal/paged_attention.rs src/model/paged_attn.rs src/metal/kernels/paged_attention_v2.metal",
        wired: wired!(crate::model::Model::metal_paged_attention),
    },
    FeatureSpec {
        name: "Fused Metal llama-layer kernel",
        status: FeatureStatus::KernelOnly,
        exec_pointer: "src/metal/fused_layer.rs",
        wired: None,
    },
    FeatureSpec {
        name: "Hand-written NEON Q8_0/Q4_K GEMV kernels",
        status: FeatureStatus::KernelOnly,
        exec_pointer: "src/quant/neon.rs",
        wired: None,
    },
    FeatureSpec {
        name: "Quantized (INT8/FP8) paged KV cache",
        status: FeatureStatus::KernelOnly,
        exec_pointer: "src/kv_cache/quantized_storage.rs",
        wired: None,
    },
    FeatureSpec {
        name: "Hybrid SSM/conv layers (Mamba-2, LIV conv) + sparse KV allocation",
        status: FeatureStatus::Planned,
        exec_pointer: "roadmap",
        wired: None,
    },
    FeatureSpec {
        name: "`granite-h` / `lfm` bench profiles",
        status: FeatureStatus::Planned,
        exec_pointer: "roadmap",
        wired: None,
    },
];
