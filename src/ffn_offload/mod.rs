//! FFN offload seam (ticket #54): per-layer SwiGLU adapters with an optional
//! Core ML / ANE execution target. No `.mlpackage` is compiled and no `MLModel`
//! runs here — both adapters execute the same CPU/AMX math, so this is a
//! correctly-named offload seam, not ANE acceleration. Wired into the forward
//! pass only when `Model::enable_ffn_offload` installs the pipeline.
//! Renamed from `ane` so the module name stops claiming hardware it never touches.

pub mod adapter;

pub use adapter::{
    AneLayerAdapter, CoreMlAneAdapter, CpuFallbackAdapter, CpuFallbackAneAdapter,
    FfnOffloadAdapter, HybridExecutionPipeline, HybridSyncStats, PlannedCoreMlAdapter,
    has_ane_support,
};
