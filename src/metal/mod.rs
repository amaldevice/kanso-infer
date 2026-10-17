pub mod context;
// Ticket-level allow (review F9): every `fused_layer` entry point binds one Metal
// kernel surface per dispatch, so 7-11 params each is the API, not a smell.
#[allow(clippy::too_many_arguments)]
pub mod fused_layer;
pub mod ops;
pub mod paged_attention;
pub mod pipeline;
pub mod shaders;

pub use context::MetalContext;
pub use fused_layer::*;
pub use paged_attention::PagedAttentionParams;
pub use pipeline::MetalPipelineCache;
