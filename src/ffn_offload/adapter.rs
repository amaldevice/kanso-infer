use crate::error::EngineError;
use crate::weights::{LayerWeights, LinearWeight};
use candle_core::{Device, Tensor};
use parking_lot::Mutex;
use std::path::PathBuf;

/// Checks whether Apple Neural Engine (ANE) hardware is present on current platform.
///
/// Always false: there is no Core ML execution path yet (`CoreMlAneAdapter`
/// runs the CPU/AMX fallback unconditionally), so claiming ANE on the basis
/// of a chip-name string match would lie on every Mac. Flip this only when a
/// real dispatch path lands behind it.
pub fn has_ane_support() -> bool {
    false
}

/// Architectural adapter trait for dispatching feedforward (SwiGLU/FFN) layer computation
/// to an offload backend or platform fallback.
pub trait FfnOffloadAdapter: Send + Sync {
    /// Forward pass for MLP / SwiGLU computation on activations.
    /// Input is a 1D slice representing [seq_len * hidden_size] or [hidden_size].
    /// Returns a 1D vector of output activations of the same size.
    fn forward_mlp(&self, input: &[f32]) -> Result<Vec<f32>, EngineError>;

    /// Indicates whether the adapter's primary acceleration hardware is available.
    fn is_available(&self) -> bool {
        true
    }

    /// Name of the execution backend.
    fn backend_name(&self) -> &str;

    /// Dispatches or updates layer weights for offloaded execution.
    fn dispatch_weights(&mut self, _weights: &LayerWeights) -> Result<(), EngineError> {
        Ok(())
    }

    /// Layer index corresponding to this adapter.
    fn layer_index(&self) -> usize {
        0
    }
}

pub use FfnOffloadAdapter as AneLayerAdapter;

/// Fallback adapter executing SwiGLU FFN on CPU or AMX (Accelerate) cores.
#[derive(Clone)]
pub struct CpuFallbackAdapter {
    pub layer_index: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub ffn_gate: LinearWeight,
    pub ffn_up: LinearWeight,
    pub ffn_down: LinearWeight,
    pub backend: crate::hardware::Backend,
    pub device: Device,
}

pub use CpuFallbackAdapter as CpuFallbackAneAdapter;

impl CpuFallbackAdapter {
    #[allow(clippy::too_many_arguments)] // #57: public API; regrouping params out of scope
    pub fn new(
        layer_index: usize,
        hidden_size: usize,
        intermediate_size: usize,
        ffn_gate: LinearWeight,
        ffn_up: LinearWeight,
        ffn_down: LinearWeight,
        backend: crate::hardware::Backend,
        device: Device,
    ) -> Self {
        Self {
            layer_index,
            hidden_size,
            intermediate_size,
            ffn_gate,
            ffn_up,
            ffn_down,
            backend,
            device,
        }
    }

    pub fn from_layer_weights(
        layer_index: usize,
        weights: &LayerWeights,
        hidden_size: usize,
        intermediate_size: usize,
        backend: crate::hardware::Backend,
    ) -> Self {
        Self::new(
            layer_index,
            hidden_size,
            intermediate_size,
            weights.ffn_gate.clone(),
            weights.ffn_up.clone(),
            weights.ffn_down.clone(),
            backend,
            weights.ffn_gate.device(),
        )
    }

    pub fn set_backend(&mut self, backend: crate::hardware::Backend) {
        self.backend = backend;
    }
}

impl FfnOffloadAdapter for CpuFallbackAdapter {
    fn forward_mlp(&self, input: &[f32]) -> Result<Vec<f32>, EngineError> {
        if input.is_empty() {
            return Err(EngineError::Model("Empty input to forward_mlp".to_string()));
        }
        if !input.len().is_multiple_of(self.hidden_size) {
            return Err(EngineError::Model(format!(
                "Input length {} is not a multiple of hidden_size {}",
                input.len(),
                self.hidden_size
            )));
        }

        let seq_len = input.len() / self.hidden_size;
        let x = Tensor::from_slice(input, (seq_len, self.hidden_size), &self.device)?;

        let amx = matches!(self.backend, crate::hardware::Backend::Amx)
            && crate::accelerate::amx::has_accelerate_support();
        // AMX GEMM only takes dense F32; quantized weights keep their own kernels.
        let project = |w: &LinearWeight, x: &Tensor| -> Result<Tensor, EngineError> {
            match w.as_dense_f32() {
                Some(dense) if amx => crate::accelerate::amx::amx_matmul(x, &dense.t()?),
                _ => w.forward(x),
            }
        };

        let gate = project(&self.ffn_gate, &x)?;
        let up = project(&self.ffn_up, &x)?;

        let silu_gate = candle_nn::ops::silu(&gate)?;
        let ffn_hidden = (silu_gate * up)?;

        let out = project(&self.ffn_down, &ffn_hidden)?;

        let flat = out.flatten_all()?.to_vec1::<f32>()?;
        Ok(flat)
    }

    fn is_available(&self) -> bool {
        true
    }

    fn backend_name(&self) -> &str {
        if matches!(self.backend, crate::hardware::Backend::Amx) {
            "amx_cpu"
        } else {
            "cpu"
        }
    }

    fn dispatch_weights(&mut self, weights: &LayerWeights) -> Result<(), EngineError> {
        self.ffn_gate = weights.ffn_gate.clone();
        self.ffn_up = weights.ffn_up.clone();
        self.ffn_down = weights.ffn_down.clone();
        Ok(())
    }

    fn layer_index(&self) -> usize {
        self.layer_index
    }
}

/// Planned Core ML adapter dispatching SwiGLU layer weights to Apple Neural Engine (ANE)
/// once `.mlpackage` compilation lands. Until then, unconditionally executes the CPU/AMX
/// fallback and honestly reports the fallback backend name.
#[derive(Clone)]
pub struct PlannedCoreMlAdapter {
    pub layer_index: usize,
    pub is_ane_available: bool,
    pub fallback: CpuFallbackAdapter,
    pub compiled_model_path: Option<PathBuf>,
}

pub use PlannedCoreMlAdapter as CoreMlAneAdapter;

impl PlannedCoreMlAdapter {
    pub fn new(
        layer_index: usize,
        hidden_size: usize,
        intermediate_size: usize,
        weights: &LayerWeights,
        backend_fallback: crate::hardware::Backend,
    ) -> Self {
        let ane_supported = has_ane_support();
        let fallback = CpuFallbackAdapter::from_layer_weights(
            layer_index,
            weights,
            hidden_size,
            intermediate_size,
            backend_fallback,
        );
        Self {
            layer_index,
            is_ane_available: ane_supported,
            fallback,
            compiled_model_path: None,
        }
    }

    pub fn set_compiled_model_path(&mut self, path: impl Into<PathBuf>) {
        self.compiled_model_path = Some(path.into());
    }
}

impl FfnOffloadAdapter for PlannedCoreMlAdapter {
    fn forward_mlp(&self, input: &[f32]) -> Result<Vec<f32>, EngineError> {
        // Honest dispatch (#62 F1): until real Core ML .mlpackage execution lands,
        // unconditionally execute through the fallback adapter.
        self.fallback.forward_mlp(input)
    }

    fn is_available(&self) -> bool {
        self.is_ane_available
    }

    fn backend_name(&self) -> &str {
        // Honest backend name (#62 F1): never report "coreml_ane" while executing CPU math.
        self.fallback.backend_name()
    }

    fn dispatch_weights(&mut self, weights: &LayerWeights) -> Result<(), EngineError> {
        self.fallback.dispatch_weights(weights)
    }

    fn layer_index(&self) -> usize {
        self.layer_index
    }
}
/// Statistics tracking hybrid layer scheduling and synchronization events.
#[derive(Debug, Clone, Default)]
pub struct HybridSyncStats {
    pub attention_dispatches: usize,
    pub adapter_dispatches: usize,
    pub fallback_dispatches: usize,
}

/// Hybrid scheduling pipeline executing attention on Metal/AMX while delegating
/// MLP feedforward computation to the Core ML / ANE adapter.
pub struct HybridExecutionPipeline {
    pub attention_backend: crate::hardware::Backend,
    pub adapters: Vec<Box<dyn FfnOffloadAdapter>>,
    stats: Mutex<HybridSyncStats>,
}

impl HybridExecutionPipeline {
    pub fn new(
        attention_backend: crate::hardware::Backend,
        adapters: Vec<Box<dyn FfnOffloadAdapter>>,
    ) -> Self {
        Self {
            attention_backend,
            adapters,
            stats: Mutex::new(HybridSyncStats::default()),
        }
    }

    /// Construct a hybrid execution pipeline from an existing Model.
    pub fn from_model(model: &crate::model::Model) -> Self {
        let attention_backend = model.backend();
        let hidden_size = model.config.hidden_size;
        let intermediate_size = model.config.intermediate_size;

        let mut adapters: Vec<Box<dyn FfnOffloadAdapter>> =
            Vec::with_capacity(model.weights.layers.len());
        for (idx, layer) in model.weights.layers.iter().enumerate() {
            let adapter = PlannedCoreMlAdapter::new(
                idx,
                hidden_size,
                intermediate_size,
                layer,
                attention_backend,
            );
            adapters.push(Box::new(adapter));
        }

        Self::new(attention_backend, adapters)
    }

    pub fn stats(&self) -> HybridSyncStats {
        self.stats.lock().clone()
    }

    pub fn reset_stats(&self) {
        *self.stats.lock() = HybridSyncStats::default();
    }

    /// Forwards MLP activation through the layer's ANE adapter.
    pub fn forward_layer_mlp(
        &self,
        layer_idx: usize,
        ffn_normed: &Tensor,
    ) -> Result<Tensor, EngineError> {
        let adapter = self.adapters.get(layer_idx).ok_or_else(|| {
            EngineError::Model(format!("No ANE adapter registered for layer {}", layer_idx))
        })?;

        {
            let mut s = self.stats.lock();
            s.adapter_dispatches += 1;
            if !adapter.is_available() {
                s.fallback_dispatches += 1;
            }
        }

        let input_vec = ffn_normed.flatten_all()?.to_vec1::<f32>()?;
        let output_vec = adapter.forward_mlp(&input_vec)?;
        Tensor::from_vec(output_vec, ffn_normed.dims(), ffn_normed.device())
            .map_err(EngineError::from)
    }

    /// Executes a single hybrid transformer layer:
    /// Attention is executed on Metal/AMX, followed by residual add, FFN norm,
    /// and delegated MLP execution on the ANE adapter with synchronized residual add.
    pub fn execute_layer(
        &self,
        layer_idx: usize,
        hidden: &Tensor,
        layer: &LayerWeights,
        model: &crate::model::Model,
    ) -> Result<Tensor, EngineError> {
        {
            let mut s = self.stats.lock();
            s.attention_dispatches += 1;
        }

        // 1. Attention Norm
        let normed = model.rmsnorm(hidden, &layer.attn_norm)?;

        // 2. Linear Q, K, V (quantized storage; Q/K in rotate-half head order)
        let (q, k, v) = model.project_qkv(layer, &normed)?;

        let head_dim = model.config.head_dim();
        let num_q_heads = model.config.num_attention_heads;
        let num_kv_heads = model.config.num_key_value_heads;
        let seq_len = hidden.dim(0)?;

        let q = q
            .reshape((seq_len, num_q_heads, head_dim))?
            .transpose(0, 1)?;
        let k = k
            .reshape((seq_len, num_kv_heads, head_dim))?
            .transpose(0, 1)?;
        let v = v
            .reshape((seq_len, num_kv_heads, head_dim))?
            .transpose(0, 1)?;

        let (k, v) = if num_q_heads > num_kv_heads {
            let factor = num_q_heads / num_kv_heads;
            (k.repeat((factor, 1, 1))?, v.repeat((factor, 1, 1))?)
        } else {
            (k, v)
        };

        let scale = 1.0 / (head_dim as f64).sqrt();
        let scores = (model.matmul(&q, &k.t()?)? * scale)?;
        let attn_weights = candle_nn::ops::softmax_last_dim(&scores)?;
        let attn_out = model.matmul(&attn_weights, &v)?;

        let attn_out = attn_out
            .transpose(0, 1)?
            .reshape((seq_len, num_q_heads * head_dim))?;
        let attn_proj = model.linear(&layer.attn_output, &attn_out)?;

        // 3. Residual connection
        let hidden = (hidden + attn_proj)?;

        // 4. FFN Norm
        let ffn_normed = model.rmsnorm(&hidden, &layer.ffn_norm)?;

        // 5. Delegate MLP computation to ANE adapter
        let ffn_out = self.forward_layer_mlp(layer_idx, &ffn_normed)?;

        // 6. Synchronized residual connection
        let hidden = (hidden + ffn_out)?;

        Ok(hidden)
    }
}
