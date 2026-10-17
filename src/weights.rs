use crate::config::{ModelArchitecture, ModelConfig};
use crate::error::EngineError;
use candle_core::Device;
use candle_core::Tensor;
use candle_core::quantized::gguf_file;
use candle_core::quantized::{GgmlDType, QMatMul, QStorage, QTensor};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// A projection weight kept in its storage format.
///
/// Real GGUF files store weights as GGML blocks (Q8_0, Q4_K, F16, ...) and #47
/// is precisely about *not* turning them into F32 at load: decode at batch 1 is
/// memory-bandwidth bound, so reading 1.06 bytes/param instead of 4 is a ~3.8x
/// cut in bytes moved per token. `QMatMul` keeps the blocks resident and runs
/// the matmul off that storage (candle's own NEON/dotprod kernels); the repo's
/// Metal kernels (`fused_llama_layer.metal`) are F32-only, so quantized Metal
/// GEMV is candle's `QMatMul`, not repo kernels. The dense variants are used for
/// synthetic/mock models, which are compared bit-for-bit by golden tests.
#[derive(Clone)]
pub struct LinearWeight(QMatMul);

impl LinearWeight {
    /// Wraps an already-dense tensor without touching its values.
    pub fn from_tensor(tensor: Tensor) -> Self {
        Self(QMatMul::Tensor(tensor))
    }

    /// Keeps a GGUF/GGML tensor in its storage format.
    ///
    /// `QMatMul::from_qtensor` expands F16 tensors to dense F32; #47 asks for F16
    /// as the floor when a file has no block-quantized weights, so F16 stays 16-bit.
    pub fn from_qtensor(qtensor: QTensor) -> Result<Self, EngineError> {
        if qtensor.dtype() == GgmlDType::F16 {
            let f16 = qtensor.dequantize_f16(&qtensor.device())?;
            return Ok(Self(QMatMul::TensorF16(f16)));
        }
        Ok(Self(QMatMul::from_qtensor(qtensor)?))
    }

    /// `x @ W^T`: the projection's last dim (`in_features`) maps to `out_features`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor, EngineError> {
        Ok(candle_core::Module::forward(&self.0, x)?)
    }

    /// Embedding lookup: dequantizes only the selected rows of a quantized table
    /// instead of materializing the whole `[vocab, hidden]` matrix. Rows are
    /// returned as F32 to match the F32 activation stream.
    pub fn embed(&self, ids: &Tensor) -> Result<Tensor, EngineError> {
        let rows = self.0.embedding(ids)?;
        Ok(if rows.dtype() == candle_core::DType::F32 {
            rows
        } else {
            rows.to_dtype(candle_core::DType::F32)?
        })
    }

    /// True when the weights are stored as quantized blocks (not dense F16/F32).
    pub fn is_quantized(&self) -> bool {
        matches!(self.0, QMatMul::QTensor(_))
    }

    /// Borrows the dense F32 storage, if this weight is dense (AMX GEMM needs it).
    pub fn as_dense_f32(&self) -> Option<&Tensor> {
        match &self.0 {
            QMatMul::Tensor(t) => Some(t),
            _ => None,
        }
    }

    /// Materializes dense F32 weights. Used for the bench F32 baseline and for
    /// adapters (ANE/Core ML) that only accept dense host buffers.
    pub fn to_dense(&self) -> Result<Tensor, EngineError> {
        Ok(match &self.0 {
            QMatMul::QTensor(t) => t.dequantize(&t.device())?,
            QMatMul::Tensor(t) | QMatMul::TensorF16(t) => t.to_dtype(candle_core::DType::F32)?,
        })
    }

    /// Re-stores the same values in `dtype` GGML blocks (dense F32 round-trip).
    ///
    /// Returns the dtype actually stored: `dtype` when the last dim divides its
    /// block size, else `Q4_0` (a GGML block may not straddle a row). The
    /// fallback prints to stderr so a bench label naming `dtype` cannot silently
    /// lie about what ran.
    pub fn requantize(&self, dtype: GgmlDType) -> Result<Self, EngineError> {
        let dense = self.to_dense()?;
        let effective = self.effective_requant_dtype(&dense, dtype);
        if effective != dtype {
            eprintln!(
                "requantize: {dtype:?} needs last-dim % {} == 0, got {} — storing Q4_0 instead",
                dtype.block_size(),
                dense.dim(candle_core::D::Minus1)?,
            );
        }
        Self::from_qtensor(QTensor::quantize(&dense, effective)?)
    }

    /// Dtype `requantize` will actually store for `dense` when asked for `dtype`.
    pub fn effective_requant_dtype(&self, dense: &Tensor, dtype: GgmlDType) -> GgmlDType {
        if dense
            .dim(candle_core::D::Minus1)
            .is_ok_and(|last| last.is_multiple_of(dtype.block_size()))
        {
            dtype
        } else {
            // A GGML block may not straddle a row: fall back to Q4_0 (32-wide
            // blocks) for the narrow matrices that Q4_K/Q8_K cannot represent.
            GgmlDType::Q4_0
        }
    }

    /// Bytes actually held in RAM by this weight's storage.
    pub fn storage_bytes(&self) -> usize {
        match &self.0 {
            QMatMul::QTensor(t) => t.storage_size_in_bytes(),
            QMatMul::Tensor(t) | QMatMul::TensorF16(t) => {
                t.elem_count() * t.dtype().size_in_bytes()
            }
        }
    }

    /// Number of stored parameters (quantized formats pack them).
    pub fn parameter_count(&self) -> usize {
        match &self.0 {
            QMatMul::QTensor(t) => t.shape().elem_count(),
            QMatMul::Tensor(t) | QMatMul::TensorF16(t) => t.elem_count(),
        }
    }

    pub fn dim(&self, axis: usize) -> usize {
        match &self.0 {
            QMatMul::QTensor(t) => t.shape().dims()[axis],
            QMatMul::Tensor(t) | QMatMul::TensorF16(t) => t.dims()[axis],
        }
    }

    /// Moves the weight (blocks or dense) onto `device`, keeping its storage format.
    pub fn to_device(&self, device: &Device) -> Result<Self, EngineError> {
        Ok(match &self.0 {
            QMatMul::Tensor(t) => Self(QMatMul::Tensor(t.to_device(device)?)),
            QMatMul::TensorF16(t) => Self(QMatMul::TensorF16(t.to_device(device)?)),
            QMatMul::QTensor(t) => {
                if t.device().same_device(device) {
                    Self(QMatMul::QTensor(t.clone()))
                } else {
                    // `QTensor` has no `to_device`: rebuild the block storage on
                    // the target device from the raw bytes.
                    let storage = QStorage::from_data(t.data()?, device, t.dtype())?;
                    Self(QMatMul::QTensor(Arc::new(QTensor::new(
                        storage,
                        t.shape().clone(),
                    )?)))
                }
            }
        })
    }

    /// Device holding the weight storage.
    pub fn device(&self) -> Device {
        match &self.0 {
            QMatMul::QTensor(t) => t.device(),
            QMatMul::Tensor(t) | QMatMul::TensorF16(t) => t.device().clone(),
        }
    }

    /// Identity of the underlying quantized storage, used to avoid counting tied
    /// weights (SmolLM2's `lm_head` reuses `token_embd`) twice.
    fn storage_id(&self) -> Option<usize> {
        match &self.0 {
            QMatMul::QTensor(t) => Some(Arc::as_ptr(t) as usize),
            QMatMul::Tensor(_) | QMatMul::TensorF16(_) => None,
        }
    }
}

#[derive(Clone)]
pub struct LayerWeights {
    pub attn_q: LinearWeight,
    pub attn_k: LinearWeight,
    pub attn_v: LinearWeight,
    pub attn_output: LinearWeight,
    pub attn_norm: Tensor,
    pub ffn_gate: LinearWeight,
    pub ffn_up: LinearWeight,
    pub ffn_down: LinearWeight,
    pub ffn_norm: Tensor,
}

/// Per-head Q/K row permutation for llama.cpp-style interleaved GGUF layouts.
///
/// Built once at load: permuting the projection output is the same permutation as
/// permuting the weight rows, but it works for any quantized block size (a GGML
/// block may not straddle a row), so the stored weights stay untouched.
#[derive(Clone)]
pub struct QkPerm {
    pub q: Tensor,
    pub k: Tensor,
}

#[derive(Clone)]
pub struct ModelWeights {
    pub embed_tokens: LinearWeight,
    pub layers: Vec<LayerWeights>,
    pub norm: Tensor,
    pub lm_head: LinearWeight,
    /// `Some` for llama.cpp GGUFs (Q/K stored interleaved per head, `x0, y0, x1, y1, ...`
    /// while `Model::rope_apply` uses the rotate-half layout); `None` for synthetic
    /// layouts that are already rotate-half.
    pub qk_perm: Option<QkPerm>,
}

/// Permutes Q/K weight matrix from llama.cpp's interleaved layout [x0, y0, x1, y1, ...]
/// to the standard rotate-half layout [x0, x1, ..., y0, y1, ...] used by custom kernels.
pub fn permute_interleaved_to_rotate_half(
    weight: &Tensor,
    num_heads: usize,
    head_dim: usize,
) -> Result<Tensor, EngineError> {
    let dev = weight.device();
    let rot_half_idx = interleaved_permutation(num_heads, head_dim, dev)?;
    let permuted = weight.index_select(&rot_half_idx, 0)?;
    Ok(permuted)
}

/// Index tensor mapping llama.cpp's interleaved per-head layout to rotate-half.
///
/// Shared by the (dense) weight permutation and the (any storage) activation
/// permutation, so the two layouts can never drift apart.
pub fn interleaved_permutation(
    num_heads: usize,
    head_dim: usize,
    dev: &Device,
) -> Result<Tensor, EngineError> {
    let half = head_dim / 2;
    let mut permutation = Vec::with_capacity(num_heads * head_dim);

    for h in 0..num_heads {
        let base = h * head_dim;
        // Evens first
        for i in 0..half {
            permutation.push((base + 2 * i) as u32);
        }
        // Odds next
        for i in 0..half {
            permutation.push((base + 2 * i + 1) as u32);
        }
    }

    Ok(Tensor::from_vec(permutation, (num_heads * head_dim,), dev)?)
}

/// Resolves a user-provided model path string by cascading through:
/// 1. Direct local file path
/// 2. Local `./models/` directory
/// 3. Standard Hugging Face hub cache directory `~/.cache/huggingface/hub/`
///
/// If no real model file is located, halts execution with an explicit error.
/// Shadow/mock model fallbacks are strictly disabled.
pub fn resolve_model_path(input: &str) -> Result<PathBuf, EngineError> {
    let direct = Path::new(input);
    if direct.is_file() {
        return Ok(direct.to_path_buf());
    }

    // Check in ./models/
    let in_models = Path::new("models").join(input);
    if in_models.is_file() {
        return Ok(in_models);
    }
    let in_models_gguf = Path::new("models").join(format!("{}.gguf", input));
    if in_models_gguf.is_file() {
        return Ok(in_models_gguf);
    }

    // Check in ~/.cache/huggingface/hub/
    if let Ok(home) = std::env::var("HOME") {
        let hf_hub = PathBuf::from(home).join(".cache/huggingface/hub");
        if hf_hub.is_dir() {
            // Search for direct snapshot file or matching pattern
            let filename = direct.file_name().and_then(|n| n.to_str()).unwrap_or(input);
            if let Ok(entries) = std::fs::read_dir(&hf_hub) {
                for entry in entries.flatten() {
                    let snapshots = entry.path().join("snapshots");
                    if snapshots.is_dir()
                        && let Ok(snaps) = std::fs::read_dir(snapshots)
                    {
                        for snap in snaps.flatten() {
                            let target = snap.path().join(filename);
                            if target.is_file() {
                                return Ok(target);
                            }
                            let target_gguf = snap.path().join(format!("{}.gguf", filename));
                            if target_gguf.is_file() {
                                return Ok(target_gguf);
                            }
                        }
                    }
                }
            }
        }
    }

    Err(EngineError::Config(format!(
        "Model file not found: '{}'. Shadow/mock model execution is disabled.\n\
         Please provide a valid GGUF file path (e.g. --model ./models/smollm2-360m.gguf) or download one via:\n\
         huggingface-cli download HuggingFaceTB/SmolLM2-360M-Instruct-GGUF smollm2-360m-instruct-q8_0.gguf",
        input
    )))
}

/// Reads one GGUF tensor from `file`, keeping the storage format recorded in the
/// file (the caller decides whether to dequantize: projection weights do not).
fn load_qtensor(
    content: &gguf_file::Content,
    file: &mut std::fs::File,
    name: &str,
    device: &Device,
) -> Result<QTensor, EngineError> {
    content
        .tensor(file, name, device)
        .map_err(|e| EngineError::Config(format!("Failed to load tensor '{}': {}", name, e)))
}

impl ModelWeights {
    /// Loads weights and architecture hyperparameters from an authentic GGUF file.
    ///
    /// Projection weights keep the storage format recorded in the file (Q8_0,
    /// Q4_K, F16, ...) so decode reads ~1 byte/param instead of 4 (#47); only the
    /// tiny 1-D norms are dequantized to F32.
    pub fn load_from_gguf(
        path: &Path,
        device: &Device,
    ) -> Result<(Self, ModelConfig), EngineError> {
        let mut file = std::fs::File::open(path)?;
        let content = gguf_file::Content::read(&mut file).map_err(|e| {
            EngineError::Config(format!(
                "Failed to parse GGUF file {}: {}",
                path.display(),
                e
            ))
        })?;

        let metadata = |key: &str| -> Result<&gguf_file::Value, EngineError> {
            content
                .metadata
                .get(key)
                .ok_or_else(|| EngineError::Config(format!("Missing GGUF metadata key '{}'", key)))
        };

        // One reader: projection weights keep the file's blocks (Q8_0/Q4_K/...),
        // while the tiny 1-D norms are dequantized to F32 for the RMSNorm math.
        let mut load_qtensor = |name: &str| load_qtensor(&content, &mut file, name, device);

        let arch = metadata("general.architecture")
            .and_then(|v| {
                v.to_string()
                    .cloned()
                    .map_err(|e| EngineError::Config(e.to_string()))
            })
            .unwrap_or_else(|_| "llama".to_string());
        let prefix = match arch.as_str() {
            "granite" => "granite",
            _ => "llama",
        };

        let num_layers = metadata(&format!("{}.block_count", prefix))?.to_u32()? as usize;
        let num_attention_heads =
            metadata(&format!("{}.attention.head_count", prefix))?.to_u32()? as usize;
        let num_kv_heads =
            metadata(&format!("{}.attention.head_count_kv", prefix))?.to_u32()? as usize;
        let hidden_size = metadata(&format!("{}.embedding_length", prefix))?.to_u32()? as usize;
        let intermediate_size =
            metadata(&format!("{}.feed_forward_length", prefix))?.to_u32()? as usize;
        let rms_norm_eps = metadata(&format!("{}.attention.layer_norm_rms_epsilon", prefix))
            .map(|v| v.to_f32().unwrap_or(1e-5) as f64)
            .unwrap_or(1e-5);
        let rope_theta = metadata(&format!("{}.rope.freq_base", prefix))
            .and_then(|v| v.to_f32().map_err(|e| EngineError::Config(e.to_string())))
            .unwrap_or(10000.0);
        let token_embd = LinearWeight::from_qtensor(load_qtensor("token_embd.weight")?)?;
        let vocab_size = token_embd.dim(0);
        let head_dim = hidden_size / num_attention_heads;

        let norm = load_qtensor("output_norm.weight")?.dequantize(device)?;
        let lm_head = if content.tensor_infos.contains_key("output.weight") {
            LinearWeight::from_qtensor(load_qtensor("output.weight")?)?
        } else {
            token_embd.clone()
        };

        let mut layers = Vec::with_capacity(num_layers);

        for i in 0..num_layers {
            let attn_norm =
                load_qtensor(&format!("blk.{}.attn_norm.weight", i))?.dequantize(device)?;
            let attn_q =
                LinearWeight::from_qtensor(load_qtensor(&format!("blk.{}.attn_q.weight", i))?)?;
            let attn_k =
                LinearWeight::from_qtensor(load_qtensor(&format!("blk.{}.attn_k.weight", i))?)?;
            let attn_v =
                LinearWeight::from_qtensor(load_qtensor(&format!("blk.{}.attn_v.weight", i))?)?;
            let attn_output = LinearWeight::from_qtensor(load_qtensor(&format!(
                "blk.{}.attn_output.weight",
                i
            ))?)?;

            let ffn_norm =
                load_qtensor(&format!("blk.{}.ffn_norm.weight", i))?.dequantize(device)?;
            let ffn_gate =
                LinearWeight::from_qtensor(load_qtensor(&format!("blk.{}.ffn_gate.weight", i))?)?;
            let ffn_up =
                LinearWeight::from_qtensor(load_qtensor(&format!("blk.{}.ffn_up.weight", i))?)?;
            let ffn_down =
                LinearWeight::from_qtensor(load_qtensor(&format!("blk.{}.ffn_down.weight", i))?)?;

            layers.push(LayerWeights {
                attn_q,
                attn_k,
                attn_v,
                attn_output,
                attn_norm,
                ffn_gate,
                ffn_up,
                ffn_down,
                ffn_norm,
            });
        }

        let config = ModelConfig {
            hidden_size,
            num_attention_heads,
            num_key_value_heads: num_kv_heads,
            num_hidden_layers: num_layers,
            intermediate_size,
            vocab_size,
            rms_norm_eps,
            block_size: 16,
            max_seq_len: 2048,
            architecture: ModelArchitecture::DenseLlama,
            attention_layers: None,
            rope_theta,
        };
        let weights = Self {
            embed_tokens: token_embd,
            layers,
            norm,
            lm_head,
            // llama.cpp GGUF layout, see `QkPerm`.
            qk_perm: Some(QkPerm {
                q: interleaved_permutation(num_attention_heads, head_dim, device)?,
                k: interleaved_permutation(num_kv_heads, head_dim, device)?,
            }),
        };

        Ok((weights, config))
    }

    /// Every projection weight, in a fixed order.
    pub fn linear_weights(&self) -> Vec<&LinearWeight> {
        let mut all: Vec<&LinearWeight> = vec![&self.embed_tokens, &self.lm_head];
        for layer in &self.layers {
            all.extend([
                &layer.attn_q,
                &layer.attn_k,
                &layer.attn_v,
                &layer.attn_output,
                &layer.ffn_gate,
                &layer.ffn_up,
                &layer.ffn_down,
            ]);
        }
        all
    }

    /// Peak-resident weight bytes; tied quantized weights are counted once.
    pub fn storage_bytes(&self) -> usize {
        let mut seen = HashSet::new();
        self.linear_weights()
            .iter()
            .filter(|w| w.storage_id().is_none_or(|id| seen.insert(id)))
            .map(|w| w.storage_bytes())
            .sum()
    }

    /// Total stored parameters (quantized formats pack them), tied weights once.
    pub fn parameter_count(&self) -> usize {
        let mut seen = HashSet::new();
        self.linear_weights()
            .iter()
            .filter(|w| w.storage_id().is_none_or(|id| seen.insert(id)))
            .map(|w| w.parameter_count())
            .sum()
    }

    /// Dense F32 copy of every weight: the pre-#47 behaviour, kept as the bench
    /// baseline and for adapters that need host-side f32 buffers.
    pub fn to_dense(&self) -> Result<Self, EngineError> {
        Ok(Self {
            embed_tokens: LinearWeight::from_tensor(self.embed_tokens.to_dense()?),
            layers: self
                .layers
                .iter()
                .map(|layer| {
                    Ok(LayerWeights {
                        attn_q: LinearWeight::from_tensor(layer.attn_q.to_dense()?),
                        attn_k: LinearWeight::from_tensor(layer.attn_k.to_dense()?),
                        attn_v: LinearWeight::from_tensor(layer.attn_v.to_dense()?),
                        attn_output: LinearWeight::from_tensor(layer.attn_output.to_dense()?),
                        attn_norm: layer.attn_norm.clone(),
                        ffn_gate: LinearWeight::from_tensor(layer.ffn_gate.to_dense()?),
                        ffn_up: LinearWeight::from_tensor(layer.ffn_up.to_dense()?),
                        ffn_down: LinearWeight::from_tensor(layer.ffn_down.to_dense()?),
                        ffn_norm: layer.ffn_norm.clone(),
                    })
                })
                .collect::<Result<Vec<_>, EngineError>>()?,
            norm: self.norm.clone(),
            lm_head: LinearWeight::from_tensor(self.lm_head.to_dense()?),
            qk_perm: self.qk_perm.clone(),
        })
    }

    /// Re-stores every weight as `dtype` GGML blocks (bench quant-fidelity runs).
    pub fn requantize(&self, dtype: GgmlDType) -> Result<Self, EngineError> {
        Ok(Self {
            embed_tokens: self.embed_tokens.requantize(dtype)?,
            layers: self
                .layers
                .iter()
                .map(|layer| {
                    Ok(LayerWeights {
                        attn_q: layer.attn_q.requantize(dtype)?,
                        attn_k: layer.attn_k.requantize(dtype)?,
                        attn_v: layer.attn_v.requantize(dtype)?,
                        attn_output: layer.attn_output.requantize(dtype)?,
                        attn_norm: layer.attn_norm.clone(),
                        ffn_gate: layer.ffn_gate.requantize(dtype)?,
                        ffn_up: layer.ffn_up.requantize(dtype)?,
                        ffn_down: layer.ffn_down.requantize(dtype)?,
                        ffn_norm: layer.ffn_norm.clone(),
                    })
                })
                .collect::<Result<Vec<_>, EngineError>>()?,
            norm: self.norm.clone(),
            lm_head: self.lm_head.requantize(dtype)?,
            qk_perm: self.qk_perm.clone(),
        })
    }

    /// Moves every weight onto `device`, preserving quantized storage.
    pub fn to_device(&self, device: &Device) -> Result<Self, EngineError> {
        Ok(Self {
            embed_tokens: self.embed_tokens.to_device(device)?,
            layers: self
                .layers
                .iter()
                .map(|layer| {
                    Ok(LayerWeights {
                        attn_q: layer.attn_q.to_device(device)?,
                        attn_k: layer.attn_k.to_device(device)?,
                        attn_v: layer.attn_v.to_device(device)?,
                        attn_output: layer.attn_output.to_device(device)?,
                        attn_norm: layer.attn_norm.to_device(device)?,
                        ffn_gate: layer.ffn_gate.to_device(device)?,
                        ffn_up: layer.ffn_up.to_device(device)?,
                        ffn_down: layer.ffn_down.to_device(device)?,
                        ffn_norm: layer.ffn_norm.to_device(device)?,
                    })
                })
                .collect::<Result<Vec<_>, EngineError>>()?,
            norm: self.norm.to_device(device)?,
            lm_head: self.lm_head.to_device(device)?,
            qk_perm: self
                .qk_perm
                .as_ref()
                .map(|p| {
                    Ok::<_, EngineError>(QkPerm {
                        q: p.q.to_device(device)?,
                        k: p.k.to_device(device)?,
                    })
                })
                .transpose()?,
        })
    }
}
