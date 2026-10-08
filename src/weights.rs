use candle_core::Tensor;
use std::path::{Path, PathBuf};
use candle_core::Device;
use candle_core::quantized::gguf_file;
use crate::config::{ModelArchitecture, ModelConfig};
use crate::error::EngineError;

#[derive(Clone)]
pub struct LayerWeights {
    pub attn_q: Tensor,
    pub attn_k: Tensor,
    pub attn_v: Tensor,
    pub attn_output: Tensor,
    pub attn_norm: Tensor,
    pub ffn_gate: Tensor,
    pub ffn_up: Tensor,
    pub ffn_down: Tensor,
    pub ffn_norm: Tensor,
}

#[derive(Clone)]
pub struct ModelWeights {
    pub embed_tokens: Tensor,
    pub layers: Vec<LayerWeights>,
    pub norm: Tensor,
    pub lm_head: Tensor,
}

/// Permutes Q/K weight matrix from llama.cpp's interleaved layout [x0, y0, x1, y1, ...]
/// to the standard rotate-half layout [x0, x1, ..., y0, y1, ...] used by custom kernels.
pub fn permute_interleaved_to_rotate_half(
    weight: &Tensor,
    num_heads: usize,
    head_dim: usize,
) -> Result<Tensor, EngineError> {
    let dev = weight.device();
    let rows = num_heads * head_dim;
    let _cols = weight.dims()[1];

    let mut permutation = Vec::with_capacity(rows);
    let half = head_dim / 2;

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

    let idx = Tensor::from_vec(permutation, (rows,), dev)?;
    let permuted = weight.index_select(&idx, 0)?;
    Ok(permuted)
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
                    if snapshots.is_dir() {
                        if let Ok(snaps) = std::fs::read_dir(snapshots) {
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
    }

    Err(EngineError::Config(format!(
        "Model file not found: '{}'. Shadow/mock model execution is disabled.\n\
         Please provide a valid GGUF file path (e.g. --model ./models/smollm2-360m.gguf) or download one via:\n\
         huggingface-cli download HuggingFaceTB/SmolLM2-360M-Instruct-GGUF smollm2-360m-instruct-q8_0.gguf",
        input
    )))
}

impl ModelWeights {
    /// Loads weights and architecture hyperparameters from an authentic GGUF file.
    pub fn load_from_gguf(path: &Path, device: &Device) -> Result<(Self, ModelConfig), EngineError> {
        let mut file = std::fs::File::open(path)?;
        let content = gguf_file::Content::read(&mut file).map_err(|e| {
            EngineError::Config(format!("Failed to parse GGUF file {}: {}", path.display(), e))
        })?;

        let metadata = |key: &str| -> Result<&gguf_file::Value, EngineError> {
            content
                .metadata
                .get(key)
                .ok_or_else(|| EngineError::Config(format!("Missing GGUF metadata key '{}'", key)))
        };

        let mut load_tensor = |name: &str| -> Result<Tensor, EngineError> {
            let tensor = content
                .tensor(&mut file, name, device)
                .map_err(|e| EngineError::Config(format!("Failed to load tensor '{}': {}", name, e)))?;
            Ok(tensor.dequantize(device)?)
        };

        let arch = metadata("general.architecture")
            .and_then(|v| v.to_string().map(|s| s.clone()).map_err(|e| EngineError::Config(e.to_string())))
            .unwrap_or_else(|_| "llama".to_string());
        let prefix = match arch.as_str() {
            "granite" => "granite",
            _ => "llama",
        };

        let num_layers = metadata(&format!("{}.block_count", prefix))?.to_u32()? as usize;
        let num_attention_heads = metadata(&format!("{}.attention.head_count", prefix))?.to_u32()? as usize;
        let num_kv_heads = metadata(&format!("{}.attention.head_count_kv", prefix))?.to_u32()? as usize;
        let hidden_size = metadata(&format!("{}.embedding_length", prefix))?.to_u32()? as usize;
        let intermediate_size = metadata(&format!("{}.feed_forward_length", prefix))?.to_u32()? as usize;
        let rms_norm_eps = metadata(&format!("{}.attention.layer_norm_rms_epsilon", prefix))
            .map(|v| v.to_f32().unwrap_or(1e-5) as f64)
            .unwrap_or(1e-5);

        let token_embd = load_tensor("token_embd.weight")?;
        let vocab_size = token_embd.dim(0)?;

        let norm = load_tensor("output_norm.weight")?;
        let lm_head = if content.tensor_infos.contains_key("output.weight") {
            load_tensor("output.weight")?
        } else {
            token_embd.clone()
        };

        let head_dim = hidden_size / num_attention_heads;
        let mut layers = Vec::with_capacity(num_layers);

        for i in 0..num_layers {
            let attn_norm = load_tensor(&format!("blk.{}.attn_norm.weight", i))?;
            let raw_q = load_tensor(&format!("blk.{}.attn_q.weight", i))?;
            let raw_k = load_tensor(&format!("blk.{}.attn_k.weight", i))?;
            let attn_v = load_tensor(&format!("blk.{}.attn_v.weight", i))?;
            let attn_output = load_tensor(&format!("blk.{}.attn_output.weight", i))?;

            let attn_q = permute_interleaved_to_rotate_half(&raw_q, num_attention_heads, head_dim)?;
            let attn_k = permute_interleaved_to_rotate_half(&raw_k, num_kv_heads, head_dim)?;

            let ffn_norm = load_tensor(&format!("blk.{}.ffn_norm.weight", i))?;
            let ffn_gate = load_tensor(&format!("blk.{}.ffn_gate.weight", i))?;
            let ffn_up = load_tensor(&format!("blk.{}.ffn_up.weight", i))?;
            let ffn_down = load_tensor(&format!("blk.{}.ffn_down.weight", i))?;

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
        };

        let weights = Self {
            embed_tokens: token_embd,
            layers,
            norm,
            lm_head,
        };

        Ok((weights, config))
    }
}
