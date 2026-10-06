use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ModelArchitecture {
    DenseLlama,
    HybridMamba2,
    HybridLfm,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f64,
    pub block_size: usize,
    pub max_seq_len: usize,
    pub architecture: ModelArchitecture,
    pub attention_layers: Option<Vec<usize>>,
}

impl ModelConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub fn num_attention_layers(&self) -> usize {
        match &self.attention_layers {
            Some(indices) => indices.len(),
            None => self.num_hidden_layers,
        }
    }

    /// Computes uncompressed KV cache bytes per token: 2 * n_attn_layers * n_kv_heads * head_dim * 4 bytes (F32)
    pub fn kv_cache_bytes_per_token_f32(&self) -> usize {
        2 * self.num_attention_layers() * self.num_key_value_heads * self.head_dim() * std::mem::size_of::<f32>()
    }

    pub fn smollm2_360m() -> Self {
        Self {
            hidden_size: 960,
            num_attention_heads: 15,
            num_key_value_heads: 3,
            num_hidden_layers: 32,
            intermediate_size: 2560,
            vocab_size: 49152,
            rms_norm_eps: 1e-5,
            block_size: 16,
            max_seq_len: 2048,
            architecture: ModelArchitecture::DenseLlama,
            attention_layers: None,
        }
    }

    pub fn granite_4_0_350m_dense() -> Self {
        Self {
            hidden_size: 1024,
            num_attention_heads: 16,
            num_key_value_heads: 4,
            num_hidden_layers: 28,
            intermediate_size: 3072,
            vocab_size: 49152,
            rms_norm_eps: 1e-5,
            block_size: 16,
            max_seq_len: 8192,
            architecture: ModelArchitecture::DenseLlama,
            attention_layers: None,
        }
    }

    pub fn granite_4_0_h_350m() -> Self {
        Self {
            hidden_size: 768,
            num_attention_heads: 12,
            num_key_value_heads: 4,
            num_hidden_layers: 32,
            intermediate_size: 2048,
            vocab_size: 49152,
            rms_norm_eps: 1e-5,
            block_size: 16,
            max_seq_len: 32768,
            architecture: ModelArchitecture::HybridMamba2,
            attention_layers: Some(vec![7, 15, 23, 31]), // 4 attention layers mixed with 28 Mamba-2 layers
        }
    }

    pub fn lfm_2_5_350m() -> Self {
        Self {
            hidden_size: 1024,
            num_attention_heads: 16,
            num_key_value_heads: 4,
            num_hidden_layers: 16,
            intermediate_size: 2560,
            vocab_size: 65536,
            rms_norm_eps: 1e-5,
            block_size: 16,
            max_seq_len: 32768,
            architecture: ModelArchitecture::HybridLfm,
            attention_layers: Some(vec![2, 5, 8, 11, 14, 15]), // 6 attention layers interleaved with 10 LIV conv
        }
    }
}
