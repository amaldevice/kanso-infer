use crate::error::EngineError;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

#[derive(Clone)]
pub struct TokenizerWrapper {
    inner: Tokenizer,
    pub eos_token_id: u32,
    pub bos_token_id: u32,
}

impl TokenizerWrapper {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, EngineError> {
        let path_ref = path.as_ref();
        let inner = Tokenizer::from_file(path_ref).map_err(|e| {
            EngineError::Config(format!(
                "Failed to load tokenizer from {}: {}",
                path_ref.display(),
                e
            ))
        })?;

        // Extract or default special tokens
        let eos_token_id = inner
            .token_to_id("<|im_end|>")
            .or_else(|| inner.token_to_id("</s>"))
            .or_else(|| inner.token_to_id("<eos>"))
            .unwrap_or(2);

        let bos_token_id = inner
            .token_to_id("<|im_start|>")
            .or_else(|| inner.token_to_id("<s>"))
            .or_else(|| inner.token_to_id("<bos>"))
            .unwrap_or(1);

        Ok(Self {
            inner,
            eos_token_id,
            bos_token_id,
        })
    }

    /// Resolves tokenizer file by searching:
    /// 1. Explicit argument path if provided
    /// 2. `./tokenizer.json`
    /// 3. `./models/tokenizer.json`
    /// 4. In the model file's parent directory
    /// 5. In Hugging Face cache snapshots
    pub fn resolve(explicit: Option<&str>, model_path: Option<&Path>) -> Result<Self, EngineError> {
        if let Some(path_str) = explicit {
            let p = Path::new(path_str);
            if p.is_file() {
                return Self::from_file(p);
            }
        }

        // Check current directory
        let local = Path::new("tokenizer.json");
        if local.is_file() {
            return Self::from_file(local);
        }

        // Check models/ directory
        let in_models = Path::new("models/tokenizer.json");
        if in_models.is_file() {
            return Self::from_file(in_models);
        }

        // Check model parent directory
        if let Some(mp) = model_path
            && let Some(parent) = mp.parent()
        {
            let candidate = parent.join("tokenizer.json");
            if candidate.is_file() {
                return Self::from_file(&candidate);
            }
        }

        // Check in ~/.cache/huggingface/hub
        if let Ok(home) = std::env::var("HOME") {
            let hf_hub = PathBuf::from(home).join(".cache/huggingface/hub");
            if hf_hub.is_dir()
                && let Ok(entries) = std::fs::read_dir(&hf_hub)
            {
                for entry in entries.flatten() {
                    let candidate = entry.path().join("snapshots");
                    if candidate.is_dir()
                        && let Ok(snaps) = std::fs::read_dir(candidate)
                    {
                        for snap in snaps.flatten() {
                            let t = snap.path().join("tokenizer.json");
                            if t.is_file() {
                                return Self::from_file(&t);
                            }
                        }
                    }
                }
            }
        }

        Err(EngineError::Config(
            "tokenizer.json not found. Please provide --tokenizer <path> or place tokenizer.json in the working directory.".to_string()
        ))
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>, EngineError> {
        let encoding = self
            .inner
            .encode(text, true)
            .map_err(|e| EngineError::Config(format!("Tokenization encoding error: {}", e)))?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String, EngineError> {
        self.inner
            .decode(ids, false)
            .map_err(|e| EngineError::Config(format!("Tokenization decoding error: {}", e)))
    }

    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.inner.id_to_token(id)
    }
}

/// Incrementally decodes token IDs into valid UTF-8 text deltas without mangling
/// multi-byte characters or leading spaces.
#[derive(Default)]
pub struct StreamDecoder {
    ids: Vec<u32>,
    prefix_offset: usize,
    read_offset: usize,
}

impl StreamDecoder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Pushes a single token ID and returns any completed UTF-8 text delta
    pub fn push(&mut self, tokenizer: &TokenizerWrapper, id: u32) -> Result<String, EngineError> {
        self.ids.push(id);
        let prefix = tokenizer.decode(&self.ids[self.prefix_offset..self.read_offset])?;
        let text = tokenizer.decode(&self.ids[self.prefix_offset..])?;

        if text.len() > prefix.len() && !text.ends_with('\u{FFFD}') {
            let delta = text.get(prefix.len()..).unwrap_or_default().to_string();
            self.prefix_offset = self.read_offset;
            self.read_offset = self.ids.len();
            Ok(delta)
        } else {
            Ok(String::new())
        }
    }

    /// Flushes any remaining bytes currently held back
    pub fn flush(&mut self, tokenizer: &TokenizerWrapper) -> Result<String, EngineError> {
        let prefix = tokenizer.decode(&self.ids[self.prefix_offset..self.read_offset])?;
        let text = tokenizer.decode(&self.ids[self.prefix_offset..])?;
        self.prefix_offset = self.ids.len();
        self.read_offset = self.ids.len();
        Ok(text.get(prefix.len()..).unwrap_or_default().to_string())
    }
}
