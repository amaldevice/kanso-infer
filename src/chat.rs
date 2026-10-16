use crate::engine::Engine;
use crate::error::EngineError;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}
use crate::tokenizer::{StreamDecoder, TokenizerWrapper};

pub struct ChatSession {
    pub engine: Engine,
    pub history: Vec<ChatMessage>,
    pub tokenizer: Option<TokenizerWrapper>,
}

impl ChatSession {
    pub fn new(engine: Engine, system_prompt: Option<String>) -> Self {
        Self::new_with_tokenizer(engine, system_prompt, None)
    }

    pub fn new_with_tokenizer(
        engine: Engine,
        system_prompt: Option<String>,
        tokenizer: Option<TokenizerWrapper>,
    ) -> Self {
        let mut history = Vec::new();
        if let Some(sys) = system_prompt {
            history.push(ChatMessage {
                role: "system".to_string(),
                content: sys,
            });
        }
        Self {
            engine,
            history,
            tokenizer,
        }
    }

    pub fn history(&self) -> &[ChatMessage] {
        &self.history
    }

    pub fn add_user_message(&mut self, content: &str) {
        self.history.push(ChatMessage {
            role: "user".to_string(),
            content: content.to_string(),
        });
    }

    /// Formats the conversation history into ChatML-style text and token IDs
    pub fn format_prompt(&self) -> (String, Vec<u32>) {
        let mut prompt_text = String::new();
        for msg in &self.history {
            prompt_text.push_str(&format!(
                "<|im_start|>{}\n{}<|im_end|>\n",
                msg.role, msg.content
            ));
        }
        prompt_text.push_str("<|im_start|>assistant\n");

        let tokens: Vec<u32> = if let Some(tok) = &self.tokenizer {
            tok.encode(&prompt_text).unwrap_or_else(|_| {
                prompt_text
                    .as_bytes()
                    .iter()
                    .map(|&b| (b % 40) as u32)
                    .collect()
            })
        } else {
            prompt_text
                .as_bytes()
                .iter()
                .map(|&b| (b % 40) as u32)
                .collect()
        };

        (prompt_text, tokens)
    }

    /// Executes streaming step generation for the current conversation state,
    /// invoking `on_token(&delta)` for each newly generated text chunk in real time.
    pub fn stream_step_reply<F>(
        &mut self,
        max_tokens: usize,
        mut on_token: F,
    ) -> Result<String, EngineError>
    where
        F: FnMut(&str),
    {
        let (_, tokens) = self.format_prompt();
        let eos_id = self.tokenizer.as_ref().map(|t| t.eos_token_id).unwrap_or(2);
        // #56: hand the engine the stop token, so EOS retires the sequence and releases
        // its KV blocks instead of generating up to max_tokens holding them.
        // F5: the REPL exposes no sampling flags, so this path keeps the greedy default
        // `SamplingParams` (`add_request_with_sampling` is the hook if it ever grows one).
        let request_id = self
            .engine
            .add_request_with_stops(tokens, max_tokens, vec![eos_id]);

        let mut generated_tokens = Vec::new();
        let mut done = false;
        let mut accumulated_reply = String::new();
        let mut decoder = StreamDecoder::new();

        while !done && self.engine.has_unfinished_requests() {
            let outputs = self.engine.step()?;
            if outputs.is_empty() {
                return Err(EngineError::Scheduler(
                    "Engine unable to allocate sufficient KV blocks to advance request".to_string(),
                ));
            }
            for out in outputs {
                if out.request_id == request_id {
                    generated_tokens.push(out.token_id);

                    // Incremental decoding delta
                    if let Some(tok) = &self.tokenizer {
                        let delta = decoder.push(tok, out.token_id)?;
                        if !delta.is_empty() {
                            let clean_delta = delta.replace("<|im_end|>", "");
                            if !clean_delta.is_empty() {
                                on_token(&clean_delta);
                            }
                            accumulated_reply.push_str(&delta);
                        }
                    } else {
                        let delta = format!("tok_{} ", out.token_id);
                        on_token(&delta);
                        accumulated_reply.push_str(&delta);
                    }

                    if out.token_id == eos_id
                        || accumulated_reply.contains("<|im_end|>")
                        || out.is_finished
                    {
                        done = true;
                    }
                }
            }
        }

        // #56: drop this turn's retired sequence so a long-lived session does not grow
        // the engine's active vector forever.
        self.engine.drain_finished();

        if let Some(tok) = &self.tokenizer {
            let remaining = decoder.flush(tok)?;
            if !remaining.is_empty() {
                let clean_rem = remaining.replace("<|im_end|>", "");
                if !clean_rem.is_empty() {
                    on_token(&clean_rem);
                }
                accumulated_reply.push_str(&remaining);
            }
        }

        let cleaned_reply = accumulated_reply
            .trim_end_matches("<|im_end|>")
            .trim()
            .to_string();

        self.history.push(ChatMessage {
            role: "assistant".to_string(),
            content: cleaned_reply.clone(),
        });

        Ok(cleaned_reply)
    }

    pub fn step_reply(&mut self, max_tokens: usize) -> Result<String, EngineError> {
        self.stream_step_reply(max_tokens, |_| {})
    }

    pub fn reset(&mut self) {
        self.history.retain(|m| m.role == "system");
    }
}
