use crate::engine::Engine;
use crate::error::EngineError;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

pub struct ChatSession {
    pub engine: Engine,
    pub history: Vec<ChatMessage>,
}

impl ChatSession {
    pub fn new(engine: Engine, system_prompt: Option<String>) -> Self {
        let mut history = Vec::new();
        if let Some(sys) = system_prompt {
            history.push(ChatMessage {
                role: "system".to_string(),
                content: sys,
            });
        }
        Self { engine, history }
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
            prompt_text.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", msg.role, msg.content));
        }
        prompt_text.push_str("<|im_start|>assistant\n");

        // Mock tokenization based on character bytes modulo vocab range
        let tokens: Vec<u32> = prompt_text
            .as_bytes()
            .iter()
            .map(|&b| (b % 40) as u32)
            .collect();

        (prompt_text, tokens)
    }

    /// Executes step generation for the current conversation state and appends assistant reply
    pub fn step_reply(&mut self, max_tokens: usize) -> Result<String, EngineError> {
        let (_, tokens) = self.format_prompt();
        let request_id = self.engine.add_request(tokens, max_tokens);

        let mut generated_tokens = Vec::new();
        let mut done = false;

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
                    if out.is_finished {
                        done = true;
                    }
                }
            }
        }

        let reply_text = format!("Response generated with {} tokens", generated_tokens.len());
        self.history.push(ChatMessage {
            role: "assistant".to_string(),
            content: reply_text.clone(),
        });

        Ok(reply_text)
    }

    pub fn reset(&mut self) {
        self.history.retain(|m| m.role == "system");
    }
}
