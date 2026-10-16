use serde::{Deserialize, Serialize};

use crate::bench::TraceEntry;
use crate::error::EngineError;

/// A single turn in a ShareGPT / LMSYS conversation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShareGptTurn {
    #[serde(alias = "role", alias = "speaker")]
    pub from: String,
    #[serde(alias = "content", alias = "text")]
    pub value: String,
}

/// A multi-turn conversation record from ShareGPT or LMSYS dataset dumps.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShareGptConversation {
    #[serde(default, deserialize_with = "deserialize_optional_id")]
    pub id: String,
    #[serde(default, alias = "turns", alias = "messages")]
    pub conversations: Vec<ShareGptTurn>,
}

fn deserialize_optional_id<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    struct IdVisitor;

    impl<'de> serde::de::Visitor<'de> for IdVisitor {
        type Value = String;

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("a string or integer id")
        }

        fn visit_str<E>(self, v: &str) -> Result<Self::Value, E> {
            Ok(v.to_string())
        }

        fn visit_i64<E>(self, v: i64) -> Result<Self::Value, E> {
            Ok(v.to_string())
        }

        fn visit_u64<E>(self, v: u64) -> Result<Self::Value, E> {
            Ok(v.to_string())
        }

        fn visit_none<E>(self) -> Result<Self::Value, E> {
            Ok(String::new())
        }

        fn visit_unit<E>(self) -> Result<Self::Value, E> {
            Ok(String::new())
        }
    }

    deserializer.deserialize_any(IdVisitor)
}

/// Normalizes raw role names to standard ChatML roles: "system", "user", "assistant".
pub fn normalize_role(from: &str) -> &str {
    match from.trim().to_ascii_lowercase().as_str() {
        "human" | "user" => "user",
        "gpt" | "chatgpt" | "assistant" | "bot" | "model" | "bing" | "bard" => "assistant",
        "system" => "system",
        _ => from.trim(),
    }
}

/// Formats a sequence of conversation turns into standard ChatML prompt format.
///
/// Output format:
/// `<|im_start|>{role}\n{content}<|im_end|>\n` for each message,
/// followed by `<|im_start|>assistant\n` to prompt assistant generation.
pub fn format_chatml_prompt(messages: &[ShareGptTurn]) -> String {
    let mut prompt_text = String::new();
    for msg in messages {
        let role = normalize_role(&msg.from);
        prompt_text.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", role, msg.value));
    }
    prompt_text.push_str("<|im_start|>assistant\n");
    prompt_text
}

/// Converts text into a deterministic token stream bounded within mock vocab range.
pub fn tokenize_chatml(text: &str) -> Vec<u32> {
    text.as_bytes().iter().map(|&b| (b % 40) as u32).collect()
}

/// Parses raw ShareGPT JSON dataset content (either a JSON array or newline-delimited JSON).
pub fn parse_conversations(json_content: &str) -> Result<Vec<ShareGptConversation>, EngineError> {
    let trimmed = json_content.trim();
    if trimmed.is_empty() {
        return Ok(Vec::new());
    }

    if trimmed.starts_with('[') {
        serde_json::from_str(trimmed)
            .map_err(|e| EngineError::Config(format!("Invalid ShareGPT JSON array: {}", e)))
    } else {
        let mut convs = Vec::new();
        for (idx, line) in trimmed.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let conv: ShareGptConversation = serde_json::from_str(line).map_err(|e| {
                EngineError::Config(format!("Invalid ShareGPT JSONL line {}: {}", idx + 1, e))
            })?;
            convs.push(conv);
        }
        if convs.is_empty() {
            let single: ShareGptConversation = serde_json::from_str(trimmed)
                .map_err(|e| EngineError::Config(format!("Invalid ShareGPT JSON object: {}", e)))?;
            convs.push(single);
        }
        Ok(convs)
    }
}

/// Ingests a ShareGPT JSON dataset and transforms multi-turn conversations into
/// structured `TraceEntry` sequences with staggered `arrival_step` intervals.
///
/// Default configuration: `turn_interval = 10`, `conv_stride = 2`.
pub fn parse_sharegpt_dataset(json_content: &str) -> Result<Vec<TraceEntry>, EngineError> {
    parse_sharegpt_dataset_with_config(json_content, 10, 2)
}

/// Ingests a ShareGPT JSON dataset with configurable turn and conversation arrival strides.
pub fn parse_sharegpt_dataset_with_config(
    json_content: &str,
    turn_interval: usize,
    conv_stride: usize,
) -> Result<Vec<TraceEntry>, EngineError> {
    let convs = parse_conversations(json_content)?;
    let mut entries = Vec::new();

    for (conv_idx, conv) in convs.iter().enumerate() {
        let mut history: Vec<ShareGptTurn> = Vec::new();
        let mut turn_idx = 0;
        let mut i = 0;

        while i < conv.conversations.len() {
            let turn = &conv.conversations[i];
            let role = normalize_role(&turn.from);

            if role == "system" {
                history.push(turn.clone());
                i += 1;
                continue;
            }

            if role == "user" {
                history.push(turn.clone());

                // Check for subsequent assistant response
                let gen_len = if i + 1 < conv.conversations.len()
                    && normalize_role(&conv.conversations[i + 1].from) == "assistant"
                {
                    let assistant_text = &conv.conversations[i + 1].value;
                    let gen_tokens = tokenize_chatml(assistant_text);
                    i += 1;
                    history.push(conv.conversations[i].clone());
                    gen_tokens.len().max(1)
                } else {
                    16
                };

                let prompt_slice =
                    if history.last().map(|t| normalize_role(&t.from)) == Some("assistant") {
                        &history[..history.len() - 1]
                    } else {
                        &history[..]
                    };

                let prompt_str = format_chatml_prompt(prompt_slice);
                let prompt_tokens = tokenize_chatml(&prompt_str);
                let prompt_len = prompt_tokens.len();
                let arrival_step = conv_idx * conv_stride + turn_idx * turn_interval;

                entries.push(TraceEntry {
                    prompt_len,
                    gen_len,
                    arrival_step,
                    prompt_tokens: Some(prompt_tokens),
                });

                turn_idx += 1;
                i += 1;
                continue;
            }

            // Other roles (e.g. unsolicited assistant message)
            history.push(turn.clone());
            i += 1;
        }
    }

    Ok(entries)
}
