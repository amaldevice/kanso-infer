use crate::kv_cache::BlockTable;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SequenceStatus {
    Waiting,
    Running,
    Preempted,
    Finished,
}

/// Per-request sampling knobs (#56). The defaults reproduce the engine's historical
/// greedy behaviour; callers override the fields they care about on the sequence.
#[derive(Debug, Clone, Copy)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub seed: u64,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            seed: 42,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Sequence {
    pub id: u64,
    pub prompt_tokens: Vec<u32>,
    pub generated_tokens: Vec<u32>,
    pub block_table: BlockTable,
    pub status: SequenceStatus,
    pub prefilled_tokens: usize,
    pub max_tokens: usize,
    pub last_active_step: usize,
    /// Token ids that terminate generation (EOS and friends); empty = run to `max_tokens`.
    pub stop_token_ids: Vec<u32>,
    pub sampling: SamplingParams,
}

impl Sequence {
    pub fn new(id: u64, prompt_tokens: Vec<u32>, max_tokens: usize) -> Self {
        Self {
            id,
            prompt_tokens,
            generated_tokens: Vec::new(),
            block_table: BlockTable::new(),
            status: SequenceStatus::Waiting,
            prefilled_tokens: 0,
            max_tokens,
            last_active_step: 0,
            stop_token_ids: Vec::new(),
            sampling: SamplingParams::default(),
        }
    }

    pub fn is_finished(&self) -> bool {
        self.status == SequenceStatus::Finished
    }

    /// #56: the one termination rule — `max_tokens` reached, or `token` (the token just
    /// appended to `generated_tokens`) is one of the configured stop ids.
    pub fn is_done_after(&self, token: u32) -> bool {
        self.generated_tokens.len() >= self.max_tokens || self.stop_token_ids.contains(&token)
    }

    pub fn total_tokens(&self) -> usize {
        self.prompt_tokens.len() + self.generated_tokens.len()
    }
}
