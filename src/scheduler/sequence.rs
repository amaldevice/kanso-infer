use crate::kv_cache::BlockTable;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SequenceStatus {
    Waiting,
    Running,
    Preempted,
    Finished,
}

#[derive(Debug, Clone)]
pub struct Sequence {
    pub id: u64,
    pub prompt_tokens: Vec<u32>,
    pub generated_tokens: Vec<u32>,
    pub block_table: BlockTable,
    pub status: SequenceStatus,
    pub max_tokens: usize,
    pub last_active_step: usize,
}

impl Sequence {
    pub fn new(id: u64, prompt_tokens: Vec<u32>, max_tokens: usize) -> Self {
        Self {
            id,
            prompt_tokens,
            generated_tokens: Vec::new(),
            block_table: BlockTable::new(),
            status: SequenceStatus::Waiting,
            max_tokens,
            last_active_step: 0,
        }
    }

    pub fn is_finished(&self) -> bool {
        self.status == SequenceStatus::Finished
    }

    pub fn total_tokens(&self) -> usize {
        self.prompt_tokens.len() + self.generated_tokens.len()
    }
}
