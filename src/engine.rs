use crate::error::EngineError;
use crate::kv_cache::{BlockPool, KvStorage, PrefixCache};
use crate::model::Model;
use crate::sampler::sample_token;
use crate::scheduler::{Sequence, SequenceStatus};

#[derive(Debug, Clone)]
pub struct StepOutput {
    pub request_id: u64,
    pub token_id: u32,
    pub is_finished: bool,
}
#[derive(Debug, Clone)]
pub struct ScratchpadWorkspace {
    pub max_batch: usize,
    pub hidden_dim: usize,
    pub intermediate_dim: usize,
    pub vocab_size: usize,
    pub hidden_states: Vec<f32>,
    pub normed_states: Vec<f32>,
    pub q_proj: Vec<f32>,
    pub k_proj: Vec<f32>,
    pub v_proj: Vec<f32>,
    pub attn_out: Vec<f32>,
    pub ffn_gate: Vec<f32>,
    pub ffn_up: Vec<f32>,
    pub ffn_down: Vec<f32>,
    pub logits: Vec<f32>,
}

impl ScratchpadWorkspace {
    pub fn new(
        max_batch: usize,
        hidden_dim: usize,
        intermediate_dim: usize,
        vocab_size: usize,
    ) -> Self {
        Self {
            max_batch,
            hidden_dim,
            intermediate_dim,
            vocab_size,
            hidden_states: vec![0.0f32; max_batch * hidden_dim],
            normed_states: vec![0.0f32; max_batch * hidden_dim],
            q_proj: vec![0.0f32; max_batch * hidden_dim],
            k_proj: vec![0.0f32; max_batch * hidden_dim],
            v_proj: vec![0.0f32; max_batch * hidden_dim],
            attn_out: vec![0.0f32; max_batch * hidden_dim],
            ffn_gate: vec![0.0f32; max_batch * intermediate_dim],
            ffn_up: vec![0.0f32; max_batch * intermediate_dim],
            ffn_down: vec![0.0f32; max_batch * hidden_dim],
            logits: vec![0.0f32; max_batch * vocab_size],
        }
    }

    #[inline(always)]
    pub fn clear(&mut self) {
        self.hidden_states.fill(0.0);
        self.normed_states.fill(0.0);
        self.q_proj.fill(0.0);
        self.k_proj.fill(0.0);
        self.v_proj.fill(0.0);
        self.attn_out.fill(0.0);
        self.ffn_gate.fill(0.0);
        self.ffn_up.fill(0.0);
        self.ffn_down.fill(0.0);
        self.logits.fill(0.0);
    }
}

pub struct Engine {
    pub model: Model,
    pub block_pool: BlockPool,
    pub prefix_cache: PrefixCache,
    pub sequences: Vec<Sequence>,
    pub max_running: usize,
    pub block_size: usize,
    pub current_step: usize,
    pub next_request_id: u64,
    pub kv_storage: KvStorage,
    pub workspace: ScratchpadWorkspace,
}

impl Engine {
    pub fn new(
        model: Model,
        max_running: usize,
        total_blocks: usize,
        block_size: usize,
    ) -> Self {
        let kv_storage = KvStorage::new(
            model.config.num_hidden_layers,
            total_blocks,
            block_size,
            model.config.num_key_value_heads,
            model.config.head_dim(),
            &model.device,
        )
        .expect("failed to initialize kv storage");

        let workspace = ScratchpadWorkspace::new(
            max_running,
            model.config.hidden_size,
            model.config.intermediate_size,
            model.config.vocab_size,
        );

        Self {
            model,
            block_pool: BlockPool::new(total_blocks),
            prefix_cache: PrefixCache::new(block_size),
            sequences: Vec::new(),
            max_running,
            block_size,
            current_step: 0,
            next_request_id: 1,
            kv_storage,
            workspace,
        }
    }
    pub fn add_request(&mut self, prompt_tokens: Vec<u32>, max_tokens: usize) -> u64 {
        let id = self.next_request_id;
        self.next_request_id += 1;
        self.sequences.push(Sequence::new(id, prompt_tokens, max_tokens));
        id
    }

    pub fn waiting_count(&self) -> usize {
        self.sequences
            .iter()
            .filter(|s| s.status == SequenceStatus::Waiting)
            .count()
    }

    pub fn running_count(&self) -> usize {
        self.sequences
            .iter()
            .filter(|s| s.status == SequenceStatus::Running)
            .count()
    }

    pub fn free_blocks_count(&self) -> usize {
        self.block_pool.free_blocks_count()
    }

    pub fn has_unfinished_requests(&self) -> bool {
        self.sequences.iter().any(|s| !s.is_finished())
    }

    pub fn step(&mut self) -> Result<Vec<StepOutput>, EngineError> {
        self.current_step += 1;
        let mut outputs = Vec::new();

        // 1. Admission stage: Admit waiting requests if capacity permits
        let running_now = self.running_count();
        let available_slots = self.max_running.saturating_sub(running_now);
        let mut admitted_count = 0;

        for seq in self.sequences.iter_mut() {
            if seq.status == SequenceStatus::Waiting && admitted_count < available_slots {
                let initial_tokens = seq.prompt_tokens.len();
                let needed_blocks = (initial_tokens + self.block_size - 1) / self.block_size;

                if self.block_pool.free_blocks_count() >= needed_blocks {
                    for _ in 0..needed_blocks {
                        if let Some(blk) = self.block_pool.allocate() {
                            seq.block_table.push_block(blk);
                        }
                    }
                    seq.block_table.set_num_tokens(initial_tokens);
                    seq.status = SequenceStatus::Running;
                    admitted_count += 1;
                }
            }
        }

        // 2. Execution stage: Run forward pass for each Running request
        let mut finished_indices = Vec::new();
        self.workspace.clear();

        for (idx, seq) in self.sequences.iter_mut().enumerate() {
            if seq.status != SequenceStatus::Running {
                continue;
            }

            let logits = if seq.generated_tokens.is_empty() {
                self.model.prefill_prompt_with_workspace(
                    &seq.prompt_tokens,
                    &seq.block_table,
                    &mut self.kv_storage,
                    Some(&mut self.workspace),
                )?
            } else {
                let last_token = *seq.generated_tokens.last().unwrap();
                let pos = seq.prompt_tokens.len() + seq.generated_tokens.len() - 1;
                self.model.forward_decode_step_with_workspace(
                    last_token,
                    pos,
                    &seq.block_table,
                    &mut self.kv_storage,
                    Some(&mut self.workspace),
                )?
            };

            let next_token = sample_token(&logits, 0.0, 1.0, 42)?;

            seq.generated_tokens.push(next_token);
            seq.block_table.advance_tokens(1);

            // Check if extra block is needed
            let current_blocks = seq.block_table.blocks().len();
            let needed_blocks =
                (seq.block_table.num_tokens() + self.block_size - 1) / self.block_size;
            if needed_blocks > current_blocks {
                if let Some(blk) = self.block_pool.allocate() {
                    seq.block_table.push_block(blk);
                } else {
                    // Out of blocks: mark preempted in real system
                }
            }

            let is_finished = seq.generated_tokens.len() >= seq.max_tokens;
            if is_finished {
                seq.status = SequenceStatus::Finished;
                finished_indices.push(idx);
            }

            outputs.push(StepOutput {
                request_id: seq.id,
                token_id: next_token,
                is_finished,
            });
        }

        // 3. Retirement stage: Free KV blocks for finished sequences
        for idx in finished_indices {
            let seq = &mut self.sequences[idx];
            for &blk in seq.block_table.blocks() {
                self.block_pool.free(blk);
            }
        }

        Ok(outputs)
    }
}
