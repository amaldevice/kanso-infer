pub mod profiler;
pub mod verifier;

pub use profiler::{SpeculativeProfiler, SpeculativeReport};
pub use verifier::{SpeculativeVerifier, VerificationResult};

use crate::engine::{Engine, StepOutput};
use crate::error::EngineError;
use crate::model::Model;
use crate::sampler::sample_token;
use crate::scheduler::SequenceStatus;

pub struct SpeculativeEngine {
    pub engine: Engine,
    pub draft_model: Option<Model>,
    pub gamma: usize,
    pub profiler: SpeculativeProfiler,
}

impl SpeculativeEngine {
    pub fn new(
        target_model: Model,
        draft_model: Option<Model>,
        max_running: usize,
        total_blocks: usize,
        block_size: usize,
        gamma: usize,
    ) -> Self {
        let engine = Engine::new(target_model, max_running, total_blocks, block_size);
        Self {
            engine,
            draft_model,
            gamma,
            profiler: SpeculativeProfiler::new(),
        }
    }

    pub fn add_request(&mut self, prompt_tokens: Vec<u32>, max_tokens: usize) -> u64 {
        self.engine.add_request(prompt_tokens, max_tokens)
    }

    pub fn has_unfinished_requests(&self) -> bool {
        self.engine.has_unfinished_requests()
    }

    pub fn waiting_count(&self) -> usize {
        self.engine.waiting_count()
    }

    pub fn running_count(&self) -> usize {
        self.engine.running_count()
    }

    pub fn free_blocks_count(&self) -> usize {
        self.engine.free_blocks_count()
    }

    pub fn step(&mut self) -> Result<Vec<StepOutput>, EngineError> {
        if self.draft_model.is_none() || self.gamma == 0 {
            return self.engine.step();
        }

        let has_prefill = self.engine.sequences.iter().any(|s| {
            s.status == SequenceStatus::Waiting
                || (s.status == SequenceStatus::Running
                    && s.prefilled_tokens < s.prompt_tokens.len())
        });
        if has_prefill {
            return self.engine.step();
        }

        let mut outputs = Vec::new();
        let running_ids: Vec<u64> = self
            .engine
            .sequences
            .iter()
            .filter(|s| s.status == SequenceStatus::Running && !s.is_finished())
            .map(|s| s.id)
            .collect();

        for req_id in running_ids {
            let seq = self
                .engine
                .sequences
                .iter()
                .find(|s| s.id == req_id)
                .unwrap();

            let mut draft_tokens = Vec::with_capacity(self.gamma);
            let mut curr = [seq.prompt_tokens.as_slice(), seq.generated_tokens.as_slice()].concat();

            let draft_model = self.draft_model.as_ref().unwrap();
            for _ in 0..self.gamma {
                let logits = draft_model.forward_single_prompt(&curr)?;
                let next_tok = sample_token(&logits, 0.0, 1.0, 42)?;
                curr.push(next_tok);
                draft_tokens.push(next_tok);
            }

            let start = std::time::Instant::now();
            let verif = self.engine.speculative_step(req_id, &draft_tokens)?;
            let duration = start.elapsed();
            self.profiler.record_step(
                draft_tokens.len(),
                verif.num_accepted,
                duration,
                verif.num_rejected > 0,
            );
            let seq_after = self
                .engine
                .sequences
                .iter()
                .find(|s| s.id == req_id)
                .unwrap();
            let is_finished = seq_after.is_finished();

            for (i, &tok) in verif.committed_tokens.iter().enumerate() {
                let tok_finished = is_finished && (i == verif.committed_tokens.len() - 1);
                outputs.push(StepOutput {
                    request_id: req_id,
                    token_id: tok,
                    is_finished: tok_finished,
                });
            }
        }

        // Retirement stage: free KV blocks for finished sequences
        for seq in self.engine.sequences.iter_mut() {
            if seq.is_finished() && !seq.block_table.blocks().is_empty() {
                for &blk in seq.block_table.blocks() {
                    self.engine.block_pool.free(blk);
                }
                seq.block_table = crate::kv_cache::BlockTable::new();
            }
        }

        Ok(outputs)
    }
}
