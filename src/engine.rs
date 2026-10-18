use candle_core::Tensor;

use crate::error::EngineError;
use crate::kv_cache::{BlockPool, BlockTable, KvStorage, PrefixCache};
use crate::model::Model;
use crate::sampler::sample_logits;
use crate::scheduler::{SamplingParams, Sequence, SequenceStatus};

#[derive(Debug, Clone)]
pub struct StepOutput {
    pub request_id: u64,
    pub token_id: u32,
    pub is_finished: bool,
}

pub struct Engine {
    pub model: Model,
    pub block_pool: BlockPool,
    pub prefix_cache: PrefixCache,
    pub sequences: Vec<Sequence>,
    pub max_running: usize,
    pub block_size: usize,
    pub token_budget: usize,
    pub current_step: usize,
    pub next_request_id: u64,
    pub kv_storage: KvStorage,
    pub prefix_matched_tokens: usize,
    pub total_prompt_tokens_admitted: usize,
    pub prefix_cache_hits: usize,
    pub prefix_cache_lookups: usize,
}

impl Engine {
    pub fn new(
        model: Model,
        max_running: usize,
        total_blocks: usize,
        block_size: usize,
    ) -> Result<Self, EngineError> {
        Self::new_with_budget(model, max_running, total_blocks, block_size, usize::MAX)
    }

    pub fn new_with_budget(
        model: Model,
        max_running: usize,
        total_blocks: usize,
        block_size: usize,
        token_budget: usize,
    ) -> Result<Self, EngineError> {
        let kv_storage = KvStorage::new(
            model.config.num_hidden_layers,
            total_blocks,
            block_size,
            model.config.num_key_value_heads,
            model.config.head_dim(),
            &model.device,
        )?;

        Ok(Self {
            model,
            block_pool: BlockPool::new(total_blocks),
            prefix_cache: PrefixCache::new(block_size),
            sequences: Vec::new(),
            max_running,
            block_size,
            token_budget,
            current_step: 0,
            next_request_id: 1,
            kv_storage,
            prefix_matched_tokens: 0,
            total_prompt_tokens_admitted: 0,
            prefix_cache_hits: 0,
            prefix_cache_lookups: 0,
        })
    }
    pub fn enable_ffn_offload(&mut self) {
        self.model.enable_ffn_offload();
    }

    pub fn set_hybrid_pipeline(&mut self, pipeline: crate::ffn_offload::HybridExecutionPipeline) {
        self.model.set_hybrid_pipeline(pipeline);
    }

    pub fn add_request(&mut self, prompt_tokens: Vec<u32>, max_tokens: usize) -> u64 {
        self.add_request_with_stops(prompt_tokens, max_tokens, Vec::new())
    }

    /// Same as [`Engine::add_request`], but generation also stops on any id in `stop_token_ids`.
    pub fn add_request_with_stops(
        &mut self,
        prompt_tokens: Vec<u32>,
        max_tokens: usize,
        stop_token_ids: Vec<u32>,
    ) -> u64 {
        self.add_request_with_sampling(
            prompt_tokens,
            max_tokens,
            stop_token_ids,
            SamplingParams::default(),
        )
    }

    /// Full constructor: stop ids plus the per-request sampling knobs the engine's sample
    /// sites read (#56, CLI/server user knobs).
    pub fn add_request_with_sampling(
        &mut self,
        prompt_tokens: Vec<u32>,
        max_tokens: usize,
        stop_token_ids: Vec<u32>,
        sampling: SamplingParams,
    ) -> u64 {
        let id = self.next_request_id;
        self.next_request_id += 1;
        let mut seq = Sequence::new(id, prompt_tokens, max_tokens);
        seq.stop_token_ids = stop_token_ids;
        seq.sampling = sampling;
        self.sequences.push(seq);
        id
    }

    pub fn prefix_cache_hit_rate(&self) -> f64 {
        if self.total_prompt_tokens_admitted == 0 {
            0.0
        } else {
            (self.prefix_matched_tokens as f64 / self.total_prompt_tokens_admitted as f64).min(1.0)
        }
    }

    pub fn reset_prefix_cache_stats(&mut self) {
        self.prefix_matched_tokens = 0;
        self.total_prompt_tokens_admitted = 0;
        self.prefix_cache_hits = 0;
        self.prefix_cache_lookups = 0;
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

    /// One scheduling step: plan, run one batched forward for everything planned (#45),
    /// commit the sampled tokens, retire finished sequences.
    ///
    /// "One model forward per step" describes this method on the serving path. Speculative
    /// traffic rides [`Engine::speculative_step_sampled`] instead, which submits its
    /// candidate block as one sequence of the same batched forward (ticket #90).
    pub fn step(&mut self) -> Result<Vec<StepOutput>, EngineError> {
        self.current_step += 1;
        let mut outputs = Vec::new();
        let mut finished_indices = Vec::new();
        let mut remaining_budget = self.token_budget;

        /// Work a sequence contributes to this step: its decode token, or the
        /// prefill chunk that carries it closer to being decodable.
        struct Planned {
            idx: usize,
            start_pos: usize,
            tokens: Vec<u32>,
            is_prefill: bool,
        }

        // 1. Priority: Active decode requests execute without stalling
        let mut planned: Vec<Planned> = Vec::new();
        // Sequences already planned this step are never preemption victims: taking their
        // blocks back would drop work the step's budget already paid for and leave the
        // plan pointing at freed blocks (ticket #61 F11).
        let mut planned_flags = vec![false; self.sequences.len()];
        for idx in 0..self.sequences.len() {
            let seq = &self.sequences[idx];
            if seq.status != SequenceStatus::Running
                || seq.is_finished()
                || seq.prefilled_tokens < seq.prompt_tokens.len()
            {
                continue;
            }
            if remaining_budget == 0 {
                break;
            }

            // #56: the block need is settled before the forward runs. If the pool is
            // short, blocks are taken from the lowest-priority (newest) sequence, which
            // then recomputes; if there is nothing left to take, this sequence waits a
            // step instead of decoding past the end of its block table.
            if self.decode_needs_block(idx) && !self.ensure_blocks(idx, 1, &planned_flags) {
                continue;
            }

            let seq = &self.sequences[idx];
            let last_token = *seq.generated_tokens.last().unwrap();
            let pos = seq.prompt_tokens.len() + seq.generated_tokens.len() - 1;
            planned.push(Planned {
                idx,
                start_pos: pos,
                tokens: vec![last_token],
                is_prefill: false,
            });
            planned_flags[idx] = true;

            remaining_budget = remaining_budget.saturating_sub(1);
        }

        // 2. Admission stage: Check waiting requests & prefix cache lookup before first chunk
        let running_now = self.running_count();
        let available_slots = self.max_running.saturating_sub(running_now);
        let mut admitted_count = 0;

        for seq in self.sequences.iter_mut() {
            if seq.status == SequenceStatus::Waiting
                && admitted_count < available_slots
                && (self.block_pool.free_blocks_count() > 0 || seq.prefilled_tokens > 0)
            {
                // Prefix cache lookup runs before the first chunk is scheduled
                if seq.prefilled_tokens == 0 && seq.generated_tokens.is_empty() {
                    self.prefix_cache_lookups += 1;
                    self.total_prompt_tokens_admitted += seq.prompt_tokens.len();
                    let (matched_tokens, matched_blocks) =
                        self.prefix_cache.find_longest_prefix(&seq.prompt_tokens);
                    if matched_tokens > 0 {
                        let mut all_claimed = true;
                        let mut claimed = Vec::new();
                        for &blk in &matched_blocks {
                            if self.block_pool.claim_block(blk) {
                                claimed.push(blk);
                            } else {
                                all_claimed = false;
                                break;
                            }
                        }
                        if all_claimed {
                            for &blk in &matched_blocks {
                                seq.block_table.push_block(blk);
                            }
                            let effective_matched = if matched_tokens == seq.prompt_tokens.len() {
                                matched_tokens.saturating_sub(1)
                            } else {
                                matched_tokens
                            };
                            seq.block_table.set_num_tokens(effective_matched);
                            seq.prefilled_tokens = effective_matched;
                            self.prefix_cache_hits += 1;
                            self.prefix_matched_tokens += matched_tokens;
                        } else {
                            for blk in claimed {
                                self.block_pool.free(blk);
                            }
                        }
                    }
                }

                seq.status = SequenceStatus::Running;
                admitted_count += 1;
            }
        }

        // 3. Prefill chunks stage: Process partial prefill chunks with remaining budget
        for idx in 0..self.sequences.len() {
            let seq = &self.sequences[idx];
            if seq.status != SequenceStatus::Running
                || seq.is_finished()
                || seq.prefilled_tokens >= seq.prompt_tokens.len()
            {
                continue;
            }
            if remaining_budget == 0 {
                continue;
            }

            let needed = seq.prompt_tokens.len() - seq.prefilled_tokens;
            let chunk_len = needed.min(remaining_budget);
            if chunk_len == 0 {
                continue;
            }

            let start_pos = seq.prefilled_tokens;
            let end_pos = start_pos + chunk_len;
            // The chunk that completes the prompt also samples its first token, so
            // reserve a slot for that token before any forward runs (#56).
            let needed_blocks = (end_pos + usize::from(end_pos == seq.prompt_tokens.len()))
                .div_ceil(self.block_size);
            let current_blocks = seq.block_table.blocks().len();
            if needed_blocks > current_blocks
                && !self.ensure_blocks(idx, needed_blocks - current_blocks, &planned_flags)
            {
                continue;
            }

            let seq = &self.sequences[idx];
            planned.push(Planned {
                idx,
                start_pos,
                tokens: seq.prompt_tokens[start_pos..end_pos].to_vec(),
                is_prefill: true,
            });
            planned_flags[idx] = true;
            remaining_budget = remaining_budget.saturating_sub(chunk_len);
        }

        // Preemption skips planned sequences, so every entry above still points at a
        // running sequence with its blocks (ticket #61 F11).
        debug_assert!(planned.iter().all(|p| {
            let seq = &self.sequences[p.idx];
            seq.status == SequenceStatus::Running && !seq.is_finished()
        }));

        // 4. One batched forward for everything planned above: decode tokens and
        // prefill chunks ride the same GEMM sweep, so the weights are read once.
        let logits = if planned.is_empty() {
            None
        } else {
            let mut tokens = Vec::new();
            let mut positions = Vec::new();
            let mut slot_mapping = Vec::new();
            let mut block_tables = Vec::with_capacity(planned.len());
            let mut context_lens = Vec::with_capacity(planned.len());
            let mut query_lens = Vec::with_capacity(planned.len());
            for p in &planned {
                let table = self.sequences[p.idx].block_table.blocks();
                for (i, &token) in p.tokens.iter().enumerate() {
                    let pos = p.start_pos + i;
                    let block_idx = pos / self.block_size;
                    let block = *table.get(block_idx).ok_or_else(|| {
                        EngineError::Model(format!(
                            "Position {} requires block index {} but block table only has {} blocks",
                            pos,
                            block_idx,
                            table.len()
                        ))
                    })?;
                    tokens.push(token);
                    positions.push(pos);
                    slot_mapping.push(block * self.block_size + pos % self.block_size);
                }
                block_tables.push(table.to_vec());
                context_lens.push(p.start_pos + p.tokens.len());
                query_lens.push(p.tokens.len());
            }
            Some(self.model.forward_batch(
                &tokens,
                &positions,
                &slot_mapping,
                &mut self.kv_storage,
                &block_tables,
                &context_lens,
                &query_lens,
            )?)
        };

        // 5. Commit stage: sample each planned sequence's next token from its logits row
        // and advance it — in plan order, so decode outputs precede prefill completions.
        //
        // Only rows that actually sample are gathered, and they cross to the host in one
        // transfer (ticket #61 F11): a partial prefill chunk's row is discarded, and one
        // `to_vec1` per sequence cost one device read per sequence.
        let sampling_rows: Vec<u32> = planned
            .iter()
            .enumerate()
            .filter(|(_, p)| {
                !p.is_prefill
                    || p.start_pos + p.tokens.len() == self.sequences[p.idx].prompt_tokens.len()
            })
            .map(|(row, _)| row as u32)
            .collect();
        let sampling_rows_count = sampling_rows.len();
        let logits_host: Option<Vec<Vec<f32>>> = match &logits {
            Some(l) => {
                let rows =
                    Tensor::from_vec(sampling_rows, (sampling_rows_count,), &self.model.device)?;
                Some(l.index_select(&rows, 0)?.to_vec2::<f32>()?)
            }
            None => None,
        };

        let mut sample_row = 0usize;
        for p in planned.iter() {
            let seq = &mut self.sequences[p.idx];

            if p.is_prefill {
                let end_pos = p.start_pos + p.tokens.len();
                seq.prefilled_tokens = end_pos;
                seq.block_table.set_num_tokens(end_pos);
                // A partial chunk produces no token; only a completed prompt samples.
                if end_pos != seq.prompt_tokens.len() {
                    continue;
                }
            }

            let row_logits = logits_host
                .as_ref()
                .and_then(|rows| rows.get(sample_row))
                .ok_or_else(|| {
                    EngineError::Model(format!(
                        "step: sequence {} has no logits row {sample_row} to sample from",
                        seq.id
                    ))
                })?;
            let next_token = sample_logits(
                row_logits,
                seq.sampling.temperature,
                seq.sampling.top_p,
                seq.sampling.seed,
            )?;
            sample_row += 1;
            seq.generated_tokens.push(next_token);
            seq.block_table.advance_tokens(1);

            // #56: EOS (any configured stop id) retires the sequence exactly like
            // max_tokens, so its blocks return to the pool in this same step.
            let is_finished = seq.is_done_after(next_token);
            if is_finished {
                seq.status = SequenceStatus::Finished;
                finished_indices.push(p.idx);
            }

            outputs.push(StepOutput {
                request_id: seq.id,
                token_id: next_token,
                is_finished,
            });
        }

        // Insert completed prompts into the prefix cache.
        for p in planned.iter().filter(|p| p.is_prefill) {
            let seq = &self.sequences[p.idx];
            if seq.prefilled_tokens == seq.prompt_tokens.len() {
                self.prefix_cache
                    .insert_prefix(&seq.prompt_tokens, seq.block_table.blocks());
            }
        }

        // 6. Retirement stage: Free KV blocks for finished sequences
        for idx in finished_indices {
            let seq = &mut self.sequences[idx];
            for &blk in seq.block_table.blocks() {
                self.block_pool.free(blk);
            }
        }

        Ok(outputs)
    }

    /// #56: drops finished requests from the active vector so a long-lived engine
    /// (server, CLI) does not rescan unbounded history every step. Returns how many
    /// were dropped. Finished sequences stay visible until the caller (chat/CLI/server)
    /// calls this; their KV blocks are already back in the pool by then.
    pub fn drain_finished(&mut self) -> usize {
        let before = self.sequences.len();
        self.sequences.retain(|s| !s.is_finished());
        before - self.sequences.len()
    }

    /// Retires a request whose caller is gone (client disconnect, ticket #61 F1): its KV
    /// blocks go back to the pool now instead of after `max_tokens` more steps of work
    /// nobody reads. `false` = no such request is live.
    pub fn cancel_request(&mut self, request_id: u64) -> bool {
        let Some(seq) = self.sequences.iter_mut().find(|s| s.id == request_id) else {
            return false;
        };
        for &blk in seq.block_table.blocks() {
            self.block_pool.free(blk);
        }
        seq.status = SequenceStatus::Finished;
        self.drain_finished();
        true
    }

    /// Would the next decoded token spill past the sequence's allocated blocks?
    fn decode_needs_block(&self, idx: usize) -> bool {
        let table = &self.sequences[idx].block_table;
        table.num_tokens() + 1 > table.blocks().len() * self.block_size
    }

    /// #56: grows sequence `idx`'s block table by `want` blocks, taking them from the
    /// lowest-priority (newest) running sequence when the pool is short — that victim is
    /// preempted and recomputes later. `false` = there was nothing left to take, so the
    /// caller must defer the sequence instead of running it outside its block table.
    ///
    /// `planned` marks the sequences this step has already planned (ticket #61 F11): they
    /// are not valid victims, because the step's budget was already spent on their forward.
    fn ensure_blocks(&mut self, idx: usize, want: usize, planned: &[bool]) -> bool {
        while self.block_pool.free_blocks_count() < want {
            if !self.preempt_lowest_priority_after(idx, planned) {
                return false;
            }
        }
        let seq = &mut self.sequences[idx];
        for _ in 0..want {
            let Some(blk) = self.block_pool.allocate() else {
                // Unreachable: the loop above just secured `want` free blocks.
                return false;
            };
            // #44: block is about to be overwritten — drop stale prefix entries.
            self.prefix_cache.invalidate_block(blk);
            seq.block_table.push_block(blk);
        }
        true
    }

    /// Picks the newest running sequence after `idx` (lowest priority) that this step has
    /// not planned yet and preempts it.
    fn preempt_lowest_priority_after(&mut self, idx: usize, planned: &[bool]) -> bool {
        let victim = (idx + 1..self.sequences.len()).rev().find(|&j| {
            !planned[j]
                && self.sequences[j].status == SequenceStatus::Running
                && !self.sequences[j].is_finished()
        });
        match victim {
            Some(v) => {
                self.preempt(v);
                true
            }
            None => false,
        }
    }

    /// Recompute preemption (#56): release the sequence's KV blocks and fold the tokens it
    /// already emitted back into its prompt, so it re-prefills from a clean block table when
    /// rescheduled. Already-reported tokens are not replayed — `max_tokens` shrinks by the
    /// folded count, so the caller still observes exactly `max_tokens` outputs.
    fn preempt(&mut self, idx: usize) {
        let seq = &mut self.sequences[idx];
        for &blk in seq.block_table.blocks() {
            self.block_pool.free(blk);
        }
        let emitted = seq.generated_tokens.len();
        seq.prompt_tokens.append(&mut seq.generated_tokens);
        seq.max_tokens = seq.max_tokens.saturating_sub(emitted);
        seq.block_table = BlockTable::new();
        seq.prefilled_tokens = 0;
        seq.status = SequenceStatus::Waiting;
    }

    /// Greedy (temperature 0) speculation: candidates are verified by argmax match.
    pub fn speculative_step(
        &mut self,
        request_id: u64,
        draft_candidates: &[u32],
    ) -> Result<crate::speculative::VerificationResult, EngineError> {
        self.speculative_step_sampled(request_id, draft_candidates, &[], SamplingParams::default())
    }

    /// Full speculation step. `draft_probs[i]` is the distribution the draft drew
    /// `draft_candidates[i]` from — required when `sampling.temperature > 0`, where the
    /// candidates are verified by rejection sampling instead of argmax match (#55).
    ///
    /// The candidates are verified in one batched forward: the `[current, candidates...]`
    /// block rides `Model::forward_batch` as a single sequence (ticket #90), so the step
    /// costs one forward and `Model::forward_batch_calls` counts it. The draft still pays
    /// one `forward_decode_step` per proposal on its own KV cache before this call.
    pub fn speculative_step_sampled(
        &mut self,
        request_id: u64,
        draft_candidates: &[u32],
        draft_probs: &[Vec<f32>],
        sampling: SamplingParams,
    ) -> Result<crate::speculative::VerificationResult, EngineError> {
        let seq_idx = self
            .sequences
            .iter()
            .position(|s| s.id == request_id)
            .ok_or_else(|| EngineError::Model(format!("Request {} not found", request_id)))?;

        let seq = &self.sequences[seq_idx];
        if seq.status != SequenceStatus::Running || seq.is_finished() {
            return Err(EngineError::Model(format!(
                "Sequence {} is not running or already finished",
                request_id
            )));
        }
        if seq.prefilled_tokens < seq.prompt_tokens.len() || seq.generated_tokens.is_empty() {
            return Err(EngineError::Model(format!(
                "Sequence {} has not completed prefill yet",
                request_id
            )));
        }

        let last_token = *seq.generated_tokens.last().unwrap();
        let start_pos = seq.prompt_tokens.len() + seq.generated_tokens.len() - 1;

        let seq_mut = &mut self.sequences[seq_idx];
        let verify_result = crate::speculative::SpeculativeVerifier::verify(
            &self.model,
            last_token,
            draft_candidates,
            draft_probs,
            sampling,
            start_pos,
            &mut seq_mut.block_table,
            &mut self.block_pool,
            &mut self.prefix_cache,
            &mut self.kv_storage,
            self.block_size,
        )?;

        let seq_mut = &mut self.sequences[seq_idx];
        let mut committed = Vec::new();
        for &tok in &verify_result.committed_tokens {
            if seq_mut.generated_tokens.len() < seq_mut.max_tokens {
                seq_mut.generated_tokens.push(tok);
                committed.push(tok);
                if seq_mut.stop_token_ids.contains(&tok) {
                    break; // #56: EOS — never commit draft tokens past the stop token
                }
            }
        }

        let total_seq_tokens = seq_mut.prompt_tokens.len() + seq_mut.generated_tokens.len();
        seq_mut.block_table.rollback_blocks(
            total_seq_tokens,
            self.block_size,
            &mut self.block_pool,
        );

        // #56/F1: same termination rule as the decode/prefill paths.
        let done = seq_mut
            .generated_tokens
            .last()
            .is_some_and(|&tok| seq_mut.is_done_after(tok));
        if done {
            seq_mut.status = SequenceStatus::Finished;
            // #56/F3: a speculative finish never reaches stage-4 retirement, so the
            // blocks would otherwise stay out of the pool until process end.
            for &blk in seq_mut.block_table.blocks() {
                self.block_pool.free(blk);
            }
        }

        let mut final_result = verify_result;
        final_result.committed_tokens = committed;
        Ok(final_result)
    }
}
