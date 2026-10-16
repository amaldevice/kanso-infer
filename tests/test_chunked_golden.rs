//! Chunked-vs-monolithic golden for #57 — teeth for #43 (chunk amnesia).
//! Compares `forward_prefill` in chunks of 4/7/16 against the monolithic
//! `forward_tokens` reference on the fixed-seed random mock.
//! EXPECTED TO FAIL until #43 is fixed (chunked prefill ignores prior-chunk
//! context: causal mask is chunk-local and attention never gathers earlier K/V).
//! DO NOT #[ignore] this test: it must stay red as proof the suite can see #43.

use kanso_infer::kv_cache::{BlockTable, KvStorage};
use kanso_infer::model::Model;

mod common;
use common::{last_token_logits, max_abs_diff};

const TOL: f32 = 1e-4;

fn mock() -> Model {
    Model::create_mock_tiny(16, 4, 2, 2, 50).unwrap()
}

fn fresh_cache(model: &Model, num_blocks: usize) -> (BlockTable, KvStorage) {
    let mut table = BlockTable::new();
    for b in 0..num_blocks {
        table.push_block(b);
    }
    let kv = KvStorage::new(
        model.config.num_hidden_layers,
        num_blocks,
        model.config.block_size,
        model.config.num_key_value_heads,
        model.config.head_dim(),
        &model.device,
    )
    .unwrap();
    (table, kv)
}

/// Full-prompt logits via sequential `forward_prefill` chunks of `chunk` tokens.
fn chunked_last_token_logits(model: &Model, prompt: &[u32], chunk: usize) -> Vec<f32> {
    let num_blocks = prompt.len().div_ceil(model.config.block_size) + 1;
    let (table, mut kv) = fresh_cache(model, num_blocks);
    let mut last: Vec<f32> = Vec::new();
    let mut pos = 0;
    while pos < prompt.len() {
        let end = (pos + chunk).min(prompt.len());
        let logits = model
            .forward_prefill(&prompt[pos..end], pos, &table, &mut kv)
            .unwrap();
        last = logits.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        pos = end;
    }
    last
}

#[test]
fn test_chunked_prefill_matches_monolithic_golden() {
    let model = mock();
    let prompt: Vec<u32> = vec![3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 7, 9, 12, 20, 33, 44];
    let reference = last_token_logits(&model, &prompt);
    for chunk in [4usize, 7, 16] {
        let got = chunked_last_token_logits(&model, &prompt, chunk);
        let d = max_abs_diff(&reference, &got);
        assert!(
            d <= TOL,
            "KNOWN-FAIL #43: chunk size {chunk}: chunked prefill diverged from monolithic \
             (max abs diff = {d}, tol = {TOL}); chunk ignores prior context"
        );
    }
}
