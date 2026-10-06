use crate::config::ModelConfig;
use crate::kv_cache::BlockPool;

pub struct SparseBlockTable {
    attention_layers: Vec<usize>,
    layer_blocks: Vec<Vec<usize>>,
}

impl SparseBlockTable {
    pub fn allocate_initial(
        config: &ModelConfig,
        pool: &mut BlockPool,
        num_tokens: usize,
    ) -> Self {
        let attention_layers = match &config.attention_layers {
            Some(layers) => layers.clone(),
            None => (0..config.num_hidden_layers).collect(),
        };

        let blocks_per_layer = (num_tokens + config.block_size - 1) / config.block_size;
        let mut layer_blocks = Vec::with_capacity(attention_layers.len());

        for _ in 0..attention_layers.len() {
            let mut blks = Vec::with_capacity(blocks_per_layer);
            for _ in 0..blocks_per_layer {
                if let Some(b) = pool.allocate() {
                    blks.push(b);
                }
            }
            layer_blocks.push(blks);
        }

        Self {
            attention_layers,
            layer_blocks,
        }
    }

    pub fn attention_layers(&self) -> &[usize] {
        &self.attention_layers
    }

    pub fn total_allocated_blocks(&self) -> usize {
        self.layer_blocks.iter().map(|b| b.len()).sum()
    }
}
