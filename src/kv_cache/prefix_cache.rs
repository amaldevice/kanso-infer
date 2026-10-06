use std::collections::HashMap;

pub struct PrefixCache {
    block_size: usize,
    cached_prefixes: HashMap<Vec<u32>, Vec<usize>>,
}

impl PrefixCache {
    pub fn new(block_size: usize) -> Self {
        Self {
            block_size,
            cached_prefixes: HashMap::new(),
        }
    }

    /// Stores prefix tokens in increments of block_size with their physical block IDs
    pub fn insert_prefix(&mut self, tokens: &[u32], blocks: &[usize]) {
        let num_complete_blocks = tokens.len() / self.block_size;
        for b in 1..=num_complete_blocks {
            let token_prefix = tokens[0..(b * self.block_size)].to_vec();
            let block_prefix = blocks[0..b].to_vec();
            self.cached_prefixes.insert(token_prefix, block_prefix);
        }
    }

    /// Finds the longest cached prefix matching the given tokens on block boundaries
    pub fn find_longest_prefix(&self, tokens: &[u32]) -> (usize, Vec<usize>) {
        let num_complete_blocks = tokens.len() / self.block_size;
        for b in (1..=num_complete_blocks).rev() {
            let slice = &tokens[0..(b * self.block_size)];
            if let Some(blocks) = self.cached_prefixes.get(slice) {
                return (b * self.block_size, blocks.clone());
            }
        }
        (0, Vec::new())
    }
}
