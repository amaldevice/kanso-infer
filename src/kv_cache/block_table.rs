use super::block_pool::BlockPool;

#[derive(Debug, Clone, Default)]
pub struct BlockTable {
    blocks: Vec<usize>,
    num_tokens: usize,
}

impl BlockTable {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_blocks(blocks: Vec<usize>, num_tokens: usize) -> Self {
        Self { blocks, num_tokens }
    }

    pub fn push_block(&mut self, block: usize) {
        self.blocks.push(block);
    }

    pub fn blocks(&self) -> &[usize] {
        &self.blocks
    }

    pub fn num_tokens(&self) -> usize {
        self.num_tokens
    }

    pub fn set_num_tokens(&mut self, n: usize) {
        self.num_tokens = n;
    }

    pub fn advance_tokens(&mut self, delta: usize) {
        self.num_tokens += delta;
    }

    pub fn required_blocks(&self, block_size: usize) -> usize {
        self.num_tokens.div_ceil(block_size)
    }

    pub fn rollback_blocks(
        &mut self,
        target_num_tokens: usize,
        block_size: usize,
        pool: &mut BlockPool,
    ) -> Vec<usize> {
        let needed_blocks = if target_num_tokens == 0 {
            0
        } else {
            target_num_tokens.div_ceil(block_size)
        };
        let mut freed = Vec::new();
        while self.blocks.len() > needed_blocks {
            if let Some(blk) = self.blocks.pop() {
                pool.free(blk);
                freed.push(blk);
            }
        }
        self.num_tokens = target_num_tokens;
        freed
    }
}
