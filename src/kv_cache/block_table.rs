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
        (self.num_tokens + block_size - 1) / block_size
    }
}
