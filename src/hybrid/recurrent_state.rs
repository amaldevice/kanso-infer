pub struct RecurrentState {
    states: Vec<Vec<f32>>,
    dim: usize,
}

impl RecurrentState {
    pub fn new(num_layers: usize, state_dim: usize) -> Self {
        Self {
            states: vec![vec![0.0f32; state_dim]; num_layers],
            dim: state_dim,
        }
    }

    /// In-place rolling update: state_{t} = 0.9 * state_{t-1} + 0.1 * input
    pub fn step_update(&mut self, layer: usize, input: &[f32]) {
        if layer < self.states.len() {
            let state = &mut self.states[layer];
            for (s, &inp) in state.iter_mut().zip(input.iter()) {
                *s = *s * 0.9 + inp * 0.1;
            }
        }
    }

    pub fn state_bytes(&self) -> usize {
        self.states.len() * self.dim * std::mem::size_of::<f32>()
    }
}
