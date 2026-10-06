pub fn cpu_rmsnorm(input: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let dim = input.len();
    let sum_sq: f32 = input.iter().map(|&x| x * x).sum();
    let mean_sq = sum_sq / (dim as f32);
    let rms = (mean_sq + eps).sqrt();

    input
        .iter()
        .zip(weight.iter())
        .map(|(&x, &w)| (x / rms) * w)
        .collect()
}

pub fn cpu_silu_gate_multiply(gate: &[f32], up: &[f32]) -> Vec<f32> {
    gate.iter()
        .zip(up.iter())
        .map(|(&g, &u)| {
            let silu = g / (1.0 + (-g).exp());
            silu * u
        })
        .collect()
}
