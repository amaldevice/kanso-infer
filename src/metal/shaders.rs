use std::collections::HashMap;

pub const RMSNORM_SHADER: &str = include_str!("kernels/rmsnorm.metal");
pub const ROPE_SHADER: &str = include_str!("kernels/rope.metal");
pub const SILU_GATE_SHADER: &str = include_str!("kernels/silu_gate.metal");
pub const PAGED_KV_SHADER: &str = include_str!("kernels/paged_kv.metal");
pub const PAGED_ATTENTION_V2_SHADER: &str = include_str!("kernels/paged_attention_v2.metal");

pub fn get_all_shaders() -> HashMap<&'static str, &'static str> {
    let mut map = HashMap::new();
    map.insert("rmsnorm.metal", RMSNORM_SHADER);
    map.insert("rope.metal", ROPE_SHADER);
    map.insert("silu_gate.metal", SILU_GATE_SHADER);
    map.insert("paged_kv.metal", PAGED_KV_SHADER);
    map.insert("paged_attention_v2.metal", PAGED_ATTENTION_V2_SHADER);
    map
}
