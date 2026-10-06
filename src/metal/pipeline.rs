use std::collections::HashMap;
use metal::{CompileOptions, ComputePipelineState, Device};
use crate::error::EngineError;
use crate::metal::shaders;

pub struct MetalPipelineCache {
    pipelines: HashMap<String, ComputePipelineState>,
}

impl MetalPipelineCache {
    pub fn new(device: &Device) -> Result<Self, EngineError> {
        let mut pipelines = HashMap::new();
        let compile_options = CompileOptions::new();

        let shader_defs = [
            (shaders::RMSNORM_SHADER, "rmsnorm_kernel", "rmsnorm"),
            (shaders::ROPE_SHADER, "rope_kernel", "rope"),
            (shaders::SILU_GATE_SHADER, "silu_gate_kernel", "silu_gate"),
            (shaders::PAGED_KV_SHADER, "paged_kv_gather_kernel", "paged_kv"),
            (shaders::PAGED_ATTENTION_V2_SHADER, "paged_attention_v2_kernel", "paged_attention_v2"),
        ];
        for (source, kernel_name, key) in shader_defs {
            let library = device
                .new_library_with_source(source, &compile_options)
                .map_err(|e| EngineError::Metal(format!("Failed to compile shader {key}: {e}")))?;
            let function = library
                .get_function(kernel_name, None)
                .map_err(|e| EngineError::Metal(format!("Failed to get kernel {kernel_name} in {key}: {e}")))?;
            let pipeline = device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|e| EngineError::Metal(format!("Failed to create pipeline state for {key}: {e}")))?;
            pipelines.insert(key.to_string(), pipeline);
        }

        Ok(Self { pipelines })
    }

    pub fn get(&self, name: &str) -> Option<&ComputePipelineState> {
        self.pipelines.get(name)
    }

    pub fn contains(&self, name: &str) -> bool {
        self.pipelines.contains_key(name)
    }

    pub fn len(&self) -> usize {
        self.pipelines.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pipelines.is_empty()
    }
}
