use crate::error::EngineError;
use crate::metal::shaders;
use metal::{CompileOptions, ComputePipelineState, Device, MTLLanguageVersion};
use std::collections::HashMap;

/// AOT `.metallib` bytes for each kernel source, written by `build.rs`.
///
/// Absent when the build machine has no Metal compiler (Command Line Tools only):
/// the pipeline cache then compiles each file from source exactly once.
#[cfg(metal_aot)]
mod aot {
    pub fn bytes(file: &str) -> Option<&'static [u8]> {
        Some(match file {
            "rmsnorm.metal" => {
                &include_bytes!(concat!(env!("OUT_DIR"), "/kernels/rmsnorm.metallib"))[..]
            }
            "rope.metal" => &include_bytes!(concat!(env!("OUT_DIR"), "/kernels/rope.metallib"))[..],
            "silu_gate.metal" => {
                &include_bytes!(concat!(env!("OUT_DIR"), "/kernels/silu_gate.metallib"))[..]
            }
            "paged_kv.metal" => {
                &include_bytes!(concat!(env!("OUT_DIR"), "/kernels/paged_kv.metallib"))[..]
            }
            "paged_attention_v2.metal" => &include_bytes!(concat!(
                env!("OUT_DIR"),
                "/kernels/paged_attention_v2.metallib"
            ))[..],
            "fused_llama_layer.metal" => &include_bytes!(concat!(
                env!("OUT_DIR"),
                "/kernels/fused_llama_layer.metallib"
            ))[..],
            _ => return None,
        })
    }
}

#[cfg(not(metal_aot))]
mod aot {
    pub fn bytes(_file: &str) -> Option<&'static [u8]> {
        None
    }
}

/// One `.metal` file and the kernels the engine needs out of it. Grouping by file is
/// what keeps `fused_llama_layer.metal` from being compiled once per kernel (ticket
/// #49: it used to be compiled three times).
#[allow(dead_code)] // `source` is unread when every library comes from an AOT metallib
struct KernelSource {
    file: &'static str,
    source: &'static str,
    kernels: &'static [(&'static str, &'static str)],
}

const KERNEL_SOURCES: &[KernelSource] = &[
    KernelSource {
        file: "rmsnorm.metal",
        source: shaders::RMSNORM_SHADER,
        kernels: &[("rmsnorm_kernel", "rmsnorm")],
    },
    KernelSource {
        file: "rope.metal",
        source: shaders::ROPE_SHADER,
        kernels: &[("rope_kernel", "rope")],
    },
    KernelSource {
        file: "silu_gate.metal",
        source: shaders::SILU_GATE_SHADER,
        kernels: &[("silu_gate_kernel", "silu_gate")],
    },
    KernelSource {
        file: "paged_kv.metal",
        source: shaders::PAGED_KV_SHADER,
        kernels: &[("paged_kv_gather_kernel", "paged_kv")],
    },
    KernelSource {
        file: "paged_attention_v2.metal",
        source: shaders::PAGED_ATTENTION_V2_SHADER,
        kernels: &[
            ("paged_attention_v2_kernel", "paged_attention_v2"),
            (
                "paged_attention_v2_reduce_kernel",
                "paged_attention_v2_reduce",
            ),
        ],
    },
    KernelSource {
        file: "fused_llama_layer.metal",
        source: shaders::FUSED_LLAMA_LAYER_SHADER,
        kernels: &[
            ("fused_rmsnorm_rope_kernel", "fused_rmsnorm_rope"),
            ("fused_rmsnorm_qkv_rope_kernel", "fused_rmsnorm_qkv_rope"),
            ("linear_gemv_kernel", "linear_gemv"),
        ],
    },
];

pub struct MetalPipelineCache {
    pipelines: HashMap<String, ComputePipelineState>,
    /// Shader files compiled from source at startup (0 when AOT metallibs are used).
    source_compiles: usize,
    /// Shader files loaded from an AOT metallib (0 on the source-compile fallback).
    aot_libraries: usize,
}

impl MetalPipelineCache {
    pub fn new(device: &Device) -> Result<Self, EngineError> {
        // Explicit options rather than the framework defaults: fast math matches the
        // `-ffast-math` the AOT metallibs are built with, and the language version is
        // pinned so a toolchain default bump cannot silently change codegen.
        let compile_options = CompileOptions::new();
        compile_options.set_fast_math_enabled(true);
        compile_options.set_language_version(MTLLanguageVersion::V2_4);

        let mut pipelines = HashMap::new();
        let mut source_compiles = 0;
        let mut aot_libraries = 0;

        for kernel_source in KERNEL_SOURCES {
            let file = kernel_source.file;
            let library = match aot::bytes(file) {
                Some(data) => {
                    aot_libraries += 1;
                    device.new_library_with_data(data).map_err(|e| {
                        EngineError::Metal(format!("Failed to load metallib for {file}: {e}"))
                    })?
                }
                None => {
                    source_compiles += 1;
                    device
                        .new_library_with_source(kernel_source.source, &compile_options)
                        .map_err(|e| {
                            EngineError::Metal(format!("Failed to compile shader {file}: {e}"))
                        })?
                }
            };

            for (function_name, key) in kernel_source.kernels {
                let function = library.get_function(function_name, None).map_err(|e| {
                    EngineError::Metal(format!(
                        "Failed to get kernel {function_name} in {file}: {e}"
                    ))
                })?;
                let pipeline = device
                    .new_compute_pipeline_state_with_function(&function)
                    .map_err(|e| {
                        EngineError::Metal(format!(
                            "Failed to create pipeline state for {key}: {e}"
                        ))
                    })?;
                pipelines.insert((*key).to_string(), pipeline);
            }
        }

        Ok(Self {
            pipelines,
            source_compiles,
            aot_libraries,
        })
    }

    /// Shader files this cache had to compile from source at startup.
    pub fn source_compiles(&self) -> usize {
        self.source_compiles
    }

    /// Shader files this cache loaded from an AOT `.metallib`.
    pub fn aot_libraries(&self) -> usize {
        self.aot_libraries
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
