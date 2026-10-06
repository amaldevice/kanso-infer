use metal::{Buffer, CommandQueue, Device, MTLResourceOptions};
use crate::error::EngineError;
use crate::metal::pipeline::MetalPipelineCache;

pub struct MetalContext {
    pub device: Device,
    pub queue: CommandQueue,
    pub pipelines: MetalPipelineCache,
}

impl MetalContext {
    pub fn new() -> Result<Self, EngineError> {
        let device = Device::system_default()
            .ok_or_else(|| EngineError::Metal("No default Metal device available".into()))?;
        let queue = device.new_command_queue();
        let pipelines = MetalPipelineCache::new(&device)?;

        Ok(Self {
            device,
            queue,
            pipelines,
        })
    }

    /// Allocate a shared memory buffer on Apple Silicon Unified Memory Architecture (zero-copy).
    pub fn alloc_buffer<T>(&self, count: usize) -> Buffer {
        let byte_len = (count * std::mem::size_of::<T>()).max(4);
        self.device.new_buffer(
            byte_len as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    /// Allocate a shared memory buffer initialized with data (zero-copy UMA accessible).
    pub fn alloc_buffer_with_data<T: Copy>(&self, data: &[T]) -> Buffer {
        let byte_len = (data.len() * std::mem::size_of::<T>()).max(4);
        self.device.new_buffer_with_data(
            data.as_ptr() as *const _,
            byte_len as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }
}
