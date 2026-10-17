use crate::error::EngineError;
use crate::metal::pipeline::MetalPipelineCache;
use metal::{Buffer, CommandBufferRef, CommandQueue, Device, MTLResourceOptions};
use std::sync::atomic::{AtomicUsize, Ordering};

pub struct MetalContext {
    pub device: Device,
    pub queue: CommandQueue,
    pub pipelines: MetalPipelineCache,
    /// Trace hooks (ticket #49). `command_buffers_created` counts what the engine
    /// submits, so a test can assert one command buffer per decode step;
    /// `buffers_allocated` counts device buffers, so it can assert resident weights
    /// are uploaded once instead of per call.
    command_buffers_created: AtomicUsize,
    buffers_allocated: AtomicUsize,
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
            command_buffers_created: AtomicUsize::new(0),
            buffers_allocated: AtomicUsize::new(0),
        })
    }

    /// Command buffer entry point for everything the engine submits.
    ///
    /// Encode the whole decode step into the buffer this returns and call
    /// `commit` + `wait_until_completed` once: per-kernel buffers make the CPU and
    /// GPU ping-pong on every dispatch.
    pub fn new_command_buffer(&self) -> &CommandBufferRef {
        self.command_buffers_created.fetch_add(1, Ordering::Relaxed);
        self.queue.new_command_buffer()
    }

    /// Command buffers submitted through [`Self::new_command_buffer`].
    pub fn command_buffers_created(&self) -> usize {
        self.command_buffers_created.load(Ordering::Relaxed)
    }

    /// Device buffers allocated through the `alloc_buffer*` helpers.
    pub fn buffers_allocated(&self) -> usize {
        self.buffers_allocated.load(Ordering::Relaxed)
    }

    /// Allocate a shared memory buffer on Apple Silicon Unified Memory Architecture.
    pub fn alloc_buffer<T>(&self, count: usize) -> Buffer {
        let byte_len = (count * std::mem::size_of::<T>()).max(4);
        self.buffers_allocated.fetch_add(1, Ordering::Relaxed);
        self.device
            .new_buffer(byte_len as u64, MTLResourceOptions::StorageModeShared)
    }

    /// Allocate a shared buffer holding a copy of `data`.
    ///
    /// UMA means the GPU reads it without an explicit blit, but `new_buffer_with_data`
    /// still copies the bytes once — this is an upload, so call it at load time (see
    /// `MetalWeights`), never per step. Avoiding that copy needs a page-aligned owner
    /// for `newBufferWithBytesNoCopy`, i.e. mmap'd weight files (ticket #47).
    pub fn alloc_buffer_with_data<T: Copy>(&self, data: &[T]) -> Buffer {
        let byte_len = std::mem::size_of_val(data).max(4);
        self.buffers_allocated.fetch_add(1, Ordering::Relaxed);
        self.device.new_buffer_with_data(
            data.as_ptr() as *const _,
            byte_len as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }
}
