// mini_engine library root
pub mod accelerate;
pub mod ane;
pub mod bench;
pub mod chat;
pub mod config;
pub mod engine;
pub mod error;
pub mod hardware;
pub mod hybrid;
pub mod kv_cache;
pub mod metal;
pub mod model;
pub mod quant;
pub mod sampler;
pub mod scheduler;
pub mod server;
pub mod speculative;
pub mod tokenizer;
pub mod weights;

pub use hardware::DEFAULT_BACKEND;
