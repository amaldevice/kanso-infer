use std::fmt;

#[derive(Debug)]
pub enum EngineError {
    Io(std::io::Error),
    Candle(candle_core::Error),
    Config(String),
    Scheduler(String),
    Model(String),
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {}", e),
            Self::Candle(e) => write!(f, "Candle error: {}", e),
            Self::Config(e) => write!(f, "Config error: {}", e),
            Self::Scheduler(e) => write!(f, "Scheduler error: {}", e),
            Self::Model(e) => write!(f, "Model error: {}", e),
        }
    }
}

impl std::error::Error for EngineError {}

impl From<std::io::Error> for EngineError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<candle_core::Error> for EngineError {
    fn from(e: candle_core::Error) -> Self {
        Self::Candle(e)
    }
}
