//! Persistence: JSONL, Parquet, rotation.

pub mod sink;
pub mod writer;

#[cfg(test)]
mod test_util;

pub use emry_core::EmryError;
pub use sink::{JsonlSink, FLUSH_EVERY, FLUSH_INTERVAL};
pub use writer::{JsonlWriter, EVENTS_FILE, METRICS_FILE};
