//! Core types, protocol, and ingest primitives for Emry.

pub mod error;
pub mod registry;
pub mod types;

pub use error::EmryError;
pub use registry::{MetricRegistry, MAX_METRICS};
pub use types::{
    AlertRecord, Event, FinishReason, MetricId, MetricRecord, Phase, RunMeta, Severity,
};
