//! Core types, protocol, and ingest primitives for Emry.

pub mod error;
pub mod types;

pub use error::EmryError;
pub use types::{
    AlertRecord, Event, FinishReason, MetricId, MetricRecord, Phase, RunMeta, Severity,
};
