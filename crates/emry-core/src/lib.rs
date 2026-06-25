//! Core types, protocol, and ingest primitives for Emry.

pub mod error;
pub mod mode;
pub mod registry;
pub mod ring;
pub mod types;

pub use error::EmryError;
pub use mode::{DeployEnv, DeployMode, ParseDeployModeError};
pub use registry::{MetricRegistry, MAX_METRICS};
pub use ring::{event_ring, event_ring_with_capacity, EventConsumer, EventProducer, RingFull};
pub use types::{
    AlertRecord, Event, FinishReason, MetricId, MetricRecord, Phase, RunMeta, Severity,
};
