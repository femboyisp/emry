//! Run engine: processors, pipeline, and `RunHandle`.

pub mod pipeline;
pub mod processor;

pub use emry_core::{EmryError, Phase, RunMeta};
pub use pipeline::{Pipeline, PipelineStats};
pub use processor::{DerivedMetric, Processor};
