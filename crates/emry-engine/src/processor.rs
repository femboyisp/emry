//! The [`Processor`] trait and its derived output.
//!
//! Processors are pure-ish stream transducers: the pipeline thread feeds each
//! [`Event`] to every processor in turn, and each returns zero or more
//! [`DerivedMetric`]s (EMA, throughput, ETA, …). Processors hold their own
//! rolling state, so they take `&mut self`.

use emry_core::Event;

/// A value computed by a [`Processor`] from the event stream.
#[derive(Debug, Clone, PartialEq)]
pub struct DerivedMetric {
    /// Stable name of the derived series (e.g. `"loss_ema"`, `"steps_per_sec"`).
    pub name: String,
    /// Computed value.
    pub value: f64,
}

impl DerivedMetric {
    /// Convenience constructor.
    pub fn new(name: impl Into<String>, value: f64) -> Self {
        Self {
            name: name.into(),
            value,
        }
    }
}

/// A stateful transducer over the event stream.
///
/// Implementations run on the single pipeline thread, so they need only be
/// [`Send`] (to be moved onto that thread), not `Sync`.
pub trait Processor: Send {
    /// Handles one event, returning any derived metrics it produced.
    fn on_event(&mut self, event: &Event) -> Vec<DerivedMetric>;
}
