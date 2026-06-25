//! Intern table mapping metric names to compact [`MetricId`] handles.
//!
//! Metrics are registered once at run start; the hot path then emits values by
//! [`MetricId`] (a `u16`) instead of re-hashing strings on every step. The
//! registry is the single source of truth for the name ↔ id mapping and backs
//! both the engine's fast `emit` path and name resolution for `metrics.jsonl`.

use crate::types::MetricId;
use std::collections::HashMap;

/// Maximum number of distinct metrics, bounded by [`MetricId`]'s `u16` index
/// (ids `0..=65_535`).
pub const MAX_METRICS: usize = u16::MAX as usize + 1;

/// Append-only intern table for metric names.
///
/// Registration is idempotent: registering a name that already exists returns
/// its existing [`MetricId`], so callers can register defensively without
/// fragmenting ids. Ids are assigned densely starting at `0` and never reused.
#[derive(Debug, Default, Clone)]
pub struct MetricRegistry {
    names: Vec<String>,
    ids: HashMap<String, MetricId>,
}

impl MetricRegistry {
    /// Creates an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers `name`, returning its stable [`MetricId`].
    ///
    /// Returns the existing id if `name` was already registered.
    ///
    /// # Panics
    ///
    /// Panics if more than [`MAX_METRICS`] distinct names are registered. Use
    /// [`MetricRegistry::try_register`] to handle the limit without panicking.
    pub fn register(&mut self, name: &str) -> MetricId {
        self.try_register(name)
            .expect("metric registry exceeded MAX_METRICS (65 536 distinct metrics)")
    }

    /// Registers `name`, returning its stable [`MetricId`], or `None` if the
    /// registry is full ([`MAX_METRICS`] distinct names already registered).
    ///
    /// Returns the existing id if `name` was already registered.
    pub fn try_register(&mut self, name: &str) -> Option<MetricId> {
        if let Some(&id) = self.ids.get(name) {
            return Some(id);
        }
        // `names.len()` is the next id. It fits in u16 for the first 65_536
        // registrations (ids 0..=65_535); once the table is full (len == 65_536)
        // the conversion fails and we return None — this is the only overflow
        // guard, not dead code.
        let id = MetricId(u16::try_from(self.names.len()).ok()?);
        self.names.push(name.to_owned());
        self.ids.insert(name.to_owned(), id);
        Some(id)
    }

    /// Resolves a [`MetricId`] back to its registered name, or `None` if the id
    /// was never registered in this registry.
    #[must_use]
    pub fn name(&self, id: MetricId) -> Option<&str> {
        self.names.get(id.index() as usize).map(String::as_str)
    }

    /// Returns the id for an already-registered `name`, without registering it.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<MetricId> {
        self.ids.get(name).copied()
    }

    /// Number of registered metrics.
    #[must_use]
    pub fn len(&self) -> usize {
        self.names.len()
    }

    /// Whether no metrics have been registered yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_registry_is_empty() {
        let reg = MetricRegistry::new();
        assert!(reg.is_empty());
        assert_eq!(reg.len(), 0);
    }

    #[test]
    fn ids_are_assigned_densely_from_zero() {
        let mut reg = MetricRegistry::new();
        assert_eq!(reg.register("loss"), MetricId(0));
        assert_eq!(reg.register("lr"), MetricId(1));
        assert_eq!(reg.register("grad_norm"), MetricId(2));
        assert_eq!(reg.len(), 3);
    }

    #[test]
    fn duplicate_register_returns_same_id() {
        let mut reg = MetricRegistry::new();
        let first = reg.register("loss");
        let other = reg.register("lr");
        let again = reg.register("loss");
        assert_eq!(first, again);
        assert_ne!(first, other);
        assert_eq!(reg.len(), 2, "re-registering must not grow the table");
    }

    #[test]
    fn name_resolves_registered_id() {
        let mut reg = MetricRegistry::new();
        let id = reg.register("throughput");
        assert_eq!(reg.name(id), Some("throughput"));
        assert_eq!(reg.get("throughput"), Some(id));
    }

    #[test]
    fn unknown_id_and_name_resolve_to_none() {
        let reg = MetricRegistry::new();
        assert_eq!(reg.name(MetricId(42)), None);
        assert_eq!(reg.get("nope"), None);
    }

    #[test]
    fn try_register_returns_none_when_full() {
        let mut reg = MetricRegistry::new();
        for i in 0..MAX_METRICS {
            assert!(reg.try_register(&format!("m{i}")).is_some());
        }
        assert_eq!(reg.len(), MAX_METRICS);
        // The table is full: a new name has nowhere to go.
        assert_eq!(reg.try_register("overflow"), None);
        // But an already-registered name still resolves.
        assert_eq!(reg.try_register("m0"), Some(MetricId(0)));
    }

    #[test]
    #[should_panic(expected = "exceeded MAX_METRICS")]
    fn register_panics_when_full() {
        let mut reg = MetricRegistry::new();
        for i in 0..MAX_METRICS {
            reg.try_register(&format!("m{i}"));
        }
        reg.register("one too many");
    }
}
