//! Synchronous JSONL writer for a run directory.
//!
//! [`JsonlWriter`] owns the two append-only logs of a run:
//!
//! - `events.jsonl` — the full [`Event`] stream (the audit trail), one
//!   adjacently-tagged JSON object per line.
//! - `metrics.jsonl` — wide [`MetricRecord`] rows for export. This file must stay
//!   parseable by external / v1 JSONL readers, so it carries resolved metric
//!   names (not [`MetricId`](emry_core::MetricId)) and a screaming-snake `phase`.
//!
//! This type is deliberately synchronous and side-effect-explicit so it can be
//! tested by reading the files straight back. The background batching/flush
//! thread lives in [`crate::sink`].

use emry_core::{EmryError, Event, MetricRecord};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

/// File name of the full event audit log within a run directory.
pub const EVENTS_FILE: &str = "events.jsonl";
/// File name of the wide metric-row log within a run directory.
pub const METRICS_FILE: &str = "metrics.jsonl";

/// Writes a run's `events.jsonl` and `metrics.jsonl`.
#[derive(Debug)]
pub struct JsonlWriter {
    events: BufWriter<File>,
    metrics: BufWriter<File>,
}

impl JsonlWriter {
    /// Creates (or truncates) both JSONL files in `dir`, which must already
    /// exist.
    ///
    /// # Errors
    ///
    /// Returns [`EmryError::Io`] if either file cannot be created.
    pub fn create(dir: &Path) -> Result<Self, EmryError> {
        Ok(Self {
            events: BufWriter::new(File::create(dir.join(EVENTS_FILE))?),
            metrics: BufWriter::new(File::create(dir.join(METRICS_FILE))?),
        })
    }

    /// Appends one [`Event`] to `events.jsonl` as a single JSON line.
    ///
    /// # Errors
    ///
    /// Returns [`EmryError::Json`] on serialization failure or [`EmryError::Io`]
    /// on write failure.
    pub fn write_event(&mut self, event: &Event) -> Result<(), EmryError> {
        write_line(&mut self.events, event)
    }

    /// Appends one [`MetricRecord`] to `metrics.jsonl` as a single JSON line.
    ///
    /// # Errors
    ///
    /// Returns [`EmryError::Json`] on serialization failure or [`EmryError::Io`]
    /// on write failure.
    pub fn write_metric(&mut self, record: &MetricRecord) -> Result<(), EmryError> {
        write_line(&mut self.metrics, record)
    }

    /// Flushes both buffered files to the OS.
    ///
    /// # Errors
    ///
    /// Returns [`EmryError::Io`] if either flush fails.
    pub fn flush(&mut self) -> Result<(), EmryError> {
        self.events.flush()?;
        self.metrics.flush()?;
        Ok(())
    }
}

/// Serializes `value` as one JSON object followed by a newline.
fn write_line<T: serde::Serialize>(out: &mut BufWriter<File>, value: &T) -> Result<(), EmryError> {
    let line = serde_json::to_string(value)?;
    out.write_all(line.as_bytes())?;
    out.write_all(b"\n")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::TempRunDir;
    use emry_core::{FinishReason, MetricId, Phase};
    use std::collections::BTreeMap;
    use std::fs;

    fn record(step: u64, loss: f64) -> MetricRecord {
        let mut values = BTreeMap::new();
        values.insert("loss".to_string(), loss);
        MetricRecord {
            step,
            epoch: 0,
            phase: Phase::Train,
            values,
        }
    }

    #[test]
    fn writes_events_and_metrics_to_separate_files() {
        let dir = TempRunDir::new();
        let mut w = JsonlWriter::create(dir.path()).unwrap();
        w.write_event(&Event::PhaseChange(Phase::Train)).unwrap();
        w.write_event(&Event::RunFinished {
            duration_secs: 1.0,
            reason: FinishReason::Completed,
        })
        .unwrap();
        w.write_metric(&record(0, 0.5)).unwrap();
        w.write_metric(&record(1, 0.4)).unwrap();
        w.flush().unwrap();

        let events = fs::read_to_string(dir.path().join(EVENTS_FILE)).unwrap();
        let metrics = fs::read_to_string(dir.path().join(METRICS_FILE)).unwrap();
        assert_eq!(events.lines().count(), 2);
        assert_eq!(metrics.lines().count(), 2);
    }

    #[test]
    fn events_roundtrip_line_by_line() {
        let dir = TempRunDir::new();
        let mut w = JsonlWriter::create(dir.path()).unwrap();
        let original = Event::Metric {
            id: MetricId(3),
            value: 0.25,
            step: 7,
            epoch: 1,
            phase: Phase::Eval,
        };
        w.write_event(&original).unwrap();
        w.flush().unwrap();

        let text = fs::read_to_string(dir.path().join(EVENTS_FILE)).unwrap();
        let parsed: Event = serde_json::from_str(text.lines().next().unwrap()).unwrap();
        assert_eq!(parsed, original);
    }

    #[test]
    fn metrics_roundtrip_and_stay_wide_v1_shape() {
        let dir = TempRunDir::new();
        let mut w = JsonlWriter::create(dir.path()).unwrap();
        let original = record(42, 0.123);
        w.write_metric(&original).unwrap();
        w.flush().unwrap();

        let line = fs::read_to_string(dir.path().join(METRICS_FILE)).unwrap();
        let line = line.lines().next().unwrap();
        // v1-readable shape: flat object with step/epoch/phase + named values.
        let raw: serde_json::Value = serde_json::from_str(line).unwrap();
        assert_eq!(raw["step"], 42);
        assert_eq!(raw["phase"], "TRAIN");
        assert_eq!(raw["values"]["loss"], 0.123);
        // And it roundtrips back into a MetricRecord.
        let parsed: MetricRecord = serde_json::from_str(line).unwrap();
        assert_eq!(parsed, original);
    }

    #[test]
    fn external_v1_metrics_line_parses() {
        // A line written by an external/v1 tool must deserialize unchanged.
        let line = r#"{"step":100,"epoch":2,"phase":"EVAL","values":{"acc":0.91,"loss":0.3}}"#;
        let parsed: MetricRecord = serde_json::from_str(line).unwrap();
        assert_eq!(parsed.step, 100);
        assert_eq!(parsed.phase, Phase::Eval);
        assert!((parsed.values["acc"] - 0.91).abs() < 1e-9);
    }

    #[test]
    fn create_fails_for_missing_directory() {
        let err = JsonlWriter::create(Path::new("/no/such/emry/dir")).unwrap_err();
        assert!(matches!(err, EmryError::Io(_)));
    }
}
