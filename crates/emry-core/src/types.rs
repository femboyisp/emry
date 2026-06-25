//! Protocol types shared across Emry crates.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Training phase for context-aware logging and UI styling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Phase {
    /// Main training loop.
    Train,
    /// Validation or evaluation.
    Eval,
    /// Test set evaluation.
    Test,
    /// Checkpoint save in progress.
    Checkpoint,
    /// Learning-rate or other warmup.
    Warmup,
}

/// Interned identifier for a registered metric.
///
/// Metrics are registered once at run start and addressed by this compact
/// `u16` on the hot path so `emit()` avoids string hashing. The interning
/// table itself lives in the metric registry (EMRY-003); this newtype is the
/// shared wire-level handle. A maximum of 65 535 distinct metrics is supported.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct MetricId(pub u16);

impl MetricId {
    /// Returns the raw `u16` index.
    #[must_use]
    pub fn index(self) -> u16 {
        self.0
    }
}

impl From<u16> for MetricId {
    fn from(value: u16) -> Self {
        Self(value)
    }
}

/// Severity of an [`AlertRecord`], from informational to fatal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Severity {
    /// Noteworthy but benign (e.g. throughput dipped).
    Info,
    /// Something worth a human glance (e.g. loss spike).
    Warning,
    /// Run integrity is compromised (e.g. NaN loss).
    Critical,
}

/// A gentle alert surfaced to observers and recorded in the event log.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AlertRecord {
    /// How serious the condition is.
    pub severity: Severity,
    /// Human-readable, calm alert copy (brand §13).
    pub message: String,
    /// Training step the alert refers to, if applicable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub step: Option<u64>,
}

/// Why a run ended, recorded in [`Event::RunFinished`] and `summary.json`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum FinishReason {
    /// The training loop ran to completion.
    Completed,
    /// A user or signal interrupted the run.
    Interrupted,
    /// The run aborted because of an error.
    Failed,
}

/// Metadata captured when a training run starts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunMeta {
    /// Unique identifier for this run.
    pub run_id: Uuid,
    /// Human-readable project or experiment name.
    pub project: String,
    /// Hyperparameters and run configuration.
    pub config: serde_json::Value,
    /// Unix timestamp (seconds) when the run started.
    pub start_time_secs: f64,
}

/// A single event in the Emry event stream.
///
/// Events are the unit of the append-only audit log (`events.jsonl`). They are
/// serialized adjacently tagged — `{"type": "...", "data": {...}}` — so that
/// every variant (struct, tuple, and newtype alike) roundtrips through JSON and
/// msgpack on one line. Hot-path metric emission uses [`Event::MetricsBatch`];
/// [`Event::Metric`] covers single dynamic values.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "data", rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Event {
    /// A single metric value at a training step.
    Metric {
        /// Registered metric handle.
        id: MetricId,
        /// Metric value.
        value: f64,
        /// Global training step.
        step: u64,
        /// Current epoch.
        epoch: u32,
        /// Phase the value was emitted in.
        phase: Phase,
    },
    /// Several metric values sharing the same step/epoch/phase (hot path).
    MetricsBatch {
        /// Global training step.
        step: u64,
        /// Current epoch.
        epoch: u32,
        /// Phase the values were emitted in.
        phase: Phase,
        /// `(metric, value)` pairs.
        values: Vec<(MetricId, f64)>,
    },
    /// The run transitioned to a new [`Phase`].
    PhaseChange(Phase),
    /// A checkpoint was written to disk.
    Checkpoint {
        /// Filesystem path of the checkpoint.
        path: String,
        /// Step the checkpoint was taken at.
        step: u64,
    },
    /// An incremental patch to the run configuration.
    ConfigPatch(serde_json::Value),
    /// An anomaly or notable condition was detected.
    Alert(AlertRecord),
    /// The run started; carries its [`RunMeta`].
    RunStarted(RunMeta),
    /// The run finished.
    RunFinished {
        /// Wall-clock duration of the run.
        duration_secs: f64,
        /// Why the run ended.
        reason: FinishReason,
    },
}

/// A wide, flattened metric row for `metrics.jsonl`.
///
/// Distinct from [`Event`]: this is the export-friendly schema consumed by
/// `emry watch`, CSV/Parquet export, and external/v1 JSONL readers. Metric
/// names are resolved (not [`MetricId`]) so the file is self-describing. The
/// [`Phase`] serializes as a screaming-snake string (e.g. `"TRAIN"`) for v1
/// compatibility.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetricRecord {
    /// Global training step.
    pub step: u64,
    /// Current epoch.
    pub epoch: u32,
    /// Phase the values were emitted in.
    pub phase: Phase,
    /// Resolved metric name → value pairs for this row.
    pub values: std::collections::BTreeMap<String, f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[test]
    fn phase_serializes_as_screaming_snake() {
        let json = serde_json::to_string(&Phase::Train).unwrap();
        assert_eq!(json, "\"TRAIN\"");
    }

    #[test]
    fn all_phases_roundtrip_json() {
        for phase in [
            Phase::Train,
            Phase::Eval,
            Phase::Test,
            Phase::Checkpoint,
            Phase::Warmup,
        ] {
            let json = serde_json::to_string(&phase).unwrap();
            let back: Phase = serde_json::from_str(&json).unwrap();
            assert_eq!(back, phase);
        }
    }

    #[test]
    fn run_meta_roundtrips_json() {
        let meta = RunMeta {
            run_id: Uuid::nil(),
            project: "llama-sft".to_string(),
            config: serde_json::json!({"lr": 2e-5}),
            start_time_secs: 1_719_000_000.0,
        };
        let json = serde_json::to_string(&meta).unwrap();
        let back: RunMeta = serde_json::from_str(&json).unwrap();
        assert_eq!(back, meta);
    }

    #[test]
    fn metric_id_is_transparent_u16() {
        let id = MetricId::from(7);
        assert_eq!(id.index(), 7);
        assert_eq!(serde_json::to_string(&id).unwrap(), "7");
        assert_eq!(serde_json::from_str::<MetricId>("7").unwrap(), id);
    }

    #[test]
    fn severity_and_finish_reason_screaming_snake() {
        assert_eq!(
            serde_json::to_string(&Severity::Critical).unwrap(),
            "\"CRITICAL\""
        );
        assert_eq!(
            serde_json::to_string(&FinishReason::Interrupted).unwrap(),
            "\"INTERRUPTED\""
        );
    }

    #[test]
    fn alert_record_omits_absent_step() {
        let alert = AlertRecord {
            severity: Severity::Warning,
            message: "Loss spiked".to_string(),
            step: None,
        };
        let json = serde_json::to_string(&alert).unwrap();
        assert!(
            !json.contains("step"),
            "absent step should be skipped: {json}"
        );
        let back: AlertRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(back, alert);
    }

    /// Adjacently-tagged form must roundtrip for every variant shape.
    #[test]
    fn every_event_variant_roundtrips() {
        let events = vec![
            Event::Metric {
                id: MetricId(0),
                value: 0.42,
                step: 100,
                epoch: 1,
                phase: Phase::Train,
            },
            Event::MetricsBatch {
                step: 101,
                epoch: 1,
                phase: Phase::Train,
                values: vec![(MetricId(0), 0.4), (MetricId(1), 2e-5)],
            },
            Event::PhaseChange(Phase::Eval),
            Event::Checkpoint {
                path: "/ckpt/step_200.pt".to_string(),
                step: 200,
            },
            Event::ConfigPatch(serde_json::json!({"lr": 1e-5})),
            Event::Alert(AlertRecord {
                severity: Severity::Critical,
                message: "Loss became NaN".to_string(),
                step: Some(12_400),
            }),
            Event::RunStarted(RunMeta {
                run_id: Uuid::nil(),
                project: "llama-sft".to_string(),
                config: serde_json::json!({}),
                start_time_secs: 1.0,
            }),
            Event::RunFinished {
                duration_secs: 3600.0,
                reason: FinishReason::Completed,
            },
        ];
        for event in events {
            let json = serde_json::to_string(&event).unwrap();
            let back: Event = serde_json::from_str(&json).unwrap();
            assert_eq!(back, event, "roundtrip failed for {event:?}");
        }
    }

    #[test]
    fn event_uses_adjacent_tag_layout() {
        let json = serde_json::to_string(&Event::PhaseChange(Phase::Eval)).unwrap();
        assert_eq!(json, r#"{"type":"PHASE_CHANGE","data":"EVAL"}"#);
    }

    /// The sidecar wire protocol (EMRY-024) frames events as msgpack. Adjacently
    /// tagged enums only roundtrip through msgpack when structs are encoded as
    /// maps, so the protocol must use `Serializer::with_struct_map` — the default
    /// compact (sequence) encoding fails to deserialize the tagged form.
    #[test]
    fn event_roundtrips_through_msgpack_as_maps() {
        use serde::Serialize;

        let event = Event::MetricsBatch {
            step: 7,
            epoch: 0,
            phase: Phase::Warmup,
            values: vec![(MetricId(3), 1.5)],
        };
        let mut bytes = Vec::new();
        event
            .serialize(&mut rmp_serde::Serializer::new(&mut bytes).with_struct_map())
            .unwrap();
        let back: Event = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(back, event);
    }

    #[test]
    fn metric_record_roundtrips_with_named_values() {
        let mut values = std::collections::BTreeMap::new();
        values.insert("loss".to_string(), 0.31);
        values.insert("lr".to_string(), 2e-5);
        let record = MetricRecord {
            step: 500,
            epoch: 2,
            phase: Phase::Train,
            values,
        };
        let json = serde_json::to_string(&record).unwrap();
        assert!(json.contains("\"phase\":\"TRAIN\""));
        let back: MetricRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(back, record);
    }
}
