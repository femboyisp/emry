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

/// Metadata captured when a training run starts.
#[derive(Debug, Clone, Serialize, Deserialize)]
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
        assert_eq!(back.project, "llama-sft");
        assert_eq!(back.run_id, Uuid::nil());
        assert!((back.start_time_secs - 1_719_000_000.0).abs() < f64::EPSILON);
    }
}
