//! End-to-end demo: drive a synthetic training loop through the engine and
//! write `events.jsonl` / `metrics.jsonl`.
//!
//! Run with: `cargo run -p emry-engine --example synthetic_run`

use emry_engine::{Engine, RunConfig};
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let run_dir: PathBuf = std::env::args()
        .nth(1)
        .map_or_else(|| std::env::temp_dir().join("emry-demo-run"), PathBuf::from);
    std::fs::create_dir_all(&run_dir)?;

    let cfg = RunConfig {
        metric_names: vec!["loss".into(), "lr".into()],
        total_steps: Some(200),
        config: serde_json::json!({ "lr": 1e-3, "demo": true }),
        ..RunConfig::new("synthetic-demo", &run_dir)
    };

    let mut run = Engine::start(cfg)?;
    let loss = run.register("loss");
    let lr = run.register("lr");

    // A decaying loss curve with one deliberate spike to exercise anomaly alerts.
    for step in 0..200u64 {
        #[allow(clippy::cast_precision_loss)]
        let base = 2.0 / (1.0 + step as f64 * 0.05);
        let value = if step == 120 { base * 25.0 } else { base };
        run.emit(&[(loss, value), (lr, 1e-3)]);
    }

    run.finish()?;
    println!("wrote run logs to {}", run_dir.display());
    Ok(())
}
