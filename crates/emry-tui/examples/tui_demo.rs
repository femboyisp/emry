//! Live TUI demo: a synthetic training run streams to the dashboard, overlaid
//! against a synthetic "previous run" baseline (the amber comparison curve).
//!
//! Run in a real terminal with: `cargo run -p emry-tui --example tui_demo`
//! Keys: `q`/`Esc` quit, `1`–`4` select metric, `p` pause.
//! Select `loss` (`1`) or `loss_ema` (`3`) to see the dashed amber baseline.

use emry_engine::{Engine, RunConfig};
use emry_tui::{run_terminal, BaselineSeries, Phase, UiState};
use std::time::Duration;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let run_dir = std::env::temp_dir().join("emry-tui-demo");
    std::fs::create_dir_all(&run_dir)?;

    let cfg = RunConfig {
        metric_names: vec!["loss".into(), "lr".into()],
        total_steps: Some(2000),
        ..RunConfig::new("tui-demo", &run_dir)
    };
    let mut run = Engine::start(cfg)?;
    let loss = run.register("loss");
    let lr = run.register("lr");
    // Pre-register the derived series so the dashboard can label them (the bus
    // carries MetricIds; the engine assigns these names the same ids).
    let loss_ema = run.register("loss_ema");
    let lr_ema = run.register("lr_ema");
    let sps = run.register("steps_per_sec");
    let eta = run.register("eta_secs");
    let events = run.bus().subscribe();

    // Feed a synthetic loss curve from a background thread, cycling phases and
    // saving "checkpoints" so the chart shows phase bands + checkpoint markers
    // alongside the comparison overlay.
    std::thread::spawn(move || {
        for step in 0..2000u64 {
            #[allow(clippy::cast_precision_loss)]
            let base = 2.0 / (1.0 + step as f64 * 0.005);
            // Modest 3x spikes: visible as anomalies without flattening the
            // baseline curve under a linear y-axis.
            let value = if step % 400 == 399 { base * 3.0 } else { base };
            // A short EVAL phase every 400 steps (the rest is TRAIN).
            if step % 400 == 0 {
                run.set_phase(if (step / 400) % 2 == 1 {
                    Phase::Eval
                } else {
                    Phase::Train
                });
            }
            run.emit(&[(loss, value), (lr, 1e-3)]);
            if step % 300 == 0 && step > 0 {
                run.checkpoint(format!("/ckpt/step_{step}.pt"), step);
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        run.finish().ok();
    });

    let mut state = UiState::with_labels(&[
        (loss, "loss"),
        (lr, "lr"),
        (loss_ema, "loss_ema"),
        (lr_ema, "lr_ema"),
        (sps, "steps_per_sec"),
        (eta, "eta_secs"),
    ]);
    // A synthetic "previous run" whose loss decays more slowly, overlaid as the
    // amber comparison baseline behind the live (terracotta) curve.
    state.set_baseline(synthetic_baseline());
    run_terminal(&events, state)?;
    Ok(())
}

/// A prior-run baseline for `loss` / `loss_ema` that sits above the live curve.
fn synthetic_baseline() -> Vec<BaselineSeries> {
    let (mut steps, mut values) = (Vec::new(), Vec::new());
    for step in (0..2000u64).step_by(5) {
        #[allow(clippy::cast_precision_loss)]
        let loss = 2.3 / (1.0 + step as f64 * 0.0032);
        steps.push(step);
        values.push(loss);
    }
    let series = |label: &str| BaselineSeries {
        label: label.to_string(),
        steps: steps.clone(),
        values: values.clone(),
    };
    vec![series("loss"), series("loss_ema")]
}
