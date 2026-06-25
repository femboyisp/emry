//! Live TUI demo: a synthetic training run streams to the dashboard.
//!
//! Run in a real terminal with: `cargo run -p emry-tui --example tui_demo`
//! Keys: `q`/`Esc` quit, `1`–`4` select metric, `p` pause.

use emry_engine::{Engine, RunConfig};
use emry_tui::{run_terminal, UiState};
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

    // Feed a synthetic loss curve from a background thread.
    std::thread::spawn(move || {
        for step in 0..2000u64 {
            #[allow(clippy::cast_precision_loss)]
            let base = 2.0 / (1.0 + step as f64 * 0.005);
            let value = if step % 400 == 399 { base * 20.0 } else { base };
            run.emit(&[(loss, value), (lr, 1e-3)]);
            std::thread::sleep(Duration::from_millis(10));
        }
        run.finish().ok();
    });

    let state = UiState::with_labels(&[
        (loss, "loss"),
        (lr, "lr"),
        (loss_ema, "loss_ema"),
        (lr_ema, "lr_ema"),
        (sps, "steps_per_sec"),
        (eta, "eta_secs"),
    ]);
    run_terminal(&events, state)?;
    Ok(())
}
