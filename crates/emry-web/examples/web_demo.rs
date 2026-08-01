//! Live web dashboard demo: a synthetic run streamed to the web server.
//!
//! `cargo run -p emry-web --example web_demo`, then open <http://127.0.0.1:8788>.
use emry_core::{Event, EventBus, MetricId, Phase, RunMeta};
use std::sync::Arc;
use std::time::Duration;

#[tokio::main]
async fn main() {
    let bus = Arc::new(EventBus::new());
    let sub = bus.subscribe();

    // Feed synthetic events into the bus from a background thread.
    let feed = Arc::clone(&bus);
    std::thread::spawn(move || {
        feed.publish(&Event::RunStarted(RunMeta {
            run_id: uuid::Uuid::new_v4(),
            project: "web-demo".into(),
            config: serde_json::json!({"lr": 1e-3}),
            start_time_secs: 0.0,
        }));
        let (loss, lr, ema) = (MetricId(0), MetricId(1), MetricId(2));
        for step in 0..100_000u64 {
            #[allow(clippy::cast_precision_loss)]
            let base = 2.0 / (1.0 + step as f64 * 0.01);
            let value = if step % 250 == 249 { base * 3.0 } else { base };
            let phase = if (step / 200) % 5 == 4 {
                Phase::Eval
            } else {
                Phase::Train
            };
            if step % 200 == 0 {
                feed.publish(&Event::PhaseChange(phase));
            }
            feed.publish(&Event::MetricsBatch {
                step,
                epoch: u32::try_from(step / 200).unwrap_or(0),
                phase,
                values: vec![(loss, value), (lr, 1e-3), (ema, base)],
            });
            if step % 300 == 0 && step > 0 {
                // Curriculum-style paths so the phase-aware chart derives a
                // distinct segment label per stage (phaseN- prefix stripped).
                let stages = ["reasoning", "knowledge", "code", "math", "polish"];
                let n = u64::try_from(stages.len()).unwrap_or(1);
                let idx = usize::try_from((step / 300 - 1) % n).unwrap_or(0);
                let stage = stages[idx];
                let phase_n = step / 300;
                feed.publish(&Event::Checkpoint {
                    path: format!("/ckpt/phase{phase_n}-{stage}/step_{step}.pt"),
                    step,
                });
            }
            if step == 100 {
                feed.publish(&Event::Alert(emry_core::AlertRecord {
                    severity: emry_core::Severity::Warning,
                    message: "Loss spiked at step 100 — this may be a transient blip.".into(),
                    step: Some(100),
                }));
            }
            std::thread::sleep(Duration::from_millis(40));
        }
    });

    // A synthetic "previous run" whose loss decays a touch slower than the live
    // run, overlaid as a dashed amber baseline.
    let baseline = synthetic_baseline();

    println!("serving on http://127.0.0.1:8788");
    let labels = [
        (MetricId(0), "loss"),
        (MetricId(1), "lr"),
        (MetricId(2), "loss_ema"),
    ];
    emry_web::serve_with_baseline(
        "127.0.0.1:8788".parse().unwrap(),
        sub,
        &labels,
        baseline,
        emry_web::WebSecurity::default(),
    )
    .await
    .unwrap();
}

/// A synthetic prior run whose loss decays a touch slower than the live run, so
/// the dashed overlay sits visibly above the live curve.
fn synthetic_baseline() -> emry_web::Baseline {
    let (mut steps, mut values) = (Vec::new(), Vec::new());
    for step in (0..100_000u64).step_by(50) {
        #[allow(clippy::cast_precision_loss)]
        let loss = 2.2 / (1.0 + step as f64 * 0.008);
        steps.push(step);
        values.push(loss);
    }
    let series = |label: &str| emry_web::BaselineSeries {
        label: label.to_string(),
        steps: steps.clone(),
        values: values.clone(),
    };
    emry_web::Baseline {
        metrics: vec![series("loss"), series("loss_ema")],
    }
}
