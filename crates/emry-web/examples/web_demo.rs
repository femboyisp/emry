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

    println!("serving on http://127.0.0.1:8788");
    let labels = [
        (MetricId(0), "loss"),
        (MetricId(1), "lr"),
        (MetricId(2), "loss_ema"),
    ];
    emry_web::serve_with_labels("127.0.0.1:8788".parse().unwrap(), sub, &labels)
        .await
        .unwrap();
}
