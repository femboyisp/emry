//! Benchmark: the registered `emit` fast path (target < 10 µs amortized).
//!
//! Measures `RunHandle::emit` with pre-registered metric ids — the call a
//! training loop makes every step. The ring drops-and-counts on overflow, so
//! this never blocks on the worker; the benchmark reflects the producer cost.
#![allow(missing_docs)] // criterion_group! generates an undocumented fn

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use emry_engine::{Engine, RunConfig};

fn registered_emit(c: &mut Criterion) {
    let dir = std::env::temp_dir().join("emry-bench-emit");
    std::fs::create_dir_all(&dir).unwrap();
    let cfg = RunConfig {
        metric_names: vec!["loss".into(), "lr".into()],
        // Keep the measured path to the raw emit cost, not derived processors.
        detect_anomalies: false,
        track_throughput: false,
        smoothing: None,
        ..RunConfig::new("bench", &dir)
    };
    let mut run = Engine::start(cfg).expect("start engine");
    let loss = run.register("loss");
    let lr = run.register("lr");

    c.bench_function("registered_emit_2_metrics", |b| {
        b.iter(|| run.emit(black_box(&[(loss, 0.5), (lr, 1e-3)])));
    });
}

criterion_group!(benches, registered_emit);
criterion_main!(benches);
