//! Benchmark: lock-free ring push/pop on the hot path.
#![allow(missing_docs)] // criterion_group! generates an undocumented fn

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use emry_core::{event_ring_with_capacity, Event, MetricId, Phase};

fn metric(step: u64) -> Event {
    Event::Metric {
        id: MetricId(0),
        value: 0.5,
        step,
        epoch: 0,
        phase: Phase::Train,
    }
}

fn ring_push_pop(c: &mut Criterion) {
    c.bench_function("ring_push_pop", |b| {
        let (mut producer, mut consumer) = event_ring_with_capacity(1024);
        let mut step = 0u64;
        b.iter(|| {
            let _ = producer.push(black_box(metric(step)));
            black_box(consumer.pop());
            step += 1;
        });
    });
}

criterion_group!(benches, ring_push_pop);
criterion_main!(benches);
