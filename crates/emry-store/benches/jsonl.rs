//! Benchmark: JSONL metric-row write + flush throughput.
#![allow(missing_docs)] // criterion_group! generates an undocumented fn

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use emry_core::{MetricRecord, Phase};
use emry_store::JsonlWriter;
use std::collections::BTreeMap;

fn record() -> MetricRecord {
    let mut values = BTreeMap::new();
    values.insert("loss".to_string(), 0.5);
    values.insert("lr".to_string(), 1e-3);
    MetricRecord {
        step: 0,
        epoch: 0,
        phase: Phase::Train,
        values,
    }
}

fn jsonl_write_and_flush(c: &mut Criterion) {
    let dir = std::env::temp_dir().join("emry-bench-jsonl");
    std::fs::create_dir_all(&dir).unwrap();
    let mut writer = JsonlWriter::create(&dir).expect("create writer");
    let rec = record();

    // Buffered append (the per-row hot cost).
    c.bench_function("jsonl_write_metric", |b| {
        b.iter(|| writer.write_metric(black_box(&rec)).unwrap());
    });

    // Append + flush to the OS (the durability cost the sink pays periodically).
    c.bench_function("jsonl_write_and_flush", |b| {
        b.iter(|| {
            writer.write_metric(black_box(&rec)).unwrap();
            writer.flush().unwrap();
        });
    });
}

criterion_group!(benches, jsonl_write_and_flush);
criterion_main!(benches);
