//! Persistence: JSONL, Parquet, rotation.

pub mod export;
pub mod meta;
pub mod sink;
pub mod writer;

#[cfg(test)]
mod test_util;

pub use emry_core::EmryError;
pub use export::export_csv;
#[cfg(feature = "parquet")]
pub use export::export_parquet;
pub use meta::{
    create_run_dir, run_dir_name, write_json, RunMetaFile, Summary, CONFIG_FILE, RUN_META_FILE,
    SUMMARY_FILE,
};
pub use sink::{JsonlSink, FLUSH_EVERY, FLUSH_INTERVAL};
pub use writer::{JsonlWriter, EVENTS_FILE, METRICS_FILE};
