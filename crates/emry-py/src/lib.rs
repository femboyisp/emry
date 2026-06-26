//! `PyO3` bindings for the Python `emry` package.
//!
//! Exposes a native `emry._native` module so Python's `embedded` deploy mode
//! drives the Rust engine in-process (the fast `emit()` path) instead of the
//! pure-Python JSONL backend.
//!
//! The `extension-module` feature is enabled by maturin when building the
//! Python wheel; it is left off for `cargo test`/`clippy` so the crate links
//! normally there.

/// Package version string exposed to Python.
#[must_use]
pub fn emry_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(feature = "extension-module")]
mod native {
    use emry_engine::{Engine, RunConfig, RunHandle};
    use pyo3::prelude::*;
    use std::collections::HashMap;
    use std::path::PathBuf;

    /// A live run, wrapping the Rust [`RunHandle`].
    ///
    /// `unsendable`: the handle owns the SPSC ring producer and is meant to be
    /// driven from the single training thread, so PyO3 pins it there (touching
    /// it from another thread raises rather than racing).
    #[pyclass(name = "RunHandle", unsendable)]
    pub struct PyRunHandle {
        inner: Option<RunHandle>,
    }

    #[pymethods]
    impl PyRunHandle {
        /// Starts a run writing to `run_dir`, pre-registering `metric_names`.
        #[new]
        #[pyo3(signature = (project, run_dir, metric_names, total_steps=None))]
        fn new(
            project: &str,
            run_dir: PathBuf,
            metric_names: Vec<String>,
            total_steps: Option<u64>,
        ) -> PyResult<Self> {
            let config = RunConfig {
                metric_names,
                total_steps,
                ..RunConfig::new(project, run_dir)
            };
            let handle = Engine::start(config)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyOSError, _>(e.to_string()))?;
            Ok(Self {
                inner: Some(handle),
            })
        }

        /// Registers (or looks up) a metric name, returning its id for `emit`.
        fn register(&self, name: &str) -> PyResult<u16> {
            Ok(self.handle()?.register(name).index())
        }

        /// Fast path: emits pre-registered `(metric_id, value)` pairs for the
        /// current step. Releases the GIL while the Rust side enqueues.
        ///
        /// Non-blocking: if the ring is full the batch is dropped and counted
        /// rather than raising — poll [`dropped`](Self::dropped) to detect
        /// saturation.
        fn emit(&mut self, py: Python<'_>, values: Vec<(u16, f64)>) -> PyResult<()> {
            let pairs: Vec<(emry_core::MetricId, f64)> = values
                .into_iter()
                .map(|(id, v)| (emry_core::MetricId(id), v))
                .collect();
            let handle = self.inner.as_mut().ok_or_else(finished_err)?;
            py.detach(|| handle.emit(&pairs));
            Ok(())
        }

        /// Slow path: emits metrics by name, registering unseen names.
        // PyO3 extracts the dict as an owned HashMap; we only borrow it.
        #[allow(clippy::needless_pass_by_value)]
        fn emit_dynamic(&mut self, py: Python<'_>, values: HashMap<String, f64>) -> PyResult<()> {
            let handle = self.inner.as_mut().ok_or_else(finished_err)?;
            py.detach(|| handle.emit_dynamic(&values));
            Ok(())
        }

        /// Sets the current epoch.
        fn set_epoch(&mut self, epoch: u32) -> PyResult<()> {
            self.inner
                .as_mut()
                .ok_or_else(finished_err)?
                .set_epoch(epoch);
            Ok(())
        }

        /// Sets the current phase from its screaming-snake name (e.g. `"TRAIN"`).
        fn set_phase(&mut self, phase: &str) -> PyResult<()> {
            let phase = parse_phase(phase)?;
            self.inner
                .as_mut()
                .ok_or_else(finished_err)?
                .set_phase(phase);
            Ok(())
        }

        /// Number of events dropped because the ring was full.
        fn dropped(&self) -> PyResult<u64> {
            Ok(self.handle()?.dropped())
        }

        /// Finishes the run, flushing all logs. Idempotent; releases the GIL
        /// while the worker drains.
        fn finish(&mut self, py: Python<'_>) -> PyResult<()> {
            if let Some(handle) = self.inner.take() {
                py.detach(|| handle.finish())
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyOSError, _>(e.to_string()))?;
            }
            Ok(())
        }
    }

    // Plain (non-`#[pymethods]`) helpers must live in a separate impl block.
    impl PyRunHandle {
        fn handle(&self) -> PyResult<&RunHandle> {
            self.inner.as_ref().ok_or_else(finished_err)
        }
    }

    fn finished_err() -> PyErr {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("run already finished")
    }

    fn parse_phase(phase: &str) -> PyResult<emry_core::Phase> {
        // Parse through Phase's serde config (the single source of truth) so new
        // variants are accepted automatically without updating a hand-written
        // match here.
        serde_json::from_value(serde_json::Value::String(phase.to_string())).map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("unknown phase {phase:?}"))
        })
    }

    /// The native module: `import emry._native`.
    #[pymodule]
    fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add("__version__", super::emry_version())?;
        m.add_class::<PyRunHandle>()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_matches_cargo_package() {
        assert_eq!(emry_version(), "0.1.0");
    }
}
