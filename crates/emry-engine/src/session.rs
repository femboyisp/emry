//! The run engine: [`Engine::start`] assembles the ring, bus, anomaly detection,
//! and JSONL persistence into a single live run and hands back a [`RunHandle`].
//!
//! ```text
//! training thread          worker thread
//! ──────────────           ─────────────
//! handle.emit()  ─ring─►    drain → events.jsonl + metrics.jsonl
//!                           drain → publish to EventBus (observers)
//!                           drain → anomaly detect → Event::Alert
//! ```
//!
//! The fast path ([`RunHandle::emit`]) takes already-registered
//! [`MetricId`](emry_core::MetricId)s and only pushes to the lock-free ring, so
//! it never locks or blocks the training thread. Name resolution for
//! `metrics.jsonl` and the slow [`RunHandle::emit_dynamic`] path use a shared
//! registry behind a mutex, off the hot path.
//!
//! The EMA/Welford/throughput processors exist as building blocks but are not
//! wired into the session yet — routing their `DerivedMetric`s to observers is
//! EMRY-022 (TUI `UiState`). The anomaly detector is wired in because its output
//! ([`Event::Alert`]) has a home in the event stream.

use crate::anomaly::AnomalyDetector;
use emry_core::{
    event_ring, DeployMode, EmryError, Event, EventBus, EventConsumer, EventProducer, FinishReason,
    MetricId, MetricRecord, MetricRegistry, Phase, RunMeta,
};
use emry_store::{JsonlSink, RunMetaFile, Summary, CONFIG_FILE, RUN_META_FILE, SUMMARY_FILE};
use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// How long the worker sleeps when the ring is momentarily empty.
const POLL_INTERVAL: Duration = Duration::from_micros(200);

/// Configuration for a run, passed to [`Engine::start`].
#[derive(Debug, Clone)]
pub struct RunConfig {
    /// Human-readable project / experiment name.
    pub project: String,
    /// Directory to write `events.jsonl` / `metrics.jsonl` into (must exist).
    pub run_dir: PathBuf,
    /// Metric names to register up front for the fast path.
    pub metric_names: Vec<String>,
    /// Hyperparameters / run configuration, persisted in the `RunStarted` event.
    pub config: serde_json::Value,
    /// Total step count, if known (for downstream ETA).
    pub total_steps: Option<u64>,
    /// Whether to watch every registered metric for NaN/Inf and spikes.
    pub detect_anomalies: bool,
    /// Deploy mode in effect, recorded in `run.meta`.
    pub mode: DeployMode,
}

impl RunConfig {
    /// Minimal config: a project name and an existing run directory.
    #[must_use]
    pub fn new(project: impl Into<String>, run_dir: impl Into<PathBuf>) -> Self {
        Self {
            project: project.into(),
            run_dir: run_dir.into(),
            metric_names: Vec::new(),
            config: serde_json::Value::Null,
            total_steps: None,
            detect_anomalies: true,
            mode: DeployMode::detect(None),
        }
    }
}

/// Entry point: starts runs.
#[derive(Debug, Default)]
pub struct Engine;

impl Engine {
    /// Starts a run: registers metrics, opens the JSONL logs, spawns the worker
    /// thread, and emits the opening `RunStarted` event.
    ///
    /// # Errors
    ///
    /// Returns [`EmryError::Io`] if the run directory's JSONL files cannot be
    /// opened.
    pub fn start(config: RunConfig) -> Result<RunHandle, EmryError> {
        let RunConfig {
            project,
            run_dir,
            metric_names,
            config: hyper_config,
            total_steps: _,
            detect_anomalies,
            mode,
        } = config;

        let mut registry = MetricRegistry::new();
        for name in &metric_names {
            registry.register(name);
        }
        let registry = Arc::new(Mutex::new(registry));

        let detectors = if detect_anomalies {
            metric_names
                .iter()
                .map(|name| {
                    let id = registry.lock().expect("registry poisoned").register(name);
                    AnomalyDetector::new(id, capitalize(name))
                })
                .collect()
        } else {
            Vec::new()
        };

        let sink = JsonlSink::spawn(&run_dir)?;
        let bus = Arc::new(EventBus::new());
        let (mut producer, consumer) = event_ring();
        let stop = Arc::new(AtomicBool::new(false));

        let start_time_secs = unix_secs_now();
        let run_id = uuid::Uuid::new_v4();

        // Metadata files (EMRY-016): run.meta + config.json at start.
        emry_store::write_json(
            &run_dir,
            RUN_META_FILE,
            &RunMetaFile {
                run_id,
                project: project.clone(),
                start_time_secs,
                mode: mode.to_string(),
            },
        )?;
        emry_store::write_json(&run_dir, CONFIG_FILE, &hyper_config)?;

        let meta = RunMeta {
            run_id,
            project: project.clone(),
            config: hyper_config,
            start_time_secs,
        };
        // Opening event; the worker persists and publishes it like any other.
        let _ = producer.push(Event::RunStarted(meta));

        let worker = Worker {
            consumer,
            bus: Arc::clone(&bus),
            sink,
            registry: Arc::clone(&registry),
            detectors,
            stop: Arc::clone(&stop),
        };
        let handle = std::thread::spawn(move || worker.run());

        Ok(RunHandle {
            producer,
            bus,
            registry,
            project,
            run_dir,
            step: 0,
            epoch: 0,
            phase: Phase::Train,
            start: Instant::now(),
            run_id,
            worker: Some(handle),
            stop,
        })
    }
}

/// Handle held by the training thread to feed a live run.
pub struct RunHandle {
    producer: EventProducer,
    bus: Arc<EventBus>,
    registry: Arc<Mutex<MetricRegistry>>,
    project: String,
    run_dir: PathBuf,
    step: u64,
    epoch: u32,
    phase: Phase,
    start: Instant,
    run_id: uuid::Uuid,
    worker: Option<JoinHandle<Result<(), EmryError>>>,
    stop: Arc<AtomicBool>,
}

impl RunHandle {
    /// This run's unique identifier.
    #[must_use]
    pub fn run_id(&self) -> uuid::Uuid {
        self.run_id
    }

    /// The event bus, for attaching observers (TUI, web, …).
    #[must_use]
    pub fn bus(&self) -> &Arc<EventBus> {
        &self.bus
    }

    /// Registers (or looks up) a metric name, returning its [`MetricId`] for the
    /// fast [`RunHandle::emit`] path. Slow path (takes the registry lock).
    pub fn register(&self, name: &str) -> MetricId {
        self.registry
            .lock()
            .expect("registry poisoned")
            .register(name)
    }

    /// Fast path: emits pre-registered metric values for the current step, then
    /// advances the step. Never blocks; a full ring drops the batch and counts it.
    pub fn emit(&mut self, values: &[(MetricId, f64)]) {
        let event = Event::MetricsBatch {
            step: self.step,
            epoch: self.epoch,
            phase: self.phase,
            values: values.to_vec(),
        };
        let _ = self.producer.push(event);
        self.step += 1;
    }

    /// Slow path: emits metrics by name, registering any unseen names. Takes the
    /// registry lock; prefer [`RunHandle::emit`] on the hot path.
    pub fn emit_dynamic(&mut self, values: &HashMap<String, f64>) {
        let resolved: Vec<(MetricId, f64)> = {
            let mut reg = self.registry.lock().expect("registry poisoned");
            values
                .iter()
                .map(|(name, value)| (reg.register(name), *value))
                .collect()
        };
        self.emit(&resolved);
    }

    /// Sets the current training phase and records the transition.
    pub fn set_phase(&mut self, phase: Phase) {
        self.phase = phase;
        let _ = self.producer.push(Event::PhaseChange(phase));
    }

    /// Sets the current epoch (stamped onto subsequent emitted metrics).
    pub fn set_epoch(&mut self, epoch: u32) {
        self.epoch = epoch;
    }

    /// Number of events dropped because the ring was full.
    #[must_use]
    pub fn dropped(&self) -> u64 {
        self.producer.dropped()
    }

    /// Finishes the run: emits `RunFinished`, drains and flushes all logs, and
    /// joins the worker.
    ///
    /// # Errors
    ///
    /// Returns any [`EmryError`] the worker hit while writing or flushing.
    ///
    /// # Panics
    ///
    /// Panics if the worker thread panicked.
    pub fn finish(mut self) -> Result<(), EmryError> {
        self.finish_with(FinishReason::Completed)
    }

    fn finish_with(&mut self, reason: FinishReason) -> Result<(), EmryError> {
        let Some(handle) = self.worker.take() else {
            return Ok(());
        };
        #[allow(clippy::cast_precision_loss)] // run durations fit f64 exactly
        let duration_secs = self.start.elapsed().as_secs_f64();
        let _ = self.producer.push(Event::RunFinished {
            duration_secs,
            reason,
        });
        // Also raise the stop flag: if the ring was full the RunFinished push
        // above was dropped, and the worker would otherwise never exit. The
        // worker drains the ring after stopping, so a pushed RunFinished is
        // still written; join() can never hang.
        self.stop.store(true, Ordering::Release);
        let worker_result = handle.join().expect("engine worker thread panicked");

        // summary.json (EMRY-016) once the worker has flushed everything.
        let summary_result = emry_store::write_json(
            &self.run_dir,
            SUMMARY_FILE,
            &Summary {
                run_id: self.run_id,
                project: self.project.clone(),
                duration_secs,
                reason,
                steps: self.step,
                dropped: self.producer.dropped(),
            },
        );
        worker_result.and(summary_result)
    }
}

impl Drop for RunHandle {
    fn drop(&mut self) {
        // Unfinished run: finish_with raises the stop flag and joins, so the
        // worker drains and exits even if the ring is full.
        if self.worker.is_some() {
            let _ = self.finish_with(FinishReason::Interrupted);
        }
    }
}

/// The worker thread: owns the consumer, sink, and detectors.
struct Worker {
    consumer: EventConsumer,
    bus: Arc<EventBus>,
    sink: JsonlSink,
    registry: Arc<Mutex<MetricRegistry>>,
    detectors: Vec<AnomalyDetector>,
    stop: Arc<AtomicBool>,
}

impl Worker {
    fn run(mut self) -> Result<(), EmryError> {
        loop {
            if self.stop.load(Ordering::Acquire) {
                break;
            }
            if let Some(event) = self.consumer.pop() {
                let finished = matches!(event, Event::RunFinished { .. });
                self.handle_event(&event);
                if finished {
                    break;
                }
            } else {
                std::thread::sleep(POLL_INTERVAL);
            }
        }
        // Drain anything still queued so the audit log is complete (bounded by
        // ring capacity), then flush and join the sink.
        while let Some(event) = self.consumer.pop() {
            self.handle_event(&event);
        }
        self.sink.finish()
    }

    /// Persist, publish, derive metric rows, and run anomaly detection for one
    /// event.
    fn handle_event(&mut self, event: &Event) {
        self.sink.write_event(event.clone());
        self.bus.publish(event);

        if let Some(record) = self.metric_record(event) {
            self.sink.write_metric(record);
        }

        for detector in &mut self.detectors {
            for alert in detector.detect(event) {
                let alert_event = Event::Alert(alert);
                self.sink.write_event(alert_event.clone());
                self.bus.publish(&alert_event);
            }
        }
    }

    /// Builds a wide [`MetricRecord`] (resolved names) from a metric event.
    fn metric_record(&self, event: &Event) -> Option<MetricRecord> {
        let (step, epoch, phase, pairs): (u64, u32, Phase, Vec<(MetricId, f64)>) = match event {
            Event::Metric {
                id,
                value,
                step,
                epoch,
                phase,
            } => (*step, *epoch, *phase, vec![(*id, *value)]),
            Event::MetricsBatch {
                step,
                epoch,
                phase,
                values,
            } => (*step, *epoch, *phase, values.clone()),
            _ => return None,
        };

        let reg = self.registry.lock().expect("registry poisoned");
        let mut values = BTreeMap::new();
        for (id, value) in pairs {
            if let Some(name) = reg.name(id) {
                values.insert(name.to_owned(), value);
            }
        }
        Some(MetricRecord {
            step,
            epoch,
            phase,
            values,
        })
    }
}

/// Seconds since the Unix epoch as an `f64`.
fn unix_secs_now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// `"loss"` -> `"Loss"` for alert labels.
fn capitalize(s: &str) -> String {
    let mut chars = s.chars();
    chars.next().map_or_else(String::new, |first| {
        first.to_uppercase().collect::<String>() + chars.as_str()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use emry_store::{EVENTS_FILE, METRICS_FILE};
    use std::path::Path;
    use std::sync::atomic::AtomicU32;

    static COUNTER: AtomicU32 = AtomicU32::new(0);

    struct TempDir(PathBuf);
    impl TempDir {
        fn new() -> Self {
            let n = COUNTER.fetch_add(1, Ordering::Relaxed);
            let p =
                std::env::temp_dir().join(format!("emry-engine-test-{}-{n}", std::process::id()));
            std::fs::create_dir_all(&p).unwrap();
            Self(p)
        }
        fn path(&self) -> &Path {
            &self.0
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn config(dir: &Path) -> RunConfig {
        RunConfig {
            metric_names: vec!["loss".into(), "lr".into()],
            ..RunConfig::new("test-run", dir)
        }
    }

    fn lines(path: &Path) -> Vec<String> {
        std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .map(str::to_owned)
            .collect()
    }

    #[test]
    fn synthetic_run_writes_events_and_metrics() {
        let dir = TempDir::new();
        let mut run = Engine::start(config(dir.path())).unwrap();
        let loss = run.register("loss");
        let lr = run.register("lr");
        for step in 0..50 {
            run.emit(&[(loss, 1.0 / f64::from(step + 1)), (lr, 1e-3)]);
        }
        run.finish().unwrap();

        let events = lines(&dir.path().join(EVENTS_FILE));
        let metrics = lines(&dir.path().join(METRICS_FILE));
        // RunStarted + 50 batches + RunFinished.
        assert_eq!(events.len(), 52);
        assert_eq!(metrics.len(), 50);
        // First and last events are the lifecycle bookends.
        assert!(events[0].contains("RUN_STARTED"));
        assert!(events.last().unwrap().contains("RUN_FINISHED"));
        // Metric rows carry resolved names.
        assert!(metrics[0].contains("\"loss\""));
        assert!(metrics[0].contains("\"lr\""));
    }

    #[test]
    fn nan_value_produces_alert_in_event_log() {
        let dir = TempDir::new();
        let mut run = Engine::start(config(dir.path())).unwrap();
        let loss = run.register("loss");
        run.emit(&[(loss, 1.0)]);
        run.emit(&[(loss, f64::NAN)]);
        run.finish().unwrap();

        let events = lines(&dir.path().join(EVENTS_FILE));
        let alerts: Vec<_> = events.iter().filter(|l| l.contains("ALERT")).collect();
        assert_eq!(alerts.len(), 1, "one NaN -> one alert");
        assert!(alerts[0].contains("CRITICAL"));
    }

    #[test]
    fn emit_dynamic_registers_and_writes_named_metrics() {
        let dir = TempDir::new();
        // No pre-registered metrics; rely on dynamic registration.
        let mut run = Engine::start(RunConfig::new("dyn", dir.path())).unwrap();
        let mut values = HashMap::new();
        values.insert("custom_metric".to_string(), 3.5);
        run.emit_dynamic(&values);
        run.finish().unwrap();

        let metrics = lines(&dir.path().join(METRICS_FILE));
        assert_eq!(metrics.len(), 1);
        assert!(metrics[0].contains("custom_metric"));
        assert!(metrics[0].contains("3.5"));
    }

    #[test]
    fn set_phase_records_transition() {
        let dir = TempDir::new();
        let mut run = Engine::start(config(dir.path())).unwrap();
        run.set_phase(Phase::Eval);
        run.finish().unwrap();

        let events = lines(&dir.path().join(EVENTS_FILE));
        assert!(events
            .iter()
            .any(|l| l.contains("PHASE_CHANGE") && l.contains("EVAL")));
    }

    #[test]
    fn drop_without_finish_flushes_as_interrupted() {
        let dir = TempDir::new();
        {
            let mut run = Engine::start(config(dir.path())).unwrap();
            let loss = run.register("loss");
            run.emit(&[(loss, 0.5)]);
            // No finish(): Drop must flush and mark the run interrupted.
        }
        let events = lines(&dir.path().join(EVENTS_FILE));
        assert!(events.last().unwrap().contains("RUN_FINISHED"));
        assert!(events.last().unwrap().contains("INTERRUPTED"));
    }

    #[test]
    fn start_fails_for_missing_directory() {
        let mut cfg = RunConfig::new("x", "/no/such/emry/dir");
        cfg.detect_anomalies = false;
        // matches! avoids requiring RunHandle: Debug (which unwrap_err needs).
        assert!(matches!(Engine::start(cfg), Err(EmryError::Io(_))));
    }

    #[test]
    fn writes_run_meta_config_and_summary_files() {
        use emry_store::{RunMetaFile, Summary, CONFIG_FILE, RUN_META_FILE, SUMMARY_FILE};

        let dir = TempDir::new();
        let mut cfg = config(dir.path());
        cfg.config = serde_json::json!({ "lr": 2e-5 });
        let mut run = Engine::start(cfg).unwrap();
        let run_id = run.run_id();
        let loss = run.register("loss");
        run.emit(&[(loss, 0.5)]);
        run.emit(&[(loss, 0.4)]);
        run.finish().unwrap();

        // run.meta carries the v4 run_id and mode.
        let meta: RunMetaFile =
            serde_json::from_str(&std::fs::read_to_string(dir.path().join(RUN_META_FILE)).unwrap())
                .unwrap();
        assert_eq!(meta.run_id, run_id);
        assert_eq!(meta.project, "test-run");
        assert!(!meta.mode.is_empty());

        // config.json holds the hyperparameters verbatim.
        let cfg_json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.path().join(CONFIG_FILE)).unwrap())
                .unwrap();
        assert_eq!(cfg_json["lr"], 2e-5);

        // summary.json records the final counts and reason.
        let summary: Summary =
            serde_json::from_str(&std::fs::read_to_string(dir.path().join(SUMMARY_FILE)).unwrap())
                .unwrap();
        assert_eq!(summary.run_id, run_id);
        assert_eq!(summary.steps, 2);
        assert_eq!(summary.reason, FinishReason::Completed);
    }

    #[test]
    fn run_id_is_uuid_v4() {
        let dir = TempDir::new();
        let run = Engine::start(config(dir.path())).unwrap();
        assert_eq!(run.run_id().get_version_num(), 4);
        run.finish().unwrap();
    }

    #[test]
    fn run_exposes_bus_for_observers() {
        let dir = TempDir::new();
        let mut run = Engine::start(config(dir.path())).unwrap();
        let observer = run.bus().subscribe();
        let loss = run.register("loss");
        run.emit(&[(loss, 0.1)]);
        run.finish().unwrap();
        // The observer saw events published during the run.
        assert!(observer.try_iter().count() >= 2); // RunStarted + batch (+finish)
    }
}
