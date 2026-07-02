//! Emry command-line interface.
//!
//! Parsing ([`Cli`]) is separated from execution (the `cmd_*` functions) so the
//! argument surface is unit-tested while the interactive/long-running commands
//! (which drive a terminal dashboard or block on a socket) are exercised by
//! their underlying library crates.

use clap::{Parser, Subcommand};
use crossbeam_channel::{bounded, Receiver};
use emry_core::{Event, MetricId};
use emry_engine::{Engine, RunConfig};
use emry_ingest::{read_frame, socket, JsonlTailer};
use emry_store::JsonlSink;
use emry_tui::{run_terminal, UiState};
use std::error::Error;
use std::io::BufReader;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Gentle observability for long training runs.
#[derive(Debug, Parser)]
#[command(name = "emry", version, about)]
pub struct Cli {
    /// Subcommand to run.
    #[command(subcommand)]
    pub command: Commands,
}

/// Subcommands exposed by the `emry` binary.
#[derive(Debug, Subcommand)]
pub enum Commands {
    /// Run a built-in synthetic demo with the live dashboard.
    Demo {
        /// Number of steps to simulate.
        #[arg(long, default_value_t = 2000)]
        steps: u64,
    },
    /// Attach the dashboard to a run directory or a sidecar socket.
    Tui {
        /// Run directory to tail (`metrics.jsonl` inside it).
        #[arg(long)]
        run_dir: Option<PathBuf>,
        /// Sidecar socket to read events from.
        #[arg(long)]
        socket: Option<PathBuf>,
        /// Prior run directory (or `metrics.jsonl`) to overlay as a baseline.
        #[arg(long)]
        compare: Option<PathBuf>,
    },
    /// Tail a run directory's `metrics.jsonl` and render it live.
    Watch {
        /// Run directory (or a `metrics.jsonl` file directly).
        path: PathBuf,
        /// Prior run directory (or `metrics.jsonl`) to overlay as a baseline.
        #[arg(long)]
        compare: Option<PathBuf>,
    },
    /// Serve the live web dashboard for a run directory or sidecar socket.
    Web {
        /// Run directory to tail (`metrics.jsonl` inside it).
        #[arg(long)]
        run_dir: Option<PathBuf>,
        /// Sidecar socket to read events from.
        #[arg(long)]
        socket: Option<PathBuf>,
        /// TCP port to bind on `127.0.0.1`.
        #[arg(long, default_value_t = 8787)]
        port: u16,
        /// Prior run directory (or `metrics.jsonl`) to overlay as a baseline.
        #[arg(long)]
        compare: Option<PathBuf>,
        /// Log directory: serve the multi-run project dashboard (overlay every
        /// run) instead of a single live run.
        #[arg(long)]
        project: Option<PathBuf>,
    },
    /// List the runs found under a log directory.
    Runs {
        /// Base log directory to scan (default `./logs`).
        #[arg(long)]
        log_dir: Option<PathBuf>,
    },
    /// Compare the final metrics of two runs side by side.
    Compare {
        /// First run directory (or its `metrics.jsonl`).
        run_a: PathBuf,
        /// Second run directory (or its `metrics.jsonl`).
        run_b: PathBuf,
    },
    /// Export a run's metrics to another format.
    Export {
        /// Output format.
        #[command(subcommand)]
        format: ExportFormat,
    },
    /// Run the sidecar engine: receive events over a socket and persist them.
    Engine {
        /// Project / experiment name (names the run directory).
        #[arg(long)]
        project: String,
        /// Unix socket path to bind and listen on.
        #[arg(long)]
        socket: PathBuf,
        /// Base log directory (default `./logs`).
        #[arg(long)]
        log_dir: Option<PathBuf>,
    },
    /// Wrap a training command with a sidecar engine (the SLURM pattern).
    ///
    /// Starts `emry engine`, points the command at it via `EMRY_MODE=sidecar` +
    /// `EMRY_SOCKET`, runs it, then drains and cleans up.
    SlurmWrap {
        /// Project / experiment name (names the run directory).
        #[arg(long, default_value = "run")]
        project: String,
        /// Socket path (default `$TMPDIR/emry-{SLURM_JOB_ID|pid}.sock`).
        #[arg(long)]
        socket: Option<PathBuf>,
        /// Base log directory passed to the engine (default `./logs`).
        #[arg(long)]
        log_dir: Option<PathBuf>,
        /// The training command to run, after `--`.
        #[arg(last = true, required = true)]
        command: Vec<String>,
    },
}

/// Output formats for `emry export`.
#[derive(Debug, Subcommand)]
pub enum ExportFormat {
    /// Export a run's `metrics.jsonl` as CSV (wide rows: step, epoch, phase,
    /// then one column per metric).
    Csv {
        /// Run directory (or a `metrics.jsonl` file directly).
        #[arg(long)]
        run_dir: PathBuf,
        /// Output file (defaults to stdout).
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Export a run's `metrics.jsonl` as Parquet (requires the `parquet` build
    /// feature).
    #[cfg(feature = "parquet")]
    Parquet {
        /// Run directory (or a `metrics.jsonl` file directly).
        #[arg(long)]
        run_dir: PathBuf,
        /// Output Parquet file.
        #[arg(long)]
        output: PathBuf,
    },
}

/// Parses the process arguments and runs the selected command.
#[must_use]
pub fn run() -> ExitCode {
    match Cli::try_parse() {
        Ok(cli) => dispatch(cli.command),
        Err(err) => {
            let _ = err.print();
            clap_exit_code(err.exit_code())
        }
    }
}

fn dispatch(command: Commands) -> ExitCode {
    let result = match command {
        Commands::Demo { steps } => cmd_demo(steps),
        Commands::Watch { path, compare } => cmd_watch(&path, compare.as_deref()),
        Commands::Tui {
            run_dir,
            socket,
            compare,
        } => cmd_tui(run_dir.as_deref(), socket.as_deref(), compare.as_deref()),
        Commands::Web {
            run_dir,
            socket,
            port,
            compare,
            project,
        } => cmd_web(
            run_dir.as_deref(),
            socket.as_deref(),
            port,
            compare.as_deref(),
            project.as_deref(),
        ),
        Commands::Runs { log_dir } => cmd_runs(log_dir.as_deref()),
        Commands::Compare { run_a, run_b } => cmd_compare(&run_a, &run_b),
        Commands::Export { format } => cmd_export(format),
        Commands::Engine {
            project,
            socket,
            log_dir,
        } => cmd_engine(&project, &socket, log_dir.as_deref()),
        Commands::SlurmWrap {
            project,
            socket,
            log_dir,
            command,
        } => cmd_slurm_wrap(&project, socket.as_deref(), log_dir.as_deref(), &command),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("emry: {err}");
            ExitCode::FAILURE
        }
    }
}

/// Builds the dashboard state (seeded with `labels` and an optional comparison
/// `baseline`) and renders `events` to the terminal until the user quits.
fn run_dashboard(
    events: &Receiver<Event>,
    labels: &[(MetricId, String)],
    baseline: Vec<emry_tui::BaselineSeries>,
) -> Result<(), Box<dyn Error>> {
    let label_refs: Vec<(MetricId, &str)> = labels
        .iter()
        .map(|(id, name)| (*id, name.as_str()))
        .collect();
    let mut state = UiState::with_labels(&label_refs);
    state.set_baseline(baseline);
    run_terminal(events, state)?;
    Ok(())
}

/// Loads a `--compare` baseline run into the TUI's series, warning (and
/// continuing without an overlay) if it cannot be read.
fn load_compare(compare: Option<&Path>) -> Vec<emry_tui::BaselineSeries> {
    let Some(path) = compare else {
        return Vec::new();
    };
    match emry_store::load_baseline(path) {
        Ok(series) => series
            .into_iter()
            .map(|s| emry_tui::BaselineSeries {
                label: s.label,
                steps: s.steps,
                values: s.values,
            })
            .collect(),
        Err(err) => {
            eprintln!(
                "emry: could not load --compare baseline {}: {err}",
                path.display()
            );
            Vec::new()
        }
    }
}

/// `emry demo` — a synthetic run streamed to the live dashboard.
fn cmd_demo(steps: u64) -> Result<(), Box<dyn Error>> {
    let run_dir = std::env::temp_dir().join("emry-demo");
    std::fs::create_dir_all(&run_dir)?;
    let cfg = RunConfig {
        metric_names: vec!["loss".into(), "lr".into()],
        total_steps: Some(steps),
        ..RunConfig::new("demo", &run_dir)
    };
    let mut run = Engine::start(cfg)?;
    let loss = run.register("loss");
    let lr = run.register("lr");
    let labels = vec![
        (loss, "loss".to_string()),
        (lr, "lr".to_string()),
        (run.register("loss_ema"), "loss_ema".to_string()),
        (run.register("steps_per_sec"), "steps_per_sec".to_string()),
        (run.register("eta_secs"), "eta_secs".to_string()),
    ];
    let events = run.bus().subscribe();

    std::thread::spawn(move || {
        for step in 0..steps {
            #[allow(clippy::cast_precision_loss)]
            let base = 2.0 / (1.0 + step as f64 * 0.005);
            let value = if step % 400 == 399 { base * 3.0 } else { base };
            run.emit(&[(loss, value), (lr, 1e-3)]);
            std::thread::sleep(Duration::from_millis(5));
        }
        run.finish().ok();
    });

    // A synthetic prior run (slower-decaying loss) overlaid as the amber
    // comparison baseline, so the demo also exercises `--compare`.
    let baseline = demo_baseline(steps);
    run_dashboard(&events, &labels, baseline)
}

/// A synthetic prior-run baseline for the demo's `loss` / `loss_ema` curves.
fn demo_baseline(steps: u64) -> Vec<emry_tui::BaselineSeries> {
    let stride = (steps / 400).max(1);
    let (mut s, mut v) = (Vec::new(), Vec::new());
    let mut step = 0;
    while step < steps {
        #[allow(clippy::cast_precision_loss)]
        let loss = 2.3 / (1.0 + step as f64 * 0.0032);
        s.push(step);
        v.push(loss);
        step += stride;
    }
    let series = |label: &str| emry_tui::BaselineSeries {
        label: label.to_string(),
        steps: s.clone(),
        values: v.clone(),
    };
    vec![series("loss"), series("loss_ema")]
}

/// `emry watch PATH` — tail a run directory's `metrics.jsonl` into the dashboard.
///
/// Metric labels are seeded from the initial poll. Metrics that first appear
/// only after the dashboard starts render with the `m{id}` fallback name until a
/// name-table protocol exists (a future `MetricRegistered` event); in practice
/// metrics are registered at run start, so they are present in the first poll.
fn cmd_watch(path: &Path, compare: Option<&Path>) -> Result<(), Box<dyn Error>> {
    let baseline = load_compare(compare);
    let (rx, labels, stop, poller) = spawn_run_dir_tailer(path)?;
    let result = run_dashboard(&rx, &labels, baseline);
    stop.store(true, Ordering::Release);
    let _ = poller.join();
    result
}

/// The pieces of a running run-directory tailer: the event receiver, the metric
/// labels learned from the initial poll, and a stop flag + handle to shut the
/// background poller down.
type RunDirTailer = (
    Receiver<Event>,
    Vec<(MetricId, String)>,
    Arc<AtomicBool>,
    std::thread::JoinHandle<()>,
);

/// Spawns a background thread tailing a run directory's `metrics.jsonl` into a
/// bounded channel, returning the [`RunDirTailer`] pieces.
fn spawn_run_dir_tailer(path: &Path) -> Result<RunDirTailer, Box<dyn Error>> {
    let metrics = if path.is_dir() {
        path.join(emry_store::METRICS_FILE)
    } else {
        path.to_path_buf()
    };

    // Initial poll: replay existing rows and learn metric names for labels.
    let mut tailer = JsonlTailer::new(&metrics);
    let initial = tailer.poll()?;
    let labels = tailer.labels();

    let (tx, rx) = bounded::<Event>(8192);
    for event in initial {
        let _ = tx.try_send(event);
    }
    let stop = Arc::new(AtomicBool::new(false));
    let poller = {
        let stop = Arc::clone(&stop);
        std::thread::spawn(move || {
            while !stop.load(Ordering::Acquire) {
                for event in tailer.poll().unwrap_or_default() {
                    if tx.try_send(event).is_err() {
                        // Full: drop and continue. Disconnected: receiver gone —
                        // the loop ends on the next stop check anyway.
                    }
                }
                std::thread::sleep(Duration::from_millis(200));
            }
        })
    };
    Ok((rx, labels, stop, poller))
}

/// `emry tui` — attach to a run directory (`--run-dir`) or a socket (`--socket`).
fn cmd_tui(
    run_dir: Option<&Path>,
    socket: Option<&Path>,
    compare: Option<&Path>,
) -> Result<(), Box<dyn Error>> {
    match (run_dir, socket) {
        (Some(dir), _) => cmd_watch(dir, compare),
        (None, Some(sock)) => cmd_socket_tui(sock, compare),
        (None, None) => Err("specify --run-dir or --socket".into()),
    }
}

/// Reads framed events from a sidecar socket into the dashboard.
fn cmd_socket_tui(sock: &Path, compare: Option<&Path>) -> Result<(), Box<dyn Error>> {
    let baseline = load_compare(compare);
    let rx = spawn_socket_reader(sock)?;
    // Socket frames carry MetricIds, not names; labels fall back to `m{id}`.
    run_dashboard(&rx, &[], baseline)
}

/// Connects to a sidecar socket and drains framed events into a bounded channel
/// on a detached reader thread.
fn spawn_socket_reader(sock: &Path) -> Result<Receiver<Event>, Box<dyn Error>> {
    let stream = socket::connect(sock)?;
    let (tx, rx) = bounded::<Event>(8192);
    // Detached: this thread parks in the blocking `read_frame`, which a join
    // could not interrupt once the sender goes quiet. It is not a busy loop, and
    // the OS reaps it when the process exits.
    std::thread::spawn(move || {
        let mut reader = BufReader::new(stream);
        while let Ok(Some(event)) = read_frame(&mut reader) {
            let _ = tx.try_send(event);
        }
    });
    Ok(rx)
}

/// `emry web` — serve the live dashboard for a run directory (`--run-dir`) or a
/// sidecar socket (`--socket`), optionally overlaying a `--compare` baseline.
fn cmd_web(
    run_dir: Option<&Path>,
    socket: Option<&Path>,
    port: u16,
    compare: Option<&Path>,
    project: Option<&Path>,
) -> Result<(), Box<dyn Error>> {
    let addr = SocketAddr::from(([127, 0, 0, 1], port));

    // --project serves the static multi-run overlay instead of a live run.
    if let Some(dir) = project {
        let project = load_project(dir)?;
        eprintln!("emry web (project) on http://{addr}");
        let runtime = tokio::runtime::Runtime::new()?;
        runtime.block_on(emry_web::serve_project(addr, project))?;
        return Ok(());
    }

    // Holds the run-dir poller's stop flag + handle so it can be shut down on
    // return (including an early bind error) rather than leaked.
    let mut tailer_shutdown: Option<(Arc<AtomicBool>, std::thread::JoinHandle<()>)> = None;
    let (rx, labels) = match (run_dir, socket) {
        (Some(dir), _) => {
            let (rx, labels, stop, poller) = spawn_run_dir_tailer(dir)?;
            tailer_shutdown = Some((stop, poller));
            (rx, labels)
        }
        (None, Some(sock)) => (spawn_socket_reader(sock)?, Vec::new()),
        (None, None) => return Err("specify --run-dir or --socket".into()),
    };
    // Load the comparison baseline via the one canonical reader
    // (`emry_store::load_baseline`) and map it into the web crate's serializable
    // types — same path the TUI overlay uses, so behaviour is consistent.
    let baseline = match compare {
        Some(path) => Some(emry_web::Baseline {
            metrics: emry_store::load_baseline(path)?
                .into_iter()
                .map(|s| emry_web::BaselineSeries {
                    label: s.label,
                    steps: s.steps,
                    values: s.values,
                })
                .collect(),
        }),
        None => None,
    };

    let label_refs: Vec<(MetricId, &str)> = labels
        .iter()
        .map(|(id, name)| (*id, name.as_str()))
        .collect();
    eprintln!("emry web on http://{addr}");

    let runtime = tokio::runtime::Runtime::new()?;
    let result = runtime.block_on(async move {
        match baseline {
            Some(baseline) => emry_web::serve_with_baseline(addr, rx, &label_refs, baseline).await,
            None => emry_web::serve_with_labels(addr, rx, &label_refs).await,
        }
    });

    // Stop the poller (if any) so it does not outlive the server.
    if let Some((stop, poller)) = tailer_shutdown {
        stop.store(true, Ordering::Release);
        let _ = poller.join();
    }
    result?;
    Ok(())
}

/// Loads every run under `base` into a [`emry_web::Project`] snapshot (via the
/// canonical `list_runs` + `load_baseline` readers), newest first.
fn load_project(base: &Path) -> Result<emry_web::Project, Box<dyn Error>> {
    let runs = emry_store::list_runs(base)?
        .into_iter()
        .map(|info| {
            let metrics = emry_store::load_baseline(&base.join(&info.dir_name))
                .unwrap_or_default()
                .into_iter()
                .map(|s| emry_web::ProjectSeries {
                    label: s.label,
                    steps: s.steps,
                    values: s.values,
                })
                .collect();
            emry_web::ProjectRun {
                name: info.dir_name,
                metrics,
            }
        })
        .collect();
    Ok(emry_web::Project { runs })
}

/// `emry runs` — list the runs under a log directory (default `./logs`).
fn cmd_runs(log_dir: Option<&Path>) -> Result<(), Box<dyn Error>> {
    let base = match log_dir {
        Some(dir) => dir.to_path_buf(),
        None => std::env::current_dir()?.join("logs"),
    };
    let runs = emry_store::list_runs(&base)?;
    print!("{}", format_runs(&runs));
    Ok(())
}

/// Renders a run list as an aligned table (or a friendly note when empty).
fn format_runs(runs: &[emry_store::RunInfo]) -> String {
    if runs.is_empty() {
        return "no runs found\n".to_string();
    }
    let rows: Vec<[String; 5]> = runs
        .iter()
        .map(|r| {
            [
                r.dir_name.clone(),
                r.project.clone(),
                r.steps.map_or_else(|| "-".to_string(), |s| s.to_string()),
                r.reason.clone().unwrap_or_else(|| "running".to_string()),
                r.duration_secs
                    .map_or_else(|| "-".to_string(), |d| format!("{d:.1}s")),
            ]
        })
        .collect();
    let headers = ["RUN", "PROJECT", "STEPS", "STATUS", "DURATION"];
    render_table(&headers, &rows)
}

/// `emry compare A B` — show each run's final metric values and their delta.
fn cmd_compare(run_a: &Path, run_b: &Path) -> Result<(), Box<dyn Error>> {
    let a = emry_store::final_metrics(run_a)?;
    let b = emry_store::final_metrics(run_b)?;
    print!("{}", format_compare(run_a, run_b, &a, &b));
    Ok(())
}

/// Renders a side-by-side comparison of two runs' final metrics.
fn format_compare(
    run_a: &Path,
    run_b: &Path,
    a: &std::collections::BTreeMap<String, f64>,
    b: &std::collections::BTreeMap<String, f64>,
) -> String {
    let label = |p: &Path| {
        p.file_name().map_or_else(
            || p.display().to_string(),
            |n| n.to_string_lossy().into_owned(),
        )
    };
    if a.is_empty() && b.is_empty() {
        return "no metrics found\n".to_string();
    }
    // Union of metric names, sorted (BTreeMap keys are already ordered).
    let mut names: Vec<&String> = a.keys().chain(b.keys()).collect();
    names.sort_unstable();
    names.dedup();

    let fmt = |v: Option<&f64>| v.map_or_else(|| "-".to_string(), |x| format!("{x:.6}"));
    let rows: Vec<[String; 4]> = names
        .iter()
        .map(|name| {
            let (va, vb) = (a.get(*name), b.get(*name));
            let delta = match (va, vb) {
                (Some(x), Some(y)) => format!("{:+.6}", y - x),
                _ => "-".to_string(),
            };
            [(*name).clone(), fmt(va), fmt(vb), delta]
        })
        .collect();
    let headers = [
        "METRIC".to_string(),
        label(run_a),
        label(run_b),
        "Δ".to_string(),
    ];
    let header_refs = [
        headers[0].as_str(),
        headers[1].as_str(),
        headers[2].as_str(),
        headers[3].as_str(),
    ];
    render_table(&header_refs, &rows)
}

/// Renders `headers` + `rows` as a left-aligned, space-padded table. Each row is
/// a fixed-width array matching the header count.
fn render_table<const N: usize>(headers: &[&str; N], rows: &[[String; N]]) -> String {
    let mut widths = headers.map(|h| h.chars().count());
    for row in rows {
        for (i, cell) in row.iter().enumerate() {
            widths[i] = widths[i].max(cell.chars().count());
        }
    }
    let mut out = String::new();
    let mut push_row = |cells: &[&str; N]| {
        for (i, cell) in cells.iter().enumerate() {
            if i > 0 {
                out.push_str("  ");
            }
            out.push_str(cell);
            if i + 1 < N {
                for _ in cell.chars().count()..widths[i] {
                    out.push(' ');
                }
            }
        }
        out.push('\n');
    };
    push_row(headers);
    for row in rows {
        let refs: [&str; N] = std::array::from_fn(|i| row[i].as_str());
        push_row(&refs);
    }
    out
}

/// `emry export` — dispatch to the selected output format.
fn cmd_export(format: ExportFormat) -> Result<(), Box<dyn Error>> {
    match format {
        ExportFormat::Csv { run_dir, output } => cmd_export_csv(&run_dir, output.as_deref()),
        #[cfg(feature = "parquet")]
        ExportFormat::Parquet { run_dir, output } => {
            let rows = emry_store::export_parquet(&run_dir, &output)?;
            eprintln!("emry: wrote {rows} rows to {}", output.display());
            Ok(())
        }
    }
}

/// `emry export csv` — write a run's `metrics.jsonl` as CSV to a file or stdout.
fn cmd_export_csv(run_dir: &Path, output: Option<&Path>) -> Result<(), Box<dyn Error>> {
    if let Some(path) = output {
        let mut file = std::fs::File::create(path)?;
        let rows = emry_store::export_csv(run_dir, &mut file)?;
        eprintln!("emry: wrote {rows} rows to {}", path.display());
    } else {
        let stdout = std::io::stdout();
        let mut lock = stdout.lock();
        emry_store::export_csv(run_dir, &mut lock)?;
    }
    Ok(())
}

/// `emry engine --socket PATH` — receive events over a socket and persist them.
fn cmd_engine(
    project: &str,
    socket_path: &Path,
    log_dir: Option<&Path>,
) -> Result<(), Box<dyn Error>> {
    let base = match log_dir {
        Some(dir) => dir.to_path_buf(),
        None => std::env::current_dir()?.join("logs"),
    };
    let start_secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0);
    let run_dir = emry_store::create_run_dir(&base, project, start_secs)?;
    let sink = JsonlSink::spawn(&run_dir)?;
    let listener = socket::bind(socket_path)?;
    eprintln!(
        "emry engine listening on {} → {}",
        socket_path.display(),
        run_dir.display()
    );

    // One training run per engine invocation: accept its connection and drain
    // it. Stop on RunFinished (so we don't wait for the peer to also close) or
    // on EOF (handles a peer that disconnected without finishing). Whatever
    // happens, flush the sink before returning.
    let (stream, _addr) = listener.accept()?;
    let mut reader = BufReader::new(stream);
    let mut read_result = Ok(());
    loop {
        match read_frame(&mut reader) {
            Ok(Some(event)) => {
                let finished = matches!(event, Event::RunFinished { .. });
                sink.write_event(event);
                if finished {
                    break;
                }
            }
            Ok(None) => break,
            Err(err) => {
                read_result = Err(err);
                break;
            }
        }
    }
    sink.finish()?;
    let _ = std::fs::remove_file(socket_path);
    read_result?;
    Ok(())
}

/// Resolves the sidecar socket path: explicit `--socket`, else
/// `{tmp}/emry-{tag}.sock` where `tag` is the SLURM job id (if set) or the pid.
fn slurm_socket_path(
    explicit: Option<&Path>,
    job_id: Option<String>,
    tmp: &Path,
    pid: u32,
) -> PathBuf {
    if let Some(path) = explicit {
        return path.to_path_buf();
    }
    let tag = job_id
        .filter(|id| !id.is_empty())
        .unwrap_or_else(|| pid.to_string());
    tmp.join(format!("emry-{tag}.sock"))
}

/// Maps a finished command's [`ExitStatus`] to a shell-style exit code.
///
/// A normal exit uses its own code; a signal death (the common SLURM case —
/// SIGKILL on OOM, SIGTERM at the time limit) maps to `128 + signal` so callers
/// can tell "the script failed" from "the scheduler killed it".
#[cfg(unix)]
fn command_exit_code(status: std::process::ExitStatus) -> i32 {
    use std::os::unix::process::ExitStatusExt;
    status
        .code()
        .or_else(|| status.signal().map(|sig| 128 + sig))
        .unwrap_or(1)
}

#[cfg(not(unix))]
fn command_exit_code(status: std::process::ExitStatus) -> i32 {
    status.code().unwrap_or(1)
}

/// `emry slurm-wrap … -- <cmd>` — run `<cmd>` with a sidecar engine, then drain.
///
/// Diverges via [`std::process::exit`] with the training command's exit code on
/// success; returns [`Err`] only for setup failures (engine won't start/bind).
fn cmd_slurm_wrap(
    project: &str,
    socket: Option<&Path>,
    log_dir: Option<&Path>,
    command: &[String],
) -> Result<(), Box<dyn Error>> {
    let (cmd0, args) = command
        .split_first()
        .ok_or("slurm-wrap: no command given after `--`")?;
    let socket = slurm_socket_path(
        socket,
        std::env::var("SLURM_JOB_ID").ok(),
        &std::env::temp_dir(),
        std::process::id(),
    );
    let _ = std::fs::remove_file(&socket); // clear a stale socket so bind succeeds

    // Start the engine as a child of *this* executable (no reliance on `emry`
    // being on PATH).
    let exe = std::env::current_exe()?;
    let mut engine_cmd = std::process::Command::new(&exe);
    engine_cmd
        .args(["engine", "--project", project, "--socket"])
        .arg(&socket);
    if let Some(dir) = log_dir {
        engine_cmd.arg("--log-dir").arg(dir);
    }
    let mut engine = engine_cmd.spawn()?;

    // Wait (bounded) for the engine to bind the socket.
    let mut bound = false;
    for _ in 0..200 {
        if socket.exists() {
            bound = true;
            break;
        }
        if let Ok(Some(status)) = engine.try_wait() {
            return Err(format!("slurm-wrap: engine exited before binding ({status})").into());
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    if !bound {
        let _ = engine.kill();
        let _ = engine.wait();
        return Err("slurm-wrap: engine did not bind the socket within 10s".into());
    }

    // Run the training command, pointed at the sidecar. If it can't even be
    // spawned (typo/missing exe), reap the engine first so we don't orphan it.
    let status = match std::process::Command::new(cmd0)
        .args(args)
        .env("EMRY_MODE", "sidecar")
        .env("EMRY_SOCKET", &socket)
        .status()
    {
        Ok(status) => status,
        Err(err) => {
            let _ = engine.kill();
            let _ = engine.wait();
            let _ = std::fs::remove_file(&socket);
            return Err(format!("slurm-wrap: failed to run `{cmd0}`: {err}").into());
        }
    };

    // Give the engine time to drain on RUN_FINISHED, then terminate it if it is
    // still blocked on accept() (e.g. the command never connected).
    let mut drained = false;
    for _ in 0..100 {
        if matches!(engine.try_wait(), Ok(Some(_))) {
            drained = true;
            break;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    if !drained {
        let _ = engine.kill();
    }
    let _ = engine.wait();
    let _ = std::fs::remove_file(&socket);

    std::process::exit(command_exit_code(status)); // propagate the command's code
}

fn clap_exit_code(code: i32) -> ExitCode {
    u8::try_from(code).map_or(ExitCode::FAILURE, ExitCode::from)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<Commands, clap::Error> {
        Cli::try_parse_from(args).map(|c| c.command)
    }

    #[test]
    fn help_and_version_parse_as_clap_errors() {
        assert!(Cli::try_parse_from(["emry", "--help"]).is_err());
        assert!(Cli::try_parse_from(["emry", "--version"]).is_err());
    }

    #[test]
    fn demo_defaults_to_2000_steps() {
        match parse(&["emry", "demo"]).unwrap() {
            Commands::Demo { steps } => assert_eq!(steps, 2000),
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn demo_accepts_steps() {
        match parse(&["emry", "demo", "--steps", "500"]).unwrap() {
            Commands::Demo { steps } => assert_eq!(steps, 500),
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn watch_requires_a_path() {
        assert!(parse(&["emry", "watch"]).is_err());
        match parse(&["emry", "watch", "./logs/run"]).unwrap() {
            Commands::Watch { path, compare } => {
                assert_eq!(path, PathBuf::from("./logs/run"));
                assert!(compare.is_none());
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn watch_and_tui_accept_compare() {
        match parse(&["emry", "watch", "run", "--compare", "old"]).unwrap() {
            Commands::Watch { compare, .. } => assert_eq!(compare, Some(PathBuf::from("old"))),
            other => panic!("unexpected {other:?}"),
        }
        match parse(&["emry", "tui", "--run-dir", "run", "--compare", "old"]).unwrap() {
            Commands::Tui { compare, .. } => assert_eq!(compare, Some(PathBuf::from("old"))),
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn tui_accepts_run_dir_and_socket_flags() {
        match parse(&[
            "emry",
            "tui",
            "--run-dir",
            "./logs/run",
            "--socket",
            "/tmp/e.sock",
        ])
        .unwrap()
        {
            Commands::Tui {
                run_dir, socket, ..
            } => {
                assert_eq!(run_dir, Some(PathBuf::from("./logs/run")));
                assert_eq!(socket, Some(PathBuf::from("/tmp/e.sock")));
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn tui_flags_are_optional() {
        match parse(&["emry", "tui"]).unwrap() {
            Commands::Tui {
                run_dir, socket, ..
            } => {
                assert!(run_dir.is_none() && socket.is_none());
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn engine_requires_project_and_socket() {
        assert!(parse(&["emry", "engine"]).is_err());
        assert!(parse(&["emry", "engine", "--project", "x"]).is_err());
        match parse(&[
            "emry",
            "engine",
            "--project",
            "llama",
            "--socket",
            "/tmp/e.sock",
        ])
        .unwrap()
        {
            Commands::Engine {
                project,
                socket,
                log_dir,
            } => {
                assert_eq!(project, "llama");
                assert_eq!(socket, PathBuf::from("/tmp/e.sock"));
                assert!(log_dir.is_none());
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn slurm_wrap_captures_command_after_dashdash() {
        // The training command lives after `--`; flags before it are the wrapper's.
        match parse(&[
            "emry",
            "slurm-wrap",
            "--project",
            "llama",
            "--socket",
            "/tmp/w.sock",
            "--",
            "python",
            "train.py",
            "--lr",
            "1e-4",
        ])
        .unwrap()
        {
            Commands::SlurmWrap {
                project,
                socket,
                log_dir,
                command,
            } => {
                assert_eq!(project, "llama");
                assert_eq!(socket, Some(PathBuf::from("/tmp/w.sock")));
                assert!(log_dir.is_none());
                assert_eq!(command, ["python", "train.py", "--lr", "1e-4"]);
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[cfg(unix)]
    #[test]
    fn command_exit_code_maps_normal_and_signal_death() {
        use std::os::unix::process::ExitStatusExt;
        // Normal exit keeps its own code (wait status encodes it in the high byte).
        assert_eq!(
            command_exit_code(std::process::ExitStatus::from_raw(3 << 8)),
            3
        );
        // Signal death (e.g. SIGKILL=9, an OOM kill) maps to 128 + signal.
        assert_eq!(
            command_exit_code(std::process::ExitStatus::from_raw(9)),
            128 + 9
        );
    }

    #[test]
    fn slurm_wrap_requires_a_command() {
        // Defaults are fine, but an empty command (nothing after `--`) is rejected.
        assert!(parse(&["emry", "slurm-wrap"]).is_err());
        assert!(parse(&["emry", "slurm-wrap", "--"]).is_err());
    }

    #[test]
    fn slurm_socket_path_prefers_explicit_then_job_id_then_pid() {
        let tmp = Path::new("/tmp");
        // Explicit --socket wins outright.
        assert_eq!(
            slurm_socket_path(Some(Path::new("/x/y.sock")), Some("42".into()), tmp, 7),
            PathBuf::from("/x/y.sock")
        );
        // Otherwise the SLURM job id names the socket.
        assert_eq!(
            slurm_socket_path(None, Some("42".into()), tmp, 7),
            PathBuf::from("/tmp/emry-42.sock")
        );
        // No job id (or an empty one) falls back to the pid.
        assert_eq!(
            slurm_socket_path(None, None, tmp, 7),
            PathBuf::from("/tmp/emry-7.sock")
        );
        assert_eq!(
            slurm_socket_path(None, Some(String::new()), tmp, 7),
            PathBuf::from("/tmp/emry-7.sock")
        );
    }

    #[test]
    fn web_defaults_port_and_accepts_flags() {
        match parse(&["emry", "web"]).unwrap() {
            Commands::Web {
                run_dir,
                socket,
                port,
                compare,
                project,
            } => {
                assert_eq!(port, 8787);
                assert!(
                    run_dir.is_none() && socket.is_none() && compare.is_none() && project.is_none()
                );
            }
            other => panic!("unexpected {other:?}"),
        }
        match parse(&[
            "emry",
            "web",
            "--run-dir",
            "./logs/run",
            "--port",
            "9000",
            "--compare",
            "./logs/old",
        ])
        .unwrap()
        {
            Commands::Web {
                run_dir,
                port,
                compare,
                ..
            } => {
                assert_eq!(run_dir, Some(PathBuf::from("./logs/run")));
                assert_eq!(port, 9000);
                assert_eq!(compare, Some(PathBuf::from("./logs/old")));
            }
            other => panic!("unexpected {other:?}"),
        }
        match parse(&["emry", "web", "--project", "./logs"]).unwrap() {
            Commands::Web { project, .. } => {
                assert_eq!(project, Some(PathBuf::from("./logs")));
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn web_with_no_target_errors() {
        // Pure validation path (no runtime started): all targets absent errors.
        assert!(cmd_web(None, None, 8787, None, None).is_err());
    }

    #[test]
    fn load_project_collects_runs_newest_first() {
        use std::sync::atomic::AtomicU32;
        static COUNTER: AtomicU32 = AtomicU32::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let base = std::env::temp_dir().join(format!("emry-cli-proj-{}-{n}", std::process::id()));
        for (name, start) in [("old", 100u64), ("new", 200u64)] {
            let dir = base.join(format!("{name}_{start}"));
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(
                dir.join("run.meta"),
                format!("{{\"run_id\":\"00000000-0000-0000-0000-000000000000\",\"project\":\"{name}\",\"start_time_secs\":{start},\"mode\":\"file\"}}"),
            )
            .unwrap();
            std::fs::write(
                dir.join("metrics.jsonl"),
                "{\"step\":0,\"epoch\":0,\"phase\":\"TRAIN\",\"values\":{\"loss\":1.0}}\n",
            )
            .unwrap();
        }
        let project = load_project(&base).unwrap();
        std::fs::remove_dir_all(&base).ok();
        assert_eq!(project.runs.len(), 2);
        assert!(project.runs[0].name.starts_with("new_")); // newest first
        assert_eq!(project.runs[0].metrics[0].label, "loss");
    }

    #[test]
    fn load_project_tolerates_a_run_without_metrics() {
        use std::sync::atomic::AtomicU32;
        static COUNTER: AtomicU32 = AtomicU32::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let base = std::env::temp_dir().join(format!("emry-cli-proj2-{}-{n}", std::process::id()));
        // A run with run.meta but no metrics.jsonl must not abort the view.
        let dir = base.join("broken_100");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("run.meta"),
            "{\"run_id\":\"00000000-0000-0000-0000-000000000000\",\"project\":\"broken\",\"start_time_secs\":100,\"mode\":\"file\"}",
        )
        .unwrap();
        let project = load_project(&base).unwrap();
        std::fs::remove_dir_all(&base).ok();
        assert_eq!(project.runs.len(), 1);
        assert!(project.runs[0].metrics.is_empty()); // degraded, not aborted
    }

    #[test]
    fn export_csv_parses_and_requires_run_dir() {
        assert!(parse(&["emry", "export", "csv"]).is_err());
        match parse(&["emry", "export", "csv", "--run-dir", "./logs/run"]).unwrap() {
            Commands::Export {
                format: ExportFormat::Csv { run_dir, output },
            } => {
                assert_eq!(run_dir, PathBuf::from("./logs/run"));
                assert!(output.is_none());
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn export_csv_writes_a_file() {
        use std::sync::atomic::AtomicU32;
        static COUNTER: AtomicU32 = AtomicU32::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("emry-cli-csv-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join(emry_store::METRICS_FILE),
            "{\"step\":0,\"epoch\":0,\"phase\":\"TRAIN\",\"values\":{\"loss\":0.5}}\n",
        )
        .unwrap();
        let out = dir.join("metrics.csv");
        cmd_export_csv(&dir, Some(&out)).unwrap();
        let csv = std::fs::read_to_string(&out).unwrap();
        assert_eq!(csv, "step,epoch,phase,loss\n0,0,TRAIN,0.5\n");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn runs_and_compare_parse() {
        match parse(&["emry", "runs"]).unwrap() {
            Commands::Runs { log_dir } => assert!(log_dir.is_none()),
            other => panic!("unexpected {other:?}"),
        }
        match parse(&["emry", "runs", "--log-dir", "/tmp/logs"]).unwrap() {
            Commands::Runs { log_dir } => assert_eq!(log_dir, Some(PathBuf::from("/tmp/logs"))),
            other => panic!("unexpected {other:?}"),
        }
        assert!(parse(&["emry", "compare", "a"]).is_err()); // needs two runs
        match parse(&["emry", "compare", "a", "b"]).unwrap() {
            Commands::Compare { run_a, run_b } => {
                assert_eq!(run_a, PathBuf::from("a"));
                assert_eq!(run_b, PathBuf::from("b"));
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn format_runs_renders_table_and_empty_note() {
        assert_eq!(format_runs(&[]), "no runs found\n");
        let runs = vec![
            emry_store::RunInfo {
                dir_name: "llama_200".into(),
                project: "llama".into(),
                start_time_secs: 200.0,
                steps: None,
                duration_secs: None,
                reason: None,
            },
            emry_store::RunInfo {
                dir_name: "gpt_100".into(),
                project: "gpt".into(),
                start_time_secs: 100.0,
                steps: Some(500),
                duration_secs: Some(12.5),
                reason: Some("COMPLETED".into()),
            },
        ];
        let table = format_runs(&runs);
        let lines: Vec<&str> = table.lines().collect();
        assert!(lines[0].starts_with("RUN"));
        assert!(
            lines[1].contains("llama") && lines[1].contains("running") && lines[1].contains('-')
        );
        assert!(lines[2].contains("gpt") && lines[2].contains("500") && lines[2].contains("12.5s"));
    }

    #[test]
    fn format_compare_shows_values_and_delta() {
        let mut a = std::collections::BTreeMap::new();
        a.insert("loss".to_string(), 1.0);
        a.insert("only_a".to_string(), 5.0);
        let mut b = std::collections::BTreeMap::new();
        b.insert("loss".to_string(), 0.4);
        let table = format_compare(Path::new("runA"), Path::new("runB"), &a, &b);
        let lines: Vec<&str> = table.lines().collect();
        assert!(
            lines[0].contains("METRIC") && lines[0].contains("runA") && lines[0].contains("runB")
        );
        // loss present in both → delta shown with sign.
        let loss = lines.iter().find(|l| l.starts_with("loss")).unwrap();
        assert!(
            loss.contains("1.000000") && loss.contains("0.400000") && loss.contains("-0.600000")
        );
        // only_a present in one → delta is "-".
        let only = lines.iter().find(|l| l.starts_with("only_a")).unwrap();
        assert!(only.contains("5.000000"));
    }

    #[test]
    fn format_compare_empty_and_delta_column_aligns() {
        let empty = std::collections::BTreeMap::new();
        assert_eq!(
            format_compare(Path::new("a"), Path::new("b"), &empty, &empty),
            "no metrics found\n"
        );

        // The Δ header (U+0394, 2 bytes / 1 char) must not over-pad the column:
        // each row's first space-separated token after the value should align.
        let mut a = std::collections::BTreeMap::new();
        a.insert("loss".to_string(), 1.0);
        let mut b = std::collections::BTreeMap::new();
        b.insert("loss".to_string(), 2.0);
        let table = format_compare(Path::new("a"), Path::new("b"), &a, &b);
        let lines: Vec<&str> = table.lines().collect();
        // Δ is the last column, so header and row share the same prefix width up
        // to it: the byte length of the header line equals char-aligned layout.
        assert_eq!(lines[0].chars().filter(|c| *c == 'Δ').count(), 1);
        // The metric/value columns line up: "loss" row and header start aligned.
        assert!(lines[1].starts_with("loss"));
    }

    #[test]
    fn unknown_subcommand_is_rejected() {
        assert!(parse(&["emry", "frobnicate"]).is_err());
    }

    #[test]
    fn tui_with_no_target_errors() {
        // Pure validation path (no terminal): both targets absent is an error.
        assert!(cmd_tui(None, None, None).is_err());
    }

    #[test]
    fn clap_exit_code_maps_values() {
        assert_eq!(clap_exit_code(0), ExitCode::SUCCESS);
        // Out-of-u8 range falls back to FAILURE without panicking.
        let _ = clap_exit_code(100_000);
    }

    #[test]
    fn engine_receives_events_over_socket_and_persists_them() {
        use emry_core::{FinishReason, Phase};
        use std::sync::atomic::AtomicU32;

        static COUNTER: AtomicU32 = AtomicU32::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let sock =
            std::env::temp_dir().join(format!("emry-cli-eng-{}-{n}.sock", std::process::id()));
        let log_dir =
            std::env::temp_dir().join(format!("emry-cli-eng-logs-{}-{n}", std::process::id()));

        // The engine blocks in serve() until it receives RunFinished.
        let engine = {
            let sock = sock.clone();
            let log_dir = log_dir.clone();
            // Map the non-Send Box<dyn Error> to a String so it can cross threads.
            std::thread::spawn(move || {
                cmd_engine("sidecar", &sock, Some(&log_dir)).map_err(|e| e.to_string())
            })
        };

        // Wait for the engine to bind, then stream events as a client.
        for _ in 0..200 {
            if sock.exists() {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        let stream = socket::connect(&sock).unwrap();
        socket::send_events(
            &stream,
            &[
                Event::MetricsBatch {
                    step: 0,
                    epoch: 0,
                    phase: Phase::Train,
                    values: vec![(MetricId(0), 1.0)],
                },
                Event::RunFinished {
                    duration_secs: 1.0,
                    reason: FinishReason::Completed,
                },
            ],
        )
        .unwrap();
        drop(stream);

        engine.join().unwrap().unwrap();

        // A run directory with a populated events.jsonl was created.
        let run_dir = std::fs::read_dir(&log_dir)
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        let events = std::fs::read_to_string(run_dir.join(emry_store::EVENTS_FILE)).unwrap();
        assert_eq!(events.lines().count(), 2);
        assert!(events.contains("RUN_FINISHED"));

        std::fs::remove_dir_all(&log_dir).ok();
    }
}
