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
    },
    /// Tail a run directory's `metrics.jsonl` and render it live.
    Watch {
        /// Run directory (or a `metrics.jsonl` file directly).
        path: PathBuf,
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
        Commands::Watch { path } => cmd_watch(&path),
        Commands::Tui { run_dir, socket } => cmd_tui(run_dir.as_deref(), socket.as_deref()),
        Commands::Engine {
            project,
            socket,
            log_dir,
        } => cmd_engine(&project, &socket, log_dir.as_deref()),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("emry: {err}");
            ExitCode::FAILURE
        }
    }
}

/// Builds the dashboard state (seeded with `labels`) and renders `events` to the
/// terminal until the user quits.
fn run_dashboard(
    events: &Receiver<Event>,
    labels: &[(MetricId, String)],
) -> Result<(), Box<dyn Error>> {
    let label_refs: Vec<(MetricId, &str)> = labels
        .iter()
        .map(|(id, name)| (*id, name.as_str()))
        .collect();
    run_terminal(events, UiState::with_labels(&label_refs))?;
    Ok(())
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

    run_dashboard(&events, &labels)
}

/// `emry watch PATH` — tail a run directory's `metrics.jsonl` into the dashboard.
///
/// Metric labels are seeded from the initial poll. Metrics that first appear
/// only after the dashboard starts render with the `m{id}` fallback name until a
/// name-table protocol exists (a future `MetricRegistered` event); in practice
/// metrics are registered at run start, so they are present in the first poll.
fn cmd_watch(path: &Path) -> Result<(), Box<dyn Error>> {
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

    let result = run_dashboard(&rx, &labels);
    stop.store(true, Ordering::Release);
    let _ = poller.join();
    result
}

/// `emry tui` — attach to a run directory (`--run-dir`) or a socket (`--socket`).
fn cmd_tui(run_dir: Option<&Path>, socket: Option<&Path>) -> Result<(), Box<dyn Error>> {
    match (run_dir, socket) {
        (Some(dir), _) => cmd_watch(dir),
        (None, Some(sock)) => cmd_socket_tui(sock),
        (None, None) => Err("specify --run-dir or --socket".into()),
    }
}

/// Reads framed events from a sidecar socket into the dashboard.
fn cmd_socket_tui(sock: &Path) -> Result<(), Box<dyn Error>> {
    let stream = socket::connect(sock)?;
    let (tx, rx) = bounded::<Event>(8192);
    // Detached: this thread parks in the blocking `read_frame`, which a join
    // could not interrupt once the sender goes quiet. It is not a busy loop, and
    // the OS reaps it when the process exits after the dashboard returns.
    std::thread::spawn(move || {
        let mut reader = BufReader::new(stream);
        while let Ok(Some(event)) = read_frame(&mut reader) {
            let _ = tx.try_send(event);
        }
    });
    // Socket frames carry MetricIds, not names; labels fall back to `m{id}`.
    run_dashboard(&rx, &[])
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
            Commands::Watch { path } => assert_eq!(path, PathBuf::from("./logs/run")),
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
            Commands::Tui { run_dir, socket } => {
                assert_eq!(run_dir, Some(PathBuf::from("./logs/run")));
                assert_eq!(socket, Some(PathBuf::from("/tmp/e.sock")));
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn tui_flags_are_optional() {
        match parse(&["emry", "tui"]).unwrap() {
            Commands::Tui { run_dir, socket } => {
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
    fn unknown_subcommand_is_rejected() {
        assert!(parse(&["emry", "frobnicate"]).is_err());
    }

    #[test]
    fn tui_with_no_target_errors() {
        // Pure validation path (no terminal): both targets absent is an error.
        assert!(cmd_tui(None, None).is_err());
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
