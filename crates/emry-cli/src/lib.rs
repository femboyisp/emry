//! Emry command-line interface library.

use clap::{Parser, Subcommand};
use std::ffi::OsString;
use std::fmt::Write as _;
use std::path::PathBuf;
use std::process::ExitCode;

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
    /// Run a built-in demo feed (development).
    Demo,
    /// Attach a TUI to a run directory or socket.
    Tui {
        /// Path to a run log directory.
        #[arg(long)]
        run_dir: Option<PathBuf>,
        /// Unix socket path for sidecar mode.
        #[arg(long)]
        socket: Option<PathBuf>,
    },
    /// Tail a metrics JSONL directory and render live.
    Watch {
        /// Run directory containing metrics.jsonl.
        path: PathBuf,
    },
}

/// Execute parsed CLI arguments and return a process exit code.
pub fn execute_from<I, T>(args: I) -> ExitCode
where
    I: IntoIterator<Item = T>,
    T: Into<OsString> + Clone,
{
    match Cli::try_parse_from(args) {
        Ok(cli) => dispatch(cli.command),
        Err(err) => {
            if err.use_stderr() {
                err.print().expect("write clap error to stderr");
            }
            clap_exit_code(err.exit_code())
        }
    }
}

fn dispatch(command: Commands) -> ExitCode {
    let mut out = String::new();
    match command {
        Commands::Demo => {
            writeln!(out, "emry demo — not implemented yet (EMRY-021)").expect("format");
        }
        Commands::Tui { run_dir, socket } => {
            writeln!(
                out,
                "emry tui — not implemented yet (run_dir={run_dir:?}, socket={socket:?})"
            )
            .expect("format");
        }
        Commands::Watch { path } => {
            writeln!(
                out,
                "emry watch — not implemented yet (path={})",
                path.display()
            )
            .expect("format");
        }
    }
    eprint!("{out}");
    ExitCode::SUCCESS
}

fn clap_exit_code(code: i32) -> ExitCode {
    match u8::try_from(code) {
        Ok(byte) => ExitCode::from(byte),
        Err(_) => ExitCode::FAILURE,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::ExitCode;

    #[test]
    fn help_succeeds() {
        let code = execute_from(["emry", "--help"]);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn demo_subcommand_succeeds() {
        let code = execute_from(["emry", "demo"]);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn watch_requires_path() {
        let code = execute_from(["emry", "watch"]);
        assert_ne!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn watch_parses_path() {
        let code = execute_from(["emry", "watch", "./logs/run"]);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn tui_accepts_flags() {
        let code = execute_from([
            "emry",
            "tui",
            "--run-dir",
            "./logs/run",
            "--socket",
            "/tmp/emry.sock",
        ]);
        assert_eq!(code, ExitCode::SUCCESS);
    }
}
