//! Deployment-mode detection: where the Emry engine runs relative to training.
//!
//! # Modes
//!
//! - [`DeployMode::Embedded`] — engine runs in-process (`PyO3`); local dev with a TTY.
//! - [`DeployMode::Sidecar`] — engine runs as a separate process reached over a
//!   socket; the default on clusters (SSH/SLURM) so training survives an engine
//!   crash.
//! - [`DeployMode::File`] — no live engine; events are written to JSONL only.
//!
//! # Resolution precedence
//!
//! 1. An explicit API argument (e.g. `mode=` passed by the caller).
//! 2. The `EMRY_MODE` environment variable (`embedded` | `sidecar` | `file`,
//!    case-insensitive; an unrecognized, empty, or whitespace-only value is
//!    ignored and falls through to auto-detection).
//! 3. Auto-detection: `SSH_CONNECTION` or `SLURM_JOB_ID` present → sidecar;
//!    otherwise a TTY on stdout → embedded; otherwise → file.
//!
//! # Environment variables
//!
//! Emry reads these `EMRY_*` variables (others are consumed by their own
//! subsystems; listed here as the canonical reference):
//!
//! - `EMRY_MODE` — override the deploy mode (see precedence above).
//! - `EMRY_LIVE` — observer to launch: `auto` | `tui` | `web` | `both` | `off`.
//! - `EMRY_SOCKET` — Unix socket path for sidecar mode.
//! - `EMRY_LOG_DIR` — base directory for run log directories.

use std::str::FromStr;

/// Where the Emry engine runs relative to the training process.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeployMode {
    /// Engine embedded in the training process (`PyO3`); local dev.
    Embedded,
    /// Engine in a separate process reached over a socket; clusters.
    Sidecar,
    /// No live engine; JSONL file output only.
    File,
}

/// Error returned when parsing an unrecognized [`DeployMode`] string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseDeployModeError(String);

impl std::fmt::Display for ParseDeployModeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "unrecognized deploy mode {:?} (expected embedded, sidecar, or file)",
            self.0
        )
    }
}

impl std::error::Error for ParseDeployModeError {}

impl FromStr for DeployMode {
    type Err = ParseDeployModeError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "embedded" => Ok(Self::Embedded),
            "sidecar" => Ok(Self::Sidecar),
            "file" => Ok(Self::File),
            _ => Err(ParseDeployModeError(s.to_owned())),
        }
    }
}

impl std::fmt::Display for DeployMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Embedded => "embedded",
            Self::Sidecar => "sidecar",
            Self::File => "file",
        })
    }
}

/// The inputs that determine the auto-detected deploy mode.
///
/// Decoupling these from the real environment keeps [`DeployMode::resolve`] a
/// pure function, so resolution is tested with deterministic table-driven cases
/// rather than racy global-env mutation.
#[derive(Debug, Clone, Default)]
pub struct DeployEnv {
    /// Raw value of `EMRY_MODE`, if set.
    pub emry_mode: Option<String>,
    /// Whether `SSH_CONNECTION` is present.
    pub ssh_connection: bool,
    /// Whether `SLURM_JOB_ID` is present.
    pub slurm_job_id: bool,
    /// Whether stdout is a terminal.
    pub stdout_is_tty: bool,
}

impl DeployEnv {
    /// Reads the real process environment and stdout TTY status.
    #[must_use]
    pub fn from_env() -> Self {
        use std::io::IsTerminal;
        Self {
            emry_mode: std::env::var("EMRY_MODE").ok(),
            ssh_connection: std::env::var_os("SSH_CONNECTION").is_some(),
            slurm_job_id: std::env::var_os("SLURM_JOB_ID").is_some(),
            stdout_is_tty: std::io::stdout().is_terminal(),
        }
    }

    /// Auto-detected mode from these inputs alone, ignoring `EMRY_MODE` and any
    /// API override.
    #[must_use]
    pub fn auto_detect(&self) -> DeployMode {
        if self.ssh_connection || self.slurm_job_id {
            DeployMode::Sidecar
        } else if self.stdout_is_tty {
            DeployMode::Embedded
        } else {
            DeployMode::File
        }
    }
}

impl DeployMode {
    /// Resolves the effective deploy mode, applying the full precedence:
    /// `api` override → `EMRY_MODE` → auto-detect.
    ///
    /// An unrecognized `EMRY_MODE` value is ignored and resolution falls through
    /// to auto-detection.
    #[must_use]
    pub fn resolve(api: Option<DeployMode>, env: &DeployEnv) -> DeployMode {
        if let Some(mode) = api {
            return mode;
        }
        if let Some(raw) = env.emry_mode.as_deref() {
            if let Ok(mode) = raw.parse::<DeployMode>() {
                return mode;
            }
        }
        env.auto_detect()
    }

    /// Convenience: resolve against the real environment with an optional API
    /// override.
    #[must_use]
    pub fn detect(api: Option<DeployMode>) -> DeployMode {
        Self::resolve(api, &DeployEnv::from_env())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_is_case_insensitive_and_trims() {
        assert_eq!("EMBEDDED".parse(), Ok(DeployMode::Embedded));
        assert_eq!("  Sidecar ".parse(), Ok(DeployMode::Sidecar));
        assert_eq!("file".parse(), Ok(DeployMode::File));
    }

    #[test]
    fn parse_rejects_unknown_and_displays_error() {
        let err = "cloud".parse::<DeployMode>().unwrap_err();
        assert!(err.to_string().contains("cloud"));
    }

    #[test]
    fn display_roundtrips_through_parse() {
        for mode in [DeployMode::Embedded, DeployMode::Sidecar, DeployMode::File] {
            assert_eq!(mode.to_string().parse(), Ok(mode));
        }
    }

    /// Table-driven coverage of every resolution branch.
    #[test]
    fn resolve_precedence_and_auto_detect_branches() {
        struct Case {
            name: &'static str,
            api: Option<DeployMode>,
            emry_mode: Option<&'static str>,
            ssh: bool,
            slurm: bool,
            tty: bool,
            expected: DeployMode,
        }

        let cases = [
            Case {
                name: "api override beats everything",
                api: Some(DeployMode::File),
                emry_mode: Some("embedded"),
                ssh: true,
                slurm: true,
                tty: true,
                expected: DeployMode::File,
            },
            Case {
                name: "EMRY_MODE beats auto-detect",
                api: None,
                emry_mode: Some("sidecar"),
                ssh: false,
                slurm: false,
                tty: true,
                expected: DeployMode::Sidecar,
            },
            Case {
                name: "invalid EMRY_MODE falls through to auto",
                api: None,
                emry_mode: Some("nonsense"),
                ssh: false,
                slurm: false,
                tty: true,
                expected: DeployMode::Embedded,
            },
            // Empty / whitespace EMRY_MODE (e.g. `export EMRY_MODE=`) is treated
            // as unset and falls through to auto-detect. Pinned so the Python
            // mirror (EMRY-031) replicates the same contract.
            Case {
                name: "empty EMRY_MODE falls through to auto",
                api: None,
                emry_mode: Some(""),
                ssh: false,
                slurm: false,
                tty: false,
                expected: DeployMode::File,
            },
            Case {
                name: "whitespace-only EMRY_MODE falls through to auto",
                api: None,
                emry_mode: Some("   "),
                ssh: true,
                slurm: false,
                tty: false,
                expected: DeployMode::Sidecar,
            },
            Case {
                name: "SSH -> sidecar",
                api: None,
                emry_mode: None,
                ssh: true,
                slurm: false,
                tty: true,
                expected: DeployMode::Sidecar,
            },
            Case {
                name: "SLURM -> sidecar",
                api: None,
                emry_mode: None,
                ssh: false,
                slurm: true,
                tty: false,
                expected: DeployMode::Sidecar,
            },
            Case {
                name: "TTY without ssh/slurm -> embedded",
                api: None,
                emry_mode: None,
                ssh: false,
                slurm: false,
                tty: true,
                expected: DeployMode::Embedded,
            },
            Case {
                name: "no signals -> file",
                api: None,
                emry_mode: None,
                ssh: false,
                slurm: false,
                tty: false,
                expected: DeployMode::File,
            },
        ];

        for case in cases {
            let env = DeployEnv {
                emry_mode: case.emry_mode.map(str::to_owned),
                ssh_connection: case.ssh,
                slurm_job_id: case.slurm,
                stdout_is_tty: case.tty,
            };
            assert_eq!(
                DeployMode::resolve(case.api, &env),
                case.expected,
                "case: {}",
                case.name
            );
        }
    }

    #[test]
    fn from_env_and_detect_run_against_real_environment() {
        // Non-deterministic result, but exercises the real-env code paths and
        // proves they return a valid variant.
        let _ = DeployEnv::from_env();
        let mode = DeployMode::detect(None);
        assert!(matches!(
            mode,
            DeployMode::Embedded | DeployMode::Sidecar | DeployMode::File
        ));
        // An explicit override is honoured regardless of environment.
        assert_eq!(DeployMode::detect(Some(DeployMode::File)), DeployMode::File);
    }
}
