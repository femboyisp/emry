# Changelog

All notable changes to Emry are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and Emry adheres to
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- **`run.stage(name)` — explicit curriculum stages.** Call `run.stage("reasoning")`
  to name each stage of a multi-phase run; the dashboards split and label the
  chart at stage boundaries (`reasoning │ knowledge │ current`) instead of only
  inferring labels from checkpoint paths. Backed by a new `StageChange` event
  that flows through all modes (file, sidecar, embedded). Explicit stages take
  precedence over checkpoint-derived boundaries when both are present.
- **Phase-aware web chart (parity with the terminal).** `emry web` now
  EMA-smooths the loss curve and, for curriculum runs, autoscales each
  checkpoint/stage segment to its own range with labeled dividers — so a healthy
  multi-phase run no longer looks like it diverges at each transition.

### Fixed

- **Web dashboard buttons are clickable again.** The metric cards rebuilt on
  every 10 Hz snapshot, so a click landing across a rebuild never registered
  (mousedown and mouseup hit different DOM nodes). Clicks are now delegated to
  the stable container and register regardless of snapshot cadence.

## [0.2.2] - 2026-07-30

Terminal-dashboard chart overhaul (readable curves for noisy, multi-phase
runs) plus dependency upgrades. Backward-compatible.

### Added

- **Phase-aware chart.** Curriculum/multi-phase runs (whose phases have very
  different loss scales) are split at checkpoint boundaries and each segment is
  autoscaled to its own range, with labeled dividers derived from the checkpoint
  path (`reasoning │ knowledge │ code │ …`). A healthy multi-phase run no longer
  looks like it diverges at each transition.
- **`events.jsonl` surfaced in file mode.** `emry watch --run-dir` now shows
  checkpoints, phase/stage changes, alerts, and the metric-name table (it
  previously tailed only `metrics.jsonl`).

### Changed

- **Readable loss curves.** The chart now draws a smoothed (EMA), *connected*
  polyline scaled to the live run. Noisy per-step loss that used to render as
  scattered dots now reads as a clean trend line, and a wide-range `--compare`
  baseline no longer crushes the live curve (off-scale baseline is hidden).
- Dependencies: **axum 0.8 + axum-server 0.8**; Rust toolchain **1.88 → 1.97**;
  in-semver lockfile bumps (serde, clap, tokio, crossbeam-channel,
  http-body-util, uuid, criterion).

## [0.2.1] - 2026-07-19

### Fixed

- **`pip install emry` now ships the `emry` CLI.** The 0.2.0 wheel contained only
  the Python SDK + native extension, so the documented `emry watch/web/compare/
  export` commands weren't runnable and the SDK's live dashboard (which launches
  `emry`) silently degraded. The compiled CLI is now bundled in the wheel and
  exposed as the `emry` console script (and `python -m emry`). The SDK also
  resolves the bundled binary directly, so the live dashboard works even when the
  interpreter's `bin/` isn't on `PATH`.

### Added

- `emry.run(name=...)` is accepted as an alias for the positional run name.

## [0.2.0] - 2026-07-07

A big batch of backward-compatible features. Upgrading from 0.1.0 requires no
code changes — everything below is additive, and existing defaults are unchanged
(the dashboard still binds loopback with no auth unless you opt in).

### Added

- **GPU telemetry.** When an NVIDIA GPU is present, Emry auto-samples
  `nvidia-smi` and charts GPU utilization, memory, and temperature alongside your
  metrics. On by default; `gpu=False` to disable.
- **NaN/Inf alerts.** Pass `alert_webhook=` (or set `EMRY_ALERT_WEBHOOK`) to get a
  Slack/Discord/generic-webhook ping the moment a metric goes non-finite —
  fire-once-per-metric, off the training thread, never blocking the loop.
- **Weights & Biases export.** `emry.to_wandb(run_dir, ...)` (or
  `python -m emry.export_wandb`) replays a finished run into a W&B run. `wandb` is
  an optional extra: `pip install emry[wandb]`.
- **Checkpoint markers.** `run.checkpoint(path, step)` records checkpoint events,
  rendered as markers on both dashboards.
- **Multi-run project dashboard.** `emry web --project ./logs` overlays every run
  in a log directory on one chart for sweep comparison.
- **SLURM sidecar helper.** `emry slurm-wrap --project NAME -- <cmd>` starts the
  sidecar engine, points your command at it, and drains/cleans up on exit —
  collapsing the batch-script boilerplate into one line.
- **Web dashboard auth + TLS.** `--auth-token` (or `EMRY_AUTH_TOKEN`) requires a
  bearer token on every route except `/healthz`; `--tls-cert`/`--tls-key` serve
  HTTPS from your own PEM files; `--host` sets the bind address (default
  loopback). All opt-in.
- **Role-based access (RBAC).** `--viewer-token` grants read-only access to
  single-run dashboards; the multi-run `--project` overlay requires the
  full-access `--auth-token` (admin).
- **Kubernetes deployment.** A multi-stage `Dockerfile` and a Helm chart under
  `deploy/helm/emry` (Deployment, Service, optional Ingress + auth Secret).
- **Terminal-dashboard parity.** `emry watch` gained the step-based chart axis,
  phase bands, checkpoint markers, and a `--compare` baseline overlay — full
  parity with the web dashboard.

### Changed

- Socket/stream observers now receive a metric name table, so live consumers show
  real metric names instead of `m{id}` placeholders.
- Minimum supported Rust version raised to 1.88 (ratatui 0.30); this affects
  building from source only, not installing the wheel.

[0.2.2]: https://github.com/femboyisp/emry/releases/tag/v0.2.2
[0.2.1]: https://github.com/femboyisp/emry/releases/tag/v0.2.1
[0.2.0]: https://github.com/femboyisp/emry/releases/tag/v0.2.0
[0.1.0]: https://github.com/femboyisp/emry/releases/tag/v0.1.0
