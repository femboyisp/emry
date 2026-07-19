# Changelog

All notable changes to Emry are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and Emry adheres to
[Semantic Versioning](https://semver.org/).

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

[0.2.1]: https://github.com/femboyisp/emry/releases/tag/v0.2.1
[0.2.0]: https://github.com/femboyisp/emry/releases/tag/v0.2.0
[0.1.0]: https://github.com/femboyisp/emry/releases/tag/v0.1.0
