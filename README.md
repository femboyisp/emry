# Emry

**Gentle observability for long training runs.**

Open-source (Apache 2.0) training observability: live TUI and web charts, cluster sidecar mode, and fast metric ingest — Rust core with a PyTorch-friendly Python SDK.

**Repository:** [github.com/femboyisp/emry](https://github.com/femboyisp/emry)

## Status

Early development (M0). The workspace compiles; CLI and Python SDK are stubs.

## Quick start (coming soon)

```python
import emry

with emry.run("llama-sft", config={"lr": 2e-5}, metrics=["loss", "lr"], live="auto") as run:
    for step in run.steps(10_000):
        run.emit(loss=loss, lr=scheduler.get_last_lr()[0])
```

```bash
emry engine --project llama-sft --socket /tmp/emry-$SLURM_JOB_ID.sock
emry tui --socket /tmp/emry-$SLURM_JOB_ID.sock
emry watch ./logs/llama-sft_20260621_120000
```

## Development

### Prerequisites

- Rust 1.85+ (`rust-toolchain.toml` pins the toolchain)
- `llvm-tools-preview` for coverage: `rustup component add llvm-tools-preview`
- `cargo-llvm-cov`: `cargo install cargo-llvm-cov`
- Python 3.10+ for hooks and the future SDK

### Commands

```bash
# Full local CI (fmt, clippy, test, ≥90% coverage)
./scripts/pre-commit-rust.sh

# Coverage only
./scripts/check-coverage.sh

# Python tests (when package grows)
pip install -e ".[dev]"
pytest
```

### Pre-commit

```bash
pip install pre-commit
pre-commit install
```

Hooks run: trailing whitespace, YAML/TOML checks, then `./scripts/pre-commit-rust.sh` (fmt + clippy + test + **90% line coverage gate**).

### Quality bar

| Check | Threshold |
|-------|-----------|
| `cargo clippy` | `-D warnings` |
| Rust line coverage | **≥ 90%** (workspace) |
| Python line coverage | **≥ 90%** (`pytest --cov-fail-under=90`) |

Planning docs (design, kanban, implementation plan) live in local `docs/` — **gitignored**, not pushed to GitHub.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
