# Running Emry on SLURM (sidecar pattern)

This runbook covers running Emry alongside a training job on a SLURM cluster
using the **sidecar** deploy mode: a long-lived `emry engine` process that owns
persistence and observation, with your training script connecting to it over a
Unix socket.

If you just want metrics on disk with zero moving parts, skip to
[File mode](#alternative-file-mode-no-engine) — it works everywhere and is the
right default for multi-node jobs.

## Why sidecar

In embedded mode the engine runs inside your training process. That is simplest
locally, but on a cluster it couples observability to the training process:

- The engine's background work (JSONL writes, anomaly detection, the web/TUI
  server) shares the training process's lifetime and fate.
- A crash or preemption of training takes the engine down with it mid-write.

The sidecar runs the engine as a **separate process**. Training emits into a
Unix socket and returns immediately; the engine drains, processes, and persists
on its own. If the engine is unreachable, training does **not** crash — it falls
back to writing a local JSONL run directory (see
[Failure behaviour](#failure-behaviour)). Observability never harms the run.

```text
                 ┌──────────────────────────── one node / allocation ───┐
  training proc  │  run.emit() ──► [Unix socket 0600] ──► emry engine    │
  (rank 0)       │                                          │            │
                 │                                          ▼            │
                 │                              logs/{project}_{ts}/     │
                 │                                events.jsonl           │
                 │                                metrics.jsonl  ◄─ tail │
                 └───────────────────────────────────────────┬──────────┘
                                                              │ shared FS
   login node:  emry web --run-dir logs/{project}_{ts}  ◄─────┘
                emry watch logs/{project}_{ts}
```

> **Host locality:** a Unix domain socket is local to a single host. The engine
> and the training process that emits to it must run on the **same node**. For
> multi-node (distributed) training, either run one engine per node, or use file
> mode and observe from the login node over the shared filesystem. Emit only
> from rank 0 — do the `all_reduce` in Python before `emit()`.

## 1. Start the engine

The engine binds a Unix socket and writes a run directory under a base log dir:

```bash
emry engine \
  --project llama-sft \
  --socket "$TMPDIR/emry.sock" \
  --log-dir "$SCRATCH/emry-logs"
```

- `--socket` — path to bind (created with `0600` perms). Put it on a node-local
  path (`$TMPDIR`) since the socket only needs to be reachable from the same
  node.
- `--log-dir` — base directory for run output. Put this on a **shared
  filesystem** (`$SCRATCH`, `$HOME`) so you can observe it from the login node.
  Defaults to `./logs` if omitted.

The engine serves one run, then drains and flushes when it receives
`RUN_FINISHED` (or the peer disconnects).

## 2. Point training at the socket

The Python SDK auto-detects sidecar mode when `SLURM_JOB_ID` is set, but be
explicit in batch scripts:

```bash
export EMRY_MODE=sidecar
export EMRY_SOCKET="$TMPDIR/emry.sock"
```

| Variable        | Purpose                                                        |
| --------------- | ------------------------------------------------------------- |
| `EMRY_MODE`     | `embedded` \| `sidecar` \| `file`. Overrides auto-detection.  |
| `EMRY_SOCKET`   | Socket path the SDK connects to in sidecar mode.              |
| `EMRY_LOG_DIR`  | Base log dir for file-mode / fallback run directories.        |

Your training code is unchanged across modes:

```python
import emry

with emry.run("llama-sft", config={"lr": 3e-4}) as run:
    for step in range(steps):
        loss = train_step()
        run.emit({"loss": loss, "lr": sched.get_last_lr()[0]})
```

## 3. Putting it in a batch script

Launch the engine in the background on the batch node, wait for its socket, run
training, then let the engine finish.

```bash
#!/bin/bash
#SBATCH --job-name=llama-sft
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --time=24:00:00

set -euo pipefail

export EMRY_SOCKET="$TMPDIR/emry.sock"
export EMRY_MODE=sidecar
LOG_DIR="$SCRATCH/emry-logs"

# Start the sidecar engine and wait for the socket to appear.
emry engine --project llama-sft --socket "$EMRY_SOCKET" --log-dir "$LOG_DIR" &
ENGINE_PID=$!
for _ in $(seq 1 100); do
  [ -S "$EMRY_SOCKET" ] && break
  sleep 0.1
done

# Run training (emits into the socket). Emit from rank 0 only.
srun python train.py

# train.py's `with emry.run(...)` sends RUN_FINISHED on exit, so the engine
# drains, flushes, and exits on its own. Wait for it to finish writing.
wait "$ENGINE_PID"
```

## 4. Observe the run live

The engine writes `metrics.jsonl` into the shared `--log-dir`, so you can watch
from the **login node** (no socket access needed — it tails the file):

```bash
# Terminal dashboard:
emry watch "$SCRATCH/emry-logs/llama-sft_20260626_120000"

# Or the web dashboard (browse to http://127.0.0.1:8787):
emry web --run-dir "$SCRATCH/emry-logs/llama-sft_20260626_120000"
```

Use `emry runs --log-dir "$SCRATCH/emry-logs"` to list runs and find the
directory name.

## Failure behaviour

- **Engine not reachable at start.** If `EMRY_SOCKET` is set but nothing is
  listening, the SDK warns and writes a local JSONL run directory instead of
  failing the job.
- **`EMRY_SOCKET` unset in sidecar mode.** Same graceful fallback to a local
  JSONL run directory, with a warning.
- **Backpressure.** Every queue between `emit()` and disk is bounded and
  drops-and-counts on overflow; `emit()` never blocks the training thread. The
  dropped count is reported in `summary.json`.
- **Preemption.** `events.jsonl` and `metrics.jsonl` are append-only, so a
  killed job leaves a valid (if incomplete) log you can still `watch`, `export`,
  or `compare`.

## Alternative: file mode (no engine)

For multi-node jobs, or when you want the fewest moving parts, skip the engine
entirely and write JSONL directly:

```bash
export EMRY_MODE=file
export EMRY_LOG_DIR="$SCRATCH/emry-logs"
srun python train.py
```

Then observe from the login node exactly as above (`emry watch` / `emry web`
read the shared `metrics.jsonl`). This is the most portable option and survives
any node going away; you only lose the live sidecar processing (anomaly alerts,
derived metrics computed engine-side).
