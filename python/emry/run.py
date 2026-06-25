"""The high-level ``emry.run()`` API (design §7).

```python
import emry

with emry.run("llama-sft", config=cfg, metrics=["loss", "lr"]) as run:
    for epoch in run.epochs(10):
        run.phase = emry.Phase.TRAIN
        for batch in run.track(train_loader):
            run.emit(loss=loss, lr=lr)
```

A [`Run`] talks to a pluggable [`Backend`]; the default is a pure-Python
[`JsonlBackend`](emry.jsonl_backend.JsonlBackend) writing the standard run-dir
layout (so the files are readable by ``emry watch``). The native PyO3 backend
(EMRY-033) and the sidecar socket client (EMRY-034) plug in later behind the same
interface, so this module needs no Rust to run or test.
"""

from __future__ import annotations

from typing import Any, Iterable, Iterator, Mapping, Optional, Protocol

from emry.coerce import coerce_metrics
from emry.phase import Phase

__all__ = ["Run", "Backend", "NullBackend", "run"]


class Backend(Protocol):
    """Sink for a run's emitted metrics and lifecycle."""

    def emit(self, step: int, epoch: int, phase: Phase, values: dict[str, float]) -> None:
        """Records one step's coerced metric values."""

    def finish(self, *, steps: int, reason: str) -> None:
        """Flushes and closes the run after `steps` emissions.

        `reason` is the screaming-snake `FinishReason` (`COMPLETED` |
        `INTERRUPTED` | `FAILED`).
        """


class NullBackend:
    """A backend that discards everything — used for ``live=False`` and tests."""

    def emit(self, step: int, epoch: int, phase: Phase, values: dict[str, float]) -> None:
        """Drops the metrics."""

    def finish(self, *, steps: int, reason: str) -> None:
        """No-op."""


class Run:
    """Handle for a live run. Construct via [`run`]; use as a context manager."""

    def __init__(
        self,
        project: str,
        backend: Backend,
        *,
        metrics: Optional[Iterable[str]] = None,
        config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.project = project
        self.metrics = list(metrics or [])
        self.config = dict(config or {})
        self._backend = backend
        self._step = 0
        self._epoch = 0
        self._phase = Phase.TRAIN
        self._finished = False

    @property
    def step(self) -> int:
        """Number of emissions so far (the next step's index)."""
        return self._step

    @property
    def epoch(self) -> int:
        """Current epoch."""
        return self._epoch

    @property
    def phase(self) -> Phase:
        """Current training phase."""
        return self._phase

    @phase.setter
    def phase(self, phase: Phase) -> None:
        self._phase = phase

    def emit(self, **values: Any) -> None:
        """Emits one step of named metric values and advances the step counter.

        Values are coerced to floats (Python numbers, tensors, numpy scalars).
        Raises ``RuntimeError`` if called after :meth:`finish`.
        """
        if self._finished:
            raise RuntimeError("emit() called after the run finished")
        coerced = coerce_metrics(values)
        self._backend.emit(self._step, self._epoch, self._phase, coerced)
        self._step += 1

    def steps(self, total: int) -> Iterator[int]:
        """Yields ``0..total`` as loop sugar; each iteration is one expected emit."""
        yield from range(total)

    def epochs(self, n: int) -> Iterator[int]:
        """Yields ``0..n``, setting :attr:`epoch` on each iteration."""
        for e in range(n):
            self._epoch = e
            yield e

    def track(self, iterable: Iterable[Any]) -> Iterator[Any]:
        """Passes an iterable (e.g. a dataloader) through unchanged."""
        yield from iterable

    def finish(self, reason: str = "COMPLETED") -> None:
        """Flushes and closes the run. Idempotent.

        `reason` is the screaming-snake `FinishReason`; defaults to `COMPLETED`.
        """
        if not self._finished:
            self._backend.finish(steps=self._step, reason=reason)
            self._finished = True

    def __enter__(self) -> "Run":
        return self

    def __exit__(self, exc_type: Optional[type], *_rest: object) -> bool:
        # Record how the run ended so a crashed `with` block isn't logged as
        # COMPLETED (mirrors the Rust FinishReason).
        if exc_type is None:
            self.finish("COMPLETED")
        elif issubclass(exc_type, KeyboardInterrupt):
            self.finish("INTERRUPTED")
        else:
            self.finish("FAILED")
        return False  # never suppress exceptions


def run(
    project: str,
    *,
    config: Optional[Mapping[str, Any]] = None,
    metrics: Optional[Iterable[str]] = None,
    live: str = "auto",  # noqa: ARG001 — observer spawning is EMRY-035
    mode: object = None,
    log_dir: Optional[str] = None,
    backend: Optional[Backend] = None,
) -> Run:
    """Starts a run and returns its [`Run`] handle.

    With no `backend`, persists to a pure-Python JSONL run directory (`file`
    mode) and, per `live`, may launch a dashboard. Pass `backend` to inject an
    alternative (e.g. a fake in tests) — observer spawning is skipped then, since
    the caller is in full control.
    """
    if backend is not None:
        return Run(project, backend, metrics=metrics, config=config)

    backend = _default_backend(
        project,
        config=dict(config or {}),
        metrics=list(metrics or []),
        mode=mode,
        log_dir=log_dir,
    )
    _spawn_live(live, backend)
    return Run(project, backend, metrics=metrics, config=config)


def _spawn_live(live: object, backend: Backend) -> None:
    """Launches the live dashboard observer(s) selected by `live`."""
    import os
    import sys

    from emry.live import resolve_live, spawn_observers

    isatty = getattr(sys.stdout, "isatty", None)
    observers = resolve_live(
        live,
        ssh="SSH_CONNECTION" in os.environ,
        tty=bool(isatty()) if callable(isatty) else False,
        force_tui=os.environ.get("EMRY_LIVE_FORCE") == "1",
    )
    if observers:
        spawn_observers(
            observers,
            run_dir=getattr(backend, "run_dir", None),
            socket_path=os.environ.get("EMRY_SOCKET"),
        )


def _default_backend(
    project: str,
    *,
    config: dict,
    metrics: list,
    mode: object,
    log_dir: Optional[str],
) -> Backend:
    """Selects the backend by deploy mode: native Rust engine for `embedded`
    (when built), the sidecar socket client for `sidecar` (when an engine is
    reachable), else the pure-Python JSONL writer. Each falls back to JSONL."""
    import os

    from emry.mode import DeployMode, detect

    if isinstance(mode, DeployMode):
        resolved = mode
    elif isinstance(mode, str):
        parsed = DeployMode.parse(mode)
        if parsed is None:
            raise ValueError(f"unknown mode {mode!r} (expected embedded, sidecar, or file)")
        resolved = parsed
    else:
        resolved = detect()

    if resolved is DeployMode.EMBEDDED:
        from emry.native_backend import try_create

        native = try_create(project, config=config, metrics=metrics, mode=resolved, log_dir=log_dir)
        if native is not None:
            return native

    if resolved is DeployMode.SIDECAR:
        socket_path = os.environ.get("EMRY_SOCKET")
        if socket_path:
            from emry.socket_backend import SocketBackend

            try:
                return SocketBackend.connect(
                    project, config=config, metrics=metrics, socket_path=socket_path
                )
            except OSError:
                import warnings

                warnings.warn(
                    f"sidecar engine not reachable at EMRY_SOCKET={socket_path!r}; "
                    "writing to a local JSONL run directory instead",
                    stacklevel=2,
                )
        elif mode is not None:
            # Sidecar was explicitly requested but no socket is configured.
            import warnings

            warnings.warn(
                "sidecar mode requested but EMRY_SOCKET is unset; writing to a "
                "local JSONL run directory instead",
                stacklevel=2,
            )

    from emry.jsonl_backend import JsonlBackend

    return JsonlBackend.create(
        project, config=config, metrics=metrics, mode=resolved, log_dir=log_dir
    )
