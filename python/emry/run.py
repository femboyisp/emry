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

    def finish(self, *, steps: int) -> None:
        """Flushes and closes the run after `steps` emissions."""


class NullBackend:
    """A backend that discards everything — used for ``live=False`` and tests."""

    def emit(self, step: int, epoch: int, phase: Phase, values: dict[str, float]) -> None:
        """Drops the metrics."""

    def finish(self, *, steps: int) -> None:
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

    def finish(self) -> None:
        """Flushes and closes the run. Idempotent."""
        if not self._finished:
            self._backend.finish(steps=self._step)
            self._finished = True

    def __enter__(self) -> "Run":
        return self

    def __exit__(self, *_exc: object) -> bool:
        self.finish()
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
    mode). Pass `backend` to inject an alternative (e.g. a fake in tests). The
    `live` observer-spawning argument is accepted now and honoured in EMRY-035.
    """
    if backend is None:
        from emry.jsonl_backend import JsonlBackend

        backend = JsonlBackend.create(
            project,
            config=dict(config or {}),
            metrics=list(metrics or []),
            mode=mode,
            log_dir=log_dir,
        )
    return Run(project, backend, metrics=metrics, config=config)
