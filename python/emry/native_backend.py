"""Backend that drives the Rust engine in-process via the PyO3 extension.

Used for ``embedded`` mode when the native ``emry._native`` module is available
(built with ``maturin develop``/installed from a wheel). It gives the fast Rust
``emit()`` path instead of the pure-Python JSONL writer. Falls back to the JSONL
backend when the extension is absent — see :func:`try_create`.

The native-only lines are marked ``# pragma: no cover``: they require the
compiled extension, which is not built in the pure-Python CI job (it is exercised
locally and by the wheel-build job in EMRY-036).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from emry.mode import DeployMode, detect
from emry.phase import Phase

__all__ = ["NativeBackend", "try_create"]


def try_create(
    project: str,
    *,
    config: Mapping[str, Any],
    metrics: Iterable[str],
    total_steps: Optional[int] = None,
    mode: object = None,
    log_dir: Optional[str] = None,
) -> Optional["NativeBackend"]:
    """Builds a native backend, or returns ``None`` if the extension is absent.

    ``config`` is currently unused by the native engine (it persists its own
    metadata); accepted for parity with the other backends.
    """
    try:
        from emry import _native  # noqa: F401
    except ImportError:
        return None
    return NativeBackend.create(  # pragma: no cover - requires the native ext
        project,
        metrics=metrics,
        total_steps=total_steps,
        mode=mode,
        log_dir=log_dir,
    )


class NativeBackend:  # pragma: no cover - requires the compiled extension
    """Wraps ``emry._native.RunHandle`` behind the SDK's ``Backend`` protocol."""

    def __init__(self, handle: Any, ids: dict[str, int]) -> None:
        self._handle = handle
        self._ids = ids
        self._phase: Optional[Phase] = None
        self._epoch = 0

    @classmethod
    def create(
        cls,
        project: str,
        *,
        metrics: Iterable[str],
        total_steps: Optional[int] = None,
        mode: object = None,
        log_dir: Optional[str] = None,
    ) -> "NativeBackend":
        from emry import _native

        mode_str = mode.value if isinstance(mode, DeployMode) else (mode or detect().value)
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        run_dir = Path(log_dir or "logs") / f"{project}_{timestamp}_{mode_str}"
        run_dir.mkdir(parents=True, exist_ok=True)
        names = list(metrics)
        handle = _native.RunHandle(project, str(run_dir), names, total_steps)
        ids = {name: handle.register(name) for name in names}
        return cls(handle, ids)

    def emit(self, step: int, epoch: int, phase: Phase, values: dict[str, float]) -> None:
        if epoch != self._epoch:
            self._handle.set_epoch(epoch)
            self._epoch = epoch
        if phase != self._phase:
            self._handle.set_phase(phase.value)
            self._phase = phase
        pairs = []
        for name, value in values.items():
            mid = self._ids.get(name)
            if mid is None:
                mid = self._handle.register(name)
                self._ids[name] = mid
            pairs.append((mid, value))
        self._handle.emit(pairs)

    def finish(self, *, steps: int, reason: str) -> None:
        # The native engine records COMPLETED; reason fidelity for the native
        # path is a follow-up (the Rust RunHandle::finish has no reason arg yet).
        self._handle.finish()
