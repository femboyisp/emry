"""GPU metric sampling via ``nvidia-smi``.

Pull-based by design: the engine's ring is single-producer, so GPU stats are
sampled on the **training thread** during ``run.emit`` (throttled to roughly once
a second) and merged into that step's metrics — never from a background thread.

A graceful no-op when there's no GPU: [`GpuSampler.available`] is just a
``which nvidia-smi`` check, and a failing query disables further sampling. No
Python GPU dependency — ``nvidia-smi`` is a system binary.

Emitted metrics, per detected GPU ``i``: ``gpu{i}_util`` (%), ``gpu{i}_mem_mb``
(MiB used), ``gpu{i}_mem_pct`` (% of total), ``gpu{i}_temp_c`` (°C).
"""

from __future__ import annotations

import shutil
import subprocess
import time
from typing import Callable, Dict

__all__ = ["GpuSampler"]

# The fields we query, in order, from nvidia-smi.
_QUERY = "index,utilization.gpu,memory.used,memory.total,temperature.gpu"
_ARGS = [
    "nvidia-smi",
    f"--query-gpu={_QUERY}",
    "--format=csv,noheader,nounits",
]


def _run_nvidia_smi() -> str:
    """Runs ``nvidia-smi`` and returns its CSV stdout (raises on failure)."""
    out = subprocess.run(  # noqa: S603 - fixed argv, no shell
        _ARGS,
        capture_output=True,
        text=True,
        timeout=5,
        check=True,
    )
    return out.stdout


class GpuSampler:
    """Throttled ``nvidia-smi`` sampler. Call [`sample`] each step; it queries at
    most once per ``interval`` seconds and returns ``{}`` in between (and forever
    once a query fails, so a transient/missing GPU never spams or stalls)."""

    def __init__(
        self,
        *,
        interval: float = 1.0,
        runner: Callable[[], str] = _run_nvidia_smi,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._interval = interval
        self._run = runner
        self._clock = clock
        self._last = float("-inf")
        self._disabled = False

    @staticmethod
    def available() -> bool:
        """Whether ``nvidia-smi`` is on ``PATH`` (cheap; no subprocess)."""
        return shutil.which("nvidia-smi") is not None

    def sample(self) -> Dict[str, float]:
        """A fresh GPU reading if at least ``interval`` has elapsed, else ``{}``.

        Returns ``{}`` (and disables further sampling) if a query errors or its
        output can't be parsed — GPU metrics are best-effort and must never
        interfere with the run.
        """
        if self._disabled:
            return {}
        now = self._clock()
        if now - self._last < self._interval:
            return {}
        self._last = now
        try:
            return _parse(self._run())
        except Exception:  # noqa: BLE001 - any failure disables, never propagates
            self._disabled = True
            return {}


def _parse(csv: str) -> Dict[str, float]:
    """Parses ``nvidia-smi`` CSV rows into a flat ``gpu{i}_*`` metric dict."""
    out: Dict[str, float] = {}
    for line in csv.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5:
            continue
        idx, util, mem_used, mem_total, temp = (float(p) for p in parts)
        i = int(idx)
        out[f"gpu{i}_util"] = util
        out[f"gpu{i}_mem_mb"] = mem_used
        out[f"gpu{i}_mem_pct"] = 100.0 * mem_used / mem_total if mem_total else 0.0
        out[f"gpu{i}_temp_c"] = temp
    return out
