"""Sidecar socket backend — streams framed msgpack events to ``emry engine``.

A pure-Python ``Backend`` for ``sidecar`` mode: it connects to the Unix socket a
running ``emry engine`` is bound to and sends length-prefixed msgpack frames
(:mod:`emry.wire`). This is the fallback when the native PyO3 extension is not
available.
"""

from __future__ import annotations

import socket
import time
from typing import Any, Iterable, Mapping, Optional

from emry import wire
from emry.phase import Phase

__all__ = ["SocketBackend"]


class SocketBackend:
    """Sends a run's events to a sidecar engine over a Unix socket."""

    def __init__(self, sock: socket.socket, start_secs: float) -> None:
        self._sock = sock
        self._start_secs = start_secs
        self._ids: dict[str, int] = {}
        self._phase: Optional[Phase] = None

    @classmethod
    def connect(
        cls,
        project: str,
        *,
        config: Mapping[str, Any],
        metrics: Iterable[str],
        socket_path: str,
    ) -> "SocketBackend":
        """Connects to the engine socket and streams events.

        The sidecar engine owns the run identity (it creates the run directory
        and `run.meta` itself), so the client does **not** send `RunStarted` — it
        only streams metrics, phase changes, and the final `RunFinished`.

        Raises ``OSError`` if the socket can't be reached (no engine listening).
        """
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(socket_path)
        backend = cls(sock, time.time())
        # Assign client-local ids in declaration order; the engine persists ids.
        for name in metrics:
            backend._ids.setdefault(name, len(backend._ids))
        return backend

    def emit(self, step: int, epoch: int, phase: Phase, values: dict[str, float]) -> None:
        # Emit a PhaseChange whenever the phase differs from the last one sent —
        # including the very first emit. Observers (e.g. the TUI's UiState) track
        # phase from PhaseChange events, not the MetricsBatch field, so the
        # initial phase must be signalled or it would read as the default.
        if phase != self._phase:
            self._send(wire.phase_change(phase.value))
        self._phase = phase
        pairs = []
        for name, value in values.items():
            mid = self._ids.setdefault(name, len(self._ids))
            pairs.append((mid, value))
        self._send(wire.metrics_batch(step, epoch, phase.value, pairs))

    def finish(self, *, steps: int, reason: str) -> None:
        try:
            self._send(wire.run_finished(time.time() - self._start_secs, reason))
        finally:
            self._sock.close()

    def _send(self, event: Mapping[str, Any]) -> None:
        self._sock.sendall(wire.frame(event))
