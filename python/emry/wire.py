"""Sidecar wire protocol — the Python encoder for ``[u32 LE len][msgpack Event]``.

Mirrors ``emry-ingest::wire`` so a pure-Python training process can stream events
to the Rust ``emry engine`` sidecar. ``Event`` is an adjacently-tagged enum
serialized as ``{"type": ..., "data": ...}`` with structs encoded as maps (the
Rust side decodes with ``rmp_serde``, which the map form matches).

``msgpack`` is imported lazily so it is only required for ``sidecar`` mode
(``pip install emry[socket]``), not for ``file``/``embedded`` runs.
"""

from __future__ import annotations

import struct
from typing import Any, Iterable, Mapping, Tuple

__all__ = [
    "MAX_FRAME_BYTES",
    "frame",
    "metrics_batch",
    "phase_change",
    "run_finished",
]

MAX_FRAME_BYTES = 16 * 1024 * 1024


def _msgpack():
    try:
        import msgpack  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - exercised via a clear message
        raise ImportError(
            "sidecar mode needs msgpack — install with `pip install emry[socket]`"
        ) from exc
    return msgpack


def frame(event: Mapping[str, Any]) -> bytes:
    """Encodes one event as ``[u32 LE length][msgpack payload]``."""
    payload = _msgpack().packb(event, use_bin_type=True)
    if len(payload) > MAX_FRAME_BYTES:
        raise ValueError(f"frame of {len(payload)} bytes exceeds {MAX_FRAME_BYTES}")
    return struct.pack("<I", len(payload)) + payload


def metrics_batch(
    step: int, epoch: int, phase: str, values: Iterable[Tuple[int, float]]
) -> dict:
    """A ``MetricsBatch`` event (`values` are `(metric_id, value)` pairs)."""
    return {
        "type": "METRICS_BATCH",
        "data": {
            "step": step,
            "epoch": epoch,
            "phase": phase,
            "values": [[int(i), float(v)] for i, v in values],
        },
    }


def phase_change(phase: str) -> dict:
    """A ``PhaseChange`` event."""
    return {"type": "PHASE_CHANGE", "data": phase}


def run_finished(duration_secs: float, reason: str) -> dict:
    """A ``RunFinished`` event."""
    return {
        "type": "RUN_FINISHED",
        "data": {"duration_secs": duration_secs, "reason": reason},
    }
