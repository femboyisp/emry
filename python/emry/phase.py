"""Training phase — the Python mirror of the Rust ``Phase`` enum.

Values are the screaming-snake strings the Rust side serializes (e.g. ``"TRAIN"``)
so phases written by Python land in the same ``metrics.jsonl`` schema that
``emry watch`` and external/v1 readers expect.
"""

from __future__ import annotations

import enum

__all__ = ["Phase"]


class Phase(enum.Enum):
    """Training phase for context-aware logging and UI styling."""

    TRAIN = "TRAIN"
    EVAL = "EVAL"
    TEST = "TEST"
    CHECKPOINT = "CHECKPOINT"
    WARMUP = "WARMUP"
