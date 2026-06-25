"""Pure-Python JSONL backend — writes the standard run-directory layout.

This is the ``file`` deploy mode without any Rust: it writes ``run.meta``,
``config.json``, ``metrics.jsonl`` (wide rows), and ``summary.json`` into
``{log_dir}/{project}_{timestamp}/``, matching the schema in
``emry-store::meta`` so the output is readable by ``emry watch`` and external/v1
tools.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, TextIO

from emry.mode import DeployMode, detect
from emry.phase import Phase

__all__ = ["JsonlBackend"]

RUN_META_FILE = "run.meta"
CONFIG_FILE = "config.json"
METRICS_FILE = "metrics.jsonl"
SUMMARY_FILE = "summary.json"


def _resolve_mode(mode: object) -> str:
    if isinstance(mode, DeployMode):
        return mode.value
    if isinstance(mode, str):
        parsed = DeployMode.parse(mode)
        return parsed.value if parsed is not None else mode
    return detect().value


class JsonlBackend:
    """Writes a run's metrics and metadata to a JSONL run directory."""

    def __init__(self, run_dir: Path, run_id: str, project: str, start_secs: float) -> None:
        self._run_dir = run_dir
        self._run_id = run_id
        self._project = project
        self._start_secs = start_secs
        self._metrics: TextIO = (run_dir / METRICS_FILE).open("w", encoding="utf-8")

    @classmethod
    def create(
        cls,
        project: str,
        *,
        config: Mapping[str, Any],
        metrics: Iterable[str],  # noqa: ARG003 — declared for parity; names come from emit()
        mode: object = None,
        log_dir: Optional[str] = None,
    ) -> "JsonlBackend":
        """Creates the run directory and writes ``run.meta`` + ``config.json``."""
        start_secs = time.time()
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime(start_secs))
        run_dir = Path(log_dir or "logs") / f"{project}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        run_id = str(uuid.uuid4())

        _write_json(
            run_dir / RUN_META_FILE,
            {
                "run_id": run_id,
                "project": project,
                "start_time_secs": start_secs,
                "mode": _resolve_mode(mode),
            },
        )
        _write_json(run_dir / CONFIG_FILE, dict(config))
        return cls(run_dir, run_id, project, start_secs)

    @property
    def run_dir(self) -> Path:
        """Directory this run is written to."""
        return self._run_dir

    def emit(self, step: int, epoch: int, phase: Phase, values: dict[str, float]) -> None:
        """Appends one wide metric row to ``metrics.jsonl``."""
        row = {"step": step, "epoch": epoch, "phase": phase.value, "values": values}
        self._metrics.write(json.dumps(row) + "\n")
        self._metrics.flush()  # keep the file live for `emry watch`

    def finish(self, *, steps: int) -> None:
        """Writes ``summary.json`` and closes ``metrics.jsonl``."""
        _write_json(
            self._run_dir / SUMMARY_FILE,
            {
                "run_id": self._run_id,
                "project": self._project,
                "duration_secs": time.time() - self._start_secs,
                "reason": "COMPLETED",
                "steps": steps,
                "dropped": 0,
            },
        )
        self._metrics.close()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")
