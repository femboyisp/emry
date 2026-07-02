"""Post-hoc Weights & Biases exporter.

Reads a finished Emry run directory (``config.json``, ``metrics.jsonl``,
``summary.json`` — the standard layout written by
:class:`~emry.jsonl_backend.JsonlBackend`) and replays it into a W&B run:
one :func:`wandb.log` call per metrics row in step order, then the run's
summary values, then :func:`wandb.finish`.

``wandb`` is an optional dependency — it is imported lazily inside
:func:`to_wandb` so importing this module (and the rest of Emry) never
requires it. Install it with ``pip install emry[wandb]``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    import wandb

__all__ = ["to_wandb"]

CONFIG_FILE = "config.json"
METRICS_FILE = "metrics.jsonl"
SUMMARY_FILE = "summary.json"
RUN_META_FILE = "run.meta"


def to_wandb(
    run_dir: Any,
    *,
    project: Optional[str] = None,
    entity: Optional[str] = None,
    name: Optional[str] = None,
    config: Optional[dict] = None,
) -> "wandb.Run":
    """Replays a finished Emry run directory into a Weights & Biases run.

    Reads the run's ``config.json`` (merged with any ``config`` override) and
    replays every ``metrics.jsonl`` row via :func:`wandb.log` in ascending step
    order, then copies ``summary.json`` onto the run's summary and finishes it.

    ``project`` defaults to the run's own project (from ``run.meta``). Raises
    :class:`ImportError` if ``wandb`` is not installed.
    """
    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "Weights & Biases export requires `pip install emry[wandb]`"
        ) from exc

    run_dir = Path(run_dir)
    meta = _read_json(run_dir / RUN_META_FILE, default={})
    run_config = _read_json(run_dir / CONFIG_FILE, default={})
    if config:
        run_config.update(config)

    run = wandb.init(
        project=project or meta.get("project"),
        entity=entity,
        name=name,
        config=run_config,
    )

    for record in _read_metrics(run_dir / METRICS_FILE):
        values = record.get("values") or {}
        run.log(values, step=record.get("step"))

    summary = _read_json(run_dir / SUMMARY_FILE, default={})
    if summary:
        run.summary.update(summary)

    wandb.finish()
    return run


def _read_json(path: Path, *, default: Any) -> Any:
    """Reads a JSON file, returning `default` if it is absent."""
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _read_metrics(path: Path) -> list[dict]:
    """Reads ``metrics.jsonl`` into a list of rows sorted by ascending step."""
    if not path.exists():
        return []
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows.sort(key=lambda row: row.get("step", 0))
    return rows


def _main(argv: Optional[list[str]] = None) -> None:
    """CLI entry point: ``python -m emry.export_wandb <run_dir> [options]``."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m emry.export_wandb",
        description="Export a finished Emry run directory to Weights & Biases.",
    )
    parser.add_argument("run_dir", help="path to a finished Emry run directory")
    parser.add_argument("--project", default=None, help="W&B project name")
    parser.add_argument("--entity", default=None, help="W&B entity (team/user)")
    parser.add_argument("--name", default=None, help="W&B run display name")
    args = parser.parse_args(argv)

    to_wandb(args.run_dir, project=args.project, entity=args.entity, name=args.name)


if __name__ == "__main__":  # pragma: no cover
    _main()
