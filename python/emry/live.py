"""Live observer spawning for ``emry.run(live=...)``.

``live`` selects whether to launch a dashboard alongside a run:

- ``"auto"`` (default): over SSH → web; on a local TTY → TUI; otherwise nothing
  (a non-interactive/batch run gets no observer). ``EMRY_LIVE_FORCE=1`` forces
  the TUI even over SSH.
- ``"tui"`` / ``"web"`` / ``"both"``: always launch the named observer(s).
- ``False`` / ``"off"`` / ``"none"``: no observer.

The resolution ([`resolve_live`]) is a pure function of its inputs so it's tested
deterministically; [`spawn_observers`] does the actual detached ``emry`` launch.
"""

from __future__ import annotations

import subprocess
import warnings
from pathlib import Path
from typing import List, Optional

__all__ = ["resolve_live", "observer_command", "spawn_observers"]


def resolve_live(live: object, *, ssh: bool, tty: bool, force_tui: bool) -> List[str]:
    """Resolves `live` to a list of observers (`"tui"`/`"web"`), possibly empty."""
    if live is False or live is None:
        return []
    if not isinstance(live, str):
        return []
    choice = live.strip().lower()
    if choice in ("off", "none", "false"):
        return []
    if choice == "tui":
        return ["tui"]
    if choice == "web":
        return ["web"]
    if choice == "both":
        return ["tui", "web"]
    if choice == "auto":
        if ssh and not force_tui:
            return ["web"]
        if tty or (ssh and force_tui):
            return ["tui"]
        return []  # non-interactive: don't spawn anything
    return []  # unknown value: be quiet rather than guess


def observer_command(
    kind: str, *, run_dir: Optional[Path], socket_path: Optional[str]
) -> Optional[List[str]]:
    """Builds the ``emry`` argv for an observer, or ``None`` if it can't run.

    A TUI attaches to the run directory (file/embedded) or the sidecar socket;
    the web observer is not built yet (EMRY-044).
    """
    if kind == "tui":
        if socket_path:
            return ["emry", "tui", "--socket", socket_path]
        if run_dir is not None:
            return ["emry", "tui", "--run-dir", str(run_dir)]
        return None
    if kind == "web":
        return None  # `emry web` lands in EMRY-044
    return None


def spawn_observers(
    observers: List[str], *, run_dir: Optional[Path], socket_path: Optional[str]
) -> List[subprocess.Popen]:
    """Launches each observer as a detached ``emry`` process. Best-effort: a
    missing binary or unavailable observer warns instead of failing the run."""
    procs: List[subprocess.Popen] = []
    for kind in observers:
        cmd = observer_command(kind, run_dir=run_dir, socket_path=socket_path)
        if cmd is None:
            warnings.warn(f"live={kind!r} observer is not available yet", stacklevel=2)
            continue
        try:
            procs.append(
                subprocess.Popen(  # noqa: S603 - fixed argv, no shell
                    cmd,
                    start_new_session=True,
                    # Detach stdio so the observer's output never bleeds into the
                    # training process's terminal/log. (A co-located local TUI
                    # can't share the training terminal — over SSH the web
                    # dashboard, EMRY-044, is the better observer.)
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            )
        except FileNotFoundError:
            warnings.warn(
                "could not launch the `emry` binary for the live dashboard "
                "(is it on PATH?)",
                stacklevel=2,
            )
    return procs
