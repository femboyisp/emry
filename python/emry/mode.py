"""Deployment-mode detection — the Python mirror of the Rust ``DeployMode``.

This must match ``emry-core``'s ``mode.rs`` (EMRY-005) exactly so embedded and
sidecar clients agree on where the engine runs:

Resolution precedence:

1. An explicit API argument (``mode=`` passed by the caller).
2. The ``EMRY_MODE`` environment variable (``embedded`` | ``sidecar`` | ``file``,
   case-insensitive; an unrecognized, empty, or whitespace-only value is ignored
   and falls through).
3. Auto-detection: ``SSH_CONNECTION`` or ``SLURM_JOB_ID`` present → sidecar;
   else stdout is a TTY → embedded; else → file.

As in Rust, detection is a pure function of a :class:`DeployEnv` so it is tested
deterministically; only :meth:`DeployEnv.from_env` touches the real environment.
"""

from __future__ import annotations

import enum
import os
import sys
from dataclasses import dataclass
from typing import Optional

__all__ = ["DeployMode", "DeployEnv", "resolve", "detect"]


class DeployMode(enum.Enum):
    """Where the Emry engine runs relative to the training process."""

    EMBEDDED = "embedded"
    SIDECAR = "sidecar"
    FILE = "file"

    @classmethod
    def parse(cls, value: Optional[str]) -> Optional["DeployMode"]:
        """Parses a mode string (case-insensitive, trimmed), or ``None`` if the
        value is missing, empty, whitespace, or unrecognized."""
        if value is None:
            return None
        try:
            return cls(value.strip().lower())
        except ValueError:
            return None


@dataclass
class DeployEnv:
    """The inputs that determine the auto-detected deploy mode."""

    emry_mode: Optional[str] = None
    ssh_connection: bool = False
    slurm_job_id: bool = False
    stdout_is_tty: bool = False

    @classmethod
    def from_env(cls) -> "DeployEnv":
        """Reads the real process environment and stdout TTY status."""
        isatty = getattr(sys.stdout, "isatty", None)
        return cls(
            emry_mode=os.environ.get("EMRY_MODE"),
            ssh_connection="SSH_CONNECTION" in os.environ,
            slurm_job_id="SLURM_JOB_ID" in os.environ,
            stdout_is_tty=bool(isatty()) if callable(isatty) else False,
        )

    def auto_detect(self) -> DeployMode:
        """Mode from these inputs alone, ignoring ``EMRY_MODE`` and any override."""
        if self.ssh_connection or self.slurm_job_id:
            return DeployMode.SIDECAR
        if self.stdout_is_tty:
            return DeployMode.EMBEDDED
        return DeployMode.FILE


def resolve(api: Optional[DeployMode], env: DeployEnv) -> DeployMode:
    """Resolves the effective mode: ``api`` → ``EMRY_MODE`` → auto-detect."""
    if api is not None:
        return api
    from_env = DeployMode.parse(env.emry_mode)
    if from_env is not None:
        return from_env
    return env.auto_detect()


def detect(api: Optional[DeployMode] = None) -> DeployMode:
    """Resolves against the real environment, honouring an optional override."""
    return resolve(api, DeployEnv.from_env())
