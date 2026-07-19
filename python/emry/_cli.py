"""Locate and run the bundled ``emry`` Rust CLI binary.

The wheel bundles the compiled ``emry`` executable under ``emry/_bin/``. This
module backs the ``emry`` console script (so ``pip install emry`` gives you the
``emry`` command) and also lets the SDK's live dashboard find the binary even
when the interpreter's ``bin/`` directory is not on ``PATH``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

__all__ = ["binary_path", "main"]

_EXE = "emry.exe" if os.name == "nt" else "emry"


def binary_path() -> Optional[Path]:
    """Path to the bundled ``emry`` binary, or ``None`` if it isn't present."""
    candidate = Path(__file__).resolve().parent / "_bin" / _EXE
    return candidate if candidate.is_file() else None


def main() -> None:
    """Console-script entry point: exec the bundled ``emry`` with our argv."""
    binary = binary_path()
    if binary is None:
        sys.exit(
            "emry: the bundled CLI binary is missing from this install.\n"
            "Reinstall with `pip install --force-reinstall emry`, or build it "
            "from source with `cargo install --path crates/emry-cli`."
        )
    argv = [str(binary), *sys.argv[1:]]
    if os.name == "nt":
        # Windows has no execv that replaces the process cleanly for a console
        # app; run as a child and propagate the exit code.
        import subprocess

        raise SystemExit(subprocess.run(argv).returncode)  # noqa: S603
    os.execv(str(binary), argv)  # replace this process with the CLI
