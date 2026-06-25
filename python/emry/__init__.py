"""Emry — gentle observability for long training runs."""

from emry.mode import DeployMode
from emry.phase import Phase
from emry.run import Run, run

__version__ = "0.1.0"

__all__ = ["DeployMode", "Phase", "Run", "run", "__version__"]
