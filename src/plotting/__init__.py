"""Scientific plotting organized by figure infrastructure and dataset family.

Use PDBScientificPlotter and PlotConfig for the Python API. The cli module owns
command-line parsing and dispatch; feature modules own their plot operations.
"""

from .config import PlotConfig, PlotKind
from .plotter import PDBScientificPlotter

__all__ = ["PDBScientificPlotter", "PlotConfig", "PlotKind"]
