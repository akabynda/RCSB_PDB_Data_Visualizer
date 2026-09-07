"""Compose the supported plot families behind the stable public plotter API."""

from __future__ import annotations

from .correlation import CorrelationPlotsMixin
from .counts import CountPlotsMixin
from .homologs import HomologPlotsMixin
from .programs import ProgramPlotsMixin
from .quality import QualityPlotsMixin
from .rendering import FigureRenderer
from .rmsd import RMSDPlotsMixin
from .weights import WeightPlotsMixin


class PDBScientificPlotter(
    CountPlotsMixin,
    ProgramPlotsMixin,
    WeightPlotsMixin,
    QualityPlotsMixin,
    HomologPlotsMixin,
    RMSDPlotsMixin,
    CorrelationPlotsMixin,
    FigureRenderer,
):
    """Render PDB datasets using focused plot families and a shared renderer.

    Feature mixins supply dataset-specific transformations and plotting methods.
    FigureRenderer owns configuration, CSV caching, common series, and output
    variants; inherited helpers are available to every plot family.
    """
