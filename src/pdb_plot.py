"""Backward-compatible plotting API and command-line entry point.

Implementation lives in :mod:`src.plotting`, grouped by plotting concern.
"""

from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.plotting import PDBScientificPlotter, PlotConfig, PlotKind
from src.plotting.cli import main, parse_args, parse_plot_kinds, parse_positive_float
from src.plotting.constants import (
    AXIS_MAJOR_TICK_LENGTH,
    AXIS_MINOR_TICK_LENGTH,
    AXIS_MINOR_TICK_SUBDIVISIONS,
    DEFAULT_FIGURE_HEIGHT_INCHES,
    JMR_TWO_COLUMN_WIDTH_INCHES,
    MAX_PLOT_YEAR,
    NMR_MONOMER_PROGRAM_CLUSTER_COLORS,
    NMR_MONOMER_PROGRAM_CLUSTER_LABELS,
    NMR_MONOMER_PROGRAM_CLUSTER_ORDER,
    NMR_PROGRAM_TOP_N,
    NMR_WEIGHT_BINS,
    NMR_WEIGHT_LABELS,
    TITLE_FONTSIZE,
    XRAY_HOMOLOG_TIMING_LABELS,
    YEAR_MAJOR_TICK_STEP,
    YEAR_MINOR_TICK_STEP,
)
from src.plotting.style import _has_arial_font as _has_arial_font

__all__ = [
    "DEFAULT_FIGURE_HEIGHT_INCHES",
    "JMR_TWO_COLUMN_WIDTH_INCHES",
    "TITLE_FONTSIZE",
    "NMR_WEIGHT_BINS",
    "NMR_WEIGHT_LABELS",
    "XRAY_HOMOLOG_TIMING_LABELS",
    "MAX_PLOT_YEAR",
    "NMR_PROGRAM_TOP_N",
    "NMR_MONOMER_PROGRAM_CLUSTER_ORDER",
    "NMR_MONOMER_PROGRAM_CLUSTER_LABELS",
    "NMR_MONOMER_PROGRAM_CLUSTER_COLORS",
    "YEAR_MAJOR_TICK_STEP",
    "YEAR_MINOR_TICK_STEP",
    "AXIS_MINOR_TICK_SUBDIVISIONS",
    "AXIS_MAJOR_TICK_LENGTH",
    "AXIS_MINOR_TICK_LENGTH",
    "PDBScientificPlotter",
    "PlotConfig",
    "PlotKind",
    "main",
    "parse_args",
    "parse_plot_kinds",
    "parse_positive_float",
]


if __name__ == "__main__":
    main()
