"""Shared figure dimensions, category definitions, colors, and tick settings."""

from __future__ import annotations

import numpy as np

DEFAULT_FIGURE_HEIGHT_INCHES: float = 5.0

JMR_TWO_COLUMN_WIDTH_INCHES: float = 2244 / 300

TITLE_FONTSIZE: float = 12.0

NMR_WEIGHT_BINS: tuple[float, ...] = (
    0.0,
    10.0,
    np.nextafter(20.0, float("inf")),
    float("inf"),
)

NMR_WEIGHT_LABELS: tuple[str, ...] = ("<10 kDa", "10-20 kDa", ">20 kDa")

XRAY_HOMOLOG_TIMING_LABELS: tuple[str, str, str] = (
    "X-ray analog released prior to deposition",
    "X-ray analog released at a later date",
    "No X-ray analog",
)

MAX_PLOT_YEAR: int = 2024

NMR_PROGRAM_TOP_N: int = 8

NMR_MONOMER_PROGRAM_CLUSTER_ORDER: tuple[str, ...] = (
    "CLUSTER1",
    "CLUSTER2",
    "CLUSTER3",
    "CLUSTER4",
    "CLUSTER5",
    "CLUSTER6",
    "CLUSTER7",
    "CLUSTER8",
    "CLUSTER9",
)

NMR_MONOMER_PROGRAM_CLUSTER_LABELS: dict[str, str] = {
    "CLUSTER1": "AMBER",
    "CLUSTER2": "ARIA",
    "CLUSTER3": "CNS",
    "CLUSTER4": "CYANA",
    "CLUSTER5": "DISCOVER",
    "CLUSTER6": "DIANA/DYANA",
    "CLUSTER7": "X-PLOR",
    "CLUSTER8": "X-PLOR NIH",
    "CLUSTER9": "OTHER",
}

NMR_MONOMER_PROGRAM_CLUSTER_COLORS: tuple[str, ...] = (
    "#4c78a8",
    "#f58518",
    "#e45756",
    "#72b7b2",
    "#54a24b",
    "#eeca3b",
    "#b279a2",
    "#ff9da6",
    "#858585",
)

YEAR_MAJOR_TICK_STEP: int = 5

YEAR_MINOR_TICK_STEP: int = 1

AXIS_MINOR_TICK_SUBDIVISIONS: int = YEAR_MAJOR_TICK_STEP // YEAR_MINOR_TICK_STEP

AXIS_MAJOR_TICK_LENGTH: float = 5.0

AXIS_MINOR_TICK_LENGTH: float = 3.0
