"""Register X-ray RMSD comparison and correlation output paths."""

from __future__ import annotations

import argparse
from pathlib import Path


def add_rmsd_output_arguments(parser: argparse.ArgumentParser) -> None:
    """Register X-ray RMSD comparison and correlation output paths."""
    parser.add_argument(
        "--nmr-monomer-xray-rmsd-output-png",
        type=Path,
        default=Path("figures/solution_nmr_monomer_xray_rmsd_by_year.png"),
        help="Output PNG for monomer X-ray RMSD(CA) by year plot.",
    )
    parser.add_argument(
        "--nmr-monomer-xray-rmsd-output-svg",
        type=Path,
        default=Path("figures/solution_nmr_monomer_xray_rmsd_by_year.svg"),
        help="Output SVG for monomer X-ray RMSD(CA) by year plot.",
    )
    parser.add_argument(
        "--nmr-monomer-xray-median-rmsd-output-png",
        type=Path,
        default=Path("figures/solution_nmr_monomer_xray_median_rmsd_by_year.png"),
        help="Output PNG for monomer X-ray median RMSD(CA) by year plot.",
    )
    parser.add_argument(
        "--nmr-monomer-xray-median-rmsd-output-svg",
        type=Path,
        default=Path("figures/solution_nmr_monomer_xray_median_rmsd_by_year.svg"),
        help="Output SVG for monomer X-ray median RMSD(CA) by year plot.",
    )
    parser.add_argument(
        "--nmr-monomer-xray-min-rmsd-output-png",
        type=Path,
        default=Path("figures/solution_nmr_monomer_xray_min_rmsd_by_year.png"),
        help="Output PNG for monomer X-ray minimum RMSD(CA) yearly-mean plot.",
    )
    parser.add_argument(
        "--nmr-monomer-xray-min-rmsd-output-svg",
        type=Path,
        default=Path("figures/solution_nmr_monomer_xray_min_rmsd_by_year.svg"),
        help="Output SVG for monomer X-ray minimum RMSD(CA) yearly-mean plot.",
    )
    parser.add_argument(
        "--nmr-monomer-xray-min-median-rmsd-output-png",
        type=Path,
        default=Path("figures/solution_nmr_monomer_xray_min_median_rmsd_by_year.png"),
        help="Output PNG for monomer X-ray minimum RMSD(CA) yearly-median plot.",
    )
    parser.add_argument(
        "--nmr-monomer-xray-min-median-rmsd-output-svg",
        type=Path,
        default=Path("figures/solution_nmr_monomer_xray_min_median_rmsd_by_year.svg"),
        help="Output SVG for monomer X-ray minimum RMSD(CA) yearly-median plot.",
    )
    parser.add_argument(
        "--nmr-monomer-xray-rmsd-extremes-mean-output-png",
        type=Path,
        default=Path(
            "figures/solution_nmr_monomer_xray_rmsd_extremes_mean_by_year.png"
        ),
        help="Output PNG for monomer X-ray RMSD extremes yearly-mean comparison plot.",
    )
    parser.add_argument(
        "--nmr-monomer-xray-rmsd-extremes-mean-output-svg",
        type=Path,
        default=Path(
            "figures/solution_nmr_monomer_xray_rmsd_extremes_mean_by_year.svg"
        ),
        help="Output SVG for monomer X-ray RMSD extremes yearly-mean comparison plot.",
    )
    parser.add_argument(
        "--nmr-monomer-xray-rmsd-extremes-median-output-png",
        type=Path,
        default=Path(
            "figures/solution_nmr_monomer_xray_rmsd_extremes_median_by_year.png"
        ),
        help="Output PNG for monomer X-ray RMSD extremes yearly-median comparison plot.",
    )
    parser.add_argument(
        "--nmr-monomer-xray-rmsd-extremes-median-output-svg",
        type=Path,
        default=Path(
            "figures/solution_nmr_monomer_xray_rmsd_extremes_median_by_year.svg"
        ),
        help="Output SVG for monomer X-ray RMSD extremes yearly-median comparison plot.",
    )
    parser.add_argument(
        "--nmr-monomer-xray-rmsd-historical-output-png",
        type=Path,
        default=Path("figures/solution_nmr_monomer_xray_rmsd_historical_by_year.png"),
        help="Output PNG for historical monomer X-ray RMSD(CA) by year plot.",
    )
    parser.add_argument(
        "--nmr-monomer-xray-rmsd-historical-output-svg",
        type=Path,
        default=Path("figures/solution_nmr_monomer_xray_rmsd_historical_by_year.svg"),
        help="Output SVG for historical monomer X-ray RMSD(CA) by year plot.",
    )
    parser.add_argument(
        "--nmr-monomer-xray-median-rmsd-historical-output-png",
        type=Path,
        default=Path(
            "figures/solution_nmr_monomer_xray_median_rmsd_historical_by_year.png"
        ),
        help="Output PNG for historical monomer X-ray median RMSD(CA) by year plot.",
    )
    parser.add_argument(
        "--nmr-monomer-xray-median-rmsd-historical-output-svg",
        type=Path,
        default=Path(
            "figures/solution_nmr_monomer_xray_median_rmsd_historical_by_year.svg"
        ),
        help="Output SVG for historical monomer X-ray median RMSD(CA) by year plot.",
    )
    parser.add_argument(
        "--nmr-monomer-xray-min-rmsd-historical-output-png",
        type=Path,
        default=Path(
            "figures/solution_nmr_monomer_xray_min_rmsd_historical_by_year.png"
        ),
        help="Output PNG for historical monomer X-ray minimum RMSD(CA) yearly-mean plot.",
    )
    parser.add_argument(
        "--nmr-monomer-xray-min-rmsd-historical-output-svg",
        type=Path,
        default=Path(
            "figures/solution_nmr_monomer_xray_min_rmsd_historical_by_year.svg"
        ),
        help="Output SVG for historical monomer X-ray minimum RMSD(CA) yearly-mean plot.",
    )
    parser.add_argument(
        "--nmr-monomer-xray-min-median-rmsd-historical-output-png",
        type=Path,
        default=Path(
            "figures/solution_nmr_monomer_xray_min_median_rmsd_historical_by_year.png"
        ),
        help="Output PNG for historical monomer X-ray minimum RMSD(CA) yearly-median plot.",
    )
    parser.add_argument(
        "--nmr-monomer-xray-min-median-rmsd-historical-output-svg",
        type=Path,
        default=Path(
            "figures/solution_nmr_monomer_xray_min_median_rmsd_historical_by_year.svg"
        ),
        help="Output SVG for historical monomer X-ray minimum RMSD(CA) yearly-median plot.",
    )
    parser.add_argument(
        "--nmr-monomer-xray-rmsd-extremes-historical-mean-output-png",
        type=Path,
        default=Path(
            "figures/solution_nmr_monomer_xray_rmsd_extremes_historical_mean_by_year.png"
        ),
        help="Output PNG for historical monomer X-ray RMSD extremes yearly-mean comparison plot.",
    )
    parser.add_argument(
        "--nmr-monomer-xray-rmsd-extremes-historical-mean-output-svg",
        type=Path,
        default=Path(
            "figures/solution_nmr_monomer_xray_rmsd_extremes_historical_mean_by_year.svg"
        ),
        help="Output SVG for historical monomer X-ray RMSD extremes yearly-mean comparison plot.",
    )
    parser.add_argument(
        "--nmr-monomer-xray-rmsd-extremes-historical-median-output-png",
        type=Path,
        default=Path(
            "figures/solution_nmr_monomer_xray_rmsd_extremes_historical_median_by_year.png"
        ),
        help="Output PNG for historical monomer X-ray RMSD extremes yearly-median comparison plot.",
    )
    parser.add_argument(
        "--nmr-monomer-xray-rmsd-extremes-historical-median-output-svg",
        type=Path,
        default=Path(
            "figures/solution_nmr_monomer_xray_rmsd_extremes_historical_median_by_year.svg"
        ),
        help="Output SVG for historical monomer X-ray RMSD extremes yearly-median comparison plot.",
    )
    parser.add_argument(
        "--nmr-monomer-xray-rmsd-precision-scatter-output-png",
        type=Path,
        default=Path(
            "figures/solution_nmr_monomer_xray_min_rmsd_precision_correlation.png"
        ),
        help="Output PNG for minimum X-ray RMSD vs STRIDE-core precision scatter plot.",
    )
    parser.add_argument(
        "--nmr-monomer-xray-rmsd-precision-scatter-output-svg",
        type=Path,
        default=Path(
            "figures/solution_nmr_monomer_xray_min_rmsd_precision_correlation.svg"
        ),
        help="Output SVG for minimum X-ray RMSD vs STRIDE-core precision scatter plot.",
    )
    parser.add_argument(
        "--nmr-monomer-xray-rmsd-precision-yearly-correlation-output-png",
        type=Path,
        default=Path(
            "figures/solution_nmr_monomer_xray_min_rmsd_precision_yearly_correlation.png"
        ),
        help="Output PNG for yearly correlation between minimum X-ray RMSD and precision.",
    )
    parser.add_argument(
        "--nmr-monomer-xray-rmsd-precision-yearly-correlation-output-svg",
        type=Path,
        default=Path(
            "figures/solution_nmr_monomer_xray_min_rmsd_precision_yearly_correlation.svg"
        ),
        help="Output SVG for yearly correlation between minimum X-ray RMSD and precision.",
    )
    parser.add_argument(
        "--nmr-monomer-xray-rmsd-precision-cumulative-correlation-output-png",
        type=Path,
        default=Path(
            "figures/solution_nmr_monomer_xray_min_rmsd_precision_cumulative_correlation.png"
        ),
        help="Output PNG for cumulative correlation between minimum X-ray RMSD and precision.",
    )
    parser.add_argument(
        "--nmr-monomer-xray-rmsd-precision-cumulative-correlation-output-svg",
        type=Path,
        default=Path(
            "figures/solution_nmr_monomer_xray_min_rmsd_precision_cumulative_correlation.svg"
        ),
        help="Output SVG for cumulative correlation between minimum X-ray RMSD and precision.",
    )
