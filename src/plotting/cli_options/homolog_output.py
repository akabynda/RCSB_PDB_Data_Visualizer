"""Register X-ray homolog share, history, and timing output paths."""

from __future__ import annotations

import argparse
from pathlib import Path


def add_homolog_output_arguments(parser: argparse.ArgumentParser) -> None:
    """Register X-ray homolog share, history, and timing output paths."""
    parser.add_argument(
        "--nmr-monomer-xray-homolog-95-output-png",
        type=Path,
        default=Path("figures/solution_nmr_monomer_xray_homologs_95_by_year.png"),
        help="Output PNG for monomer X-ray homolog share plot at 95%% sequence identity.",
    )
    parser.add_argument(
        "--nmr-monomer-xray-homolog-95-output-svg",
        type=Path,
        default=Path("figures/solution_nmr_monomer_xray_homologs_95_by_year.svg"),
        help="Output SVG for monomer X-ray homolog share plot at 95%% sequence identity.",
    )
    parser.add_argument(
        "--nmr-monomer-xray-homolog-100-output-png",
        type=Path,
        default=Path("figures/solution_nmr_monomer_xray_homologs_100_by_year.png"),
        help="Output PNG for monomer X-ray homolog share plot at 100%% sequence identity.",
    )
    parser.add_argument(
        "--nmr-monomer-xray-homolog-100-output-svg",
        type=Path,
        default=Path("figures/solution_nmr_monomer_xray_homologs_100_by_year.svg"),
        help="Output SVG for monomer X-ray homolog share plot at 100%% sequence identity.",
    )
    parser.add_argument(
        "--nmr-monomer-xray-homolog-95-cumulative-output-png",
        type=Path,
        default=Path(
            "figures/solution_nmr_monomer_xray_homologs_95_cumulative_share_by_year.png"
        ),
        help="Output PNG for cumulative monomer X-ray homolog share plot at 95%% sequence identity.",
    )
    parser.add_argument(
        "--nmr-monomer-xray-homolog-95-cumulative-output-svg",
        type=Path,
        default=Path(
            "figures/solution_nmr_monomer_xray_homologs_95_cumulative_share_by_year.svg"
        ),
        help="Output SVG for cumulative monomer X-ray homolog share plot at 95%% sequence identity.",
    )
    parser.add_argument(
        "--nmr-monomer-xray-homolog-100-cumulative-output-png",
        type=Path,
        default=Path(
            "figures/solution_nmr_monomer_xray_homologs_100_cumulative_share_by_year.png"
        ),
        help="Output PNG for cumulative monomer X-ray homolog share plot at 100%% sequence identity.",
    )
    parser.add_argument(
        "--nmr-monomer-xray-homolog-100-cumulative-output-svg",
        type=Path,
        default=Path(
            "figures/solution_nmr_monomer_xray_homologs_100_cumulative_share_by_year.svg"
        ),
        help="Output SVG for cumulative monomer X-ray homolog share plot at 100%% sequence identity.",
    )
    parser.add_argument(
        "--nmr-monomer-xray-homolog-95-historical-output-png",
        type=Path,
        default=Path(
            "figures/solution_nmr_monomer_xray_homologs_95_historical_by_year.png"
        ),
        help="Output PNG for historical monomer X-ray homolog share plot at 95%% sequence identity.",
    )
    parser.add_argument(
        "--nmr-monomer-xray-homolog-95-historical-output-svg",
        type=Path,
        default=Path(
            "figures/solution_nmr_monomer_xray_homologs_95_historical_by_year.svg"
        ),
        help="Output SVG for historical monomer X-ray homolog share plot at 95%% sequence identity.",
    )
    parser.add_argument(
        "--nmr-monomer-xray-homolog-100-historical-output-png",
        type=Path,
        default=Path(
            "figures/solution_nmr_monomer_xray_homologs_100_historical_by_year.png"
        ),
        help="Output PNG for historical monomer X-ray homolog share plot at 100%% sequence identity.",
    )
    parser.add_argument(
        "--nmr-monomer-xray-homolog-100-historical-output-svg",
        type=Path,
        default=Path(
            "figures/solution_nmr_monomer_xray_homologs_100_historical_by_year.svg"
        ),
        help="Output SVG for historical monomer X-ray homolog share plot at 100%% sequence identity.",
    )
    parser.add_argument(
        "--nmr-monomer-xray-homolog-95-historical-cumulative-output-png",
        type=Path,
        default=Path(
            "figures/solution_nmr_monomer_xray_homologs_95_historical_cumulative_share_by_year.png"
        ),
        help="Output PNG for cumulative historical monomer X-ray homolog share plot at 95%% sequence identity.",
    )
    parser.add_argument(
        "--nmr-monomer-xray-homolog-95-historical-cumulative-output-svg",
        type=Path,
        default=Path(
            "figures/solution_nmr_monomer_xray_homologs_95_historical_cumulative_share_by_year.svg"
        ),
        help="Output SVG for cumulative historical monomer X-ray homolog share plot at 95%% sequence identity.",
    )
    parser.add_argument(
        "--nmr-monomer-xray-homolog-100-historical-cumulative-output-png",
        type=Path,
        default=Path(
            "figures/solution_nmr_monomer_xray_homologs_100_historical_cumulative_share_by_year.png"
        ),
        help="Output PNG for cumulative historical monomer X-ray homolog share plot at 100%% sequence identity.",
    )
    parser.add_argument(
        "--nmr-monomer-xray-homolog-100-historical-cumulative-output-svg",
        type=Path,
        default=Path(
            "figures/solution_nmr_monomer_xray_homologs_100_historical_cumulative_share_by_year.svg"
        ),
        help="Output SVG for cumulative historical monomer X-ray homolog share plot at 100%% sequence identity.",
    )
    parser.add_argument(
        "--nmr-monomer-xray-homolog-95-timing-share-output-png",
        type=Path,
        default=Path(
            "figures/solution_nmr_monomer_xray_homologs_95_timing_share_by_year.png"
        ),
        help="Output PNG for monomer X-ray homolog timing share plot at 95%% sequence identity.",
    )
    parser.add_argument(
        "--nmr-monomer-xray-homolog-95-timing-share-output-svg",
        type=Path,
        default=Path(
            "figures/solution_nmr_monomer_xray_homologs_95_timing_share_by_year.svg"
        ),
        help="Output SVG for monomer X-ray homolog timing share plot at 95%% sequence identity.",
    )
    parser.add_argument(
        "--nmr-monomer-xray-homolog-100-timing-share-output-png",
        type=Path,
        default=Path(
            "figures/solution_nmr_monomer_xray_homologs_100_timing_share_by_year.png"
        ),
        help="Output PNG for monomer X-ray homolog timing share plot at 100%% sequence identity.",
    )
    parser.add_argument(
        "--nmr-monomer-xray-homolog-100-timing-share-output-svg",
        type=Path,
        default=Path(
            "figures/solution_nmr_monomer_xray_homologs_100_timing_share_by_year.svg"
        ),
        help="Output SVG for monomer X-ray homolog timing share plot at 100%% sequence identity.",
    )
    parser.add_argument(
        "--nmr-monomer-xray-homolog-95-timing-counts-output-csv",
        type=Path,
        default=Path(
            "data/solution_nmr_monomer_xray_homologs_95_timing_counts_by_year.csv"
        ),
        help="Output CSV for yearly monomer X-ray homolog timing counts at 95%% sequence identity.",
    )
    parser.add_argument(
        "--nmr-monomer-xray-homolog-100-timing-counts-output-csv",
        type=Path,
        default=Path(
            "data/solution_nmr_monomer_xray_homologs_100_timing_counts_by_year.csv"
        ),
        help="Output CSV for yearly monomer X-ray homolog timing counts at 100%% sequence identity.",
    )
