"""Register method-count, membrane, and software output paths."""

from __future__ import annotations

import argparse
from pathlib import Path


def add_count_output_arguments(parser: argparse.ArgumentParser) -> None:
    """Register method-count, membrane, and software output paths."""
    parser.add_argument(
        "--annual-output-png",
        type=Path,
        default=Path("figures/pdb_method_trends.png"),
        help="Output PNG for annual method-count figure.",
    )
    parser.add_argument(
        "--annual-output-svg",
        type=Path,
        default=Path("figures/pdb_method_trends.svg"),
        help="Output SVG for annual method-count figure.",
    )
    parser.add_argument(
        "--cumulative-output-png",
        type=Path,
        default=Path("figures/pdb_method_trends_cumulative.png"),
        help="Output PNG for cumulative method-count figure.",
    )
    parser.add_argument(
        "--cumulative-output-svg",
        type=Path,
        default=Path("figures/pdb_method_trends_cumulative.svg"),
        help="Output SVG for cumulative method-count figure.",
    )
    parser.add_argument(
        "--membrane-annual-output-png",
        type=Path,
        default=Path("figures/membrane_protein_counts_by_year.png"),
        help="Output PNG for annual membrane-protein count figure.",
    )
    parser.add_argument(
        "--membrane-annual-output-svg",
        type=Path,
        default=Path("figures/membrane_protein_counts_by_year.svg"),
        help="Output SVG for annual membrane-protein count figure.",
    )
    parser.add_argument(
        "--membrane-cumulative-output-png",
        type=Path,
        default=Path("figures/membrane_protein_counts_cumulative_by_year.png"),
        help="Output PNG for cumulative membrane-protein count figure.",
    )
    parser.add_argument(
        "--membrane-cumulative-output-svg",
        type=Path,
        default=Path("figures/membrane_protein_counts_cumulative_by_year.svg"),
        help="Output SVG for cumulative membrane-protein count figure.",
    )
    parser.add_argument(
        "--membrane-method-annual-output-png",
        type=Path,
        default=Path("figures/membrane_protein_method_counts_by_year.png"),
        help="Output PNG for annual membrane-protein method-count figure.",
    )
    parser.add_argument(
        "--membrane-method-annual-output-svg",
        type=Path,
        default=Path("figures/membrane_protein_method_counts_by_year.svg"),
        help="Output SVG for annual membrane-protein method-count figure.",
    )
    parser.add_argument(
        "--membrane-method-cumulative-output-png",
        type=Path,
        default=Path("figures/membrane_protein_method_counts_cumulative_by_year.png"),
        help="Output PNG for cumulative membrane-protein method-count figure.",
    )
    parser.add_argument(
        "--membrane-method-cumulative-output-svg",
        type=Path,
        default=Path("figures/membrane_protein_method_counts_cumulative_by_year.svg"),
        help="Output SVG for cumulative membrane-protein method-count figure.",
    )
    parser.add_argument(
        "--nmr-program-annual-output-png",
        type=Path,
        default=Path("figures/solution_nmr_program_trends_by_year.png"),
        help="Output PNG for annual SOLUTION NMR refinement program trend figure.",
    )
    parser.add_argument(
        "--nmr-program-annual-output-svg",
        type=Path,
        default=Path("figures/solution_nmr_program_trends_by_year.svg"),
        help="Output SVG for annual SOLUTION NMR refinement program trend figure.",
    )
    parser.add_argument(
        "--nmr-monomer-program-cluster-share-output-png",
        type=Path,
        default=Path("figures/solution_nmr_monomer_program_cluster_share_by_year.png"),
        help="Output PNG for SOLUTION NMR monomer program cluster share stacked-area plot.",
    )
    parser.add_argument(
        "--nmr-monomer-program-cluster-share-output-svg",
        type=Path,
        default=Path("figures/solution_nmr_monomer_program_cluster_share_by_year.svg"),
        help="Output SVG for SOLUTION NMR monomer program cluster share stacked-area plot.",
    )
    parser.add_argument(
        "--nmr-monomer-program-cluster-share-without-other-output-png",
        type=Path,
        default=Path(
            "figures/solution_nmr_monomer_program_cluster_share_without_other_by_year.png"
        ),
        help=(
            "Output PNG for SOLUTION NMR monomer program cluster share "
            "stacked-area plot excluding OTHER."
        ),
    )
    parser.add_argument(
        "--nmr-monomer-program-cluster-share-without-other-output-svg",
        type=Path,
        default=Path(
            "figures/solution_nmr_monomer_program_cluster_share_without_other_by_year.svg"
        ),
        help=(
            "Output SVG for SOLUTION NMR monomer program cluster share "
            "stacked-area plot excluding OTHER."
        ),
    )
