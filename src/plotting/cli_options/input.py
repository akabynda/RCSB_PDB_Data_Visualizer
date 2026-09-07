"""Register input CSV paths for all supported datasets."""

from __future__ import annotations

import argparse
from pathlib import Path


def add_input_arguments(parser: argparse.ArgumentParser) -> None:
    """Register input CSV paths for all supported datasets."""
    parser.add_argument(
        "--counts-input",
        type=Path,
        default=Path("data/pdb_method_counts_by_year.csv"),
        help="Input CSV for method_counts plot.",
    )
    parser.add_argument(
        "--membrane-counts-input",
        type=Path,
        default=Path("data/membrane_protein_counts_by_year.csv"),
        help="Input CSV for membrane_protein_counts plot.",
    )
    parser.add_argument(
        "--membrane-method-counts-input",
        type=Path,
        default=Path("data/membrane_protein_method_counts_by_year.csv"),
        help="Input CSV for membrane-protein counts split by experimental method.",
    )
    parser.add_argument(
        "--nmr-weights-input",
        type=Path,
        default=Path("data/solution_nmr_structure_weights.csv"),
        help="Input CSV for SOLUTION NMR weight-based plots.",
    )
    parser.add_argument(
        "--nmr-program-counts-input",
        type=Path,
        default=Path("data/solution_nmr_program_counts_by_year.csv"),
        help="Input CSV for SOLUTION NMR refinement program trend plot.",
    )
    parser.add_argument(
        "--nmr-monomer-program-cluster-summary-input",
        type=Path,
        default=Path("data/solution_nmr_monomer_program_cluster_quality_by_year.csv"),
        help="Input CSV for SOLUTION NMR monomer program cluster share plots.",
    )
    parser.add_argument(
        "--nmr-monomer-stride-modeled-first-model-input",
        type=Path,
        default=Path("data/solution_nmr_monomer_stride_modeled_first_model.csv"),
        help=(
            "Input CSV for SOLUTION NMR monomer STRIDE modeled-first-model "
            "secondary-structure plot."
        ),
    )
    parser.add_argument(
        "--nmr-monomer-precision-stride-modeled-first-model-input",
        type=Path,
        default=Path(
            "data/solution_nmr_monomer_precision_stride_modeled_first_model.csv"
        ),
        help=(
            "Input CSV for SOLUTION NMR monomer precision plots "
            "with STRIDE-defined modeled-first-model core."
        ),
    )
    parser.add_argument(
        "--nmr-monomer-quality-input",
        type=Path,
        default=Path("data/solution_nmr_monomer_quality_metrics.csv"),
        help="Input CSV for SOLUTION NMR monomer quality-metrics plots.",
    )
    parser.add_argument(
        "--nmr-monomer-xray-homolog-95-input",
        type=Path,
        default=Path("data/solution_nmr_monomer_xray_homologs_95.csv"),
        help="Input CSV for SOLUTION NMR monomer X-ray homologs at 95%% sequence identity.",
    )
    parser.add_argument(
        "--nmr-monomer-xray-homolog-100-input",
        type=Path,
        default=Path("data/solution_nmr_monomer_xray_homologs_100.csv"),
        help="Input CSV for SOLUTION NMR monomer X-ray homologs at 100%% sequence identity.",
    )
    parser.add_argument(
        "--nmr-monomer-xray-homolog-95-historical-input",
        type=Path,
        default=Path("data/solution_nmr_monomer_xray_homologs_95_historical.csv"),
        help=(
            "Input CSV for 95%% X-ray homologs released no later than the "
            "NMR entry deposit date."
        ),
    )
    parser.add_argument(
        "--nmr-monomer-xray-homolog-100-historical-input",
        type=Path,
        default=Path("data/solution_nmr_monomer_xray_homologs_100_historical.csv"),
        help=(
            "Input CSV for 100%% X-ray homologs released no later than the "
            "NMR entry deposit date."
        ),
    )
    parser.add_argument(
        "--nmr-monomer-xray-rmsd-input",
        type=Path,
        default=Path("data/solution_nmr_monomer_xray_rmsd.csv"),
        help="Input CSV for SOLUTION NMR monomer X-ray RMSD plot.",
    )
    parser.add_argument(
        "--nmr-monomer-xray-rmsd-historical-input",
        type=Path,
        default=Path("data/solution_nmr_monomer_xray_rmsd_historical.csv"),
        help="Input CSV for historical SOLUTION NMR monomer X-ray RMSD plot.",
    )
    parser.add_argument(
        "--nmr-monomer-xray-rmsd-extremes-input",
        type=Path,
        default=Path("data/solution_nmr_monomer_xray_rmsd_extremes.csv"),
        help="Input CSV for SOLUTION NMR monomer X-ray RMSD extremes plots.",
    )
    parser.add_argument(
        "--nmr-monomer-xray-rmsd-extremes-historical-input",
        type=Path,
        default=Path("data/solution_nmr_monomer_xray_rmsd_extremes_historical.csv"),
        help="Input CSV for historical SOLUTION NMR monomer X-ray RMSD extremes plots.",
    )
