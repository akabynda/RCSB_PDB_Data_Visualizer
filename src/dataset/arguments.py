"""Command-line arguments for the dataset builder."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.dataset.config import (
    DEFAULT_MAX_WORKERS,
    DEFAULT_PDB_CACHE_VALIDATION_HOURS,
    DEFAULT_STRIDE_CACHE_DIR,
    DEFAULT_STRIDE_INSTALL_DIR,
    DatasetKind,
)


def parse_dataset_kinds(raw_value: str) -> list[DatasetKind]:
    """Parse comma-separated dataset names into DatasetKind values."""
    if raw_value.strip().lower() == "all":
        return [
            DatasetKind.METHOD_COUNTS,
            DatasetKind.MEMBRANE_PROTEIN_COUNTS,
            DatasetKind.SOLUTION_NMR_PROGRAM_COUNTS,
            DatasetKind.SOLUTION_NMR_MONOMER_PROGRAM_CLUSTERS,
            DatasetKind.SOLUTION_NMR_WEIGHTS,
            DatasetKind.SOLUTION_NMR_MONOMER_STRIDE_MODELED_FIRST_MODEL,
            DatasetKind.SOLUTION_NMR_MONOMER_PRECISION_STRIDE_MODELED_FIRST_MODEL,
            DatasetKind.SOLUTION_NMR_MONOMER_QUALITY,
            DatasetKind.SOLUTION_NMR_MONOMER_EXPERIMENTS,
            DatasetKind.SOLUTION_NMR_MONOMER_XRAY_HOMOLOGS,
            DatasetKind.SOLUTION_NMR_MONOMER_XRAY_HOMOLOGS_HISTORICAL,
            DatasetKind.SOLUTION_NMR_MONOMER_XRAY_RMSD,
            DatasetKind.SOLUTION_NMR_MONOMER_XRAY_RMSD_HISTORICAL,
            DatasetKind.SOLUTION_NMR_MONOMER_XRAY_RMSD_EXTREMES,
            DatasetKind.SOLUTION_NMR_MONOMER_XRAY_RMSD_EXTREMES_HISTORICAL,
        ]
    raw_items = [item.strip() for item in raw_value.split(",") if item.strip()]
    selected: list[DatasetKind] = []
    for item in raw_items:
        try:
            selected.append(DatasetKind(item))
        except ValueError as exc:
            valid = ", ".join(dataset.value for dataset in DatasetKind)
            raise argparse.ArgumentTypeError(
                f"Unknown dataset '{item}'. Use one of: {valid}, all."
            ) from exc
    if not selected:
        raise argparse.ArgumentTypeError("No datasets selected.")
    return selected


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the builder CLI."""
    parser = argparse.ArgumentParser(
        description="Collect extensible PDB datasets from RCSB APIs."
    )
    parser.add_argument(
        "--datasets",
        type=parse_dataset_kinds,
        default=[
            DatasetKind.METHOD_COUNTS,
            DatasetKind.SOLUTION_NMR_WEIGHTS,
            DatasetKind.SOLUTION_NMR_MONOMER_STRIDE_MODELED_FIRST_MODEL,
        ],
        help=(
            "Comma-separated dataset kinds or 'all'. Available: "
            + ", ".join(dataset.value for dataset in DatasetKind)
            + ", all."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume homolog, precision, and X-ray RMSD calculations from "
            "existing results instead of rebuilding them from scratch."
        ),
    )
    parser.add_argument(
        "--counts-output",
        type=Path,
        default=Path("data/pdb_method_counts_by_year.csv"),
        help="Output CSV path for method_counts dataset.",
    )
    parser.add_argument(
        "--membrane-counts-output",
        type=Path,
        default=Path("data/membrane_protein_counts_by_year.csv"),
        help="Output CSV path for membrane_protein_counts dataset.",
    )
    parser.add_argument(
        "--membrane-method-counts-output",
        type=Path,
        default=Path("data/membrane_protein_method_counts_by_year.csv"),
        help=(
            "Output CSV path for membrane protein counts split by experimental method."
        ),
    )
    parser.add_argument(
        "--solution-nmr-output",
        type=Path,
        default=Path("data/solution_nmr_structure_weights.csv"),
        help="Output CSV path for solution_nmr_weights dataset.",
    )
    parser.add_argument(
        "--solution-nmr-monomer-experiments-output",
        type=Path,
        default=Path("data/solution_nmr_monomer_experiments.csv"),
        help="Output CSV path for NMR experiments of monomeric SOLUTION NMR entries.",
    )
    parser.add_argument(
        "--solution-nmr-program-counts-output",
        type=Path,
        default=Path("data/solution_nmr_program_counts_by_year.csv"),
        help="Output CSV path for solution_nmr_program_counts dataset.",
    )
    parser.add_argument(
        "--solution-nmr-program-cache-dir",
        type=Path,
        default=Path("data/pdb_cache"),
        help="Directory to cache downloaded PDB files for solution_nmr_program_counts dataset.",
    )
    parser.add_argument(
        "--pdb-cache-validation-hours",
        type=float,
        default=DEFAULT_PDB_CACHE_VALIDATION_HOURS,
        help=(
            "Hours between conditional remote validations of cached PDB files "
            "(default: 24; use 0 to validate on every access)."
        ),
    )
    parser.add_argument(
        "--solution-nmr-monomer-stride-modeled-first-model-output",
        type=Path,
        default=Path("data/solution_nmr_monomer_stride_modeled_first_model.csv"),
        help=(
            "Output CSV path for solution_nmr_monomer_stride_modeled_first_model "
            "dataset (STRIDE for modeled residues of the first model only)."
        ),
    )
    parser.add_argument(
        "--solution-nmr-monomer-cache-dir",
        type=Path,
        default=Path("data/pdb_cache"),
        help=(
            "Directory to cache PDB files used by the base solution_nmr_monomer "
            "model-length filter and coordinate-level monomer datasets."
        ),
    )
    parser.add_argument(
        "--stride-cache-dir",
        type=Path,
        default=DEFAULT_STRIDE_CACHE_DIR,
        help="Directory to cache first-model STRIDE state maps by structure.",
    )
    parser.add_argument(
        "--stride-install-dir",
        type=Path,
        default=DEFAULT_STRIDE_INSTALL_DIR,
        help=(
            "Root directory under which the automatically downloaded and built "
            "STRIDE source is stored by revision and platform (default: "
            "data/stride)."
        ),
    )
    parser.add_argument(
        "--solution-nmr-monomer-stride-executable",
        type=str,
        default="",
        help=(
            "Explicit STRIDE executable for STRIDE-based datasets. If omitted, "
            "the builder checks PATH and local builds, then downloads and builds "
            "the pinned STRIDE source automatically."
        ),
    )
    parser.add_argument(
        "--solution-nmr-monomer-precision-stride-modeled-first-model-output",
        type=Path,
        default=Path(
            "data/solution_nmr_monomer_precision_stride_modeled_first_model.csv"
        ),
        help=(
            "Output CSV path for solution_nmr_monomer_precision_stride_modeled_first_model "
            "dataset."
        ),
    )
    parser.add_argument(
        "--solution-nmr-monomer-quality-output",
        type=Path,
        default=Path("data/solution_nmr_monomer_quality_metrics.csv"),
        help="Output CSV path for solution_nmr_monomer_quality dataset.",
    )
    parser.add_argument(
        "--solution-nmr-monomer-program-cluster-input",
        type=Path,
        default=Path("data/solution_nmr_monomer_quality_metrics.csv"),
        help=(
            "Input CSV path for solution_nmr_monomer_program_clusters dataset. "
            "Expected format: solution_nmr_monomer_quality_metrics.csv."
        ),
    )
    parser.add_argument(
        "--solution-nmr-monomer-program-cluster-cache-dir",
        type=Path,
        default=Path("data/pdb_cache"),
        help=(
            "Directory with cached PDB files for solution_nmr_monomer_program_clusters "
            "dataset."
        ),
    )
    parser.add_argument(
        "--solution-nmr-monomer-program-cluster-assignment-output",
        type=Path,
        default=Path("data/solution_nmr_monomer_program_cluster_assignments.csv"),
        help=(
            "Output CSV path for per-entry assignments in "
            "solution_nmr_monomer_program_clusters dataset."
        ),
    )
    parser.add_argument(
        "--solution-nmr-monomer-program-cluster-summary-output",
        type=Path,
        default=Path("data/solution_nmr_monomer_program_cluster_quality_by_year.csv"),
        help=(
            "Output CSV path for yearly cluster summary in "
            "solution_nmr_monomer_program_clusters dataset."
        ),
    )
    parser.add_argument(
        "--solution-nmr-monomer-program-cluster-yearly-summary-output",
        type=Path,
        default=Path(
            "data/solution_nmr_monomer_program_cluster_quality_total_by_year.csv"
        ),
        help=(
            "Output CSV path for yearly totals across all program clusters in "
            "solution_nmr_monomer_program_clusters dataset."
        ),
    )
    parser.add_argument(
        "--solution-nmr-monomer-program-cluster-total-output",
        type=Path,
        default=Path("data/solution_nmr_monomer_program_cluster_quality_total.csv"),
        help=(
            "Output CSV path for totals across all years per program cluster in "
            "solution_nmr_monomer_program_clusters dataset."
        ),
    )
    parser.add_argument(
        "--solution-nmr-monomer-xray-homolog-95-output",
        type=Path,
        default=Path("data/solution_nmr_monomer_xray_homologs_95.csv"),
        help=(
            "Output CSV path for solution_nmr_monomer_xray_homologs dataset at "
            "95%% sequence identity; rejected candidates use sibling *_rejected.csv."
        ),
    )
    parser.add_argument(
        "--solution-nmr-monomer-xray-homolog-100-output",
        type=Path,
        default=Path("data/solution_nmr_monomer_xray_homologs_100.csv"),
        help=(
            "Output CSV path for solution_nmr_monomer_xray_homologs dataset at "
            "100%% sequence identity; rejected candidates use sibling *_rejected.csv."
        ),
    )
    parser.add_argument(
        "--solution-nmr-monomer-xray-homolog-95-historical-output",
        type=Path,
        default=Path("data/solution_nmr_monomer_xray_homologs_95_historical.csv"),
        help=(
            "Output CSV path for 95%% X-ray homologs released no later than "
            "the NMR entry deposit date."
        ),
    )
    parser.add_argument(
        "--solution-nmr-monomer-xray-homolog-100-historical-output",
        type=Path,
        default=Path("data/solution_nmr_monomer_xray_homologs_100_historical.csv"),
        help=(
            "Output CSV path for 100%% X-ray homologs released no later than "
            "the NMR entry deposit date."
        ),
    )
    parser.add_argument(
        "--solution-nmr-monomer-xray-rmsd-output",
        type=Path,
        default=Path("data/solution_nmr_monomer_xray_rmsd.csv"),
        help="Output CSV path for solution_nmr_monomer_xray_rmsd dataset.",
    )
    parser.add_argument(
        "--solution-nmr-monomer-xray-rmsd-historical-output",
        type=Path,
        default=Path("data/solution_nmr_monomer_xray_rmsd_historical.csv"),
        help=(
            "Output CSV path for X-ray RMSD calculated from already-released "
            "historical homologs."
        ),
    )
    parser.add_argument(
        "--solution-nmr-monomer-xray-rmsd-extremes-output",
        type=Path,
        default=Path("data/solution_nmr_monomer_xray_rmsd_extremes.csv"),
        help=("Output CSV path for solution_nmr_monomer_xray_rmsd_extremes dataset."),
    )
    parser.add_argument(
        "--solution-nmr-monomer-xray-rmsd-extremes-historical-output",
        type=Path,
        default=Path("data/solution_nmr_monomer_xray_rmsd_extremes_historical.csv"),
        help=(
            "Output CSV path for X-ray RMSD extremes calculated from "
            "already-released historical homologs."
        ),
    )
    parser.add_argument(
        "--precision-cache-dir",
        type=Path,
        default=Path("data/pdb_cache"),
        help="Directory to cache downloaded PDB files for precision calculation.",
    )
    parser.add_argument(
        "--precision-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help="Parallel workers for RMSD precision computation.",
    )
    parser.add_argument(
        "--xray-rmsd-cache-dir",
        type=Path,
        default=Path("data/pdb_cache"),
        help="Directory to cache downloaded PDB files for X-ray RMSD calculation.",
    )
    parser.add_argument(
        "--xray-rmsd-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help="Parallel workers for X-ray RMSD computation.",
    )
    parser.add_argument(
        "--xray-rmsd-sequence-identity",
        type=int,
        choices=(95, 100),
        default=100,
        help=(
            "Sequence identity cutoff for selecting X-ray homologs from the "
            "STRIDE-core homolog CSV (95 or 100)."
        ),
    )
    parser.add_argument(
        "--page-size", type=int, default=10000, help="Search API page size."
    )
    parser.add_argument(
        "--batch-size", type=int, default=300, help="GraphQL batch size."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help="Parallel workers for GraphQL calls.",
    )
    parser.add_argument(
        "--log-level", default="INFO", help="Logging level (DEBUG/INFO/WARNING/ERROR)."
    )
    return parser.parse_args()
