"""Coordinate requested dataset builds while preserving CLI execution order."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.dataset.arguments import parse_args
from src.dataset.client import RCSBClient
from src.dataset.config import (
    DEFAULT_MAX_WORKERS,
    DatasetBuildConfig,
    DatasetKind,
)
from src.dataset.reporting import (
    _configure_dataset_filtered_csvs,
    _configure_dataset_warning_logs,
    _set_active_dataset_warning_logs,
)
from src.dataset.workflows.counts import (
    run_membrane_counts,
    run_method_counts,
)
from src.dataset.workflows.historical_homologs import run_historical_homologs
from src.dataset.workflows.homologs import run_homologs
from src.dataset.workflows.nmr import (
    run_experiments,
    run_program_counts,
    run_weights,
)
from src.dataset.workflows.quality import (
    run_program_clusters,
    run_quality,
)
from src.dataset.workflows.stride import (
    run_precision,
    run_stride,
)
from src.dataset.workflows.xray import run_xray_rmsd


def _selected_dataset_output_paths(
    args: argparse.Namespace,
) -> dict[DatasetKind, tuple[Path, ...]]:
    """Map selected dataset kinds to every CSV file they produce."""
    all_paths: dict[DatasetKind, tuple[Path, ...]] = {
        DatasetKind.METHOD_COUNTS: (Path(args.counts_output),),
        DatasetKind.MEMBRANE_PROTEIN_COUNTS: (
            Path(args.membrane_counts_output),
            Path(args.membrane_method_counts_output),
        ),
        DatasetKind.SOLUTION_NMR_PROGRAM_COUNTS: (
            Path(args.solution_nmr_program_counts_output),
        ),
        DatasetKind.SOLUTION_NMR_MONOMER_PROGRAM_CLUSTERS: (
            Path(args.solution_nmr_monomer_program_cluster_assignment_output),
            Path(args.solution_nmr_monomer_program_cluster_summary_output),
            Path(args.solution_nmr_monomer_program_cluster_yearly_summary_output),
            Path(args.solution_nmr_monomer_program_cluster_total_output),
        ),
        DatasetKind.SOLUTION_NMR_WEIGHTS: (Path(args.solution_nmr_output),),
        DatasetKind.SOLUTION_NMR_MONOMER_STRIDE_MODELED_FIRST_MODEL: (
            Path(args.solution_nmr_monomer_stride_modeled_first_model_output),
        ),
        DatasetKind.SOLUTION_NMR_MONOMER_PRECISION_STRIDE_MODELED_FIRST_MODEL: (
            Path(args.solution_nmr_monomer_precision_stride_modeled_first_model_output),
        ),
        DatasetKind.SOLUTION_NMR_MONOMER_QUALITY: (
            Path(args.solution_nmr_monomer_quality_output),
        ),
        DatasetKind.SOLUTION_NMR_MONOMER_EXPERIMENTS: (
            Path(args.solution_nmr_monomer_experiments_output),
        ),
        DatasetKind.SOLUTION_NMR_MONOMER_XRAY_HOMOLOGS: (
            Path(args.solution_nmr_monomer_xray_homolog_95_output),
            Path(args.solution_nmr_monomer_xray_homolog_100_output),
        ),
        DatasetKind.SOLUTION_NMR_MONOMER_XRAY_HOMOLOGS_HISTORICAL: (
            Path(args.solution_nmr_monomer_xray_homolog_95_historical_output),
            Path(args.solution_nmr_monomer_xray_homolog_100_historical_output),
        ),
        DatasetKind.SOLUTION_NMR_MONOMER_XRAY_RMSD: (
            Path(args.solution_nmr_monomer_xray_rmsd_output),
        ),
        DatasetKind.SOLUTION_NMR_MONOMER_XRAY_RMSD_HISTORICAL: (
            Path(args.solution_nmr_monomer_xray_rmsd_historical_output),
        ),
        DatasetKind.SOLUTION_NMR_MONOMER_XRAY_RMSD_EXTREMES: (
            Path(args.solution_nmr_monomer_xray_rmsd_extremes_output),
        ),
        DatasetKind.SOLUTION_NMR_MONOMER_XRAY_RMSD_EXTREMES_HISTORICAL: (
            Path(args.solution_nmr_monomer_xray_rmsd_extremes_historical_output),
        ),
    }
    return {dataset: all_paths[dataset] for dataset in args.datasets}


def main() -> None:
    """Run the requested dataset collection workflow from the CLI."""
    print(f"Using up to {DEFAULT_MAX_WORKERS} worker threads for concurrent tasks")
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    output_paths_by_dataset = _selected_dataset_output_paths(args)
    _configure_dataset_warning_logs(output_paths_by_dataset)
    preserved_filter_outputs: list[Path] = []
    if args.resume and DatasetKind.SOLUTION_NMR_MONOMER_XRAY_HOMOLOGS in args.datasets:
        preserved_filter_outputs.extend(
            output_paths_by_dataset[DatasetKind.SOLUTION_NMR_MONOMER_XRAY_HOMOLOGS]
        )
    _configure_dataset_filtered_csvs(
        output_paths_by_dataset,
        preserve_existing_output_paths=preserved_filter_outputs,
    )
    config = DatasetBuildConfig(
        page_size=args.page_size,
        graphql_batch_size=args.batch_size,
        max_workers=args.workers,
        pdb_cache_validation_hours=args.pdb_cache_validation_hours,
    )
    client = RCSBClient(
        config=config,
        solution_nmr_monomer_cache_dir=Path(args.solution_nmr_monomer_cache_dir),
    )

    # Keep prerequisites before the workflows that consume their CSV outputs.
    runners = (
        (DatasetKind.METHOD_COUNTS, run_method_counts),
        (DatasetKind.MEMBRANE_PROTEIN_COUNTS, run_membrane_counts),
        (DatasetKind.SOLUTION_NMR_PROGRAM_COUNTS, run_program_counts),
        (DatasetKind.SOLUTION_NMR_WEIGHTS, run_weights),
        (DatasetKind.SOLUTION_NMR_MONOMER_EXPERIMENTS, run_experiments),
        (DatasetKind.SOLUTION_NMR_MONOMER_STRIDE_MODELED_FIRST_MODEL, run_stride),
        (
            DatasetKind.SOLUTION_NMR_MONOMER_PRECISION_STRIDE_MODELED_FIRST_MODEL,
            run_precision,
        ),
        (DatasetKind.SOLUTION_NMR_MONOMER_QUALITY, run_quality),
        (DatasetKind.SOLUTION_NMR_MONOMER_PROGRAM_CLUSTERS, run_program_clusters),
        (DatasetKind.SOLUTION_NMR_MONOMER_XRAY_HOMOLOGS, run_homologs),
        (
            DatasetKind.SOLUTION_NMR_MONOMER_XRAY_HOMOLOGS_HISTORICAL,
            run_historical_homologs,
        ),
    )
    for dataset_kind, runner in runners:
        if dataset_kind in args.datasets:
            _set_active_dataset_warning_logs(output_paths_by_dataset[dataset_kind])
            runner(args, client, config)

    run_xray_rmsd(args, client, config, output_paths_by_dataset)
