"""Coordinate the shared current and historical X-ray RMSD computation."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.dataset.builders.rmsd_outputs import (
    build_solution_nmr_monomer_xray_rmsd_outputs_to_csv,
)
from src.dataset.client import RCSBClient
from src.dataset.config import (
    DEFAULT_STRIDE_CACHE_DIR,
    DEFAULT_STRIDE_INSTALL_DIR,
    DatasetBuildConfig,
    DatasetKind,
)
from src.dataset.reporting import _set_active_dataset_warning_logs
from src.dataset.stride_install import ensure_stride_executable


def run_xray_rmsd(
    args: argparse.Namespace,
    client: RCSBClient,
    config: DatasetBuildConfig,
    output_paths_by_dataset: dict[DatasetKind, tuple[Path, ...]],
) -> None:
    """Build all requested current and historical X-ray RMSD outputs together."""
    rmsd_dataset_kinds = {
        DatasetKind.SOLUTION_NMR_MONOMER_XRAY_RMSD,
        DatasetKind.SOLUTION_NMR_MONOMER_XRAY_RMSD_HISTORICAL,
        DatasetKind.SOLUTION_NMR_MONOMER_XRAY_RMSD_EXTREMES,
        DatasetKind.SOLUTION_NMR_MONOMER_XRAY_RMSD_EXTREMES_HISTORICAL,
    }

    selected_rmsd_kinds = rmsd_dataset_kinds.intersection(args.datasets)
    if not selected_rmsd_kinds:
        return

    residue_selection = getattr(args, "rmsd_residue_selection", "core")
    stride_executable = ""
    if residue_selection == "secondary-structure":
        stride_executable = ensure_stride_executable(
            getattr(args, "solution_nmr_monomer_stride_executable", ""),
            Path(getattr(args, "stride_install_dir", DEFAULT_STRIDE_INSTALL_DIR)),
        )
        if stride_executable is None:
            raise RuntimeError(
                "STRIDE executable not found for secondary-structure X-ray RMSD. "
                "Provide a valid --solution-nmr-monomer-stride-executable path."
            )

    current_selected = bool(
        selected_rmsd_kinds
        & {
            DatasetKind.SOLUTION_NMR_MONOMER_XRAY_RMSD,
            DatasetKind.SOLUTION_NMR_MONOMER_XRAY_RMSD_EXTREMES,
        }
    )
    historical_selected = bool(
        selected_rmsd_kinds
        & {
            DatasetKind.SOLUTION_NMR_MONOMER_XRAY_RMSD_HISTORICAL,
            DatasetKind.SOLUTION_NMR_MONOMER_XRAY_RMSD_EXTREMES_HISTORICAL,
        }
    )
    selected_rmsd_output_paths = tuple(
        path
        for dataset_kind in selected_rmsd_kinds
        for path in output_paths_by_dataset[dataset_kind]
    )
    _set_active_dataset_warning_logs(selected_rmsd_output_paths)
    current_homolog_input_path = (
        (
            Path(args.solution_nmr_monomer_xray_homolog_95_output)
            if args.xray_rmsd_sequence_identity == 95
            else Path(args.solution_nmr_monomer_xray_homolog_100_output)
        )
        if current_selected
        else None
    )
    historical_homolog_input_path = (
        (
            Path(args.solution_nmr_monomer_xray_homolog_95_historical_output)
            if args.xray_rmsd_sequence_identity == 95
            else Path(args.solution_nmr_monomer_xray_homolog_100_historical_output)
        )
        if historical_selected
        else None
    )
    build_solution_nmr_monomer_xray_rmsd_outputs_to_csv(
        client=client,
        config=config,
        current_homolog_input_path=current_homolog_input_path,
        historical_homolog_input_path=historical_homolog_input_path,
        ordinary_output_path=(
            Path(args.solution_nmr_monomer_xray_rmsd_output)
            if DatasetKind.SOLUTION_NMR_MONOMER_XRAY_RMSD in selected_rmsd_kinds
            else None
        ),
        historical_ordinary_output_path=(
            Path(args.solution_nmr_monomer_xray_rmsd_historical_output)
            if DatasetKind.SOLUTION_NMR_MONOMER_XRAY_RMSD_HISTORICAL
            in selected_rmsd_kinds
            else None
        ),
        extremes_output_path=(
            Path(args.solution_nmr_monomer_xray_rmsd_extremes_output)
            if DatasetKind.SOLUTION_NMR_MONOMER_XRAY_RMSD_EXTREMES
            in selected_rmsd_kinds
            else None
        ),
        historical_extremes_output_path=(
            Path(args.solution_nmr_monomer_xray_rmsd_extremes_historical_output)
            if DatasetKind.SOLUTION_NMR_MONOMER_XRAY_RMSD_EXTREMES_HISTORICAL
            in selected_rmsd_kinds
            else None
        ),
        cache_dir=Path(args.xray_rmsd_cache_dir),
        rmsd_workers=args.xray_rmsd_workers,
        sequence_identity_percent=args.xray_rmsd_sequence_identity,
        resume=args.resume,
        residue_selection=residue_selection,
        stride_executable=stride_executable,
        stride_cache_dir=Path(
            getattr(args, "stride_cache_dir", DEFAULT_STRIDE_CACHE_DIR)
        ),
    )
