"""Atomic CSV outputs for the membrane cohort and annotation audit."""

from __future__ import annotations

from pathlib import Path

from src.dataset.builders.membrane import (
    CHAIN_COLUMNS,
    DUPLICATE_CANDIDATE_COLUMNS,
    ENTITY_COLUMNS,
    ENTRY_COLUMNS,
    SEGMENT_COLUMNS,
    MembraneBuildResult,
)
from src.dataset.io.common import _atomic_write_csv_rows
from src.dataset.opm_annotations import OPM_ENTRY_COLUMNS, OPM_RESIDUE_COLUMNS, OPM_SUMMARY_COLUMNS


def membrane_output_paths(output_path: Path) -> dict[str, Path]:
    """Derive companion paths from the selected entry CSV stem."""
    output_path = Path(output_path)
    return {
        "entries": output_path,
        **{
            name: output_path.with_name(f"{output_path.stem}_{name}.csv")
            for name in ("entities", "chains", "segments", "duplicate_candidates", "opm_residues")
        },
    }


def write_solution_nmr_membrane_csvs(
    result: MembraneBuildResult, output_path: Path
) -> dict[str, Path]:
    """Write one row per stated unit; None becomes blank and booleans true/false."""
    paths = membrane_output_paths(output_path)
    tables = (
        ("entries", (*ENTRY_COLUMNS, *OPM_ENTRY_COLUMNS), result.entry_rows),
        ("entities", ENTITY_COLUMNS, result.entity_rows),
        ("chains", (*CHAIN_COLUMNS, *OPM_SUMMARY_COLUMNS), result.chain_rows),
        ("segments", SEGMENT_COLUMNS, result.segment_rows),
        ("duplicate_candidates", DUPLICATE_CANDIDATE_COLUMNS, result.duplicate_candidate_rows),
        ("opm_residues", OPM_RESIDUE_COLUMNS, result.opm_residue_rows),
    )
    for name, columns, rows in tables:
        _atomic_write_csv_rows(
            output_path=paths[name],
            header=columns,
            rows=(
                tuple(
                    str(value).lower() if isinstance(value, bool) else value
                    for value in (
                        row.get(column) if column.startswith("opm_") else row[column]
                        for column in columns
                    )
                )
                for row in rows
            ),
        )
    return paths
