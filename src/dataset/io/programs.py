"""CSV serialization for refinement-program counts and cluster summaries."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING

from src.dataset.io.common import (
    write_csv_rows,
)
from src.dataset.records import (
    SolutionNMRMonomerProgramClusterAssignmentRecord,
)

if TYPE_CHECKING:
    from src.dataset.records import (
        SolutionNMRMonomerProgramClusterSummaryRecord,
        SolutionNMRMonomerProgramClusterTotalRecord,
        SolutionNMRMonomerProgramClusterYearlySummaryRecord,
        SolutionNMRProgramYearlyCountRecord,
    )


def write_solution_nmr_program_counts_csv(
    records: list[SolutionNMRProgramYearlyCountRecord], output_path: Path
) -> None:
    """Write yearly SOLUTION NMR program counts to CSV."""
    write_csv_rows(
        output_path=output_path,
        header=["year", "program", "count"],
        rows=((r.year, r.program, r.count) for r in records),
    )


def write_solution_nmr_monomer_program_cluster_assignments_csv(
    records: list[SolutionNMRMonomerProgramClusterAssignmentRecord],
    output_path: Path,
) -> None:
    """Write program-cluster assignment records to CSV."""
    write_csv_rows(
        output_path=output_path,
        header=[
            "entry_id",
            "year",
            "cluster_id",
            "cluster_name",
            "cluster_score",
            "has_program_text",
            "program_text",
        ],
        rows=(
            (
                r.entry_id,
                r.year,
                r.cluster_id,
                r.cluster_name,
                f"{r.cluster_score:.12g}",
                int(r.has_program_text),
                r.program_text,
            )
            for r in records
        ),
    )


def read_solution_nmr_monomer_program_cluster_assignments_csv(
    input_path: Path,
) -> list[SolutionNMRMonomerProgramClusterAssignmentRecord]:
    """Read program-cluster assignment records from CSV."""
    if not input_path.exists():
        return []
    records: list[SolutionNMRMonomerProgramClusterAssignmentRecord] = []
    with input_path.open("r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if not row:
                continue
            records.append(
                SolutionNMRMonomerProgramClusterAssignmentRecord(
                    entry_id=str(row["entry_id"]),
                    year=int(row["year"]),
                    cluster_id=str(row["cluster_id"]),
                    cluster_name=str(row["cluster_name"]),
                    cluster_score=float(row.get("cluster_score") or 1.0),
                    has_program_text=bool(int(row["has_program_text"])),
                    program_text=str(row["program_text"]),
                )
            )
    return records


def write_solution_nmr_monomer_program_cluster_summary_csv(
    records: list[SolutionNMRMonomerProgramClusterSummaryRecord],
    output_path: Path,
) -> None:
    """Write yearly program-cluster quality summary rows to CSV."""
    write_csv_rows(
        output_path=output_path,
        header=[
            "year",
            "cluster_id",
            "cluster_name",
            "structure_count",
            "avg_ramachandran_outliers_percent",
            "avg_sidechain_outliers_percent",
            "avg_clashscore",
        ],
        rows=(
            (
                r.year,
                r.cluster_id,
                r.cluster_name,
                f"{r.structure_count:.12g}",
                (
                    f"{r.avg_ramachandran_outliers_percent:.4f}"
                    if r.avg_ramachandran_outliers_percent is not None
                    else ""
                ),
                (
                    f"{r.avg_sidechain_outliers_percent:.4f}"
                    if r.avg_sidechain_outliers_percent is not None
                    else ""
                ),
                f"{r.avg_clashscore:.4f}" if r.avg_clashscore is not None else "",
            )
            for r in records
        ),
    )


def write_solution_nmr_monomer_program_cluster_yearly_summary_csv(
    records: list[SolutionNMRMonomerProgramClusterYearlySummaryRecord],
    output_path: Path,
) -> None:
    """Write overall yearly program-cluster quality summaries to CSV."""
    write_csv_rows(
        output_path=output_path,
        header=[
            "year",
            "structure_count",
            "avg_ramachandran_outliers_percent",
            "avg_sidechain_outliers_percent",
            "avg_clashscore",
        ],
        rows=(
            (
                r.year,
                r.structure_count,
                (
                    f"{r.avg_ramachandran_outliers_percent:.4f}"
                    if r.avg_ramachandran_outliers_percent is not None
                    else ""
                ),
                (
                    f"{r.avg_sidechain_outliers_percent:.4f}"
                    if r.avg_sidechain_outliers_percent is not None
                    else ""
                ),
                f"{r.avg_clashscore:.4f}" if r.avg_clashscore is not None else "",
            )
            for r in records
        ),
    )


def write_solution_nmr_monomer_program_cluster_total_csv(
    records: list[SolutionNMRMonomerProgramClusterTotalRecord],
    output_path: Path,
) -> None:
    """Write total program-cluster quality summaries to CSV."""
    write_csv_rows(
        output_path=output_path,
        header=[
            "cluster_name",
            "structure_count",
            "avg_ramachandran_outliers_percent",
            "avg_sidechain_outliers_percent",
            "avg_clashscore",
        ],
        rows=(
            (
                r.cluster_name,
                f"{r.structure_count:.12g}",
                (
                    f"{r.avg_ramachandran_outliers_percent:.4f}"
                    if r.avg_ramachandran_outliers_percent is not None
                    else ""
                ),
                (
                    f"{r.avg_sidechain_outliers_percent:.4f}"
                    if r.avg_sidechain_outliers_percent is not None
                    else ""
                ),
                f"{r.avg_clashscore:.4f}" if r.avg_clashscore is not None else "",
            )
            for r in records
        ),
    )
