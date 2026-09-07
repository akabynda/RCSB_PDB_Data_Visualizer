"""Weighted refinement-program cluster quality summaries."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.dataset.config import (
    LOGGER,
)
from src.dataset.programs import (
    PROGRAM_CLUSTER_DEFINITIONS,
)
from src.dataset.records import (
    SolutionNMRMonomerProgramClusterTotalRecord,
    SolutionNMRMonomerProgramClusterYearlySummaryRecord,
)

if TYPE_CHECKING:
    from src.dataset.records import (
        SolutionNMRMonomerProgramClusterAssignmentRecord,
        SolutionNMRMonomerQualityRecord,
    )


def summarize_solution_nmr_monomer_program_cluster_quality_by_year(
    assignment_records: list[SolutionNMRMonomerProgramClusterAssignmentRecord],
    quality_records: list[SolutionNMRMonomerQualityRecord],
) -> list[SolutionNMRMonomerProgramClusterYearlySummaryRecord]:
    """Summarize monomer quality metrics by year and program cluster."""
    if not assignment_records or not quality_records:
        return []

    quality_by_key = {
        (record.entry_id.upper(), record.year): record for record in quality_records
    }
    yearly_totals: dict[int, dict[str, float | int]] = {}
    assignment_keys = {
        (record.entry_id.upper(), record.year) for record in assignment_records
    }
    matched_quality_keys: set[tuple[str, int]] = set()
    missing_quality_count = 0

    for key in sorted(assignment_keys):
        quality_record = quality_by_key.get(key)
        if quality_record is None:
            missing_quality_count += 1
            continue
        matched_quality_keys.add(key)
        total_row = yearly_totals.setdefault(
            quality_record.year,
            {
                "count": 0,
                "rama_sum": 0.0,
                "side_sum": 0.0,
                "clash_sum": 0.0,
            },
        )
        total_row["count"] += 1
        total_row["rama_sum"] += quality_record.ramachandran_outliers_percent
        total_row["side_sum"] += quality_record.sidechain_outliers_percent
        total_row["clash_sum"] += quality_record.clashscore

    unmatched_quality_count = len(quality_by_key) - len(matched_quality_keys)
    LOGGER.info(
        (
            "SOLUTION NMR monomer program-cluster yearly totals: years=%d, "
            "matched_entries=%d, missing_quality=%d, unmatched_quality=%d"
        ),
        len(yearly_totals),
        len(matched_quality_keys),
        missing_quality_count,
        unmatched_quality_count,
    )

    return [
        SolutionNMRMonomerProgramClusterYearlySummaryRecord(
            year=year,
            structure_count=int(total_row["count"]),
            avg_ramachandran_outliers_percent=(
                float(total_row["rama_sum"]) / int(total_row["count"])
                if int(total_row["count"]) > 0
                else None
            ),
            avg_sidechain_outliers_percent=(
                float(total_row["side_sum"]) / int(total_row["count"])
                if int(total_row["count"]) > 0
                else None
            ),
            avg_clashscore=(
                float(total_row["clash_sum"]) / int(total_row["count"])
                if int(total_row["count"]) > 0
                else None
            ),
        )
        for year, total_row in sorted(yearly_totals.items())
    ]


def summarize_solution_nmr_monomer_program_cluster_quality_total(
    assignment_records: list[SolutionNMRMonomerProgramClusterAssignmentRecord],
    quality_records: list[SolutionNMRMonomerQualityRecord],
) -> list[SolutionNMRMonomerProgramClusterTotalRecord]:
    """Summarize program-cluster quality metrics across all years."""
    if not assignment_records or not quality_records:
        return []

    quality_by_key = {
        (record.entry_id.upper(), record.year): record for record in quality_records
    }
    totals_by_cluster_id: dict[str, dict[str, float | int | str]] = {
        cluster_id: {
            "cluster_name": cluster_name,
            "count": 0,
            "rama_sum": 0.0,
            "side_sum": 0.0,
            "clash_sum": 0.0,
        }
        for cluster_id, cluster_name in PROGRAM_CLUSTER_DEFINITIONS
    }
    matched_quality_keys: set[tuple[str, int]] = set()
    missing_quality_count = 0

    for assignment_record in assignment_records:
        key = (assignment_record.entry_id.upper(), assignment_record.year)
        quality_record = quality_by_key.get(key)
        if quality_record is None:
            missing_quality_count += 1
            continue
        matched_quality_keys.add(key)
        total_row = totals_by_cluster_id.setdefault(
            assignment_record.cluster_id,
            {
                "cluster_name": assignment_record.cluster_name,
                "count": 0,
                "rama_sum": 0.0,
                "side_sum": 0.0,
                "clash_sum": 0.0,
            },
        )
        score = assignment_record.cluster_score
        total_row["count"] += score
        total_row["rama_sum"] += quality_record.ramachandran_outliers_percent * score
        total_row["side_sum"] += quality_record.sidechain_outliers_percent * score
        total_row["clash_sum"] += quality_record.clashscore * score

    unmatched_quality_count = len(quality_by_key) - len(matched_quality_keys)
    LOGGER.info(
        (
            "SOLUTION NMR monomer program-cluster totals: clusters=%d, "
            "matched_entries=%d, missing_quality=%d, unmatched_quality=%d"
        ),
        len(totals_by_cluster_id),
        len(matched_quality_keys),
        missing_quality_count,
        unmatched_quality_count,
    )

    ordered_records: list[SolutionNMRMonomerProgramClusterTotalRecord] = []
    for cluster_id, cluster_name in PROGRAM_CLUSTER_DEFINITIONS:
        total_row = totals_by_cluster_id.get(
            cluster_id,
            {
                "cluster_name": cluster_name,
                "count": 0,
                "rama_sum": 0.0,
                "side_sum": 0.0,
                "clash_sum": 0.0,
            },
        )
        count = float(total_row["count"])
        ordered_records.append(
            SolutionNMRMonomerProgramClusterTotalRecord(
                cluster_name=str(total_row["cluster_name"]),
                structure_count=count,
                avg_ramachandran_outliers_percent=(
                    float(total_row["rama_sum"]) / count if count > 0 else None
                ),
                avg_sidechain_outliers_percent=(
                    float(total_row["side_sum"]) / count if count > 0 else None
                ),
                avg_clashscore=(
                    float(total_row["clash_sum"]) / count if count > 0 else None
                ),
            )
        )
    return ordered_records
