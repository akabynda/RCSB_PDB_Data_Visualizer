"""Validate homolog resume data against paired CSVs and rejection audits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.dataset.config import LOGGER
from src.dataset.io.homologs import (
    _read_rejected_xray_homolog_csv_with_status,
    _read_xray_homolog_resume_checkpoint,
    _write_xray_homolog_resume_checkpoint_statuses,
    read_solution_nmr_monomer_xray_homolog_csv,
)
from src.dataset.records import (
    RejectedXrayHomologRecord,
    SolutionNMRMonomerXrayHomologRecord,
)


@dataclass(frozen=True)
class HomologOutputPaths:
    """Files updated together while building the two homolog identity cutoffs."""

    homolog_95: Path
    homolog_100: Path
    rejected_95: Path
    rejected_100: Path
    checkpoint: Path


@dataclass
class HomologResumeState:
    """Audited records and seeds safe to preserve when continuing a build."""

    existing_records_95: list[SolutionNMRMonomerXrayHomologRecord]
    existing_records_100: list[SolutionNMRMonomerXrayHomologRecord]
    existing_rejected_95: list[RejectedXrayHomologRecord]
    existing_rejected_100: list[RejectedXrayHomologRecord]
    skip_homolog_entry_ids: set[str]


def load_homolog_resume_state(
    resume: bool, paths: HomologOutputPaths
) -> HomologResumeState:
    """Keep only complete audited record pairs, retrying all other eligible seeds."""
    existing_records_95: list[SolutionNMRMonomerXrayHomologRecord] = []
    existing_records_100: list[SolutionNMRMonomerXrayHomologRecord] = []
    existing_rejected_95: list[RejectedXrayHomologRecord] = []
    existing_rejected_100: list[RejectedXrayHomologRecord] = []
    skip_homolog_entry_ids: set[str] = set()
    if resume:
        records_95_by_entry_id = {
            record.entry_id: record
            for record in read_solution_nmr_monomer_xray_homolog_csv(paths.homolog_95)
            if record.sequence_identity_percent == 95
            and record.nmr_query_sequence_length > 10
            and record.nmr_core_start_seq_id is not None
            and record.nmr_core_end_seq_id is not None
        }
        records_100_by_entry_id = {
            record.entry_id: record
            for record in read_solution_nmr_monomer_xray_homolog_csv(paths.homolog_100)
            if record.sequence_identity_percent == 100
            and record.nmr_query_sequence_length > 10
            and record.nmr_core_start_seq_id is not None
            and record.nmr_core_end_seq_id is not None
        }
        paired_csv_entry_ids = set(records_95_by_entry_id) & set(
            records_100_by_entry_id
        )
        checkpoint_statuses = _read_xray_homolog_resume_checkpoint(paths.checkpoint)
        audited_entry_ids = {
            entry_id
            for entry_id, status in checkpoint_statuses.items()
            if status == "success_with_rejected_audit"
        }
        paired_entry_ids = paired_csv_entry_ids & audited_entry_ids
        unaudited_pair_count = len(paired_csv_entry_ids - paired_entry_ids)
        if unaudited_pair_count:
            LOGGER.warning(
                "SOLUTION NMR monomer X-ray homolog resume: recomputing %d "
                "record pairs without a completed rejected-candidate audit",
                unaudited_pair_count,
            )
        (
            rejected_records_95,
            rejected_report_95_is_valid,
        ) = _read_rejected_xray_homolog_csv_with_status(paths.rejected_95)
        (
            rejected_records_100,
            rejected_report_100_is_valid,
        ) = _read_rejected_xray_homolog_csv_with_status(paths.rejected_100)
        rejected_report_95_is_valid = rejected_report_95_is_valid and all(
            record.sequence_identity_percent == 95 for record in rejected_records_95
        )
        rejected_report_100_is_valid = rejected_report_100_is_valid and all(
            record.sequence_identity_percent == 100 for record in rejected_records_100
        )
        if not (rejected_report_95_is_valid and rejected_report_100_is_valid):
            LOGGER.warning(
                "SOLUTION NMR monomer X-ray homolog resume: rejected-candidate "
                "reports are missing or incomplete; recomputing %d previously "
                "completed record pairs",
                len(paired_entry_ids),
            )
            _write_xray_homolog_resume_checkpoint_statuses(
                paths.checkpoint,
                (
                    (entry_id, "pending_rejected_audit")
                    for entry_id in sorted(paired_entry_ids)
                ),
                mode="a",
            )
            paired_entry_ids = set()
        existing_records_95 = sorted(
            (records_95_by_entry_id[entry_id] for entry_id in paired_entry_ids),
            key=lambda record: (record.year, record.entry_id),
        )
        existing_records_100 = sorted(
            (records_100_by_entry_id[entry_id] for entry_id in paired_entry_ids),
            key=lambda record: (record.year, record.entry_id),
        )
        existing_rejected_95 = sorted(
            (
                record
                for record in rejected_records_95
                if record.nmr_entry_id in paired_entry_ids
                and record.sequence_identity_percent == 95
            ),
            key=lambda record: (
                record.nmr_year,
                record.nmr_entry_id,
                record.xray_entity_id,
            ),
        )
        existing_rejected_100 = sorted(
            (
                record
                for record in rejected_records_100
                if record.nmr_entry_id in paired_entry_ids
                and record.sequence_identity_percent == 100
            ),
            key=lambda record: (
                record.nmr_year,
                record.nmr_entry_id,
                record.xray_entity_id,
            ),
        )
        completed_ineligible_entry_ids = {
            entry_id
            for entry_id, status in checkpoint_statuses.items()
            if status == "ineligible"
        }
        skip_homolog_entry_ids = paired_entry_ids | completed_ineligible_entry_ids
        LOGGER.info(
            "SOLUTION NMR monomer X-ray homolog resume: keeping %d completed record pairs and %d ineligible entries; retrying all other seeds",
            len(paired_entry_ids),
            len(completed_ineligible_entry_ids),
        )
    else:
        _write_xray_homolog_resume_checkpoint_statuses(
            paths.checkpoint,
            (),
            mode="w",
        )
    return HomologResumeState(
        existing_records_95=existing_records_95,
        existing_records_100=existing_records_100,
        existing_rejected_95=existing_rejected_95,
        existing_rejected_100=existing_rejected_100,
        skip_homolog_entry_ids=skip_homolog_entry_ids,
    )
