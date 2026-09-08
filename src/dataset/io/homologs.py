"""X-ray homolog CSVs, rejection reports, and resume checkpoints."""

from __future__ import annotations

import csv
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from src.dataset.config import (
    LOGGER,
)
from src.dataset.io.common import (
    _atomic_write_csv_rows,
    write_csv_rows,
)
from src.dataset.records import (
    parse_residue_id,
    RejectedXrayHomologRecord,
    SolutionNMRMonomerXrayHomologRecord,
)

SOLUTION_NMR_MONOMER_XRAY_HOMOLOG_HEADER = [
    "entry_id",
    "year",
    "sequence_identity_percent",
    "nmr_core_start_seq_id",
    "nmr_core_end_seq_id",
    "nmr_query_sequence_length",
    "has_xray_homolog",
    "xray_homolog_count",
    "xray_homolog_entry_ids",
    "xray_homolog_entity_ids",
]


REJECTED_XRAY_HOMOLOG_HEADER = [
    "nmr_entry_id",
    "nmr_year",
    "nmr_chain_id",
    "sequence_identity_percent",
    "nmr_core_start_seq_id",
    "nmr_core_end_seq_id",
    "nmr_query_sequence_length",
    "xray_entry_id",
    "xray_entity_id",
    "xray_chain_ids",
    "reason",
]


def rejected_xray_homologs_csv_path(homolog_output_path: Path) -> Path:
    """Return the sibling CSV for candidates rejected from a homolog output."""
    homolog_output_path = Path(homolog_output_path)
    return homolog_output_path.with_name(f"{homolog_output_path.stem}_rejected.csv")


def _rejected_xray_homolog_csv_row(
    record: RejectedXrayHomologRecord,
) -> tuple[Any, ...]:
    """Convert one rejected X-ray candidate to its stable CSV representation."""
    return (
        record.nmr_entry_id,
        record.nmr_year,
        record.nmr_chain_id,
        record.sequence_identity_percent,
        (
            record.nmr_core_start_seq_id
            if record.nmr_core_start_seq_id is not None
            else ""
        ),
        record.nmr_core_end_seq_id if record.nmr_core_end_seq_id is not None else "",
        record.nmr_query_sequence_length,
        record.xray_entry_id,
        record.xray_entity_id,
        ";".join(record.xray_chain_ids),
        record.reason,
    )


def write_rejected_xray_homolog_csv(
    records: Iterable[RejectedXrayHomologRecord], output_path: Path
) -> None:
    """Write X-ray sequence hits rejected by homolog eligibility checks."""
    _atomic_write_csv_rows(
        output_path=output_path,
        header=REJECTED_XRAY_HOMOLOG_HEADER,
        rows=(_rejected_xray_homolog_csv_row(record) for record in records),
    )


def _read_rejected_xray_homolog_csv_with_status(
    input_path: Path,
) -> tuple[list[RejectedXrayHomologRecord], bool]:
    """Read a rejection report and indicate whether its full schema is valid."""
    if not input_path.exists():
        return [], False
    records: list[RejectedXrayHomologRecord] = []
    with input_path.open("r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        if reader.fieldnames != REJECTED_XRAY_HOMOLOG_HEADER:
            LOGGER.warning(
                "Ignoring incompatible rejected X-ray homolog CSV %s",
                input_path,
            )
            return [], False
        is_valid = True
        try:
            numbered_rows = enumerate(reader, start=2)
            for line_number, row in numbered_rows:
                if not row or not any(value for value in row.values() if value):
                    continue
                try:
                    if None in row:
                        raise ValueError("row has unexpected extra columns")
                    nmr_entry_id = str(row.get("nmr_entry_id") or "").strip()
                    xray_entry_id = str(row.get("xray_entry_id") or "").strip()
                    xray_entity_id = str(row.get("xray_entity_id") or "").strip()
                    reason = str(row.get("reason") or "").strip()
                    if not all((nmr_entry_id, xray_entry_id, xray_entity_id, reason)):
                        raise ValueError("required text field is empty")
                    nmr_core_start_raw = row.get("nmr_core_start_seq_id")
                    nmr_core_end_raw = row.get("nmr_core_end_seq_id")
                    record = RejectedXrayHomologRecord(
                        nmr_entry_id=nmr_entry_id,
                        nmr_year=int(row["nmr_year"]),
                        nmr_chain_id=str(row.get("nmr_chain_id") or ""),
                        sequence_identity_percent=int(row["sequence_identity_percent"]),
                        nmr_core_start_seq_id=(
                            parse_residue_id(nmr_core_start_raw)
                            if nmr_core_start_raw not in {None, ""}
                            else None
                        ),
                        nmr_core_end_seq_id=(
                            parse_residue_id(nmr_core_end_raw)
                            if nmr_core_end_raw not in {None, ""}
                            else None
                        ),
                        nmr_query_sequence_length=int(
                            row.get("nmr_query_sequence_length") or 0
                        ),
                        xray_entry_id=xray_entry_id,
                        xray_entity_id=xray_entity_id,
                        xray_chain_ids=tuple(
                            chain_id.strip()
                            for chain_id in str(row.get("xray_chain_ids") or "").split(
                                ";"
                            )
                            if chain_id.strip()
                        ),
                        reason=reason,
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    is_valid = False
                    LOGGER.warning(
                        "Ignoring malformed row %d in rejected X-ray homolog "
                        "CSV %s: %s",
                        line_number,
                        input_path,
                        exc,
                    )
                    continue
                records.append(record)
        except csv.Error as exc:
            LOGGER.warning(
                "Ignoring malformed tail in rejected X-ray homolog CSV %s: %s",
                input_path,
                exc,
            )
            is_valid = False
    return records, is_valid


def read_rejected_xray_homolog_csv(
    input_path: Path,
) -> list[RejectedXrayHomologRecord]:
    """Read valid rows from a homolog rejection report when present."""
    records, _ = _read_rejected_xray_homolog_csv_with_status(input_path)
    return records


def _solution_nmr_monomer_xray_homolog_csv_row(
    record: SolutionNMRMonomerXrayHomologRecord,
) -> tuple[Any, ...]:
    """Convert an X-ray homolog record into a CSV row dictionary."""
    return (
        record.entry_id,
        record.year,
        record.sequence_identity_percent,
        (
            record.nmr_core_start_seq_id
            if record.nmr_core_start_seq_id is not None
            else ""
        ),
        record.nmr_core_end_seq_id if record.nmr_core_end_seq_id is not None else "",
        record.nmr_query_sequence_length,
        int(record.has_xray_homolog),
        len(record.xray_homolog_entity_ids),
        ";".join(record.xray_homolog_entry_ids),
        ";".join(record.xray_homolog_entity_ids),
    )


def write_solution_nmr_monomer_xray_homolog_csv(
    records: list[SolutionNMRMonomerXrayHomologRecord], output_path: Path
) -> None:
    """Write X-ray homolog records to CSV."""
    write_csv_rows(
        output_path=output_path,
        header=SOLUTION_NMR_MONOMER_XRAY_HOMOLOG_HEADER,
        rows=(_solution_nmr_monomer_xray_homolog_csv_row(r) for r in records),
    )


def read_solution_nmr_monomer_xray_homolog_csv(
    input_path: Path,
) -> list[SolutionNMRMonomerXrayHomologRecord]:
    """Read X-ray homolog records from CSV."""
    if not input_path.exists():
        return []
    records: list[SolutionNMRMonomerXrayHomologRecord] = []
    with input_path.open("r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if not row:
                continue
            nmr_core_start_raw = row.get("nmr_core_start_seq_id")
            nmr_core_end_raw = row.get("nmr_core_end_seq_id")
            xray_entry_ids = tuple(
                item.strip()
                for item in str(row.get("xray_homolog_entry_ids") or "").split(";")
                if item.strip()
            )
            xray_entity_ids = tuple(
                item.strip()
                for item in str(row.get("xray_homolog_entity_ids") or "").split(";")
                if item.strip()
            )
            records.append(
                SolutionNMRMonomerXrayHomologRecord(
                    entry_id=str(row["entry_id"]),
                    year=int(row["year"]),
                    sequence_identity_percent=int(row["sequence_identity_percent"]),
                    nmr_core_start_seq_id=(
                        parse_residue_id(nmr_core_start_raw)
                        if nmr_core_start_raw not in {None, ""}
                        else None
                    ),
                    nmr_core_end_seq_id=(
                        parse_residue_id(nmr_core_end_raw)
                        if nmr_core_end_raw not in {None, ""}
                        else None
                    ),
                    nmr_query_sequence_length=int(
                        row.get("nmr_query_sequence_length") or 0
                    ),
                    xray_homolog_entry_ids=xray_entry_ids,
                    xray_homolog_entity_ids=xray_entity_ids,
                    has_xray_homolog=bool(int(row.get("has_xray_homolog") or 0)),
                )
            )
    return records


def _xray_homolog_resume_checkpoint_path(output_95_path: Path) -> Path:
    """Return the completion checkpoint path shared by a 95%/100% CSV pair."""
    return output_95_path.with_suffix(".resume.tsv")


def _read_xray_homolog_resume_checkpoint(checkpoint_path: Path) -> dict[str, str]:
    """Read the latest recognized completion status for each NMR entry."""
    statuses: dict[str, str] = {}
    if not checkpoint_path.exists():
        return statuses
    with checkpoint_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            entry_id, separator, status = line.rstrip("\n").partition("\t")
            if (
                separator
                and entry_id
                and status
                in {
                    "success",
                    "success_with_rejected_audit",
                    "pending_rejected_audit",
                    "ineligible",
                }
            ):
                statuses[entry_id] = status
    return statuses


def _write_xray_homolog_resume_checkpoint_statuses(
    checkpoint_path: Path,
    statuses: Iterable[tuple[str, str]],
    *,
    mode: str,
) -> None:
    """Durably replace or append homolog completion statuses."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_path.open(mode, encoding="utf-8") as checkpoint_file:
        for entry_id, status in statuses:
            checkpoint_file.write(f"{entry_id}\t{status}\n")
        checkpoint_file.flush()
        os.fsync(checkpoint_file.fileno())
