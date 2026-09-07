"""NMR-to-X-ray RMSD CSV schemas and serialization."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from src.dataset.io.common import (
    write_csv_rows,
)
from src.dataset.records import (
    SolutionNMRMonomerXrayRmsdExtremesRecord,
    SolutionNMRMonomerXrayRmsdRecord,
)


def read_solution_nmr_monomer_xray_rmsd_csv(
    input_path: Path,
) -> list[SolutionNMRMonomerXrayRmsdRecord]:
    """Read NMR-to-X-ray RMSD records from CSV."""
    if not input_path.exists():
        return []
    records: list[SolutionNMRMonomerXrayRmsdRecord] = []
    with input_path.open("r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if not row:
                continue
            nmr_core_start_raw = row.get("nmr_core_start_seq_id")
            nmr_core_end_raw = row.get("nmr_core_end_seq_id")
            xray_core_start_raw = row.get("xray_core_start_seq_id")
            xray_core_end_raw = row.get("xray_core_end_seq_id")
            records.append(
                SolutionNMRMonomerXrayRmsdRecord(
                    entry_id=str(row["entry_id"]),
                    year=int(row["year"]),
                    sequence_identity_percent=int(row["sequence_identity_percent"]),
                    nmr_chain_id=str(row["nmr_chain_id"]),
                    nmr_core_start_seq_id=(
                        int(nmr_core_start_raw)
                        if nmr_core_start_raw not in {None, ""}
                        else None
                    ),
                    nmr_core_end_seq_id=(
                        int(nmr_core_end_raw)
                        if nmr_core_end_raw not in {None, ""}
                        else None
                    ),
                    nmr_query_sequence_length=int(
                        row.get("nmr_query_sequence_length") or 0
                    ),
                    xray_homolog_entity_id=str(row.get("xray_homolog_entity_id") or ""),
                    xray_homolog_count=int(row.get("xray_homolog_count") or 0),
                    xray_entry_id=str(row["xray_entry_id"]),
                    xray_chain_id=str(row["xray_chain_id"]),
                    xray_core_start_seq_id=(
                        int(xray_core_start_raw)
                        if xray_core_start_raw not in {None, ""}
                        else None
                    ),
                    xray_core_end_seq_id=(
                        int(xray_core_end_raw)
                        if xray_core_end_raw not in {None, ""}
                        else None
                    ),
                    xray_resolution_angstrom=float(row["xray_resolution_angstrom"]),
                    n_common_ca=int(row["n_common_ca"]),
                    rmsd_ca_angstrom=float(row["rmsd_ca_angstrom"]),
                )
            )
    return records


SOLUTION_NMR_MONOMER_XRAY_RMSD_HEADER: tuple[str, ...] = (
    "entry_id",
    "year",
    "sequence_identity_percent",
    "nmr_chain_id",
    "nmr_core_start_seq_id",
    "nmr_core_end_seq_id",
    "nmr_query_sequence_length",
    "xray_homolog_entity_id",
    "xray_homolog_count",
    "xray_entry_id",
    "xray_chain_id",
    "xray_core_start_seq_id",
    "xray_core_end_seq_id",
    "xray_resolution_angstrom",
    "n_common_ca",
    "rmsd_ca_angstrom",
)


def _solution_nmr_monomer_xray_rmsd_csv_row(
    record: SolutionNMRMonomerXrayRmsdRecord,
) -> tuple[Any, ...]:
    """Convert an NMR-to-X-ray RMSD record into a CSV row dictionary."""
    return (
        record.entry_id,
        record.year,
        record.sequence_identity_percent,
        record.nmr_chain_id,
        (
            record.nmr_core_start_seq_id
            if record.nmr_core_start_seq_id is not None
            else ""
        ),
        record.nmr_core_end_seq_id if record.nmr_core_end_seq_id is not None else "",
        record.nmr_query_sequence_length,
        record.xray_homolog_entity_id,
        record.xray_homolog_count,
        record.xray_entry_id,
        record.xray_chain_id,
        (
            record.xray_core_start_seq_id
            if record.xray_core_start_seq_id is not None
            else ""
        ),
        record.xray_core_end_seq_id if record.xray_core_end_seq_id is not None else "",
        f"{record.xray_resolution_angstrom:.4f}",
        record.n_common_ca,
        f"{record.rmsd_ca_angstrom:.4f}",
    )


def write_solution_nmr_monomer_xray_rmsd_csv(
    records: list[SolutionNMRMonomerXrayRmsdRecord], output_path: Path
) -> None:
    """Write NMR-to-X-ray RMSD records to CSV."""
    write_csv_rows(
        output_path=output_path,
        header=list(SOLUTION_NMR_MONOMER_XRAY_RMSD_HEADER),
        rows=(_solution_nmr_monomer_xray_rmsd_csv_row(r) for r in records),
    )


SOLUTION_NMR_MONOMER_XRAY_RMSD_EXTREMES_HEADER: tuple[str, ...] = (
    "entry_id",
    "year",
    "sequence_identity_percent",
    "nmr_chain_id",
    "nmr_core_start_seq_id",
    "nmr_core_end_seq_id",
    "nmr_query_sequence_length",
    "xray_homolog_count",
    "successful_xray_homolog_count",
    "best_xray_homolog_entity_id",
    "best_xray_entry_id",
    "best_xray_chain_id",
    "best_xray_resolution_angstrom",
    "best_xray_core_start_seq_id",
    "best_xray_core_end_seq_id",
    "best_n_common_ca",
    "best_rmsd_ca_angstrom",
    "worst_xray_homolog_entity_id",
    "worst_xray_entry_id",
    "worst_xray_chain_id",
    "worst_xray_resolution_angstrom",
    "worst_xray_core_start_seq_id",
    "worst_xray_core_end_seq_id",
    "worst_n_common_ca",
    "worst_rmsd_ca_angstrom",
    "rmsd_delta_angstrom",
)


def _solution_nmr_monomer_xray_rmsd_extremes_csv_row(
    record: SolutionNMRMonomerXrayRmsdExtremesRecord,
) -> tuple[Any, ...]:
    """Convert an RMSD extremes record into a CSV row dictionary."""
    return (
        record.entry_id,
        record.year,
        record.sequence_identity_percent,
        record.nmr_chain_id,
        (
            record.nmr_core_start_seq_id
            if record.nmr_core_start_seq_id is not None
            else ""
        ),
        record.nmr_core_end_seq_id if record.nmr_core_end_seq_id is not None else "",
        record.nmr_query_sequence_length,
        record.xray_homolog_count,
        record.successful_xray_homolog_count,
        record.best_xray_homolog_entity_id,
        record.best_xray_entry_id,
        record.best_xray_chain_id,
        f"{record.best_xray_resolution_angstrom:.4f}",
        (
            record.best_xray_core_start_seq_id
            if record.best_xray_core_start_seq_id is not None
            else ""
        ),
        (
            record.best_xray_core_end_seq_id
            if record.best_xray_core_end_seq_id is not None
            else ""
        ),
        record.best_n_common_ca,
        f"{record.best_rmsd_ca_angstrom:.4f}",
        record.worst_xray_homolog_entity_id,
        record.worst_xray_entry_id,
        record.worst_xray_chain_id,
        f"{record.worst_xray_resolution_angstrom:.4f}",
        (
            record.worst_xray_core_start_seq_id
            if record.worst_xray_core_start_seq_id is not None
            else ""
        ),
        (
            record.worst_xray_core_end_seq_id
            if record.worst_xray_core_end_seq_id is not None
            else ""
        ),
        record.worst_n_common_ca,
        f"{record.worst_rmsd_ca_angstrom:.4f}",
        f"{record.rmsd_delta_angstrom:.4f}",
    )


def read_solution_nmr_monomer_xray_rmsd_extremes_csv(
    input_path: Path,
) -> list[SolutionNMRMonomerXrayRmsdExtremesRecord]:
    """Read NMR-to-X-ray RMSD extremes records from CSV."""
    if not input_path.exists():
        return []
    records: list[SolutionNMRMonomerXrayRmsdExtremesRecord] = []
    with input_path.open("r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if not row:
                continue
            nmr_core_start_raw = row.get("nmr_core_start_seq_id")
            nmr_core_end_raw = row.get("nmr_core_end_seq_id")
            best_core_start_raw = row.get("best_xray_core_start_seq_id")
            best_core_end_raw = row.get("best_xray_core_end_seq_id")
            worst_core_start_raw = row.get("worst_xray_core_start_seq_id")
            worst_core_end_raw = row.get("worst_xray_core_end_seq_id")
            records.append(
                SolutionNMRMonomerXrayRmsdExtremesRecord(
                    entry_id=str(row["entry_id"]),
                    year=int(row["year"]),
                    sequence_identity_percent=int(row["sequence_identity_percent"]),
                    nmr_chain_id=str(row["nmr_chain_id"]),
                    nmr_core_start_seq_id=(
                        int(nmr_core_start_raw)
                        if nmr_core_start_raw not in {None, ""}
                        else None
                    ),
                    nmr_core_end_seq_id=(
                        int(nmr_core_end_raw)
                        if nmr_core_end_raw not in {None, ""}
                        else None
                    ),
                    nmr_query_sequence_length=int(
                        row.get("nmr_query_sequence_length") or 0
                    ),
                    xray_homolog_count=int(row.get("xray_homolog_count") or 0),
                    successful_xray_homolog_count=int(
                        row.get("successful_xray_homolog_count") or 0
                    ),
                    best_xray_homolog_entity_id=str(row["best_xray_homolog_entity_id"]),
                    best_xray_entry_id=str(row["best_xray_entry_id"]),
                    best_xray_chain_id=str(row["best_xray_chain_id"]),
                    best_xray_resolution_angstrom=float(
                        row["best_xray_resolution_angstrom"]
                    ),
                    best_xray_core_start_seq_id=(
                        int(best_core_start_raw)
                        if best_core_start_raw not in {None, ""}
                        else None
                    ),
                    best_xray_core_end_seq_id=(
                        int(best_core_end_raw)
                        if best_core_end_raw not in {None, ""}
                        else None
                    ),
                    best_n_common_ca=int(row["best_n_common_ca"]),
                    best_rmsd_ca_angstrom=float(row["best_rmsd_ca_angstrom"]),
                    worst_xray_homolog_entity_id=str(
                        row["worst_xray_homolog_entity_id"]
                    ),
                    worst_xray_entry_id=str(row["worst_xray_entry_id"]),
                    worst_xray_chain_id=str(row["worst_xray_chain_id"]),
                    worst_xray_resolution_angstrom=float(
                        row["worst_xray_resolution_angstrom"]
                    ),
                    worst_xray_core_start_seq_id=(
                        int(worst_core_start_raw)
                        if worst_core_start_raw not in {None, ""}
                        else None
                    ),
                    worst_xray_core_end_seq_id=(
                        int(worst_core_end_raw)
                        if worst_core_end_raw not in {None, ""}
                        else None
                    ),
                    worst_n_common_ca=int(row["worst_n_common_ca"]),
                    worst_rmsd_ca_angstrom=float(row["worst_rmsd_ca_angstrom"]),
                    rmsd_delta_angstrom=float(row["rmsd_delta_angstrom"]),
                )
            )
    return records


def write_solution_nmr_monomer_xray_rmsd_extremes_csv(
    records: list[SolutionNMRMonomerXrayRmsdExtremesRecord],
    output_path: Path,
) -> None:
    """Write NMR-to-X-ray RMSD extremes records to CSV."""
    write_csv_rows(
        output_path=output_path,
        header=list(SOLUTION_NMR_MONOMER_XRAY_RMSD_EXTREMES_HEADER),
        rows=(_solution_nmr_monomer_xray_rmsd_extremes_csv_row(r) for r in records),
    )
