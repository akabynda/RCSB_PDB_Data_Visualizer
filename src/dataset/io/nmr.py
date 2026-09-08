"""CSV schemas and serialization for counts and NMR metrics."""

from __future__ import annotations

import csv
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.dataset.io.common import (
    write_csv_rows,
)
from src.dataset.records import (
    parse_residue_id,
    SolutionNMRMonomerPrecisionRecord,
    SolutionNMRMonomerQualityRecord,
)

if TYPE_CHECKING:
    from src.dataset.records import (
        MembraneYearlyCountRecord,
        SolutionNMRMonomerExperimentsRecord,
        SolutionNMRMonomerStrideModeledFirstModelRecord,
        SolutionNMRWeightRecord,
        YearlyCountRecord,
    )


def write_method_counts_csv(
    records: list[YearlyCountRecord], output_path: Path
) -> None:
    """Write yearly experimental-method counts to CSV."""
    write_csv_rows(
        output_path=output_path,
        header=["year", "method", "count"],
        rows=((r.year, r.method, r.count) for r in records),
    )


def read_solution_nmr_monomer_quality_csv(
    input_path: Path,
) -> list[SolutionNMRMonomerQualityRecord]:
    """Read SOLUTION NMR monomer quality records from CSV."""
    if not input_path.exists():
        return []
    records: list[SolutionNMRMonomerQualityRecord] = []
    with input_path.open("r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if not row:
                continue
            records.append(
                SolutionNMRMonomerQualityRecord(
                    entry_id=str(row["entry_id"]),
                    year=int(row["year"]),
                    clashscore=float(row["clashscore"]),
                    ramachandran_outliers_percent=float(
                        row["ramachandran_outliers_percent"]
                    ),
                    sidechain_outliers_percent=float(row["sidechain_outliers_percent"]),
                )
            )
    return records


def write_membrane_counts_csv(
    records: list[MembraneYearlyCountRecord], output_path: Path
) -> None:
    """Write membrane-protein yearly count records to CSV."""
    write_csv_rows(
        output_path=output_path,
        header=["year", "count"],
        rows=((r.year, r.count) for r in records),
    )


def write_solution_nmr_weights_csv(
    records: list[SolutionNMRWeightRecord], output_path: Path
) -> None:
    """Write SOLUTION NMR molecular-weight records to CSV."""
    write_csv_rows(
        output_path=output_path,
        header=["entry_id", "year", "molecular_weight_kda"],
        rows=(
            (
                r.entry_id,
                r.year,
                f"{r.molecular_weight_kda:.3f}",
            )
            for r in records
        ),
    )


def write_solution_nmr_monomer_experiments_csv(
    records: list[SolutionNMRMonomerExperimentsRecord], output_path: Path
) -> None:
    """Write one combined NMR-experiment field per monomeric entry."""
    write_csv_rows(
        output_path=output_path,
        header=["entry_id", "year", "nmr_experiments_conducted"],
        rows=(
            (
                record.entry_id,
                record.year,
                "; ".join(record.nmr_experiments_conducted),
            )
            for record in records
        ),
    )


def stream_solution_nmr_monomer_stride_modeled_first_model_csv(
    records: Iterator[SolutionNMRMonomerStrideModeledFirstModelRecord],
    output_path: Path,
) -> int:
    """Stream STRIDE modeled-first-model records directly to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "entry_id",
                "year",
                "chain_id",
                "modeled_start_seq_id",
                "modeled_end_seq_id",
                "modeled_sequence_length",
                "stride_alpha_helix_fraction",
                "stride_3_10_helix_fraction",
                "stride_pi_helix_fraction",
                "stride_beta_strand_fraction",
                "stride_isolated_beta_bridge_fraction",
                "stride_turn_fraction",
                "stride_coil_fraction",
                "stride_secondary_structure_percent",
            ]
        )
        csvfile.flush()
        for record in records:
            writer.writerow(
                (
                    record.entry_id,
                    record.year,
                    record.chain_id,
                    record.modeled_start_seq_id,
                    record.modeled_end_seq_id,
                    record.modeled_sequence_length,
                    f"{record.stride_alpha_helix_fraction:.6f}",
                    f"{record.stride_3_10_helix_fraction:.6f}",
                    f"{record.stride_pi_helix_fraction:.6f}",
                    f"{record.stride_beta_strand_fraction:.6f}",
                    f"{record.stride_isolated_beta_bridge_fraction:.6f}",
                    f"{record.stride_turn_fraction:.6f}",
                    f"{record.stride_coil_fraction:.6f}",
                    f"{record.stride_secondary_structure_percent:.3f}",
                )
            )
            csvfile.flush()
            count += 1
    return count


def read_solution_nmr_monomer_precision_csv(
    input_path: Path,
) -> list[SolutionNMRMonomerPrecisionRecord]:
    """Read SOLUTION NMR monomer precision records from CSV."""
    if not input_path.exists():
        return []
    records: list[SolutionNMRMonomerPrecisionRecord] = []
    with input_path.open("r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if not row:
                continue
            n_ca_core_used_raw = row.get("n_ca_core_used")
            if n_ca_core_used_raw in {None, ""}:
                n_ca_core_used_raw = row.get("n_ca_core")
            if n_ca_core_used_raw in {None, ""}:
                continue
            n_ca_core_raw_raw = row.get("n_ca_core_raw")
            if n_ca_core_raw_raw in {None, ""}:
                n_ca_core_raw_raw = n_ca_core_used_raw
            records.append(
                SolutionNMRMonomerPrecisionRecord(
                    entry_id=str(row["entry_id"]),
                    year=int(row["year"]),
                    chain_id=str(row["chain_id"]),
                    core_start_seq_id=parse_residue_id(row["core_start_seq_id"]),
                    core_end_seq_id=parse_residue_id(row["core_end_seq_id"]),
                    n_models=int(row["n_models"]),
                    n_ca_core_used=int(str(n_ca_core_used_raw)),
                    n_ca_core_raw=int(str(n_ca_core_raw_raw)),
                    mean_rmsd_angstrom=float(row["mean_rmsd_angstrom"]),
                )
            )
    return records


SOLUTION_NMR_MONOMER_PRECISION_HEADER: tuple[str, ...] = (
    "entry_id",
    "year",
    "chain_id",
    "core_start_seq_id",
    "core_end_seq_id",
    "n_models",
    "n_ca_core_used",
    "n_ca_core_raw",
    "mean_rmsd_angstrom",
)


def _solution_nmr_monomer_precision_csv_row(
    record: SolutionNMRMonomerPrecisionRecord,
) -> tuple[Any, ...]:
    """Convert a precision record into a CSV row dictionary."""
    return (
        record.entry_id,
        record.year,
        record.chain_id,
        record.core_start_seq_id,
        record.core_end_seq_id,
        record.n_models,
        record.n_ca_core_used,
        record.n_ca_core_raw,
        f"{record.mean_rmsd_angstrom:.6f}",
    )


def write_solution_nmr_monomer_quality_csv(
    records: list[SolutionNMRMonomerQualityRecord], output_path: Path
) -> None:
    """Write SOLUTION NMR monomer quality records to CSV."""
    write_csv_rows(
        output_path=output_path,
        header=[
            "entry_id",
            "year",
            "clashscore",
            "ramachandran_outliers_percent",
            "sidechain_outliers_percent",
        ],
        rows=(
            (
                r.entry_id,
                r.year,
                f"{r.clashscore:.4f}",
                f"{r.ramachandran_outliers_percent:.4f}",
                f"{r.sidechain_outliers_percent:.4f}",
            )
            for r in records
        ),
    )
