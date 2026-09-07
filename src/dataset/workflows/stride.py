"""Dataset collection workflows for stride."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.dataset.builders.precision import (
    SolutionNMRMonomerPrecisionStrideModeledFirstModelBuilder,
)
from src.dataset.builders.stride import SolutionNMRMonomerStrideModeledFirstModelBuilder
from src.dataset.client import RCSBClient
from src.dataset.config import (
    LOGGER,
    DatasetBuildConfig,
)
from src.dataset.io.nmr import (
    SOLUTION_NMR_MONOMER_PRECISION_HEADER,
    _solution_nmr_monomer_precision_csv_row,
    read_solution_nmr_monomer_precision_csv,
    stream_solution_nmr_monomer_stride_modeled_first_model_csv,
)
from src.dataset.records import SolutionNMRMonomerPrecisionRecord
from src.dataset.stride_install import ensure_stride_executable


def run_stride(
    args: argparse.Namespace, client: RCSBClient, config: DatasetBuildConfig
) -> None:
    """Stream modeled first-model STRIDE records to CSV."""
    modeled_first_stride_executable = ensure_stride_executable(
        args.solution_nmr_monomer_stride_executable,
        Path(args.stride_install_dir),
    )
    if modeled_first_stride_executable is None:
        raise RuntimeError(
            "STRIDE executable not found for solution_nmr_monomer_stride_modeled_first_model. "
            "Provide a valid --solution-nmr-monomer-stride-executable path."
        )
    LOGGER.info(
        ("SOLUTION NMR monomer-stride-modeled-first-model: using STRIDE executable %s"),
        modeled_first_stride_executable,
    )
    modeled_stride_builder = SolutionNMRMonomerStrideModeledFirstModelBuilder(
        client=client,
        config=config,
        stride_executable=modeled_first_stride_executable,
        cache_dir=Path(args.solution_nmr_monomer_cache_dir),
        stride_cache_dir=Path(args.stride_cache_dir),
    )
    modeled_stride_count = stream_solution_nmr_monomer_stride_modeled_first_model_csv(
        records=modeled_stride_builder.iter_records(),
        output_path=Path(args.solution_nmr_monomer_stride_modeled_first_model_output),
    )
    LOGGER.info(
        "Saved %d records to %s",
        modeled_stride_count,
        args.solution_nmr_monomer_stride_modeled_first_model_output,
    )


def run_precision(
    args: argparse.Namespace, client: RCSBClient, config: DatasetBuildConfig
) -> None:
    """Build STRIDE-core precision records with optional resume."""
    precision_stride_executable = ensure_stride_executable(
        args.solution_nmr_monomer_stride_executable,
        Path(args.stride_install_dir),
    )
    if precision_stride_executable is None:
        raise RuntimeError(
            "STRIDE executable not found for "
            "solution_nmr_monomer_precision_stride_modeled_first_model. "
            "Provide a valid --solution-nmr-monomer-stride-executable path."
        )

    existing_records = []
    skip_entry_ids = set()
    precision_stride_output_path = Path(
        args.solution_nmr_monomer_precision_stride_modeled_first_model_output
    )
    if args.resume and precision_stride_output_path.exists():
        existing_records = read_solution_nmr_monomer_precision_csv(
            precision_stride_output_path
        )
        skip_entry_ids = {record.entry_id for record in existing_records}
        LOGGER.info(
            "SOLUTION NMR precision STRIDE modeled-first-model: loaded %d existing records for resume",
            len(existing_records),
        )

    precision_stride_builder = (
        SolutionNMRMonomerPrecisionStrideModeledFirstModelBuilder(
            client=client,
            config=config,
            cache_dir=Path(args.precision_cache_dir),
            precision_workers=args.precision_workers,
            stride_executable=precision_stride_executable,
            stride_cache_dir=Path(args.stride_cache_dir),
        )
    )
    existing_records = sorted(existing_records, key=lambda r: (r.year, r.entry_id))
    precision_stride_output_path.parent.mkdir(parents=True, exist_ok=True)
    with precision_stride_output_path.open(
        "w", newline="", encoding="utf-8"
    ) as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(SOLUTION_NMR_MONOMER_PRECISION_HEADER)
        for record in existing_records:
            writer.writerow(_solution_nmr_monomer_precision_csv_row(record))
        csvfile.flush()

        def _on_precision_stride_record(
            record: SolutionNMRMonomerPrecisionRecord,
        ) -> None:
            """Persist one STRIDE-core precision record while tracking processed seeds."""
            writer.writerow(_solution_nmr_monomer_precision_csv_row(record))
            csvfile.flush()

        new_records = precision_stride_builder.build(
            skip_entry_ids=skip_entry_ids,
            on_record=_on_precision_stride_record,
        )

    LOGGER.info(
        "Saved %d records to %s (new: %d)",
        len(existing_records) + len(new_records),
        args.solution_nmr_monomer_precision_stride_modeled_first_model_output,
        len(new_records),
    )
