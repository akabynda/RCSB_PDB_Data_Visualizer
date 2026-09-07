"""Dataset collection workflows for homologs."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from src.dataset.builders.homologs import SolutionNMRMonomerXrayHomologBuilder
from src.dataset.client import RCSBClient
from src.dataset.config import (
    LOGGER,
    DatasetBuildConfig,
)
from src.dataset.io.homologs import (
    SOLUTION_NMR_MONOMER_XRAY_HOMOLOG_HEADER,
    _rejected_xray_homolog_csv_row,
    _solution_nmr_monomer_xray_homolog_csv_row,
    _xray_homolog_resume_checkpoint_path,
    rejected_xray_homologs_csv_path,
    write_rejected_xray_homolog_csv,
)
from src.dataset.records import (
    RejectedXrayHomologRecord,
    SolutionNMRMonomerXrayHomologRecord,
)
from src.dataset.stride_install import ensure_stride_executable
from src.dataset.workflows.homolog_resume import (
    HomologOutputPaths,
    HomologResumeState,
    load_homolog_resume_state,
)


def run_homologs(
    args: argparse.Namespace, client: RCSBClient, config: DatasetBuildConfig
) -> None:
    """Build and checkpoint current X-ray homologs and rejection audits."""
    homolog_stride_executable = ensure_stride_executable(
        args.solution_nmr_monomer_stride_executable,
        Path(args.stride_install_dir),
    )
    if homolog_stride_executable is None:
        raise SystemExit(
            "STRIDE executable not found for solution_nmr_monomer_xray_homologs. "
            "Provide a valid --solution-nmr-monomer-stride-executable path."
        )
    LOGGER.info(
        "SOLUTION NMR monomer X-ray homologs: using STRIDE executable %s",
        homolog_stride_executable,
    )
    homolog_builder = SolutionNMRMonomerXrayHomologBuilder(
        client=client,
        config=config,
        stride_executable=homolog_stride_executable,
        cache_dir=Path(args.solution_nmr_monomer_cache_dir),
        stride_cache_dir=Path(args.stride_cache_dir),
    )
    homolog_95_output_path = Path(args.solution_nmr_monomer_xray_homolog_95_output)
    homolog_100_output_path = Path(args.solution_nmr_monomer_xray_homolog_100_output)
    paths = HomologOutputPaths(
        homolog_95=homolog_95_output_path,
        homolog_100=homolog_100_output_path,
        rejected_95=rejected_xray_homologs_csv_path(homolog_95_output_path),
        rejected_100=rejected_xray_homologs_csv_path(homolog_100_output_path),
        checkpoint=_xray_homolog_resume_checkpoint_path(homolog_95_output_path),
    )
    for output_path in (
        paths.homolog_95,
        paths.homolog_100,
        paths.rejected_95,
        paths.rejected_100,
    ):
        output_path.parent.mkdir(parents=True, exist_ok=True)
    state = load_homolog_resume_state(args.resume, paths)
    _stream_homolog_records(homolog_builder, paths, state)


def _persist_rejected_xray_homologs(
    records: Iterable[RejectedXrayHomologRecord],
    writer: Any,
    output_file: Any,
    known_keys: set[tuple[str, int, str]],
) -> None:
    """Append new candidate rejections before checkpointing the seed."""
    wrote_row = False
    for rejected_record in records:
        key = (
            rejected_record.nmr_entry_id,
            rejected_record.sequence_identity_percent,
            rejected_record.xray_entity_id,
        )
        if key in known_keys:
            continue
        writer.writerow(_rejected_xray_homolog_csv_row(rejected_record))
        known_keys.add(key)
        wrote_row = True
    if wrote_row:
        output_file.flush()


def _stream_homolog_records(
    homolog_builder: SolutionNMRMonomerXrayHomologBuilder,
    paths: HomologOutputPaths,
    state: HomologResumeState,
) -> None:
    """Persist paired records, rejection audits, and checkpoints as seeds finish."""
    rejected_by_key_95 = {
        (
            record.nmr_entry_id,
            record.sequence_identity_percent,
            record.xray_entity_id,
        ): record
        for record in state.existing_rejected_95
    }
    rejected_by_key_100 = {
        (
            record.nmr_entry_id,
            record.sequence_identity_percent,
            record.xray_entity_id,
        ): record
        for record in state.existing_rejected_100
    }
    write_rejected_xray_homolog_csv(rejected_by_key_95.values(), paths.rejected_95)
    write_rejected_xray_homolog_csv(rejected_by_key_100.values(), paths.rejected_100)

    with (
        paths.homolog_95.open("w", newline="", encoding="utf-8") as file_95,
        paths.homolog_100.open("w", newline="", encoding="utf-8") as file_100,
        paths.rejected_95.open("a", newline="", encoding="utf-8") as rejected_file_95,
        paths.rejected_100.open("a", newline="", encoding="utf-8") as rejected_file_100,
        paths.checkpoint.open("a", encoding="utf-8") as checkpoint_file,
    ):
        writer_95 = csv.writer(file_95)
        writer_100 = csv.writer(file_100)
        rejected_writer_95 = csv.writer(rejected_file_95)
        rejected_writer_100 = csv.writer(rejected_file_100)
        writer_95.writerow(SOLUTION_NMR_MONOMER_XRAY_HOMOLOG_HEADER)
        writer_100.writerow(SOLUTION_NMR_MONOMER_XRAY_HOMOLOG_HEADER)
        for record in state.existing_records_95:
            writer_95.writerow(_solution_nmr_monomer_xray_homolog_csv_row(record))
        for record in state.existing_records_100:
            writer_100.writerow(_solution_nmr_monomer_xray_homolog_csv_row(record))
        rejected_keys_95 = set(rejected_by_key_95)
        rejected_keys_100 = set(rejected_by_key_100)
        file_95.flush()
        file_100.flush()

        def _on_homolog_record_pair(
            record_95: SolutionNMRMonomerXrayHomologRecord,
            record_100: SolutionNMRMonomerXrayHomologRecord,
        ) -> None:
            """Persist homolog and rejected-candidate records for one seed."""
            _persist_rejected_xray_homologs(
                record_95.rejected_xray_homologs,
                rejected_writer_95,
                rejected_file_95,
                rejected_keys_95,
            )
            _persist_rejected_xray_homologs(
                record_100.rejected_xray_homologs,
                rejected_writer_100,
                rejected_file_100,
                rejected_keys_100,
            )
            writer_95.writerow(_solution_nmr_monomer_xray_homolog_csv_row(record_95))
            writer_100.writerow(_solution_nmr_monomer_xray_homolog_csv_row(record_100))
            file_95.flush()
            file_100.flush()

        def _on_homolog_entry_complete(entry_id: str, status: str) -> None:
            """Checkpoint successes and intentional exclusions, not failures."""
            checkpoint_status = (
                "success_with_rejected_audit" if status == "success" else status
            )
            checkpoint_file.write(f"{entry_id}\t{checkpoint_status}\n")
            checkpoint_file.flush()

        new_records_95, new_records_100 = homolog_builder.build(
            on_record_pair=_on_homolog_record_pair,
            skip_entry_ids=state.skip_homolog_entry_ids,
            on_entry_complete=_on_homolog_entry_complete,
            retain_rejected_xray_homologs=False,
        )
    records_95 = state.existing_records_95 + new_records_95
    records_100 = state.existing_records_100 + new_records_100
    LOGGER.info(
        "Saved %d records to %s",
        len(records_95),
        paths.homolog_95,
    )
    LOGGER.info(
        "Saved %d records to %s",
        len(records_100),
        paths.homolog_100,
    )
    LOGGER.info(
        "Saved %d rejected X-ray candidates to %s",
        len(rejected_keys_95),
        paths.rejected_95,
    )
    LOGGER.info(
        "Saved %d rejected X-ray candidates to %s",
        len(rejected_keys_100),
        paths.rejected_100,
    )
