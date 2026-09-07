"""Stream and resume ordinary, extreme, and historical RMSD outputs."""

from __future__ import annotations

import csv
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.dataset.builders.homologs import (
    SolutionNMRMonomerXrayHomologBuilder,
)
from src.dataset.builders.xray_rmsd import (
    SolutionNMRMonomerXrayRmsdBuilder,
    _build_xray_rmsd_extremes_record_from_candidates,
    _select_ordinary_xray_rmsd_record,
)
from src.dataset.config import (
    LOGGER,
)
from src.dataset.io.homologs import (
    read_solution_nmr_monomer_xray_homolog_csv,
)
from src.dataset.io.rmsd import (
    SOLUTION_NMR_MONOMER_XRAY_RMSD_EXTREMES_HEADER,
    SOLUTION_NMR_MONOMER_XRAY_RMSD_HEADER,
    _solution_nmr_monomer_xray_rmsd_csv_row,
    _solution_nmr_monomer_xray_rmsd_extremes_csv_row,
    read_solution_nmr_monomer_xray_rmsd_csv,
    read_solution_nmr_monomer_xray_rmsd_extremes_csv,
)
from src.dataset.reporting import (
    _import_filtered_structures,
    _record_filtered_structure,
    _set_active_dataset_filtered_csvs,
)

if TYPE_CHECKING:
    from src.dataset.client import (
        RCSBClient,
    )
    from src.dataset.config import (
        DatasetBuildConfig,
    )
    from src.dataset.records import (
        SolutionNMRMonomerXrayHomologRecord,
        SolutionNMRMonomerXrayRmsdExtremesRecord,
        SolutionNMRMonomerXrayRmsdRecord,
    )


def build_solution_nmr_monomer_xray_rmsd_to_csv(
    client: RCSBClient,
    config: DatasetBuildConfig,
    homolog_input_path: Path,
    output_path: Path,
    cache_dir: Path,
    rmsd_workers: int,
    sequence_identity_percent: int,
    resume: bool,
    log_label: str,
) -> None:
    """Collect best-match NMR-to-X-ray RMSD records and stream them to CSV."""
    _import_filtered_structures(homolog_input_path)
    existing_records: list[SolutionNMRMonomerXrayRmsdRecord] = []
    valid_existing_records: list[SolutionNMRMonomerXrayRmsdRecord] = []
    skip_entry_ids: set[str] = set()
    homolog_records = read_solution_nmr_monomer_xray_homolog_csv(homolog_input_path)
    if not homolog_records:
        raise SystemExit(
            f"No X-ray homolog records found for {log_label}. Run the matching "
            f"homolog dataset first or provide the expected CSV at {homolog_input_path}."
        )
    LOGGER.info(
        "%s %d%%: loaded %d homolog records from %s",
        log_label,
        sequence_identity_percent,
        len(homolog_records),
        homolog_input_path,
    )
    if resume and output_path.exists():
        existing_records = read_solution_nmr_monomer_xray_rmsd_csv(output_path)
        existing_records = [
            record
            for record in existing_records
            if record.sequence_identity_percent == sequence_identity_percent
        ]
        valid_existing_records = [
            record
            for record in existing_records
            if record.nmr_core_start_seq_id is not None
            and record.nmr_core_end_seq_id is not None
            and record.xray_core_start_seq_id is not None
            and record.xray_core_end_seq_id is not None
            and record.xray_homolog_entity_id
        ]
        dropped_existing = len(existing_records) - len(valid_existing_records)
        skip_entry_ids = {record.entry_id for record in valid_existing_records}
        LOGGER.info(
            "%s %d%%: loaded %d existing records for resume (outdated=%d)",
            log_label,
            sequence_identity_percent,
            len(valid_existing_records),
            dropped_existing,
        )

    rmsd_builder = SolutionNMRMonomerXrayRmsdBuilder(
        client=client,
        config=config,
        cache_dir=cache_dir,
        rmsd_workers=rmsd_workers,
        homolog_records=homolog_records,
        sequence_identity_percent=sequence_identity_percent,
    )
    valid_existing_records = sorted(
        valid_existing_records, key=lambda r: (r.year, r.entry_id)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(SOLUTION_NMR_MONOMER_XRAY_RMSD_HEADER)
        for record in valid_existing_records:
            writer.writerow(_solution_nmr_monomer_xray_rmsd_csv_row(record))
        csvfile.flush()

        def _on_xray_rmsd_record(record: SolutionNMRMonomerXrayRmsdRecord) -> None:
            """Persist one RMSD record while tracking processed seeds."""
            writer.writerow(_solution_nmr_monomer_xray_rmsd_csv_row(record))
            csvfile.flush()

        new_records = rmsd_builder.build(
            skip_entry_ids=skip_entry_ids,
            on_record=_on_xray_rmsd_record,
        )

    LOGGER.info(
        "Saved %d records to %s (new: %d, identity=%d%%)",
        len(valid_existing_records) + len(new_records),
        output_path,
        len(new_records),
        sequence_identity_percent,
    )


def build_solution_nmr_monomer_xray_rmsd_extremes_to_csv(
    client: RCSBClient,
    config: DatasetBuildConfig,
    homolog_input_path: Path,
    output_path: Path,
    cache_dir: Path,
    rmsd_workers: int,
    sequence_identity_percent: int,
    resume: bool,
    log_label: str,
) -> None:
    """Collect NMR-to-X-ray RMSD extremes and stream them to CSV."""
    _import_filtered_structures(homolog_input_path)
    existing_records: list[SolutionNMRMonomerXrayRmsdExtremesRecord] = []
    valid_existing_records: list[SolutionNMRMonomerXrayRmsdExtremesRecord] = []
    skip_entry_ids: set[str] = set()
    homolog_records = read_solution_nmr_monomer_xray_homolog_csv(homolog_input_path)
    if not homolog_records:
        raise SystemExit(
            f"No X-ray homolog records found for {log_label}. Run the matching "
            f"homolog dataset first or provide the expected CSV at {homolog_input_path}."
        )
    LOGGER.info(
        "%s %d%%: loaded %d homolog records from %s",
        log_label,
        sequence_identity_percent,
        len(homolog_records),
        homolog_input_path,
    )
    if resume and output_path.exists():
        existing_records = read_solution_nmr_monomer_xray_rmsd_extremes_csv(output_path)
        existing_records = [
            record
            for record in existing_records
            if record.sequence_identity_percent == sequence_identity_percent
        ]
        valid_existing_records = [
            record
            for record in existing_records
            if record.best_xray_homolog_entity_id
            and record.worst_xray_homolog_entity_id
        ]
        dropped_existing = len(existing_records) - len(valid_existing_records)
        skip_entry_ids = {record.entry_id for record in valid_existing_records}
        LOGGER.info(
            "%s %d%%: loaded %d existing records for resume (outdated=%d)",
            log_label,
            sequence_identity_percent,
            len(valid_existing_records),
            dropped_existing,
        )

    rmsd_builder = SolutionNMRMonomerXrayRmsdBuilder(
        client=client,
        config=config,
        cache_dir=cache_dir,
        rmsd_workers=rmsd_workers,
        homolog_records=homolog_records,
        sequence_identity_percent=sequence_identity_percent,
    )
    valid_existing_records = sorted(
        valid_existing_records, key=lambda r: (r.year, r.entry_id)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(SOLUTION_NMR_MONOMER_XRAY_RMSD_EXTREMES_HEADER)
        for record in valid_existing_records:
            writer.writerow(_solution_nmr_monomer_xray_rmsd_extremes_csv_row(record))
        csvfile.flush()

        def _on_xray_rmsd_extremes_record(
            record: SolutionNMRMonomerXrayRmsdExtremesRecord,
        ) -> None:
            """Persist one RMSD extremes record while tracking processed seeds."""
            writer.writerow(_solution_nmr_monomer_xray_rmsd_extremes_csv_row(record))
            csvfile.flush()

        new_records = rmsd_builder.build_extremes(
            skip_entry_ids=skip_entry_ids,
            on_record=_on_xray_rmsd_extremes_record,
        )

    LOGGER.info(
        "Saved %d records to %s (new: %d, identity=%d%%)",
        len(valid_existing_records) + len(new_records),
        output_path,
        len(new_records),
        sequence_identity_percent,
    )


def build_solution_nmr_monomer_xray_rmsd_outputs_to_csv(
    *,
    client: RCSBClient,
    config: DatasetBuildConfig,
    current_homolog_input_path: Path | None,
    historical_homolog_input_path: Path | None,
    ordinary_output_path: Path | None,
    historical_ordinary_output_path: Path | None,
    extremes_output_path: Path | None,
    historical_extremes_output_path: Path | None,
    cache_dir: Path,
    rmsd_workers: int,
    sequence_identity_percent: int,
    resume: bool = False,
) -> None:
    """Build selected ordinary/extremes/current/historical CSVs in one pass."""
    output_specs: dict[str, tuple[Path, str, bool]] = {}
    if ordinary_output_path is not None:
        output_specs["ordinary_current"] = (
            Path(ordinary_output_path),
            "current",
            False,
        )
    if historical_ordinary_output_path is not None:
        output_specs["ordinary_historical"] = (
            Path(historical_ordinary_output_path),
            "historical",
            False,
        )
    if extremes_output_path is not None:
        output_specs["extremes_current"] = (
            Path(extremes_output_path),
            "current",
            True,
        )
    if historical_extremes_output_path is not None:
        output_specs["extremes_historical"] = (
            Path(historical_extremes_output_path),
            "historical",
            True,
        )
    if not output_specs:
        return

    output_paths_by_view = {
        view: tuple(
            path for path, spec_view, _ in output_specs.values() if spec_view == view
        )
        for view in ("current", "historical")
    }
    homolog_maps: dict[str, dict[str, SolutionNMRMonomerXrayHomologRecord]] = {}
    for view, input_path in (
        ("current", current_homolog_input_path),
        ("historical", historical_homolog_input_path),
    ):
        if not output_paths_by_view[view]:
            continue
        if input_path is None:
            raise SystemExit(f"Missing {view} X-ray homolog CSV for RMSD outputs")
        _set_active_dataset_filtered_csvs(output_paths_by_view[view])
        _import_filtered_structures(input_path)
        records = [
            record
            for record in read_solution_nmr_monomer_xray_homolog_csv(input_path)
            if record.sequence_identity_percent == sequence_identity_percent
        ]
        if not records:
            raise SystemExit(
                f"No {view} X-ray homolog records at {sequence_identity_percent}% "
                f"identity in {input_path}. Regenerate the homolog CSV first."
            )
        homolog_maps[view] = {record.entry_id: record for record in records}
        LOGGER.info(
            "SOLUTION NMR unified X-ray RMSD %s %d%%: loaded %d homolog records from %s",
            view,
            sequence_identity_percent,
            len(records),
            input_path,
        )

    if "current" in homolog_maps and "historical" in homolog_maps:
        current_by_entry_id = homolog_maps["current"]
        historical_by_entry_id = homolog_maps["historical"]
        unexpected_historical_ids = set(historical_by_entry_id) - set(
            current_by_entry_id
        )
        if unexpected_historical_ids:
            first_entry_id = min(unexpected_historical_ids)
            raise SystemExit(
                "Historical homolog CSV is inconsistent with the current CSV: "
                f"{first_entry_id} is absent from current records. Regenerate both "
                "homolog CSVs together."
            )
        for entry_id, historical in historical_by_entry_id.items():
            current = current_by_entry_id[entry_id]
            current_metadata = (
                current.year,
                current.sequence_identity_percent,
                current.nmr_core_start_seq_id,
                current.nmr_core_end_seq_id,
                current.nmr_query_sequence_length,
            )
            historical_metadata = (
                historical.year,
                historical.sequence_identity_percent,
                historical.nmr_core_start_seq_id,
                historical.nmr_core_end_seq_id,
                historical.nmr_query_sequence_length,
            )
            if current_metadata != historical_metadata or not set(
                historical.xray_homolog_entity_ids
            ).issubset(current.xray_homolog_entity_ids):
                raise SystemExit(
                    "Historical homolog CSV is stale or inconsistent for "
                    f"{entry_id}. Regenerate current and historical homolog CSVs "
                    "together before RMSD calculation."
                )

    existing_records_by_output: dict[str, list[Any]] = {}
    target_entry_ids_by_output: dict[str, set[str]] = {}
    for output_name, (output_path, view, is_extremes) in output_specs.items():
        existing: list[Any] = []
        if resume and output_path.exists():
            loaded: list[Any]
            if is_extremes:
                loaded = read_solution_nmr_monomer_xray_rmsd_extremes_csv(output_path)
                existing = [
                    record
                    for record in loaded
                    if record.sequence_identity_percent == sequence_identity_percent
                    and record.best_xray_homolog_entity_id
                    and record.worst_xray_homolog_entity_id
                ]
            else:
                loaded = read_solution_nmr_monomer_xray_rmsd_csv(output_path)
                existing = [
                    record
                    for record in loaded
                    if record.sequence_identity_percent == sequence_identity_percent
                    and record.nmr_core_start_seq_id is not None
                    and record.nmr_core_end_seq_id is not None
                    and record.xray_core_start_seq_id is not None
                    and record.xray_core_end_seq_id is not None
                    and record.xray_homolog_entity_id
                ]
        existing_by_entry_id = {record.entry_id: record for record in existing}
        existing = sorted(
            existing_by_entry_id.values(),
            key=lambda record: (record.year, record.entry_id),
        )
        existing_records_by_output[output_name] = existing
        existing_entry_ids = set(existing_by_entry_id)
        eligible_missing = sorted(
            (
                record
                for record in homolog_maps[view].values()
                if record.entry_id not in existing_entry_ids
                and record.nmr_core_start_seq_id is not None
                and record.nmr_core_end_seq_id is not None
                and record.xray_homolog_entity_ids
            ),
            key=lambda record: (record.year, record.entry_id),
        )
        target_entry_ids_by_output[output_name] = {
            record.entry_id for record in eligible_missing
        }
        LOGGER.info(
            "SOLUTION NMR unified X-ray RMSD %s: mode=%s, retained rows=%d, entries to process=%d",
            output_name,
            "resume" if resume else "rebuild",
            len(existing),
            len(eligible_missing),
        )

    work_entry_ids = set().union(*target_entry_ids_by_output.values())
    merged_homolog_records: list[SolutionNMRMonomerXrayHomologRecord] = []
    for entry_id in sorted(work_entry_ids):
        needed_views = {
            output_specs[output_name][1]
            for output_name, target_ids in target_entry_ids_by_output.items()
            if entry_id in target_ids
        }
        base_view = "current" if "current" in needed_views else "historical"
        base_record = homolog_maps[base_view][entry_id]
        entity_ids = tuple(
            dict.fromkeys(
                entity_id
                for view in ("current", "historical")
                if view in needed_views
                for entity_id in homolog_maps[view][entry_id].xray_homolog_entity_ids
            )
        )
        merged_homolog_records.append(
            replace(
                base_record,
                xray_homolog_entry_ids=(
                    SolutionNMRMonomerXrayHomologBuilder._entry_ids_from_polymer_entity_ids(
                        entity_ids
                    )
                ),
                xray_homolog_entity_ids=entity_ids,
                has_xray_homolog=bool(entity_ids),
            )
        )

    _set_active_dataset_filtered_csvs(
        tuple(path for path, _, _ in output_specs.values())
    )
    new_record_counts = {output_name: 0 for output_name in output_specs}
    with ExitStack() as stack:
        csvfiles: dict[str, Any] = {}
        writers: dict[str, Any] = {}
        for output_name, (output_path, _, is_extremes) in output_specs.items():
            output_path.parent.mkdir(parents=True, exist_ok=True)
            csvfile = stack.enter_context(
                output_path.open("w", newline="", encoding="utf-8")
            )
            writer = csv.writer(csvfile)
            writer.writerow(
                SOLUTION_NMR_MONOMER_XRAY_RMSD_EXTREMES_HEADER
                if is_extremes
                else SOLUTION_NMR_MONOMER_XRAY_RMSD_HEADER
            )
            for record in existing_records_by_output[output_name]:
                writer.writerow(
                    _solution_nmr_monomer_xray_rmsd_extremes_csv_row(record)
                    if is_extremes
                    else _solution_nmr_monomer_xray_rmsd_csv_row(record)
                )
            csvfile.flush()
            csvfiles[output_name] = csvfile
            writers[output_name] = writer

        def _on_candidate_set(
            computed_homolog: SolutionNMRMonomerXrayHomologRecord,
            candidate_records: tuple[SolutionNMRMonomerXrayRmsdRecord, ...],
        ) -> None:
            """Project one shared pair set and checkpoint every missing output."""
            entry_id = computed_homolog.entry_id
            for output_name, (_, view, is_extremes) in output_specs.items():
                if entry_id not in target_entry_ids_by_output[output_name]:
                    continue
                view_homolog = homolog_maps[view][entry_id]
                if is_extremes:
                    record = _build_xray_rmsd_extremes_record_from_candidates(
                        homolog=view_homolog,
                        candidate_records=candidate_records,
                    )
                else:
                    record = _select_ordinary_xray_rmsd_record(
                        homolog=view_homolog,
                        candidate_records=candidate_records,
                    )
                if record is None:
                    _record_filtered_structure(
                        entry_id,
                        f"no successful X-ray RMSD candidate in the {view} homolog view",
                        year=view_homolog.year,
                    )
                    continue
                writers[output_name].writerow(
                    _solution_nmr_monomer_xray_rmsd_extremes_csv_row(record)
                    if is_extremes
                    else _solution_nmr_monomer_xray_rmsd_csv_row(record)
                )
                csvfiles[output_name].flush()
                new_record_counts[output_name] += 1

        if merged_homolog_records:
            rmsd_builder = SolutionNMRMonomerXrayRmsdBuilder(
                client=client,
                config=config,
                cache_dir=cache_dir,
                rmsd_workers=rmsd_workers,
                homolog_records=merged_homolog_records,
                sequence_identity_percent=sequence_identity_percent,
            )
            rmsd_builder.build_candidate_sets(on_candidate_set=_on_candidate_set)

    for output_name, (output_path, _, _) in output_specs.items():
        LOGGER.info(
            "Saved %d records to %s (new: %d, unified identity=%d%%)",
            len(existing_records_by_output[output_name])
            + new_record_counts[output_name],
            output_path,
            new_record_counts[output_name],
            sequence_identity_percent,
        )
