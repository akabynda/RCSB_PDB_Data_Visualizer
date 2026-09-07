"""Filter X-ray homologs by deposition and release dates."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.dataset.builders.homologs import (
    SolutionNMRMonomerXrayHomologBuilder,
)
from src.dataset.config import (
    LOGGER,
)
from src.dataset.records import (
    SolutionNMRMonomerXrayHomologRecord,
)
from src.dataset.reporting import (
    _record_filtered_structure,
)
from src.dataset.utils import (
    chunked,
    collect_batch_results,
    parse_rcsb_datetime,
)

if TYPE_CHECKING:
    from src.dataset.client import (
        RCSBClient,
    )
    from src.dataset.config import (
        DatasetBuildConfig,
    )


def filter_xray_homolog_records_by_deposit_date(
    records: list[SolutionNMRMonomerXrayHomologRecord],
    client: RCSBClient,
    config: DatasetBuildConfig,
) -> list[SolutionNMRMonomerXrayHomologRecord]:
    """Keep homolog records whose X-ray release timing matches the mode."""
    eligible_records: list[SolutionNMRMonomerXrayHomologRecord] = []
    for record in records:
        if record.nmr_query_sequence_length <= 10:
            _record_filtered_structure(
                record.entry_id,
                f"NMR query sequence is too short ({record.nmr_query_sequence_length} residues)",
                year=record.year,
            )
            continue
        if record.nmr_core_start_seq_id is None or record.nmr_core_end_seq_id is None:
            _record_filtered_structure(
                record.entry_id, "NMR core range is missing", year=record.year
            )
            continue
        eligible_records.append(record)
    records = eligible_records
    xray_entity_ids = sorted(
        {
            entity_id
            for record in records
            for entity_id in record.xray_homolog_entity_ids
        }
    )
    if not xray_entity_ids:
        return [
            SolutionNMRMonomerXrayHomologRecord(
                entry_id=record.entry_id,
                year=record.year,
                sequence_identity_percent=record.sequence_identity_percent,
                nmr_core_start_seq_id=record.nmr_core_start_seq_id,
                nmr_core_end_seq_id=record.nmr_core_end_seq_id,
                nmr_query_sequence_length=record.nmr_query_sequence_length,
                xray_homolog_entry_ids=tuple(),
                xray_homolog_entity_ids=tuple(),
                has_xray_homolog=False,
            )
            for record in records
        ]

    xray_entry_id_by_entity_id = {
        entity_id: str(entity_id).split("_", 1)[0].strip()
        for entity_id in xray_entity_ids
    }
    entry_ids = sorted(
        {record.entry_id for record in records}
        | {entry_id for entry_id in xray_entry_id_by_entity_id.values() if entry_id}
    )
    accession_dates_by_entry_id: dict[str, tuple[str | None, str | None]] = {}
    for batch_dates in collect_batch_results(
        batches=list(chunked(entry_ids, config.graphql_batch_size)),
        max_workers=config.max_workers,
        fetch_fn=client.fetch_accession_dates_by_entry_id_for_ids,
        progress_label="Historical homolog accession dates",
    ):
        accession_dates_by_entry_id.update(batch_dates)

    historical_records: list[SolutionNMRMonomerXrayHomologRecord] = []
    for record in records:
        if not record.xray_homolog_entity_ids:
            historical_records.append(
                SolutionNMRMonomerXrayHomologRecord(
                    entry_id=record.entry_id,
                    year=record.year,
                    sequence_identity_percent=record.sequence_identity_percent,
                    nmr_core_start_seq_id=record.nmr_core_start_seq_id,
                    nmr_core_end_seq_id=record.nmr_core_end_seq_id,
                    nmr_query_sequence_length=record.nmr_query_sequence_length,
                    xray_homolog_entry_ids=tuple(),
                    xray_homolog_entity_ids=tuple(),
                    has_xray_homolog=False,
                )
            )
            continue
        nmr_deposit_date = parse_rcsb_datetime(
            (accession_dates_by_entry_id.get(record.entry_id) or (None, None))[0]
        )
        if nmr_deposit_date is None:
            LOGGER.warning(
                "Excluding NMR entry %s from historical homology: missing NMR deposit date",
                record.entry_id,
            )
            _record_filtered_structure(
                record.entry_id, "NMR deposit date is missing", year=record.year
            )
            continue
        kept_entity_ids_list: list[str] = []
        dates_complete = True
        for entity_id in record.xray_homolog_entity_ids:
            xray_entry_id = xray_entry_id_by_entity_id.get(entity_id)
            if not xray_entry_id:
                dates_complete = False
                break
            xray_release_date = (
                accession_dates_by_entry_id.get(xray_entry_id) or (None, None)
            )[1]
            parsed_xray_release_date = parse_rcsb_datetime(xray_release_date)
            if parsed_xray_release_date is None:
                dates_complete = False
                break
            if parsed_xray_release_date <= nmr_deposit_date:
                kept_entity_ids_list.append(entity_id)
        if not dates_complete:
            LOGGER.warning(
                "Excluding NMR entry %s from historical homology: incomplete X-ray release dates",
                record.entry_id,
            )
            _record_filtered_structure(
                record.entry_id,
                "one or more X-ray release dates are missing",
                year=record.year,
            )
            continue
        kept_entity_ids = tuple(kept_entity_ids_list)
        kept_entry_ids = (
            SolutionNMRMonomerXrayHomologBuilder._entry_ids_from_polymer_entity_ids(
                kept_entity_ids
            )
        )
        historical_records.append(
            SolutionNMRMonomerXrayHomologRecord(
                entry_id=record.entry_id,
                year=record.year,
                sequence_identity_percent=record.sequence_identity_percent,
                nmr_core_start_seq_id=record.nmr_core_start_seq_id,
                nmr_core_end_seq_id=record.nmr_core_end_seq_id,
                nmr_query_sequence_length=record.nmr_query_sequence_length,
                xray_homolog_entry_ids=kept_entry_ids,
                xray_homolog_entity_ids=kept_entity_ids,
                has_xray_homolog=bool(kept_entity_ids),
            )
        )
    return historical_records
