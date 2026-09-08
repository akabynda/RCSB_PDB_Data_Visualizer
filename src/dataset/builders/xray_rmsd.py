"""Evaluate first-model NMR-to-X-ray RMSD candidates."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from src.dataset.ca_cache import (
    load_cached_first_model_ca_data,
)
from src.dataset.config import (
    LOGGER,
)
from src.dataset.coordinates import (
    parse_first_model_ca_residues,
    parse_models_ca_coords,
)
from src.dataset.downloads import (
    download_pdb_chain_subset_if_needed,
    download_pdb_if_needed,
)
from src.dataset.geometry import (
    _superposed_rmsd,
)
from src.dataset.matching import (
    _ca_residue_has_hetatm,
    find_modeled_ca_core_identity_matches,
)
from src.dataset.records import (
    ResidueId,
    SolutionNMRMonomerXrayRmsdExtremesRecord,
    SolutionNMRMonomerXrayRmsdRecord,
)
from src.dataset.reporting import (
    _record_filtered_structure,
)
from src.dataset.structures import (
    load_cached_chain_id_map,
)
from src.dataset.utils import (
    chunked,
    collect_batch_results,
)

if TYPE_CHECKING:
    from src.dataset.client import (
        RCSBClient,
    )
    from src.dataset.config import (
        DatasetBuildConfig,
    )
    from src.dataset.records import (
        PreparedNMRCoreData,
        SolutionNMRMonomerXrayHomologRecord,
        XrayPolymerEntityCandidateRecord,
    )


def _xray_rmsd_records_for_homolog_view(
    homolog: SolutionNMRMonomerXrayHomologRecord,
    candidate_records: Sequence[SolutionNMRMonomerXrayRmsdRecord],
) -> tuple[SolutionNMRMonomerXrayRmsdRecord, ...]:
    """Filter shared successful calculations to one current/historical view."""
    allowed_entity_ids = set(homolog.xray_homolog_entity_ids)
    return tuple(
        record
        for record in candidate_records
        if record.xray_homolog_entity_id in allowed_entity_ids
    )


def _select_ordinary_xray_rmsd_record(
    *,
    homolog: SolutionNMRMonomerXrayHomologRecord,
    candidate_records: Sequence[SolutionNMRMonomerXrayRmsdRecord],
) -> SolutionNMRMonomerXrayRmsdRecord | None:
    """Project the first resolution-ordered successful pair to one view."""
    eligible_records = _xray_rmsd_records_for_homolog_view(homolog, candidate_records)
    if not eligible_records:
        return None
    return replace(
        eligible_records[0],
        year=homolog.year,
        sequence_identity_percent=homolog.sequence_identity_percent,
        nmr_core_start_seq_id=homolog.nmr_core_start_seq_id,
        nmr_core_end_seq_id=homolog.nmr_core_end_seq_id,
        nmr_query_sequence_length=homolog.nmr_query_sequence_length,
        xray_homolog_count=len(homolog.xray_homolog_entity_ids),
    )


def _build_xray_rmsd_extremes_record_from_candidates(
    *,
    homolog: SolutionNMRMonomerXrayHomologRecord,
    candidate_records: Sequence[SolutionNMRMonomerXrayRmsdRecord],
) -> SolutionNMRMonomerXrayRmsdExtremesRecord | None:
    """Project minimum and maximum shared candidate RMSDs to one view."""
    eligible_records = _xray_rmsd_records_for_homolog_view(homolog, candidate_records)
    if not eligible_records:
        return None
    best = min(
        eligible_records,
        key=lambda record: (
            record.rmsd_ca_angstrom,
            -record.n_common_ca,
            np.isnan(record.xray_resolution_angstrom),
            record.xray_resolution_angstrom,
            record.xray_entry_id,
            record.xray_homolog_entity_id,
        ),
    )
    worst = max(
        eligible_records,
        key=lambda record: (
            record.rmsd_ca_angstrom,
            record.n_common_ca,
            not np.isnan(record.xray_resolution_angstrom),
            (
                -record.xray_resolution_angstrom
                if not np.isnan(record.xray_resolution_angstrom)
                else 0.0
            ),
            record.xray_entry_id,
            record.xray_homolog_entity_id,
        ),
    )
    return SolutionNMRMonomerXrayRmsdExtremesRecord(
        entry_id=homolog.entry_id,
        year=homolog.year,
        sequence_identity_percent=homolog.sequence_identity_percent,
        nmr_chain_id=best.nmr_chain_id,
        nmr_core_start_seq_id=homolog.nmr_core_start_seq_id,
        nmr_core_end_seq_id=homolog.nmr_core_end_seq_id,
        nmr_query_sequence_length=homolog.nmr_query_sequence_length,
        xray_homolog_count=len(homolog.xray_homolog_entity_ids),
        successful_xray_homolog_count=len(eligible_records),
        best_xray_homolog_entity_id=best.xray_homolog_entity_id,
        best_xray_entry_id=best.xray_entry_id,
        best_xray_chain_id=best.xray_chain_id,
        best_xray_resolution_angstrom=best.xray_resolution_angstrom,
        best_xray_core_start_seq_id=best.xray_core_start_seq_id,
        best_xray_core_end_seq_id=best.xray_core_end_seq_id,
        best_n_common_ca=best.n_common_ca,
        best_rmsd_ca_angstrom=best.rmsd_ca_angstrom,
        worst_xray_homolog_entity_id=worst.xray_homolog_entity_id,
        worst_xray_entry_id=worst.xray_entry_id,
        worst_xray_chain_id=worst.xray_chain_id,
        worst_xray_resolution_angstrom=worst.xray_resolution_angstrom,
        worst_xray_core_start_seq_id=worst.xray_core_start_seq_id,
        worst_xray_core_end_seq_id=worst.xray_core_end_seq_id,
        worst_n_common_ca=worst.n_common_ca,
        worst_rmsd_ca_angstrom=worst.rmsd_ca_angstrom,
        rmsd_delta_angstrom=worst.rmsd_ca_angstrom - best.rmsd_ca_angstrom,
    )


class SolutionNMRMonomerXrayRmsdBuilder:
    """Compute alpha-carbon RMSDs between NMR entries and X-ray homologs."""

    def __init__(
        self,
        client: RCSBClient,
        config: DatasetBuildConfig,
        cache_dir: Path,
        rmsd_workers: int,
        homolog_records: list[SolutionNMRMonomerXrayHomologRecord],
        sequence_identity_percent: int = 100,
    ) -> None:
        """Initialize NMR-to-X-ray RMSD collection."""
        if sequence_identity_percent not in {95, 100}:
            raise ValueError("sequence_identity_percent must be 95 or 100")
        self.client = client
        self.config = config
        self.cache_dir = cache_dir
        self.rmsd_workers = max(1, rmsd_workers)
        self.homolog_records = homolog_records
        self.sequence_identity_percent = sequence_identity_percent
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _download_pdb_if_needed(self, entry_id: str) -> Path:
        """Download or reuse a PDB coordinate file for RMSD work."""
        return download_pdb_if_needed(
            session=self.client.session,
            config=self.config,
            cache_dir=self.cache_dir,
            entry_id=entry_id,
        )

    def _compute_candidate_record(
        self,
        homolog: SolutionNMRMonomerXrayHomologRecord,
        nmr_chain_id: str,
        nmr_pdb_path: Path,
        parsed_nmr_chain_id: str,
        prepared_nmr_core: PreparedNMRCoreData,
        candidate: XrayPolymerEntityCandidateRecord,
    ) -> SolutionNMRMonomerXrayRmsdRecord | None:
        """Compute RMSD metrics for one NMR and X-ray candidate pair."""
        best_chain_result: tuple[str, int, float, int, int, int, int] | None = None
        for xray_chain_id in candidate.chain_ids:
            try:
                (
                    xray_pdb_path,
                    xray_chain_map,
                ) = download_pdb_chain_subset_if_needed(
                    session=self.client.session,
                    config=self.config,
                    cache_dir=self.cache_dir,
                    entry_id=candidate.entry_id,
                    chain_ids=(xray_chain_id,),
                )
            except Exception as exc:
                LOGGER.debug(
                    "Skipping X-ray RMSD chain %s/%s for %s: %s",
                    candidate.entry_id,
                    xray_chain_id,
                    homolog.entry_id,
                    exc,
                )
                continue
            parsed_xray_chain_id = xray_chain_map.get(
                xray_chain_id,
                xray_chain_id,
            )
            try:
                rmsd_result = self._compute_ca_rmsd_to_xray(
                    nmr_pdb_path=nmr_pdb_path,
                    nmr_chain_id=parsed_nmr_chain_id,
                    nmr_core_start_seq_id=homolog.nmr_core_start_seq_id,
                    nmr_core_end_seq_id=homolog.nmr_core_end_seq_id,
                    xray_pdb_path=xray_pdb_path,
                    xray_chain_id=parsed_xray_chain_id,
                    sequence_identity_percent=self.sequence_identity_percent,
                    prepared_nmr_core=prepared_nmr_core,
                )
            except Exception as exc:
                LOGGER.debug(
                    "Skipping X-ray RMSD chain %s/%s for %s: %s",
                    candidate.entry_id,
                    xray_chain_id,
                    homolog.entry_id,
                    exc,
                )
                continue
            if rmsd_result is None:
                continue
            (
                n_common_ca,
                rmsd_ca,
                _nmr_core_start,
                _nmr_core_end,
                xray_core_start,
                xray_core_end,
            ) = rmsd_result
            chain_candidate = (
                xray_chain_id,
                n_common_ca,
                rmsd_ca,
                homolog.nmr_core_start_seq_id,
                homolog.nmr_core_end_seq_id,
                xray_core_start,
                xray_core_end,
            )
            if (
                best_chain_result is None
                or chain_candidate[1] > best_chain_result[1]
                or (
                    chain_candidate[1] == best_chain_result[1]
                    and chain_candidate[2] < best_chain_result[2]
                )
            ):
                best_chain_result = chain_candidate

        if best_chain_result is None:
            return None
        (
            xray_chain_id,
            n_common_ca,
            rmsd_ca,
            _nmr_core_start,
            _nmr_core_end,
            xray_core_start,
            xray_core_end,
        ) = best_chain_result
        return SolutionNMRMonomerXrayRmsdRecord(
            entry_id=homolog.entry_id,
            year=homolog.year,
            sequence_identity_percent=self.sequence_identity_percent,
            nmr_chain_id=nmr_chain_id,
            nmr_core_start_seq_id=homolog.nmr_core_start_seq_id,
            nmr_core_end_seq_id=homolog.nmr_core_end_seq_id,
            nmr_query_sequence_length=homolog.nmr_query_sequence_length,
            xray_homolog_entity_id=candidate.polymer_entity_id,
            xray_homolog_count=len(homolog.xray_homolog_entity_ids),
            xray_entry_id=candidate.entry_id,
            xray_chain_id=xray_chain_id,
            xray_core_start_seq_id=xray_core_start,
            xray_core_end_seq_id=xray_core_end,
            xray_resolution_angstrom=candidate.resolution_angstrom,
            n_common_ca=n_common_ca,
            rmsd_ca_angstrom=rmsd_ca,
        )

    @staticmethod
    def _prepare_nmr_core_data(
        nmr_pdb_path: Path,
        nmr_chain_id: str,
        nmr_core_start_seq_id: ResidueId | int,
        nmr_core_end_seq_id: ResidueId | int,
    ) -> PreparedNMRCoreData | None:
        """Parse invariant NMR core inputs once for all X-ray candidates."""
        nmr_residues = parse_first_model_ca_residues(
            nmr_pdb_path,
            nmr_chain_id,
            start_seq_id=nmr_core_start_seq_id,
            end_seq_id=nmr_core_end_seq_id,
            include_hetatm=True,
        )
        if len(nmr_residues) <= 10:
            return None
        if any(_ca_residue_has_hetatm(record) for record in nmr_residues):
            return None

        nmr_model_maps = parse_models_ca_coords(
            nmr_pdb_path,
            nmr_chain_id,
            start_seq_id=nmr_core_start_seq_id,
            end_seq_id=nmr_core_end_seq_id,
        )
        if not nmr_model_maps:
            return None
        return tuple(nmr_residues), nmr_model_maps[0]

    @staticmethod
    def _compute_ca_rmsd_to_xray(
        nmr_pdb_path: Path,
        nmr_chain_id: str,
        nmr_core_start_seq_id: ResidueId | int,
        nmr_core_end_seq_id: ResidueId | int,
        xray_pdb_path: Path,
        xray_chain_id: str,
        sequence_identity_percent: int,
        prepared_nmr_core: PreparedNMRCoreData | None = None,
    ) -> (
        tuple[int, float, ResidueId | int, ResidueId | int, ResidueId, ResidueId] | None
    ):
        """Compute CA RMSD after aligning an NMR model core to an X-ray chain."""
        _ = sequence_identity_percent
        if prepared_nmr_core is None:
            prepared_nmr_core = (
                SolutionNMRMonomerXrayRmsdBuilder._prepare_nmr_core_data(
                    nmr_pdb_path=nmr_pdb_path,
                    nmr_chain_id=nmr_chain_id,
                    nmr_core_start_seq_id=nmr_core_start_seq_id,
                    nmr_core_end_seq_id=nmr_core_end_seq_id,
                )
            )
        if prepared_nmr_core is None:
            return None
        cached_nmr_residues, nmr_first_model_map = prepared_nmr_core
        nmr_residues = list(cached_nmr_residues)

        cached_xray_residues, xray_first_model_map = load_cached_first_model_ca_data(
            pdb_path=xray_pdb_path,
            chain_id=xray_chain_id,
        )
        xray_residues = list(cached_xray_residues)
        matched_pair_sets = find_modeled_ca_core_identity_matches(
            nmr_residues=nmr_residues,
            xray_residues=xray_residues,
            sequence_identity_percent=100,
        )
        if not matched_pair_sets:
            return None
        if not xray_first_model_map:
            return None

        best_result: (
            tuple[int, float, ResidueId | int, ResidueId | int, ResidueId, ResidueId]
            | None
        ) = None
        for matched_pairs in matched_pair_sets:
            rmsd_pairs = [
                (nmr_record.key, xray_record.key)
                for nmr_record, xray_record in matched_pairs
                if nmr_record.is_standard_atom and xray_record.is_standard_atom
            ]
            if len(rmsd_pairs) < 3:
                continue

            nmr_common_resids = [nmr_resid for nmr_resid, _ in rmsd_pairs]
            xray_common_resids = [xray_resid for _, xray_resid in rmsd_pairs]
            if any(resid not in nmr_first_model_map for resid in nmr_common_resids):
                continue
            if any(resid not in xray_first_model_map for resid in xray_common_resids):
                continue

            nmr_coords = np.asarray(
                [nmr_first_model_map[resid] for resid in nmr_common_resids],
                dtype=float,
            )
            xray_coords = np.asarray(
                [xray_first_model_map[resid] for resid in xray_common_resids],
                dtype=float,
            )
            rmsd_value = _superposed_rmsd(nmr_coords, xray_coords)
            xray_matched_resids = [xray_record.key for _, xray_record in matched_pairs]
            result = (
                len(rmsd_pairs),
                rmsd_value,
                nmr_core_start_seq_id,
                nmr_core_end_seq_id,
                xray_matched_resids[0],
                xray_matched_resids[-1],
            )
            if best_result is None or result[1] < best_result[1]:
                best_result = result

        return best_result

    def _compute_candidate_records(
        self,
        homolog: SolutionNMRMonomerXrayHomologRecord,
        nmr_chain_id: str,
        candidates: tuple[XrayPolymerEntityCandidateRecord, ...],
    ) -> tuple[SolutionNMRMonomerXrayRmsdRecord, ...]:
        """Compute every successful pair once in candidate sort order."""
        if homolog.nmr_core_start_seq_id is None or homolog.nmr_core_end_seq_id is None:
            _record_filtered_structure(
                homolog.entry_id, "NMR core range is missing", year=homolog.year
            )
            return tuple()
        if not candidates:
            _record_filtered_structure(
                homolog.entry_id,
                "no usable X-ray homolog candidates",
                year=homolog.year,
            )
            return tuple()

        try:
            nmr_pdb_path = self._download_pdb_if_needed(homolog.entry_id)
            nmr_chain_map = load_cached_chain_id_map(self.cache_dir, homolog.entry_id)
            parsed_nmr_chain_id = nmr_chain_map.get(nmr_chain_id, nmr_chain_id)
            prepared_nmr_core = self._prepare_nmr_core_data(
                nmr_pdb_path=nmr_pdb_path,
                nmr_chain_id=parsed_nmr_chain_id,
                nmr_core_start_seq_id=homolog.nmr_core_start_seq_id,
                nmr_core_end_seq_id=homolog.nmr_core_end_seq_id,
            )
            if prepared_nmr_core is None:
                _record_filtered_structure(
                    homolog.entry_id,
                    "NMR core cannot be prepared for X-ray RMSD",
                    year=homolog.year,
                )
                return tuple()

            candidate_records: list[SolutionNMRMonomerXrayRmsdRecord] = []
            for candidate in candidates:
                record = self._compute_candidate_record(
                    homolog=homolog,
                    nmr_chain_id=nmr_chain_id,
                    nmr_pdb_path=nmr_pdb_path,
                    parsed_nmr_chain_id=parsed_nmr_chain_id,
                    prepared_nmr_core=prepared_nmr_core,
                    candidate=candidate,
                )
                if record is not None:
                    candidate_records.append(record)
            if not candidate_records:
                _record_filtered_structure(
                    homolog.entry_id,
                    "no X-ray homolog candidate produced a usable CA RMSD",
                    year=homolog.year,
                )
            return tuple(candidate_records)
        except Exception as exc:
            LOGGER.warning(
                "X-ray RMSD calculation failed for %s: %s", homolog.entry_id, exc
            )
            _record_filtered_structure(
                homolog.entry_id,
                f"X-ray RMSD calculation failed: {exc}",
                year=homolog.year,
            )
            return tuple()

    def _compute_record(
        self,
        homolog: SolutionNMRMonomerXrayHomologRecord,
        nmr_chain_id: str,
        candidates: tuple[XrayPolymerEntityCandidateRecord, ...],
    ) -> SolutionNMRMonomerXrayRmsdRecord | None:
        """Compute and project the ordinary X-ray RMSD record for one seed."""
        candidate_records = self._compute_candidate_records(
            homolog=homolog,
            nmr_chain_id=nmr_chain_id,
            candidates=candidates,
        )
        return _select_ordinary_xray_rmsd_record(
            homolog=homolog,
            candidate_records=candidate_records,
        )

    def _compute_extremes_record(
        self,
        homolog: SolutionNMRMonomerXrayHomologRecord,
        nmr_chain_id: str,
        candidates: tuple[XrayPolymerEntityCandidateRecord, ...],
    ) -> SolutionNMRMonomerXrayRmsdExtremesRecord | None:
        """Compute and project X-ray RMSD extremes for one NMR seed."""
        candidate_records = self._compute_candidate_records(
            homolog=homolog,
            nmr_chain_id=nmr_chain_id,
            candidates=candidates,
        )
        return _build_xray_rmsd_extremes_record_from_candidates(
            homolog=homolog,
            candidate_records=candidate_records,
        )

    def _prepare_work_items(
        self,
        skip_entry_ids: set[str],
        progress_prefix: str,
    ) -> list[
        tuple[
            SolutionNMRMonomerXrayHomologRecord,
            str,
            tuple[XrayPolymerEntityCandidateRecord, ...],
        ]
    ]:
        """Prepare NMR seeds and candidate lists for RMSD collection."""
        filtered_homologs: list[SolutionNMRMonomerXrayHomologRecord] = []
        for record in self.homolog_records:
            if record.sequence_identity_percent != self.sequence_identity_percent:
                continue
            if record.entry_id in skip_entry_ids:
                continue
            if (
                record.nmr_core_start_seq_id is None
                or record.nmr_core_end_seq_id is None
            ):
                _record_filtered_structure(
                    record.entry_id, "NMR core range is missing", year=record.year
                )
                continue
            if not record.xray_homolog_entity_ids:
                _record_filtered_structure(
                    record.entry_id,
                    f"no X-ray homologs at {self.sequence_identity_percent}% sequence identity",
                    year=record.year,
                )
                continue
            filtered_homologs.append(record)
        filtered_homologs = sorted(
            filtered_homologs, key=lambda r: (r.year, r.entry_id)
        )

        entry_ids = sorted({record.entry_id for record in filtered_homologs})
        chain_by_entry_id: dict[str, str] = {}
        for batch_seeds in collect_batch_results(
            batches=list(chunked(entry_ids, self.config.graphql_batch_size)),
            max_workers=self.config.max_workers,
            fetch_fn=self.client.fetch_solution_nmr_monomer_xray_homolog_seed_records_for_ids,
            progress_label=f"{progress_prefix} NMR chain lookup",
        ):
            for seed in batch_seeds:
                chain_by_entry_id[seed.entry_id] = seed.chain_id

        xray_entity_ids = sorted(
            {
                entity_id
                for record in filtered_homologs
                for entity_id in record.xray_homolog_entity_ids
            }
        )
        candidate_by_entity_id: dict[str, XrayPolymerEntityCandidateRecord] = {}
        for batch_candidates in collect_batch_results(
            batches=list(chunked(xray_entity_ids, self.config.graphql_batch_size)),
            max_workers=self.config.max_workers,
            fetch_fn=self.client.fetch_xray_polymer_entity_candidates_for_ids,
            progress_label=f"{progress_prefix} X-ray candidate metadata",
        ):
            for candidate in batch_candidates:
                candidate_by_entity_id[candidate.polymer_entity_id] = candidate

        work_items: list[
            tuple[
                SolutionNMRMonomerXrayHomologRecord,
                str,
                tuple[XrayPolymerEntityCandidateRecord, ...],
            ]
        ] = []
        for homolog in filtered_homologs:
            nmr_chain_id = chain_by_entry_id.get(homolog.entry_id)
            if not nmr_chain_id:
                _record_filtered_structure(
                    homolog.entry_id,
                    "NMR chain metadata is missing",
                    year=homolog.year,
                )
                continue
            candidates = tuple(
                sorted(
                    (
                        candidate_by_entity_id[entity_id]
                        for entity_id in homolog.xray_homolog_entity_ids
                        if entity_id in candidate_by_entity_id
                    ),
                    key=lambda c: (
                        np.isnan(c.resolution_angstrom),
                        c.resolution_angstrom,
                        c.entry_id,
                        c.polymer_entity_id,
                    ),
                )
            )
            if not candidates:
                _record_filtered_structure(
                    homolog.entry_id,
                    "X-ray homolog candidate metadata is missing",
                    year=homolog.year,
                )
                continue
            work_items.append((homolog, nmr_chain_id, candidates))

        LOGGER.info(
            "%s %d%%: entries to process=%d, unique X-ray entities=%d",
            progress_prefix,
            self.sequence_identity_percent,
            len(work_items),
            len(candidate_by_entity_id),
        )
        return work_items

    def build(
        self,
        skip_entry_ids: set[str] | None = None,
        on_record: Callable[[SolutionNMRMonomerXrayRmsdRecord], None] | None = None,
    ) -> list[SolutionNMRMonomerXrayRmsdRecord]:
        """Collect best-match NMR-to-X-ray RMSD records."""
        skip_entry_ids = skip_entry_ids or set()
        work_items = self._prepare_work_items(
            skip_entry_ids=skip_entry_ids,
            progress_prefix="SOLUTION NMR X-ray RMSD",
        )

        records: list[SolutionNMRMonomerXrayRmsdRecord] = []
        with ThreadPoolExecutor(max_workers=self.rmsd_workers) as executor:
            future_map = {
                executor.submit(
                    self._compute_record,
                    homolog=homolog,
                    nmr_chain_id=nmr_chain_id,
                    candidates=candidates,
                ): idx
                for idx, (homolog, nmr_chain_id, candidates) in enumerate(
                    work_items, start=1
                )
            }
            total = len(future_map)
            completed = 0
            for future in as_completed(future_map):
                record = future.result()
                if record is not None:
                    records.append(record)
                    if on_record is not None:
                        on_record(record)
                completed += 1
                if total > 0 and (completed % 50 == 0 or completed == total):
                    LOGGER.info(
                        "SOLUTION NMR X-ray RMSD %d%%: processed %d/%d entries",
                        self.sequence_identity_percent,
                        completed,
                        total,
                    )

        return sorted(records, key=lambda r: (r.year, r.entry_id))

    def build_extremes(
        self,
        skip_entry_ids: set[str] | None = None,
        on_record: (
            Callable[[SolutionNMRMonomerXrayRmsdExtremesRecord], None] | None
        ) = None,
    ) -> list[SolutionNMRMonomerXrayRmsdExtremesRecord]:
        """Collect minimum and maximum NMR-to-X-ray RMSD records."""
        skip_entry_ids = skip_entry_ids or set()
        work_items = self._prepare_work_items(
            skip_entry_ids=skip_entry_ids,
            progress_prefix="SOLUTION NMR X-ray RMSD extremes",
        )

        records: list[SolutionNMRMonomerXrayRmsdExtremesRecord] = []
        with ThreadPoolExecutor(max_workers=self.rmsd_workers) as executor:
            future_map = {
                executor.submit(
                    self._compute_extremes_record,
                    homolog=homolog,
                    nmr_chain_id=nmr_chain_id,
                    candidates=candidates,
                ): idx
                for idx, (homolog, nmr_chain_id, candidates) in enumerate(
                    work_items, start=1
                )
            }
            total = len(future_map)
            completed = 0
            for future in as_completed(future_map):
                record = future.result()
                if record is not None:
                    records.append(record)
                    if on_record is not None:
                        on_record(record)
                completed += 1
                if total > 0 and (completed % 50 == 0 or completed == total):
                    LOGGER.info(
                        "SOLUTION NMR X-ray RMSD extremes %d%%: processed %d/%d entries",
                        self.sequence_identity_percent,
                        completed,
                        total,
                    )

        return sorted(records, key=lambda r: (r.year, r.entry_id))

    def build_candidate_sets(
        self,
        skip_entry_ids: set[str] | None = None,
        on_candidate_set: (
            Callable[
                [
                    SolutionNMRMonomerXrayHomologRecord,
                    tuple[SolutionNMRMonomerXrayRmsdRecord, ...],
                ],
                None,
            ]
            | None
        ) = None,
    ) -> list[
        tuple[
            SolutionNMRMonomerXrayHomologRecord,
            tuple[SolutionNMRMonomerXrayRmsdRecord, ...],
        ]
    ]:
        """Compute shared candidate pairs once for all requested RMSD views."""
        work_items = self._prepare_work_items(
            skip_entry_ids=skip_entry_ids or set(),
            progress_prefix="SOLUTION NMR unified X-ray RMSD",
        )
        results: list[
            tuple[
                SolutionNMRMonomerXrayHomologRecord,
                tuple[SolutionNMRMonomerXrayRmsdRecord, ...],
            ]
        ] = []
        with ThreadPoolExecutor(max_workers=self.rmsd_workers) as executor:
            future_map = {
                executor.submit(
                    self._compute_candidate_records,
                    homolog=homolog,
                    nmr_chain_id=nmr_chain_id,
                    candidates=candidates,
                ): homolog
                for homolog, nmr_chain_id, candidates in work_items
            }
            total = len(future_map)
            for completed, future in enumerate(as_completed(future_map), start=1):
                homolog = future_map[future]
                candidate_records = future.result()
                results.append((homolog, candidate_records))
                if on_candidate_set is not None:
                    on_candidate_set(homolog, candidate_records)
                if total > 0 and (completed % 50 == 0 or completed == total):
                    LOGGER.info(
                        "SOLUTION NMR unified X-ray RMSD %d%%: processed %d/%d entries",
                        self.sequence_identity_percent,
                        completed,
                        total,
                    )
        return sorted(results, key=lambda item: (item[0].year, item[0].entry_id))
