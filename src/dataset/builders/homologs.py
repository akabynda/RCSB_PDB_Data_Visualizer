"""Find and audit X-ray homologs for eligible NMR monomer cores."""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from src.dataset.ca_cache import (
    load_cached_first_model_ca_data,
)
from src.dataset.config import (
    LOGGER,
    XRAY_HOMOLOG_HETATM_REJECTION_REASON,
    XRAY_HOMOLOG_METHOD_REJECTION_REASON,
    XRAY_HOMOLOG_SEARCH_MAX_ATTEMPTS,
)
from src.dataset.coordinates import (
    parse_first_model_ca_residues,
    parse_first_model_modeled_ca_auth_seq_ids,
)
from src.dataset.downloads import (
    download_pdb_chain_subset_if_needed,
    download_pdb_if_needed,
)
from src.dataset.errors import (
    NMRCoreContainsHetatmError,
    NMRHomologyQueryIneligibleError,
    XrayHomologEvaluationError,
    _is_http_server_error,
)
from src.dataset.matching import (
    _ca_residue_has_hetatm,
    find_modeled_ca_core_identity_matches,
)
from src.dataset.records import (
    RejectedXrayHomologRecord,
    SolutionNMRMonomerXrayHomologRecord,
    XrayPolymerEntityCandidateRecord,
)
from src.dataset.reporting import (
    _record_filtered_structure,
)
from src.dataset.stride import (
    compute_stride_core_range_for_modeled_auth_seq_ids_in_first_model,
)
from src.dataset.structures import (
    load_cached_chain_id_map,
)
from src.dataset.utils import (
    chunked,
    collect_batch_results,
    fetch_solution_nmr_entry_ids,
)

if TYPE_CHECKING:
    from src.dataset.client import (
        RCSBClient,
    )
    from src.dataset.config import (
        DatasetBuildConfig,
    )
    from src.dataset.records import (
        CAResidueRecord,
        SolutionNMRMonomerXrayHomologSeedRecord,
    )


_MISSING = object()


class SolutionNMRMonomerXrayHomologBuilder:
    """Find sequence-identical X-ray homologs for solution-NMR monomers."""

    def __init__(
        self,
        client: RCSBClient,
        config: DatasetBuildConfig,
        stride_executable: str,
        cache_dir: Path,
        stride_cache_dir: Path,
    ) -> None:
        """Initialize X-ray homolog collection for SOLUTION NMR monomers."""
        self.client = client
        self.config = config
        self.stride_executable = stride_executable
        self.cache_dir = cache_dir
        self.stride_cache_dir = stride_cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.stride_cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _entry_ids_from_polymer_entity_ids(
        entity_ids: Sequence[str],
    ) -> tuple[str, ...]:
        """Extract entry IDs from polymer entity identifiers."""
        seen: set[str] = set()
        entry_ids: list[str] = []
        for entity_id in entity_ids:
            entry_id = str(entity_id).split("_", 1)[0].strip()
            if not entry_id or entry_id in seen:
                continue
            seen.add(entry_id)
            entry_ids.append(entry_id)
        return tuple(entry_ids)

    def _build_stride_core_query_sequence(
        self,
        seed: SolutionNMRMonomerXrayHomologSeedRecord,
    ) -> tuple[str, int, int, list[CAResidueRecord]]:
        """Build the STRIDE-core query sequence used for homolog searches."""
        pdb_path = download_pdb_if_needed(
            session=self.client.session,
            config=self.config,
            cache_dir=self.cache_dir,
            entry_id=seed.entry_id,
        )
        chain_map = load_cached_chain_id_map(self.cache_dir, seed.entry_id)
        parsed_chain_id = chain_map.get(seed.chain_id, seed.chain_id)
        modeled_auth_seq_ids = parse_first_model_modeled_ca_auth_seq_ids(
            pdb_path=pdb_path,
            chain_id=parsed_chain_id,
        )
        if not modeled_auth_seq_ids:
            raise NMRHomologyQueryIneligibleError(
                seed.entry_id, "no usable first-model modeled CA residues"
            )
        core_range = compute_stride_core_range_for_modeled_auth_seq_ids_in_first_model(
            pdb_path=pdb_path,
            entry_id=seed.entry_id,
            chain_id=parsed_chain_id,
            modeled_auth_seq_ids=modeled_auth_seq_ids,
            stride_executable=self.stride_executable,
            stride_cache_dir=self.stride_cache_dir,
        )
        if core_range is None:
            raise NMRHomologyQueryIneligibleError(
                seed.entry_id, "STRIDE found no modeled core residues"
            )
        core_start, core_end = core_range
        nmr_core_residues = parse_first_model_ca_residues(
            pdb_path=pdb_path,
            chain_id=parsed_chain_id,
            start_seq_id=core_start,
            end_seq_id=core_end,
            include_hetatm=True,
        )
        if any(_ca_residue_has_hetatm(record) for record in nmr_core_residues):
            raise NMRCoreContainsHetatmError(seed.entry_id)
        if len(nmr_core_residues) <= 10:
            raise NMRHomologyQueryIneligibleError(
                seed.entry_id,
                f"STRIDE core is too short ({len(nmr_core_residues)} CA residues)",
            )
        identities = [record.identity for record in nmr_core_residues]
        if any(len(identity) != 1 or not identity.isalpha() for identity in identities):
            raise NMRHomologyQueryIneligibleError(
                seed.entry_id, "STRIDE core contains unusable residue identities"
            )
        query_sequence = "".join(identities)
        if not query_sequence:
            raise NMRHomologyQueryIneligibleError(
                seed.entry_id, "STRIDE core query sequence is empty"
            )
        return query_sequence, core_start, core_end, nmr_core_residues

    def _xray_candidate_has_modeled_core_match(
        self,
        nmr_core_residues: list[CAResidueRecord],
        candidate: XrayPolymerEntityCandidateRecord,
        sequence_identity_percent: int,
        chain_residue_cache: (
            dict[tuple[str, str], tuple[CAResidueRecord, ...]] | None
        ) = None,
    ) -> bool:
        """Check whether an X-ray candidate models the NMR core sequence."""
        chain_errors: list[tuple[str, Exception]] = []
        for xray_chain_id in candidate.chain_ids:
            cache_key = (candidate.entry_id, xray_chain_id)
            cached_residues = (
                chain_residue_cache.get(cache_key)
                if chain_residue_cache is not None
                else None
            )
            if cached_residues is None:
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
                    parsed_xray_chain_id = xray_chain_map.get(
                        xray_chain_id, xray_chain_id
                    )
                    cached_residues, _ = load_cached_first_model_ca_data(
                        pdb_path=xray_pdb_path,
                        chain_id=parsed_xray_chain_id,
                    )
                except Exception as exc:
                    chain_errors.append((xray_chain_id, exc))
                    continue
                if chain_residue_cache is not None:
                    chain_residue_cache[cache_key] = cached_residues
            if find_modeled_ca_core_identity_matches(
                nmr_residues=nmr_core_residues,
                xray_residues=list(cached_residues),
                sequence_identity_percent=sequence_identity_percent,
            ):
                return True
        if chain_errors:
            failed_chain_id, error = chain_errors[0]
            raise XrayHomologEvaluationError(
                "Could not evaluate X-ray homolog candidate "
                f"{candidate.polymer_entity_id}: chain {failed_chain_id} failed: "
                f"{error} ({len(chain_errors)}/{len(candidate.chain_ids)} chains failed)"
            ) from error
        return False

    def _filter_modeled_xray_homolog_entity_ids(
        self,
        xray_entity_ids: tuple[str, ...],
        nmr_core_residues: list[CAResidueRecord],
        sequence_identity_percent: int,
        candidate_cache: (dict[str, XrayPolymerEntityCandidateRecord] | None) = None,
        chain_residue_cache: (
            dict[tuple[str, str], tuple[CAResidueRecord, ...]] | None
        ) = None,
        rejection_reason_by_entity_id: dict[str, str] | None = None,
    ) -> tuple[str, ...]:
        """Filter candidates by exact X-ray method and modeled core."""
        if not xray_entity_ids:
            return tuple()

        candidate_by_entity_id = candidate_cache if candidate_cache is not None else {}
        uncached_entity_ids = [
            entity_id
            for entity_id in xray_entity_ids
            if entity_id not in candidate_by_entity_id
        ]
        candidates: list[XrayPolymerEntityCandidateRecord] = []
        for entity_id_batch in chunked(
            uncached_entity_ids, self.config.graphql_batch_size
        ):
            candidates.extend(
                self.client.fetch_xray_polymer_entity_candidates_for_ids(
                    entity_id_batch,
                )
            )
        candidate_by_entity_id.update(
            {candidate.polymer_entity_id: candidate for candidate in candidates}
        )
        missing_entity_ids = [
            entity_id
            for entity_id in xray_entity_ids
            if entity_id not in candidate_by_entity_id
        ]
        if missing_entity_ids:
            raise XrayHomologEvaluationError(
                "Missing X-ray candidate metadata for: " + ", ".join(missing_entity_ids)
            )

        filtered_entity_ids: list[str] = []
        for entity_id in xray_entity_ids:
            candidate = candidate_by_entity_id.get(entity_id)
            if candidate is None:
                continue
            if frozenset(candidate.experimental_methods) != frozenset(
                {"X-RAY DIFFRACTION"}
            ):
                if rejection_reason_by_entity_id is not None:
                    rejection_reason_by_entity_id[entity_id] = (
                        XRAY_HOMOLOG_METHOD_REJECTION_REASON
                    )
                continue
            if self._xray_candidate_has_modeled_core_match(
                nmr_core_residues=nmr_core_residues,
                candidate=candidate,
                sequence_identity_percent=sequence_identity_percent,
                chain_residue_cache=chain_residue_cache,
            ):
                filtered_entity_ids.append(entity_id)
            elif rejection_reason_by_entity_id is not None:
                rejection_reason_by_entity_id[entity_id] = (
                    XRAY_HOMOLOG_HETATM_REJECTION_REASON
                )
        return tuple(filtered_entity_ids)

    def _build_record(
        self,
        seed: SolutionNMRMonomerXrayHomologSeedRecord,
        sequence_identity_percent: int,
        core_query: (tuple[str, int, int, list[CAResidueRecord]] | object) = _MISSING,
        candidate_cache: (dict[str, XrayPolymerEntityCandidateRecord] | None) = None,
        chain_residue_cache: (
            dict[tuple[str, str], tuple[CAResidueRecord, ...]] | None
        ) = None,
    ) -> SolutionNMRMonomerXrayHomologRecord:
        """Build one X-ray homolog summary record from a seed."""
        if core_query is _MISSING:
            core_query = self._build_stride_core_query_sequence(seed)
        assert core_query is not _MISSING
        query_sequence, core_start, core_end, nmr_core_residues = core_query
        raw_xray_entity_ids = tuple(
            self.client.fetch_xray_polymer_entity_ids_by_sequence(
                sequence=query_sequence,
                sequence_identity_percent=sequence_identity_percent,
            )
        )
        resolved_candidate_cache = (
            candidate_cache if candidate_cache is not None else {}
        )
        rejection_reason_by_entity_id: dict[str, str] = {}
        xray_entity_ids = self._filter_modeled_xray_homolog_entity_ids(
            xray_entity_ids=raw_xray_entity_ids,
            nmr_core_residues=nmr_core_residues,
            sequence_identity_percent=sequence_identity_percent,
            candidate_cache=resolved_candidate_cache,
            chain_residue_cache=chain_residue_cache,
            rejection_reason_by_entity_id=rejection_reason_by_entity_id,
        )
        kept_entity_ids = set(xray_entity_ids)
        rejected_xray_homologs: list[RejectedXrayHomologRecord] = []
        seen_rejected_entity_ids: set[str] = set()
        for entity_id in raw_xray_entity_ids:
            if entity_id in kept_entity_ids or entity_id in seen_rejected_entity_ids:
                continue
            candidate = resolved_candidate_cache.get(entity_id)
            if candidate is None:
                candidate = XrayPolymerEntityCandidateRecord(
                    polymer_entity_id=entity_id,
                    entry_id=str(entity_id).split("_", 1)[0],
                    chain_ids=(),
                    resolution_angstrom=float("nan"),
                )
            seen_rejected_entity_ids.add(entity_id)
            rejected_xray_homologs.append(
                RejectedXrayHomologRecord(
                    nmr_entry_id=seed.entry_id,
                    nmr_year=seed.year,
                    nmr_chain_id=seed.chain_id,
                    sequence_identity_percent=sequence_identity_percent,
                    nmr_core_start_seq_id=core_start,
                    nmr_core_end_seq_id=core_end,
                    nmr_query_sequence_length=len(query_sequence),
                    xray_entry_id=candidate.entry_id,
                    xray_entity_id=candidate.polymer_entity_id,
                    xray_chain_ids=candidate.chain_ids,
                    reason=rejection_reason_by_entity_id.get(
                        entity_id, XRAY_HOMOLOG_HETATM_REJECTION_REASON
                    ),
                )
            )
        xray_entry_ids = self._entry_ids_from_polymer_entity_ids(xray_entity_ids)
        return SolutionNMRMonomerXrayHomologRecord(
            entry_id=seed.entry_id,
            year=seed.year,
            sequence_identity_percent=sequence_identity_percent,
            nmr_core_start_seq_id=core_start,
            nmr_core_end_seq_id=core_end,
            nmr_query_sequence_length=len(query_sequence),
            xray_homolog_entry_ids=xray_entry_ids,
            xray_homolog_entity_ids=xray_entity_ids,
            has_xray_homolog=bool(xray_entity_ids),
            rejected_xray_homologs=tuple(rejected_xray_homologs),
        )

    def _build_record_pair(
        self,
        seed: SolutionNMRMonomerXrayHomologSeedRecord,
    ) -> tuple[
        SolutionNMRMonomerXrayHomologRecord, SolutionNMRMonomerXrayHomologRecord
    ]:
        """Build current and historical homolog records for one seed."""
        core_query = self._build_stride_core_query_sequence(seed)
        candidate_cache: dict[str, XrayPolymerEntityCandidateRecord] = {}
        chain_residue_cache: dict[tuple[str, str], tuple[CAResidueRecord, ...]] = {}
        return (
            self._build_record(
                seed,
                sequence_identity_percent=95,
                core_query=core_query,
                candidate_cache=candidate_cache,
                chain_residue_cache=chain_residue_cache,
            ),
            self._build_record(
                seed,
                sequence_identity_percent=100,
                core_query=core_query,
                candidate_cache=candidate_cache,
                chain_residue_cache=chain_residue_cache,
            ),
        )

    def build(
        self,
        on_record_pair: (
            Callable[
                [
                    SolutionNMRMonomerXrayHomologRecord,
                    SolutionNMRMonomerXrayHomologRecord,
                ],
                None,
            ]
            | None
        ) = None,
        skip_entry_ids: set[str] | None = None,
        on_entry_complete: Callable[[str, str], None] | None = None,
        retain_rejected_xray_homologs: bool = True,
    ) -> tuple[
        list[SolutionNMRMonomerXrayHomologRecord],
        list[SolutionNMRMonomerXrayHomologRecord],
    ]:
        """Collect X-ray homolog records for SOLUTION NMR monomer seeds.

        Streaming callers can disable retention of candidate-level rejection
        details after ``on_record_pair`` has persisted them.
        """
        skip_entry_ids = skip_entry_ids or set()
        entry_ids = fetch_solution_nmr_entry_ids(
            client=self.client,
            log_label="SOLUTION NMR monomer X-ray homologs",
        )
        batches = list(chunked(entry_ids, self.config.graphql_batch_size))
        seeds: list[SolutionNMRMonomerXrayHomologSeedRecord] = []

        for batch_seeds in collect_batch_results(
            batches=batches,
            max_workers=self.config.max_workers,
            fetch_fn=(
                self.client.fetch_solution_nmr_monomer_xray_homolog_seed_records_for_ids
            ),
            progress_label="SOLUTION NMR monomer X-ray homolog seeds",
        ):
            seeds.extend(batch_seeds)
        seeds = [seed for seed in seeds if seed.entry_id not in skip_entry_ids]
        LOGGER.info("SOLUTION NMR monomer X-ray homolog seeds: %d", len(seeds))

        records_95: list[SolutionNMRMonomerXrayHomologRecord] = []
        records_100: list[SolutionNMRMonomerXrayHomologRecord] = []
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_map = {
                executor.submit(self._build_record_pair, seed): seed for seed in seeds
            }
            attempt_by_entry_id = {seed.entry_id: 1 for seed in seeds}
            pending = set(future_map)
            total = len(pending)
            completed_count = 0
            error_count = 0
            rejected_candidate_count_95 = 0
            rejected_candidate_count_100 = 0
            exclusion_reason_counts: Counter[str] = Counter()
            last_progress_log = time.monotonic()

            while pending:
                done, pending = wait(
                    pending,
                    timeout=30.0,
                    return_when=FIRST_COMPLETED,
                )
                if not done:
                    LOGGER.info(
                        "SOLUTION NMR monomer X-ray homolog sequence searches: "
                        "processed %d/%d entries (95%% hits=%d, 100%% hits=%d, errors=%d)",
                        completed_count,
                        total,
                        sum(1 for record in records_95 if record.has_xray_homolog),
                        sum(1 for record in records_100 if record.has_xray_homolog),
                        error_count,
                    )
                    continue

                for future in done:
                    seed = future_map.pop(future)
                    try:
                        record_95, record_100 = future.result()
                    except NMRHomologyQueryIneligibleError as exc:
                        completed_count += 1
                        exclusion_reason_counts[exc.reason] += 1
                        _record_filtered_structure(
                            seed.entry_id, exc.reason, year=seed.year
                        )
                        LOGGER.info(
                            "Excluding NMR entry %s from X-ray homology: %s",
                            seed.entry_id,
                            exc.reason,
                        )
                        if on_entry_complete is not None:
                            on_entry_complete(seed.entry_id, "ineligible")
                        continue
                    except Exception as exc:
                        attempt = attempt_by_entry_id[seed.entry_id]
                        is_server_error = _is_http_server_error(exc)
                        if (
                            is_server_error
                            and attempt < XRAY_HOMOLOG_SEARCH_MAX_ATTEMPTS
                        ):
                            next_attempt = attempt + 1
                            attempt_by_entry_id[seed.entry_id] = next_attempt
                            retry_future = executor.submit(
                                self._build_record_pair, seed
                            )
                            future_map[retry_future] = seed
                            pending.add(retry_future)
                            LOGGER.warning(
                                "SOLUTION NMR monomer X-ray homolog sequence search "
                                "returned an HTTP 5xx error for %s; moved to the end "
                                "of the queue (next attempt %d/%d): %s",
                                seed.entry_id,
                                next_attempt,
                                XRAY_HOMOLOG_SEARCH_MAX_ATTEMPTS,
                                exc,
                            )
                            continue
                        completed_count += 1
                        error_count += 1
                        _record_filtered_structure(
                            seed.entry_id,
                            f"X-ray homology search failed: {exc}",
                            year=seed.year,
                        )
                        if is_server_error:
                            LOGGER.warning(
                                "SOLUTION NMR monomer X-ray homolog sequence search "
                                "failed for %s after %d attempts: %s",
                                seed.entry_id,
                                attempt,
                                exc,
                            )
                        else:
                            LOGGER.warning(
                                "SOLUTION NMR monomer X-ray homolog sequence search "
                                "failed for %s: %s",
                                seed.entry_id,
                                exc,
                            )
                        continue

                    completed_count += 1
                    rejected_candidate_count_95 += len(record_95.rejected_xray_homologs)
                    rejected_candidate_count_100 += len(
                        record_100.rejected_xray_homologs
                    )
                    records_95.append(
                        record_95
                        if retain_rejected_xray_homologs
                        else replace(record_95, rejected_xray_homologs=())
                    )
                    records_100.append(
                        record_100
                        if retain_rejected_xray_homologs
                        else replace(record_100, rejected_xray_homologs=())
                    )
                    if on_record_pair is not None:
                        on_record_pair(record_95, record_100)
                    if on_entry_complete is not None:
                        on_entry_complete(seed.entry_id, "success")

                    now = time.monotonic()
                    if (
                        completed_count % 25 == 0
                        or completed_count == total
                        or now - last_progress_log >= 30.0
                    ):
                        LOGGER.info(
                            "SOLUTION NMR monomer X-ray homolog sequence searches: "
                            "processed %d/%d entries (95%% hits=%d, 100%% hits=%d, errors=%d)",
                            completed_count,
                            total,
                            sum(1 for record in records_95 if record.has_xray_homolog),
                            sum(1 for record in records_100 if record.has_xray_homolog),
                            error_count,
                        )
                        last_progress_log = now

        LOGGER.info(
            "SOLUTION NMR monomer X-ray homologs: found X-ray hits (%d%%=%d/%d, %d%%=%d/%d); entries without a performed search=%d",
            95,
            sum(1 for record in records_95 if record.has_xray_homolog),
            len(records_95),
            100,
            sum(1 for record in records_100 if record.has_xray_homolog),
            len(records_100),
            sum(exclusion_reason_counts.values()),
        )
        LOGGER.info(
            "SOLUTION NMR monomer X-ray candidates rejected by method or "
            "modeled-core filters: %d%%=%d, %d%%=%d",
            95,
            rejected_candidate_count_95,
            100,
            rejected_candidate_count_100,
        )
        for reason, count in sorted(exclusion_reason_counts.items()):
            LOGGER.info(
                "SOLUTION NMR monomer X-ray homolog exclusions: %s=%d",
                reason,
                count,
            )

        def key_fn(record):
            """Return the stable chronological sort key for a homolog record."""
            return record.year, record.entry_id

        return sorted(records_95, key=key_fn), sorted(records_100, key=key_fn)
