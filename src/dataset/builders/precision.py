"""Measure NMR ensemble precision over modeled STRIDE cores."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from src.dataset.config import (
    LOGGER,
)
from src.dataset.coordinates import (
    parse_first_model_modeled_ca_auth_seq_ids,
    parse_models_ca_coords_with_stats,
)
from src.dataset.downloads import (
    download_pdb_if_needed,
)
from src.dataset.geometry import (
    _ca_rmsd_to_mean_structure,
    _coordinates_aligned_to_first_model,
)
from src.dataset.records import (
    ResidueId,
    SolutionNMRMonomerPrecisionRecord,
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
        SolutionNMRMonomerModeledFirstModelSeedRecord,
    )


class SolutionNMRMonomerPrecisionBuilder:
    """Compute ensemble precision for solution-NMR monomer cores."""

    def __init__(
        self,
        client: RCSBClient,
        config: DatasetBuildConfig,
        cache_dir: Path,
        precision_workers: int,
    ) -> None:
        """Initialize shared state for monomer precision builders."""
        self.client = client
        self.config = config
        self.cache_dir = cache_dir
        self.precision_workers = max(1, precision_workers)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _download_pdb_if_needed(self, entry_id: str) -> Path:
        """Download or reuse the PDB file needed for precision calculations."""
        return download_pdb_if_needed(
            session=self.client.session,
            config=self.config,
            cache_dir=self.cache_dir,
            entry_id=entry_id,
        )

    @staticmethod
    def _compute_mean_rmsd_to_average(
        pdb_path: Path,
        chain_id: str,
        start_seq_id: ResidueId | int,
        end_seq_id: ResidueId | int,
    ) -> tuple[tuple[int, int, int, float] | None, str | None]:
        """Compute ensemble CA RMSD to per-residue mean coordinates."""
        model_maps, raw_ca_counts_per_model = parse_models_ca_coords_with_stats(
            pdb_path=pdb_path,
            chain_id=chain_id,
            start_seq_id=start_seq_id,
            end_seq_id=end_seq_id,
        )
        if len(model_maps) < 2:
            return (
                None,
                f"fewer than 2 coordinate models in core range (found {len(model_maps)})",
            )

        common_resids = set(model_maps[0].keys())
        for model_map in model_maps[1:]:
            common_resids &= set(model_map.keys())
        if len(common_resids) < 3:
            return (
                None,
                (
                    "fewer than 3 CA residues common to all models in core range "
                    f"(found {len(common_resids)})"
                ),
            )
        sorted_resids = sorted(common_resids)

        coords = np.asarray(
            [[model_map[resid] for resid in sorted_resids] for model_map in model_maps],
            dtype=float,
        )
        aligned_coords = _coordinates_aligned_to_first_model(coords)
        ensemble_rmsd = _ca_rmsd_to_mean_structure(aligned_coords)

        raw_ca_counts_common = [
            sum(raw_counts.get(resid, 0) for resid in sorted_resids)
            for raw_counts in raw_ca_counts_per_model
        ]
        n_ca_core_raw = (
            min(raw_ca_counts_common) if raw_ca_counts_common else len(sorted_resids)
        )
        return (
            (
                len(model_maps),
                len(sorted_resids),
                int(n_ca_core_raw),
                ensemble_rmsd,
            ),
            None,
        )

    def _build_record_from_core_range(
        self,
        pdb_path: Path,
        entry_id: str,
        year: int,
        chain_id: str,
        core_start_seq_id: ResidueId | int,
        core_end_seq_id: ResidueId | int,
        parsed_chain_id: str | None = None,
    ) -> SolutionNMRMonomerPrecisionRecord | None:
        """Build a precision record from a modeled residue core range."""
        result, skip_reason = self._compute_mean_rmsd_to_average(
            pdb_path=pdb_path,
            chain_id=parsed_chain_id or chain_id,
            start_seq_id=core_start_seq_id,
            end_seq_id=core_end_seq_id,
        )
        if result is None:
            LOGGER.info(
                "Skipping precision entry %s chain %s: %s",
                entry_id,
                chain_id,
                skip_reason,
            )
            _record_filtered_structure(
                entry_id,
                f"precision calculation rejected the structural core: {skip_reason}",
                year=year,
            )
            return None
        n_models, n_ca_core_used, n_ca_core_raw, mean_rmsd = result
        return SolutionNMRMonomerPrecisionRecord(
            entry_id=entry_id,
            year=year,
            chain_id=chain_id,
            core_start_seq_id=core_start_seq_id,
            core_end_seq_id=core_end_seq_id,
            n_models=n_models,
            n_ca_core_used=n_ca_core_used,
            n_ca_core_raw=n_ca_core_raw,
            mean_rmsd_angstrom=mean_rmsd,
        )


class SolutionNMRMonomerPrecisionStrideModeledFirstModelBuilder(
    SolutionNMRMonomerPrecisionBuilder
):
    """Build precision records using STRIDE-defined modeled core ranges."""

    def __init__(
        self,
        client: RCSBClient,
        config: DatasetBuildConfig,
        cache_dir: Path,
        precision_workers: int,
        stride_executable: str,
        stride_cache_dir: Path,
    ) -> None:
        """Initialize STRIDE-core modeled-first-model precision collection."""
        super().__init__(
            client=client,
            config=config,
            cache_dir=cache_dir,
            precision_workers=precision_workers,
        )
        self.stride_executable = stride_executable
        self.stride_cache_dir = stride_cache_dir
        self.stride_cache_dir.mkdir(parents=True, exist_ok=True)

    def _compute_record_from_seed(
        self,
        seed: SolutionNMRMonomerModeledFirstModelSeedRecord,
    ) -> SolutionNMRMonomerPrecisionRecord | None:
        """Compute one STRIDE-core precision record from seed metadata."""
        try:
            pdb_path = self._download_pdb_if_needed(seed.entry_id)
            chain_map = load_cached_chain_id_map(self.cache_dir, seed.entry_id)
            parsed_chain_id = chain_map.get(seed.chain_id, seed.chain_id)
            modeled_auth_seq_ids = parse_first_model_modeled_ca_auth_seq_ids(
                pdb_path=pdb_path,
                chain_id=parsed_chain_id,
            )
            if not modeled_auth_seq_ids:
                LOGGER.info(
                    "Skipping precision entry %s chain %s: no usable first-model modeled CA residues",
                    seed.entry_id,
                    seed.chain_id,
                )
                _record_filtered_structure(
                    seed.entry_id,
                    "no usable first-model modeled CA residues",
                    year=seed.year,
                )
                return None
            core_range = (
                compute_stride_core_range_for_modeled_auth_seq_ids_in_first_model(
                    pdb_path=pdb_path,
                    entry_id=seed.entry_id,
                    chain_id=parsed_chain_id,
                    modeled_auth_seq_ids=modeled_auth_seq_ids,
                    stride_executable=self.stride_executable,
                    stride_cache_dir=self.stride_cache_dir,
                )
            )
            if core_range is None:
                LOGGER.info(
                    "Skipping precision entry %s chain %s: STRIDE found no modeled core residues",
                    seed.entry_id,
                    seed.chain_id,
                )
                _record_filtered_structure(
                    seed.entry_id,
                    "STRIDE found no modeled core residues",
                    year=seed.year,
                )
                return None
            core_start, core_end = core_range
            return self._build_record_from_core_range(
                pdb_path=pdb_path,
                entry_id=seed.entry_id,
                year=seed.year,
                chain_id=seed.chain_id,
                core_start_seq_id=core_start,
                core_end_seq_id=core_end,
                parsed_chain_id=parsed_chain_id,
            )
        except Exception as exc:
            LOGGER.warning(
                "STRIDE modeled-first-model precision calculation failed for %s: %s",
                seed.entry_id,
                exc,
            )
            _record_filtered_structure(
                seed.entry_id,
                f"precision calculation failed: {exc}",
                year=seed.year,
            )
            return None

    def build(
        self,
        skip_entry_ids: set[str] | None = None,
        on_record: Callable[[SolutionNMRMonomerPrecisionRecord], None] | None = None,
    ) -> list[SolutionNMRMonomerPrecisionRecord]:
        """Collect STRIDE-core precision records for all eligible seeds."""
        skip_entry_ids = skip_entry_ids or set()
        entry_ids = fetch_solution_nmr_entry_ids(
            client=self.client,
            log_label="SOLUTION NMR precision STRIDE modeled-first-model",
        )

        batches = list(chunked(entry_ids, self.config.graphql_batch_size))
        seed_records: list[SolutionNMRMonomerModeledFirstModelSeedRecord] = []
        for batch_seeds in collect_batch_results(
            batches=batches,
            max_workers=self.config.max_workers,
            fetch_fn=(
                self.client.fetch_solution_nmr_monomer_modeled_first_model_seed_records_for_ids
            ),
            progress_label="SOLUTION NMR precision STRIDE modeled-first-model seeds",
        ):
            seed_records.extend(batch_seeds)

        filtered_seeds = [
            record for record in seed_records if record.entry_id not in skip_entry_ids
        ]
        filtered_seeds = sorted(filtered_seeds, key=lambda r: (r.year, r.entry_id))
        LOGGER.info(
            "SOLUTION NMR precision STRIDE modeled-first-model: entries to process after filters: %d",
            len(filtered_seeds),
        )

        precision_records: list[SolutionNMRMonomerPrecisionRecord] = []
        with ThreadPoolExecutor(max_workers=self.precision_workers) as executor:
            future_map = {
                executor.submit(self._compute_record_from_seed, seed): idx
                for idx, seed in enumerate(filtered_seeds, start=1)
            }
            total = len(future_map)
            for future in as_completed(future_map):
                record = future.result()
                if record is not None:
                    precision_records.append(record)
                    if on_record is not None:
                        on_record(record)
                idx = future_map[future]
                if total > 0 and (idx % 50 == 0 or idx == total):
                    LOGGER.info(
                        "SOLUTION NMR precision STRIDE modeled-first-model RMSD: processed %d/%d entries",
                        idx,
                        total,
                    )

        return sorted(precision_records, key=lambda r: (r.year, r.entry_id))
