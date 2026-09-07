"""Calculate STRIDE coverage records for solution NMR monomers."""

from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from ..config import LOGGER
from ..coordinates import parse_first_model_modeled_ca_auth_seq_ids
from ..downloads import download_pdb_if_needed
from ..records import SolutionNMRMonomerStrideModeledFirstModelRecord
from ..reporting import _record_filtered_structure
from ..stride import compute_stride_state_coverages_for_chain_modeled_first_model
from ..structures import load_cached_chain_id_map
from ..utils import _record_entries_missing_from_response


class SolutionNMRStrideMixin:
    """Build first-model secondary-structure records for eligible NMR entries."""

    def iter_solution_nmr_monomer_stride_modeled_first_model_records_for_ids(
        self,
        entry_ids: list[str],
        stride_executable: str,
        pdb_cache_dir: Path,
        stride_cache_dir: Path,
    ) -> Iterator[SolutionNMRMonomerStrideModeledFirstModelRecord]:
        """Yield STRIDE first-model monomer records for SOLUTION NMR entries."""
        query = """
        query($ids:[String!]!) {
          entries(entry_ids:$ids) {
            rcsb_id
            rcsb_entry_info {
              deposited_model_count
            }
            rcsb_accession_info {
              deposit_date
            }
            polymer_entities {
              entity_poly {
                type
                rcsb_entity_polymer_type
                pdbx_strand_id
              }
              polymer_entity_instances {
                rcsb_id
              }
            }
          }
        }
        """
        payload = {"query": query, "variables": {"ids": entry_ids}}
        data = self._post_json(self.config.graphql_url, payload)
        entries = data.get("data", {}).get("entries") or []
        _record_entries_missing_from_response(entry_ids, entries)

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_map = {
                executor.submit(
                    self._compute_solution_nmr_monomer_stride_modeled_first_model_for_entry,
                    entry=entry,
                    stride_executable=stride_executable,
                    pdb_cache_dir=pdb_cache_dir,
                    stride_cache_dir=stride_cache_dir,
                ): idx
                for idx, entry in enumerate(entries, start=1)
            }
            for future in as_completed(future_map):
                record = future.result()
                if record is None:
                    continue
                yield record

    def _compute_solution_nmr_monomer_stride_modeled_first_model_for_entry(
        self,
        entry: dict[str, Any] | None,
        stride_executable: str,
        pdb_cache_dir: Path,
        stride_cache_dir: Path,
    ) -> SolutionNMRMonomerStrideModeledFirstModelRecord | None:
        """Compute one entry-level STRIDE modeled-first-model record."""
        if not entry:
            return None
        context = self._extract_solution_nmr_monomer_context(entry)
        if context is None:
            return None
        entry_id, year, _, polymer_entity, chain_id = context

        instances = polymer_entity.get("polymer_entity_instances") or []
        if len(instances) != 1:
            _record_filtered_structure(
                entry_id,
                f"expected exactly 1 polymer entity instance (found {len(instances)})",
                year=year,
            )
            return None
        try:
            pdb_path = download_pdb_if_needed(
                session=self.session,
                config=self.config,
                cache_dir=pdb_cache_dir,
                entry_id=entry_id,
            )
            chain_map = load_cached_chain_id_map(pdb_cache_dir, entry_id)
            parsed_chain_id = chain_map.get(chain_id, chain_id)
            modeled_auth_seq_ids = parse_first_model_modeled_ca_auth_seq_ids(
                pdb_path=pdb_path,
                chain_id=parsed_chain_id,
            )
        except Exception as exc:
            LOGGER.debug(
                "Skipping STRIDE modeled-first-model entry %s: %s",
                entry_id,
                exc,
            )
            _record_filtered_structure(
                entry_id, f"coordinate preparation failed: {exc}", year=year
            )
            return None

        modeled_sequence_length = len(modeled_auth_seq_ids)
        if modeled_sequence_length <= 0:
            _record_filtered_structure(
                entry_id, "no usable first-model modeled CA residues", year=year
            )
            return None
        modeled_start_seq_id = min(modeled_auth_seq_ids)
        modeled_end_seq_id = max(modeled_auth_seq_ids)

        try:
            stride_coverages, _, _ = (
                compute_stride_state_coverages_for_chain_modeled_first_model(
                    session=self.session,
                    config=self.config,
                    cache_dir=pdb_cache_dir,
                    stride_cache_dir=stride_cache_dir,
                    entry_id=entry_id,
                    chain_id=chain_id,
                    modeled_sequence_length=modeled_sequence_length,
                    modeled_auth_seq_ids=modeled_auth_seq_ids,
                    stride_executable=stride_executable,
                )
            )
        except Exception as exc:
            LOGGER.warning(
                "Skipping STRIDE modeled-first-model entry %s: %s", entry_id, exc
            )
            _record_filtered_structure(
                entry_id, f"STRIDE calculation failed: {exc}", year=year
            )
            return None
        stride_coil_fraction = stride_coverages["C"]
        stride_secondary_percent = (1.0 - stride_coil_fraction) * 100.0

        return SolutionNMRMonomerStrideModeledFirstModelRecord(
            entry_id=entry_id,
            year=year,
            chain_id=chain_id,
            modeled_start_seq_id=modeled_start_seq_id,
            modeled_end_seq_id=modeled_end_seq_id,
            modeled_sequence_length=modeled_sequence_length,
            stride_alpha_helix_fraction=stride_coverages["H"],
            stride_3_10_helix_fraction=stride_coverages["G"],
            stride_pi_helix_fraction=stride_coverages["I"],
            stride_beta_strand_fraction=stride_coverages["E"],
            stride_isolated_beta_bridge_fraction=stride_coverages["B"],
            stride_turn_fraction=stride_coverages["T"],
            stride_coil_fraction=stride_coil_fraction,
            stride_secondary_structure_percent=stride_secondary_percent,
        )

    def fetch_solution_nmr_monomer_stride_modeled_first_model_records_for_ids(
        self,
        entry_ids: list[str],
        stride_executable: str,
        pdb_cache_dir: Path,
        stride_cache_dir: Path,
    ) -> list[SolutionNMRMonomerStrideModeledFirstModelRecord]:
        """Fetch all STRIDE modeled-first-model records for SOLUTION NMR monomers."""
        return list(
            self.iter_solution_nmr_monomer_stride_modeled_first_model_records_for_ids(
                entry_ids=entry_ids,
                stride_executable=stride_executable,
                pdb_cache_dir=pdb_cache_dir,
                stride_cache_dir=stride_cache_dir,
            )
        )
