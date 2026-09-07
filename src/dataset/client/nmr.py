"""Validate solution NMR monomers and retrieve their dataset records."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import LOGGER, PROTEIN_MONOMER_ENTITY_TYPES, PROTEIN_POLYMER_TYPE
from ..coordinates import parse_models_ca_coords_with_stats
from ..downloads import download_pdb_if_needed
from ..records import (
    SolutionNMRMonomerExperimentsRecord,
    SolutionNMRMonomerModeledFirstModelSeedRecord,
    SolutionNMRMonomerQualityRecord,
    SolutionNMRMonomerXrayHomologSeedRecord,
    SolutionNMRWeightRecord,
)
from ..reporting import _record_filtered_structure
from ..structures import load_cached_chain_id_map
from ..utils import _record_entries_missing_from_response, extract_year


class SolutionNMRMixin:
    """Fetch NMR metadata and enforce the common monomer eligibility checks."""

    def _extract_solution_nmr_monomer_context(
        self,
        entry: dict[str, Any],
    ) -> tuple[str, int, int, dict[str, Any], str] | None:
        """Extract context for a monomer whose models have equal chain lengths."""
        entry_id = entry.get("rcsb_id")
        if not entry_id:
            return None
        entry_id = str(entry_id)
        year = extract_year(
            (entry.get("rcsb_accession_info") or {}).get("deposit_date")
        )

        model_count_raw = (entry.get("rcsb_entry_info") or {}).get(
            "deposited_model_count"
        )
        if model_count_raw is None:
            _record_filtered_structure(
                entry_id, "deposited model count is missing", year=year
            )
            return None
        try:
            model_count = int(model_count_raw)
        except (TypeError, ValueError):
            _record_filtered_structure(
                entry_id, "deposited model count is invalid", year=year
            )
            return None
        if model_count <= 1:
            _record_filtered_structure(
                entry_id, "fewer than 2 deposited models", year=year
            )
            return None

        if year is None:
            _record_filtered_structure(entry_id, "deposit year is missing or invalid")
            return None

        polymer_entities = entry.get("polymer_entities") or []
        if len(polymer_entities) != 1:
            _record_filtered_structure(
                entry_id,
                f"expected exactly 1 polymer entity (found {len(polymer_entities)})",
                year=year,
            )
            return None
        polymer_entity = polymer_entities[0] or {}
        entity_poly = polymer_entity.get("entity_poly") or {}

        if entity_poly.get("type") not in PROTEIN_MONOMER_ENTITY_TYPES:
            _record_filtered_structure(
                entry_id,
                "polymer entity type is not polypeptide(L) or polypeptide(D)",
                year=year,
            )
            return None
        if entity_poly.get("rcsb_entity_polymer_type") != PROTEIN_POLYMER_TYPE:
            _record_filtered_structure(
                entry_id, "polymer entity is not a protein", year=year
            )
            return None

        chain_id = str(entity_poly.get("pdbx_strand_id") or "").strip()
        if not chain_id or "," in chain_id:
            _record_filtered_structure(
                entry_id,
                "polymer entity does not have exactly 1 chain ID",
                year=year,
            )
            return None

        if not self._solution_nmr_monomer_models_have_equal_lengths(
            entry_id=entry_id,
            chain_id=chain_id,
        ):
            _record_filtered_structure(
                entry_id,
                "coordinate models do not have equal full-chain lengths",
                year=year,
            )
            return None

        return entry_id, year, model_count, polymer_entity, chain_id

    def _download_solution_nmr_monomer_pdb_if_needed(self, entry_id: str) -> Path:
        """Download or reuse coordinates needed by the base monomer filter."""
        return download_pdb_if_needed(
            session=self.session,
            config=self.config,
            cache_dir=self.solution_nmr_monomer_cache_dir,
            entry_id=entry_id,
        )

    def _solution_nmr_monomer_models_have_equal_lengths(
        self,
        entry_id: str,
        chain_id: str,
    ) -> bool:
        """Return whether every coordinate model has the same full-chain length."""
        try:
            pdb_path = self._download_solution_nmr_monomer_pdb_if_needed(entry_id)
            chain_map = load_cached_chain_id_map(
                self.solution_nmr_monomer_cache_dir,
                entry_id,
            )
            parsed_chain_id = chain_map.get(chain_id, chain_id)
            model_maps, _ = parse_models_ca_coords_with_stats(
                pdb_path=pdb_path,
                chain_id=parsed_chain_id,
            )
        except Exception as exc:
            LOGGER.warning(
                "Skipping SOLUTION NMR monomer %s: model-length check failed: %s",
                entry_id,
                exc,
            )
            return False

        if len(model_maps) < 2:
            LOGGER.info(
                "Skipping SOLUTION NMR monomer %s chain %s: fewer than 2 coordinate models",
                entry_id,
                chain_id,
            )
            return False

        model_lengths = [len(model_map) for model_map in model_maps]
        if len(set(model_lengths)) != 1:
            LOGGER.info(
                "Skipping SOLUTION NMR monomer %s chain %s: coordinate models have different full-chain lengths (%s)",
                entry_id,
                chain_id,
                model_lengths,
            )
            return False
        return True

    def fetch_solution_nmr_weight_records_for_ids(
        self, entry_ids: list[str]
    ) -> list[SolutionNMRWeightRecord]:
        """Build molecular-weight records for SOLUTION NMR entries."""
        query = """
        query($ids:[String!]!) {
          entries(entry_ids:$ids) {
            rcsb_id
            rcsb_accession_info {
              deposit_date
            }
            rcsb_entry_info {
              molecular_weight
            }
          }
        }
        """
        payload = {"query": query, "variables": {"ids": entry_ids}}
        data = self._post_json(self.config.graphql_url, payload)
        entries = data.get("data", {}).get("entries") or []
        _record_entries_missing_from_response(entry_ids, entries)

        records: list[SolutionNMRWeightRecord] = []
        for entry in entries:
            if not entry:
                continue
            entry_id = entry.get("rcsb_id")
            year = extract_year(
                (entry.get("rcsb_accession_info") or {}).get("deposit_date")
            )
            if not entry_id:
                continue
            if year is None:
                _record_filtered_structure(
                    entry_id, "deposit year is missing or invalid", year=year
                )
                continue
            entry_mw_raw = (entry.get("rcsb_entry_info") or {}).get("molecular_weight")
            try:
                molecular_weight_kda = float(entry_mw_raw)
            except (TypeError, ValueError):
                _record_filtered_structure(
                    entry_id,
                    "entry molecular weight is missing or invalid",
                    year=year,
                )
                continue

            records.append(
                SolutionNMRWeightRecord(
                    entry_id=entry_id,
                    year=year,
                    molecular_weight_kda=molecular_weight_kda,
                )
            )
        return records

    def fetch_solution_nmr_monomer_experiment_records_for_ids(
        self, entry_ids: list[str]
    ) -> list[SolutionNMRMonomerExperimentsRecord]:
        """Fetch NMR experiments for eligible monomeric SOLUTION NMR entries."""
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
            }
            pdbx_nmr_exptl {
              type
            }
          }
        }
        """
        payload = {"query": query, "variables": {"ids": entry_ids}}
        data = self._post_json(self.config.graphql_url, payload)
        entries = data.get("data", {}).get("entries") or []
        _record_entries_missing_from_response(entry_ids, entries)

        records: list[SolutionNMRMonomerExperimentsRecord] = []
        for entry in entries:
            if not entry:
                continue
            context = self._extract_solution_nmr_monomer_context(entry)
            if context is None:
                continue
            entry_id, year, _, _, _ = context
            experiments = tuple(
                str(item.get("type")).strip()
                for item in entry.get("pdbx_nmr_exptl") or []
                if item and item.get("type")
            )
            records.append(
                SolutionNMRMonomerExperimentsRecord(
                    entry_id=entry_id,
                    year=year,
                    nmr_experiments_conducted=experiments,
                )
            )
        return records

    def fetch_solution_nmr_monomer_quality_records_for_ids(
        self, entry_ids: list[str]
    ) -> list[SolutionNMRMonomerQualityRecord]:
        """Fetch validation quality metrics for SOLUTION NMR monomers."""
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
            pdbx_vrpt_summary_geometry {
              clashscore
              percent_ramachandran_outliers
              percent_rotamer_outliers
            }
            polymer_entities {
              entity_poly {
                type
                rcsb_entity_polymer_type
                pdbx_strand_id
              }
            }
          }
        }
        """
        payload = {"query": query, "variables": {"ids": entry_ids}}
        data = self._post_json(self.config.graphql_url, payload)
        entries = data.get("data", {}).get("entries") or []
        _record_entries_missing_from_response(entry_ids, entries)
        records: list[SolutionNMRMonomerQualityRecord] = []

        for entry in entries:
            if not entry:
                continue
            context = self._extract_solution_nmr_monomer_context(entry)
            if context is None:
                continue
            entry_id, year, _, _, _ = context
            quality_items = entry.get("pdbx_vrpt_summary_geometry") or []
            if not quality_items:
                _record_filtered_structure(
                    entry_id, "validation quality metrics are missing", year=year
                )
                continue
            quality = quality_items[0] or {}
            clashscore = quality.get("clashscore")
            rama = quality.get("percent_ramachandran_outliers")
            rotamer = quality.get("percent_rotamer_outliers")
            if clashscore is None or rama is None or rotamer is None:
                _record_filtered_structure(
                    entry_id,
                    "one or more validation quality metrics are missing",
                    year=year,
                )
                continue
            try:
                clashscore_value = float(clashscore)
                rama_value = float(rama)
                rotamer_value = float(rotamer)
            except (TypeError, ValueError):
                _record_filtered_structure(
                    entry_id,
                    "one or more validation quality metrics are invalid",
                    year=year,
                )
                continue

            records.append(
                SolutionNMRMonomerQualityRecord(
                    entry_id=entry_id,
                    year=year,
                    clashscore=clashscore_value,
                    ramachandran_outliers_percent=rama_value,
                    sidechain_outliers_percent=rotamer_value,
                )
            )
        return records

    def fetch_solution_nmr_monomer_modeled_first_model_seed_records_for_ids(
        self, entry_ids: list[str]
    ) -> list[SolutionNMRMonomerModeledFirstModelSeedRecord]:
        """Fetch seed data needed for modeled-first-model precision analysis."""
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
            }
          }
        }
        """
        payload = {"query": query, "variables": {"ids": entry_ids}}
        data = self._post_json(self.config.graphql_url, payload)
        entries = data.get("data", {}).get("entries") or []
        _record_entries_missing_from_response(entry_ids, entries)
        records: list[SolutionNMRMonomerModeledFirstModelSeedRecord] = []

        for entry in entries:
            if not entry:
                continue
            context = self._extract_solution_nmr_monomer_context(entry)
            if context is None:
                continue
            entry_id, year, _, _, chain_id = context

            records.append(
                SolutionNMRMonomerModeledFirstModelSeedRecord(
                    entry_id=str(entry_id),
                    year=year,
                    chain_id=chain_id,
                )
            )
        return records

    def fetch_solution_nmr_monomer_xray_homolog_seed_records_for_ids(
        self, entry_ids: list[str]
    ) -> list[SolutionNMRMonomerXrayHomologSeedRecord]:
        """Fetch seed data needed to search X-ray homologs for NMR monomers."""
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
            }
          }
        }
        """
        payload = {"query": query, "variables": {"ids": entry_ids}}
        data = self._post_json(self.config.graphql_url, payload)
        entries = data.get("data", {}).get("entries") or []
        _record_entries_missing_from_response(entry_ids, entries)
        records: list[SolutionNMRMonomerXrayHomologSeedRecord] = []

        for entry in entries:
            if not entry:
                continue
            context = self._extract_solution_nmr_monomer_context(entry)
            if context is None:
                continue
            entry_id, year, _, _, chain_id = context
            records.append(
                SolutionNMRMonomerXrayHomologSeedRecord(
                    entry_id=str(entry_id),
                    year=year,
                    chain_id=chain_id,
                )
            )
        return records
