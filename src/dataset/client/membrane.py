"""Collect membrane-annotated NMR candidates and their deposited metadata."""

from __future__ import annotations

from typing import Any

from src.dataset.api_cache import CachedJSONRequests
from src.dataset.config import LOGGER, MEMBRANE_ANNOTATION_TYPES, SOLUTION_NMR_METHOD


MEMBRANE_ENTRY_QUERY = """
query($ids: [String!]!) {
  entries(entry_ids: $ids) {
    rcsb_id
    struct { title }
    rcsb_accession_info { deposit_date initial_release_date revision_date }
    exptl { method }
    rcsb_entry_info { deposited_model_count polymer_entity_count_protein }
    rcsb_primary_citation {
      title year pdbx_database_id_DOI pdbx_database_id_PubMed
    }
    pdbx_database_related { db_id db_name details content_type }
    pdbx_nmr_ensemble {
      conformers_submitted_total_number representative_conformer
    }
    pdbx_nmr_representative { conformer_id selection_criteria }
    pdbx_nmr_sample_details {
      solution_id contents details label solvent_system type
    }
    assemblies {
      rcsb_id
      pdbx_struct_assembly_gen { asym_id_list oper_expression }
      rcsb_assembly_info { polymer_entity_instance_count_protein }
      rcsb_struct_symmetry { kind type symbol oligomeric_state stoichiometry }
    }
    polymer_entities {
      rcsb_id
      entity_poly {
        type rcsb_entity_polymer_type pdbx_seq_one_letter_code_can
        rcsb_sample_sequence_length
      }
      rcsb_polymer_entity { pdbx_description }
      rcsb_polymer_entity_container_identifiers {
        entity_id asym_ids auth_asym_ids uniprot_ids
      }
      rcsb_polymer_entity_annotation {
        type name annotation_id provenance_source
      }
      polymer_entity_instances {
        rcsb_id
        rcsb_polymer_entity_instance_container_identifiers {
          asym_id auth_asym_id entity_id
        }
        rcsb_polymer_instance_info { modeled_residue_count }
        rcsb_polymer_instance_feature {
          type name description assignment_version provenance_source
          feature_positions { beg_seq_id end_seq_id }
        }
      }
    }
  }
}
"""


def membrane_search_query() -> dict[str, Any]:
    """Require a protein entity, SOLUTION NMR membership and a membrane marker.

    Search membership permits hybrid methods; exact-method eligibility is
    assessed from the retrieved entry metadata by the dataset builder.
    """
    return {
        "type": "group",
        "logical_operator": "and",
        "nodes": [
            {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": attribute,
                    "operator": operator,
                    "value": value,
                },
            }
            for attribute, operator, value in (
                ("rcsb_entry_info.polymer_entity_count_protein", "greater_or_equal", 1),
                ("exptl.method", "exact_match", SOLUTION_NMR_METHOD),
                (
                    "rcsb_polymer_entity_annotation.type",
                    "in",
                    list(MEMBRANE_ANNOTATION_TYPES),
                ),
            )
        ],
    }


class MembraneEntriesMixin:
    """Retrieve candidate IDs and raw entry metadata through the API cache."""

    def fetch_solution_nmr_membrane_entry_ids(
        self, cache: CachedJSONRequests
    ) -> list[str]:
        """Fetch every candidate once, validating pagination completeness."""
        if self.config.page_size < 1:
            raise ValueError("RCSB search page size must be positive")
        all_ids: list[str] = []
        seen: set[str] = set()
        total_count: int | None = None
        while total_count is None or len(all_ids) < total_count:
            payload = {
                "query": membrane_search_query(),
                "return_type": "entry",
                "request_options": {
                    "paginate": {"start": len(all_ids), "rows": self.config.page_size},
                    "results_content_type": ["experimental"],
                    "sort": [{"sort_by": "rcsb_id", "direction": "asc"}],
                },
            }
            data = cache.post_json(self.config.search_url, payload)
            page_total = data.get("total_count")
            if (
                not isinstance(page_total, int)
                or isinstance(page_total, bool)
                or page_total < 0
            ):
                raise RuntimeError("RCSB search response has no valid total_count")
            if total_count is not None and total_count != page_total:
                raise RuntimeError("RCSB search total_count changed during pagination")
            total_count = page_total
            result_set = data.get("result_set", [])
            if not isinstance(result_set, list):
                raise RuntimeError("RCSB search result_set is not a list")
            batch_ids = [
                item.get("identifier") if isinstance(item, dict) else None
                for item in result_set
            ]
            if any(not isinstance(item, str) or not item.strip() for item in batch_ids):
                raise RuntimeError("RCSB search returned an invalid identifier")
            if len(batch_ids) != len(set(batch_ids)) or seen.intersection(batch_ids):
                raise RuntimeError("RCSB search repeated identifiers during pagination")
            if len(all_ids) + len(batch_ids) > total_count:
                raise RuntimeError(
                    "RCSB search returned more identifiers than total_count"
                )
            if not batch_ids and len(all_ids) < total_count:
                raise RuntimeError(
                    "RCSB search returned an empty page before completion"
                )
            all_ids.extend(batch_ids)
            seen.update(batch_ids)
            LOGGER.info(
                "SOLUTION NMR membrane candidates: fetched %d/%d IDs",
                len(all_ids),
                total_count,
            )
        return sorted(all_ids)

    def fetch_membrane_entries_for_ids(
        self, entry_ids: list[str], cache: CachedJSONRequests
    ) -> list[dict[str, Any]]:
        """Fetch one complete metadata batch; missing entries are retrieval errors."""
        if not entry_ids:
            return []
        if len(set(entry_ids)) != len(entry_ids):
            raise ValueError("Membrane metadata batch contains duplicate identifiers")
        payload = {
            "query": MEMBRANE_ENTRY_QUERY,
            "variables": {"ids": sorted(entry_ids)},
        }
        data = cache.post_json(self.config.graphql_url, payload)
        entries = (data.get("data") or {}).get("entries")
        if not isinstance(entries, list) or any(
            not isinstance(entry, dict) for entry in entries
        ):
            raise RuntimeError(
                "RCSB membrane metadata contains missing or invalid entries"
            )
        returned_ids = [entry.get("rcsb_id") for entry in entries]
        if any(not isinstance(entry_id, str) for entry_id in returned_ids):
            raise RuntimeError("RCSB membrane metadata has an invalid entry identifier")
        if len(set(returned_ids)) != len(returned_ids) or set(returned_ids) != set(
            entry_ids
        ):
            missing = sorted(set(entry_ids) - set(returned_ids))
            unexpected = sorted(set(returned_ids) - set(entry_ids))
            raise RuntimeError(
                f"RCSB membrane metadata IDs differ (missing={missing}, unexpected={unexpected})"
            )
        return sorted(entries, key=lambda entry: entry["rcsb_id"])
