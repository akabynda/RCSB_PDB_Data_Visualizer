"""Search sequence homologs and retrieve polymer-entity group mappings."""

from __future__ import annotations

import time
from collections.abc import Iterable
from typing import Any

import requests

from ..config import LOGGER, SEQUENCE_IDENTITY_AGGREGATION_METHOD
from ..records import XrayEntityGroupMappingRecord, XrayPolymerEntityCandidateRecord
from ..utils import chunked


class SequenceHomologyMixin:
    """Find X-ray polymer entities using sequence identity and group membership."""

    @staticmethod
    def _normalize_similarity_cutoff(raw_cutoff: Any) -> int | None:
        """Normalize raw sequence-identity cutoff values to integer percentages."""
        try:
            return int(round(float(raw_cutoff)))
        except (OverflowError, TypeError, ValueError):
            return None

    @classmethod
    def _extract_sequence_identity_groups(
        cls,
        memberships: Iterable[dict[str, Any] | None],
        allowed_cutoffs: set[int] | None = None,
    ) -> dict[int, str]:
        """Extract matching sequence-identity group IDs from GraphQL entity data."""
        groups: dict[int, str] = {}
        for membership in memberships:
            if not membership:
                continue
            if (
                membership.get("aggregation_method")
                != SEQUENCE_IDENTITY_AGGREGATION_METHOD
            ):
                continue
            cutoff = cls._normalize_similarity_cutoff(
                membership.get("similarity_cutoff")
            )
            group_id = membership.get("group_id")
            if cutoff is None or not group_id:
                continue
            if allowed_cutoffs is not None and cutoff not in allowed_cutoffs:
                continue
            groups[cutoff] = str(group_id)
        return groups

    def fetch_xray_polymer_entity_ids_for_group_ids(
        self, group_ids: list[str]
    ) -> list[str]:
        """Fetch X-ray polymer entity IDs for sequence-identity group IDs."""
        if not group_ids:
            return []
        query = {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "exptl.method",
                        "operator": "exact_match",
                        "value": "X-RAY DIFFRACTION",
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_polymer_entity_group_membership.group_id",
                        "operator": "in",
                        "value": group_ids,
                    },
                },
            ],
        }
        return self._fetch_paginated_identifiers(
            query=query,
            return_type="polymer_entity",
        )

    def fetch_polymer_entity_group_mapping_for_ids(
        self, entity_ids: list[str], similarity_cutoff: int
    ) -> list[XrayEntityGroupMappingRecord]:
        """Fetch sequence-identity group mappings for polymer entity IDs."""
        if not entity_ids:
            return []
        query = """
        query($ids:[String!]!) {
          polymer_entities(entity_ids:$ids) {
            rcsb_id
            entity_poly {
              pdbx_strand_id
            }
            rcsb_polymer_entity_container_identifiers {
              entry_id
            }
            rcsb_polymer_entity_group_membership {
              aggregation_method
              similarity_cutoff
              group_id
            }
          }
        }
        """
        payload = {"query": query, "variables": {"ids": entity_ids}}
        data = self._post_json(self.config.graphql_url, payload)
        entities = data.get("data", {}).get("polymer_entities") or []

        records: list[XrayEntityGroupMappingRecord] = []
        for entity in entities:
            if not entity:
                continue
            polymer_entity_id = entity.get("rcsb_id")
            entry_id = (
                entity.get("rcsb_polymer_entity_container_identifiers", {}) or {}
            ).get("entry_id")
            chain_ids = str(
                (entity.get("entity_poly", {}) or {}).get("pdbx_strand_id") or ""
            ).strip()
            chain_id_tuple = tuple(
                item.strip() for item in chain_ids.split(",") if item.strip()
            )
            if not polymer_entity_id or not entry_id or not chain_id_tuple:
                continue

            memberships = entity.get("rcsb_polymer_entity_group_membership") or []
            matched_group_id: str | None = None
            for membership in memberships:
                if not membership:
                    continue
                if (
                    membership.get("aggregation_method")
                    != SEQUENCE_IDENTITY_AGGREGATION_METHOD
                ):
                    continue
                group_id = membership.get("group_id")
                cutoff = self._normalize_similarity_cutoff(
                    membership.get("similarity_cutoff")
                )
                if cutoff is None or not group_id:
                    continue
                if cutoff == similarity_cutoff:
                    matched_group_id = str(group_id)
                    break
            if not matched_group_id:
                continue
            records.append(
                XrayEntityGroupMappingRecord(
                    polymer_entity_id=str(polymer_entity_id),
                    entry_id=str(entry_id),
                    chain_ids=chain_id_tuple,
                    group_id=matched_group_id,
                )
            )
        return records

    def fetch_xray_polymer_entity_candidates_for_ids(
        self,
        entity_ids: list[str],
    ) -> list[XrayPolymerEntityCandidateRecord]:
        """Fetch candidate X-ray polymer entities and optional resolution metadata."""
        if not entity_ids:
            return []
        query = """
        query($ids:[String!]!) {
          polymer_entities(entity_ids:$ids) {
            rcsb_id
            entity_poly {
              pdbx_strand_id
            }
            rcsb_polymer_entity_container_identifiers {
              entry_id
            }
          }
        }
        """
        payload = {"query": query, "variables": {"ids": entity_ids}}
        data = self._post_json(self.config.graphql_url, payload)
        entities = data.get("data", {}).get("polymer_entities") or []

        entity_rows: list[tuple[str, str, tuple[str, ...]]] = []
        for entity in entities:
            if not entity:
                continue
            polymer_entity_id = entity.get("rcsb_id")
            entry_id = (
                entity.get("rcsb_polymer_entity_container_identifiers", {}) or {}
            ).get("entry_id")
            chain_ids = str(
                (entity.get("entity_poly", {}) or {}).get("pdbx_strand_id") or ""
            ).strip()
            chain_id_tuple = tuple(
                item.strip() for item in chain_ids.split(",") if item.strip()
            )
            if not polymer_entity_id or not entry_id or not chain_id_tuple:
                continue
            entity_rows.append((str(polymer_entity_id), str(entry_id), chain_id_tuple))

        resolution_by_entry_id: dict[str, float] = {}
        methods_by_entry_id: dict[str, tuple[str, ...]] = {}
        unknown_entries = sorted({entry_id for _, entry_id, _ in entity_rows})
        for entry_batch in chunked(unknown_entries, self.config.graphql_batch_size):
            resolution_by_entry_id.update(
                self.fetch_entry_resolution_for_ids(entry_batch)
            )
            methods_by_entry_id.update(
                self.fetch_entry_experimental_methods_for_ids(entry_batch)
            )

        records: list[XrayPolymerEntityCandidateRecord] = []
        for polymer_entity_id, entry_id, chain_ids in entity_rows:
            resolution = resolution_by_entry_id.get(entry_id)
            records.append(
                XrayPolymerEntityCandidateRecord(
                    polymer_entity_id=polymer_entity_id,
                    entry_id=entry_id,
                    chain_ids=chain_ids,
                    resolution_angstrom=(
                        resolution if resolution is not None else float("nan")
                    ),
                    experimental_methods=methods_by_entry_id.get(entry_id, ()),
                )
            )
        return records

    def fetch_sequence_identity_group_ids_for_polymer_entity_ids(
        self, entity_ids: list[str], similarity_cutoff: int
    ) -> set[str]:
        """Fetch sequence-identity group IDs for polymer entity IDs."""
        if not entity_ids:
            return set()
        query = """
        query($ids:[String!]!) {
          polymer_entities(entity_ids:$ids) {
            rcsb_polymer_entity_group_membership {
              aggregation_method
              similarity_cutoff
              group_id
            }
          }
        }
        """
        payload = {"query": query, "variables": {"ids": entity_ids}}
        data = self._post_json(self.config.graphql_url, payload)
        entities = data.get("data", {}).get("polymer_entities") or []
        matching_group_ids: set[str] = set()
        for entity in entities:
            if not entity:
                continue
            memberships = entity.get("rcsb_polymer_entity_group_membership") or []
            group_ids = self._extract_sequence_identity_groups(
                memberships,
                allowed_cutoffs={similarity_cutoff},
            )
            matching_group_ids.update(group_ids.values())
        return matching_group_ids

    def fetch_xray_polymer_entity_ids_by_sequence(
        self,
        sequence: str,
        sequence_identity_percent: int,
    ) -> list[str]:
        """Search RCSB for X-ray polymer entities matching a query sequence."""
        if sequence_identity_percent not in {95, 100}:
            raise ValueError("sequence_identity_percent must be 95 or 100")
        sequence = "".join(sequence.split()).upper()
        if not sequence:
            return []
        if len(sequence) < 10:
            return []

        query = {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "sequence",
                    "parameters": {
                        "evalue_cutoff": 0.1,
                        "identity_cutoff": sequence_identity_percent / 100.0,
                        "sequence_type": "protein",
                        "target": "pdb_protein_sequence",
                        "value": sequence,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "exptl.method",
                        "operator": "exact_match",
                        "value": "X-RAY DIFFRACTION",
                    },
                },
            ],
        }

        entity_ids: list[str] = []
        start = 0
        total_count: int | None = None
        while total_count is None or start < total_count:
            payload = {
                "query": query,
                "return_type": "polymer_entity",
                "request_options": {
                    "paginate": {"start": start, "rows": self.config.page_size},
                    "results_verbosity": "compact",
                    "scoring_strategy": "sequence",
                },
            }
            response: requests.Response | None = None
            last_error: Exception | None = None
            for attempt in range(1, self.config.retries + 1):
                try:
                    response = self.session.post(
                        self.config.search_url,
                        json=payload,
                        timeout=self.config.timeout_seconds,
                    )
                    if response.status_code in {429} or response.status_code >= 500:
                        last_error = requests.HTTPError(
                            f"{response.status_code} Server Error for url: {self.config.search_url}"
                        )
                        LOGGER.warning(
                            "Sequence search request returned HTTP %d (attempt %d/%d)",
                            response.status_code,
                            attempt,
                            self.config.retries,
                        )
                        if attempt < self.config.retries:
                            time.sleep(self.config.backoff_seconds * attempt)
                            continue
                    break
                except requests.RequestException as exc:
                    last_error = exc
                    LOGGER.warning(
                        "Sequence search request failed (attempt %d/%d): %s",
                        attempt,
                        self.config.retries,
                        exc,
                    )
                    if attempt < self.config.retries:
                        time.sleep(self.config.backoff_seconds * attempt)
            if response is None:
                raise RuntimeError(
                    "Sequence search request failed after "
                    f"{self.config.retries} attempts: {last_error}"
                )
            if response.status_code == 204:
                break
            if response.status_code == 400 and "minimum length" in response.text:
                break
            response.raise_for_status()
            data = response.json()
            total_count = int(data.get("total_count", 0))
            batch_ids: list[str] = []
            for item in data.get("result_set") or []:
                if isinstance(item, str):
                    batch_ids.append(item)
                elif isinstance(item, dict) and item.get("identifier"):
                    batch_ids.append(str(item["identifier"]))
            entity_ids.extend(batch_ids)
            start += len(batch_ids)
            if not batch_ids:
                break
        return entity_ids
