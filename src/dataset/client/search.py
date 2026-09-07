"""Search entry identifiers and filter experimental-method categories."""

from __future__ import annotations

from typing import Any

from ..config import LOGGER, SOLUTION_NMR_METHOD, ExperimentalMethod
from ..reporting import _record_filtered_structure
from ..utils import chunked, extract_year


class EntrySearchMixin:
    """Search RCSB entries by method or membrane annotations."""

    def _fetch_paginated_identifiers(
        self,
        query: dict[str, Any],
        return_type: str,
        progress_label: str | None = None,
    ) -> list[str]:
        """Run a paginated RCSB search query and return all identifiers."""
        all_ids: list[str] = []
        start = 0
        total_count: int | None = None
        while total_count is None or start < total_count:
            payload = {
                "query": query,
                "return_type": return_type,
                "request_options": {
                    "paginate": {"start": start, "rows": self.config.page_size}
                },
            }
            data = self._post_json(self.config.search_url, payload)
            total_count = int(data.get("total_count", 0))
            batch_ids = [
                item["identifier"]
                for item in data.get("result_set") or []
                if "identifier" in item
            ]
            all_ids.extend(batch_ids)
            start += len(batch_ids)
            if not batch_ids:
                break
            if progress_label:
                LOGGER.info(
                    "%s: fetched %d/%d entry IDs",
                    progress_label,
                    len(all_ids),
                    total_count,
                )
        return all_ids

    def fetch_entry_ids_for_method(
        self,
        method_label: str,
        query_value: str,
        require_protein_entities: bool = False,
    ) -> list[str]:
        """Fetch entry IDs for one experimental method."""
        return self.fetch_entry_ids_for_method_set(
            method_label=method_label,
            method_values=(query_value,),
            require_protein_entities=require_protein_entities,
        )

    def fetch_entry_ids_for_method_set(
        self,
        method_label: str,
        method_values: tuple[str, ...],
        require_protein_entities: bool = False,
    ) -> list[str]:
        """Fetch entry IDs whose experimental methods exactly match a set."""
        if not method_values:
            raise ValueError("method_values must not be empty")

        method_queries: list[dict[str, Any]] = [
            {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "exptl.method",
                    "operator": "exact_match",
                    "value": method_value,
                },
            }
            for method_value in method_values
        ]
        method_query: dict[str, Any]
        if len(method_queries) == 1:
            method_query = method_queries[0]
        else:
            method_query = {
                "type": "group",
                "logical_operator": "and",
                "nodes": method_queries,
            }
        matched_entry_ids = self._fetch_paginated_identifiers(
            query=method_query,
            return_type="entry",
            progress_label=f"{method_label} ({' + '.join(method_values)})",
        )
        if not matched_entry_ids:
            return []

        filtered_entry_ids: list[str] = []
        for batch in chunked(matched_entry_ids, self.config.graphql_batch_size):
            filtered_entry_ids.extend(
                self._filter_entry_ids_by_exact_methods(
                    entry_ids=batch,
                    method_values=method_values,
                    require_protein_entities=require_protein_entities,
                    record_exclusions=(method_label == SOLUTION_NMR_METHOD),
                )
            )

        LOGGER.info(
            "%s (%s): kept %d/%d entries with the exact method set",
            method_label,
            " + ".join(method_values),
            len(filtered_entry_ids),
            len(matched_entry_ids),
        )
        return filtered_entry_ids

    def fetch_entry_ids_for_method_category(
        self,
        method: ExperimentalMethod,
        require_protein_entities: bool = False,
    ) -> list[str]:
        """Fetch and filter all candidates for one experimental-method category."""
        candidate_entry_ids: set[str] = set()
        for method_values in method.exact_method_sets:
            method_queries = [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "exptl.method",
                        "operator": "exact_match",
                        "value": method_value,
                    },
                }
                for method_value in method_values
            ]
            query: dict[str, Any]
            if len(method_queries) == 1:
                query = method_queries[0]
            else:
                query = {
                    "type": "group",
                    "logical_operator": "and",
                    "nodes": method_queries,
                }
            candidate_entry_ids.update(
                self._fetch_paginated_identifiers(
                    query=query,
                    return_type="entry",
                    progress_label=(
                        f"{method.label} candidates ({' + '.join(method_values)})"
                    ),
                )
            )

        filtered_entry_ids: list[str] = []
        sorted_candidates = sorted(candidate_entry_ids)
        for batch in chunked(sorted_candidates, self.config.graphql_batch_size):
            filtered_entry_ids.extend(
                self._filter_entry_ids_by_allowed_method_sets(
                    entry_ids=batch,
                    allowed_method_sets=method.exact_method_sets,
                    require_protein_entities=require_protein_entities,
                    record_exclusions=True,
                )
            )
        LOGGER.info(
            "%s: kept %d/%d entries after category filters",
            method.label,
            len(filtered_entry_ids),
            len(sorted_candidates),
        )
        return sorted(filtered_entry_ids)

    def _filter_entry_ids_by_exact_single_method(
        self, entry_ids: list[str], method_value: str
    ) -> list[str]:
        """Keep entries whose experimental method list is exactly the requested method."""
        return self._filter_entry_ids_by_exact_methods(entry_ids, (method_value,))

    def _filter_entry_ids_by_exact_methods(
        self,
        entry_ids: list[str],
        method_values: tuple[str, ...],
        require_protein_entities: bool = False,
        record_exclusions: bool = False,
    ) -> list[str]:
        """Keep entries whose experimental methods exactly match the requested set."""
        return self._filter_entry_ids_by_allowed_method_sets(
            entry_ids=entry_ids,
            allowed_method_sets=(method_values,),
            require_protein_entities=require_protein_entities,
            record_exclusions=record_exclusions,
        )

    def _filter_entry_ids_by_allowed_method_sets(
        self,
        entry_ids: list[str],
        allowed_method_sets: tuple[tuple[str, ...], ...],
        require_protein_entities: bool = False,
        record_exclusions: bool = False,
    ) -> list[str]:
        """Keep entries matching an allowed method set and protein requirement."""
        if not entry_ids:
            return []

        query = """
        query($ids:[String!]!) {
          entries(entry_ids:$ids) {
            rcsb_id
            rcsb_accession_info {
              deposit_date
            }
            exptl {
              method
            }
            rcsb_entry_info {
              polymer_entity_count_protein
            }
          }
        }
        """
        payload = {"query": query, "variables": {"ids": entry_ids}}
        data = self._post_json(self.config.graphql_url, payload)
        entries = data.get("data", {}).get("entries") or []

        filtered: list[str] = []
        returned_entry_ids: set[str] = set()
        allowed_method_sets_normalized = {
            frozenset(method_set) for method_set in allowed_method_sets
        }
        for entry in entries:
            if not entry:
                continue
            entry_id = entry.get("rcsb_id")
            if not entry_id:
                continue
            normalized_entry_id = str(entry_id)
            returned_entry_ids.add(normalized_entry_id)
            year = extract_year(
                (entry.get("rcsb_accession_info") or {}).get("deposit_date")
            )

            methods = []
            for exptl in entry.get("exptl") or []:
                if not exptl:
                    continue
                method = exptl.get("method")
                if method:
                    methods.append(str(method))

            unique_methods = frozenset(methods)
            if unique_methods not in allowed_method_sets_normalized:
                if not record_exclusions:
                    continue
                actual_methods = ", ".join(sorted(unique_methods)) or "none"
                expected_methods = " or ".join(
                    f"[{', '.join(method_set)}]" for method_set in allowed_method_sets
                )
                _record_filtered_structure(
                    normalized_entry_id,
                    "experimental method set does not exactly match "
                    f"{expected_methods} (found [{actual_methods}])",
                    year=year,
                )
                continue

            if require_protein_entities:
                protein_count_raw = (entry.get("rcsb_entry_info") or {}).get(
                    "polymer_entity_count_protein"
                )
                try:
                    protein_count = int(protein_count_raw)
                except (TypeError, ValueError):
                    if record_exclusions:
                        _record_filtered_structure(
                            normalized_entry_id,
                            "protein polymer entity count is missing or invalid",
                            year=year,
                        )
                    continue
                if protein_count < 1:
                    if record_exclusions:
                        _record_filtered_structure(
                            normalized_entry_id,
                            "entry has no protein polymer entities",
                            year=year,
                        )
                    continue

            filtered.append(normalized_entry_id)

        if record_exclusions:
            for missing_entry_id in sorted(set(entry_ids) - returned_entry_ids):
                _record_filtered_structure(
                    missing_entry_id,
                    "entry metadata missing from RCSB GraphQL response",
                )

        return filtered

    def fetch_entry_ids_for_membrane_annotations(
        self, annotation_types: tuple[str, ...]
    ) -> list[str]:
        """Fetch entry IDs with membrane-protein annotations."""
        query = {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_polymer_entity_annotation.type",
                "operator": "in",
                "value": list(annotation_types),
            },
        }
        return self._fetch_paginated_identifiers(
            query=query,
            return_type="entry",
            progress_label=f"Membrane proteins ({','.join(annotation_types)})",
        )
