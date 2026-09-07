"""Retrieve accession dates, resolutions, and experimental methods."""

from __future__ import annotations

from ..reporting import _record_filtered_structure
from ..utils import _record_entries_missing_from_response, extract_year


class EntryMetadataMixin:
    """Fetch entry-level metadata from the RCSB GraphQL service."""

    def fetch_deposit_dates_for_ids(self, entry_ids: list[str]) -> list[str]:
        """Fetch deposit date strings for entry IDs."""
        query = """
        query($ids:[String!]!) {
          entries(entry_ids:$ids) {
            rcsb_id
            rcsb_accession_info {
              deposit_date
            }
          }
        }
        """
        payload = {"query": query, "variables": {"ids": entry_ids}}
        data = self._post_json(self.config.graphql_url, payload)
        entries = data.get("data", {}).get("entries") or []
        _record_entries_missing_from_response(entry_ids, entries)
        for entry in entries:
            if not entry or not entry.get("rcsb_id"):
                continue
            if (
                extract_year(
                    (entry.get("rcsb_accession_info") or {}).get("deposit_date")
                )
                is None
            ):
                _record_filtered_structure(
                    entry["rcsb_id"],
                    "deposit year is missing or invalid",
                    year=(entry.get("rcsb_accession_info") or {}).get("deposit_date"),
                )
        return [
            (entry.get("rcsb_accession_info") or {}).get("deposit_date")
            for entry in entries
            if entry and (entry.get("rcsb_accession_info") or {}).get("deposit_date")
        ]

    def fetch_deposit_year_by_entry_id_for_ids(
        self, entry_ids: list[str]
    ) -> dict[str, int]:
        """Fetch deposit years keyed by entry ID."""
        if not entry_ids:
            return {}
        query = """
        query($ids:[String!]!) {
          entries(entry_ids:$ids) {
            rcsb_id
            rcsb_accession_info {
              deposit_date
            }
          }
        }
        """
        payload = {"query": query, "variables": {"ids": entry_ids}}
        data = self._post_json(self.config.graphql_url, payload)
        entries = data.get("data", {}).get("entries") or []
        entry_year_by_id: dict[str, int] = {}
        for entry in entries:
            if not entry:
                continue
            entry_id = entry.get("rcsb_id")
            if not entry_id:
                continue
            year = extract_year(
                (entry.get("rcsb_accession_info") or {}).get("deposit_date")
            )
            if year is None:
                continue
            entry_year_by_id[str(entry_id)] = year
        return entry_year_by_id

    def fetch_deposit_date_by_entry_id_for_ids(
        self, entry_ids: list[str]
    ) -> dict[str, str]:
        """Fetch deposit datetimes keyed by entry ID."""
        if not entry_ids:
            return {}
        query = """
        query($ids:[String!]!) {
          entries(entry_ids:$ids) {
            rcsb_id
            rcsb_accession_info {
              deposit_date
            }
          }
        }
        """
        payload = {"query": query, "variables": {"ids": entry_ids}}
        data = self._post_json(self.config.graphql_url, payload)
        entries = data.get("data", {}).get("entries") or []
        entry_date_by_id: dict[str, str] = {}
        for entry in entries:
            if not entry:
                continue
            entry_id = entry.get("rcsb_id")
            deposit_date = (entry.get("rcsb_accession_info", {}) or {}).get(
                "deposit_date"
            )
            if not entry_id or not deposit_date:
                continue
            entry_date_by_id[str(entry_id)] = str(deposit_date)
        return entry_date_by_id

    def fetch_accession_dates_by_entry_id_for_ids(
        self, entry_ids: list[str]
    ) -> dict[str, tuple[str | None, str | None]]:
        """Fetch initial release datetimes keyed by entry ID."""
        if not entry_ids:
            return {}
        query = """
        query($ids:[String!]!) {
          entries(entry_ids:$ids) {
            rcsb_id
            rcsb_accession_info {
              deposit_date
              initial_release_date
            }
          }
        }
        """
        payload = {"query": query, "variables": {"ids": entry_ids}}
        data = self._post_json(self.config.graphql_url, payload)
        entries = data.get("data", {}).get("entries") or []
        entry_dates_by_id: dict[str, tuple[str | None, str | None]] = {}
        for entry in entries:
            if not entry:
                continue
            entry_id = entry.get("rcsb_id")
            accession_info = entry.get("rcsb_accession_info", {}) or {}
            if not entry_id:
                continue
            entry_dates_by_id[str(entry_id)] = (
                accession_info.get("deposit_date"),
                accession_info.get("initial_release_date"),
            )
        return entry_dates_by_id

    def fetch_entry_resolution_for_ids(self, entry_ids: list[str]) -> dict[str, float]:
        """Fetch crystallographic resolution values keyed by entry ID."""
        query = """
        query($ids:[String!]!) {
          entries(entry_ids:$ids) {
            rcsb_id
            rcsb_entry_info {
              resolution_combined
            }
          }
        }
        """
        payload = {"query": query, "variables": {"ids": entry_ids}}
        data = self._post_json(self.config.graphql_url, payload)
        entries = data.get("data", {}).get("entries") or []
        resolutions: dict[str, float] = {}
        for entry in entries:
            if not entry:
                continue
            entry_id = entry.get("rcsb_id")
            if not entry_id:
                continue
            combined = (entry.get("rcsb_entry_info") or {}).get("resolution_combined")
            if not combined:
                continue
            try:
                value = min(float(item) for item in combined if item is not None)
            except (TypeError, ValueError):
                continue
            resolutions[str(entry_id)] = value
        return resolutions

    def fetch_entry_experimental_methods_for_ids(
        self, entry_ids: list[str]
    ) -> dict[str, tuple[str, ...]]:
        """Fetch reported experimental methods keyed by entry ID."""
        if not entry_ids:
            return {}
        query = """
        query($ids:[String!]!) {
          entries(entry_ids:$ids) {
            rcsb_id
            exptl {
              method
            }
          }
        }
        """
        payload = {"query": query, "variables": {"ids": entry_ids}}
        data = self._post_json(self.config.graphql_url, payload)
        entries = data.get("data", {}).get("entries") or []
        methods_by_entry_id: dict[str, tuple[str, ...]] = {}
        for entry in entries:
            if not entry:
                continue
            entry_id = entry.get("rcsb_id")
            if not entry_id:
                continue
            methods: list[str] = []
            for item in entry.get("exptl") or []:
                if not item:
                    continue
                method = item.get("method")
                if method:
                    methods.append(str(method))
            methods_by_entry_id[str(entry_id)] = tuple(methods)
        return methods_by_entry_id
