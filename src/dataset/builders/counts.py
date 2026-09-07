"""Collect annual experimental-method and membrane-protein counts."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import TYPE_CHECKING

from src.dataset.config import (
    LOGGER,
    MEMBRANE_ANNOTATION_TYPES,
)
from src.dataset.records import (
    MembraneYearlyCountRecord,
    YearlyCountRecord,
)
from src.dataset.reporting import (
    _record_filtered_structure,
)
from src.dataset.utils import (
    chunked,
    collect_batch_results,
    extract_year,
)

if TYPE_CHECKING:
    from src.dataset.client import (
        RCSBClient,
    )
    from src.dataset.config import (
        DatasetBuildConfig,
        ExperimentalMethod,
    )


class PDBMethodYearlyBuilder:
    """Build annual PDB structure counts by experimental method."""

    def __init__(self, client: RCSBClient, config: DatasetBuildConfig) -> None:
        """Initialize a yearly experimental-method count builder."""
        self.client = client
        self.config = config

    def _fetch_method_records(
        self, method: ExperimentalMethod
    ) -> list[YearlyCountRecord]:
        """Fetch yearly count records for one experimental method."""
        entry_ids = self.client.fetch_entry_ids_for_method_category(
            method=method,
            require_protein_entities=True,
        )
        LOGGER.info("%s: total unique IDs collected: %d", method.label, len(entry_ids))
        year_counter: Counter[int] = Counter()

        batches = list(chunked(entry_ids, self.config.graphql_batch_size))
        for batch_dates in collect_batch_results(
            batches=batches,
            max_workers=self.config.max_workers,
            fetch_fn=self.client.fetch_deposit_dates_for_ids,
            progress_label=method.label,
        ):
            years = filter(
                None, (extract_year(date_value) for date_value in batch_dates)
            )
            year_counter.update(years)

        return [
            YearlyCountRecord(year=year, method=method.label, count=count)
            for year, count in sorted(year_counter.items())
        ]

    def build(self, methods: Iterable[ExperimentalMethod]) -> list[YearlyCountRecord]:
        """Collect yearly counts for all requested experimental methods."""
        records: list[YearlyCountRecord] = []
        for method in methods:
            records.extend(self._fetch_method_records(method))
        return sorted(records, key=lambda record: (record.year, record.method))


class MembraneProteinYearlyBuilder:
    """Build yearly membrane-protein counts overall and by method."""

    def __init__(self, client: RCSBClient, config: DatasetBuildConfig) -> None:
        """Initialize the membrane-protein yearly count builder."""
        self.client = client
        self.config = config

    def _fetch_membrane_entry_ids(self) -> list[str]:
        """Fetch entry IDs annotated as membrane proteins."""
        entry_ids = sorted(
            set(
                self.client.fetch_entry_ids_for_membrane_annotations(
                    MEMBRANE_ANNOTATION_TYPES
                )
            )
        )
        LOGGER.info("Membrane proteins: total unique IDs collected: %d", len(entry_ids))
        return entry_ids

    def _count_entry_years(
        self,
        entry_ids: list[str],
        progress_label: str,
    ) -> Counter[int]:
        """Count entries by deposit year from an entry ID list."""
        year_counter: Counter[int] = Counter()

        batches = list(chunked(entry_ids, self.config.graphql_batch_size))
        for batch_dates in collect_batch_results(
            batches=batches,
            max_workers=self.config.max_workers,
            fetch_fn=self.client.fetch_deposit_dates_for_ids,
            progress_label=progress_label,
        ):
            years = filter(
                None, (extract_year(date_value) for date_value in batch_dates)
            )
            year_counter.update(years)
        return year_counter

    def build(self) -> list[MembraneYearlyCountRecord]:
        """Collect yearly membrane-protein entry counts."""
        entry_ids = self._fetch_membrane_entry_ids()
        year_counter = self._count_entry_years(
            entry_ids=entry_ids,
            progress_label="Membrane proteins",
        )

        return [
            MembraneYearlyCountRecord(year=year, count=count)
            for year, count in sorted(year_counter.items())
        ]

    def build_by_method(
        self,
        methods: Iterable[ExperimentalMethod],
    ) -> list[YearlyCountRecord]:
        """Collect yearly membrane-protein counts split by experimental method."""
        membrane_entry_ids = set(self._fetch_membrane_entry_ids())
        records: list[YearlyCountRecord] = []

        for method in methods:
            method_entry_ids = self.client.fetch_entry_ids_for_method_category(
                method=method,
                require_protein_entities=True,
            )
            membrane_method_entry_ids = [
                entry_id
                for entry_id in method_entry_ids
                if entry_id in membrane_entry_ids
            ]
            excluded_entry_ids = sorted(set(method_entry_ids) - membrane_entry_ids)
            year_by_entry_id: dict[str, int] = {}
            for batch_years in collect_batch_results(
                batches=list(
                    chunked(excluded_entry_ids, self.config.graphql_batch_size)
                ),
                max_workers=self.config.max_workers,
                fetch_fn=self.client.fetch_deposit_year_by_entry_id_for_ids,
                progress_label=f"Non-membrane {method.label} deposit years",
            ):
                year_by_entry_id.update(batch_years)
            for entry_id in excluded_entry_ids:
                _record_filtered_structure(
                    entry_id,
                    f"entry has no supported membrane-protein annotation for {method.label}",
                    year=year_by_entry_id.get(entry_id),
                )
            LOGGER.info(
                "Membrane proteins %s: kept %d/%d method entries",
                method.label,
                len(membrane_method_entry_ids),
                len(method_entry_ids),
            )
            year_counter = self._count_entry_years(
                entry_ids=membrane_method_entry_ids,
                progress_label=f"Membrane proteins {method.label}",
            )
            records.extend(
                YearlyCountRecord(year=year, method=method.label, count=count)
                for year, count in sorted(year_counter.items())
            )

        return sorted(records, key=lambda record: (record.year, record.method))
