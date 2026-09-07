"""Collect solution-NMR weights, experiments, and validation quality."""

from __future__ import annotations

from typing import TYPE_CHECKING

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
        SolutionNMRMonomerExperimentsRecord,
        SolutionNMRMonomerQualityRecord,
        SolutionNMRWeightRecord,
    )


class SolutionNMRWeightBuilder:
    """Build molecular-weight records for eligible solution-NMR entries."""

    def __init__(self, client: RCSBClient, config: DatasetBuildConfig) -> None:
        """Initialize the SOLUTION NMR molecular-weight builder."""
        self.client = client
        self.config = config

    def build(self) -> list[SolutionNMRWeightRecord]:
        """Collect molecular-weight records for SOLUTION NMR entries."""
        entry_ids = fetch_solution_nmr_entry_ids(
            client=self.client,
            log_label="SOLUTION NMR",
            require_protein_entities=True,
        )
        batches = list(chunked(entry_ids, self.config.graphql_batch_size))

        records: list[SolutionNMRWeightRecord] = []
        for batch_records in collect_batch_results(
            batches=batches,
            max_workers=self.config.max_workers,
            fetch_fn=self.client.fetch_solution_nmr_weight_records_for_ids,
            progress_label="SOLUTION NMR weights",
        ):
            records.extend(batch_records)
        return sorted(records, key=lambda record: (record.year, record.entry_id))


class SolutionNMRMonomerExperimentsBuilder:
    """Build reported-experiment records for solution-NMR monomers."""

    def __init__(self, client: RCSBClient, config: DatasetBuildConfig) -> None:
        """Initialize the SOLUTION NMR monomer experiment builder."""
        self.client = client
        self.config = config

    def build(self) -> list[SolutionNMRMonomerExperimentsRecord]:
        """Collect experiment descriptions for all eligible NMR monomers."""
        entry_ids = fetch_solution_nmr_entry_ids(
            client=self.client,
            log_label="SOLUTION NMR monomer experiments",
        )
        batches = list(chunked(entry_ids, self.config.graphql_batch_size))
        records: list[SolutionNMRMonomerExperimentsRecord] = []
        for batch_records in collect_batch_results(
            batches=batches,
            max_workers=self.config.max_workers,
            fetch_fn=self.client.fetch_solution_nmr_monomer_experiment_records_for_ids,
            progress_label="SOLUTION NMR monomer experiments",
        ):
            records.extend(batch_records)
        return sorted(records, key=lambda record: (record.year, record.entry_id))


class SolutionNMRMonomerQualityBuilder:
    """Build validation-quality records for solution-NMR monomers."""

    def __init__(self, client: RCSBClient, config: DatasetBuildConfig) -> None:
        """Initialize the SOLUTION NMR monomer quality builder."""
        self.client = client
        self.config = config

    def build(self) -> list[SolutionNMRMonomerQualityRecord]:
        """Collect validation quality records for SOLUTION NMR monomers."""
        entry_ids = fetch_solution_nmr_entry_ids(
            client=self.client,
            log_label="SOLUTION NMR quality",
        )
        batches = list(chunked(entry_ids, self.config.graphql_batch_size))
        records: list[SolutionNMRMonomerQualityRecord] = []

        for batch_records in collect_batch_results(
            batches=batches,
            max_workers=self.config.max_workers,
            fetch_fn=self.client.fetch_solution_nmr_monomer_quality_records_for_ids,
            progress_label="SOLUTION NMR quality",
        ):
            records.extend(batch_records)
        return sorted(records, key=lambda r: (r.year, r.entry_id))
