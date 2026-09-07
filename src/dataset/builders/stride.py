"""Collect secondary-structure composition for NMR monomers."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

from src.dataset.config import (
    LOGGER,
)
from src.dataset.utils import (
    chunked,
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
        SolutionNMRMonomerStrideModeledFirstModelRecord,
    )


class SolutionNMRMonomerStrideModeledFirstModelBuilder:
    """Build STRIDE records for first modeled solution-NMR monomer chains."""

    def __init__(
        self,
        client: RCSBClient,
        config: DatasetBuildConfig,
        stride_executable: str,
        cache_dir: Path,
        stride_cache_dir: Path,
    ) -> None:
        """Initialize streaming STRIDE modeled-first-model collection."""
        self.client = client
        self.config = config
        self.stride_executable = stride_executable
        self.cache_dir = cache_dir
        self.stride_cache_dir = stride_cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.stride_cache_dir.mkdir(parents=True, exist_ok=True)

    def iter_batches(
        self,
    ) -> Iterator[list[SolutionNMRMonomerStrideModeledFirstModelRecord]]:
        """Yield STRIDE modeled-first-model records batch by batch."""
        entry_ids = fetch_solution_nmr_entry_ids(
            client=self.client,
            log_label="SOLUTION NMR monomer-stride-modeled-first-model",
        )
        batches = list(chunked(entry_ids, self.config.graphql_batch_size))
        if not batches:
            return

        for batch_idx, batch in enumerate(batches, start=1):
            batch_records = self.client.fetch_solution_nmr_monomer_stride_modeled_first_model_records_for_ids(
                entry_ids=batch,
                stride_executable=self.stride_executable,
                pdb_cache_dir=self.cache_dir,
                stride_cache_dir=self.stride_cache_dir,
            )
            LOGGER.info(
                (
                    "SOLUTION NMR monomer-stride-modeled-first-model: "
                    "processed batch %d/%d (entries: %d)"
                ),
                batch_idx,
                len(batches),
                len(batch_records),
            )
            yield batch_records

    def iter_records(self) -> Iterator[SolutionNMRMonomerStrideModeledFirstModelRecord]:
        """Yield individual STRIDE modeled-first-model records."""
        for batch_records in self.iter_batches():
            for record in batch_records:
                yield record

    def build(self) -> list[SolutionNMRMonomerStrideModeledFirstModelRecord]:
        """Collect all STRIDE modeled-first-model records into a list."""
        records = list(self.iter_records())
        return sorted(records, key=lambda record: (record.year, record.entry_id))
