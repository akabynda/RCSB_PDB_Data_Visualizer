"""Batch collection and shared metadata normalization helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import TYPE_CHECKING, Any, TypeVar

from src.dataset.config import (
    LOGGER,
    SOLUTION_NMR_METHOD,
)
from src.dataset.reporting import (
    _record_filtered_structure,
)

if TYPE_CHECKING:
    from src.dataset.client import (
        RCSBClient,
    )


T = TypeVar("T")


def chunked(items: list[str], size: int) -> Iterator[list[str]]:
    """Yield fixed-size batches from a list of item identifiers."""
    for i in range(0, len(items), size):
        yield items[i : i + size]


def extract_year(deposit_date: str | None) -> int | None:
    """Extract a deposit year from an RCSB date string."""
    if not deposit_date:
        return None
    try:
        return int(deposit_date[:4])
    except (TypeError, ValueError):
        return None


def parse_rcsb_datetime(value: str | None) -> datetime | None:
    """Parse an RCSB date or datetime string into a datetime object."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def collect_batch_results(
    batches: list[list[str]],
    max_workers: int,
    fetch_fn: Callable[[list[str]], T],
    progress_label: str,
) -> list[T]:
    """Run batched collection work in parallel and collect successful records."""
    if not batches:
        return []
    results: list[T] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(fetch_fn, batch): idx
            for idx, batch in enumerate(batches, start=1)
        }
        for future in as_completed(future_map):
            results.append(future.result())
            batch_idx = future_map[future]
            LOGGER.info(
                "%s: processed batch %d/%d", progress_label, batch_idx, len(batches)
            )
    return results


def _record_entries_missing_from_response(
    requested_entry_ids: Iterable[str],
    entries: Iterable[dict[str, Any] | None],
) -> None:
    """Record requested PDB entries omitted from a GraphQL batch response."""
    returned_entry_ids = {
        str(entry.get("rcsb_id")) for entry in entries if entry and entry.get("rcsb_id")
    }
    for entry_id in sorted(set(requested_entry_ids) - returned_entry_ids):
        _record_filtered_structure(
            entry_id,
            "entry metadata missing from RCSB GraphQL response",
        )


def fetch_solution_nmr_entry_ids(
    client: RCSBClient,
    log_label: str,
    require_protein_entities: bool = False,
) -> list[str]:
    """Fetch all entry IDs assigned to the SOLUTION NMR method."""
    entry_ids = sorted(
        set(
            client.fetch_entry_ids_for_method(
                method_label=SOLUTION_NMR_METHOD,
                query_value=SOLUTION_NMR_METHOD,
                require_protein_entities=require_protein_entities,
            )
        )
    )
    LOGGER.info("%s: total unique IDs collected: %d", log_label, len(entry_ids))
    return entry_ids


def contains_noesy_experiment(experiments: Iterable[str]) -> bool:
    """Return whether at least one experiment contains NOESY."""
    return any("NOESY" in experiment.upper() for experiment in experiments)
