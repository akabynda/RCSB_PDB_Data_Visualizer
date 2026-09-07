"""Per-dataset warning logs and rejected-structure audit reports."""

from __future__ import annotations

import csv
import logging
from collections.abc import Iterable
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any

from src.dataset.io.common import (
    write_csv_rows,
)

if TYPE_CHECKING:
    from src.dataset.config import (
        DatasetKind,
    )


_ACTIVE_DATASET_WARNING_LOG_PATHS: frozenset[Path] = frozenset()


_ACTIVE_DATASET_FILTERED_CSV_PATHS: frozenset[Path] = frozenset()


_FILTERED_STRUCTURE_ROWS_BY_OUTPUT: dict[Path, dict[tuple[str, str], str]] = {}


_FILTERED_STRUCTURE_CSV_LOCK = Lock()


class ActiveDatasetWarningLogFilter(logging.Filter):
    """Route warnings to the log files of the dataset currently being built."""

    def __init__(self, csv_output_path: Path) -> None:
        """Track the CSV whose active build should receive warning records."""
        super().__init__()
        self.csv_output_path = csv_output_path

    def filter(self, record: logging.LogRecord) -> bool:
        """Return whether ``record`` belongs in this dataset's warning log."""
        return self.csv_output_path in _ACTIVE_DATASET_WARNING_LOG_PATHS


def _set_active_dataset_warning_logs(output_paths: Iterable[Path]) -> None:
    """Select which per-CSV warning logs and filter reports are active."""
    global _ACTIVE_DATASET_WARNING_LOG_PATHS, _ACTIVE_DATASET_FILTERED_CSV_PATHS
    paths = frozenset(Path(path) for path in output_paths)
    _ACTIVE_DATASET_WARNING_LOG_PATHS = paths
    _ACTIVE_DATASET_FILTERED_CSV_PATHS = paths


def _set_active_dataset_filtered_csvs(output_paths: Iterable[Path]) -> None:
    """Override the active filtered-structure reports within a dataset build."""
    global _ACTIVE_DATASET_FILTERED_CSV_PATHS
    _ACTIVE_DATASET_FILTERED_CSV_PATHS = frozenset(Path(path) for path in output_paths)


def filtered_structures_csv_path(output_path: Path) -> Path:
    """Return the sibling CSV used to explain filtered-out structures."""
    output_path = Path(output_path)
    return output_path.with_name(f"{output_path.stem}_filtered.csv")


def _configure_dataset_filtered_csvs(
    output_paths_by_dataset: dict[DatasetKind, tuple[Path, ...]],
    preserve_existing_output_paths: Iterable[Path] = (),
) -> None:
    """Create fresh per-output CSV reports for structures rejected by filters."""
    global _FILTERED_STRUCTURE_ROWS_BY_OUTPUT
    output_paths = {
        Path(path) for paths in output_paths_by_dataset.values() for path in paths
    }
    preserved_paths = {Path(path) for path in preserve_existing_output_paths}
    with _FILTERED_STRUCTURE_CSV_LOCK:
        rows_by_output: dict[Path, dict[tuple[str, str], str]] = {}
        for output_path in sorted(output_paths, key=str):
            filtered_path = filtered_structures_csv_path(output_path)
            existing_rows: dict[tuple[str, str], str] = {}
            if output_path in preserved_paths and filtered_path.exists():
                with filtered_path.open(newline="", encoding="utf-8") as csvfile:
                    for row in csv.DictReader(csvfile):
                        entry_id = str(row.get("entry_id") or "").strip()
                        year = _normalize_filtered_structure_year(row.get("year"))
                        reason = " ".join(str(row.get("reason") or "").split())
                        if entry_id and reason:
                            existing_rows[(entry_id, reason)] = year
            rows_by_output[output_path] = existing_rows
            write_csv_rows(
                output_path=filtered_path,
                header=("entry_id", "year", "reason"),
                rows=(
                    (entry_id, year, reason)
                    for (entry_id, reason), year in sorted(existing_rows.items())
                ),
            )
        _FILTERED_STRUCTURE_ROWS_BY_OUTPUT = rows_by_output


def _normalize_filtered_structure_year(year: Any) -> str:
    """Normalize a deposit year for filtered-structure CSV serialization."""
    raw_year = str(year or "").strip()
    if len(raw_year) >= 4 and raw_year[:4].isdigit():
        return raw_year[:4]
    return ""


def _record_filtered_structure(entry_id: Any, reason: str, year: Any = None) -> None:
    """Append one deduplicated filter decision to every active dataset report."""
    normalized_entry_id = str(entry_id or "").strip()
    normalized_year = _normalize_filtered_structure_year(year)
    normalized_reason = " ".join(str(reason).split())
    if not normalized_entry_id or not normalized_reason:
        return
    with _FILTERED_STRUCTURE_CSV_LOCK:
        for output_path in _ACTIVE_DATASET_FILTERED_CSV_PATHS:
            rows = _FILTERED_STRUCTURE_ROWS_BY_OUTPUT.get(output_path)
            if rows is None:
                continue
            key = (normalized_entry_id, normalized_reason)
            existing_year = rows.get(key)
            if existing_year is not None:
                if not existing_year and normalized_year:
                    rows[key] = normalized_year
                    filtered_path = filtered_structures_csv_path(output_path)
                    write_csv_rows(
                        output_path=filtered_path,
                        header=("entry_id", "year", "reason"),
                        rows=(
                            (stored_entry_id, stored_year, stored_reason)
                            for (
                                stored_entry_id,
                                stored_reason,
                            ), stored_year in sorted(rows.items())
                        ),
                    )
                continue
            filtered_path = filtered_structures_csv_path(output_path)
            with filtered_path.open("a", newline="", encoding="utf-8") as csvfile:
                csv.writer(csvfile).writerow(
                    (normalized_entry_id, normalized_year, normalized_reason)
                )
            rows[key] = normalized_year


def _import_filtered_structures(input_dataset_path: Path) -> None:
    """Copy upstream filter decisions into the currently active reports."""
    input_filtered_path = filtered_structures_csv_path(input_dataset_path)
    if not input_filtered_path.exists():
        return
    with input_filtered_path.open(newline="", encoding="utf-8") as csvfile:
        for row in csv.DictReader(csvfile):
            _record_filtered_structure(
                row.get("entry_id"),
                row.get("reason") or "",
                year=row.get("year"),
            )


def _configure_dataset_warning_logs(
    output_paths_by_dataset: dict[DatasetKind, tuple[Path, ...]],
) -> list[logging.FileHandler]:
    """Create fresh WARNING/ERROR log files next to selected output CSV files."""
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    root_logger = logging.getLogger()
    handlers: list[logging.FileHandler] = []
    unique_output_paths = {
        path for paths in output_paths_by_dataset.values() for path in paths
    }
    for output_path in sorted(unique_output_paths, key=str):
        log_path = output_path.with_suffix(".log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        handler.setLevel(logging.WARNING)
        handler.setFormatter(formatter)
        handler.addFilter(ActiveDatasetWarningLogFilter(output_path))
        root_logger.addHandler(handler)
        handlers.append(handler)
    return handlers
