"""Shared CSV writing, atomic replacement and resume checks."""

from __future__ import annotations

import csv
import os
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any


def write_csv_rows(
    output_path: Path,
    header: Sequence[str],
    rows: Iterable[Sequence[Any]],
) -> None:
    """Write dataclass records to CSV with the requested field order."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(rows)


def _atomic_write_csv_rows(
    output_path: Path,
    header: Sequence[str],
    rows: Iterable[Sequence[Any]],
) -> None:
    """Atomically replace a CSV file using a collision-safe temporary path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            newline="",
            encoding="utf-8",
            suffix=f"{output_path.suffix}.tmp",
            prefix=f".{output_path.stem}.",
            dir=str(output_path.parent),
            delete=False,
        ) as csvfile:
            temp_path = Path(csvfile.name)
            writer = csv.writer(csvfile)
            writer.writerow(header)
            writer.writerows(rows)
            csvfile.flush()
            os.fsync(csvfile.fileno())
        temp_path.replace(output_path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def require_complete_last_row(input_path: Path) -> None:
    """Refuse to resume from a CSV whose last row lacks its line terminator."""
    # csv.writer ends every row with a newline, so a missing one marks a write
    # cut off mid-row (for example by a full disk); its values may be partial.
    with input_path.open("rb") as handle:
        if handle.seek(0, os.SEEK_END) == 0:
            return
        handle.seek(-1, os.SEEK_END)
        if handle.read(1) == b"\n":
            return
    raise SystemExit(
        f"Cannot resume {input_path}: incomplete last row (interrupted write). "
        "Remove that row and run again."
    )
