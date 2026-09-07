"""Cache CSV inputs and validate shared tabular data contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .constants import (
    MAX_PLOT_YEAR,
)


class TablePreparationMixin:
    """Cache CSV inputs and validate shared tabular data contracts."""

    def _read_csv(self, data_path: Path) -> pd.DataFrame:
        """Read and cache the CSV table at ``data_path``."""
        resolved = data_path.resolve()
        if resolved not in self._csv_cache:
            self._csv_cache[resolved] = pd.read_csv(resolved)
        return self._csv_cache[resolved]

    @staticmethod
    def _limit_year_column(table: pd.DataFrame) -> pd.DataFrame:
        """Return rows through the configured maximum plotting year."""
        if "year" not in table.columns:
            return table
        return table.loc[table["year"] <= MAX_PLOT_YEAR].copy()

    @staticmethod
    def _validate_required_columns(
        df: pd.DataFrame, required_columns: set[str], dataset_name: str
    ) -> None:
        """Raise an error when ``df`` lacks required dataset columns."""
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(
                f"{dataset_name} is missing required columns: {', '.join(sorted(missing))}"
            )

    @classmethod
    def _prepare_typed_table(
        cls,
        df: pd.DataFrame,
        required_columns: set[str],
        column_types: dict[str, type[Any]],
        dataset_name: str,
    ) -> pd.DataFrame:
        """Validate, copy, and cast a dataset table to requested column types."""
        cls._validate_required_columns(df, required_columns, dataset_name)
        prepared = df.copy()
        for column, dtype in column_types.items():
            prepared[column] = prepared[column].astype(dtype)
        return prepared
