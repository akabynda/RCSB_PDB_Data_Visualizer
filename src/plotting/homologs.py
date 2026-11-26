"""Plot X-ray homolog availability, historical shares, and release timing."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .constants import (
    XRAY_HOMOLOG_TIMING_LABELS,
)


class HomologPlotsMixin:
    """Plot X-ray homolog availability, historical shares, and release timing."""

    def _render_homolog_timing_stackplot(
        self,
        table: pd.DataFrame,
        output_png: Path,
        output_svg: Path,
        title: str,
    ) -> None:
        """Render homolog availability timing shares as a stacked plot."""

        def draw(ax: plt.Axes) -> None:
            """Draw cumulative layers for the three homolog timing states."""
            base = pd.Series(0.0, index=table.index, dtype=float)
            x_step_edges = self._step_edges(table.index)
            for idx, label in enumerate(XRAY_HOMOLOG_TIMING_LABELS):
                values = table[label].astype(float)
                top = base + values
                ax.fill_between(
                    x_step_edges,
                    self._step_values(base),
                    self._step_values(top),
                    step="post",
                    color=self.config.homolog_timing_colors[
                        idx % len(self.config.homolog_timing_colors)
                    ],
                    alpha=0.85,
                    label=label,
                )
                base = top
            ax.set_ylim(0.0, 100.0)
            ax.set_xlim(float(x_step_edges[0]), float(x_step_edges[-1]))
            self._add_legend(ax, loc="upper left")

        self._render_figure(
            output_png=output_png,
            output_svg=output_svg,
            title=title,
            y_label=self.config.nmr_monomer_xray_homolog_timing_share_y_label,
            draw_fn=draw,
        )

    @staticmethod
    def _homolog_share_series(table: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """Return annual and cumulative shares of entries with X-ray homologs."""
        yearly_share = (
            table.groupby("year", as_index=True)["has_xray_homolog"]
            .mean()
            .mul(100.0)
            .sort_index()
        )
        yearly_counts = table.groupby("year", as_index=True)["entry_id"].count()
        yearly_yes = table.groupby("year", as_index=True)["has_xray_homolog"].sum()
        cumulative_share = (
            yearly_yes.cumsum().div(yearly_counts.cumsum()).mul(100.0).sort_index()
        )
        return yearly_share, cumulative_share

    @classmethod
    def _prepare_monomer_xray_homolog_table(
        cls, df: pd.DataFrame, filter_rows: bool = True
    ) -> pd.DataFrame:
        """Validate homolog-search records and optionally apply plot filters."""
        prepared = cls._prepare_typed_table(
            df=df,
            required_columns={
                "entry_id",
                "year",
                "sequence_identity_percent",
                "nmr_query_sequence_length",
                "has_xray_homolog",
            },
            column_types={
                "year": int,
                "sequence_identity_percent": int,
                "nmr_query_sequence_length": int,
                "has_xray_homolog": int,
            },
            dataset_name="Monomer X-ray homolog CSV",
        )
        if not filter_rows:
            return prepared
        return cls._filter_monomer_xray_homolog_rows(prepared)

    @classmethod
    def _filter_monomer_xray_homolog_rows(cls, table: pd.DataFrame) -> pd.DataFrame:
        """Keep query sequences longer than 10 residues through the plot year."""
        filtered = table[table["nmr_query_sequence_length"] > 10].copy()
        return cls._limit_year_column(filtered)

    def plot_solution_nmr_monomer_xray_homologs(
        self,
        data_95_path: Path,
        data_100_path: Path,
        output_95_png: Path,
        output_95_svg: Path,
        output_100_png: Path,
        output_100_svg: Path,
        cumulative_output_95_png: Path,
        cumulative_output_95_svg: Path,
        cumulative_output_100_png: Path,
        cumulative_output_100_svg: Path,
    ) -> None:
        """Render annual and cumulative shares for current homolog searches."""
        table_95 = self._prepare_monomer_xray_homolog_table(
            self._read_csv(data_95_path)
        )
        table_100 = self._prepare_monomer_xray_homolog_table(
            self._read_csv(data_100_path)
        )
        self._scientific_style()

        yearly_95, cumulative_share_95 = self._homolog_share_series(table_95)
        yearly_100, cumulative_share_100 = self._homolog_share_series(table_100)

        self._render_bar_series(
            output_png=output_95_png,
            output_svg=output_95_svg,
            title=self.config.nmr_monomer_xray_homolog_95_title,
            y_label=self.config.nmr_monomer_xray_homolog_y_label,
            x_values=yearly_95.index,
            y_values=yearly_95,
            color="#1f77b4",
            y_limits=(0.0, 100.0),
        )
        self._render_bar_series(
            output_png=output_100_png,
            output_svg=output_100_svg,
            title=self.config.nmr_monomer_xray_homolog_100_title,
            y_label=self.config.nmr_monomer_xray_homolog_y_label,
            x_values=yearly_100.index,
            y_values=yearly_100,
            color="#2ca02c",
            y_limits=(0.0, 100.0),
        )
        self._render_line_series(
            output_png=cumulative_output_95_png,
            output_svg=cumulative_output_95_svg,
            title=self.config.nmr_monomer_xray_homolog_95_cumulative_title,
            y_label=self.config.nmr_monomer_xray_homolog_y_label,
            x_values=cumulative_share_95.index,
            y_values=cumulative_share_95,
            color="#1f77b4",
            y_limits=(0.0, 100.0),
        )
        self._render_line_series(
            output_png=cumulative_output_100_png,
            output_svg=cumulative_output_100_svg,
            title=self.config.nmr_monomer_xray_homolog_100_cumulative_title,
            y_label=self.config.nmr_monomer_xray_homolog_y_label,
            x_values=cumulative_share_100.index,
            y_values=cumulative_share_100,
            color="#2ca02c",
            y_limits=(0.0, 100.0),
        )

    def plot_solution_nmr_monomer_xray_homologs_historical(
        self,
        data_95_path: Path,
        data_100_path: Path,
        output_95_png: Path,
        output_95_svg: Path,
        output_100_png: Path,
        output_100_svg: Path,
        cumulative_output_95_png: Path,
        cumulative_output_95_svg: Path,
        cumulative_output_100_png: Path,
        cumulative_output_100_svg: Path,
    ) -> None:
        """Render shares for X-ray homologs available by NMR deposition time."""
        table_95 = self._prepare_monomer_xray_homolog_table(
            self._read_csv(data_95_path)
        )
        table_100 = self._prepare_monomer_xray_homolog_table(
            self._read_csv(data_100_path)
        )
        self._scientific_style()

        yearly_95, cumulative_share_95 = self._homolog_share_series(table_95)
        yearly_100, cumulative_share_100 = self._homolog_share_series(table_100)

        self._render_bar_series(
            output_png=output_95_png,
            output_svg=output_95_svg,
            title=self.config.nmr_monomer_xray_homolog_95_historical_title,
            y_label=self.config.nmr_monomer_xray_homolog_y_label,
            x_values=yearly_95.index,
            y_values=yearly_95,
            color="#1f77b4",
            y_limits=(0.0, 100.0),
        )
        self._render_bar_series(
            output_png=output_100_png,
            output_svg=output_100_svg,
            title=self.config.nmr_monomer_xray_homolog_100_historical_title,
            y_label=self.config.nmr_monomer_xray_homolog_y_label,
            x_values=yearly_100.index,
            y_values=yearly_100,
            color="#2ca02c",
            y_limits=(0.0, 100.0),
        )
        self._render_line_series(
            output_png=cumulative_output_95_png,
            output_svg=cumulative_output_95_svg,
            title=self.config.nmr_monomer_xray_homolog_95_historical_cumulative_title,
            y_label=self.config.nmr_monomer_xray_homolog_y_label,
            x_values=cumulative_share_95.index,
            y_values=cumulative_share_95,
            color="#1f77b4",
            y_limits=(0.0, 100.0),
        )
        self._render_line_series(
            output_png=cumulative_output_100_png,
            output_svg=cumulative_output_100_svg,
            title=self.config.nmr_monomer_xray_homolog_100_historical_cumulative_title,
            y_label=self.config.nmr_monomer_xray_homolog_y_label,
            x_values=cumulative_share_100.index,
            y_values=cumulative_share_100,
            color="#2ca02c",
            y_limits=(0.0, 100.0),
        )

    def plot_solution_nmr_monomer_xray_homolog_timing_share(
        self,
        regular_data_95_path: Path,
        regular_data_100_path: Path,
        historical_data_95_path: Path,
        historical_data_100_path: Path,
        counts_output_95_csv: Path,
        counts_output_100_csv: Path,
        output_95_png: Path,
        output_95_svg: Path,
        output_100_png: Path,
        output_100_svg: Path,
    ) -> None:
        """Write timing counts and render homolog-availability share plots."""
        self._scientific_style()

        counts_95 = self._build_xray_homolog_timing_count_table(
            regular_table=self._prepare_monomer_xray_homolog_table(
                self._read_csv(regular_data_95_path), filter_rows=False
            ),
            historical_table=self._prepare_monomer_xray_homolog_table(
                self._read_csv(historical_data_95_path), filter_rows=False
            ),
        )
        counts_100 = self._build_xray_homolog_timing_count_table(
            regular_table=self._prepare_monomer_xray_homolog_table(
                self._read_csv(regular_data_100_path), filter_rows=False
            ),
            historical_table=self._prepare_monomer_xray_homolog_table(
                self._read_csv(historical_data_100_path), filter_rows=False
            ),
        )
        self._write_xray_homolog_timing_counts_csv(
            counts_95,
            counts_output_95_csv,
        )
        self._write_xray_homolog_timing_counts_csv(
            counts_100,
            counts_output_100_csv,
        )

        self._render_homolog_timing_stackplot(
            table=self._xray_homolog_timing_share_from_counts(counts_95),
            output_png=output_95_png,
            output_svg=output_95_svg,
            title=self.config.nmr_monomer_xray_homolog_95_timing_share_title,
        )
        self._render_homolog_timing_stackplot(
            table=self._xray_homolog_timing_share_from_counts(counts_100),
            output_png=output_100_png,
            output_svg=output_100_svg,
            title=self.config.nmr_monomer_xray_homolog_100_timing_share_title,
        )

    @staticmethod
    def _residue_id_text(value: object) -> str:
        """Return a residue ID cell as text, whether read as a number or not."""
        if pd.isna(value):
            return ""
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value).strip()

    @staticmethod
    def _homolog_entity_id_set(value: object) -> frozenset[str]:
        """Parse a semicolon-separated X-ray entity-ID cell."""
        if pd.isna(value):
            return frozenset()
        return frozenset(item.strip() for item in str(value).split(";") if item.strip())

    @classmethod
    def _build_xray_homolog_timing_count_table(
        cls,
        regular_table: pd.DataFrame,
        historical_table: pd.DataFrame,
    ) -> pd.DataFrame:
        """Check unfiltered current and historical searches, then count timings."""
        metadata_columns = [
            "year",
            "sequence_identity_percent",
            "nmr_core_start_seq_id",
            "nmr_core_end_seq_id",
            "nmr_query_sequence_length",
        ]
        columns = [
            "entry_id",
            *metadata_columns,
            "has_xray_homolog",
            "xray_homolog_entity_ids",
        ]
        for table, name in (
            (regular_table, "current homolog CSV"),
            (historical_table, "historical homolog CSV"),
        ):
            cls._validate_required_columns(table, set(columns), name)
            duplicates = table.loc[table["entry_id"].duplicated(), "entry_id"]
            if not duplicates.empty:
                raise ValueError(f"Duplicate entry_id in {name}: {duplicates.iloc[0]}")

        regular, historical = (
            table[columns].assign(
                **{
                    column: table[column].map(cls._residue_id_text)
                    for column in ("nmr_core_start_seq_id", "nmr_core_end_seq_id")
                }
            )
            for table in (regular_table, historical_table)
        )
        merged = regular.merge(
            historical,
            on="entry_id",
            how="outer",
            suffixes=("", "_historical"),
            indicator=True,
        )
        regenerate = " Regenerate current and historical homolog CSVs together."
        historical_only = merged.loc[merged["_merge"] == "right_only", "entry_id"]
        if not historical_only.empty:
            raise ValueError(
                f"Historical homolog record {sorted(historical_only)[0]} is absent "
                f"from the current homolog CSV.{regenerate}"
            )
        # The historical builder drops short queries and records with candidates
        # whose dates are incomplete; the latter are excluded from the counts.
        missing = merged["_merge"] == "left_only"
        missing_without_candidates = merged.loc[
            missing
            & ~merged["has_xray_homolog"].astype(bool)
            & (merged["nmr_query_sequence_length"] > 10),
            "entry_id",
        ]
        if not missing_without_candidates.empty:
            raise ValueError(
                f"Current homolog record {sorted(missing_without_candidates)[0]} "
                f"has no candidates but is absent from the historical homolog "
                f"CSV.{regenerate}"
            )
        for row in merged.loc[~missing].itertuples(index=False):
            current_metadata = tuple(
                getattr(row, column) for column in metadata_columns
            )
            historical_metadata = tuple(
                getattr(row, f"{column}_historical") for column in metadata_columns
            )
            current_ids = cls._homolog_entity_id_set(row.xray_homolog_entity_ids)
            historical_ids = cls._homolog_entity_id_set(
                row.xray_homolog_entity_ids_historical
            )
            if (
                current_metadata != historical_metadata
                or not historical_ids <= current_ids
                or (row.has_xray_homolog_historical and not row.has_xray_homolog)
            ):
                raise ValueError(
                    "Historical homolog CSV is stale or inconsistent for "
                    f"{row.entry_id}.{regenerate}"
                )
        merged = cls._filter_monomer_xray_homolog_rows(merged.loc[~missing])
        merged = merged.rename(
            columns={
                "has_xray_homolog": "has_any_xray_homolog",
                "has_xray_homolog_historical": "has_historical_xray_homolog",
            }
        )
        merged["status"] = XRAY_HOMOLOG_TIMING_LABELS[2]
        merged.loc[
            merged["has_any_xray_homolog"].astype(bool),
            "status",
        ] = XRAY_HOMOLOG_TIMING_LABELS[1]
        merged.loc[
            merged["has_historical_xray_homolog"].astype(bool),
            "status",
        ] = XRAY_HOMOLOG_TIMING_LABELS[0]

        return (
            merged.groupby(["year", "status"], observed=False)
            .size()
            .unstack(fill_value=0)
            .reindex(columns=XRAY_HOMOLOG_TIMING_LABELS, fill_value=0)
            .sort_index()
        )

    @staticmethod
    def _xray_homolog_timing_share_from_counts(counts: pd.DataFrame) -> pd.DataFrame:
        """Convert yearly homolog timing counts into percentage shares."""
        return counts.div(counts.sum(axis=1), axis=0).fillna(0.0) * 100.0

    @staticmethod
    def _write_xray_homolog_timing_counts_csv(
        counts: pd.DataFrame,
        output_path: Path,
    ) -> None:
        """Write yearly homolog timing counts to ``output_path``."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output = (
            counts.rename(
                columns={
                    XRAY_HOMOLOG_TIMING_LABELS[0]: (
                        "already_released_xray_homolog_count"
                    ),
                    XRAY_HOMOLOG_TIMING_LABELS[1]: (
                        "later_released_xray_homolog_count"
                    ),
                    XRAY_HOMOLOG_TIMING_LABELS[2]: "no_xray_homolog_count",
                }
            )
            .reset_index()
            .rename(columns={"index": "year"})
        )
        output.to_csv(output_path, index=False)
