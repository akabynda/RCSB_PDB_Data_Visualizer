"""Plot annual and cumulative experimental-method and membrane counts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


class CountPlotsMixin:
    """Plot annual and cumulative experimental-method and membrane counts."""

    @classmethod
    def _prepare_method_count_table(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and pivot raw experimental-method counts by year."""
        prepared = cls._prepare_typed_table(
            df=df,
            required_columns={"year", "method", "count"},
            column_types={"year": int, "method": str, "count": int},
            dataset_name="Method count CSV",
        )
        limited = cls._limit_year_column(prepared)
        return (
            limited.pivot(index="year", columns="method", values="count")
            .fillna(0)
            .sort_index()
            .astype(int)
        )

    @classmethod
    def _prepare_membrane_count_table(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and normalize yearly membrane-protein counts."""
        prepared = cls._prepare_typed_table(
            df=df,
            required_columns={"year", "count"},
            column_types={"year": int, "count": int},
            dataset_name="Membrane count CSV",
        )
        return cls._limit_year_column(prepared).sort_values("year")

    def plot_method_counts(
        self,
        data_path: Path,
        annual_output_png: Path,
        annual_output_svg: Path,
        cumulative_output_png: Path,
        cumulative_output_svg: Path,
    ) -> None:
        """Render annual and cumulative experimental-method counts from CSV."""
        table = self._prepare_method_count_table(self._read_csv(data_path))
        cumulative_table = table.cumsum()
        self._scientific_style()

        def draw(ax: plt.Axes, source: pd.DataFrame, use_step: bool) -> None:
            """Draw the available method columns from a yearly table."""
            for col, color in [
                ("X-ray", self.config.xray_color),
                ("NMR", self.config.nmr_color),
                ("cryo-EM", self.config.cryoem_color),
            ]:
                if col in source.columns:
                    if use_step:
                        self._plot_step_series(
                            ax=ax,
                            x_values=source.index,
                            y_values=source[col],
                            color=color,
                            linewidth=2.0,
                            label=col,
                        )
                    else:
                        ax.plot(
                            source.index,
                            source[col],
                            color=color,
                            linewidth=2.0,
                            label=col,
                        )
            self._add_legend(ax, loc="upper left")

        self._render_figure(
            annual_output_png,
            annual_output_svg,
            self.config.annual_title,
            self.config.annual_y_label,
            lambda ax: draw(ax, table, use_step=True),
        )
        self._render_figure(
            cumulative_output_png,
            cumulative_output_svg,
            self.config.cumulative_title,
            self.config.cumulative_y_label,
            lambda ax: draw(ax, cumulative_table, use_step=False),
        )

    def _plot_method_count_table(
        self,
        table: pd.DataFrame,
        annual_output_png: Path,
        annual_output_svg: Path,
        cumulative_output_png: Path,
        cumulative_output_svg: Path,
        annual_title: str,
        annual_y_label: str,
        cumulative_title: str,
        cumulative_y_label: str,
    ) -> None:
        """Render annual and cumulative plots from a prepared method table."""
        cumulative_table = table.cumsum()
        self._scientific_style()

        def draw(ax: plt.Axes, source: pd.DataFrame, use_step: bool) -> None:
            """Draw the prepared method series as steps or lines."""
            for col, color in [
                ("X-ray", self.config.xray_color),
                ("NMR", self.config.nmr_color),
                ("cryo-EM", self.config.cryoem_color),
            ]:
                if col in source.columns:
                    if use_step:
                        self._plot_step_series(
                            ax=ax,
                            x_values=source.index,
                            y_values=source[col],
                            color=color,
                            linewidth=2.0,
                            label=col,
                        )
                    else:
                        ax.plot(
                            source.index,
                            source[col],
                            color=color,
                            linewidth=2.0,
                            label=col,
                        )
            self._add_legend(ax, loc="upper left")

        self._render_figure(
            annual_output_png,
            annual_output_svg,
            annual_title,
            annual_y_label,
            lambda ax: draw(ax, table, use_step=True),
        )
        self._render_figure(
            cumulative_output_png,
            cumulative_output_svg,
            cumulative_title,
            cumulative_y_label,
            lambda ax: draw(ax, cumulative_table, use_step=False),
        )

    def plot_membrane_protein_counts(
        self,
        data_path: Path,
        method_data_path: Path,
        annual_output_png: Path,
        annual_output_svg: Path,
        cumulative_output_png: Path,
        cumulative_output_svg: Path,
        method_annual_output_png: Path,
        method_annual_output_svg: Path,
        method_cumulative_output_png: Path,
        method_cumulative_output_svg: Path,
    ) -> None:
        """Render overall and method-specific membrane-protein counts."""
        table = self._prepare_membrane_count_table(self._read_csv(data_path))
        cumulative = table.copy()
        cumulative["count"] = cumulative["count"].cumsum()
        self._scientific_style()
        self._render_bar_series(
            output_png=annual_output_png,
            output_svg=annual_output_svg,
            title=self.config.membrane_annual_title,
            y_label=self.config.membrane_annual_y_label,
            x_values=table["year"],
            y_values=table["count"],
            color="#17becf",
            y_bottom=0.0,
        )
        self._render_line_series(
            output_png=cumulative_output_png,
            output_svg=cumulative_output_svg,
            title=self.config.membrane_cumulative_title,
            y_label=self.config.membrane_cumulative_y_label,
            x_values=cumulative["year"],
            y_values=cumulative["count"],
            color="#17becf",
        )
        method_table = self._prepare_method_count_table(
            self._read_csv(method_data_path)
        )
        self._plot_method_count_table(
            table=method_table,
            annual_output_png=method_annual_output_png,
            annual_output_svg=method_annual_output_svg,
            cumulative_output_png=method_cumulative_output_png,
            cumulative_output_svg=method_cumulative_output_svg,
            annual_title=self.config.membrane_method_annual_title,
            annual_y_label=self.config.membrane_annual_y_label,
            cumulative_title=self.config.membrane_method_cumulative_title,
            cumulative_y_label=self.config.membrane_cumulative_y_label,
        )
