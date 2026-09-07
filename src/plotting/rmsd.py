"""Plot NMR-to-X-ray RMSD summaries and best/worst homolog comparisons."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class RMSDPlotsMixin:
    """Plot NMR-to-X-ray RMSD summaries and best/worst homolog comparisons."""

    @classmethod
    def _prepare_monomer_xray_rmsd_table(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and normalize NMR-to-X-ray RMSD records."""
        prepared = cls._prepare_typed_table(
            df=df,
            required_columns={"entry_id", "year", "rmsd_ca_angstrom"},
            column_types={"year": int, "rmsd_ca_angstrom": float},
            dataset_name="Monomer X-ray RMSD CSV",
        )
        return cls._limit_year_column(prepared)

    @classmethod
    def _prepare_monomer_xray_rmsd_extremes_table(
        cls,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Validate and normalize best- and worst-match RMSD records."""
        prepared = cls._prepare_typed_table(
            df=df,
            required_columns={
                "entry_id",
                "year",
                "best_rmsd_ca_angstrom",
                "worst_rmsd_ca_angstrom",
            },
            column_types={
                "year": int,
                "best_rmsd_ca_angstrom": float,
                "worst_rmsd_ca_angstrom": float,
            },
            dataset_name="Monomer X-ray RMSD extremes CSV",
        )
        return cls._limit_year_column(prepared)

    @staticmethod
    def _xray_rmsd_extremes_yearly_table(
        rmsd_table: pd.DataFrame,
        extremes_table: pd.DataFrame,
        statistic: str,
    ) -> pd.DataFrame:
        """Aggregate three X-ray RMSD selection strategies by year."""
        regular = (
            rmsd_table.groupby("year", as_index=True)["rmsd_ca_angstrom"]
            .agg(statistic)
            .rename("best_resolution_rmsd")
        )
        best = (
            extremes_table.groupby("year", as_index=True)["best_rmsd_ca_angstrom"]
            .agg(statistic)
            .rename("best_rmsd")
        )
        worst = (
            extremes_table.groupby("year", as_index=True)["worst_rmsd_ca_angstrom"]
            .agg(statistic)
            .rename("worst_rmsd")
        )
        return pd.concat([regular, best, worst], axis=1).sort_index()

    def plot_solution_nmr_monomer_xray_rmsd(
        self,
        data_path: Path,
        extremes_data_path: Path,
        mean_output_png: Path,
        mean_output_svg: Path,
        median_output_png: Path,
        median_output_svg: Path,
        min_mean_output_png: Path,
        min_mean_output_svg: Path,
        min_median_output_png: Path,
        min_median_output_svg: Path,
        extremes_mean_output_png: Path,
        extremes_mean_output_svg: Path,
        extremes_median_output_png: Path,
        extremes_median_output_svg: Path,
        title_suffix: str = "",
    ) -> None:
        """Render yearly NMR-to-X-ray RMSD summaries and comparisons."""
        table = self._prepare_monomer_xray_rmsd_table(self._read_csv(data_path))
        extremes_table = self._prepare_monomer_xray_rmsd_extremes_table(
            self._read_csv(extremes_data_path)
        )
        title = (
            (lambda value: f"{value} {title_suffix}")
            if title_suffix
            else (lambda value: value)
        )
        self._scientific_style()
        yearly_rmsd = (
            table.groupby("year", as_index=True)["rmsd_ca_angstrom"]
            .agg(["mean", "median"])
            .sort_index()
        )
        self._render_bar_series(
            output_png=mean_output_png,
            output_svg=mean_output_svg,
            title=title(self.config.nmr_monomer_xray_rmsd_title),
            y_label=self.config.nmr_monomer_xray_rmsd_y_label,
            x_values=yearly_rmsd.index,
            y_values=yearly_rmsd["mean"],
            color="#9467bd",
            y_bottom=0.0,
        )
        self._render_bar_series(
            output_png=median_output_png,
            output_svg=median_output_svg,
            title=title(self.config.nmr_monomer_xray_median_rmsd_title),
            y_label=self.config.nmr_monomer_xray_median_rmsd_y_label,
            x_values=yearly_rmsd.index,
            y_values=yearly_rmsd["median"],
            color=self.config.median_color,
            y_bottom=0.0,
        )
        yearly_min_rmsd = (
            extremes_table.groupby("year", as_index=True)["best_rmsd_ca_angstrom"]
            .agg(["mean", "median"])
            .sort_index()
        )
        self._render_bar_series(
            output_png=min_mean_output_png,
            output_svg=min_mean_output_svg,
            title=title(self.config.nmr_monomer_xray_min_rmsd_title),
            y_label=self.config.nmr_monomer_xray_min_rmsd_y_label,
            x_values=yearly_min_rmsd.index,
            y_values=yearly_min_rmsd["mean"],
            color="#2ca02c",
            y_bottom=0.0,
        )
        self._render_bar_series(
            output_png=min_median_output_png,
            output_svg=min_median_output_svg,
            title=title(self.config.nmr_monomer_xray_min_median_rmsd_title),
            y_label=self.config.nmr_monomer_xray_min_median_rmsd_y_label,
            x_values=yearly_min_rmsd.index,
            y_values=yearly_min_rmsd["median"],
            color=self.config.median_color,
            y_bottom=0.0,
        )
        colors = {
            "best_resolution_rmsd": "#9467bd",
            "best_rmsd": "#2ca02c",
            "worst_rmsd": "#d62728",
        }
        labels = {
            "best_resolution_rmsd": "Best resolution X-ray",
            "best_rmsd": "Minimum RMSD X-ray",
            "worst_rmsd": "Maximum RMSD X-ray",
        }
        draw_order = ["worst_rmsd", "best_rmsd", "best_resolution_rmsd"]
        self._render_multi_line_series(
            output_png=extremes_mean_output_png,
            output_svg=extremes_mean_output_svg,
            title=title(self.config.nmr_monomer_xray_rmsd_extremes_mean_title),
            y_label=self.config.nmr_monomer_xray_rmsd_extremes_y_label,
            table=self._xray_rmsd_extremes_yearly_table(
                rmsd_table=table,
                extremes_table=extremes_table,
                statistic="mean",
            ),
            colors=colors,
            labels=labels,
            y_bottom=0.0,
            use_step=True,
            draw_order=draw_order,
        )
        self._render_multi_line_series(
            output_png=extremes_median_output_png,
            output_svg=extremes_median_output_svg,
            title=title(self.config.nmr_monomer_xray_rmsd_extremes_median_title),
            y_label=self.config.nmr_monomer_xray_rmsd_extremes_y_label,
            table=self._xray_rmsd_extremes_yearly_table(
                rmsd_table=table,
                extremes_table=extremes_table,
                statistic="median",
            ),
            colors=colors,
            labels=labels,
            y_bottom=0.0,
            use_step=True,
            draw_order=draw_order,
        )
