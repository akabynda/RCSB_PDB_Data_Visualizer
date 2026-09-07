"""Plot NMR secondary structure, ensemble precision, and validation metrics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


class QualityPlotsMixin:
    """Plot NMR secondary structure, ensemble precision, and validation metrics."""

    @classmethod
    def _prepare_monomer_stride_modeled_first_model_table(
        cls,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Validate and normalize first-model STRIDE composition records."""
        prepared = cls._prepare_typed_table(
            df=df,
            required_columns={
                "entry_id",
                "year",
                "stride_alpha_helix_fraction",
                "stride_3_10_helix_fraction",
                "stride_pi_helix_fraction",
                "stride_beta_strand_fraction",
                "stride_isolated_beta_bridge_fraction",
            },
            column_types={
                "year": int,
                "stride_alpha_helix_fraction": float,
                "stride_3_10_helix_fraction": float,
                "stride_pi_helix_fraction": float,
                "stride_beta_strand_fraction": float,
                "stride_isolated_beta_bridge_fraction": float,
            },
            dataset_name="Monomer STRIDE CSV",
        )
        return cls._limit_year_column(prepared)

    def plot_solution_nmr_monomer_stride_modeled_first_model(
        self, data_path: Path, output_png: Path, output_svg: Path
    ) -> None:
        """Render individual and annual mean structured-residue percentages."""
        table = self._prepare_monomer_stride_modeled_first_model_table(
            self._read_csv(data_path)
        )
        self._scientific_style()
        table = table.copy()
        table["stride_hgieb_percent"] = (
            table["stride_alpha_helix_fraction"]
            + table["stride_3_10_helix_fraction"]
            + table["stride_pi_helix_fraction"]
            + table["stride_beta_strand_fraction"]
            + table["stride_isolated_beta_bridge_fraction"]
        ) * 100.0
        filtered = table.loc[
            (table["stride_hgieb_percent"] >= 0.0)
            & (table["stride_hgieb_percent"] <= 100.0)
        ].copy()
        yearly_mean = (
            filtered.groupby("year", as_index=True)["stride_hgieb_percent"]
            .mean()
            .sort_index()
        )

        def draw(ax: plt.Axes) -> None:
            """Draw entry-level observations and the yearly mean series."""
            ax.scatter(
                filtered["year"],
                filtered["stride_hgieb_percent"],
                s=10,
                alpha=0.2,
                color="#7f7f7f",
                label="Individual structures",
                clip_on=False,
                zorder=3,
            )
            self._plot_step_series(
                ax=ax,
                x_values=yearly_mean.index,
                y_values=yearly_mean.values,
                linewidth=2.2,
                color=self.config.nmr_color,
                label="Yearly mean",
                zorder=4,
            )
            ax.set_ylim(0, 100)
            self._add_legend(ax, loc="upper left")

        self._render_figure(
            output_png=output_png,
            output_svg=output_svg,
            title=self.config.nmr_monomer_stride_modeled_first_model_title,
            y_label=self.config.nmr_monomer_stride_modeled_first_model_y_label,
            draw_fn=draw,
        )

    @classmethod
    def _prepare_monomer_precision_table(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and normalize NMR ensemble-precision records."""
        prepared = cls._prepare_typed_table(
            df=df,
            required_columns={"entry_id", "year", "mean_rmsd_angstrom"},
            column_types={"year": int, "mean_rmsd_angstrom": float},
            dataset_name="Monomer precision CSV",
        )
        return cls._limit_year_column(prepared)

    def _plot_solution_nmr_monomer_precision_stat(
        self,
        data_path: Path,
        output_png: Path,
        output_svg: Path,
        statistic: str,
        title: str,
        y_label: str,
    ) -> None:
        """Aggregate and render one annual ensemble-precision statistic."""
        table = self._prepare_monomer_precision_table(self._read_csv(data_path))
        self._scientific_style()
        yearly_rmsd = (
            table.groupby("year", as_index=True)["mean_rmsd_angstrom"]
            .agg(statistic)
            .sort_index()
        )

        self._render_bar_series(
            output_png=output_png,
            output_svg=output_svg,
            title=title,
            y_label=y_label,
            x_values=yearly_rmsd.index,
            y_values=yearly_rmsd,
            color="#8c564b",
            y_bottom=0.0,
        )

    def plot_solution_nmr_monomer_precision_stride_modeled_first_model_mean(
        self, data_path: Path, output_png: Path, output_svg: Path
    ) -> None:
        """Render annual mean precision for STRIDE-defined modeled cores."""
        self._plot_solution_nmr_monomer_precision_stat(
            data_path=data_path,
            output_png=output_png,
            output_svg=output_svg,
            statistic="mean",
            title=self.config.nmr_monomer_precision_stride_mean_title,
            y_label=self.config.nmr_monomer_precision_stride_mean_y_label,
        )

    def plot_solution_nmr_monomer_precision_stride_modeled_first_model_median(
        self, data_path: Path, output_png: Path, output_svg: Path
    ) -> None:
        """Render annual median precision for STRIDE-defined modeled cores."""
        self._plot_solution_nmr_monomer_precision_stat(
            data_path=data_path,
            output_png=output_png,
            output_svg=output_svg,
            statistic="median",
            title=self.config.nmr_monomer_precision_stride_median_title,
            y_label=self.config.nmr_monomer_precision_stride_median_y_label,
        )

    @classmethod
    def _prepare_monomer_quality_table(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and normalize solution-NMR validation metrics."""
        prepared = cls._prepare_typed_table(
            df=df,
            required_columns={
                "entry_id",
                "year",
                "clashscore",
                "ramachandran_outliers_percent",
                "sidechain_outliers_percent",
            },
            column_types={
                "year": int,
                "clashscore": float,
                "ramachandran_outliers_percent": float,
                "sidechain_outliers_percent": float,
            },
            dataset_name="Monomer quality CSV",
        )
        return cls._limit_year_column(prepared)

    def plot_solution_nmr_monomer_quality(
        self,
        data_path: Path,
        clash_output_png: Path,
        clash_output_svg: Path,
        rama_output_png: Path,
        rama_output_svg: Path,
        side_output_png: Path,
        side_output_svg: Path,
    ) -> None:
        """Render annual clash, Ramachandran, and side-chain quality means."""
        table = self._prepare_monomer_quality_table(self._read_csv(data_path))
        self._scientific_style()
        yearly = (
            table.groupby("year", as_index=True).mean(numeric_only=True).sort_index()
        )

        self._render_bar_series(
            output_png=clash_output_png,
            output_svg=clash_output_svg,
            title=self.config.nmr_monomer_quality_clash_title,
            y_label=self.config.nmr_monomer_quality_clash_y_label,
            x_values=yearly.index,
            y_values=yearly["clashscore"],
            color="#8c564b",
            y_bottom=0.0,
        )
        self._render_bar_series(
            output_png=rama_output_png,
            output_svg=rama_output_svg,
            title=self.config.nmr_monomer_quality_rama_title,
            y_label=self.config.nmr_monomer_quality_rama_y_label,
            x_values=yearly.index,
            y_values=yearly["ramachandran_outliers_percent"],
            color="#1f77b4",
            y_bottom=0.0,
        )
        self._render_bar_series(
            output_png=side_output_png,
            output_svg=side_output_svg,
            title=self.config.nmr_monomer_quality_side_title,
            y_label=self.config.nmr_monomer_quality_side_y_label,
            x_values=yearly.index,
            y_values=yearly["sidechain_outliers_percent"],
            color="#ff7f0e",
            y_bottom=0.0,
        )
