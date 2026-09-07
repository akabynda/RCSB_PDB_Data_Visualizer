"""Compare ensemble precision with X-ray RMSD by entry and deposition year."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator


class CorrelationPlotsMixin:
    """Compare ensemble precision with X-ray RMSD by entry and deposition year."""

    @classmethod
    def _prepare_xray_rmsd_precision_correlation_table(
        cls,
        precision_df: pd.DataFrame,
        extremes_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Merge precision and minimum X-ray RMSD values by entry identifier."""
        precision = cls._prepare_monomer_precision_table(precision_df)[
            ["entry_id", "year", "mean_rmsd_angstrom"]
        ].copy()
        extremes = cls._prepare_monomer_xray_rmsd_extremes_table(
            extremes_df
        )[["entry_id", "best_rmsd_ca_angstrom"]].copy()
        precision = precision.drop_duplicates(subset=["entry_id"], keep="first")
        extremes = extremes.drop_duplicates(subset=["entry_id"], keep="first")
        merged = precision.merge(extremes, on="entry_id", how="inner")
        merged = merged.rename(
            columns={
                "mean_rmsd_angstrom": "precision_rmsd",
                "best_rmsd_ca_angstrom": "min_xray_rmsd",
            }
        )
        merged = merged.replace([np.inf, -np.inf], np.nan).dropna(
            subset=["year", "precision_rmsd", "min_xray_rmsd"]
        )
        return cls._limit_year_column(merged)

    def plot_solution_nmr_monomer_xray_rmsd_precision_correlation(
        self,
        precision_data_path: Path,
        extremes_data_path: Path,
        scatter_output_png: Path,
        scatter_output_svg: Path,
        yearly_correlation_output_png: Path,
        yearly_correlation_output_svg: Path,
        cumulative_correlation_output_png: Path,
        cumulative_correlation_output_svg: Path,
        yearly_min_count: int = 3,
    ) -> None:
        """Render scatter, yearly, and cumulative RMSD correlations."""
        table = self._prepare_xray_rmsd_precision_correlation_table(
            precision_df=self._read_csv(precision_data_path),
            extremes_df=self._read_csv(extremes_data_path),
        )
        if table.empty:
            raise ValueError(
                "No overlapping entries between precision and X-ray RMSD extremes CSVs."
            )
        self._scientific_style()

        pearson = float(table["precision_rmsd"].corr(table["min_xray_rmsd"]))
        spearman = float(
            table["precision_rmsd"].rank().corr(table["min_xray_rmsd"].rank())
        )
        x = table["precision_rmsd"].to_numpy(dtype=float)
        y = table["min_xray_rmsd"].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x, y, 1) if len(table) >= 2 else (np.nan, np.nan)
        axis_limit = max(float(np.max(x)), float(np.max(y)))
        marker_area_points = 12.0
        marker_linewidth_points = 1.0

        def draw_scatter(ax: plt.Axes) -> None:
            """Draw entry-level RMSDs, linear regression, and correlations."""
            ax.scatter(
                table["precision_rmsd"],
                table["min_xray_rmsd"],
                s=marker_area_points,
                alpha=1.0,
                facecolors="none",
                edgecolors="#4c78a8",
                linewidths=marker_linewidth_points,
                clip_on=False,
                zorder=3,
            )
            if len(table) >= 2:
                x_values = np.linspace(0.0, axis_limit, 100)
                ax.plot(
                    x_values,
                    slope * x_values + intercept,
                    color="#d62728",
                    linewidth=2.0,
                    zorder=4,
                )
            # Convert the marker radius plus one full outline width from points
            # to data units. Add that padding to both upper limits so circles on
            # either maximum remain fully visible beyond their outlines.
            ax.set_xlim(0.0, axis_limit)
            ax.set_ylim(0.0, axis_limit)
            ax.set_aspect("equal", adjustable="box")
            ax.figure.canvas.draw()
            marker_padding_pixels = (
                np.sqrt(marker_area_points) / 2.0 + marker_linewidth_points
            ) * ax.figure.dpi / 72.0
            axes_extent = ax.get_window_extent()
            axes_size_pixels = min(axes_extent.width, axes_extent.height)
            marker_padding_data = (
                axis_limit
                * marker_padding_pixels
                / (axes_size_pixels - marker_padding_pixels)
            )
            padded_axis_limit = axis_limit + marker_padding_data
            ax.set_xlim(0.0, padded_axis_limit)
            ax.set_ylim(0.0, padded_axis_limit)
            ax.xaxis.set_major_locator(MultipleLocator(5.0))
            ax.yaxis.set_major_locator(MultipleLocator(5.0))
            ax.text(
                0.96,
                0.94,
                f"n = {len(table)}\n$r$ = {pearson:.2f}\n$\\rho$ = {spearman:.2f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=12,
                bbox={
                    "boxstyle": "square,pad=0.25",
                    "facecolor": "white",
                    "edgecolor": "black",
                    "linewidth": 0.6,
                    "alpha": 0.95,
                },
            )

        self._render_figure(
            output_png=scatter_output_png,
            output_svg=scatter_output_svg,
            title=self.config.nmr_monomer_xray_rmsd_precision_scatter_title,
            y_label=r"RMSD$_{\mathrm{NMR,X\!-\!ray}}$ (Å)",
            x_label=r"RMSD$_{\mathrm{NMR}}$ (Å)",
            draw_fn=draw_scatter,
            use_year_x_ticks=False,
            height_scale=self.config.aspect_ratio,
        )

        yearly_correlations: dict[int, float] = {}
        for year, group in table.groupby("year"):
            if len(group) < yearly_min_count:
                continue
            if (
                group["precision_rmsd"].nunique() < 2
                or group["min_xray_rmsd"].nunique() < 2
            ):
                continue
            yearly_correlations[int(year)] = float(
                group["precision_rmsd"].corr(group["min_xray_rmsd"])
            )
        yearly_corr = pd.Series(yearly_correlations, dtype=float).sort_index()

        def draw_yearly_corr(ax: plt.Axes) -> None:
            """Draw within-year Pearson correlations above the sample cutoff."""
            ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
            self._plot_step_series(
                ax=ax,
                x_values=yearly_corr.index,
                y_values=yearly_corr.values,
                color="#9467bd",
                linewidth=2.2,
                label=f"Pearson r, n>={yearly_min_count}",
            )
            ax.set_ylim(-1.0, 1.0)
            self._add_legend(ax, loc="lower right")

        self._render_figure(
            output_png=yearly_correlation_output_png,
            output_svg=yearly_correlation_output_svg,
            title=self.config.nmr_monomer_xray_rmsd_precision_yearly_corr_title,
            y_label="Pearson correlation",
            draw_fn=draw_yearly_corr,
        )

        cumulative_correlations: dict[int, float] = {}
        for year in sorted(table["year"].unique()):
            subset = table.loc[table["year"] <= year]
            if len(subset) < yearly_min_count:
                continue
            if (
                subset["precision_rmsd"].nunique() < 2
                or subset["min_xray_rmsd"].nunique() < 2
            ):
                continue
            cumulative_correlations[int(year)] = float(
                subset["precision_rmsd"].corr(subset["min_xray_rmsd"])
            )
        cumulative_corr = pd.Series(cumulative_correlations, dtype=float).sort_index()

        def draw_cumulative_corr(ax: plt.Axes) -> None:
            """Draw correlations accumulated through each deposition year."""
            ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
            self._plot_step_series(
                ax=ax,
                x_values=cumulative_corr.index,
                y_values=cumulative_corr.values,
                color="#2ca02c",
                linewidth=2.2,
                label=f"Cumulative Pearson r, n>={yearly_min_count}",
            )
            ax.set_ylim(-1.0, 1.0)
            self._add_legend(ax, loc="lower right")

        self._render_figure(
            output_png=cumulative_correlation_output_png,
            output_svg=cumulative_correlation_output_svg,
            title=self.config.nmr_monomer_xray_rmsd_precision_cumulative_corr_title,
            y_label="Pearson correlation",
            draw_fn=draw_cumulative_corr,
        )
