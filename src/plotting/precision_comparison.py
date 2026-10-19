"""Compare core and secondary-structure precision alongside Figure 4."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import AutoMinorLocator, MaxNLocator


CORE_COLOR = "#CC79A7"
SECONDARY_COLOR = "#0072B2"
HISTOGRAM_EDGES = np.linspace(0.0, 3.0, 31)
RMSD_LABEL = r"RMSD$_{\mathrm{NMR}}$ (Å)"


class PrecisionComparisonPlotsMixin:
    """Retain the original core trend and add a distribution comparison."""

    @classmethod
    def _prepare_precision_comparison_tables(
        cls, core_df: pd.DataFrame, secondary_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Use the Figure 4 cohort, retaining core rows without secondary RMSD."""
        tables = []
        for frame, selection in (
            (core_df, "core"),
            (secondary_df, "secondary-structure"),
        ):
            table = cls._prepare_monomer_precision_table(frame)
            if table.empty:
                raise ValueError(
                    f"No {selection} precision entries in the plotting years"
                )
            modes = (
                set(table["residue_selection"].fillna("core"))
                if "residue_selection" in table
                else {"core"}
            )
            if modes != {selection}:
                raise ValueError(f"Expected {selection} precision data; found {modes}")
            if table["entry_id"].isna().any() or table["entry_id"].duplicated().any():
                raise ValueError(
                    f"Missing or duplicate entry_id in {selection} precision"
                )
            values = table["mean_rmsd_angstrom"].to_numpy()
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f"Invalid RMSD values in {selection} precision")
            tables.append(table)
        core, secondary = tables
        secondary = secondary.loc[secondary["entry_id"].isin(core["entry_id"])].copy()
        if secondary.empty:
            raise ValueError("No secondary-structure results for the Figure 4 cohort")
        core_by_id = core.set_index("entry_id").loc[secondary["entry_id"]]
        for column in ("year", "chain_id", "core_start_seq_id", "core_end_seq_id"):
            if column in core and column in secondary:
                if not np.array_equal(
                    core_by_id[column].astype(str).to_numpy(),
                    secondary[column].astype(str).to_numpy(),
                ):
                    raise ValueError(
                        f"Core and secondary precision disagree on {column}"
                    )
        return core, secondary

    @staticmethod
    def _precision_histogram_counts(values: np.ndarray) -> tuple[np.ndarray, int]:
        """Count common 0.1 Å bins and a separate ≥3 Å overflow category."""
        values = np.asarray(values, dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Histogram RMSD values must be finite and nonnegative")
        regular = values[values < HISTOGRAM_EDGES[-1]]
        counts, _ = np.histogram(regular, bins=HISTOGRAM_EDGES)
        return counts, int(np.count_nonzero(values >= HISTOGRAM_EDGES[-1]))

    def _draw_precision_histogram(
        self,
        ax: plt.Axes,
        core: pd.DataFrame,
        secondary: pd.DataFrame,
        *,
        fontsize: float = 8.0,
    ) -> None:
        """Overlay counts with a visibly separated, explicitly pooled tail."""
        ax.set_facecolor("white")
        ax.axvspan(3.05, 3.32, color="#f2f2f2", zorder=0)
        for table, color, label in (
            (core, CORE_COLOR, "Core"),
            (secondary, SECONDARY_COLOR, "Sec. struc."),
        ):
            counts, overflow = self._precision_histogram_counts(
                table["mean_rmsd_angstrom"].to_numpy()
            )
            ax.stairs(
                counts,
                HISTOGRAM_EDGES,
                fill=True,
                color=color,
                alpha=0.25,
                linewidth=0,
                zorder=2,
            )
            ax.stairs(
                counts,
                HISTOGRAM_EDGES,
                color=color,
                linewidth=1.1,
                label=f"{label} (n = {len(table):,})",
                zorder=3,
            )
            ax.bar(3.18, overflow, width=0.18, color=color, alpha=0.25, zorder=2)
            ax.bar(
                3.18,
                overflow,
                width=0.18,
                facecolor="none",
                edgecolor=color,
                linewidth=1.0,
                hatch="///",
                zorder=3,
            )
        ax.set_xlim(0, 3.32)
        ax.set_ylim(bottom=0)
        ax.set_xticks([0, 1, 2, 3.18], ["0", "1", "2", "≥3"])
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(axis="both", which="major", labelsize=fontsize, length=3)
        ax.tick_params(axis="both", which="minor", length=2)
        ax.set_xlabel(RMSD_LABEL, fontsize=fontsize, labelpad=2)
        ax.set_ylabel("Number of structures", fontsize=fontsize, labelpad=3)
        ax.grid(axis="x", visible=False)
        ax.grid(axis="y", alpha=0.18)
        ax.legend(
            loc="upper right",
            frameon=False,
            fontsize=fontsize,
            handlelength=1.8,
            labelspacing=0.3,
            borderaxespad=0.4,
        )
        self._configure_boxed_axes(ax)

    def _draw_core_precision_trend(self, ax: plt.Axes, yearly: pd.Series) -> None:
        """Use the same step coordinates, color and statistic as Figure 4."""
        self._plot_step_series(
            ax=ax,
            x_values=yearly.index,
            y_values=yearly,
            color="#8c564b",
            linewidth=2.2,
        )
        ax.set_ylim(bottom=0)
        ax.margins(x=0)

    def _render_precision_comparison_panels(
        self,
        core: pd.DataFrame,
        secondary: pd.DataFrame,
        yearly: pd.Series,
        output_png: Path,
        output_svg: Path,
    ) -> None:
        """Put the histogram below the unchanged core trend as a second panel."""
        for with_title, open_axes in (
            (True, False),
            (False, False),
            (True, True),
            (False, True),
        ):
            fig, (trend, histogram) = plt.subplots(
                2,
                1,
                figsize=self.config.figure_size(1.5),
                gridspec_kw={"height_ratios": [1.1, 1.0]},
            )
            self._draw_core_precision_trend(trend, yearly)
            self._configure_year_axis_ticks(trend)
            self._remove_zero_y_tick(trend)
            self._configure_minor_ticks(trend, use_year_x_ticks=True)
            trend.set_xlabel(self.config.x_label)
            trend.set_ylabel(self.config.nmr_monomer_precision_stride_median_y_label)
            self._draw_precision_histogram(histogram, core, secondary, fontsize=10.0)
            for ax, label in ((trend, "A"), (histogram, "B")):
                self._configure_boxed_axes(ax)
                if open_axes and ax is trend:
                    self._configure_open_axes(ax)
                ax.text(
                    -0.12,
                    1.02,
                    label,
                    transform=ax.transAxes,
                    fontsize=12,
                    fontweight="bold",
                    va="bottom",
                )
            if with_title:
                self._set_title(
                    trend, self.config.nmr_monomer_precision_stride_median_title
                )
                self._set_title(
                    histogram, "Core and secondary-structure RMSD distributions"
                )
            png, svg = output_png, output_svg
            if not with_title:
                png, svg = (
                    self._titleless_output_path(png),
                    self._titleless_output_path(svg),
                )
            if open_axes:
                png, svg = (
                    self._open_axes_output_path(png),
                    self._open_axes_output_path(svg),
                )
            fig.tight_layout(h_pad=2.3)
            self._save_figure_files(fig, png, svg)
            plt.close(fig)

    def plot_solution_nmr_monomer_precision_comparison(
        self,
        core_data_path: Path,
        secondary_data_path: Path,
        output_png: Path,
        output_svg: Path,
    ) -> None:
        """Save standalone histograms and two optional Figure 4 layouts."""
        core, secondary = self._prepare_precision_comparison_tables(
            self._read_csv(core_data_path), self._read_csv(secondary_data_path)
        )
        yearly = core.groupby("year")["mean_rmsd_angstrom"].median().sort_index()
        self._scientific_style()

        def paths(suffix: str) -> tuple[Path, Path]:
            return (
                output_png.with_stem(output_png.stem + suffix),
                output_svg.with_stem(output_svg.stem + suffix),
            )

        def draw_inset(ax: plt.Axes) -> None:
            self._draw_core_precision_trend(ax, yearly)
            ax.set_ylim(0, max(1.6, float(yearly.max()) * 1.17))
            inset = ax.inset_axes([0.29, 0.56, 0.68, 0.39])
            self._draw_precision_histogram(inset, core, secondary)

        png, svg = paths("_with_histogram")
        self._render_figure(
            output_png=png,
            output_svg=svg,
            title=self.config.nmr_monomer_precision_stride_median_title,
            y_label=self.config.nmr_monomer_precision_stride_median_y_label,
            draw_fn=draw_inset,
        )
        png, svg = paths("_histogram")
        self._render_figure(
            output_png=png,
            output_svg=svg,
            title="Core and secondary-structure RMSD distributions",
            y_label="Number of structures",
            x_label=RMSD_LABEL,
            use_year_x_ticks=False,
            force_boxed_axes=True,
            draw_fn=lambda ax: self._draw_precision_histogram(
                ax, core, secondary, fontsize=11.0
            ),
        )
        png, svg = paths("_comparison_panels")
        self._render_precision_comparison_panels(core, secondary, yearly, png, svg)
