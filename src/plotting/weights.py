"""Plot molecular-weight summaries, period distributions, and category shares."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .constants import (
    NMR_WEIGHT_BINS,
    NMR_WEIGHT_LABELS,
)


class WeightPlotsMixin:
    """Plot molecular-weight summaries, period distributions, and category shares."""

    @staticmethod
    def _build_weight_category_yearly_counts(table: pd.DataFrame) -> pd.DataFrame:
        """Return yearly structure counts split into molecular-weight categories."""
        categorized = table.copy()
        categorized["weight_category"] = pd.cut(
            categorized["molecular_weight_kda"],
            bins=NMR_WEIGHT_BINS,
            labels=NMR_WEIGHT_LABELS,
            right=False,
        )
        return (
            categorized.groupby(["year", "weight_category"], observed=False)
            .size()
            .unstack(fill_value=0)
            .reindex(columns=NMR_WEIGHT_LABELS, fill_value=0)
            .sort_index()
        )

    def _render_weight_category_stackplot(
        self,
        table: pd.DataFrame,
        output_png: Path,
        output_svg: Path,
        title: str,
        y_label: str,
        y_limits: tuple[float, float] | None = None,
        x_left: float | None = None,
        x_right: float | None = None,
        use_step_segments: bool = False,
        expand_step_xlim: bool = True,
        legend_loc: str = "upper left",
    ) -> None:
        """Render yearly molecular-weight categories as a stacked area plot."""

        def draw(ax: plt.Axes) -> None:
            """Draw continuous or step-aligned stacked category areas."""
            x_step_edges: np.ndarray | None = None
            if use_step_segments:
                base = pd.Series(0.0, index=table.index, dtype=float)
                x_step_edges = self._step_edges(table.index)
                for idx, label in enumerate(NMR_WEIGHT_LABELS):
                    values = table[label].astype(float)
                    top = base + values
                    color = self.config.area_colors[idx % len(self.config.area_colors)]
                    ax.fill_between(
                        x_step_edges,
                        self._step_values(base),
                        self._step_values(top),
                        step="post",
                        color=color,
                        alpha=0.85,
                        label=label,
                    )
                    base = top
            else:
                ax.stackplot(
                    table.index,
                    *(table[label] for label in NMR_WEIGHT_LABELS),
                    labels=NMR_WEIGHT_LABELS,
                    colors=self.config.area_colors,
                    alpha=0.85,
                )
            if y_limits is not None:
                ax.set_ylim(*y_limits)
            if x_left is not None or x_right is not None:
                current_left, current_right = ax.get_xlim()
                effective_left = x_left if x_left is not None else current_left
                effective_right = x_right if x_right is not None else current_right
                if expand_step_xlim and use_step_segments and x_step_edges is not None:
                    effective_left = min(effective_left, float(x_step_edges[0]))
                    effective_right = max(effective_right, float(x_step_edges[-1]))
                ax.set_xlim(
                    left=effective_left,
                    right=effective_right,
                )
            self._add_legend(ax, loc=legend_loc, title="Weight range")

        self._render_figure(
            output_png=output_png,
            output_svg=output_svg,
            title=title,
            y_label=y_label,
            draw_fn=draw,
        )

    @classmethod
    def _prepare_nmr_weight_table(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and normalize solution-NMR molecular-weight records."""
        prepared = cls._prepare_typed_table(
            df=df,
            required_columns={"entry_id", "year", "molecular_weight_kda"},
            column_types={"year": int, "molecular_weight_kda": float},
            dataset_name="NMR weight CSV",
        )
        return cls._limit_year_column(prepared)

    @staticmethod
    def _period_series(table: pd.DataFrame) -> dict[str, pd.Series]:
        """Split molecular weights into the three historical periods."""
        return {
            "Before 1996": table.loc[table["year"] < 1996, "molecular_weight_kda"],
            "1996-2006": table.loc[
                (table["year"] >= 1996) & (table["year"] <= 2006),
                "molecular_weight_kda",
            ],
            "After 2006": table.loc[table["year"] > 2006, "molecular_weight_kda"],
        }

    def plot_solution_nmr_weight_stats(
        self,
        data_path: Path,
        avg_output_png: Path,
        avg_output_svg: Path,
        median_output_png: Path,
        median_output_svg: Path,
        max_output_png: Path,
        max_output_svg: Path,
    ) -> None:
        """Render annual mean, median, and maximum NMR molecular weights."""
        table = self._prepare_nmr_weight_table(self._read_csv(data_path))
        stats = (
            table.groupby("year", as_index=True)["molecular_weight_kda"]
            .agg(["mean", "median", "max"])
            .sort_index()
        )
        self._scientific_style()

        self._render_bar_series(
            output_png=avg_output_png,
            output_svg=avg_output_svg,
            title=self.config.nmr_avg_title,
            y_label=self.config.nmr_avg_y_label,
            x_values=stats.index,
            y_values=stats["mean"],
            color=self.config.avg_color,
            y_bottom=0.0,
        )
        self._render_bar_series(
            output_png=median_output_png,
            output_svg=median_output_svg,
            title=self.config.nmr_median_title,
            y_label=self.config.nmr_median_y_label,
            x_values=stats.index,
            y_values=stats["median"],
            color=self.config.median_color,
            y_bottom=0.0,
        )
        self._render_bar_series(
            output_png=max_output_png,
            output_svg=max_output_svg,
            title=self.config.nmr_max_title,
            y_label=self.config.nmr_max_y_label,
            x_values=stats.index,
            y_values=stats["max"],
            color=self.config.max_color,
            y_bottom=0.0,
        )

    def plot_solution_nmr_period_boxplot(
        self, data_path: Path, output_png: Path, output_svg: Path
    ) -> None:
        """Render molecular-weight distributions for three historical periods."""
        table = self._prepare_nmr_weight_table(self._read_csv(data_path))
        periods = self._period_series(table)
        labels = list(periods.keys())
        values = [periods[label].values for label in labels]
        self._scientific_style()

        def draw(ax: plt.Axes) -> None:
            """Draw and color the period-specific box plots."""
            bp = ax.boxplot(
                values, tick_labels=labels, patch_artist=True, showfliers=False
            )
            for patch, color in zip(
                bp["boxes"],
                [
                    self.config.before_color,
                    self.config.middle_color,
                    self.config.after_color,
                ],
            ):
                patch.set_facecolor(color)
                patch.set_alpha(0.5)

        self._render_figure(
            output_png=output_png,
            output_svg=output_svg,
            title=self.config.nmr_boxplot_title,
            y_label="Molecular weight (kDa)",
            draw_fn=draw,
            x_label="Period",
            use_year_x_ticks=False,
        )

    def plot_solution_nmr_period_area(
        self, data_path: Path, output_png: Path, output_svg: Path
    ) -> None:
        """Render cumulative NMR counts by molecular-weight category."""
        table = self._prepare_nmr_weight_table(self._read_csv(data_path))
        self._scientific_style()
        yearly = self._build_weight_category_yearly_counts(table)
        cumulative = yearly.cumsum()
        self._render_weight_category_stackplot(
            table=cumulative,
            output_png=output_png,
            output_svg=output_svg,
            title=self.config.nmr_area_title,
            y_label=self.config.nmr_area_y_label,
        )

    def plot_solution_nmr_period_area_share(
        self, data_path: Path, output_png: Path, output_svg: Path
    ) -> None:
        """Render annual shares of NMR structures by weight category."""
        table = self._prepare_nmr_weight_table(self._read_csv(data_path))
        self._scientific_style()
        yearly_counts = self._build_weight_category_yearly_counts(table)
        yearly_share = (
            yearly_counts.div(yearly_counts.sum(axis=1), axis=0).fillna(0.0) * 100.0
        )
        self._render_weight_category_stackplot(
            table=yearly_share,
            output_png=output_png,
            output_svg=output_svg,
            title=self.config.nmr_area_share_title,
            y_label=self.config.nmr_area_share_y_label,
            y_limits=(0.0, 100.0),
            x_left=float(yearly_share.index.min()),
            x_right=float(yearly_share.index.max()),
            use_step_segments=True,
            legend_loc="lower left",
        )

    def plot_solution_nmr_period_area_cumulative_share(
        self, data_path: Path, output_png: Path, output_svg: Path
    ) -> None:
        """Render cumulative shares of NMR structures by weight category."""
        table = self._prepare_nmr_weight_table(self._read_csv(data_path))
        self._scientific_style()
        yearly_counts = self._build_weight_category_yearly_counts(table)
        cumulative_counts = yearly_counts.cumsum()
        cumulative_share = (
            cumulative_counts.div(cumulative_counts.sum(axis=1), axis=0).fillna(0.0)
            * 100.0
        )
        self._render_weight_category_stackplot(
            table=cumulative_share,
            output_png=output_png,
            output_svg=output_svg,
            title=self.config.nmr_area_cumulative_share_title,
            y_label=self.config.nmr_area_cumulative_share_y_label,
            y_limits=(0.0, 100.0),
            x_left=1979,
            x_right=float(cumulative_share.index.max()),
        )
