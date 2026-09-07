"""Plot refinement software trends and software-cluster shares."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .constants import (
    NMR_MONOMER_PROGRAM_CLUSTER_COLORS,
    NMR_MONOMER_PROGRAM_CLUSTER_LABELS,
    NMR_MONOMER_PROGRAM_CLUSTER_ORDER,
    NMR_PROGRAM_TOP_N,
)


class ProgramPlotsMixin:
    """Plot refinement software trends and software-cluster shares."""

    @classmethod
    def _prepare_nmr_program_count_table(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and pivot solution-NMR refinement-program counts by year."""
        prepared = cls._prepare_typed_table(
            df=df,
            required_columns={"year", "program", "count"},
            column_types={"year": int, "program": str, "count": int},
            dataset_name="Solution NMR program count CSV",
        )
        limited = cls._limit_year_column(prepared)
        return (
            limited.pivot(index="year", columns="program", values="count")
            .fillna(0)
            .sort_index()
            .astype(int)
        )

    @classmethod
    def _prepare_nmr_monomer_program_cluster_table(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and normalize the NMR software-cluster summary table."""
        cls._validate_required_columns(
            df=df,
            required_columns={
                "year",
                "cluster_id",
                "cluster_name",
                "structure_count",
            },
            dataset_name="Solution NMR monomer program cluster summary CSV",
        )
        prepared = df.copy()
        prepared["year"] = prepared["year"].astype(int)
        prepared["cluster_id"] = prepared["cluster_id"].astype(str)
        prepared["cluster_name"] = prepared["cluster_name"].astype(str)
        prepared["structure_count"] = prepared["structure_count"].astype(float)
        limited = cls._limit_year_column(prepared)
        return limited.sort_values(["cluster_id", "year"]).reset_index(drop=True)

    @staticmethod
    def _display_cluster_label(cluster_id: str, cluster_name: str) -> str:
        """Return the preferred display label for a software cluster."""
        display = NMR_MONOMER_PROGRAM_CLUSTER_LABELS.get(cluster_id)
        if display is not None:
            return display
        return cluster_name.replace("_", " ")

    @classmethod
    def _cluster_column_labels(cls) -> list[str]:
        """Return display labels in the canonical software-cluster order."""
        return [
            cls._display_cluster_label(cluster_id=cluster_id, cluster_name=cluster_id)
            for cluster_id in NMR_MONOMER_PROGRAM_CLUSTER_ORDER
        ]

    @classmethod
    def _build_cluster_yearly_table(
        cls,
        table: pd.DataFrame,
        value_column: str,
        fill_value: float | None,
    ) -> pd.DataFrame:
        """Pivot one cluster metric into a year-by-cluster table."""
        pivoted = (
            table.pivot(index="year", columns="cluster_id", values=value_column)
            .reindex(columns=list(NMR_MONOMER_PROGRAM_CLUSTER_ORDER))
            .sort_index()
        )
        if fill_value is not None:
            pivoted = pivoted.fillna(fill_value)
        pivoted.columns = cls._cluster_column_labels()
        return pivoted

    def _render_cluster_stackplot(
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
        legend_outside: bool = False,
        height_scale: float = 1.0,
    ) -> None:
        """Render yearly software-cluster values as a stacked area plot."""
        cluster_labels = list(table.columns)

        def draw(ax: plt.Axes) -> None:
            """Draw continuous or step-aligned software-cluster layers."""
            x_step_edges: np.ndarray | None = None
            if use_step_segments:
                base = pd.Series(0.0, index=table.index, dtype=float)
                x_step_edges = self._step_edges(table.index)
                for idx, label in enumerate(cluster_labels):
                    values = table[label].astype(float)
                    top = base + values
                    ax.fill_between(
                        x_step_edges,
                        self._step_values(base),
                        self._step_values(top),
                        step="post",
                        color=NMR_MONOMER_PROGRAM_CLUSTER_COLORS[
                            idx % len(NMR_MONOMER_PROGRAM_CLUSTER_COLORS)
                        ],
                        alpha=0.85,
                        label=label,
                    )
                    base = top
            else:
                ax.stackplot(
                    table.index,
                    *(table[label] for label in cluster_labels),
                    labels=cluster_labels,
                    colors=NMR_MONOMER_PROGRAM_CLUSTER_COLORS[: len(cluster_labels)],
                    alpha=0.85,
                )
            if y_limits is not None:
                ax.set_ylim(*y_limits)
            if x_left is not None or x_right is not None:
                current_left, current_right = ax.get_xlim()
                effective_left = x_left if x_left is not None else current_left
                effective_right = x_right if x_right is not None else current_right
                if use_step_segments and x_step_edges is not None:
                    effective_left = min(effective_left, float(x_step_edges[0]))
                    effective_right = max(effective_right, float(x_step_edges[-1]))
                ax.set_xlim(
                    left=effective_left,
                    right=effective_right,
                )
            if legend_outside:
                self._add_legend(
                    ax,
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.13),
                    borderaxespad=0.0,
                    title="Program cluster",
                    ncol=5,
                )
            else:
                self._add_legend(ax, loc="upper left", title="Program cluster", ncol=1)

        if legend_outside:
            self._render_cluster_stackplot_with_bottom_legend(
                output_png=output_png,
                output_svg=output_svg,
                title=title,
                y_label=y_label,
                draw_fn=draw,
                height_scale=height_scale,
            )
            return

        self._render_figure(
            output_png=output_png,
            output_svg=output_svg,
            title=title,
            y_label=y_label,
            draw_fn=draw,
            savefig_pad_inches=0.1,
            height_scale=height_scale,
        )

    def _render_cluster_stackplot_with_bottom_legend(
        self,
        output_png: Path,
        output_svg: Path,
        title: str,
        y_label: str,
        draw_fn: Callable[[plt.Axes], None],
        height_scale: float = 1.0,
    ) -> None:
        """Render stackplot variants with a shared legend below the axes."""

        def render_variant(
            variant_output_png: Path,
            variant_output_svg: Path,
            *,
            with_title: bool,
            open_axes: bool = False,
        ) -> None:
            """Render one title and axes-style variant with a bottom legend."""
            fig = plt.figure(figsize=self.config.figure_size(height_scale))
            ax = fig.add_axes((0.11, 0.285, 0.84, 0.585 if with_title else 0.665))
            draw_fn(ax)
            self._configure_year_axis_ticks(ax)
            y_bottom, _ = ax.get_ylim()
            if y_bottom < 0.0:
                ax.set_ylim(bottom=0.0)
            self._remove_zero_y_tick(ax)
            self._configure_minor_ticks(ax=ax, use_year_x_ticks=True)
            if with_title:
                self._set_title(ax=ax, title=title)
            ax.set_xlabel(self.config.x_label)
            ax.set_ylabel(y_label)
            self._configure_boxed_axes(ax)
            if open_axes:
                self._configure_open_axes(ax)
            ax.margins(x=0)

            handles, labels = ax.get_legend_handles_labels()
            legend = fig.legend(
                handles,
                labels,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.025),
                ncol=5,
                frameon=True,
                fancybox=False,
                framealpha=1.0,
                facecolor="white",
                edgecolor="black",
            )
            legend.get_frame().set_linewidth(0.8)
            if ax.legend_ is not None:
                ax.legend_.remove()

            self._save_figure_files(
                fig=fig,
                output_png=variant_output_png,
                output_svg=variant_output_svg,
            )
            plt.close(fig)

        render_variant(output_png, output_svg, with_title=True)
        render_variant(
            self._titleless_output_path(output_png),
            self._titleless_output_path(output_svg),
            with_title=False,
        )
        render_variant(
            self._open_axes_output_path(output_png),
            self._open_axes_output_path(output_svg),
            with_title=True,
            open_axes=True,
        )
        render_variant(
            self._open_axes_output_path(self._titleless_output_path(output_png)),
            self._open_axes_output_path(self._titleless_output_path(output_svg)),
            with_title=False,
            open_axes=True,
        )

    def plot_solution_nmr_program_counts(
        self,
        data_path: Path,
        annual_output_png: Path,
        annual_output_svg: Path,
        top_n: int = NMR_PROGRAM_TOP_N,
    ) -> None:
        """Render annual counts for the most-used NMR refinement programs."""
        table = self._prepare_nmr_program_count_table(self._read_csv(data_path))
        if table.empty:
            raise ValueError("Solution NMR program count CSV is empty.")
        top_programs = (
            table.sum(axis=0)
            .sort_values(ascending=False)
            .head(max(1, top_n))
            .index.tolist()
        )
        filtered_table = table[top_programs]
        self._scientific_style()

        def draw(ax: plt.Axes) -> None:
            """Draw one step series for each selected refinement program."""
            colors = plt.cm.tab10.colors
            for idx, program in enumerate(filtered_table.columns):
                self._plot_step_series(
                    ax=ax,
                    x_values=filtered_table.index,
                    y_values=filtered_table[program],
                    color=colors[idx % len(colors)],
                    linewidth=2.0,
                    label=program,
                )
            self._add_legend(
                ax,
                loc="upper left",
                ncol=2 if len(filtered_table.columns) > 4 else 1,
                title="Program",
            )

        self._render_figure(
            output_png=annual_output_png,
            output_svg=annual_output_svg,
            title=self.config.nmr_program_annual_title,
            y_label=self.config.nmr_program_annual_y_label,
            draw_fn=draw,
        )

    def plot_solution_nmr_monomer_program_clusters(
        self,
        data_path: Path,
        share_output_png: Path,
        share_output_svg: Path,
        share_without_other_output_png: Path,
        share_without_other_output_svg: Path,
    ) -> None:
        """Render software-cluster share plots."""
        table = self._prepare_nmr_monomer_program_cluster_table(
            self._read_csv(data_path)
        )
        if table.empty:
            raise ValueError(
                "Solution NMR monomer program cluster summary CSV is empty."
            )
        count_table = self._build_cluster_yearly_table(
            table=table,
            value_column="structure_count",
            fill_value=0.0,
        )
        count_share_table = (
            count_table.div(count_table.sum(axis=1), axis=0).fillna(0.0) * 100.0
        )
        count_without_other_table = count_table.drop(
            columns=[NMR_MONOMER_PROGRAM_CLUSTER_LABELS["CLUSTER9"]],
            errors="ignore",
        )
        count_share_without_other_table = (
            count_without_other_table.div(
                count_without_other_table.sum(axis=1),
                axis=0,
            ).fillna(0.0)
            * 100.0
        )

        self._scientific_style()
        self._render_cluster_stackplot(
            table=count_share_table,
            output_png=share_output_png,
            output_svg=share_output_svg,
            title=self.config.nmr_monomer_program_cluster_share_title,
            y_label=self.config.nmr_monomer_program_cluster_share_y_label,
            y_limits=(0.0, 100.0),
            x_left=float(count_share_table.index.min()),
            x_right=float(count_share_table.index.max()),
            use_step_segments=True,
            legend_outside=True,
        )
        self._render_cluster_stackplot(
            table=count_share_without_other_table,
            output_png=share_without_other_output_png,
            output_svg=share_without_other_output_svg,
            title=(
                self.config.nmr_monomer_program_cluster_share_title
                + " (excluding OTHER)"
            ),
            y_label=self.config.nmr_monomer_program_cluster_share_y_label,
            y_limits=(0.0, 100.0),
            x_left=float(count_share_without_other_table.index.min()),
            x_right=float(count_share_without_other_table.index.max()),
            use_step_segments=True,
            legend_outside=True,
        )
