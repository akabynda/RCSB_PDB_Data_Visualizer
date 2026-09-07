"""Render and save consistent figure variants and common series."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import PlotConfig
from .style import PlotStyleMixin
from .tables import TablePreparationMixin


class FigureRenderer(PlotStyleMixin, TablePreparationMixin):
    """Render and save consistent figure variants and common series."""

    def __init__(self, config: PlotConfig, generate_svg: bool = False) -> None:
        """Initialize the plotter with output styling and SVG preferences."""
        self.config = config
        self.generate_svg = generate_svg
        self._csv_cache: dict[Path, pd.DataFrame] = {}

    def _titleless_output_path(self, path: Path) -> Path:
        """Return the output path for a figure variant without a title."""
        return path.with_name(
            f"{path.stem}{self.config.titleless_suffix}{path.suffix}"
        )

    def _open_axes_output_path(self, path: Path) -> Path:
        """Return the output path for the open-axes figure variant."""
        return path.with_name(
            f"{path.stem}{self.config.open_axes_suffix}{path.suffix}"
        )

    def _grouped_figure_output_path(self, path: Path) -> Path:
        """Place related figure variants in a directory named for the base plot."""
        stem = path.stem
        for suffix in (
            f"{self.config.titleless_suffix}{self.config.open_axes_suffix}",
            self.config.open_axes_suffix,
            self.config.titleless_suffix,
        ):
            if stem.endswith(suffix):
                stem = stem.removesuffix(suffix)
                break
        if path.parent.name == stem:
            return path
        return path.parent / stem / path.name

    @staticmethod
    def _apply_tight_layout(
        fig: plt.Figure,
        tight_layout_rect: tuple[float, float, float, float] | None,
    ) -> None:
        """Apply tight layout with an optional normalized bounding rectangle."""
        if tight_layout_rect is None:
            fig.tight_layout()
        else:
            fig.tight_layout(rect=tight_layout_rect)

    def _save_figure_files(
        self,
        fig: plt.Figure,
        output_png: Path,
        output_svg: Path,
        *,
        savefig_bbox_inches: str | None = None,
        savefig_pad_inches: float = 0.1,
    ) -> None:
        """Save a PNG and, when enabled, an SVG using configured paths."""
        output_png = self._grouped_figure_output_path(output_png)
        output_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output_png,
            dpi=self.config.dpi,
            bbox_inches=savefig_bbox_inches,
            pad_inches=savefig_pad_inches,
        )
        if self.generate_svg:
            output_svg = self._grouped_figure_output_path(output_svg)
            output_svg.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                output_svg,
                bbox_inches=savefig_bbox_inches,
                pad_inches=savefig_pad_inches,
            )

    def _render_figure(
        self,
        output_png: Path,
        output_svg: Path,
        title: str,
        y_label: str,
        draw_fn: Callable[[plt.Axes], None],
        x_label: str | None = None,
        use_year_x_ticks: bool = True,
        tight_layout_rect: tuple[float, float, float, float] | None = None,
        savefig_bbox_inches: str | None = None,
        savefig_pad_inches: float = 0.1,
        height_scale: float = 1.0,
    ) -> None:
        """Draw and save titled, titleless, boxed, and open-axes variants."""

        def render_variant(
            variant_output_png: Path,
            variant_output_svg: Path,
            *,
            with_title: bool,
            open_axes: bool = False,
        ) -> None:
            """Render one requested title and axes-style combination."""
            fig, ax = plt.subplots(figsize=self.config.figure_size(height_scale))
            draw_fn(ax)
            if use_year_x_ticks:
                self._configure_year_axis_ticks(ax)
            y_bottom, _ = ax.get_ylim()
            if y_bottom < 0.0:
                ax.set_ylim(bottom=0.0)
            self._remove_zero_y_tick(ax)
            self._configure_minor_ticks(ax=ax, use_year_x_ticks=use_year_x_ticks)
            if with_title:
                self._set_title(ax=ax, title=title)
            ax.set_xlabel(x_label if x_label is not None else self.config.x_label)
            ax.set_ylabel(y_label)
            self._configure_boxed_axes(ax)
            if open_axes:
                self._configure_open_axes(ax)
            ax.margins(x=0)
            self._apply_tight_layout(fig=fig, tight_layout_rect=tight_layout_rect)
            self._save_figure_files(
                fig=fig,
                output_png=variant_output_png,
                output_svg=variant_output_svg,
                savefig_bbox_inches=savefig_bbox_inches,
                savefig_pad_inches=savefig_pad_inches,
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

    @staticmethod
    def _step_edges(
        x_values: pd.Index[Any] | pd.Series[Any] | np.ndarray,
    ) -> np.ndarray:
        """Convert step centers into bin-edge coordinates."""
        x_array = np.asarray(x_values, dtype=float)
        if x_array.size == 0:
            return np.array([], dtype=float)
        if x_array.size == 1:
            center = x_array[0]
            return np.array([center - 0.5, center + 0.5], dtype=float)
        midpoints = (x_array[:-1] + x_array[1:]) / 2.0
        left_edge = x_array[0] - (midpoints[0] - x_array[0])
        right_edge = x_array[-1] + (x_array[-1] - midpoints[-1])
        return np.concatenate(([left_edge], midpoints, [right_edge]))

    @staticmethod
    def _step_values(y_values: pd.Series[Any] | np.ndarray) -> np.ndarray:
        """Extend y values by one point for explicit post-step drawing."""
        y_array = np.asarray(y_values, dtype=float)
        if y_array.size == 0:
            return np.array([], dtype=float)
        return np.concatenate((y_array, [y_array[-1]]))

    def _plot_step_series(
        self,
        ax: plt.Axes,
        x_values: pd.Index[Any] | pd.Series[Any],
        y_values: pd.Series[Any] | np.ndarray,
        color: str,
        linewidth: float,
        label: str | None = None,
        zorder: int | None = None,
    ) -> None:
        """Draw one unfilled stair-step series on an axes object."""
        ax.stairs(
            values=np.asarray(y_values, dtype=float),
            edges=self._step_edges(x_values),
            baseline=None,
            color=color,
            linewidth=linewidth,
            label=label,
            fill=False,
            zorder=zorder,
        )

    def _render_line_series(
        self,
        output_png: Path,
        output_svg: Path,
        title: str,
        y_label: str,
        x_values: pd.Index[Any] | pd.Series[Any],
        y_values: pd.Series[Any],
        color: str,
        linewidth: float = 2.2,
        label: str | None = None,
        y_limits: tuple[float, float] | None = None,
        x_left: float | None = None,
    ) -> None:
        """Render one line series to the requested PNG and SVG outputs."""

        def draw(ax: plt.Axes) -> None:
            """Draw the configured line and optional limits and legend."""
            ax.plot(
                x_values,
                y_values,
                linewidth=linewidth,
                color=color,
                label=label,
            )
            if y_limits is not None:
                ax.set_ylim(*y_limits)
            if x_left is not None:
                ax.set_xlim(left=x_left)
            if label:
                self._add_legend(ax, loc="upper left")

        self._render_figure(
            output_png=output_png,
            output_svg=output_svg,
            title=title,
            y_label=y_label,
            draw_fn=draw,
        )

    def _render_multi_line_series(
        self,
        output_png: Path,
        output_svg: Path,
        title: str,
        y_label: str,
        table: pd.DataFrame,
        colors: dict[str, str],
        labels: dict[str, str],
        linewidth: float = 2.2,
        y_bottom: float | None = None,
        use_step: bool = False,
        draw_order: Sequence[str] | None = None,
    ) -> None:
        """Render multiple aligned line or step series from ``table``."""

        def draw(ax: plt.Axes) -> None:
            """Draw each selected table column with consistent ordering."""
            columns = [
                column
                for column in (draw_order or list(table.columns))
                if column in table.columns
            ]
            for zorder, column in enumerate(columns, start=2):
                if use_step:
                    self._plot_step_series(
                        ax=ax,
                        x_values=table.index,
                        y_values=table[column],
                        color=colors.get(column),
                        linewidth=linewidth,
                        label=labels.get(column, column),
                        zorder=zorder,
                    )
                else:
                    ax.plot(
                        table.index,
                        table[column],
                        linewidth=linewidth,
                        color=colors.get(column),
                        label=labels.get(column, column),
                        zorder=zorder,
                    )
            if y_bottom is not None:
                ax.set_ylim(bottom=y_bottom)
            self._add_legend(ax, loc="upper left")

        self._render_figure(
            output_png=output_png,
            output_svg=output_svg,
            title=title,
            y_label=y_label,
            draw_fn=draw,
        )

    def _render_bar_series(
        self,
        output_png: Path,
        output_svg: Path,
        title: str,
        y_label: str,
        x_values: pd.Index[Any] | pd.Series[Any],
        y_values: pd.Series[Any],
        color: str,
        width: float = 0.8,
        y_limits: tuple[float, float] | None = None,
        y_bottom: float | None = None,
        x_left: float | None = None,
    ) -> None:
        """Render a compatibility bar-series request as a step series."""

        def draw(ax: plt.Axes) -> None:
            """Draw the step representation and apply requested limits."""
            # Keep `width` in the signature for compatibility with existing calls.
            _ = width
            self._plot_step_series(
                ax=ax,
                x_values=x_values,
                y_values=y_values,
                color=color,
                linewidth=2.2,
            )
            if y_limits is not None:
                ax.set_ylim(*y_limits)
            if y_bottom is not None:
                ax.set_ylim(bottom=y_bottom)
            if x_left is not None:
                ax.set_xlim(left=x_left)

        self._render_figure(
            output_png=output_png,
            output_svg=output_svg,
            title=title,
            y_label=y_label,
            draw_fn=draw,
        )
