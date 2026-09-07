"""Apply shared Matplotlib styling, axis ticks, titles, and legends."""

from __future__ import annotations

import warnings
from functools import lru_cache
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import AutoMinorLocator, FuncFormatter, MultipleLocator

from .constants import (
    AXIS_MAJOR_TICK_LENGTH,
    AXIS_MINOR_TICK_LENGTH,
    AXIS_MINOR_TICK_SUBDIVISIONS,
    MAX_PLOT_YEAR,
    TITLE_FONTSIZE,
    YEAR_MAJOR_TICK_STEP,
    YEAR_MINOR_TICK_STEP,
)


@lru_cache(maxsize=1)
def _has_arial_font() -> bool:
    """Return whether Matplotlib can resolve the preferred Arial font."""
    target = "arial"
    for font in font_manager.fontManager.ttflist:
        if font.name.strip().casefold() == target:
            return True
    return False


class PlotStyleMixin:
    """Apply shared Matplotlib styling, axis ticks, titles, and legends."""

    @staticmethod
    def _scientific_style() -> None:
        """Apply the shared publication-oriented Matplotlib style."""
        if not _has_arial_font():
            warnings.warn(
                "Arial is not available in matplotlib font registry. "
                "Matplotlib will fallback to another sans-serif font.",
                RuntimeWarning,
                stacklevel=2,
            )
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams.update(
            {
                "font.family": ["Arial"],
                "font.sans-serif": ["Arial"],
                "axes.titlesize": TITLE_FONTSIZE,
                "axes.titleweight": "bold",
                "axes.labelsize": 12,
                "xtick.labelsize": 11,
                "ytick.labelsize": 11,
                "legend.fontsize": 11,
                "axes.linewidth": 0.9,
                "grid.alpha": 0.25,
            }
        )

    @staticmethod
    def _configure_year_axis_ticks(ax: plt.Axes) -> None:
        """Configure major and minor year ticks on ``ax``."""
        ax.xaxis.set_major_locator(MultipleLocator(YEAR_MAJOR_TICK_STEP))
        ax.xaxis.set_major_formatter(
            FuncFormatter(
                lambda value, _pos: (
                    f"{int(value)}"
                    if abs(value - round(value)) < 1e-6
                    and int(round(value)) <= MAX_PLOT_YEAR
                    else ""
                )
            )
        )
        ax.xaxis.set_minor_locator(MultipleLocator(YEAR_MINOR_TICK_STEP))
        ax.tick_params(axis="x", which="major", length=AXIS_MAJOR_TICK_LENGTH)
        ax.tick_params(axis="x", which="minor", length=AXIS_MINOR_TICK_LENGTH)

    @staticmethod
    def _visible_major_step(axis: plt.Axis) -> float | None:
        """Return the uniform spacing between visible major ticks, if any."""
        tick_locs = axis.get_majorticklocs()
        if len(tick_locs) < 2:
            return None
        view_min, view_max = sorted(axis.get_view_interval())
        visible = [
            float(tick)
            for tick in tick_locs
            if view_min - 1e-9 <= float(tick) <= view_max + 1e-9
        ]
        if len(visible) < 2:
            return None
        diffs = [
            right - left
            for left, right in zip(visible, visible[1:])
            if (right - left) > 1e-9
        ]
        if not diffs:
            return None
        step = min(diffs)
        if step <= 0.0:
            return None
        if max(diffs) - min(diffs) > step * 1e-6:
            return None
        return step

    @staticmethod
    def _has_categorical_x_ticks(ax: plt.Axes) -> bool:
        """Return whether visible x tick labels represent categories."""
        labels = [label.get_text().strip() for label in ax.get_xticklabels()]
        nonempty = [label for label in labels if label]
        if not nonempty:
            return False
        try:
            for label in nonempty:
                float(label.replace("\N{MINUS SIGN}", "-"))
        except ValueError:
            return True
        return False

    @classmethod
    def _configure_minor_ticks(cls, ax: plt.Axes, use_year_x_ticks: bool) -> None:
        """Give numeric x and y axes matching major and minor tick marks."""
        ax.yaxis.set_minor_locator(AutoMinorLocator(AXIS_MINOR_TICK_SUBDIVISIONS))
        ax.tick_params(axis="y", which="major", length=AXIS_MAJOR_TICK_LENGTH)
        ax.tick_params(
            axis="y",
            which="minor",
            length=AXIS_MINOR_TICK_LENGTH,
            labelleft=False,
        )

        if use_year_x_ticks or cls._has_categorical_x_ticks(ax):
            return
        x_step = cls._visible_major_step(ax.xaxis)
        if x_step is None:
            return
        ax.xaxis.set_minor_locator(AutoMinorLocator(AXIS_MINOR_TICK_SUBDIVISIONS))
        ax.tick_params(axis="x", which="major", length=AXIS_MAJOR_TICK_LENGTH)
        ax.tick_params(axis="x", which="minor", length=AXIS_MINOR_TICK_LENGTH)

    @staticmethod
    def _remove_zero_y_tick(ax: plt.Axes) -> None:
        """Remove the zero label from the y axis while retaining other ticks."""
        y_bottom, y_top = ax.get_ylim()
        if y_bottom >= 0.0:
            filtered_ticks = [
                tick
                for tick in ax.get_yticks()
                if 0.0 < float(tick) <= y_top + 1e-9
            ]
        else:
            filtered_ticks = [
                tick
                for tick in ax.get_yticks()
                if y_bottom - 1e-9 <= float(tick) <= y_top + 1e-9
                and abs(float(tick)) > 1e-9
            ]
        if filtered_ticks:
            ax.set_yticks(filtered_ticks)

    @staticmethod
    def _configure_boxed_axes(
        ax: plt.Axes,
        *,
        y_tick_labels_on_both_sides: bool = False,
    ) -> None:
        """Show a full axes box and optionally label both y-axis sides."""
        for spine in ax.spines.values():
            spine.set_visible(True)

        ax.tick_params(
            axis="y",
            which="major",
            left=True,
            right=True,
            labelleft=True,
            labelright=y_tick_labels_on_both_sides,
        )
        ax.tick_params(
            axis="y",
            which="minor",
            left=True,
            right=True,
            labelleft=False,
            labelright=False,
        )
        ax.tick_params(
            axis="x",
            which="both",
            bottom=True,
            top=True,
            labelbottom=True,
            labeltop=False,
        )

    @staticmethod
    def _configure_open_axes(ax: plt.Axes) -> None:
        """Hide the top and right spines for the open-axes output variant."""
        ax.spines["left"].set_visible(True)
        ax.spines["bottom"].set_visible(True)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.tick_params(axis="y", which="both", left=True, right=False)
        ax.tick_params(axis="x", which="both", bottom=True, top=False)

    @staticmethod
    def _set_title(
        ax: plt.Axes,
        title: str,
    ) -> None:
        """Set the configured scientific figure title on ``ax``."""
        ax.set_title(
            title,
            pad=10,
            fontsize=TITLE_FONTSIZE,
            fontweight="bold",
        )

    @staticmethod
    def _add_legend(ax: plt.Axes, **kwargs: Any) -> None:
        """Add a consistently framed legend using supplied options."""
        legend = ax.legend(
            frameon=True,
            fancybox=False,
            framealpha=1.0,
            facecolor="white",
            edgecolor="black",
            **kwargs,
        )
        if legend:
            legend.get_frame().set_linewidth(0.8)
