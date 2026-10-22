"""Plot the solution-NMR membrane cohort and availability of PDBTM segments."""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from .constants import MAX_PLOT_YEAR, YEAR_MAJOR_TICK_STEP


MODEL_GROUPS = ("One model", "More than one model", "Model count unknown")
PDBTM_COVERAGE_GROUPS = (
    "No PDBTM\nannotation",
    "Label only:\nno valid segments",
    "Partial\nsegment coverage",
    "Complete\nsegment coverage",
)


class MembranePlotsMixin:
    """Render entry counts without treating missing annotations as zero length."""

    @classmethod
    def _prepare_solution_nmr_membrane_table(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Validate one row per entry and restrict deposition years to the plot cutoff."""
        name = "SOLUTION NMR membrane protein CSV"
        cls._validate_required_columns(
            df,
            {"entry_id", "year", "deposited_model_count", "has_pdbtm_annotation",
             "pdbtm_segment_status"},
            name,
        )
        table = df.copy()
        entry_ids = table["entry_id"].astype("string").str.strip().str.upper()
        if entry_ids.isna().any() or (entry_ids == "").any():
            raise ValueError(f"{name} contains a missing entry ID")
        if entry_ids.duplicated().any():
            raise ValueError(f"{name} must contain exactly one row per entry ID")
        table["entry_id"] = entry_ids

        raw_years = table["year"].astype("string").str.strip()
        years = pd.to_numeric(raw_years.mask(raw_years == ""), errors="raise")
        known_years = years.dropna()
        if (not np.isfinite(known_years).all() or (known_years <= 0).any()
                or (known_years % 1 != 0).any()):
            raise ValueError(f"{name} contains an invalid deposition year")
        table["year"] = years.astype("Int64")

        raw_models = table["deposited_model_count"].astype("string").str.strip()
        raw_models = raw_models.mask(raw_models == "")
        models = pd.to_numeric(raw_models, errors="raise")
        known = models.dropna()
        if not np.isfinite(known).all() or (known < 1).any() or (known % 1 != 0).any():
            raise ValueError(f"{name} contains an invalid deposited model count")
        table["deposited_model_count"] = models
        table["model_group"] = MODEL_GROUPS[2]
        table.loc[models == 1, "model_group"] = MODEL_GROUPS[0]
        table.loc[models > 1, "model_group"] = MODEL_GROUPS[1]

        annotations = table["has_pdbtm_annotation"].astype("string").str.lower().str.strip()
        if not annotations.isin(("true", "false")).all():
            raise ValueError(f"{name} requires true/false PDBTM annotation values")
        statuses = table["pdbtm_segment_status"].astype("string").str.lower().str.strip()
        if not statuses.isin(("complete", "partial", "missing")).all():
            raise ValueError(f"{name} contains an invalid PDBTM segment status")
        table["has_pdbtm_annotation"] = annotations == "true"
        table["pdbtm_segment_status"] = statuses
        table["pdbtm_coverage_group"] = statuses.map({
            "missing": PDBTM_COVERAGE_GROUPS[1],
            "partial": PDBTM_COVERAGE_GROUPS[2],
            "complete": PDBTM_COVERAGE_GROUPS[3],
        })
        table.loc[~table["has_pdbtm_annotation"], "pdbtm_coverage_group"] = (
            PDBTM_COVERAGE_GROUPS[0]
        )
        unknown_year_ids = table.loc[table["year"].isna(), "entry_id"].tolist()
        if unknown_year_ids:
            warnings.warn(
                f"{name}: omitting {len(unknown_year_ids)} entries with unknown "
                f"deposition year from plots through {MAX_PLOT_YEAR}: "
                + ", ".join(unknown_year_ids),
                RuntimeWarning,
                stacklevel=2,
            )
            table = table.loc[table["year"].notna()].copy()
        table["year"] = table["year"].astype(int)
        return cls._limit_year_column(table).sort_values(["year", "entry_id"])

    @staticmethod
    def _summarize_solution_nmr_membrane_table(
        table: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Count PDB entries by year/model group and by segment-coverage category."""
        annual = pd.crosstab(table["year"], table["model_group"]).reindex(
            columns=MODEL_GROUPS, fill_value=0
        )
        if not annual.empty:
            annual = annual.reindex(range(int(annual.index.min()), int(annual.index.max()) + 1),
                                    fill_value=0)
        coverage = table["pdbtm_coverage_group"].value_counts().reindex(
            PDBTM_COVERAGE_GROUPS, fill_value=0
        )
        return annual.astype(int), coverage.astype(int)

    def plot_solution_nmr_membrane_proteins(
        self,
        data_path: Path,
        annual_output_png: Path,
        annual_output_svg: Path,
        coverage_output_png: Path,
        coverage_output_svg: Path,
    ) -> None:
        """Plot entry-level model counts and PDBTM coverage from the same cohort CSV."""
        table = self._prepare_solution_nmr_membrane_table(self._read_csv(data_path))
        annual, coverage = self._summarize_solution_nmr_membrane_table(table)
        colors = ("#56B4E9", "#E69F00", "#999999", "#009E73")
        hatches = ("//", "\\\\", "..", "xx")
        use_year_ticks = (not annual.empty and
                          annual.index.max() - annual.index.min() >= YEAR_MAJOR_TICK_STEP)

        def draw_annual(ax: plt.Axes) -> None:
            bottom = np.zeros(len(annual), dtype=int)
            for index, label in enumerate(MODEL_GROUPS):
                values = annual[label].to_numpy()
                ax.bar(annual.index, values, bottom=bottom, width=0.8,
                       label=label, color=colors[index], hatch=hatches[index],
                       edgecolor="black", linewidth=0.6)
                bottom += values
            maximum = int(bottom.max()) if len(bottom) else 0
            ax.set_ylim(0, maximum + max(1, maximum * 0.25))
            if annual.empty:
                ax.text(0.5, 0.5, f"No entries through {MAX_PLOT_YEAR}", transform=ax.transAxes,
                        ha="center", va="center")
            else:
                ax.set_xlim(annual.index.min() - 0.6, annual.index.max() + 0.6)
            if not use_year_ticks:
                ax.set_xticks(annual.index)
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            self._add_legend(ax, loc="upper center", ncol=3, fontsize=10)

        def draw_coverage(ax: plt.Axes) -> None:
            bars = ax.bar(range(len(coverage)), coverage.to_numpy(), width=0.65,
                          color=colors, edgecolor="black", linewidth=0.8)
            for bar, hatch in zip(bars, hatches):
                bar.set_hatch(hatch)
            ax.set_xticks(range(len(coverage)), PDBTM_COVERAGE_GROUPS, fontsize=10)
            ax.set_xlim(-0.6, len(coverage) - 0.4)
            maximum = int(coverage.max())
            ax.set_ylim(0, max(1, maximum * 1.15))
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.bar_label(bars, padding=4, fontsize=11)

        with plt.rc_context():
            self._scientific_style()
            self._render_figure(
                annual_output_png, annual_output_svg,
                self.config.nmr_membrane_annual_title, "Number of PDB entries", draw_annual,
                use_year_x_ticks=use_year_ticks,
            )
            self._render_figure(
                coverage_output_png, coverage_output_svg,
                self.config.nmr_membrane_coverage_title, "Number of PDB entries", draw_coverage,
                x_label="", use_year_x_ticks=False, auto_minor_ticks=False,
            )
