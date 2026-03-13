from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import pandas as pd


class PlotKind(str, Enum):
    METHOD_COUNTS = "method_counts"
    SOLUTION_NMR_WEIGHT_STATS = "solution_nmr_weight_stats"
    SOLUTION_NMR_PERIOD_BOXPLOT = "solution_nmr_period_boxplot"
    SOLUTION_NMR_PERIOD_AREA = "solution_nmr_period_area"
    SOLUTION_NMR_PERIOD_AREA_SHARE = "solution_nmr_period_area_share"
    SOLUTION_NMR_PERIOD_AREA_CUMULATIVE_SHARE = (
        "solution_nmr_period_area_cumulative_share"
    )
    SOLUTION_NMR_MONOMER_SECONDARY = "solution_nmr_monomer_secondary"
    SOLUTION_NMR_MONOMER_PRECISION = "solution_nmr_monomer_precision"
    SOLUTION_NMR_MONOMER_QUALITY = "solution_nmr_monomer_quality"


@dataclass(frozen=True)
class PlotConfig:
    width_inches: float = 8.6
    height_inches: float = 5.0
    dpi: int = 600
    x_label: str = "Deposition year"
    annual_title: str = "Annual Number of PDB Structures by Experimental Method"
    annual_y_label: str = "Number of deposited structures"
    cumulative_title: str = "Cumulative Number of PDB Structures by Experimental Method"
    cumulative_y_label: str = "Cumulative number of deposited structures"
    nmr_avg_title: str = "SOLUTION NMR: Mean Structure Molecular Weight by Year"
    nmr_avg_y_label: str = "Mean molecular weight (kDa)"
    nmr_median_title: str = "SOLUTION NMR: Median Structure Molecular Weight by Year"
    nmr_median_y_label: str = "Median molecular weight (kDa)"
    nmr_max_title: str = "SOLUTION NMR: Maximum Structure Molecular Weight by Year"
    nmr_max_y_label: str = "Maximum molecular weight (kDa)"
    nmr_boxplot_title: str = "SOLUTION NMR: Molecular Weight by Period"
    nmr_area_title: str = "SOLUTION NMR: Cumulative Structures by Weight Category"
    nmr_area_y_label: str = "Cumulative number of structures"
    nmr_area_share_title: str = "SOLUTION NMR: Category Share by Year"
    nmr_area_share_y_label: str = "Share of structures (%)"
    nmr_area_cumulative_share_title: str = (
        "SOLUTION NMR: Category Share by Cumulative Sum"
    )
    nmr_area_cumulative_share_y_label: str = "Share of cumulative structures (%)"
    nmr_monomer_secondary_title: str = (
        "SOLUTION NMR Monomeric Proteins: Secondary Structure Content by Year"
    )
    nmr_monomer_secondary_y_label: str = "Secondary structure content (%)"
    nmr_monomer_precision_title: str = (
        "SOLUTION NMR Monomeric Proteins: Mean Ensemble RMSD by Year"
    )
    nmr_monomer_precision_y_label: str = "Mean RMSD to average structure (Å)"
    nmr_monomer_quality_clash_title: str = (
        "SOLUTION NMR Monomeric Proteins: Mean Clashscore by Year"
    )
    nmr_monomer_quality_clash_y_label: str = "Mean clashscore"
    nmr_monomer_quality_rama_title: str = (
        "SOLUTION NMR Monomeric Proteins: Mean Ramachandran Outliers by Year"
    )
    nmr_monomer_quality_rama_y_label: str = "Mean Ramachandran outliers (%)"
    nmr_monomer_quality_side_title: str = (
        "SOLUTION NMR Monomeric Proteins: Mean Sidechain Outliers by Year"
    )
    nmr_monomer_quality_side_y_label: str = "Mean sidechain outliers (%)"
    xray_color: str = "#1f77b4"
    cryoem_color: str = "#d62728"
    nmr_color: str = "#2ca02c"
    avg_color: str = "#1f77b4"
    median_color: str = "#ff7f0e"
    max_color: str = "#d62728"
    before_color: str = "#4c78a8"
    middle_color: str = "#54a24b"
    after_color: str = "#e45756"
    area_colors: tuple[str, str, str] = ("#4c78a8", "#f58518", "#54a24b")


def parse_plot_kinds(raw_value: str) -> list[PlotKind]:
    if raw_value.strip().lower() == "all":
        return [
            PlotKind.METHOD_COUNTS,
            PlotKind.SOLUTION_NMR_WEIGHT_STATS,
            PlotKind.SOLUTION_NMR_PERIOD_BOXPLOT,
            PlotKind.SOLUTION_NMR_PERIOD_AREA,
            PlotKind.SOLUTION_NMR_PERIOD_AREA_SHARE,
            PlotKind.SOLUTION_NMR_PERIOD_AREA_CUMULATIVE_SHARE,
            PlotKind.SOLUTION_NMR_MONOMER_SECONDARY,
            PlotKind.SOLUTION_NMR_MONOMER_PRECISION,
            PlotKind.SOLUTION_NMR_MONOMER_QUALITY,
        ]
    raw_items = [item.strip() for item in raw_value.split(",") if item.strip()]
    selected: list[PlotKind] = []
    for item in raw_items:
        try:
            selected.append(PlotKind(item))
        except ValueError as exc:
            valid = ", ".join(plot.value for plot in PlotKind)
            raise argparse.ArgumentTypeError(
                f"Unknown plot '{item}'. Use one of: {valid}, all."
            ) from exc
    if not selected:
        raise argparse.ArgumentTypeError("No plots selected.")
    return selected


class PDBScientificPlotter:
    def __init__(self, config: PlotConfig) -> None:
        self.config = config

    @staticmethod
    def _scientific_style() -> None:
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams.update(
            {
                "font.family": "serif",
                "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                "axes.titlesize": 12,
                "axes.labelsize": 11,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "legend.fontsize": 10,
                "axes.linewidth": 0.9,
                "grid.alpha": 0.25,
            }
        )

    def _render_figure(
        self,
        output_png: Path,
        output_svg: Path,
        title: str,
        y_label: str,
        draw_fn: Callable[[plt.Axes], None],
    ) -> None:
        fig, ax = plt.subplots(
            figsize=(self.config.width_inches, self.config.height_inches)
        )
        draw_fn(ax)
        ax.set_title(title, pad=10)
        ax.set_xlabel(self.config.x_label)
        ax.set_ylabel(y_label)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.margins(x=0.01)
        fig.tight_layout()
        output_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_png, dpi=self.config.dpi)
        fig.savefig(output_svg)
        plt.close(fig)

    @staticmethod
    def _prepare_method_count_table(df: pd.DataFrame) -> pd.DataFrame:
        required_columns = {"year", "method", "count"}
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(
                f"Method count CSV is missing required columns: {', '.join(sorted(missing))}"
            )
        return (
            df.pivot(index="year", columns="method", values="count")
            .fillna(0)
            .sort_index()
            .astype(int)
        )

    @staticmethod
    def _prepare_nmr_weight_table(df: pd.DataFrame) -> pd.DataFrame:
        required_columns = {"entry_id", "year", "molecular_weight_kda"}
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(
                f"NMR weight CSV is missing required columns: {', '.join(sorted(missing))}"
            )
        prepared = df.copy()
        prepared["year"] = prepared["year"].astype(int)
        prepared["molecular_weight_kda"] = prepared["molecular_weight_kda"].astype(
            float
        )
        return prepared

    @staticmethod
    def _period_series(table: pd.DataFrame) -> dict[str, pd.Series]:
        return {
            "Before 1996": table.loc[table["year"] < 1996, "molecular_weight_kda"],
            "1996-2006": table.loc[
                (table["year"] >= 1996) & (table["year"] <= 2006),
                "molecular_weight_kda",
            ],
            "After 2006": table.loc[table["year"] > 2006, "molecular_weight_kda"],
        }

    @staticmethod
    def _place_bottom_legend(ax: plt.Axes) -> None:
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.14),
            frameon=False,
            ncol=3,
            title="Weight category",
        )

    @staticmethod
    def _place_right_legend(ax: plt.Axes) -> None:
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            title="Weight category",
        )

    def plot_method_counts(
        self,
        data_path: Path,
        annual_output_png: Path,
        annual_output_svg: Path,
        cumulative_output_png: Path,
        cumulative_output_svg: Path,
    ) -> None:
        table = self._prepare_method_count_table(pd.read_csv(data_path))
        cumulative_table = table.cumsum()
        self._scientific_style()

        def draw(ax: plt.Axes, source: pd.DataFrame) -> None:
            for col, color in [
                ("X-ray", self.config.xray_color),
                ("cryo-EM", self.config.cryoem_color),
                ("NMR", self.config.nmr_color),
            ]:
                if col in source.columns:
                    ax.plot(
                        source.index, source[col], color=color, linewidth=2.0, label=col
                    )
            ax.legend(loc="upper left", frameon=False)

        self._render_figure(
            annual_output_png,
            annual_output_svg,
            self.config.annual_title,
            self.config.annual_y_label,
            lambda ax: draw(ax, table),
        )
        self._render_figure(
            cumulative_output_png,
            cumulative_output_svg,
            self.config.cumulative_title,
            self.config.cumulative_y_label,
            lambda ax: draw(ax, cumulative_table),
        )

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
        table = self._prepare_nmr_weight_table(pd.read_csv(data_path))
        stats = (
            table.groupby("year", as_index=True)["molecular_weight_kda"]
            .agg(["mean", "median", "max"])
            .sort_index()
        )
        self._scientific_style()

        self._render_figure(
            avg_output_png,
            avg_output_svg,
            self.config.nmr_avg_title,
            self.config.nmr_avg_y_label,
            lambda ax: ax.plot(
                stats.index, stats["mean"], color=self.config.avg_color, linewidth=2.0
            ),
        )
        self._render_figure(
            median_output_png,
            median_output_svg,
            self.config.nmr_median_title,
            self.config.nmr_median_y_label,
            lambda ax: ax.plot(
                stats.index,
                stats["median"],
                color=self.config.median_color,
                linewidth=2.0,
            ),
        )
        self._render_figure(
            max_output_png,
            max_output_svg,
            self.config.nmr_max_title,
            self.config.nmr_max_y_label,
            lambda ax: ax.plot(
                stats.index, stats["max"], color=self.config.max_color, linewidth=2.0
            ),
        )

    def plot_solution_nmr_period_boxplot(
        self, data_path: Path, output_png: Path, output_svg: Path
    ) -> None:
        table = self._prepare_nmr_weight_table(pd.read_csv(data_path))
        periods = self._period_series(table)
        labels = list(periods.keys())
        values = [periods[label].values for label in labels]
        self._scientific_style()

        fig, ax = plt.subplots(
            figsize=(self.config.width_inches, self.config.height_inches)
        )
        bp = ax.boxplot(values, tick_labels=labels, patch_artist=True, showfliers=False)
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
        ax.set_title(self.config.nmr_boxplot_title, pad=10)
        ax.set_xlabel("Period")
        ax.set_ylabel("Molecular weight (kDa)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        output_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_png, dpi=self.config.dpi)
        fig.savefig(output_svg)
        plt.close(fig)

    def plot_solution_nmr_period_area(
        self, data_path: Path, output_png: Path, output_svg: Path
    ) -> None:
        table = self._prepare_nmr_weight_table(pd.read_csv(data_path))
        self._scientific_style()

        # Three broad categories for readable long-term cumulative trends.
        bins = [0.0, 10.0, 25.0, float("inf")]
        labels = ["<10 kDa", "10-25 kDa", ">25 kDa"]
        table["weight_category"] = pd.cut(
            table["molecular_weight_kda"], bins=bins, labels=labels, right=False
        )

        yearly = (
            table.groupby(["year", "weight_category"], observed=False)
            .size()
            .unstack(fill_value=0)
            .reindex(columns=labels, fill_value=0)
            .sort_index()
        )
        cumulative = yearly.cumsum()

        fig, ax = plt.subplots(
            figsize=(self.config.width_inches, self.config.height_inches)
        )
        ax.stackplot(
            cumulative.index,
            cumulative[labels[0]],
            cumulative[labels[1]],
            cumulative[labels[2]],
            labels=labels,
            colors=self.config.area_colors,
            alpha=0.85,
        )
        ax.set_title(self.config.nmr_area_title, pad=10)
        ax.set_xlabel(self.config.x_label)
        ax.set_ylabel(self.config.nmr_area_y_label)
        ax.legend(loc="upper left", frameon=False, title="Weight category")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        output_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_png, dpi=self.config.dpi)
        fig.savefig(output_svg)
        plt.close(fig)

    def plot_solution_nmr_period_area_share(
        self, data_path: Path, output_png: Path, output_svg: Path
    ) -> None:
        table = self._prepare_nmr_weight_table(pd.read_csv(data_path))
        self._scientific_style()

        bins = [0.0, 10.0, 25.0, float("inf")]
        labels = ["<10 kDa", "10-25 kDa", ">25 kDa"]
        table["weight_category"] = pd.cut(
            table["molecular_weight_kda"], bins=bins, labels=labels, right=False
        )

        yearly_counts = (
            table.groupby(["year", "weight_category"], observed=False)
            .size()
            .unstack(fill_value=0)
            .reindex(columns=labels, fill_value=0)
            .sort_index()
        )
        yearly_share = (
            yearly_counts.div(yearly_counts.sum(axis=1), axis=0).fillna(0.0) * 100.0
        )

        fig, ax = plt.subplots(
            figsize=(self.config.width_inches, self.config.height_inches)
        )
        ax.stackplot(
            yearly_share.index,
            yearly_share[labels[0]],
            yearly_share[labels[1]],
            yearly_share[labels[2]],
            labels=labels,
            colors=self.config.area_colors,
            alpha=0.85,
        )
        ax.set_title(self.config.nmr_area_share_title, pad=10)
        ax.set_xlabel(self.config.x_label)
        ax.set_ylabel(self.config.nmr_area_share_y_label)
        ax.set_ylim(0, 100)
        ax.set_xlim(left=1978)
        ax.legend(loc="upper left", frameon=False, title="Weight category")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        output_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_png, dpi=self.config.dpi)
        fig.savefig(output_svg)
        plt.close(fig)

    def plot_solution_nmr_period_area_cumulative_share(
        self, data_path: Path, output_png: Path, output_svg: Path
    ) -> None:
        table = self._prepare_nmr_weight_table(pd.read_csv(data_path))
        self._scientific_style()

        bins = [0.0, 10.0, 25.0, float("inf")]
        labels = ["<10 kDa", "10-25 kDa", ">25 kDa"]
        table["weight_category"] = pd.cut(
            table["molecular_weight_kda"], bins=bins, labels=labels, right=False
        )

        yearly_counts = (
            table.groupby(["year", "weight_category"], observed=False)
            .size()
            .unstack(fill_value=0)
            .reindex(columns=labels, fill_value=0)
            .sort_index()
        )
        cumulative_counts = yearly_counts.cumsum()
        cumulative_share = (
            cumulative_counts.div(cumulative_counts.sum(axis=1), axis=0).fillna(0.0)
            * 100.0
        )

        fig, ax = plt.subplots(
            figsize=(self.config.width_inches, self.config.height_inches)
        )
        ax.stackplot(
            cumulative_share.index,
            cumulative_share[labels[0]],
            cumulative_share[labels[1]],
            cumulative_share[labels[2]],
            labels=labels,
            colors=self.config.area_colors,
            alpha=0.85,
        )
        ax.set_title(self.config.nmr_area_cumulative_share_title, pad=10)
        ax.set_xlabel(self.config.x_label)
        ax.set_ylabel(self.config.nmr_area_cumulative_share_y_label)
        ax.set_ylim(0, 100)
        ax.set_xlim(left=1978)
        ax.legend(loc="upper left", frameon=False, title="Weight category")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        output_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_png, dpi=self.config.dpi)
        fig.savefig(output_svg)
        plt.close(fig)

    @staticmethod
    def _prepare_monomer_secondary_table(df: pd.DataFrame) -> pd.DataFrame:
        required_columns = {"entry_id", "year", "secondary_structure_percent"}
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(
                f"Monomer secondary CSV is missing required columns: {', '.join(sorted(missing))}"
            )
        prepared = df.copy()
        prepared["year"] = prepared["year"].astype(int)
        prepared["secondary_structure_percent"] = prepared[
            "secondary_structure_percent"
        ].astype(float)
        return prepared

    def plot_solution_nmr_monomer_secondary(
        self, data_path: Path, output_png: Path, output_svg: Path
    ) -> None:
        table = self._prepare_monomer_secondary_table(pd.read_csv(data_path))
        self._scientific_style()
        yearly_mean = (
            table.groupby("year", as_index=True)["secondary_structure_percent"]
            .mean()
            .sort_index()
        )

        fig, ax = plt.subplots(
            figsize=(self.config.width_inches, self.config.height_inches)
        )
        ax.scatter(
            table["year"],
            table["secondary_structure_percent"],
            s=10,
            alpha=0.2,
            color="#7f7f7f",
            label="Individual structures",
        )
        ax.plot(
            yearly_mean.index,
            yearly_mean.values,
            linewidth=2.2,
            color=self.config.nmr_color,
            label="Yearly mean",
        )
        ax.set_title(self.config.nmr_monomer_secondary_title, pad=10)
        ax.set_xlabel(self.config.x_label)
        ax.set_ylabel(self.config.nmr_monomer_secondary_y_label)
        ax.set_ylim(0, 100)
        ax.legend(loc="upper left", frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        output_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_png, dpi=self.config.dpi)
        fig.savefig(output_svg)
        plt.close(fig)

    @staticmethod
    def _prepare_monomer_precision_table(df: pd.DataFrame) -> pd.DataFrame:
        required_columns = {"entry_id", "year", "mean_rmsd_angstrom"}
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(
                f"Monomer precision CSV is missing required columns: {', '.join(sorted(missing))}"
            )
        prepared = df.copy()
        prepared["year"] = prepared["year"].astype(int)
        prepared["mean_rmsd_angstrom"] = prepared["mean_rmsd_angstrom"].astype(float)
        return prepared

    def plot_solution_nmr_monomer_precision(
        self, data_path: Path, output_png: Path, output_svg: Path
    ) -> None:
        table = self._prepare_monomer_precision_table(pd.read_csv(data_path))
        self._scientific_style()
        yearly_mean_rmsd = (
            table.groupby("year", as_index=True)["mean_rmsd_angstrom"]
            .mean()
            .sort_index()
        )

        fig, ax = plt.subplots(
            figsize=(self.config.width_inches, self.config.height_inches)
        )
        ax.plot(
            yearly_mean_rmsd.index,
            yearly_mean_rmsd.values,
            linewidth=2.2,
            color="#8c564b",
            label="Yearly mean RMSD",
        )
        ax.set_title(self.config.nmr_monomer_precision_title, pad=10)
        ax.set_xlabel(self.config.x_label)
        ax.set_ylabel(self.config.nmr_monomer_precision_y_label)
        ax.legend(loc="upper left", frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        output_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_png, dpi=self.config.dpi)
        fig.savefig(output_svg)
        plt.close(fig)

    @staticmethod
    def _prepare_monomer_quality_table(df: pd.DataFrame) -> pd.DataFrame:
        required_columns = {
            "entry_id",
            "year",
            "clashscore",
            "ramachandran_outliers_percent",
            "sidechain_outliers_percent",
        }
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(
                f"Monomer quality CSV is missing required columns: {', '.join(sorted(missing))}"
            )
        prepared = df.copy()
        prepared["year"] = prepared["year"].astype(int)
        prepared["clashscore"] = prepared["clashscore"].astype(float)
        prepared["ramachandran_outliers_percent"] = prepared[
            "ramachandran_outliers_percent"
        ].astype(float)
        prepared["sidechain_outliers_percent"] = prepared[
            "sidechain_outliers_percent"
        ].astype(float)
        return prepared

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
        table = self._prepare_monomer_quality_table(pd.read_csv(data_path))
        self._scientific_style()
        yearly = table.groupby("year", as_index=True).mean(numeric_only=True).sort_index()

        self._render_figure(
            clash_output_png,
            clash_output_svg,
            self.config.nmr_monomer_quality_clash_title,
            self.config.nmr_monomer_quality_clash_y_label,
            lambda ax: ax.plot(
                yearly.index,
                yearly["clashscore"],
                linewidth=2.2,
                color="#8c564b",
            ),
        )
        self._render_figure(
            rama_output_png,
            rama_output_svg,
            self.config.nmr_monomer_quality_rama_title,
            self.config.nmr_monomer_quality_rama_y_label,
            lambda ax: ax.plot(
                yearly.index,
                yearly["ramachandran_outliers_percent"],
                linewidth=2.2,
                color="#1f77b4",
            ),
        )
        self._render_figure(
            side_output_png,
            side_output_svg,
            self.config.nmr_monomer_quality_side_title,
            self.config.nmr_monomer_quality_side_y_label,
            lambda ax: ax.plot(
                yearly.index,
                yearly["sidechain_outliers_percent"],
                linewidth=2.2,
                color="#ff7f0e",
            ),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot publication-ready figures from PDB CSV datasets."
    )
    parser.add_argument(
        "--plots",
        type=parse_plot_kinds,
        default=[
            PlotKind.METHOD_COUNTS,
            PlotKind.SOLUTION_NMR_WEIGHT_STATS,
            PlotKind.SOLUTION_NMR_PERIOD_BOXPLOT,
            PlotKind.SOLUTION_NMR_PERIOD_AREA,
            PlotKind.SOLUTION_NMR_PERIOD_AREA_SHARE,
            PlotKind.SOLUTION_NMR_PERIOD_AREA_CUMULATIVE_SHARE,
            PlotKind.SOLUTION_NMR_MONOMER_SECONDARY,
            PlotKind.SOLUTION_NMR_MONOMER_PRECISION,
            PlotKind.SOLUTION_NMR_MONOMER_QUALITY,
        ],
        help="Comma-separated plot kinds or 'all'. Available: method_counts, solution_nmr_weight_stats, solution_nmr_period_boxplot, solution_nmr_period_area, solution_nmr_period_area_share, solution_nmr_period_area_cumulative_share, solution_nmr_monomer_secondary, solution_nmr_monomer_precision, solution_nmr_monomer_quality (default: all).",
    )
    parser.add_argument(
        "--counts-input",
        type=Path,
        default=Path("data/pdb_method_counts_by_year.csv"),
        help="Input CSV for method_counts plot.",
    )
    parser.add_argument(
        "--nmr-weights-input",
        type=Path,
        default=Path("data/solution_nmr_structure_weights.csv"),
        help="Input CSV for SOLUTION NMR weight-based plots.",
    )
    parser.add_argument(
        "--nmr-monomer-secondary-input",
        type=Path,
        default=Path("data/solution_nmr_monomer_secondary_structure.csv"),
        help="Input CSV for SOLUTION NMR monomer secondary-structure plot.",
    )
    parser.add_argument(
        "--nmr-monomer-precision-input",
        type=Path,
        default=Path("data/solution_nmr_monomer_precision.csv"),
        help="Input CSV for SOLUTION NMR monomer precision (RMSD) plot.",
    )
    parser.add_argument(
        "--nmr-monomer-quality-input",
        type=Path,
        default=Path("data/solution_nmr_monomer_quality_metrics.csv"),
        help="Input CSV for SOLUTION NMR monomer quality-metrics plots.",
    )

    parser.add_argument(
        "--annual-output-png",
        type=Path,
        default=Path("figures/pdb_method_trends.png"),
        help="Output PNG for annual method-count figure.",
    )
    parser.add_argument(
        "--annual-output-svg",
        type=Path,
        default=Path("figures/pdb_method_trends.svg"),
        help="Output SVG for annual method-count figure.",
    )
    parser.add_argument(
        "--cumulative-output-png",
        type=Path,
        default=Path("figures/pdb_method_trends_cumulative.png"),
        help="Output PNG for cumulative method-count figure.",
    )
    parser.add_argument(
        "--cumulative-output-svg",
        type=Path,
        default=Path("figures/pdb_method_trends_cumulative.svg"),
        help="Output SVG for cumulative method-count figure.",
    )

    parser.add_argument(
        "--nmr-avg-output-png",
        type=Path,
        default=Path("figures/solution_nmr_mean_weight_by_year.png"),
        help="Output PNG for SOLUTION NMR mean molecular weight figure.",
    )
    parser.add_argument(
        "--nmr-avg-output-svg",
        type=Path,
        default=Path("figures/solution_nmr_mean_weight_by_year.svg"),
        help="Output SVG for SOLUTION NMR mean molecular weight figure.",
    )
    parser.add_argument(
        "--nmr-median-output-png",
        type=Path,
        default=Path("figures/solution_nmr_median_weight_by_year.png"),
        help="Output PNG for SOLUTION NMR median molecular weight figure.",
    )
    parser.add_argument(
        "--nmr-median-output-svg",
        type=Path,
        default=Path("figures/solution_nmr_median_weight_by_year.svg"),
        help="Output SVG for SOLUTION NMR median molecular weight figure.",
    )
    parser.add_argument(
        "--nmr-max-output-png",
        type=Path,
        default=Path("figures/solution_nmr_max_weight_by_year.png"),
        help="Output PNG for SOLUTION NMR max molecular weight figure.",
    )
    parser.add_argument(
        "--nmr-max-output-svg",
        type=Path,
        default=Path("figures/solution_nmr_max_weight_by_year.svg"),
        help="Output SVG for SOLUTION NMR max molecular weight figure.",
    )

    parser.add_argument(
        "--nmr-boxplot-output-png",
        type=Path,
        default=Path("figures/solution_nmr_weight_boxplot_by_period.png"),
        help="Output PNG for SOLUTION NMR period boxplot.",
    )
    parser.add_argument(
        "--nmr-boxplot-output-svg",
        type=Path,
        default=Path("figures/solution_nmr_weight_boxplot_by_period.svg"),
        help="Output SVG for SOLUTION NMR period boxplot.",
    )
    parser.add_argument(
        "--nmr-area-output-png",
        type=Path,
        default=Path("figures/solution_nmr_cumulative_area_by_weight_category.png"),
        help="Output PNG for SOLUTION NMR cumulative area chart.",
    )
    parser.add_argument(
        "--nmr-area-output-svg",
        type=Path,
        default=Path("figures/solution_nmr_cumulative_area_by_weight_category.svg"),
        help="Output SVG for SOLUTION NMR cumulative area chart.",
    )
    parser.add_argument(
        "--nmr-area-share-output-png",
        type=Path,
        default=Path("figures/solution_nmr_area_share_by_weight_category.png"),
        help="Output PNG for SOLUTION NMR share area chart.",
    )
    parser.add_argument(
        "--nmr-area-share-output-svg",
        type=Path,
        default=Path("figures/solution_nmr_area_share_by_weight_category.svg"),
        help="Output SVG for SOLUTION NMR share area chart.",
    )
    parser.add_argument(
        "--nmr-area-cumulative-share-output-png",
        type=Path,
        default=Path(
            "figures/solution_nmr_cumulative_share_area_by_weight_category.png"
        ),
        help="Output PNG for SOLUTION NMR cumulative-share area chart.",
    )
    parser.add_argument(
        "--nmr-area-cumulative-share-output-svg",
        type=Path,
        default=Path(
            "figures/solution_nmr_cumulative_share_area_by_weight_category.svg"
        ),
        help="Output SVG for SOLUTION NMR cumulative-share area chart.",
    )
    parser.add_argument(
        "--nmr-monomer-secondary-output-png",
        type=Path,
        default=Path("figures/solution_nmr_monomer_secondary_by_year.png"),
        help="Output PNG for SOLUTION NMR monomer secondary-structure plot.",
    )
    parser.add_argument(
        "--nmr-monomer-secondary-output-svg",
        type=Path,
        default=Path("figures/solution_nmr_monomer_secondary_by_year.svg"),
        help="Output SVG for SOLUTION NMR monomer secondary-structure plot.",
    )
    parser.add_argument(
        "--nmr-monomer-precision-output-png",
        type=Path,
        default=Path("figures/solution_nmr_monomer_precision_rmsd_by_year.png"),
        help="Output PNG for SOLUTION NMR monomer precision (RMSD) plot.",
    )
    parser.add_argument(
        "--nmr-monomer-precision-output-svg",
        type=Path,
        default=Path("figures/solution_nmr_monomer_precision_rmsd_by_year.svg"),
        help="Output SVG for SOLUTION NMR monomer precision (RMSD) plot.",
    )
    parser.add_argument(
        "--nmr-monomer-quality-clash-output-png",
        type=Path,
        default=Path("figures/solution_nmr_monomer_quality_clashscore_by_year.png"),
        help="Output PNG for monomer quality clashscore plot.",
    )
    parser.add_argument(
        "--nmr-monomer-quality-clash-output-svg",
        type=Path,
        default=Path("figures/solution_nmr_monomer_quality_clashscore_by_year.svg"),
        help="Output SVG for monomer quality clashscore plot.",
    )
    parser.add_argument(
        "--nmr-monomer-quality-rama-output-png",
        type=Path,
        default=Path(
            "figures/solution_nmr_monomer_quality_ramachandran_outliers_by_year.png"
        ),
        help="Output PNG for monomer quality Ramachandran-outliers plot.",
    )
    parser.add_argument(
        "--nmr-monomer-quality-rama-output-svg",
        type=Path,
        default=Path(
            "figures/solution_nmr_monomer_quality_ramachandran_outliers_by_year.svg"
        ),
        help="Output SVG for monomer quality Ramachandran-outliers plot.",
    )
    parser.add_argument(
        "--nmr-monomer-quality-side-output-png",
        type=Path,
        default=Path("figures/solution_nmr_monomer_quality_sidechain_outliers_by_year.png"),
        help="Output PNG for monomer quality sidechain-outliers plot.",
    )
    parser.add_argument(
        "--nmr-monomer-quality-side-output-svg",
        type=Path,
        default=Path("figures/solution_nmr_monomer_quality_sidechain_outliers_by_year.svg"),
        help="Output SVG for monomer quality sidechain-outliers plot.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plotter = PDBScientificPlotter(config=PlotConfig())

    if PlotKind.METHOD_COUNTS in args.plots:
        plotter.plot_method_counts(
            data_path=args.counts_input,
            annual_output_png=args.annual_output_png,
            annual_output_svg=args.annual_output_svg,
            cumulative_output_png=args.cumulative_output_png,
            cumulative_output_svg=args.cumulative_output_svg,
        )

    if PlotKind.SOLUTION_NMR_WEIGHT_STATS in args.plots:
        plotter.plot_solution_nmr_weight_stats(
            data_path=args.nmr_weights_input,
            avg_output_png=args.nmr_avg_output_png,
            avg_output_svg=args.nmr_avg_output_svg,
            median_output_png=args.nmr_median_output_png,
            median_output_svg=args.nmr_median_output_svg,
            max_output_png=args.nmr_max_output_png,
            max_output_svg=args.nmr_max_output_svg,
        )

    if PlotKind.SOLUTION_NMR_PERIOD_BOXPLOT in args.plots:
        plotter.plot_solution_nmr_period_boxplot(
            data_path=args.nmr_weights_input,
            output_png=args.nmr_boxplot_output_png,
            output_svg=args.nmr_boxplot_output_svg,
        )

    if PlotKind.SOLUTION_NMR_PERIOD_AREA in args.plots:
        plotter.plot_solution_nmr_period_area(
            data_path=args.nmr_weights_input,
            output_png=args.nmr_area_output_png,
            output_svg=args.nmr_area_output_svg,
        )

    if PlotKind.SOLUTION_NMR_PERIOD_AREA_SHARE in args.plots:
        plotter.plot_solution_nmr_period_area_share(
            data_path=args.nmr_weights_input,
            output_png=args.nmr_area_share_output_png,
            output_svg=args.nmr_area_share_output_svg,
        )

    if PlotKind.SOLUTION_NMR_PERIOD_AREA_CUMULATIVE_SHARE in args.plots:
        plotter.plot_solution_nmr_period_area_cumulative_share(
            data_path=args.nmr_weights_input,
            output_png=args.nmr_area_cumulative_share_output_png,
            output_svg=args.nmr_area_cumulative_share_output_svg,
        )

    if PlotKind.SOLUTION_NMR_MONOMER_SECONDARY in args.plots:
        plotter.plot_solution_nmr_monomer_secondary(
            data_path=args.nmr_monomer_secondary_input,
            output_png=args.nmr_monomer_secondary_output_png,
            output_svg=args.nmr_monomer_secondary_output_svg,
        )

    if PlotKind.SOLUTION_NMR_MONOMER_PRECISION in args.plots:
        plotter.plot_solution_nmr_monomer_precision(
            data_path=args.nmr_monomer_precision_input,
            output_png=args.nmr_monomer_precision_output_png,
            output_svg=args.nmr_monomer_precision_output_svg,
        )

    if PlotKind.SOLUTION_NMR_MONOMER_QUALITY in args.plots:
        plotter.plot_solution_nmr_monomer_quality(
            data_path=args.nmr_monomer_quality_input,
            clash_output_png=args.nmr_monomer_quality_clash_output_png,
            clash_output_svg=args.nmr_monomer_quality_clash_output_svg,
            rama_output_png=args.nmr_monomer_quality_rama_output_png,
            rama_output_svg=args.nmr_monomer_quality_rama_output_svg,
            side_output_png=args.nmr_monomer_quality_side_output_png,
            side_output_svg=args.nmr_monomer_quality_side_output_svg,
        )


if __name__ == "__main__":
    main()
