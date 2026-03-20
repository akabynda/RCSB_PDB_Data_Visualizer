from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import pandas as pd


class PlotKind(str, Enum):
    METHOD_COUNTS = "method_counts"
    MEMBRANE_PROTEIN_COUNTS = "membrane_protein_counts"
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
    SOLUTION_NMR_MONOMER_XRAY_HOMOLOGS = "solution_nmr_monomer_xray_homologs"
    SOLUTION_NMR_MONOMER_XRAY_RMSD = "solution_nmr_monomer_xray_rmsd"


@dataclass(frozen=True)
class PlotConfig:
    width_inches: float = 8.6
    height_inches: float = 5.0
    dpi: int = 600
    x_label: str = "Deposition year"
    annual_title: str = "Annual Number of PDB Structures by Experimental Method"
    annual_y_label: str = "Number of deposited structures"
    membrane_annual_title: str = "Annual Number of Membrane Protein Structures in PDB"
    membrane_annual_y_label: str = "Number of deposited membrane protein structures"
    membrane_cumulative_title: str = (
        "Cumulative Number of Membrane Protein Structures in PDB"
    )
    membrane_cumulative_y_label: str = (
        "Cumulative number of deposited membrane protein structures"
    )
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
    nmr_monomer_xray_homolog_95_title: str = (
        "SOLUTION NMR Monomeric Proteins: X-ray Analogs (95% Sequence Identity) by Year"
    )
    nmr_monomer_xray_homolog_100_title: str = (
        "SOLUTION NMR Monomeric Proteins: X-ray Analogs (100% Sequence Identity) by Year"
    )
    nmr_monomer_xray_homolog_y_label: str = "Structures with X-ray analog (%)"
    nmr_monomer_xray_homolog_95_cumulative_title: str = (
        "SOLUTION NMR Monomeric Proteins: Cumulative X-ray Analogs (95% Sequence Identity)"
    )
    nmr_monomer_xray_homolog_100_cumulative_title: str = (
        "SOLUTION NMR Monomeric Proteins: Cumulative X-ray Analogs (100% Sequence Identity)"
    )
    nmr_monomer_xray_rmsd_title: str = (
        "SOLUTION NMR Monomeric Proteins: Mean RMSD(CA) to Best-Resolution X-ray Analog by Year"
    )
    nmr_monomer_xray_rmsd_y_label: str = "Mean RMSD(CA) (Å)"
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


NMR_WEIGHT_BINS: tuple[float, ...] = (0.0, 10.0, 25.0, float("inf"))
NMR_WEIGHT_LABELS: tuple[str, ...] = ("<10 kDa", "10-25 kDa", ">25 kDa")
MAX_PLOT_YEAR: int = 2024


def parse_plot_kinds(raw_value: str) -> list[PlotKind]:
    if raw_value.strip().lower() == "all":
        return [
            PlotKind.METHOD_COUNTS,
            PlotKind.MEMBRANE_PROTEIN_COUNTS,
            PlotKind.SOLUTION_NMR_WEIGHT_STATS,
            PlotKind.SOLUTION_NMR_PERIOD_BOXPLOT,
            PlotKind.SOLUTION_NMR_PERIOD_AREA,
            PlotKind.SOLUTION_NMR_PERIOD_AREA_SHARE,
            PlotKind.SOLUTION_NMR_PERIOD_AREA_CUMULATIVE_SHARE,
            PlotKind.SOLUTION_NMR_MONOMER_SECONDARY,
            PlotKind.SOLUTION_NMR_MONOMER_PRECISION,
            PlotKind.SOLUTION_NMR_MONOMER_QUALITY,
            PlotKind.SOLUTION_NMR_MONOMER_XRAY_HOMOLOGS,
            PlotKind.SOLUTION_NMR_MONOMER_XRAY_RMSD,
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
    def __init__(self, config: PlotConfig, generate_svg: bool = True) -> None:
        self.config = config
        self.generate_svg = generate_svg
        self._csv_cache: dict[Path, pd.DataFrame] = {}

    def _read_csv(self, data_path: Path) -> pd.DataFrame:
        resolved = data_path.resolve()
        if resolved not in self._csv_cache:
            self._csv_cache[resolved] = pd.read_csv(resolved)
        return self._csv_cache[resolved]

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
        x_label: str | None = None,
    ) -> None:
        fig, ax = plt.subplots(
            figsize=(self.config.width_inches, self.config.height_inches)
        )
        draw_fn(ax)
        y_bottom, y_top = ax.get_ylim()
        """
        if y_bottom < 0.0 < y_top:
            ax.set_ylim(bottom=0.0)
        """
        ax.set_title(title, pad=10)
        ax.set_xlabel(x_label if x_label is not None else self.config.x_label)
        ax.set_ylabel(y_label)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.margins(x=0.01)
        fig.tight_layout()
        output_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_png, dpi=self.config.dpi)
        if self.generate_svg:
            fig.savefig(output_svg)
        plt.close(fig)

    @staticmethod
    def _limit_year_column(table: pd.DataFrame) -> pd.DataFrame:
        if "year" not in table.columns:
            return table
        return table.loc[table["year"] <= MAX_PLOT_YEAR].copy()

    @staticmethod
    def _validate_required_columns(
        df: pd.DataFrame, required_columns: set[str], dataset_name: str
    ) -> None:
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(
                f"{dataset_name} is missing required columns: {', '.join(sorted(missing))}"
            )

    @classmethod
    def _prepare_typed_table(
        cls,
        df: pd.DataFrame,
        required_columns: set[str],
        column_types: dict[str, type[Any]],
        dataset_name: str,
    ) -> pd.DataFrame:
        cls._validate_required_columns(df, required_columns, dataset_name)
        prepared = df.copy()
        for column, dtype in column_types.items():
            prepared[column] = prepared[column].astype(dtype)
        return prepared

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
        def draw(ax: plt.Axes) -> None:
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
                ax.legend(loc="upper left", frameon=False)

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
        x_left: float | None = None,
    ) -> None:
        def draw(ax: plt.Axes) -> None:
            ax.bar(x_values, y_values, color=color, width=width)
            if y_limits is not None:
                ax.set_ylim(*y_limits)
            if x_left is not None:
                ax.set_xlim(left=x_left)

        self._render_figure(
            output_png=output_png,
            output_svg=output_svg,
            title=title,
            y_label=y_label,
            draw_fn=draw,
        )

    @staticmethod
    def _build_weight_category_yearly_counts(table: pd.DataFrame) -> pd.DataFrame:
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
    ) -> None:
        def draw(ax: plt.Axes) -> None:
            ax.stackplot(
                table.index,
                *(table[label] for label in NMR_WEIGHT_LABELS),
                labels=NMR_WEIGHT_LABELS,
                colors=self.config.area_colors,
                alpha=0.85,
            )
            if y_limits is not None:
                ax.set_ylim(*y_limits)
            if x_left is not None:
                ax.set_xlim(left=x_left)
            ax.legend(loc="upper left", frameon=False, title="Weight category")

        self._render_figure(
            output_png=output_png,
            output_svg=output_svg,
            title=title,
            y_label=y_label,
            draw_fn=draw,
        )

    @staticmethod
    def _homolog_share_series(table: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        yearly_share = (
            table.groupby("year", as_index=True)["has_xray_homolog"]
            .mean()
            .mul(100.0)
            .sort_index()
        )
        yearly_counts = table.groupby("year", as_index=True)["entry_id"].count()
        yearly_yes = table.groupby("year", as_index=True)["has_xray_homolog"].sum()
        cumulative_share = (
            yearly_yes.cumsum().div(yearly_counts.cumsum()).mul(100.0).sort_index()
        )
        return yearly_share, cumulative_share

    @staticmethod
    def _prepare_method_count_table(df: pd.DataFrame) -> pd.DataFrame:
        PDBScientificPlotter._validate_required_columns(
            df=df,
            required_columns={"year", "method", "count"},
            dataset_name="Method count CSV",
        )
        limited = PDBScientificPlotter._limit_year_column(df)
        return (
            limited.pivot(index="year", columns="method", values="count")
            .fillna(0)
            .sort_index()
            .astype(int)
        )

    @staticmethod
    def _prepare_membrane_count_table(df: pd.DataFrame) -> pd.DataFrame:
        prepared = PDBScientificPlotter._prepare_typed_table(
            df=df,
            required_columns={"year", "count"},
            column_types={"year": int, "count": int},
            dataset_name="Membrane count CSV",
        )
        return PDBScientificPlotter._limit_year_column(prepared).sort_values("year")

    @staticmethod
    def _prepare_nmr_weight_table(df: pd.DataFrame) -> pd.DataFrame:
        prepared = PDBScientificPlotter._prepare_typed_table(
            df=df,
            required_columns={"entry_id", "year", "molecular_weight_kda"},
            column_types={"year": int, "molecular_weight_kda": float},
            dataset_name="NMR weight CSV",
        )
        return PDBScientificPlotter._limit_year_column(prepared)

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

    def plot_method_counts(
        self,
        data_path: Path,
        annual_output_png: Path,
        annual_output_svg: Path,
        cumulative_output_png: Path,
        cumulative_output_svg: Path,
    ) -> None:
        table = self._prepare_method_count_table(self._read_csv(data_path))
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
        )
        self._render_bar_series(
            output_png=median_output_png,
            output_svg=median_output_svg,
            title=self.config.nmr_median_title,
            y_label=self.config.nmr_median_y_label,
            x_values=stats.index,
            y_values=stats["median"],
            color=self.config.median_color,
        )
        self._render_bar_series(
            output_png=max_output_png,
            output_svg=max_output_svg,
            title=self.config.nmr_max_title,
            y_label=self.config.nmr_max_y_label,
            x_values=stats.index,
            y_values=stats["max"],
            color=self.config.max_color,
        )

    def plot_membrane_protein_counts(
        self,
        data_path: Path,
        annual_output_png: Path,
        annual_output_svg: Path,
        cumulative_output_png: Path,
        cumulative_output_svg: Path,
    ) -> None:
        table = self._prepare_membrane_count_table(self._read_csv(data_path))
        cumulative = table.copy()
        cumulative["count"] = cumulative["count"].cumsum()
        self._scientific_style()
        self._render_line_series(
            output_png=annual_output_png,
            output_svg=annual_output_svg,
            title=self.config.membrane_annual_title,
            y_label=self.config.membrane_annual_y_label,
            x_values=table["year"],
            y_values=table["count"],
            color="#17becf",
        )
        self._render_line_series(
            output_png=cumulative_output_png,
            output_svg=cumulative_output_svg,
            title=self.config.membrane_cumulative_title,
            y_label=self.config.membrane_cumulative_y_label,
            x_values=cumulative["year"],
            y_values=cumulative["count"],
            color="#17becf",
        )

    def plot_solution_nmr_period_boxplot(
        self, data_path: Path, output_png: Path, output_svg: Path
    ) -> None:
        table = self._prepare_nmr_weight_table(self._read_csv(data_path))
        periods = self._period_series(table)
        labels = list(periods.keys())
        values = [periods[label].values for label in labels]
        self._scientific_style()

        def draw(ax: plt.Axes) -> None:
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
        )

    def plot_solution_nmr_period_area(
        self, data_path: Path, output_png: Path, output_svg: Path
    ) -> None:
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
            x_left=1979,
        )

    def plot_solution_nmr_period_area_cumulative_share(
        self, data_path: Path, output_png: Path, output_svg: Path
    ) -> None:
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
        )

    @staticmethod
    def _prepare_monomer_secondary_table(df: pd.DataFrame) -> pd.DataFrame:
        prepared = PDBScientificPlotter._prepare_typed_table(
            df=df,
            required_columns={"entry_id", "year", "secondary_structure_percent"},
            column_types={"year": int, "secondary_structure_percent": float},
            dataset_name="Monomer secondary CSV",
        )
        return PDBScientificPlotter._limit_year_column(prepared)

    def plot_solution_nmr_monomer_secondary(
        self, data_path: Path, output_png: Path, output_svg: Path
    ) -> None:
        table = self._prepare_monomer_secondary_table(self._read_csv(data_path))
        self._scientific_style()
        yearly_mean = (
            table.groupby("year", as_index=True)["secondary_structure_percent"]
            .mean()
            .sort_index()
        )

        def draw(ax: plt.Axes) -> None:
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
            ax.set_ylim(0, 100)
            ax.legend(loc="upper left", frameon=False)

        self._render_figure(
            output_png=output_png,
            output_svg=output_svg,
            title=self.config.nmr_monomer_secondary_title,
            y_label=self.config.nmr_monomer_secondary_y_label,
            draw_fn=draw,
        )

    @staticmethod
    def _prepare_monomer_precision_table(df: pd.DataFrame) -> pd.DataFrame:
        prepared = PDBScientificPlotter._prepare_typed_table(
            df=df,
            required_columns={"entry_id", "year", "mean_rmsd_angstrom"},
            column_types={"year": int, "mean_rmsd_angstrom": float},
            dataset_name="Monomer precision CSV",
        )
        return PDBScientificPlotter._limit_year_column(prepared)

    def plot_solution_nmr_monomer_precision(
        self, data_path: Path, output_png: Path, output_svg: Path
    ) -> None:
        table = self._prepare_monomer_precision_table(self._read_csv(data_path))
        self._scientific_style()
        yearly_mean_rmsd = (
            table.groupby("year", as_index=True)["mean_rmsd_angstrom"]
            .mean()
            .sort_index()
        )

        self._render_line_series(
            output_png=output_png,
            output_svg=output_svg,
            title=self.config.nmr_monomer_precision_title,
            y_label=self.config.nmr_monomer_precision_y_label,
            x_values=yearly_mean_rmsd.index,
            y_values=yearly_mean_rmsd,
            color="#8c564b",
        )

    @staticmethod
    def _prepare_monomer_quality_table(df: pd.DataFrame) -> pd.DataFrame:
        prepared = PDBScientificPlotter._prepare_typed_table(
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
        return PDBScientificPlotter._limit_year_column(prepared)

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
        table = self._prepare_monomer_quality_table(self._read_csv(data_path))
        self._scientific_style()
        yearly = (
            table.groupby("year", as_index=True).mean(numeric_only=True).sort_index()
        )

        self._render_line_series(
            output_png=clash_output_png,
            output_svg=clash_output_svg,
            title=self.config.nmr_monomer_quality_clash_title,
            y_label=self.config.nmr_monomer_quality_clash_y_label,
            x_values=yearly.index,
            y_values=yearly["clashscore"],
            color="#8c564b",
        )
        self._render_line_series(
            output_png=rama_output_png,
            output_svg=rama_output_svg,
            title=self.config.nmr_monomer_quality_rama_title,
            y_label=self.config.nmr_monomer_quality_rama_y_label,
            x_values=yearly.index,
            y_values=yearly["ramachandran_outliers_percent"],
            color="#1f77b4",
        )
        self._render_line_series(
            output_png=side_output_png,
            output_svg=side_output_svg,
            title=self.config.nmr_monomer_quality_side_title,
            y_label=self.config.nmr_monomer_quality_side_y_label,
            x_values=yearly.index,
            y_values=yearly["sidechain_outliers_percent"],
            color="#ff7f0e",
        )

    @staticmethod
    def _prepare_monomer_xray_homolog_table(df: pd.DataFrame) -> pd.DataFrame:
        prepared = PDBScientificPlotter._prepare_typed_table(
            df=df,
            required_columns={
                "entry_id",
                "year",
                "sequence_identity_percent",
                "has_xray_homolog",
            },
            column_types={
                "year": int,
                "sequence_identity_percent": int,
                "has_xray_homolog": int,
            },
            dataset_name="Monomer X-ray homolog CSV",
        )
        return PDBScientificPlotter._limit_year_column(prepared)

    def plot_solution_nmr_monomer_xray_homologs(
        self,
        data_95_path: Path,
        data_100_path: Path,
        output_95_png: Path,
        output_95_svg: Path,
        output_100_png: Path,
        output_100_svg: Path,
        cumulative_output_95_png: Path,
        cumulative_output_95_svg: Path,
        cumulative_output_100_png: Path,
        cumulative_output_100_svg: Path,
    ) -> None:
        table_95 = self._prepare_monomer_xray_homolog_table(
            self._read_csv(data_95_path)
        )
        table_100 = self._prepare_monomer_xray_homolog_table(
            self._read_csv(data_100_path)
        )
        self._scientific_style()

        yearly_95, cumulative_share_95 = self._homolog_share_series(table_95)
        yearly_100, cumulative_share_100 = self._homolog_share_series(table_100)

        self._render_line_series(
            output_png=output_95_png,
            output_svg=output_95_svg,
            title=self.config.nmr_monomer_xray_homolog_95_title,
            y_label=self.config.nmr_monomer_xray_homolog_y_label,
            x_values=yearly_95.index,
            y_values=yearly_95,
            color="#1f77b4",
        )
        self._render_line_series(
            output_png=output_100_png,
            output_svg=output_100_svg,
            title=self.config.nmr_monomer_xray_homolog_100_title,
            y_label=self.config.nmr_monomer_xray_homolog_y_label,
            x_values=yearly_100.index,
            y_values=yearly_100,
            color="#2ca02c",
        )
        self._render_line_series(
            output_png=cumulative_output_95_png,
            output_svg=cumulative_output_95_svg,
            title=self.config.nmr_monomer_xray_homolog_95_cumulative_title,
            y_label=self.config.nmr_monomer_xray_homolog_y_label,
            x_values=cumulative_share_95.index,
            y_values=cumulative_share_95,
            color="#1f77b4",
        )
        self._render_line_series(
            output_png=cumulative_output_100_png,
            output_svg=cumulative_output_100_svg,
            title=self.config.nmr_monomer_xray_homolog_100_cumulative_title,
            y_label=self.config.nmr_monomer_xray_homolog_y_label,
            x_values=cumulative_share_100.index,
            y_values=cumulative_share_100,
            color="#2ca02c",
        )

    @staticmethod
    def _prepare_monomer_xray_rmsd_table(df: pd.DataFrame) -> pd.DataFrame:
        prepared = PDBScientificPlotter._prepare_typed_table(
            df=df,
            required_columns={"entry_id", "year", "rmsd_ca_angstrom"},
            column_types={"year": int, "rmsd_ca_angstrom": float},
            dataset_name="Monomer X-ray RMSD CSV",
        )
        return PDBScientificPlotter._limit_year_column(prepared)

    def plot_solution_nmr_monomer_xray_rmsd(
        self, data_path: Path, output_png: Path, output_svg: Path
    ) -> None:
        table = self._prepare_monomer_xray_rmsd_table(self._read_csv(data_path))
        self._scientific_style()
        yearly_mean_rmsd = (
            table.groupby("year", as_index=True)["rmsd_ca_angstrom"].mean().sort_index()
        )
        self._render_line_series(
            output_png=output_png,
            output_svg=output_svg,
            title=self.config.nmr_monomer_xray_rmsd_title,
            y_label=self.config.nmr_monomer_xray_rmsd_y_label,
            x_values=yearly_mean_rmsd.index,
            y_values=yearly_mean_rmsd,
            color="#9467bd",
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
            PlotKind.MEMBRANE_PROTEIN_COUNTS,
            PlotKind.SOLUTION_NMR_WEIGHT_STATS,
            PlotKind.SOLUTION_NMR_PERIOD_BOXPLOT,
            PlotKind.SOLUTION_NMR_PERIOD_AREA,
            PlotKind.SOLUTION_NMR_PERIOD_AREA_SHARE,
            PlotKind.SOLUTION_NMR_PERIOD_AREA_CUMULATIVE_SHARE,
            PlotKind.SOLUTION_NMR_MONOMER_SECONDARY,
            PlotKind.SOLUTION_NMR_MONOMER_PRECISION,
            PlotKind.SOLUTION_NMR_MONOMER_QUALITY,
            PlotKind.SOLUTION_NMR_MONOMER_XRAY_HOMOLOGS,
            PlotKind.SOLUTION_NMR_MONOMER_XRAY_RMSD,
        ],
        help="Comma-separated plot kinds or 'all'. Available: method_counts, membrane_protein_counts, solution_nmr_weight_stats, solution_nmr_period_boxplot, solution_nmr_period_area, solution_nmr_period_area_share, solution_nmr_period_area_cumulative_share, solution_nmr_monomer_secondary, solution_nmr_monomer_precision, solution_nmr_monomer_quality, solution_nmr_monomer_xray_homologs, solution_nmr_monomer_xray_rmsd (default: all).",
    )
    parser.add_argument(
        "--counts-input",
        type=Path,
        default=Path("data/pdb_method_counts_by_year.csv"),
        help="Input CSV for method_counts plot.",
    )
    parser.add_argument(
        "--membrane-counts-input",
        type=Path,
        default=Path("data/membrane_protein_counts_by_year.csv"),
        help="Input CSV for membrane_protein_counts plot.",
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
        "--nmr-monomer-xray-homolog-95-input",
        type=Path,
        default=Path("data/solution_nmr_monomer_xray_homologs_95.csv"),
        help="Input CSV for SOLUTION NMR monomer X-ray homologs at 95%% sequence identity.",
    )
    parser.add_argument(
        "--nmr-monomer-xray-homolog-100-input",
        type=Path,
        default=Path("data/solution_nmr_monomer_xray_homologs_100.csv"),
        help="Input CSV for SOLUTION NMR monomer X-ray homologs at 100%% sequence identity.",
    )
    parser.add_argument(
        "--nmr-monomer-xray-rmsd-input",
        type=Path,
        default=Path("data/solution_nmr_monomer_xray_rmsd.csv"),
        help="Input CSV for SOLUTION NMR monomer X-ray RMSD plot.",
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
        "--membrane-annual-output-png",
        type=Path,
        default=Path("figures/membrane_protein_counts_by_year.png"),
        help="Output PNG for annual membrane-protein count figure.",
    )
    parser.add_argument(
        "--membrane-annual-output-svg",
        type=Path,
        default=Path("figures/membrane_protein_counts_by_year.svg"),
        help="Output SVG for annual membrane-protein count figure.",
    )
    parser.add_argument(
        "--membrane-cumulative-output-png",
        type=Path,
        default=Path("figures/membrane_protein_counts_cumulative_by_year.png"),
        help="Output PNG for cumulative membrane-protein count figure.",
    )
    parser.add_argument(
        "--membrane-cumulative-output-svg",
        type=Path,
        default=Path("figures/membrane_protein_counts_cumulative_by_year.svg"),
        help="Output SVG for cumulative membrane-protein count figure.",
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
        default=Path(
            "figures/solution_nmr_monomer_quality_sidechain_outliers_by_year.png"
        ),
        help="Output PNG for monomer quality sidechain-outliers plot.",
    )
    parser.add_argument(
        "--nmr-monomer-quality-side-output-svg",
        type=Path,
        default=Path(
            "figures/solution_nmr_monomer_quality_sidechain_outliers_by_year.svg"
        ),
        help="Output SVG for monomer quality sidechain-outliers plot.",
    )
    parser.add_argument(
        "--nmr-monomer-xray-homolog-95-output-png",
        type=Path,
        default=Path("figures/solution_nmr_monomer_xray_homologs_95_by_year.png"),
        help="Output PNG for monomer X-ray homolog share plot at 95%% sequence identity.",
    )
    parser.add_argument(
        "--nmr-monomer-xray-homolog-95-output-svg",
        type=Path,
        default=Path("figures/solution_nmr_monomer_xray_homologs_95_by_year.svg"),
        help="Output SVG for monomer X-ray homolog share plot at 95%% sequence identity.",
    )
    parser.add_argument(
        "--nmr-monomer-xray-homolog-100-output-png",
        type=Path,
        default=Path("figures/solution_nmr_monomer_xray_homologs_100_by_year.png"),
        help="Output PNG for monomer X-ray homolog share plot at 100%% sequence identity.",
    )
    parser.add_argument(
        "--nmr-monomer-xray-homolog-100-output-svg",
        type=Path,
        default=Path("figures/solution_nmr_monomer_xray_homologs_100_by_year.svg"),
        help="Output SVG for monomer X-ray homolog share plot at 100%% sequence identity.",
    )
    parser.add_argument(
        "--nmr-monomer-xray-homolog-95-cumulative-output-png",
        type=Path,
        default=Path(
            "figures/solution_nmr_monomer_xray_homologs_95_cumulative_share_by_year.png"
        ),
        help="Output PNG for cumulative monomer X-ray homolog share plot at 95%% sequence identity.",
    )
    parser.add_argument(
        "--nmr-monomer-xray-homolog-95-cumulative-output-svg",
        type=Path,
        default=Path(
            "figures/solution_nmr_monomer_xray_homologs_95_cumulative_share_by_year.svg"
        ),
        help="Output SVG for cumulative monomer X-ray homolog share plot at 95%% sequence identity.",
    )
    parser.add_argument(
        "--nmr-monomer-xray-homolog-100-cumulative-output-png",
        type=Path,
        default=Path(
            "figures/solution_nmr_monomer_xray_homologs_100_cumulative_share_by_year.png"
        ),
        help="Output PNG for cumulative monomer X-ray homolog share plot at 100%% sequence identity.",
    )
    parser.add_argument(
        "--nmr-monomer-xray-homolog-100-cumulative-output-svg",
        type=Path,
        default=Path(
            "figures/solution_nmr_monomer_xray_homologs_100_cumulative_share_by_year.svg"
        ),
        help="Output SVG for cumulative monomer X-ray homolog share plot at 100%% sequence identity.",
    )
    parser.add_argument(
        "--nmr-monomer-xray-rmsd-output-png",
        type=Path,
        default=Path("figures/solution_nmr_monomer_xray_rmsd_by_year.png"),
        help="Output PNG for monomer X-ray RMSD(CA) by year plot.",
    )
    parser.add_argument(
        "--nmr-monomer-xray-rmsd-output-svg",
        type=Path,
        default=Path("figures/solution_nmr_monomer_xray_rmsd_by_year.svg"),
        help="Output SVG for monomer X-ray RMSD(CA) by year plot.",
    )
    parser.add_argument(
        "--no-svg",
        action="store_true",
        help="Disable SVG output generation.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plotter = PDBScientificPlotter(config=PlotConfig(), generate_svg=not args.no_svg)

    if PlotKind.METHOD_COUNTS in args.plots:
        plotter.plot_method_counts(
            data_path=args.counts_input,
            annual_output_png=args.annual_output_png,
            annual_output_svg=args.annual_output_svg,
            cumulative_output_png=args.cumulative_output_png,
            cumulative_output_svg=args.cumulative_output_svg,
        )

    if PlotKind.MEMBRANE_PROTEIN_COUNTS in args.plots:
        plotter.plot_membrane_protein_counts(
            data_path=args.membrane_counts_input,
            annual_output_png=args.membrane_annual_output_png,
            annual_output_svg=args.membrane_annual_output_svg,
            cumulative_output_png=args.membrane_cumulative_output_png,
            cumulative_output_svg=args.membrane_cumulative_output_svg,
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

    if PlotKind.SOLUTION_NMR_MONOMER_XRAY_HOMOLOGS in args.plots:
        plotter.plot_solution_nmr_monomer_xray_homologs(
            data_95_path=args.nmr_monomer_xray_homolog_95_input,
            data_100_path=args.nmr_monomer_xray_homolog_100_input,
            output_95_png=args.nmr_monomer_xray_homolog_95_output_png,
            output_95_svg=args.nmr_monomer_xray_homolog_95_output_svg,
            output_100_png=args.nmr_monomer_xray_homolog_100_output_png,
            output_100_svg=args.nmr_monomer_xray_homolog_100_output_svg,
            cumulative_output_95_png=args.nmr_monomer_xray_homolog_95_cumulative_output_png,
            cumulative_output_95_svg=args.nmr_monomer_xray_homolog_95_cumulative_output_svg,
            cumulative_output_100_png=args.nmr_monomer_xray_homolog_100_cumulative_output_png,
            cumulative_output_100_svg=args.nmr_monomer_xray_homolog_100_cumulative_output_svg,
        )

    if PlotKind.SOLUTION_NMR_MONOMER_XRAY_RMSD in args.plots:
        plotter.plot_solution_nmr_monomer_xray_rmsd(
            data_path=args.nmr_monomer_xray_rmsd_input,
            output_png=args.nmr_monomer_xray_rmsd_output_png,
            output_svg=args.nmr_monomer_xray_rmsd_output_svg,
        )


if __name__ == "__main__":
    main()
