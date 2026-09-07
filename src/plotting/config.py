"""Public plot selection and immutable figure configuration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .constants import (
    DEFAULT_FIGURE_HEIGHT_INCHES,
    JMR_TWO_COLUMN_WIDTH_INCHES,
)


class PlotKind(str, Enum):
    """Identify each plot family supported by the command-line interface."""

    METHOD_COUNTS = "method_counts"
    MEMBRANE_PROTEIN_COUNTS = "membrane_protein_counts"
    SOLUTION_NMR_PROGRAM_COUNTS = "solution_nmr_program_counts"
    SOLUTION_NMR_MONOMER_PROGRAM_CLUSTERS = "solution_nmr_monomer_program_clusters"
    SOLUTION_NMR_WEIGHT_STATS = "solution_nmr_weight_stats"
    SOLUTION_NMR_PERIOD_BOXPLOT = "solution_nmr_period_boxplot"
    SOLUTION_NMR_PERIOD_AREA = "solution_nmr_period_area"
    SOLUTION_NMR_PERIOD_AREA_SHARE = "solution_nmr_period_area_share"
    SOLUTION_NMR_PERIOD_AREA_CUMULATIVE_SHARE = (
        "solution_nmr_period_area_cumulative_share"
    )
    SOLUTION_NMR_MONOMER_STRIDE_MODELED_FIRST_MODEL = (
        "solution_nmr_monomer_stride_modeled_first_model"
    )
    SOLUTION_NMR_MONOMER_PRECISION_STRIDE_MODELED_FIRST_MODEL_MEAN = (
        "solution_nmr_monomer_precision_stride_modeled_first_model_mean"
    )
    SOLUTION_NMR_MONOMER_PRECISION_STRIDE_MODELED_FIRST_MODEL_MEDIAN = (
        "solution_nmr_monomer_precision_stride_modeled_first_model_median"
    )
    SOLUTION_NMR_MONOMER_QUALITY = "solution_nmr_monomer_quality"
    SOLUTION_NMR_MONOMER_XRAY_HOMOLOGS = "solution_nmr_monomer_xray_homologs"
    SOLUTION_NMR_MONOMER_XRAY_HOMOLOGS_HISTORICAL = (
        "solution_nmr_monomer_xray_homologs_historical"
    )
    SOLUTION_NMR_MONOMER_XRAY_HOMOLOG_TIMING_SHARE = (
        "solution_nmr_monomer_xray_homolog_timing_share"
    )
    SOLUTION_NMR_MONOMER_XRAY_RMSD = "solution_nmr_monomer_xray_rmsd"
    SOLUTION_NMR_MONOMER_XRAY_RMSD_HISTORICAL = (
        "solution_nmr_monomer_xray_rmsd_historical"
    )
    SOLUTION_NMR_MONOMER_XRAY_RMSD_PRECISION_CORRELATION = (
        "solution_nmr_monomer_xray_rmsd_precision_correlation"
    )


@dataclass(frozen=True)
class PlotConfig:
    """Configure plot dimensions, labels, colors, and output variants."""

    height_inches: float = DEFAULT_FIGURE_HEIGHT_INCHES
    aspect_ratio: float = (
        JMR_TWO_COLUMN_WIDTH_INCHES / DEFAULT_FIGURE_HEIGHT_INCHES
    )
    titleless_suffix: str = "_no_title"
    open_axes_suffix: str = "_open_axes"
    dpi: int = 600
    x_label: str = "Deposition year"
    annual_title: str = (
        "Number of PDB structures by deposition year and experimental method"
    )
    annual_y_label: str = "Number of structures"
    membrane_annual_title: str = (
        "Number of membrane protein structures in PDB by deposition year"
    )
    membrane_annual_y_label: str = "Number of membrane protein structures"
    membrane_cumulative_title: str = (
        "Cumulative number of membrane protein structures in PDB by deposition year"
    )
    membrane_cumulative_y_label: str = (
        "Cumulative number of membrane protein structures"
    )
    membrane_method_annual_title: str = (
        "Number of membrane protein structures by deposition year and experimental method"
    )
    membrane_method_cumulative_title: str = (
        "Cumulative number of membrane protein structures by deposition year and experimental method"
    )
    nmr_program_annual_title: str = (
        "Number of NMR structures by deposition year and refinement program"
    )
    nmr_program_annual_y_label: str = "Number of structures"
    nmr_monomer_program_cluster_share_title: str = (
        "Share of NMR structures by structure-determination software"
    )
    nmr_monomer_program_cluster_share_y_label: str = "Share of structures (%)"
    cumulative_title: str = (
        "Cumulative number of PDB structures by deposition year and experimental method"
    )
    cumulative_y_label: str = "Cumulative number of structures"
    nmr_avg_title: str = (
        "Mean molecular weight of NMR structures by deposition year"
    )
    nmr_avg_y_label: str = "Mean molecular weight (kDa)"
    nmr_median_title: str = (
        "Median molecular weight of NMR structures by deposition year"
    )
    nmr_median_y_label: str = "Median molecular weight (kDa)"
    nmr_max_title: str = (
        "Maximum molecular weight of NMR structures by deposition year"
    )
    nmr_max_y_label: str = "Maximum molecular weight (kDa)"
    nmr_boxplot_title: str = (
        "Molecular weight distribution of NMR structures by period"
    )
    nmr_area_title: str = (
        "Cumulative number of NMR structures by weight range"
    )
    nmr_area_y_label: str = "Cumulative number of structures"
    nmr_area_share_title: str = (
        "Share of NMR structures by weight range"
    )
    nmr_area_share_y_label: str = "Share of structures (%)"
    nmr_area_cumulative_share_title: str = (
        "Cumulative share of NMR structures by weight range"
    )
    nmr_area_cumulative_share_y_label: str = "Share of cumulative structures (%)"
    nmr_monomer_stride_modeled_first_model_title: str = (
        "Secondary-structure content of NMR structures by deposition year"
    )
    nmr_monomer_stride_modeled_first_model_y_label: str = (
        "Secondary-structure content (%)"
    )
    nmr_monomer_precision_stride_mean_title: str = (
        "Mean ensemble RMSD of NMR structures by deposition year"
    )
    nmr_monomer_precision_stride_mean_y_label: str = (
        "Mean RMSD to average structure (Å)"
    )
    nmr_monomer_precision_stride_median_title: str = (
        "Median ensemble RMSD of NMR structures by deposition year"
    )
    nmr_monomer_precision_stride_median_y_label: str = (
        "Median RMSD to average structure (Å)"
    )
    nmr_monomer_quality_clash_title: str = (
        "Mean clash score for NMR structures by deposition year"
    )
    nmr_monomer_quality_clash_y_label: str = "Mean clash score"
    nmr_monomer_quality_rama_title: str = (
        "Mean percentage of Ramachandran outliers for NMR structures by deposition year"
    )
    nmr_monomer_quality_rama_y_label: str = (
        "Mean fraction of Ramachandran outliers (%)"
    )
    nmr_monomer_quality_side_title: str = (
        "Mean percentage of side-chain outliers for NMR structures by deposition year"
    )
    nmr_monomer_quality_side_y_label: str = "Mean fraction of side-chain outliers (%)"
    nmr_monomer_xray_homolog_95_title: str = (
        "Share of NMR structures with X-ray analog by deposition year (95% identity)"
    )
    nmr_monomer_xray_homolog_100_title: str = (
        "Share of NMR structures with X-ray analog by deposition year (100% identity)"
    )
    nmr_monomer_xray_homolog_y_label: str = "Structures with X-ray analog (%)"
    nmr_monomer_xray_homolog_95_historical_title: str = (
        "Share of NMR structures with prior X-ray analog by deposition year (95% identity)"
    )
    nmr_monomer_xray_homolog_100_historical_title: str = (
        "Share of NMR structures with prior X-ray analog by deposition year (100% identity)"
    )
    nmr_monomer_xray_homolog_95_cumulative_title: str = (
        "Cumulative share of NMR structures with X-ray analog (95% identity)"
    )
    nmr_monomer_xray_homolog_100_cumulative_title: str = (
        "Cumulative share of NMR structures with X-ray analog (100% identity)"
    )
    nmr_monomer_xray_homolog_95_historical_cumulative_title: str = (
        "Cumulative share of NMR structures with prior X-ray analog (95% identity)"
    )
    nmr_monomer_xray_homolog_100_historical_cumulative_title: str = (
        "Cumulative share of NMR structures with prior X-ray analog (100% identity)"
    )
    nmr_monomer_xray_homolog_95_timing_share_title: str = (
        "Share of NMR structures by availability of X-ray analog (95% identity)"
    )
    nmr_monomer_xray_homolog_100_timing_share_title: str = (
        "Share of NMR structures by availability of X-ray analog (100% identity)"
    )
    nmr_monomer_xray_homolog_timing_share_y_label: str = "Share of structures (%)"
    nmr_monomer_xray_rmsd_title: str = (
        "Mean RMSD(CA) of NMR structures to best-resolution X-ray analog by deposition year"
    )
    nmr_monomer_xray_rmsd_y_label: str = "Mean RMSD(CA) (Å)"
    nmr_monomer_xray_median_rmsd_title: str = (
        "Median RMSD(CA) between NMR structures and their best-resolution "
        "X-ray analogs by NMR deposition year"
    )
    nmr_monomer_xray_median_rmsd_y_label: str = (
        "Median RMSD to X-ray structure (Å)"
    )
    nmr_monomer_xray_min_rmsd_title: str = (
        "Mean RMSD(CA) of NMR structures to minimum-RMSD X-ray analog by deposition year"
    )
    nmr_monomer_xray_min_rmsd_y_label: str = "Mean minimum RMSD(CA) (Å)"
    nmr_monomer_xray_min_median_rmsd_title: str = (
        "Median RMSD between NMR and analog X-ray structures by deposition year"
    )
    nmr_monomer_xray_min_median_rmsd_y_label: str = "Median RMSD(CA) (Å)"
    nmr_monomer_xray_rmsd_extremes_mean_title: str = (
        "Mean RMSD(CA) to X-ray analogs by deposition year"
    )
    nmr_monomer_xray_rmsd_extremes_median_title: str = (
        "Median RMSD(CA) to X-ray analogs by deposition year"
    )
    nmr_monomer_xray_rmsd_extremes_y_label: str = "RMSD(CA) (Å)"
    nmr_monomer_xray_rmsd_precision_scatter_title: str = (
        "NMR / X-ray RMSD vs. NMR ensemble RMSD"
    )
    nmr_monomer_xray_rmsd_precision_yearly_corr_title: str = (
        "Within-year correlation between minimum X-ray RMSD(CA) and NMR precision"
    )
    nmr_monomer_xray_rmsd_precision_cumulative_corr_title: str = (
        "Cumulative correlation between minimum X-ray RMSD(CA) and NMR precision"
    )
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
    homolog_timing_colors: tuple[str, str, str] = (
        "#4c78a8",
        "#f58518",
        "#bab0ac",
    )

    def figure_size(self, height_scale: float = 1.0) -> tuple[float, float]:
        """Return figure width and scaled height in inches."""
        width = self.height_inches * self.aspect_ratio
        height = self.height_inches * height_scale
        return (width, height)
