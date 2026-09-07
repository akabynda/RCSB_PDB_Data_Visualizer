"""Parse plotting command-line options and dispatch the selected plot families."""

from __future__ import annotations

import argparse

import numpy as np

from .cli_options.count_output import add_count_output_arguments
from .cli_options.homolog_output import add_homolog_output_arguments
from .cli_options.input import add_input_arguments
from .cli_options.nmr_output import add_nmr_output_arguments
from .cli_options.rmsd_output import add_rmsd_output_arguments
from .config import PlotConfig, PlotKind
from .plotter import PDBScientificPlotter


def parse_plot_kinds(raw_value: str) -> list[PlotKind]:
    """Parse a comma-separated plot selection into plot-kind values."""
    if raw_value.strip().lower() == "all":
        return [
            PlotKind.METHOD_COUNTS,
            PlotKind.MEMBRANE_PROTEIN_COUNTS,
            PlotKind.SOLUTION_NMR_PROGRAM_COUNTS,
            PlotKind.SOLUTION_NMR_MONOMER_PROGRAM_CLUSTERS,
            PlotKind.SOLUTION_NMR_WEIGHT_STATS,
            PlotKind.SOLUTION_NMR_PERIOD_BOXPLOT,
            PlotKind.SOLUTION_NMR_PERIOD_AREA,
            PlotKind.SOLUTION_NMR_PERIOD_AREA_SHARE,
            PlotKind.SOLUTION_NMR_PERIOD_AREA_CUMULATIVE_SHARE,
            PlotKind.SOLUTION_NMR_MONOMER_STRIDE_MODELED_FIRST_MODEL,
            PlotKind.SOLUTION_NMR_MONOMER_PRECISION_STRIDE_MODELED_FIRST_MODEL_MEAN,
            PlotKind.SOLUTION_NMR_MONOMER_PRECISION_STRIDE_MODELED_FIRST_MODEL_MEDIAN,
            PlotKind.SOLUTION_NMR_MONOMER_QUALITY,
            PlotKind.SOLUTION_NMR_MONOMER_XRAY_HOMOLOGS,
            PlotKind.SOLUTION_NMR_MONOMER_XRAY_HOMOLOGS_HISTORICAL,
            PlotKind.SOLUTION_NMR_MONOMER_XRAY_HOMOLOG_TIMING_SHARE,
            PlotKind.SOLUTION_NMR_MONOMER_XRAY_RMSD,
            PlotKind.SOLUTION_NMR_MONOMER_XRAY_RMSD_HISTORICAL,
            PlotKind.SOLUTION_NMR_MONOMER_XRAY_RMSD_PRECISION_CORRELATION,
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


def parse_positive_float(raw_value: str) -> float:
    """Parse a positive floating-point command-line value."""
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Expected a positive number, got '{raw_value}'."
        ) from exc
    if not np.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError(
            f"Expected a positive number, got {value}."
        )
    return value


def parse_args() -> argparse.Namespace:
    """Parse plot selections, dataset paths, and figure output options."""
    parser = argparse.ArgumentParser(
        description="Plot publication-ready figures from PDB CSV datasets."
    )
    parser.add_argument(
        "--plots",
        type=parse_plot_kinds,
        default=[
            PlotKind.METHOD_COUNTS,
            PlotKind.MEMBRANE_PROTEIN_COUNTS,
            PlotKind.SOLUTION_NMR_PROGRAM_COUNTS,
            PlotKind.SOLUTION_NMR_MONOMER_PROGRAM_CLUSTERS,
            PlotKind.SOLUTION_NMR_WEIGHT_STATS,
            PlotKind.SOLUTION_NMR_PERIOD_BOXPLOT,
            PlotKind.SOLUTION_NMR_PERIOD_AREA,
            PlotKind.SOLUTION_NMR_PERIOD_AREA_SHARE,
            PlotKind.SOLUTION_NMR_PERIOD_AREA_CUMULATIVE_SHARE,
            PlotKind.SOLUTION_NMR_MONOMER_STRIDE_MODELED_FIRST_MODEL,
            PlotKind.SOLUTION_NMR_MONOMER_PRECISION_STRIDE_MODELED_FIRST_MODEL_MEAN,
            PlotKind.SOLUTION_NMR_MONOMER_PRECISION_STRIDE_MODELED_FIRST_MODEL_MEDIAN,
            PlotKind.SOLUTION_NMR_MONOMER_QUALITY,
            PlotKind.SOLUTION_NMR_MONOMER_XRAY_HOMOLOGS,
            PlotKind.SOLUTION_NMR_MONOMER_XRAY_HOMOLOGS_HISTORICAL,
            PlotKind.SOLUTION_NMR_MONOMER_XRAY_HOMOLOG_TIMING_SHARE,
            PlotKind.SOLUTION_NMR_MONOMER_XRAY_RMSD,
            PlotKind.SOLUTION_NMR_MONOMER_XRAY_RMSD_HISTORICAL,
            PlotKind.SOLUTION_NMR_MONOMER_XRAY_RMSD_PRECISION_CORRELATION,
        ],
        help="Comma-separated plot kinds or 'all'. Available: method_counts, membrane_protein_counts, solution_nmr_program_counts, solution_nmr_monomer_program_clusters, solution_nmr_weight_stats, solution_nmr_period_boxplot, solution_nmr_period_area, solution_nmr_period_area_share, solution_nmr_period_area_cumulative_share, solution_nmr_monomer_stride_modeled_first_model, solution_nmr_monomer_precision_stride_modeled_first_model_mean, solution_nmr_monomer_precision_stride_modeled_first_model_median, solution_nmr_monomer_quality, solution_nmr_monomer_xray_homologs, solution_nmr_monomer_xray_homologs_historical, solution_nmr_monomer_xray_homolog_timing_share, solution_nmr_monomer_xray_rmsd, solution_nmr_monomer_xray_rmsd_historical, solution_nmr_monomer_xray_rmsd_precision_correlation (default: all).",
    )
    add_input_arguments(parser)
    add_count_output_arguments(parser)
    add_nmr_output_arguments(parser)
    add_homolog_output_arguments(parser)
    add_rmsd_output_arguments(parser)
    parser.add_argument(
        "--svg",
        action="store_true",
        help="Also generate SVG files. By default only PNG files are generated.",
    )
    parser.add_argument(
        "--aspect-ratio",
        type=parse_positive_float,
        default=PlotConfig().aspect_ratio,
        help=(
            "Figure width-to-height ratio. Lower values make plots narrower "
            "(default: 1.496, matching a 7.48 x 5 in two-column figure)."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Generate every plot selected through command-line arguments."""
    args = parse_args()
    plotter = PDBScientificPlotter(
        config=PlotConfig(aspect_ratio=args.aspect_ratio),
        generate_svg=args.svg,
    )

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
            method_data_path=args.membrane_method_counts_input,
            annual_output_png=args.membrane_annual_output_png,
            annual_output_svg=args.membrane_annual_output_svg,
            cumulative_output_png=args.membrane_cumulative_output_png,
            cumulative_output_svg=args.membrane_cumulative_output_svg,
            method_annual_output_png=args.membrane_method_annual_output_png,
            method_annual_output_svg=args.membrane_method_annual_output_svg,
            method_cumulative_output_png=args.membrane_method_cumulative_output_png,
            method_cumulative_output_svg=args.membrane_method_cumulative_output_svg,
        )

    if PlotKind.SOLUTION_NMR_PROGRAM_COUNTS in args.plots:
        plotter.plot_solution_nmr_program_counts(
            data_path=args.nmr_program_counts_input,
            annual_output_png=args.nmr_program_annual_output_png,
            annual_output_svg=args.nmr_program_annual_output_svg,
        )

    if PlotKind.SOLUTION_NMR_MONOMER_PROGRAM_CLUSTERS in args.plots:
        plotter.plot_solution_nmr_monomer_program_clusters(
            data_path=args.nmr_monomer_program_cluster_summary_input,
            share_output_png=args.nmr_monomer_program_cluster_share_output_png,
            share_output_svg=args.nmr_monomer_program_cluster_share_output_svg,
            share_without_other_output_png=(
                args.nmr_monomer_program_cluster_share_without_other_output_png
            ),
            share_without_other_output_svg=(
                args.nmr_monomer_program_cluster_share_without_other_output_svg
            ),
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

    if PlotKind.SOLUTION_NMR_MONOMER_STRIDE_MODELED_FIRST_MODEL in args.plots:
        plotter.plot_solution_nmr_monomer_stride_modeled_first_model(
            data_path=args.nmr_monomer_stride_modeled_first_model_input,
            output_png=args.nmr_monomer_stride_modeled_first_model_output_png,
            output_svg=args.nmr_monomer_stride_modeled_first_model_output_svg,
        )

    if (
        PlotKind.SOLUTION_NMR_MONOMER_PRECISION_STRIDE_MODELED_FIRST_MODEL_MEAN
        in args.plots
    ):
        plotter.plot_solution_nmr_monomer_precision_stride_modeled_first_model_mean(
            data_path=args.nmr_monomer_precision_stride_modeled_first_model_input,
            output_png=args.nmr_monomer_precision_stride_mean_output_png,
            output_svg=args.nmr_monomer_precision_stride_mean_output_svg,
        )

    if (
        PlotKind.SOLUTION_NMR_MONOMER_PRECISION_STRIDE_MODELED_FIRST_MODEL_MEDIAN
        in args.plots
    ):
        plotter.plot_solution_nmr_monomer_precision_stride_modeled_first_model_median(
            data_path=args.nmr_monomer_precision_stride_modeled_first_model_input,
            output_png=args.nmr_monomer_precision_stride_median_output_png,
            output_svg=args.nmr_monomer_precision_stride_median_output_svg,
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

    if PlotKind.SOLUTION_NMR_MONOMER_XRAY_HOMOLOGS_HISTORICAL in args.plots:
        plotter.plot_solution_nmr_monomer_xray_homologs_historical(
            data_95_path=args.nmr_monomer_xray_homolog_95_historical_input,
            data_100_path=args.nmr_monomer_xray_homolog_100_historical_input,
            output_95_png=args.nmr_monomer_xray_homolog_95_historical_output_png,
            output_95_svg=args.nmr_monomer_xray_homolog_95_historical_output_svg,
            output_100_png=args.nmr_monomer_xray_homolog_100_historical_output_png,
            output_100_svg=args.nmr_monomer_xray_homolog_100_historical_output_svg,
            cumulative_output_95_png=args.nmr_monomer_xray_homolog_95_historical_cumulative_output_png,
            cumulative_output_95_svg=args.nmr_monomer_xray_homolog_95_historical_cumulative_output_svg,
            cumulative_output_100_png=args.nmr_monomer_xray_homolog_100_historical_cumulative_output_png,
            cumulative_output_100_svg=args.nmr_monomer_xray_homolog_100_historical_cumulative_output_svg,
        )

    if PlotKind.SOLUTION_NMR_MONOMER_XRAY_HOMOLOG_TIMING_SHARE in args.plots:
        plotter.plot_solution_nmr_monomer_xray_homolog_timing_share(
            regular_data_95_path=args.nmr_monomer_xray_homolog_95_input,
            regular_data_100_path=args.nmr_monomer_xray_homolog_100_input,
            historical_data_95_path=args.nmr_monomer_xray_homolog_95_historical_input,
            historical_data_100_path=args.nmr_monomer_xray_homolog_100_historical_input,
            counts_output_95_csv=(
                args.nmr_monomer_xray_homolog_95_timing_counts_output_csv
            ),
            counts_output_100_csv=(
                args.nmr_monomer_xray_homolog_100_timing_counts_output_csv
            ),
            output_95_png=args.nmr_monomer_xray_homolog_95_timing_share_output_png,
            output_95_svg=args.nmr_monomer_xray_homolog_95_timing_share_output_svg,
            output_100_png=args.nmr_monomer_xray_homolog_100_timing_share_output_png,
            output_100_svg=args.nmr_monomer_xray_homolog_100_timing_share_output_svg,
        )

    if PlotKind.SOLUTION_NMR_MONOMER_XRAY_RMSD in args.plots:
        plotter.plot_solution_nmr_monomer_xray_rmsd(
            data_path=args.nmr_monomer_xray_rmsd_input,
            extremes_data_path=args.nmr_monomer_xray_rmsd_extremes_input,
            mean_output_png=args.nmr_monomer_xray_rmsd_output_png,
            mean_output_svg=args.nmr_monomer_xray_rmsd_output_svg,
            median_output_png=args.nmr_monomer_xray_median_rmsd_output_png,
            median_output_svg=args.nmr_monomer_xray_median_rmsd_output_svg,
            min_mean_output_png=args.nmr_monomer_xray_min_rmsd_output_png,
            min_mean_output_svg=args.nmr_monomer_xray_min_rmsd_output_svg,
            min_median_output_png=args.nmr_monomer_xray_min_median_rmsd_output_png,
            min_median_output_svg=args.nmr_monomer_xray_min_median_rmsd_output_svg,
            extremes_mean_output_png=(
                args.nmr_monomer_xray_rmsd_extremes_mean_output_png
            ),
            extremes_mean_output_svg=(
                args.nmr_monomer_xray_rmsd_extremes_mean_output_svg
            ),
            extremes_median_output_png=(
                args.nmr_monomer_xray_rmsd_extremes_median_output_png
            ),
            extremes_median_output_svg=(
                args.nmr_monomer_xray_rmsd_extremes_median_output_svg
            ),
        )

    if PlotKind.SOLUTION_NMR_MONOMER_XRAY_RMSD_HISTORICAL in args.plots:
        plotter.plot_solution_nmr_monomer_xray_rmsd(
            data_path=args.nmr_monomer_xray_rmsd_historical_input,
            extremes_data_path=args.nmr_monomer_xray_rmsd_extremes_historical_input,
            mean_output_png=args.nmr_monomer_xray_rmsd_historical_output_png,
            mean_output_svg=args.nmr_monomer_xray_rmsd_historical_output_svg,
            median_output_png=args.nmr_monomer_xray_median_rmsd_historical_output_png,
            median_output_svg=args.nmr_monomer_xray_median_rmsd_historical_output_svg,
            min_mean_output_png=args.nmr_monomer_xray_min_rmsd_historical_output_png,
            min_mean_output_svg=args.nmr_monomer_xray_min_rmsd_historical_output_svg,
            min_median_output_png=(
                args.nmr_monomer_xray_min_median_rmsd_historical_output_png
            ),
            min_median_output_svg=(
                args.nmr_monomer_xray_min_median_rmsd_historical_output_svg
            ),
            extremes_mean_output_png=(
                args.nmr_monomer_xray_rmsd_extremes_historical_mean_output_png
            ),
            extremes_mean_output_svg=(
                args.nmr_monomer_xray_rmsd_extremes_historical_mean_output_svg
            ),
            extremes_median_output_png=(
                args.nmr_monomer_xray_rmsd_extremes_historical_median_output_png
            ),
            extremes_median_output_svg=(
                args.nmr_monomer_xray_rmsd_extremes_historical_median_output_svg
            ),
            title_suffix="(already-released X-ray analogs)",
        )

    if PlotKind.SOLUTION_NMR_MONOMER_XRAY_RMSD_PRECISION_CORRELATION in args.plots:
        plotter.plot_solution_nmr_monomer_xray_rmsd_precision_correlation(
            precision_data_path=(
                args.nmr_monomer_precision_stride_modeled_first_model_input
            ),
            extremes_data_path=args.nmr_monomer_xray_rmsd_extremes_input,
            scatter_output_png=(
                args.nmr_monomer_xray_rmsd_precision_scatter_output_png
            ),
            scatter_output_svg=(
                args.nmr_monomer_xray_rmsd_precision_scatter_output_svg
            ),
            yearly_correlation_output_png=(
                args.nmr_monomer_xray_rmsd_precision_yearly_correlation_output_png
            ),
            yearly_correlation_output_svg=(
                args.nmr_monomer_xray_rmsd_precision_yearly_correlation_output_svg
            ),
            cumulative_correlation_output_png=(
                args.nmr_monomer_xray_rmsd_precision_cumulative_correlation_output_png
            ),
            cumulative_correlation_output_svg=(
                args.nmr_monomer_xray_rmsd_precision_cumulative_correlation_output_svg
            ),
        )
