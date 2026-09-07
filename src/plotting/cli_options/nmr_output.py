"""Register molecular-weight, secondary-structure, and quality output paths."""

from __future__ import annotations

import argparse
from pathlib import Path


def add_nmr_output_arguments(parser: argparse.ArgumentParser) -> None:
    """Register molecular-weight, secondary-structure, and quality output paths."""
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
        "--nmr-monomer-stride-modeled-first-model-output-png",
        type=Path,
        default=Path(
            "figures/solution_nmr_monomer_stride_modeled_first_model_by_year.png"
        ),
        help="Output PNG for SOLUTION NMR monomer STRIDE modeled-first-model secondary-structure plot.",
    )
    parser.add_argument(
        "--nmr-monomer-stride-modeled-first-model-output-svg",
        type=Path,
        default=Path(
            "figures/solution_nmr_monomer_stride_modeled_first_model_by_year.svg"
        ),
        help="Output SVG for SOLUTION NMR monomer STRIDE modeled-first-model secondary-structure plot.",
    )
    parser.add_argument(
        "--nmr-monomer-precision-stride-mean-output-png",
        type=Path,
        default=Path(
            "figures/solution_nmr_monomer_precision_stride_modeled_first_model_mean_rmsd_by_year.png"
        ),
        help="Output PNG for STRIDE-core monomer precision yearly-mean RMSD plot.",
    )
    parser.add_argument(
        "--nmr-monomer-precision-stride-mean-output-svg",
        type=Path,
        default=Path(
            "figures/solution_nmr_monomer_precision_stride_modeled_first_model_mean_rmsd_by_year.svg"
        ),
        help="Output SVG for STRIDE-core monomer precision yearly-mean RMSD plot.",
    )
    parser.add_argument(
        "--nmr-monomer-precision-stride-median-output-png",
        type=Path,
        default=Path(
            "figures/solution_nmr_monomer_precision_stride_modeled_first_model_median_rmsd_by_year.png"
        ),
        help="Output PNG for STRIDE-core monomer precision yearly-median RMSD plot.",
    )
    parser.add_argument(
        "--nmr-monomer-precision-stride-median-output-svg",
        type=Path,
        default=Path(
            "figures/solution_nmr_monomer_precision_stride_modeled_first_model_median_rmsd_by_year.svg"
        ),
        help="Output SVG for STRIDE-core monomer precision yearly-median RMSD plot.",
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
