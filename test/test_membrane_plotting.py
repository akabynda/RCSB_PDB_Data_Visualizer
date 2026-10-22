"""Validate cohort-level membrane counts, missing values and plotting routes."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.pdb_plot import PDBScientificPlotter, PlotConfig, PlotKind
from src.plotting import cli
from src.plotting.membrane import MODEL_GROUPS, PDBTM_COVERAGE_GROUPS


def cohort_table() -> pd.DataFrame:
    return pd.DataFrame({
        "entry_id": ["1aaa", "1bbb", "1ccc", "1ddd", "1eee", "1fff"],
        "year": [2022, 2022, 2022, 2024, 2024, 2025],
        "deposited_model_count": [1, 20, "", 2, None, 10],
        "has_pdbtm_annotation": [False, True, True, True, False, True],
        "pdbtm_segment_status": ["missing", "missing", "partial", "complete",
                                 "missing", "complete"],
    })


class MembraneTableTests(unittest.TestCase):
    def setUp(self) -> None:
        self.plotter = PDBScientificPlotter(PlotConfig())

    def test_counts_preserve_unknown_models_and_exclude_future_years(self) -> None:
        source = cohort_table()
        table = self.plotter._prepare_solution_nmr_membrane_table(source)
        annual, coverage = self.plotter._summarize_solution_nmr_membrane_table(table)

        self.assertEqual(len(source), 6)
        self.assertEqual(table["entry_id"].tolist(), ["1AAA", "1BBB", "1CCC", "1DDD", "1EEE"])
        self.assertEqual(annual.index.tolist(), [2022, 2023, 2024])
        self.assertEqual(annual.columns.tolist(), list(MODEL_GROUPS))
        self.assertEqual(annual.to_numpy().tolist(), [[1, 1, 1], [0, 0, 0], [0, 1, 1]])
        self.assertEqual(coverage.index.tolist(), list(PDBTM_COVERAGE_GROUPS))
        self.assertEqual(coverage.tolist(), [2, 1, 1, 1])
        self.assertEqual(int(annual.to_numpy().sum()), 5)
        self.assertEqual(int(coverage.sum()), 5)

    def test_empty_table_retains_category_contract(self) -> None:
        table = self.plotter._prepare_solution_nmr_membrane_table(cohort_table().iloc[:0])
        annual, coverage = self.plotter._summarize_solution_nmr_membrane_table(table)
        self.assertTrue(annual.empty)
        self.assertEqual(annual.columns.tolist(), list(MODEL_GROUPS))
        self.assertEqual(coverage.tolist(), [0, 0, 0, 0])

    def test_duplicate_entries_are_rejected_instead_of_counted_twice(self) -> None:
        data = cohort_table()
        data.loc[1, "entry_id"] = " 1AAA "
        with self.assertRaisesRegex(ValueError, "one row per entry"):
            self.plotter._prepare_solution_nmr_membrane_table(data)

    def test_missing_years_are_reported_and_excluded_from_both_summaries(self) -> None:
        data = cohort_table().astype(object)
        data.loc[0, "year"] = np.nan
        data.loc[3, "year"] = " "
        with self.assertWarnsRegex(RuntimeWarning, "2 entries.*1AAA, 1DDD"):
            table = self.plotter._prepare_solution_nmr_membrane_table(data)
        annual, coverage = self.plotter._summarize_solution_nmr_membrane_table(table)
        self.assertEqual(table["entry_id"].tolist(), ["1BBB", "1CCC", "1EEE"])
        self.assertEqual(annual.to_numpy().tolist(), [[0, 1, 1], [0, 0, 0], [0, 0, 1]])
        self.assertEqual(coverage.tolist(), [1, 1, 1, 0])
        self.assertEqual(len(data), 6)

    def test_invalid_models_are_not_relabelled_as_unknown(self) -> None:
        for value in (0, -1, 1.5, np.inf, "invalid"):
            with self.subTest(value=value):
                data = cohort_table()
                data.loc[0, "deposited_model_count"] = value
                with self.assertRaises(ValueError):
                    self.plotter._prepare_solution_nmr_membrane_table(data)

    def test_required_fields_and_invalid_metadata_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            self.plotter._prepare_solution_nmr_membrane_table(cohort_table().drop(columns="year"))
        for column, value in (("entry_id", None), ("year", -1), ("year", 0),
                              ("year", np.inf), ("year", "invalid"),
                              ("year", 2023.5), ("has_pdbtm_annotation", "unknown"),
                              ("pdbtm_segment_status", None)):
            with self.subTest(column=column, value=value):
                data = cohort_table().astype(object)
                data.loc[0, column] = value
                with self.assertRaises(ValueError):
                    self.plotter._prepare_solution_nmr_membrane_table(data)


class MembranePlotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.plotter = PDBScientificPlotter(PlotConfig())

    def test_cli_all_and_explicit_route_include_cohort(self) -> None:
        self.assertIn(PlotKind.SOLUTION_NMR_MEMBRANE_PROTEINS, cli.parse_plot_kinds("all"))
        with patch.object(sys, "argv", ["pdb_plot.py", "--plots", "solution_nmr_membrane_proteins",
                                        "--nmr-membrane-input", "custom.csv"]):
            args = cli.parse_args()
        with (patch.object(cli, "parse_args", return_value=args),
              patch.object(cli, "PDBScientificPlotter") as plotter_class):
            cli.main()
        plotted = plotter_class.return_value
        self.assertEqual(len(plotted.method_calls), 1)
        plotted.plot_solution_nmr_membrane_proteins.assert_called_once_with(
            data_path=Path("custom.csv"),
            annual_output_png=Path("figures/solution_nmr_membrane_proteins_by_year.png"),
            annual_output_svg=Path("figures/solution_nmr_membrane_proteins_by_year.svg"),
            coverage_output_png=Path("figures/solution_nmr_membrane_pdbtm_coverage.png"),
            coverage_output_svg=Path("figures/solution_nmr_membrane_pdbtm_coverage.svg"),
        )

    def test_four_variants_per_plot_preserve_counts_and_hatches(self) -> None:
        rendered = []

        def inspect_figure(**kwargs) -> None:
            ax = kwargs["fig"].axes[0]
            rendered.append((kwargs["output_png"].name,
                             [bar.get_height() for bar in ax.patches],
                             [bar.get_hatch() for bar in ax.patches],
                             ax.get_ylim()[0]))

        with (patch.object(self.plotter, "_read_csv", return_value=cohort_table()),
              patch.object(self.plotter, "_scientific_style"),
              patch.object(self.plotter, "_save_figure_files", side_effect=inspect_figure)):
            self.plotter.plot_solution_nmr_membrane_proteins(
                Path("input.csv"), Path("annual.png"), Path("annual.svg"),
                Path("coverage.png"), Path("coverage.svg"),
            )
        self.assertEqual([row[0] for row in rendered], [
            "annual.png", "annual_no_title.png", "annual_open_axes.png",
            "annual_no_title_open_axes.png", "coverage.png", "coverage_no_title.png",
            "coverage_open_axes.png", "coverage_no_title_open_axes.png",
        ])
        for _, heights, hatches, bottom in rendered:
            self.assertEqual(sum(heights), 5)
            self.assertTrue(all(hatches))
            self.assertEqual(bottom, 0)
        self.assertEqual(rendered[4][1], [2, 1, 1, 1])
        self.assertEqual(plt.get_fignums(), [])

    def test_empty_cohort_still_exports_both_plot_families(self) -> None:
        with (patch.object(self.plotter, "_read_csv", return_value=cohort_table().iloc[:0]),
              patch.object(self.plotter, "_scientific_style"),
              patch.object(self.plotter, "_save_figure_files") as save):
            self.plotter.plot_solution_nmr_membrane_proteins(
                Path("input.csv"), Path("annual.png"), Path("annual.svg"),
                Path("coverage.png"), Path("coverage.svg"),
            )
        self.assertEqual(save.call_count, 8)
        self.assertEqual(plt.get_fignums(), [])


if __name__ == "__main__":
    unittest.main()
