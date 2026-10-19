"""Check the Figure 4 cohort and complete precision histogram counts."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from src.plotting.config import PlotConfig
from src.plotting.plotter import PDBScientificPlotter
from src.plotting.precision_comparison import HISTOGRAM_EDGES


class PrecisionComparisonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.core = pd.DataFrame(
            {
                "entry_id": ["a", "b", "c", "future"],
                "year": [2000, 2000, 2024, 2025],
                "chain_id": ["A"] * 4,
                "core_start_seq_id": [2] * 4,
                "core_end_seq_id": [40] * 4,
                "mean_rmsd_angstrom": [0.2, 1.2, 0.6, 10.0],
            }
        )
        self.secondary = pd.DataFrame(
            {
                "entry_id": ["c", "a", "extra", "future"],
                "year": [2024, 2000, 2020, 2025],
                "chain_id": ["A"] * 4,
                "core_start_seq_id": [2] * 4,
                "core_end_seq_id": [40] * 4,
                "mean_rmsd_angstrom": [0.3, 0.1, 0.4, 0.1],
                "residue_selection": ["secondary-structure"] * 4,
            }
        )

    def test_keeps_full_figure4_core_cohort_and_omits_secondary_extras(self) -> None:
        core, secondary = PDBScientificPlotter._prepare_precision_comparison_tables(
            self.core, self.secondary
        )

        self.assertEqual(core["entry_id"].tolist(), ["a", "b", "c"])
        self.assertEqual(secondary["entry_id"].tolist(), ["c", "a"])
        self.assertAlmostEqual(
            core.groupby("year")["mean_rmsd_angstrom"].median().loc[2000], 0.7
        )
        self.assertEqual(len(self.core), 4)
        self.assertEqual(len(self.secondary), 4)

    def test_accepts_explicit_core_selection(self) -> None:
        core, _ = PDBScientificPlotter._prepare_precision_comparison_tables(
            self.core.assign(residue_selection="core"), self.secondary
        )
        self.assertEqual(len(core), 3)

    def test_public_comparison_pairs_histograms_but_keeps_full_core_trend(self) -> None:
        plotter = PDBScientificPlotter(PlotConfig())

        def draw_without_saving(**kwargs: object) -> None:
            kwargs["draw_fn"](Mock())

        with (
            patch.object(plotter, "_read_csv", side_effect=[self.core, self.secondary]),
            patch.object(plotter, "_scientific_style"),
            patch.object(plotter, "_render_figure", side_effect=draw_without_saving),
            patch.object(plotter, "_draw_precision_histogram") as draw_histogram,
            patch.object(plotter, "_draw_core_precision_trend") as draw_trend,
            patch.object(plotter, "_render_precision_comparison_panels") as panels,
        ):
            plotter.plot_solution_nmr_monomer_precision_comparison(
                Path("core.csv"), Path("secondary.csv"),
                Path("figure4.png"), Path("figure4.svg"),
            )

        self.assertEqual(draw_histogram.call_count, 2)
        panels.assert_called_once()
        histogram_pairs = [
            call.args[1:3] for call in draw_histogram.call_args_list
        ]
        histogram_pairs.append(panels.call_args.args[:2])
        for core, secondary in histogram_pairs:
            self.assertEqual(core["entry_id"].tolist(), secondary["entry_id"].tolist())
            self.assertEqual(set(core["entry_id"]), {"a", "c"})
            self.assertEqual(len(core), 2)
            self.assertEqual(len(secondary), 2)

        draw_trend.assert_called_once()
        for yearly in (draw_trend.call_args.args[1], panels.call_args.args[2]):
            self.assertAlmostEqual(yearly.loc[2000], 0.7)
            self.assertAlmostEqual(yearly.loc[2024], 0.6)
            self.assertEqual(yearly.index.tolist(), [2000, 2024])

    def test_rejects_mislabelled_or_missing_secondary_selection(self) -> None:
        cases = (
            (self.core, self.secondary.drop(columns="residue_selection")),
            (self.core, self.secondary.assign(residue_selection="core")),
            (
                self.core,
                self.secondary.assign(
                    residue_selection=["core", "secondary-structure"] * 2
                ),
            ),
            (self.core.assign(residue_selection="secondary-structure"), self.secondary),
        )
        for core, secondary in cases:
            with self.subTest(core_selection=core.get("residue_selection")):
                with self.assertRaisesRegex(ValueError, "Expected .* precision data"):
                    PDBScientificPlotter._prepare_precision_comparison_tables(
                        core, secondary
                    )

    def test_rejects_mismatched_metadata_for_the_same_entry(self) -> None:
        for column, replacement in (
            ("year", 2001),
            ("chain_id", "B"),
            ("core_start_seq_id", 3),
            ("core_end_seq_id", 41),
        ):
            with self.subTest(column=column):
                secondary = self.secondary.copy()
                secondary.loc[secondary["entry_id"] == "a", column] = replacement
                with self.assertRaisesRegex(ValueError, f"disagree on {column}"):
                    PDBScientificPlotter._prepare_precision_comparison_tables(
                        self.core, secondary
                    )

    def test_rejects_missing_or_duplicate_entry_identifiers(self) -> None:
        for dataset in ("core", "secondary"):
            for invalid_id in (None, "a"):
                with self.subTest(dataset=dataset, entry_id=invalid_id):
                    core, secondary = self.core.copy(), self.secondary.copy()
                    frame = core if dataset == "core" else secondary
                    frame.loc[frame["entry_id"] != "a", "entry_id"] = invalid_id
                    with self.assertRaisesRegex(
                        ValueError, "Missing or duplicate entry_id"
                    ):
                        PDBScientificPlotter._prepare_precision_comparison_tables(
                            core, secondary
                        )

    def test_rejects_invalid_rmsd_before_plotting(self) -> None:
        for dataset in ("core", "secondary"):
            for invalid_rmsd in (-0.1, np.nan, np.inf):
                with self.subTest(dataset=dataset, rmsd=invalid_rmsd):
                    core, secondary = self.core.copy(), self.secondary.copy()
                    frame = core if dataset == "core" else secondary
                    frame.loc[frame["entry_id"] == "a", "mean_rmsd_angstrom"] = (
                        invalid_rmsd
                    )
                    with self.assertRaisesRegex(ValueError, "Invalid RMSD values"):
                        PDBScientificPlotter._prepare_precision_comparison_tables(
                            core, secondary
                        )

    def test_rejects_comparison_with_no_shared_structures(self) -> None:
        secondary = self.secondary.loc[self.secondary["entry_id"] == "extra"]
        with self.assertRaisesRegex(ValueError, "No secondary-structure results"):
            PDBScientificPlotter._prepare_precision_comparison_tables(
                self.core, secondary
            )

    def test_shared_bins_and_overflow_include_every_observation(self) -> None:
        np.testing.assert_allclose(np.diff(HISTOGRAM_EDGES), 0.1)
        for values, expected_counts, expected_overflow in (
            (
                [0.0, 0.05, 0.1, 0.15, 1.05, 2.999, 3.0, 4.0, 28.0],
                {0: 2, 1: 2, 10: 1, 29: 1},
                3,
            ),
            ([0.15, 0.25, 3.0], {1: 1, 2: 1}, 1),
            ([], {}, 0),
        ):
            with self.subTest(values=values):
                counts, overflow = PDBScientificPlotter._precision_histogram_counts(
                    np.asarray(values)
                )
                expected = np.zeros(30, dtype=int)
                for index, count in expected_counts.items():
                    expected[index] = count
                np.testing.assert_array_equal(counts, expected)
                self.assertEqual(overflow, expected_overflow)
                self.assertEqual(int(counts.sum()) + overflow, len(values))

    def test_histogram_rejects_values_that_cannot_be_counted(self) -> None:
        for values in ([-0.1], [np.nan], [np.inf]):
            with self.subTest(values=values):
                with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
                    PDBScientificPlotter._precision_histogram_counts(np.asarray(values))


if __name__ == "__main__":
    unittest.main()
