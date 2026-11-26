"""Check the Figure 3 structured-residue percentages built from stored fractions."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.plotting.config import PlotConfig
from src.plotting.plotter import PDBScientificPlotter

FRACTION_COLUMNS = (
    "stride_alpha_helix_fraction",
    "stride_3_10_helix_fraction",
    "stride_pi_helix_fraction",
    "stride_beta_strand_fraction",
    "stride_isolated_beta_bridge_fraction",
)


class StrideCompositionPlotTests(unittest.TestCase):
    """Keep fully structured chains despite six-decimal fraction rounding."""

    def _plotted_percentages(self, rows: list[tuple[float, ...]]) -> np.ndarray:
        """Plot rows of H/G/I/E/B fractions and return the scattered percentages."""
        table = pd.DataFrame(rows, columns=FRACTION_COLUMNS)
        table.insert(0, "year", 2000)
        table.insert(0, "entry_id", [f"{index}ABC" for index in range(len(rows))])
        ax = MagicMock()
        plotter = PDBScientificPlotter(PlotConfig())
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "stride.csv"
            table.to_csv(csv_path, index=False)
            with patch.object(
                plotter,
                "_render_figure",
                side_effect=lambda **kwargs: kwargs["draw_fn"](ax),
            ):
                plotter.plot_solution_nmr_monomer_stride_modeled_first_model(
                    csv_path, Path("unused.png"), Path("unused.svg")
                )
        return np.asarray(ax.scatter.call_args.args[1], dtype=float)

    def test_rounding_overshoot_is_kept_as_100_percent(self) -> None:
        # 1/6 + 1/6 + 4/6 written with six decimals sums to 1.000001.
        percentages = self._plotted_percentages(
            [(0.166667, 0.166667, 0.0, 0.666667, 0.0), (0.5, 0.0, 0.0, 0.0, 0.0)]
        )

        np.testing.assert_allclose(percentages, [100.0, 50.0])

    def test_values_beyond_rounding_error_are_removed(self) -> None:
        percentages = self._plotted_percentages(
            [(0.5, 0.0, 0.0, 0.50001, 0.0), (0.5, 0.0, 0.0, 0.0, 0.0)]
        )

        np.testing.assert_allclose(percentages, [50.0])


if __name__ == "__main__":
    unittest.main()
