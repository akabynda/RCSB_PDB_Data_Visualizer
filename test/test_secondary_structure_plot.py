"""Check selection of secondary-structure RMSD datasets and figure labels."""

from __future__ import annotations

import sys
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.plotting import cli
from src.plotting.config import PlotConfig
from src.plotting.plotter import PDBScientificPlotter


class SecondaryStructurePlotTests(unittest.TestCase):
    def parse(self, *arguments: str) -> Namespace:
        with patch.object(sys, "argv", ["pdb_plot.py", *arguments]):
            return cli.parse_args()

    def test_secondary_selection_changes_only_rmsd_default_paths(self) -> None:
        core = self.parse()
        secondary = self.parse("--rmsd-residue-selection", "secondary-structure")

        self.assertEqual(core.rmsd_residue_selection, "core")
        for name, path in vars(core).items():
            if not isinstance(path, Path):
                continue
            with self.subTest(argument=name):
                affected = name.startswith("nmr_monomer_precision_") or (
                    name.startswith("nmr_monomer_xray_") and "rmsd" in name
                )
                expected = (
                    path.with_name(f"{path.stem}_secondary_structure{path.suffix}")
                    if affected
                    else path
                )
                self.assertEqual(getattr(secondary, name), expected)

    def test_secondary_selection_preserves_explicit_paths_even_core_defaults(self) -> None:
        args = self.parse(
            "--rmsd-residue-selection=secondary-structure",
            "--nmr-monomer-precision-stride-modeled-first-model-input",
            "data/solution_nmr_monomer_precision_stride_modeled_first_model.csv",
            "--nmr-monomer-xray-rmsd-extremes-input=custom/extremes.csv",
            "--nmr-monomer-xray-rmsd-precision-scatter-output-png",
            "custom/figure7.png",
        )

        self.assertEqual(
            args.nmr_monomer_precision_stride_modeled_first_model_input,
            Path("data/solution_nmr_monomer_precision_stride_modeled_first_model.csv"),
        )
        self.assertEqual(
            args.nmr_monomer_xray_rmsd_extremes_input,
            Path("custom/extremes.csv"),
        )
        self.assertEqual(
            args.nmr_monomer_xray_rmsd_precision_scatter_output_png,
            Path("custom/figure7.png"),
        )

    def test_figure7_dispatch_uses_secondary_inputs_and_isolated_output(self) -> None:
        args = self.parse(
            "--plots", "solution_nmr_monomer_xray_rmsd_precision_correlation",
            "--rmsd-residue-selection", "secondary-structure",
        )
        with (
            patch.object(cli, "parse_args", return_value=args),
            patch.object(cli, "PDBScientificPlotter") as plotter_class,
        ):
            cli.main()

        self.assertEqual(
            plotter_class.call_args.kwargs["config"].rmsd_residue_selection,
            "secondary-structure",
        )
        plot = plotter_class.return_value.plot_solution_nmr_monomer_xray_rmsd_precision_correlation
        plot.assert_called_once()
        self.assertEqual(
            plot.call_args.kwargs["precision_data_path"],
            Path("data/solution_nmr_monomer_precision_stride_modeled_first_model_secondary_structure.csv"),
        )
        self.assertEqual(
            plot.call_args.kwargs["extremes_data_path"],
            Path("data/solution_nmr_monomer_xray_rmsd_extremes_secondary_structure.csv"),
        )
        plotter = PDBScientificPlotter(PlotConfig())
        output = plotter._grouped_figure_output_path(
            plot.call_args.kwargs["scatter_output_png"]
        )
        self.assertEqual(
            output,
            Path("figures")
            / "solution_nmr_monomer_xray_min_rmsd_precision_correlation_secondary_structure"
            / "solution_nmr_monomer_xray_min_rmsd_precision_correlation_secondary_structure.png",
        )

    def test_figure7_labels_identify_selected_atoms_on_both_axes(self) -> None:
        plotter = PDBScientificPlotter(
            PlotConfig(rmsd_residue_selection="secondary-structure")
        )
        precision = pd.DataFrame({
            "entry_id": ["a", "b", "c"],
            "year": [2020, 2020, 2020],
            "mean_rmsd_angstrom": [1.0, 2.0, 3.0],
            "residue_selection": ["secondary-structure"] * 3,
        })
        extremes = pd.DataFrame({
            "entry_id": ["a", "b", "c"],
            "year": [2020, 2020, 2020],
            "best_rmsd_ca_angstrom": [0.5, 2.0, 2.5],
            "worst_rmsd_ca_angstrom": [2.0, 3.0, 4.0],
            "residue_selection": ["secondary-structure"] * 3,
        })
        with (
            patch.object(plotter, "_read_csv", side_effect=[precision, extremes]),
            patch.object(plotter, "_scientific_style"),
            patch.object(plotter, "_render_figure") as render,
        ):
            plotter.plot_solution_nmr_monomer_xray_rmsd_precision_correlation(
                Path("precision.csv"), Path("extremes.csv"),
                Path("scatter.png"), Path("scatter.svg"),
                Path("yearly.png"), Path("yearly.svg"),
                Path("cumulative.png"), Path("cumulative.svg"),
            )
        self.assertEqual(render.call_count, 3)
        for call in render.call_args_list:
            self.assertIn(
                "Secondary-structure CA atoms (STRIDE H/G/I/E/B)",
                call.kwargs["title"],
            )
        scatter = render.call_args_list[0].kwargs
        self.assertIn("Secondary-structure CA atoms", scatter["x_label"])
        self.assertIn("Secondary-structure CA atoms", scatter["y_label"])

    def test_figure7_rejects_legacy_core_and_mixed_selection_inputs(self) -> None:
        precision = pd.DataFrame({
            "entry_id": ["a", "b"],
            "year": [2020, 2020],
            "mean_rmsd_angstrom": [1.0, 2.0],
        })
        extremes = pd.DataFrame({
            "entry_id": ["a", "b"],
            "year": [2020, 2020],
            "best_rmsd_ca_angstrom": [0.5, 2.0],
            "worst_rmsd_ca_angstrom": [2.0, 3.0],
        })
        cases = (
            ("secondary-structure", None, None, "Monomer precision CSV"),
            ("secondary-structure", ["core", "core"], None, "Monomer precision CSV"),
            (
                "secondary-structure",
                ["secondary-structure", "secondary-structure"],
                None,
                "X-ray RMSD extremes CSV",
            ),
            (
                "secondary-structure",
                ["secondary-structure", "secondary-structure"],
                ["secondary-structure", "core"],
                "X-ray RMSD extremes CSV",
            ),
            (
                "core",
                ["secondary-structure", "secondary-structure"],
                ["secondary-structure", "secondary-structure"],
                "Monomer precision CSV",
            ),
        )
        for selection, precision_states, extremes_states, dataset_name in cases:
            with self.subTest(
                selection=selection,
                precision_states=precision_states,
                extremes_states=extremes_states,
            ):
                precision_input, extremes_input = precision.copy(), extremes.copy()
                if precision_states is not None:
                    precision_input["residue_selection"] = precision_states
                if extremes_states is not None:
                    extremes_input["residue_selection"] = extremes_states
                plotter = PDBScientificPlotter(
                    PlotConfig(rmsd_residue_selection=selection)
                )
                with (
                    patch.object(
                        plotter, "_read_csv",
                        side_effect=[precision_input, extremes_input],
                    ),
                    patch.object(plotter, "_render_figure") as render,
                    self.assertRaisesRegex(
                        ValueError, f"{dataset_name} residue_selection.*expected",
                    ),
                ):
                    plotter.plot_solution_nmr_monomer_xray_rmsd_precision_correlation(
                        Path("custom/precision.csv"), Path("custom/extremes.csv"),
                        Path("scatter.png"), Path("scatter.svg"),
                        Path("yearly.png"), Path("yearly.svg"),
                        Path("cumulative.png"), Path("cumulative.svg"),
                    )
                render.assert_not_called()


if __name__ == "__main__":
    unittest.main()
