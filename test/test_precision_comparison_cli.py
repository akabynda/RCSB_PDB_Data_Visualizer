"""Check opt-in Figure 4 histogram comparison dispatch and argument validation."""

from __future__ import annotations

import io
import sys
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from src.plotting import cli


MEDIAN_PLOT = "solution_nmr_monomer_precision_stride_modeled_first_model_median"
SECONDARY_INPUT_FLAG = "--nmr-monomer-precision-secondary-structure-input"


class PrecisionComparisonCliTests(unittest.TestCase):
    def parse(self, *arguments: str) -> Namespace:
        with patch.object(sys, "argv", ["pdb_plot.py", *arguments]):
            return cli.parse_args()

    def test_comparison_dispatch_preserves_explicit_inputs_and_output_base(
        self,
    ) -> None:
        args = self.parse(
            "--plots",
            MEDIAN_PLOT,
            "--nmr-monomer-precision-stride-modeled-first-model-input",
            "core.csv",
            SECONDARY_INPUT_FLAG,
            "secondary.csv",
            "--nmr-monomer-precision-stride-median-output-png",
            "custom/figure4.png",
            "--nmr-monomer-precision-stride-median-output-svg",
            "custom/figure4.svg",
            "--svg",
        )
        with (
            patch.object(cli, "parse_args", return_value=args),
            patch.object(cli, "PDBScientificPlotter") as plotter_class,
        ):
            cli.main()

        plotter = plotter_class.return_value
        plotter.plot_solution_nmr_monomer_precision_comparison.assert_called_once_with(
            core_data_path=Path("core.csv"),
            secondary_data_path=Path("secondary.csv"),
            output_png=Path("custom/figure4.png"),
            output_svg=Path("custom/figure4.svg"),
        )
        plotter.plot_solution_nmr_monomer_precision_stride_modeled_first_model_median.assert_not_called()
        self.assertEqual(
            plotter_class.call_args.kwargs["config"].rmsd_residue_selection, "core"
        )
        self.assertTrue(plotter_class.call_args.kwargs["generate_svg"])

    def test_default_plots_do_not_require_or_dispatch_comparison(self) -> None:
        for arguments in ((), ("--plots", "all"), ("--plots", MEDIAN_PLOT)):
            with self.subTest(arguments=arguments):
                args = self.parse(*arguments)
                self.assertIsNone(args.nmr_monomer_precision_secondary_structure_input)
                with (
                    patch.object(cli, "parse_args", return_value=args),
                    patch.object(cli, "PDBScientificPlotter") as plotter_class,
                ):
                    cli.main()

                plotter = plotter_class.return_value
                plotter.plot_solution_nmr_monomer_precision_comparison.assert_not_called()
                plotter.plot_solution_nmr_monomer_precision_stride_modeled_first_model_median.assert_called_once_with(
                    data_path=args.nmr_monomer_precision_stride_modeled_first_model_input,
                    output_png=args.nmr_monomer_precision_stride_median_output_png,
                    output_svg=args.nmr_monomer_precision_stride_median_output_svg,
                )

    def test_secondary_selection_without_comparison_keeps_existing_dispatch(
        self,
    ) -> None:
        args = self.parse(
            "--plots",
            MEDIAN_PLOT,
            "--rmsd-residue-selection",
            "secondary-structure",
        )
        with (
            patch.object(cli, "parse_args", return_value=args),
            patch.object(cli, "PDBScientificPlotter") as plotter_class,
        ):
            cli.main()

        plotter = plotter_class.return_value
        plotter.plot_solution_nmr_monomer_precision_comparison.assert_not_called()
        plotter.plot_solution_nmr_monomer_precision_stride_modeled_first_model_median.assert_called_once_with(
            data_path=Path(
                "data/solution_nmr_monomer_precision_stride_modeled_first_model_secondary_structure.csv"
            ),
            output_png=args.nmr_monomer_precision_stride_median_output_png,
            output_svg=args.nmr_monomer_precision_stride_median_output_svg,
        )

    def test_comparison_rejects_incompatible_selection_and_plot(self) -> None:
        cases = (
            (
                (
                    "--plots",
                    MEDIAN_PLOT,
                    "--rmsd-residue-selection",
                    "secondary-structure",
                ),
                "requires --rmsd-residue-selection core",
            ),
            (("--plots", "method_counts"), f"requires the {MEDIAN_PLOT} plot"),
            (
                (
                    "--plots",
                    "solution_nmr_monomer_precision_stride_modeled_first_model_mean",
                ),
                f"requires the {MEDIAN_PLOT} plot",
            ),
        )
        for arguments, message in cases:
            with self.subTest(arguments=arguments):
                with (
                    patch.object(sys, "stderr", new_callable=io.StringIO) as stderr,
                    self.assertRaises(SystemExit) as raised,
                ):
                    self.parse(*arguments, SECONDARY_INPUT_FLAG, "secondary.csv")
                self.assertEqual(raised.exception.code, 2)
                self.assertIn(message, stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
