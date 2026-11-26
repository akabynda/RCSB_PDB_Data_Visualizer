"""Tests for the dataset builder's public command-line switches."""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import src.dataset.workflows.quality as quality_workflows
from src.pdb_dataset_builder import parse_args


class DatasetBuilderCliFlagTests(unittest.TestCase):
    """Keep long-running build controls small and behaviorally consistent."""

    @staticmethod
    def _parse(*arguments: str):
        """Parse an isolated builder command line."""
        with patch.object(sys, "argv", ["pdb_dataset_builder.py", *arguments]):
            return parse_args()

    def test_rebuild_is_default_and_resume_is_explicit(self) -> None:
        """Rebuild outputs unless continuation is explicitly requested."""
        self.assertFalse(self._parse().resume)
        self.assertTrue(self._parse("--resume").resume)

    def test_program_clusters_read_the_quality_output_unless_input_is_given(
        self,
    ) -> None:
        """A combined run must cluster the quality CSV it has just written."""
        cases = (
            (
                ("--solution-nmr-monomer-quality-output", "new/quality.csv"),
                Path("new/quality.csv"),
            ),
            (
                (
                    "--solution-nmr-monomer-quality-output",
                    "new/quality.csv",
                    "--solution-nmr-monomer-program-cluster-input",
                    "old/quality.csv",
                ),
                Path("old/quality.csv"),
            ),
        )
        for arguments, expected_path in cases:
            with self.subTest(arguments=arguments):
                args = self._parse(
                    "--datasets", "solution_nmr_monomer_quality", *arguments
                )
                with (
                    patch.object(quality_workflows, "_import_filtered_structures"),
                    patch.object(
                        quality_workflows,
                        "read_solution_nmr_monomer_quality_csv",
                        return_value=[],
                    ) as read_quality,
                    self.assertRaises(RuntimeError),
                ):
                    quality_workflows.run_program_clusters(
                        args, MagicMock(), MagicMock()
                    )
                read_quality.assert_called_once_with(expected_path)


if __name__ == "__main__":
    unittest.main()
