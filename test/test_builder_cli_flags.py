"""Tests for the dataset builder's public command-line switches."""

import sys
import unittest
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()
