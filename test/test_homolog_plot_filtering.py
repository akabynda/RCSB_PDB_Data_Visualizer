"""Tests for filtering X-ray homolog records before plotting."""

import unittest

import pandas as pd

from src.pdb_plot import PDBScientificPlotter


class HomologPlotFilteringTests(unittest.TestCase):
    """Verify that plot inputs contain only performed homolog searches."""

    def test_omits_entries_without_a_performed_sequence_search(self) -> None:
        """Exclude records whose query sequence was too short for a search."""
        table = pd.DataFrame(
            [
                {
                    "entry_id": "VALID",
                    "year": 2000,
                    "sequence_identity_percent": 100,
                    "nmr_query_sequence_length": 11,
                    "has_xray_homolog": 0,
                },
                {
                    "entry_id": "NOT_SEARCHED",
                    "year": 2000,
                    "sequence_identity_percent": 100,
                    "nmr_query_sequence_length": 0,
                    "has_xray_homolog": 0,
                },
            ]
        )

        prepared = PDBScientificPlotter._prepare_monomer_xray_homolog_table(table)

        self.assertEqual(prepared["entry_id"].tolist(), ["VALID"])


if __name__ == "__main__":
    unittest.main()
