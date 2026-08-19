"""Tests for molecular-weight category boundaries in NMR plots."""

import unittest

import numpy as np
import pandas as pd

from src.pdb_plot import NMR_WEIGHT_LABELS, PDBScientificPlotter


class NMRWeightCategoryTests(unittest.TestCase):
    """Verify exact inclusion boundaries for molecular-weight bins."""

    def test_twenty_kda_is_in_middle_category(self) -> None:
        """Assign exactly 20 kDa to the middle category and larger values above."""
        table = pd.DataFrame(
            {
                "year": [2020, 2020, 2020, 2020],
                "molecular_weight_kda": [9.0, 10.0, 20.0, np.nextafter(20.0, np.inf)],
            }
        )

        counts = PDBScientificPlotter._build_weight_category_yearly_counts(table)

        self.assertEqual(list(counts.columns), list(NMR_WEIGHT_LABELS))
        self.assertEqual(counts.loc[2020, "<10 kDa"], 1)
        self.assertEqual(counts.loc[2020, "10-20 kDa"], 2)
        self.assertEqual(counts.loc[2020, ">20 kDa"], 1)


if __name__ == "__main__":
    unittest.main()
