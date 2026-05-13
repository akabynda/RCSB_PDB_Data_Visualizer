import math
import unittest

import numpy as np

from pdb_data_collector import _ca_rmsd_to_mean_structure


class CaRmsdToMeanStructureTests(unittest.TestCase):
    def test_uses_root_mean_squared_deviation_across_models_and_atoms(self) -> None:
        coords = np.array(
            [
                [[0.0, 0.0, 0.0]],
                [[2.0, 0.0, 0.0]],
                [[5.0, 0.0, 0.0]],
            ],
            dtype=float,
        )

        self.assertAlmostEqual(
            _ca_rmsd_to_mean_structure(coords),
            math.sqrt(38.0) / 3.0,
        )


if __name__ == "__main__":
    unittest.main()
