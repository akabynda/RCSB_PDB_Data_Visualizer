import tempfile
import unittest
from pathlib import Path

from pdb_data_collector import (
    parse_first_model_modeled_ca_auth_seq_ids,
    parse_models_ca_coords_with_stats,
)


def _ca_line(serial: int, resid: int, occupancy: float, x: float = 0.0) -> str:
    return (
        f"ATOM  {serial:5d}  CA  ALA A{resid:4d}    "
        f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}{occupancy:6.2f}{20.0:6.2f}"
        "           C\n"
    )


class PdbModeledCaTests(unittest.TestCase):
    def test_first_model_modeled_ids_skip_zero_occupancy_and_preserve_gaps(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_path = Path(tmpdir) / "modeled.pdb"
            pdb_path.write_text(
                "".join(
                    [
                        "MODEL        1\n",
                        _ca_line(1, 10, 1.0),
                        _ca_line(2, 11, 0.0),
                        _ca_line(3, 13, 0.5),
                        "ENDMDL\n",
                        "MODEL        2\n",
                        _ca_line(4, 10, 1.0, x=1.0),
                        _ca_line(5, 11, 1.0, x=1.0),
                        _ca_line(6, 13, 1.0, x=1.0),
                        "ENDMDL\n",
                    ]
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                parse_first_model_modeled_ca_auth_seq_ids(pdb_path, "A"),
                {10, 13},
            )

    def test_model_coordinate_maps_skip_zero_occupancy_ca_atoms(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_path = Path(tmpdir) / "coords.pdb"
            pdb_path.write_text(
                "".join(
                    [
                        "MODEL        1\n",
                        _ca_line(1, 1, 1.0),
                        _ca_line(2, 2, 0.0),
                        "ENDMDL\n",
                        "MODEL        2\n",
                        _ca_line(3, 1, 1.0),
                        _ca_line(4, 2, 0.0),
                        "ENDMDL\n",
                    ]
                ),
                encoding="utf-8",
            )

            model_maps, raw_counts = parse_models_ca_coords_with_stats(pdb_path, "A")

            self.assertEqual([set(model_map) for model_map in model_maps], [{1}, {1}])
            self.assertEqual(raw_counts, [{1: 1}, {1: 1}])

    def test_first_model_modeled_ids_skip_excluded_seqadv_residues(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_path = Path(tmpdir) / "artifact.pdb"
            pdb_path.write_text(
                "".join(
                    [
                        "SEQADV 1CWW GLY A   -5  UNP  O14727              CLONING ARTIFACT\n",
                        "SEQADV 2M2E GLN A   -4  UNP  Q99543              EXPRESSION TAG\n",
                        "MODEL        1\n",
                        _ca_line(1, -5, 1.0),
                        _ca_line(2, -4, 1.0),
                        _ca_line(3, -3, 1.0),
                        _ca_line(4, 1, 1.0),
                        "ENDMDL\n",
                    ]
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                parse_first_model_modeled_ca_auth_seq_ids(pdb_path, "A"),
                {-3, 1},
            )

    def test_model_coordinate_maps_skip_excluded_seqadv_residues(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_path = Path(tmpdir) / "artifact_coords.pdb"
            pdb_path.write_text(
                "".join(
                    [
                        "SEQADV 1CWW GLY A   -5  UNP  O14727              CLONING ARTIFACT\n",
                        "SEQADV 2M2E GLN A   -4  UNP  Q99543              EXPRESSION TAG\n",
                        "MODEL        1\n",
                        _ca_line(1, -5, 1.0),
                        _ca_line(2, -4, 1.0),
                        _ca_line(3, 1, 1.0),
                        "ENDMDL\n",
                        "MODEL        2\n",
                        _ca_line(4, -5, 1.0),
                        _ca_line(5, -4, 1.0),
                        _ca_line(6, 1, 1.0),
                        "ENDMDL\n",
                    ]
                ),
                encoding="utf-8",
            )

            model_maps, raw_counts = parse_models_ca_coords_with_stats(pdb_path, "A")

            self.assertEqual([set(model_map) for model_map in model_maps], [{1}, {1}])
            self.assertEqual(raw_counts, [{1: 1}, {1: 1}])


if __name__ == "__main__":
    unittest.main()
