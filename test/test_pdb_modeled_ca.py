"""Tests for parsing modeled alpha-carbon residues from PDB files."""

import tempfile
import unittest
from pathlib import Path

from src.pdb_dataset_builder import (
    parse_first_model_ca_residues,
    parse_first_model_modeled_ca_auth_seq_ids,
    parse_models_ca_coords_with_stats,
)


def _ca_line(serial: int, resid: int, occupancy: float, x: float = 0.0) -> str:
    """Return a formatted ATOM alpha-carbon record for test fixtures."""
    return (
        f"ATOM  {serial:5d}  CA  ALA A{resid:4d}    "
        f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}{occupancy:6.2f}{20.0:6.2f}"
        "           C\n"
    )


def _hetatm_ca_line(serial: int, resname: str, resid: int, occupancy: float) -> str:
    """Return a formatted HETATM alpha-carbon record for test fixtures."""
    return (
        f"HETATM{serial:5d}  CA  {resname} A{resid:4d}    "
        f"{0.0:8.3f}{0.0:8.3f}{0.0:8.3f}{occupancy:6.2f}{20.0:6.2f}"
        "           C\n"
    )


class PdbModeledCaTests(unittest.TestCase):
    """Verify modeled-residue and coordinate parsing edge cases."""

    def test_first_model_modeled_ids_skip_zero_occupancy_and_preserve_gaps(
        self,
    ) -> None:
        """Skip zero-occupancy residues without collapsing sequence gaps."""
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
        """Exclude zero-occupancy alpha carbons from per-model maps."""
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

    def test_first_model_modeled_ids_ignore_seqadv_labels(self) -> None:
        """Ignore SEQADV annotations when collecting first-model residue IDs."""
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
                {-5, -4, -3, 1},
            )

    def test_model_coordinate_maps_ignore_seqadv_labels(self) -> None:
        """Ignore SEQADV annotations when parsing model coordinates."""
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

            self.assertEqual(
                [set(model_map) for model_map in model_maps],
                [{-5, -4, 1}, {-5, -4, 1}],
            )
            self.assertEqual(
                raw_counts,
                [{-5: 1, -4: 1, 1: 1}, {-5: 1, -4: 1, 1: 1}],
            )

    def test_nmr_modeled_ids_include_hetatm_records(self) -> None:
        """Include HETATM alpha carbons in modeled NMR residue identifiers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_path = Path(tmpdir) / "hetatm.pdb"
            pdb_path.write_text(
                "".join(
                    [
                        "MODEL        1\n",
                        _ca_line(1, 1, 1.0),
                        _hetatm_ca_line(2, "NLE", 2, 1.0),
                        _hetatm_ca_line(3, "A1BEB", 3, 1.0),
                        "ENDMDL\n",
                    ]
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                parse_first_model_modeled_ca_auth_seq_ids(pdb_path, "A"),
                {1, 2, 3},
            )

    def test_model_coordinate_maps_include_hetatm_ca_atoms(self) -> None:
        """Include HETATM alpha carbons in per-model coordinate maps."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_path = Path(tmpdir) / "hetatm_coords.pdb"
            pdb_path.write_text(
                "".join(
                    [
                        "MODEL        1\n",
                        _ca_line(1, 1, 1.0),
                        _hetatm_ca_line(2, "A1BEB", 2, 1.0),
                        "ENDMDL\n",
                    ]
                ),
                encoding="utf-8",
            )

            model_maps, raw_counts = parse_models_ca_coords_with_stats(pdb_path, "A")

            self.assertEqual([set(model_map) for model_map in model_maps], [{1, 2}])
            self.assertEqual(raw_counts, [{1: 1, 2: 1}])

    def test_xray_parser_keeps_long_name_hetatm_ca_records(self) -> None:
        """Preserve HETATM alpha carbons with nonstandard long residue names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_path = Path(tmpdir) / "long_hetatm.pdb"
            pdb_path.write_text(
                "".join(
                    [
                        "MODEL        1\n",
                        _ca_line(1, 1, 1.0),
                        _hetatm_ca_line(2, "A1BEB", 30, 1.0),
                        "ENDMDL\n",
                    ]
                ),
                encoding="utf-8",
            )

            residues = parse_first_model_ca_residues(
                pdb_path=pdb_path,
                chain_id="A",
                include_hetatm=True,
            )

            self.assertEqual([record.resid for record in residues], [1, 30])
            self.assertEqual(
                [(record.identity, record.is_standard_atom) for record in residues],
                [("A", True), ("HET:A1BEB", False)],
            )


if __name__ == "__main__":
    unittest.main()
