import tempfile
import unittest
from pathlib import Path

from pdb_data_collector import SolutionNMRMonomerXrayRmsdCollector


def _ca_line(serial: int, resid: int, x: float, y: float, z: float) -> str:
    return (
        f"ATOM  {serial:5d}  CA  ALA A{resid:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{20.0:6.2f}"
        "           C\n"
    )


def _coord_for_residue(resid: int) -> tuple[float, float, float]:
    return float(resid), float((resid * resid) % 7), float(resid % 5)


class XrayRmsdFirstModelTests(unittest.TestCase):
    def test_ignores_missing_ca_atoms_in_later_nmr_models(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            nmr_path = Path(tmpdir) / "nmr.pdb"
            xray_path = Path(tmpdir) / "xray.pdb"

            nmr_lines = ["MODEL        1\n"]
            serial = 1
            for resid in range(1, 12):
                nmr_lines.append(_ca_line(serial, resid, *_coord_for_residue(resid)))
                serial += 1
            nmr_lines.append("ENDMDL\nMODEL        2\n")
            for resid in range(1, 12):
                if resid == 5:
                    continue
                x, y, z = _coord_for_residue(resid)
                nmr_lines.append(_ca_line(serial, resid, x + 5.0, y, z))
                serial += 1
            nmr_lines.append("ENDMDL\n")
            nmr_path.write_text("".join(nmr_lines), encoding="utf-8")

            xray_lines = []
            for resid in range(1, 12):
                x, y, z = _coord_for_residue(resid)
                xray_lines.append(_ca_line(resid, resid, x + 10.0, y - 3.0, z + 2.0))
            xray_path.write_text("".join(xray_lines), encoding="utf-8")

            result = SolutionNMRMonomerXrayRmsdCollector._compute_ca_rmsd_to_xray(
                nmr_pdb_path=nmr_path,
                nmr_chain_id="A",
                nmr_core_start_seq_id=1,
                nmr_core_end_seq_id=11,
                xray_pdb_path=xray_path,
                xray_chain_id="A",
                sequence_identity_percent=100,
            )

            self.assertIsNotNone(result)
            assert result is not None
            self.assertEqual(result[0], 11)
            self.assertAlmostEqual(result[1], 0.0, places=5)

    def test_ignores_numbering_gaps_in_first_nmr_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            nmr_path = Path(tmpdir) / "nmr_gap.pdb"
            xray_path = Path(tmpdir) / "xray_gap.pdb"
            nmr_resids = list(range(1, 6)) + list(range(10, 16))

            nmr_lines = ["MODEL        1\n"]
            for serial, resid in enumerate(nmr_resids, start=1):
                nmr_lines.append(_ca_line(serial, resid, *_coord_for_residue(serial)))
            nmr_lines.append("ENDMDL\n")
            nmr_path.write_text("".join(nmr_lines), encoding="utf-8")

            xray_lines = []
            for serial in range(1, 12):
                x, y, z = _coord_for_residue(serial)
                xray_lines.append(_ca_line(serial, serial, x + 2.0, y - 4.0, z + 6.0))
            xray_path.write_text("".join(xray_lines), encoding="utf-8")

            result = SolutionNMRMonomerXrayRmsdCollector._compute_ca_rmsd_to_xray(
                nmr_pdb_path=nmr_path,
                nmr_chain_id="A",
                nmr_core_start_seq_id=1,
                nmr_core_end_seq_id=15,
                xray_pdb_path=xray_path,
                xray_chain_id="A",
                sequence_identity_percent=100,
            )

            self.assertIsNotNone(result)
            assert result is not None
            self.assertEqual(result[0], 11)
            self.assertAlmostEqual(result[1], 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
