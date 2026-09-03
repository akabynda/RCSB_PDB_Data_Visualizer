"""Tests for first-model NMR-to-X-ray RMSD preparation."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.pdb_dataset_builder import (
    CAResidueRecord,
    DatasetBuildConfig,
    RCSBClient,
    SolutionNMRMonomerXrayHomologRecord,
    SolutionNMRMonomerXrayRmsdBuilder,
    XrayPolymerEntityCandidateRecord,
)


def _ca_line(serial: int, resid: int, x: float, y: float, z: float) -> str:
    """Return a formatted ATOM alpha-carbon record for a test coordinate."""
    return (
        f"ATOM  {serial:5d}  CA  ALA A{resid:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{20.0:6.2f}"
        "           C\n"
    )


def _hetatm_ca_line(serial: int, resid: int, x: float, y: float, z: float) -> str:
    """Return a formatted HETATM alpha-carbon test record."""
    return (
        f"HETATM{serial:5d}  CA  MSE A{resid:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{20.0:6.2f}"
        "           C\n"
    )


def _coord_for_residue(resid: int) -> tuple[float, float, float]:
    """Return a deterministic three-dimensional coordinate for ``resid``."""
    return float(resid), float((resid * resid) % 7), float(resid % 5)


class XrayRmsdFirstModelTests(unittest.TestCase):
    """Verify residue selection for NMR-to-X-ray RMSD calculations."""

    def test_rejects_nmr_core_with_hetatm_ca_residue(self) -> None:
        """Reject an NMR structural core containing a HETATM alpha carbon."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nmr_path = Path(tmpdir) / "nmr_hetatm.pdb"
            xray_path = Path(tmpdir) / "xray.pdb"
            nmr_lines = ["MODEL        1\n"]
            for resid in range(1, 12):
                nmr_lines.append(_ca_line(resid, resid, *_coord_for_residue(resid)))
            nmr_lines.append(_hetatm_ca_line(12, 12, *_coord_for_residue(12)))
            nmr_lines.append("ENDMDL\n")
            nmr_path.write_text("".join(nmr_lines), encoding="utf-8")
            xray_path.write_text(
                "".join(
                    _ca_line(resid, resid, *_coord_for_residue(resid))
                    for resid in range(1, 13)
                ),
                encoding="utf-8",
            )

            result = SolutionNMRMonomerXrayRmsdBuilder._compute_ca_rmsd_to_xray(
                nmr_pdb_path=nmr_path,
                nmr_chain_id="A",
                nmr_core_start_seq_id=1,
                nmr_core_end_seq_id=12,
                xray_pdb_path=xray_path,
                xray_chain_id="A",
                sequence_identity_percent=100,
            )

            self.assertIsNone(result)

    def test_uses_clean_xray_region_instead_of_lower_rmsd_dirty_repeat(
        self,
    ) -> None:
        """Never select a repeated X-ray core that contains HETATM CA."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nmr_path = Path(tmpdir) / "nmr.pdb"
            xray_path = Path(tmpdir) / "xray_repeats.pdb"
            nmr_path.write_text(
                "".join(
                    _ca_line(resid, resid, *_coord_for_residue(resid))
                    for resid in range(1, 12)
                ),
                encoding="utf-8",
            )

            xray_lines = [
                _ca_line(resid, resid, *_coord_for_residue(resid))
                for resid in range(1, 12)
            ]
            xray_lines.append(_hetatm_ca_line(12, 5, *_coord_for_residue(5)))
            xray_lines.append(_hetatm_ca_line(13, 50, 0.0, 0.0, 0.0))
            for offset, resid in enumerate(range(101, 112), start=1):
                x, y, z = _coord_for_residue(offset)
                if offset == 6:
                    z += 2.0
                xray_lines.append(_ca_line(13 + offset, resid, x, y, z))
            xray_path.write_text("".join(xray_lines), encoding="utf-8")

            result = SolutionNMRMonomerXrayRmsdBuilder._compute_ca_rmsd_to_xray(
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
            self.assertEqual(result[4:], (101, 111))

    def test_ignores_missing_ca_atoms_in_later_nmr_models(self) -> None:
        """Base shared-residue selection on the complete first NMR model."""
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

            result = SolutionNMRMonomerXrayRmsdBuilder._compute_ca_rmsd_to_xray(
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
        """Retain modeled residues across numbering gaps in the first model."""
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

            result = SolutionNMRMonomerXrayRmsdBuilder._compute_ca_rmsd_to_xray(
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

    def test_extremes_prepares_nmr_core_once_for_all_candidates(self) -> None:
        """Reuse invariant NMR parsing across an entry's RMSD candidates."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = DatasetBuildConfig()
            homolog = SolutionNMRMonomerXrayHomologRecord(
                entry_id="1NMR",
                year=2000,
                sequence_identity_percent=100,
                nmr_core_start_seq_id=1,
                nmr_core_end_seq_id=11,
                nmr_query_sequence_length=11,
                xray_homolog_entry_ids=("1AAA", "2BBB"),
                xray_homolog_entity_ids=("1AAA_1", "2BBB_1"),
                has_xray_homolog=True,
            )
            candidates = tuple(
                XrayPolymerEntityCandidateRecord(
                    polymer_entity_id=f"{entry_id}_1",
                    entry_id=entry_id,
                    chain_ids=("A",),
                    resolution_angstrom=2.0,
                )
                for entry_id in ("1AAA", "2BBB")
            )
            builder = SolutionNMRMonomerXrayRmsdBuilder(
                client=RCSBClient(config),
                config=config,
                cache_dir=root,
                rmsd_workers=1,
                homolog_records=[homolog],
            )
            prepared_core = (
                tuple(CAResidueRecord(resid, "A", True) for resid in range(1, 12)),
                {},
            )

            with (
                patch.object(
                    builder,
                    "_download_pdb_if_needed",
                    return_value=root / "1NMR.pdb",
                ),
                patch(
                    "src.pdb_dataset_builder.load_cached_chain_id_map",
                    return_value={},
                ),
                patch.object(
                    builder,
                    "_prepare_nmr_core_data",
                    return_value=prepared_core,
                ) as prepare_core,
                patch.object(
                    builder,
                    "_compute_candidate_record",
                    return_value=None,
                ) as compute_candidate,
            ):
                result = builder._compute_extremes_record(
                    homolog=homolog,
                    nmr_chain_id="A",
                    candidates=candidates,
                )

            self.assertIsNone(result)
            self.assertEqual(prepare_core.call_count, 1)
            self.assertEqual(compute_candidate.call_count, 2)
            for call in compute_candidate.call_args_list:
                self.assertIs(call.kwargs["prepared_nmr_core"], prepared_core)


if __name__ == "__main__":
    unittest.main()
