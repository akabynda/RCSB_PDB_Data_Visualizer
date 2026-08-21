"""Tests for logging skipped NMR precision calculations."""

import tempfile
import unittest
from pathlib import Path

from src.pdb_dataset_builder import (
    DatasetBuildConfig,
    RCSBClient,
    SolutionNMRMonomerPrecisionBuilder,
)


def _ca_line(serial: int, resid: int) -> str:
    """Return a minimal alpha-carbon PDB record for ``resid``."""
    return (
        f"ATOM  {serial:5d}  CA  ALA A{resid:4d}    "
        f"{float(resid):8.3f}{0.0:8.3f}{0.0:8.3f}{1.0:6.2f}{20.0:6.2f}"
        "           C\n"
    )


class PrecisionSkipLoggingTests(unittest.TestCase):
    """Verify diagnostic logging for unusable precision inputs."""

    def test_logs_when_core_has_fewer_than_two_coordinate_models(self) -> None:
        """Log a skip reason when fewer than two models cover the core."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_path = Path(tmpdir) / "one_model.pdb"
            pdb_path.write_text(
                "".join(
                    [
                        "MODEL        1\n",
                        _ca_line(1, 1),
                        _ca_line(2, 2),
                        _ca_line(3, 3),
                        "ENDMDL\n",
                    ]
                ),
                encoding="utf-8",
            )
            builder = SolutionNMRMonomerPrecisionBuilder(
                client=RCSBClient(DatasetBuildConfig()),
                config=DatasetBuildConfig(),
                cache_dir=Path(tmpdir),
                precision_workers=1,
            )

            with self.assertLogs("src.pdb_dataset_builder", level="INFO") as logs:
                record = builder._build_record_from_core_range(
                    pdb_path=pdb_path,
                    entry_id="TEST",
                    year=2000,
                    chain_id="A",
                    core_start_seq_id=1,
                    core_end_seq_id=3,
                )

            self.assertIsNone(record)
            self.assertTrue(
                any(
                    "Skipping precision entry TEST chain A: fewer than 2 coordinate models"
                    in message
                    for message in logs.output
                )
            )

    def test_skips_models_with_different_lengths(self) -> None:
        """Exclude an entry when its coordinate models have unequal lengths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_path = Path(tmpdir) / "different_lengths.pdb"
            pdb_path.write_text(
                "".join(
                    [
                        "MODEL        1\n",
                        _ca_line(1, 1),
                        _ca_line(2, 2),
                        _ca_line(3, 3),
                        "ENDMDL\n",
                        "MODEL        2\n",
                        _ca_line(4, 1),
                        _ca_line(5, 2),
                        "ENDMDL\n",
                    ]
                ),
                encoding="utf-8",
            )
            builder = SolutionNMRMonomerPrecisionBuilder(
                client=RCSBClient(DatasetBuildConfig()),
                config=DatasetBuildConfig(),
                cache_dir=Path(tmpdir),
                precision_workers=1,
            )

            with self.assertLogs("src.pdb_dataset_builder", level="INFO") as logs:
                record = builder._build_record_from_core_range(
                    pdb_path=pdb_path,
                    entry_id="UNEQUAL",
                    year=2000,
                    chain_id="A",
                    core_start_seq_id=1,
                    core_end_seq_id=3,
                )

            self.assertIsNone(record)
            self.assertTrue(
                any(
                    "Skipping precision entry UNEQUAL chain A: coordinate models "
                    "have different lengths" in message
                    for message in logs.output
                )
            )

    def test_computes_precision_for_models_with_equal_lengths(self) -> None:
        """Keep an entry when all coordinate models have the same length."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_path = Path(tmpdir) / "equal_lengths.pdb"
            pdb_path.write_text(
                "".join(
                    [
                        "MODEL        1\n",
                        _ca_line(1, 1),
                        _ca_line(2, 2),
                        _ca_line(3, 3),
                        "ENDMDL\n",
                        "MODEL        2\n",
                        _ca_line(4, 1),
                        _ca_line(5, 2),
                        _ca_line(6, 3),
                        "ENDMDL\n",
                    ]
                ),
                encoding="utf-8",
            )
            builder = SolutionNMRMonomerPrecisionBuilder(
                client=RCSBClient(DatasetBuildConfig()),
                config=DatasetBuildConfig(),
                cache_dir=Path(tmpdir),
                precision_workers=1,
            )

            record = builder._build_record_from_core_range(
                pdb_path=pdb_path,
                entry_id="EQUAL",
                year=2000,
                chain_id="A",
                core_start_seq_id=1,
                core_end_seq_id=3,
            )

            self.assertIsNotNone(record)
            assert record is not None
            self.assertEqual(record.n_models, 2)
            self.assertEqual(record.n_ca_core_used, 3)
            self.assertAlmostEqual(record.mean_rmsd_angstrom, 0.0)


if __name__ == "__main__":
    unittest.main()
