"""Tests for the base SOLUTION NMR monomer model-length filter."""

import tempfile
import unittest
from pathlib import Path

from src.pdb_dataset_builder import DatasetBuildConfig, RCSBClient


def _ca_line(serial: int, resid: int) -> str:
    """Return a minimal alpha-carbon PDB record for ``resid``."""
    return (
        f"ATOM  {serial:5d}  CA  ALA A{resid:4d}    "
        f"{float(resid):8.3f}{0.0:8.3f}{0.0:8.3f}{1.0:6.2f}{20.0:6.2f}"
        "           C\n"
    )


def _monomer_entry(entry_id: str) -> dict:
    """Return minimal metadata accepted by the base monomer filter."""
    return {
        "rcsb_id": entry_id,
        "rcsb_entry_info": {"deposited_model_count": 2},
        "rcsb_accession_info": {"deposit_date": "2000-01-01"},
        "polymer_entities": [
            {
                "entity_poly": {
                    "type": "polypeptide(L)",
                    "rcsb_entity_polymer_type": "Protein",
                    "pdbx_strand_id": "A",
                }
            }
        ],
    }


class _LocalPDBClient(RCSBClient):
    """Use local coordinate fixtures for base-filter tests."""

    def __init__(self, cache_dir: Path, pdb_paths: dict[str, Path]) -> None:
        """Initialize the client with deterministic local coordinate paths."""
        super().__init__(
            DatasetBuildConfig(),
            solution_nmr_monomer_cache_dir=cache_dir,
        )
        self.pdb_paths = pdb_paths

    def _download_solution_nmr_monomer_pdb_if_needed(self, entry_id: str) -> Path:
        """Return the local coordinate fixture for ``entry_id``."""
        return self.pdb_paths[entry_id]


class SolutionNMRMonomerModelLengthTests(unittest.TestCase):
    """Verify model length is part of base monomer eligibility."""

    def test_base_filter_keeps_equal_full_chain_lengths(self) -> None:
        """Keep a monomer when every model has the same full-chain length."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pdb_path = root / "EQUAL.pdb"
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
            client = _LocalPDBClient(root, {"EQUAL": pdb_path})

            context = client._extract_solution_nmr_monomer_context(
                _monomer_entry("EQUAL")
            )

            self.assertIsNotNone(context)

    def test_base_filter_rejects_different_full_chain_lengths(self) -> None:
        """Reject a monomer even when the length difference is outside its core."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pdb_path = root / "UNEQUAL.pdb"
            pdb_path.write_text(
                "".join(
                    [
                        "MODEL        1\n",
                        _ca_line(1, 1),
                        _ca_line(2, 2),
                        _ca_line(3, 3),
                        _ca_line(4, 4),
                        "ENDMDL\n",
                        "MODEL        2\n",
                        _ca_line(5, 1),
                        _ca_line(6, 2),
                        _ca_line(7, 3),
                        "ENDMDL\n",
                    ]
                ),
                encoding="utf-8",
            )
            client = _LocalPDBClient(root, {"UNEQUAL": pdb_path})

            with self.assertLogs("src.pdb_dataset_builder", level="INFO") as logs:
                context = client._extract_solution_nmr_monomer_context(
                    _monomer_entry("UNEQUAL")
                )

            self.assertIsNone(context)
            self.assertTrue(
                any(
                    "Skipping SOLUTION NMR monomer UNEQUAL chain A: coordinate "
                    "models have different full-chain lengths" in message
                    for message in logs.output
                )
            )


if __name__ == "__main__":
    unittest.main()
