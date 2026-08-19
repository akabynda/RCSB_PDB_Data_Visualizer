"""Tests for building solution-NMR monomer experiment datasets."""

import csv
import tempfile
import unittest
from pathlib import Path

from src.pdb_dataset_builder import (
    DatasetBuildConfig,
    RCSBClient,
    SolutionNMRMonomerExperimentsRecord,
    contains_noesy_experiment,
    write_solution_nmr_monomer_experiments_csv,
)


class _ExperimentsClient(RCSBClient):
    """Provide deterministic experiment-query responses for builder tests."""

    def __init__(self) -> None:
        """Initialize the fake client with the default build configuration."""
        super().__init__(DatasetBuildConfig())

    def _post_json(self, url: str, payload: dict) -> dict:
        """Return fixture data selected from identifiers in ``payload``."""
        return {
            "data": {
                "entries": [
                    {
                        "rcsb_id": "1ABC",
                        "rcsb_entry_info": {"deposited_model_count": 20},
                        "rcsb_accession_info": {"deposit_date": "2001-02-03"},
                        "polymer_entities": [
                            {
                                "entity_poly": {
                                    "type": "polypeptide(L)",
                                    "rcsb_entity_polymer_type": "Protein",
                                    "pdbx_strand_id": "A",
                                }
                            }
                        ],
                        "pdbx_nmr_exptl": [
                            {"type": "3D 1H-15N NOESY"},
                            {"type": "2D 1H-1H NOESY"},
                        ],
                    },
                    {
                        "rcsb_id": "2DEF",
                        "rcsb_entry_info": {"deposited_model_count": 1},
                        "rcsb_accession_info": {"deposit_date": "2002-03-04"},
                        "polymer_entities": [
                            {
                                "entity_poly": {
                                    "type": "polypeptide(L)",
                                    "rcsb_entity_polymer_type": "Protein",
                                    "pdbx_strand_id": "A",
                                }
                            }
                        ],
                        "pdbx_nmr_exptl": [{"type": "NOESY"}],
                    },
                ]
            }
        }


class SolutionNMRMonomerExperimentsTests(unittest.TestCase):
    """Verify experiment eligibility, normalization, and CSV serialization."""

    def test_fetches_experiments_only_for_eligible_monomers(self) -> None:
        """Fetch experiment metadata only for eligible monomer entries."""
        records = _ExperimentsClient().fetch_solution_nmr_monomer_experiment_records_for_ids(
            ["1ABC", "2DEF"]
        )

        self.assertEqual(
            records,
            [
                SolutionNMRMonomerExperimentsRecord(
                    entry_id="1ABC",
                    year=2001,
                    nmr_experiments_conducted=(
                        "3D 1H-15N NOESY",
                        "2D 1H-1H NOESY",
                    ),
                )
            ],
        )

    def test_noesy_is_counted_once_per_structure(self) -> None:
        """Count repeated NOESY records only once for an entry."""
        self.assertTrue(contains_noesy_experiment(("NOESY", "3D NOESY")))
        self.assertTrue(contains_noesy_experiment(("3d noesy-hsqc",)))
        self.assertTrue(contains_noesy_experiment(("3D_13C-SEPARATED_NOESY",)))
        self.assertFalse(contains_noesy_experiment(("HSQC", "TOCSY")))

    def test_writes_all_experiments_in_one_csv_cell(self) -> None:
        """Serialize all normalized experiments into one CSV field."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "experiments.csv"
            write_solution_nmr_monomer_experiments_csv(
                records=[
                    SolutionNMRMonomerExperimentsRecord(
                        entry_id="1ABC",
                        year=2001,
                        nmr_experiments_conducted=("HSQC", "3D NOESY"),
                    )
                ],
                output_path=output_path,
            )

            with output_path.open(newline="", encoding="utf-8") as handle:
                self.assertEqual(
                    list(csv.DictReader(handle)),
                    [
                        {
                            "entry_id": "1ABC",
                            "year": "2001",
                            "nmr_experiments_conducted": "HSQC; 3D NOESY",
                        }
                    ],
                )


if __name__ == "__main__":
    unittest.main()
