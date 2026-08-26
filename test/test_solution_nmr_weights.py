"""Tests for building solution-NMR molecular-weight datasets."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from src.pdb_dataset_builder import (
    DatasetBuildConfig,
    RCSBClient,
    SolutionNMRWeightBuilder,
    SolutionNMRWeightRecord,
    write_solution_nmr_weights_csv,
)


class _WeightClient(RCSBClient):
    """Provide deterministic weight-query responses for builder tests."""

    def __init__(self) -> None:
        """Initialize the fake client with the default build configuration."""
        super().__init__(DatasetBuildConfig())
        self.last_payload: dict | None = None

    def _post_json(self, url: str, payload: dict) -> dict:
        """Return weight fixture data selected from ``payload`` identifiers."""
        self.last_payload = payload
        return {
            "data": {
                "entries": [
                    {
                        "rcsb_id": "1ABC",
                        "rcsb_accession_info": {"deposit_date": "2001-02-03"},
                        "rcsb_entry_info": {"molecular_weight": 12.3456},
                    },
                    {
                        "rcsb_id": "2DEF",
                        "rcsb_accession_info": {"deposit_date": "2002-03-04"},
                        "rcsb_entry_info": {"molecular_weight": None},
                    },
                ]
            }
        }


class SolutionNMRWeightTests(unittest.TestCase):
    """Verify molecular-weight retrieval and CSV serialization."""

    def test_builder_requires_at_least_one_protein_entity(self) -> None:
        """Filter Figure 2 entries to structures containing a protein entity."""
        client = Mock(spec=RCSBClient)
        client.fetch_entry_ids_for_method.return_value = []

        records = SolutionNMRWeightBuilder(
            client=client,
            config=DatasetBuildConfig(),
        ).build()

        self.assertEqual(records, [])
        client.fetch_entry_ids_for_method.assert_called_once_with(
            method_label="SOLUTION NMR",
            query_value="SOLUTION NMR",
            require_protein_entities=True,
        )

    def test_fetches_only_entry_total_weight(self) -> None:
        """Use only the total deposited entry molecular weight."""
        client = _WeightClient()

        records = client.fetch_solution_nmr_weight_records_for_ids(["1ABC", "2DEF"])

        self.assertEqual(
            records,
            [
                SolutionNMRWeightRecord(
                    entry_id="1ABC",
                    year=2001,
                    molecular_weight_kda=12.3456,
                )
            ],
        )
        query = client.last_payload["query"]
        self.assertIn("molecular_weight", query)
        self.assertNotIn("polymer_entities", query)

    def test_writes_single_weight_column(self) -> None:
        """Write one normalized molecular-weight column to CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "weights.csv"

            write_solution_nmr_weights_csv(
                records=[
                    SolutionNMRWeightRecord(
                        entry_id="1ABC",
                        year=2001,
                        molecular_weight_kda=12.3456,
                    )
                ],
                output_path=output_path,
            )

            self.assertEqual(
                output_path.read_text(encoding="utf-8"),
                "entry_id,year,molecular_weight_kda\n1ABC,2001,12.346\n",
            )


if __name__ == "__main__":
    unittest.main()
