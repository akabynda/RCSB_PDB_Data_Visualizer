import tempfile
import unittest
from pathlib import Path

from pdb_data_collector import (
    CollectorConfig,
    RCSBClient,
    SolutionNMRWeightRecord,
    write_solution_nmr_weights_csv,
)


class _WeightClient(RCSBClient):
    def __init__(self) -> None:
        super().__init__(CollectorConfig())
        self.last_payload: dict | None = None

    def _post_json(self, url: str, payload: dict) -> dict:
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
    def test_fetches_only_entry_total_weight(self) -> None:
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
