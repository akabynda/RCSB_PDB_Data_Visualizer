"""Tests for per-dataset reports of structures rejected by filters."""

import csv
import tempfile
import unittest
from pathlib import Path

from src.pdb_dataset_builder import (
    DatasetBuildConfig,
    DatasetKind,
    ExperimentalMethod,
    RCSBClient,
    _configure_dataset_filtered_csvs,
    _import_filtered_structures,
    _record_filtered_structure,
    _set_active_dataset_filtered_csvs,
    _set_active_dataset_warning_logs,
    filtered_structures_csv_path,
)


class _FilteredWeightClient(RCSBClient):
    """Return one valid weight, one missing weight, and one missing entry."""

    def __init__(self) -> None:
        """Initialize a deterministic client fixture."""
        super().__init__(DatasetBuildConfig())

    def _post_json(self, url: str, payload: dict) -> dict:
        """Return fixed GraphQL metadata for weight filtering."""
        return {
            "data": {
                "entries": [
                    {
                        "rcsb_id": "KEEP",
                        "rcsb_accession_info": {"deposit_date": "2001-02-03"},
                        "rcsb_entry_info": {"molecular_weight": 12.3},
                    },
                    {
                        "rcsb_id": "NO_WEIGHT",
                        "rcsb_accession_info": {"deposit_date": "2002-03-04"},
                        "rcsb_entry_info": {"molecular_weight": None},
                    },
                ]
            }
        }


class FilteredStructureCsvTests(unittest.TestCase):
    """Verify creation, routing, and content of filtered-structure CSVs."""

    def tearDown(self) -> None:
        """Prevent active report paths from leaking into another test."""
        _set_active_dataset_warning_logs(())
        _configure_dataset_filtered_csvs({})

    @staticmethod
    def _read_rows(path: Path) -> list[dict[str, str]]:
        """Read a filtered-structure report as dictionaries."""
        with path.open(newline="", encoding="utf-8") as csvfile:
            return list(csv.DictReader(csvfile))

    def test_creates_sibling_csv_and_deduplicates_filter_decisions(self) -> None:
        """Write one row per unique entry/reason pair to the active report."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "dataset.csv"
            _configure_dataset_filtered_csvs(
                {DatasetKind.SOLUTION_NMR_WEIGHTS: (output_path,)}
            )
            _set_active_dataset_filtered_csvs((output_path,))

            _record_filtered_structure("1ABC", "missing   molecular weight")
            _record_filtered_structure("1ABC", "missing molecular weight")

            filtered_path = filtered_structures_csv_path(output_path)
            self.assertEqual(filtered_path.name, "dataset_filtered.csv")
            self.assertEqual(
                self._read_rows(filtered_path),
                [{"entry_id": "1ABC", "reason": "missing molecular weight"}],
            )

    def test_routes_a_decision_only_to_selected_output(self) -> None:
        """Allow multi-output datasets to target the relevant paired report."""
        with tempfile.TemporaryDirectory() as tmpdir:
            first_output = Path(tmpdir) / "first.csv"
            second_output = Path(tmpdir) / "second.csv"
            _configure_dataset_filtered_csvs(
                {
                    DatasetKind.MEMBRANE_PROTEIN_COUNTS: (
                        first_output,
                        second_output,
                    )
                }
            )
            _set_active_dataset_filtered_csvs((second_output,))

            _record_filtered_structure("2DEF", "not a membrane protein")

            self.assertEqual(
                self._read_rows(filtered_structures_csv_path(first_output)), []
            )
            self.assertEqual(
                self._read_rows(filtered_structures_csv_path(second_output)),
                [{"entry_id": "2DEF", "reason": "not a membrane protein"}],
            )

    def test_weight_fetch_records_every_rejected_requested_entry(self) -> None:
        """Explain both invalid metadata and an omitted GraphQL entry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "weights.csv"
            _configure_dataset_filtered_csvs(
                {DatasetKind.SOLUTION_NMR_WEIGHTS: (output_path,)}
            )
            _set_active_dataset_filtered_csvs((output_path,))

            records = _FilteredWeightClient().fetch_solution_nmr_weight_records_for_ids(
                ["KEEP", "NO_WEIGHT", "NO_RESPONSE"]
            )

            self.assertEqual([record.entry_id for record in records], ["KEEP"])
            self.assertCountEqual(
                self._read_rows(filtered_structures_csv_path(output_path)),
                [
                    {
                        "entry_id": "NO_WEIGHT",
                        "reason": "entry molecular weight is missing or invalid",
                    },
                    {
                        "entry_id": "NO_RESPONSE",
                        "reason": "entry metadata missing from RCSB GraphQL response",
                    },
                ],
            )

    def test_imports_upstream_reasons_into_a_derived_dataset(self) -> None:
        """Carry exclusions forward when a dataset consumes another CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            upstream_output = Path(tmpdir) / "quality.csv"
            derived_output = Path(tmpdir) / "clusters.csv"
            upstream_filtered = filtered_structures_csv_path(upstream_output)
            upstream_filtered.write_text(
                "entry_id,reason\nUPSTREAM,quality metrics are missing\n",
                encoding="utf-8",
            )
            _configure_dataset_filtered_csvs(
                {
                    DatasetKind.SOLUTION_NMR_MONOMER_PROGRAM_CLUSTERS: (
                        derived_output,
                    )
                }
            )
            _set_active_dataset_filtered_csvs((derived_output,))

            _import_filtered_structures(upstream_output)

            self.assertEqual(
                self._read_rows(filtered_structures_csv_path(derived_output)),
                [
                    {
                        "entry_id": "UPSTREAM",
                        "reason": "quality metrics are missing",
                    }
                ],
            )

    def test_method_category_reports_server_side_filter_reasons(self) -> None:
        """Report method and protein filters after collecting raw candidates."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "method_counts.csv"
            _configure_dataset_filtered_csvs(
                {DatasetKind.METHOD_COUNTS: (output_path,)}
            )
            _set_active_dataset_filtered_csvs((output_path,))
            client = RCSBClient(DatasetBuildConfig(graphql_batch_size=10))
            client._fetch_paginated_identifiers = lambda **kwargs: [
                "SINGLE",
                "PAIR",
                "EXTRA",
                "NO_PROTEIN",
            ]
            client._post_json = lambda url, payload: {
                "data": {
                    "entries": [
                        {
                            "rcsb_id": "SINGLE",
                            "exptl": [{"method": "SOLUTION NMR"}],
                            "rcsb_entry_info": {"polymer_entity_count_protein": 1},
                        },
                        {
                            "rcsb_id": "PAIR",
                            "exptl": [
                                {"method": "SOLUTION NMR"},
                                {"method": "SOLID-STATE NMR"},
                            ],
                            "rcsb_entry_info": {"polymer_entity_count_protein": 1},
                        },
                        {
                            "rcsb_id": "EXTRA",
                            "exptl": [
                                {"method": "SOLUTION NMR"},
                                {"method": "X-RAY DIFFRACTION"},
                            ],
                            "rcsb_entry_info": {"polymer_entity_count_protein": 1},
                        },
                        {
                            "rcsb_id": "NO_PROTEIN",
                            "exptl": [{"method": "SOLUTION NMR"}],
                            "rcsb_entry_info": {"polymer_entity_count_protein": 0},
                        },
                    ]
                }
            }

            entry_ids = client.fetch_entry_ids_for_method_category(
                method=ExperimentalMethod.NMR,
                require_protein_entities=True,
            )

            self.assertEqual(entry_ids, ["PAIR", "SINGLE"])
            rows = self._read_rows(filtered_structures_csv_path(output_path))
            self.assertEqual({row["entry_id"] for row in rows}, {"EXTRA", "NO_PROTEIN"})
            self.assertTrue(
                any("experimental method set" in row["reason"] for row in rows)
            )
            self.assertIn(
                {"entry_id": "NO_PROTEIN", "reason": "entry has no protein polymer entities"},
                rows,
            )

    def test_base_monomer_filter_records_the_first_failed_rule(self) -> None:
        """Expose the concrete monomer eligibility reason in the paired CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "experiments.csv"
            _configure_dataset_filtered_csvs(
                {DatasetKind.SOLUTION_NMR_MONOMER_EXPERIMENTS: (output_path,)}
            )
            _set_active_dataset_filtered_csvs((output_path,))
            client = RCSBClient(DatasetBuildConfig())

            context = client._extract_solution_nmr_monomer_context(
                {
                    "rcsb_id": "ONE_MODEL",
                    "rcsb_entry_info": {"deposited_model_count": 1},
                    "rcsb_accession_info": {"deposit_date": "2000-01-01"},
                    "polymer_entities": [],
                }
            )

            self.assertIsNone(context)
            self.assertEqual(
                self._read_rows(filtered_structures_csv_path(output_path)),
                [
                    {
                        "entry_id": "ONE_MODEL",
                        "reason": "fewer than 2 deposited models",
                    }
                ],
            )


if __name__ == "__main__":
    unittest.main()
