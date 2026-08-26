"""Tests for NMR categorization in experimental-method counts."""

import unittest
from unittest.mock import Mock, call

from src.pdb_dataset_builder import (
    DatasetBuildConfig,
    ExperimentalMethod,
    PDBMethodYearlyBuilder,
    RCSBClient,
    YearlyCountRecord,
)


NMR_METHOD_SETS = (
    ("SOLUTION NMR",),
    ("SOLID-STATE NMR",),
    ("SOLID-STATE NMR", "SOLUTION NMR"),
)


class MethodCountsNMRTests(unittest.TestCase):
    """Verify that every supported NMR method contributes to the NMR category."""

    def test_nmr_category_contains_all_three_method_sets(self) -> None:
        """Expose both single methods and their exact two-method combination."""
        self.assertEqual(ExperimentalMethod.NMR.exact_method_sets, NMR_METHOD_SETS)

    def test_all_three_methods_are_counted_under_nmr_label(self) -> None:
        """Aggregate entries from every NMR method under the common NMR label."""
        client = Mock(spec=RCSBClient)
        entry_ids_by_method_set = {
            method_values: [f"NMR{index}"]
            for index, method_values in enumerate(NMR_METHOD_SETS, start=1)
        }
        client.fetch_entry_ids_for_method_set.side_effect = (
            lambda method_label, method_values, require_protein_entities: (
                entry_ids_by_method_set[method_values]
            )
        )
        client.fetch_deposit_dates_for_ids.return_value = [
            "2020-01-01",
            "2020-02-01",
            "2020-03-01",
        ]
        builder = PDBMethodYearlyBuilder(
            client=client,
            config=DatasetBuildConfig(max_workers=1),
        )

        records = builder.build([ExperimentalMethod.NMR])

        self.assertEqual(
            records,
            [YearlyCountRecord(year=2020, method="NMR", count=3)],
        )
        self.assertEqual(
            client.fetch_entry_ids_for_method_set.call_args_list,
            [
                call(
                    method_label="NMR",
                    method_values=method_values,
                    require_protein_entities=True,
                )
                for method_values in NMR_METHOD_SETS
            ],
        )

    def test_two_method_case_requires_exactly_the_nmr_pair(self) -> None:
        """Reject single-method entries and pairs containing an unrelated method."""
        client = RCSBClient(DatasetBuildConfig())
        client._post_json = Mock(
            return_value={
                "data": {
                    "entries": [
                        {
                            "rcsb_id": "PAIR",
                            "exptl": [
                                {"method": "SOLID-STATE NMR"},
                                {"method": "SOLUTION NMR"},
                            ],
                        },
                        {
                            "rcsb_id": "SINGLE",
                            "exptl": [{"method": "SOLUTION NMR"}],
                        },
                        {
                            "rcsb_id": "EXTRA",
                            "exptl": [
                                {"method": "SOLID-STATE NMR"},
                                {"method": "SOLUTION NMR"},
                                {"method": "X-RAY DIFFRACTION"},
                            ],
                        },
                    ]
                }
            }
        )

        entry_ids = client._filter_entry_ids_by_exact_methods(
            ["PAIR", "SINGLE", "EXTRA"],
            ("SOLID-STATE NMR", "SOLUTION NMR"),
        )

        self.assertEqual(entry_ids, ["PAIR"])


if __name__ == "__main__":
    unittest.main()
