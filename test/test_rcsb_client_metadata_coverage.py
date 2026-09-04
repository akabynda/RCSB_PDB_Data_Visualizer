"""Network-free coverage for RCSB metadata parsing and validation branches."""

from __future__ import annotations

import copy
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests

from src import pdb_dataset_builder as builder


class RCSBClientMetadataTests(unittest.TestCase):
    """Validate GraphQL response parsing with representative partial payloads."""

    def setUp(self) -> None:
        self.client = builder.RCSBClient(
            builder.DatasetBuildConfig(page_size=2, graphql_batch_size=2, max_workers=2)
        )

    def tearDown(self) -> None:
        builder._configure_dataset_filtered_csvs({})

    @staticmethod
    def _monomer_entry(entry_id: str = "1ABC") -> dict:
        return {
            "rcsb_id": entry_id,
            "rcsb_entry_info": {"deposited_model_count": 2},
            "rcsb_accession_info": {"deposit_date": "2020-01-02"},
            "polymer_entities": [
                {
                    "entity_poly": {
                        "type": "polypeptide(L)",
                        "rcsb_entity_polymer_type": builder.PROTEIN_POLYMER_TYPE,
                        "pdbx_strand_id": "A",
                    }
                }
            ],
        }

    def test_similarity_memberships_are_normalized_and_filtered(self) -> None:
        self.assertEqual(self.client._normalize_similarity_cutoff("94.6"), 95)
        self.assertIsNone(self.client._normalize_similarity_cutoff(None))
        self.assertIsNone(self.client._normalize_similarity_cutoff("bad"))
        memberships = [
            None,
            {"aggregation_method": "other", "similarity_cutoff": 95, "group_id": "x"},
            {
                "aggregation_method": builder.SEQUENCE_IDENTITY_AGGREGATION_METHOD,
                "similarity_cutoff": "bad",
                "group_id": "bad",
            },
            {
                "aggregation_method": builder.SEQUENCE_IDENTITY_AGGREGATION_METHOD,
                "similarity_cutoff": 95.0,
                "group_id": "group-95",
            },
            {
                "aggregation_method": builder.SEQUENCE_IDENTITY_AGGREGATION_METHOD,
                "similarity_cutoff": 100,
                "group_id": "group-100",
            },
        ]
        self.assertEqual(
            self.client._extract_sequence_identity_groups(memberships),
            {95: "group-95", 100: "group-100"},
        )
        self.assertEqual(
            self.client._extract_sequence_identity_groups(memberships, {100}),
            {100: "group-100"},
        )

    def test_monomer_context_rejects_every_invalid_metadata_layer(self) -> None:
        with patch.object(
            self.client,
            "_solution_nmr_monomer_models_have_equal_lengths",
            return_value=True,
        ) as equal_lengths:
            self.assertIsNone(self.client._extract_solution_nmr_monomer_context({}))

            cases: list[dict] = []
            missing_count = self._monomer_entry("MISSING_COUNT")
            missing_count["rcsb_entry_info"] = None
            cases.append(missing_count)
            bad_count = self._monomer_entry("BAD_COUNT")
            bad_count["rcsb_entry_info"]["deposited_model_count"] = "many"
            cases.append(bad_count)
            one_model = self._monomer_entry("ONE")
            one_model["rcsb_entry_info"]["deposited_model_count"] = 1
            cases.append(one_model)
            no_accession = self._monomer_entry("NO_DATE")
            no_accession["rcsb_accession_info"] = None
            cases.append(no_accession)
            no_entity = self._monomer_entry("NO_ENTITY")
            no_entity["polymer_entities"] = []
            cases.append(no_entity)
            two_entities = self._monomer_entry("TWO_ENTITIES")
            two_entities["polymer_entities"] *= 2
            cases.append(two_entities)
            wrong_type = self._monomer_entry("WRONG_TYPE")
            wrong_type["polymer_entities"][0]["entity_poly"]["type"] = (
                "polydeoxyribonucleotide"
            )
            cases.append(wrong_type)
            wrong_polymer = self._monomer_entry("WRONG_POLYMER")
            wrong_polymer["polymer_entities"][0]["entity_poly"][
                "rcsb_entity_polymer_type"
            ] = "DNA"
            cases.append(wrong_polymer)
            no_chain = self._monomer_entry("NO_CHAIN")
            no_chain["polymer_entities"][0]["entity_poly"]["pdbx_strand_id"] = ""
            cases.append(no_chain)
            many_chains = self._monomer_entry("MANY_CHAINS")
            many_chains["polymer_entities"][0]["entity_poly"]["pdbx_strand_id"] = "A,B"
            cases.append(many_chains)

            for entry in cases:
                with self.subTest(entry=entry["rcsb_id"]):
                    self.assertIsNone(
                        self.client._extract_solution_nmr_monomer_context(entry)
                    )

            equal_lengths.return_value = False
            self.assertIsNone(
                self.client._extract_solution_nmr_monomer_context(
                    self._monomer_entry("UNEQUAL")
                )
            )
            equal_lengths.return_value = True
            context = self.client._extract_solution_nmr_monomer_context(
                self._monomer_entry("VALID")
            )
            self.assertIsNotNone(context)
            self.assertEqual((context or ())[0:3], ("VALID", 2020, 2))

    def test_paginated_search_stops_at_total_and_on_empty_page(self) -> None:
        self.client._post_json = Mock(
            side_effect=[
                {
                    "total_count": 3,
                    "result_set": [{"identifier": "A"}, {"score": 1}],
                },
                {
                    "total_count": 3,
                    "result_set": [
                        {"identifier": "B"},
                        {"identifier": "C"},
                    ],
                },
            ]
        )
        self.assertEqual(
            self.client._fetch_paginated_identifiers(
                {"type": "terminal"}, "entry", "test search"
            ),
            ["A", "B", "C"],
        )
        starts = [
            call.args[1]["request_options"]["paginate"]["start"]
            for call in self.client._post_json.call_args_list
        ]
        self.assertEqual(starts, [0, 1])

        self.client._post_json = Mock(return_value={"total_count": 5, "result_set": []})
        self.assertEqual(self.client._fetch_paginated_identifiers({}, "entry"), [])

    def test_method_and_annotation_queries_delegate_expected_filters(self) -> None:
        with patch.object(
            self.client, "_fetch_paginated_identifiers", return_value=[]
        ) as paginate:
            self.assertEqual(
                self.client.fetch_entry_ids_for_membrane_annotations(("A", "B")), []
            )
            self.assertEqual(paginate.call_args.kwargs["return_type"], "entry")
            self.assertEqual(
                paginate.call_args.kwargs["query"]["parameters"]["value"], ["A", "B"]
            )
            self.assertEqual(
                self.client.fetch_xray_polymer_entity_ids_for_group_ids([]), []
            )
            paginate.return_value = ["1ABC_1"]
            self.assertEqual(
                self.client.fetch_xray_polymer_entity_ids_for_group_ids(["group"]),
                ["1ABC_1"],
            )
            self.assertEqual(paginate.call_args.kwargs["return_type"], "polymer_entity")

        with self.assertRaises(ValueError):
            self.client.fetch_entry_ids_for_method_set("empty", ())
        with patch.object(self.client, "_fetch_paginated_identifiers", return_value=[]):
            self.assertEqual(
                self.client.fetch_entry_ids_for_method("NMR", "SOLUTION NMR"), []
            )

        with (
            patch.object(
                self.client,
                "_fetch_paginated_identifiers",
                return_value=["A", "B", "C"],
            ) as paginate,
            patch.object(
                self.client,
                "_filter_entry_ids_by_exact_methods",
                side_effect=lambda entry_ids, **_: entry_ids,
            ) as exact_filter,
        ):
            kept = self.client.fetch_entry_ids_for_method_set(
                "combined", ("METHOD A", "METHOD B"), True
            )
            self.assertEqual(kept, ["A", "B", "C"])
            self.assertEqual(
                paginate.call_args.kwargs["query"]["logical_operator"], "and"
            )
            self.assertEqual(exact_filter.call_count, 2)

    def test_exact_method_filter_handles_partial_entries_and_bad_protein_counts(
        self,
    ) -> None:
        """Skip malformed GraphQL rows and report invalid or omitted metadata."""
        self.assertEqual(
            self.client._filter_entry_ids_by_allowed_method_sets([], (("M",),)),
            [],
        )
        self.client._post_json = Mock(
            return_value={
                "data": {
                    "entries": [
                        None,
                        {},
                        {
                            "rcsb_id": "VALID",
                            "exptl": [None, {}, {"method": "SOLUTION NMR"}],
                            "rcsb_entry_info": {"polymer_entity_count_protein": 1},
                        },
                        {
                            "rcsb_id": "BAD_PROTEIN",
                            "exptl": [{"method": "SOLUTION NMR"}],
                            "rcsb_entry_info": {
                                "polymer_entity_count_protein": "invalid"
                            },
                        },
                    ]
                }
            }
        )
        with patch.object(builder, "_record_filtered_structure") as record:
            result = self.client._filter_entry_ids_by_exact_single_method(
                ["VALID", "BAD_PROTEIN", "MISSING"], "SOLUTION NMR"
            )
            self.assertEqual(result, ["VALID", "BAD_PROTEIN"])

            result = self.client._filter_entry_ids_by_allowed_method_sets(
                ["VALID", "BAD_PROTEIN", "MISSING"],
                (("SOLUTION NMR",),),
                require_protein_entities=True,
                record_exclusions=True,
            )
        self.assertEqual(result, ["VALID"])
        reasons = [call.args[1] for call in record.call_args_list]
        self.assertIn("protein polymer entity count is missing or invalid", reasons)
        self.assertIn("entry metadata missing from RCSB GraphQL response", reasons)

    def test_date_and_resolution_fetchers_tolerate_partial_nested_objects(self) -> None:
        entries = [
            {
                "rcsb_id": "VALID",
                "rcsb_accession_info": {
                    "deposit_date": "2020-01-02",
                    "initial_release_date": "2021-02-03",
                },
                "rcsb_entry_info": {"resolution_combined": [2.0, None, 1.5]},
            },
            {
                "rcsb_id": "BAD_DATE",
                "rcsb_accession_info": {"deposit_date": "bad"},
                "rcsb_entry_info": {"resolution_combined": ["bad"]},
            },
            {
                "rcsb_id": "NULL_NESTED",
                "rcsb_accession_info": None,
                "rcsb_entry_info": None,
            },
            None,
            {"rcsb_accession_info": {"deposit_date": "2019-01-01"}},
        ]
        self.client._post_json = Mock(return_value={"data": {"entries": entries}})
        self.assertEqual(
            self.client.fetch_deposit_dates_for_ids(
                ["VALID", "BAD_DATE", "NULL_NESTED", "MISSING"]
            ),
            ["2020-01-02", "bad", "2019-01-01"],
        )
        self.assertEqual(
            self.client.fetch_deposit_year_by_entry_id_for_ids(
                ["VALID", "BAD_DATE", "NULL_NESTED"]
            ),
            {"VALID": 2020},
        )
        self.assertEqual(self.client.fetch_deposit_year_by_entry_id_for_ids([]), {})
        self.assertEqual(
            self.client.fetch_deposit_date_by_entry_id_for_ids(
                ["VALID", "BAD_DATE", "NULL_NESTED"]
            ),
            {"VALID": "2020-01-02", "BAD_DATE": "bad"},
        )
        self.assertEqual(self.client.fetch_deposit_date_by_entry_id_for_ids([]), {})
        self.assertEqual(
            self.client.fetch_accession_dates_by_entry_id_for_ids(
                ["VALID", "NULL_NESTED"]
            ),
            {
                "VALID": ("2020-01-02", "2021-02-03"),
                "BAD_DATE": ("bad", None),
                "NULL_NESTED": (None, None),
            },
        )
        self.assertEqual(self.client.fetch_accession_dates_by_entry_id_for_ids([]), {})
        self.assertEqual(
            self.client.fetch_entry_resolution_for_ids(["VALID", "BAD_DATE"]),
            {"VALID": 1.5},
        )

    def test_entry_experimental_methods_parse_pure_hybrid_empty_and_null(self) -> None:
        """Preserve every reported method while tolerating absent experiment data."""
        self.client._post_json = Mock(
            return_value={
                "data": {
                    "entries": [
                        {
                            "rcsb_id": "PURE",
                            "exptl": [{"method": "X-RAY DIFFRACTION"}],
                        },
                        {
                            "rcsb_id": "HYBRID",
                            "exptl": [
                                {"method": "X-RAY DIFFRACTION"},
                                {"method": "NEUTRON DIFFRACTION"},
                            ],
                        },
                        {"rcsb_id": "EMPTY", "exptl": []},
                        {"rcsb_id": "NULL", "exptl": None},
                        None,
                    ]
                }
            }
        )

        methods = self.client.fetch_entry_experimental_methods_for_ids(
            ["PURE", "HYBRID", "EMPTY", "NULL"]
        )

        self.assertEqual(
            methods,
            {
                "PURE": ("X-RAY DIFFRACTION",),
                "HYBRID": ("X-RAY DIFFRACTION", "NEUTRON DIFFRACTION"),
                "EMPTY": (),
                "NULL": (),
            },
        )
        self.assertEqual(self.client.fetch_entry_experimental_methods_for_ids([]), {})

    def test_group_mapping_and_candidate_metadata(self) -> None:
        entities = [
            {
                "rcsb_id": "1ABC_1",
                "entity_poly": {"pdbx_strand_id": " A, B "},
                "rcsb_polymer_entity_container_identifiers": {"entry_id": "1ABC"},
                "rcsb_polymer_entity_group_membership": [
                    None,
                    {
                        "aggregation_method": "wrong",
                        "similarity_cutoff": 95,
                        "group_id": "wrong",
                    },
                    {
                        "aggregation_method": builder.SEQUENCE_IDENTITY_AGGREGATION_METHOD,
                        "similarity_cutoff": "95",
                        "group_id": "group-95",
                    },
                ],
            },
            {
                "rcsb_id": "2DEF_1",
                "entity_poly": {"pdbx_strand_id": "C"},
                "rcsb_polymer_entity_container_identifiers": {"entry_id": "2DEF"},
                "rcsb_polymer_entity_group_membership": [],
            },
            {"rcsb_id": "NO_CHAIN", "entity_poly": {}},
            None,
        ]
        self.client._post_json = Mock(
            return_value={"data": {"polymer_entities": entities}}
        )
        self.assertEqual(
            self.client.fetch_polymer_entity_group_mapping_for_ids([], 95), []
        )
        mappings = self.client.fetch_polymer_entity_group_mapping_for_ids(
            ["1ABC_1", "2DEF_1"], 95
        )
        self.assertEqual(
            mappings,
            [
                builder.XrayEntityGroupMappingRecord(
                    "1ABC_1", "1ABC", ("A", "B"), "group-95"
                )
            ],
        )

        with (
            patch.object(
                self.client,
                "fetch_entry_resolution_for_ids",
                return_value={"1ABC": 1.8},
            ) as resolutions,
            patch.object(
                self.client,
                "fetch_entry_experimental_methods_for_ids",
                return_value={
                    "1ABC": ("X-RAY DIFFRACTION",),
                    "2DEF": ("X-RAY DIFFRACTION", "NEUTRON DIFFRACTION"),
                },
            ) as methods,
        ):
            candidates = self.client.fetch_xray_polymer_entity_candidates_for_ids(
                ["1ABC_1", "2DEF_1"]
            )
        self.assertEqual(resolutions.call_count, 1)
        self.assertEqual(methods.call_count, 1)
        self.assertEqual(candidates[0].resolution_angstrom, 1.8)
        self.assertEqual(candidates[0].experimental_methods, ("X-RAY DIFFRACTION",))
        self.assertTrue(math.isnan(candidates[1].resolution_angstrom))
        self.assertEqual(
            candidates[1].experimental_methods,
            ("X-RAY DIFFRACTION", "NEUTRON DIFFRACTION"),
        )
        self.assertEqual(
            self.client.fetch_xray_polymer_entity_candidates_for_ids([]), []
        )

        groups = self.client.fetch_sequence_identity_group_ids_for_polymer_entity_ids(
            ["1ABC_1"], 95
        )
        self.assertEqual(groups, {"group-95"})
        self.assertEqual(
            self.client.fetch_sequence_identity_group_ids_for_polymer_entity_ids(
                [], 95
            ),
            set(),
        )

    def test_sequence_search_validates_retries_and_parses_result_shapes(self) -> None:
        """Handle sequence-search short circuits, throttling, and mixed results."""
        with self.assertRaises(ValueError):
            self.client.fetch_xray_polymer_entity_ids_by_sequence("A" * 10, 90)
        self.client.session = Mock()
        self.assertEqual(
            self.client.fetch_xray_polymer_entity_ids_by_sequence("  ", 95), []
        )
        self.assertEqual(
            self.client.fetch_xray_polymer_entity_ids_by_sequence("A" * 9, 100),
            [],
        )
        self.client.session.post.assert_not_called()

        throttled = Mock(status_code=429, text="slow down")
        successful = Mock(status_code=200, text="")
        successful.json.return_value = {
            "total_count": 2,
            "result_set": [
                "1ABC_1",
                {"identifier": "2DEF_2"},
                {"score": 0.5},
            ],
        }
        self.client.session.post.side_effect = [throttled, successful]
        with patch.object(builder.time, "sleep") as sleep:
            result = self.client.fetch_xray_polymer_entity_ids_by_sequence(
                " acdefghikL ", 95
            )
        self.assertEqual(result, ["1ABC_1", "2DEF_2"])
        sleep.assert_called_once_with(self.client.config.backoff_seconds)
        successful.raise_for_status.assert_called_once_with()

    def test_sequence_search_returns_raw_xray_condition_hits_in_order(self) -> None:
        """Leave exact-method filtering to the homolog candidate evaluator."""
        response = Mock(status_code=200, text="")
        response.json.return_value = {
            "total_count": 3,
            "result_set": [
                {"identifier": "2HYB_1"},
                {"identifier": "1PURE_2"},
                {"identifier": "1PURE_1"},
            ],
        }
        self.client.session = Mock()
        self.client.session.post.return_value = response

        with patch.object(
            self.client,
            "_filter_entry_ids_by_exact_single_method",
        ) as exact_method_filter:
            result = self.client.fetch_xray_polymer_entity_ids_by_sequence(
                "ACDEFGHIKL", 100
            )

        self.assertEqual(
            result,
            ["2HYB_1", "1PURE_2", "1PURE_1"],
        )
        exact_method_filter.assert_not_called()

    def test_sequence_search_handles_empty_http_outcomes_and_transport_failure(
        self,
    ) -> None:
        """Treat documented empty responses as no hits and expose exhausted retries."""
        for response in (
            Mock(status_code=204, text=""),
            Mock(status_code=400, text="query is below minimum length"),
        ):
            with self.subTest(status=response.status_code):
                self.client.session = Mock()
                self.client.session.post.return_value = response
                self.assertEqual(
                    self.client.fetch_xray_polymer_entity_ids_by_sequence(
                        "A" * 10, 100
                    ),
                    [],
                )
                response.raise_for_status.assert_not_called()

        failing = builder.RCSBClient(
            builder.DatasetBuildConfig(retries=2, backoff_seconds=0)
        )
        failing.session = Mock()
        failing.session.post.side_effect = requests.Timeout("offline")
        with (
            patch.object(builder.time, "sleep") as sleep,
            self.assertRaisesRegex(RuntimeError, "failed after 2 attempts"),
        ):
            failing.fetch_xray_polymer_entity_ids_by_sequence("A" * 10, 95)
        sleep.assert_called_once_with(0)

    def test_quality_and_seed_fetchers_skip_invalid_rows(self) -> None:
        quality_entries = [
            {
                "rcsb_id": "GOOD",
                "pdbx_vrpt_summary_geometry": [
                    {
                        "clashscore": "1.5",
                        "percent_ramachandran_outliers": 2,
                        "percent_rotamer_outliers": 3,
                    }
                ],
            },
            {"rcsb_id": "NO_QUALITY", "pdbx_vrpt_summary_geometry": []},
            {
                "rcsb_id": "MISSING_METRIC",
                "pdbx_vrpt_summary_geometry": [
                    {
                        "clashscore": 1,
                        "percent_ramachandran_outliers": None,
                        "percent_rotamer_outliers": 3,
                    }
                ],
            },
            {
                "rcsb_id": "BAD_METRIC",
                "pdbx_vrpt_summary_geometry": [
                    {
                        "clashscore": "bad",
                        "percent_ramachandran_outliers": 2,
                        "percent_rotamer_outliers": 3,
                    }
                ],
            },
            {"rcsb_id": "SKIP"},
            None,
        ]
        self.client._post_json = Mock(
            return_value={"data": {"entries": quality_entries}}
        )
        with patch.object(
            self.client,
            "_extract_solution_nmr_monomer_context",
            side_effect=lambda entry: (
                None
                if entry.get("rcsb_id") == "SKIP"
                else (entry["rcsb_id"], 2020, 2, {}, "A")
            ),
        ):
            records = self.client.fetch_solution_nmr_monomer_quality_records_for_ids(
                ["GOOD", "NO_QUALITY", "MISSING_METRIC", "BAD_METRIC", "SKIP"]
            )
        self.assertEqual(
            records,
            [builder.SolutionNMRMonomerQualityRecord("GOOD", 2020, 1.5, 2.0, 3.0)],
        )

        seed_entries = [{"rcsb_id": "GOOD"}, {"rcsb_id": "SKIP"}, None]
        self.client._post_json = Mock(return_value={"data": {"entries": seed_entries}})
        with patch.object(
            self.client,
            "_extract_solution_nmr_monomer_context",
            side_effect=lambda entry: (
                ("GOOD", 2020, 2, {}, "A") if entry.get("rcsb_id") == "GOOD" else None
            ),
        ):
            modeled = self.client.fetch_solution_nmr_monomer_modeled_first_model_seed_records_for_ids(
                ["GOOD", "SKIP"]
            )
            homolog = self.client.fetch_solution_nmr_monomer_xray_homolog_seed_records_for_ids(
                ["GOOD", "SKIP"]
            )
        self.assertEqual(
            modeled,
            [builder.SolutionNMRMonomerModeledFirstModelSeedRecord("GOOD", 2020, "A")],
        )
        self.assertEqual(
            homolog,
            [builder.SolutionNMRMonomerXrayHomologSeedRecord("GOOD", 2020, "A")],
        )

    def test_weight_fetch_tolerates_null_nested_metadata(self) -> None:
        """Treat explicitly null GraphQL subobjects as missing values."""
        self.client._post_json = Mock(
            return_value={
                "data": {
                    "entries": [
                        {
                            "rcsb_id": "NULL_METADATA",
                            "rcsb_accession_info": None,
                            "rcsb_entry_info": None,
                        }
                    ]
                }
            }
        )
        self.assertEqual(
            self.client.fetch_solution_nmr_weight_records_for_ids(["NULL_METADATA"]),
            [],
        )

    def test_stride_record_computation_and_iterator_paths(self) -> None:
        entry = self._monomer_entry("STRIDE")
        polymer_entity = entry["polymer_entities"][0]
        polymer_entity["polymer_entity_instances"] = [{"rcsb_id": "STRIDE.A"}]
        context = ("STRIDE", 2020, 2, polymer_entity, "A")
        coverages = {state: 0.0 for state in builder.STRIDE_STATE_CODES}
        coverages["C"] = 0.25
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch.object(
                self.client,
                "_extract_solution_nmr_monomer_context",
                return_value=context,
            ),
            patch.object(
                builder, "download_pdb_if_needed", return_value=Path(tmpdir) / "x.pdb"
            ),
            patch.object(builder, "load_cached_chain_id_map", return_value={"A": "Z"}),
            patch.object(
                builder,
                "parse_first_model_modeled_ca_auth_seq_ids",
                return_value={10, 11},
            ),
            patch.object(
                builder,
                "compute_stride_state_coverages_for_chain_modeled_first_model",
                return_value=(coverages, 2, 1),
            ),
        ):
            record = self.client._compute_solution_nmr_monomer_stride_modeled_first_model_for_entry(
                entry, "/stride", Path(tmpdir), Path(tmpdir) / "stride"
            )
        self.assertIsNotNone(record)
        self.assertEqual((record or Mock()).modeled_sequence_length, 2)
        self.assertEqual((record or Mock()).stride_secondary_structure_percent, 75.0)

        self.assertIsNone(
            self.client._compute_solution_nmr_monomer_stride_modeled_first_model_for_entry(
                None, "/stride", Path("cache"), Path("stride")
            )
        )
        invalid_entity = copy.deepcopy(entry)
        invalid_context = (
            "STRIDE",
            2020,
            2,
            invalid_entity["polymer_entities"][0],
            "A",
        )
        invalid_context[3]["polymer_entity_instances"] = []
        with patch.object(
            self.client,
            "_extract_solution_nmr_monomer_context",
            return_value=invalid_context,
        ):
            self.assertIsNone(
                self.client._compute_solution_nmr_monomer_stride_modeled_first_model_for_entry(
                    invalid_entity, "/stride", Path("cache"), Path("stride")
                )
            )

        self.client._post_json = Mock(
            return_value={
                "data": {"entries": [{"rcsb_id": "A"}, None, {"rcsb_id": "B"}]}
            }
        )
        with patch.object(
            self.client,
            "_compute_solution_nmr_monomer_stride_modeled_first_model_for_entry",
            side_effect=["first", None, "second"],
        ):
            yielded = list(
                self.client.iter_solution_nmr_monomer_stride_modeled_first_model_records_for_ids(
                    ["A", "B"], "/stride", Path("cache"), Path("stride")
                )
            )
        self.assertCountEqual(yielded, ["first", "second"])

    def test_model_length_check_contains_download_and_short_model_failures(
        self,
    ) -> None:
        """Treat coordinate failures and a single parsed model as ineligible."""
        with patch.object(
            self.client,
            "_download_solution_nmr_monomer_pdb_if_needed",
            side_effect=RuntimeError("offline"),
        ):
            self.assertFalse(
                self.client._solution_nmr_monomer_models_have_equal_lengths(
                    "BROKEN", "A"
                )
            )

        with (
            patch.object(
                self.client,
                "_download_solution_nmr_monomer_pdb_if_needed",
                return_value=Path("one.pdb"),
            ),
            patch.object(builder, "load_cached_chain_id_map", return_value={}),
            patch.object(
                builder,
                "parse_models_ca_coords_with_stats",
                return_value=([{1: Mock()}], [{1: 1}]),
            ),
        ):
            self.assertFalse(
                self.client._solution_nmr_monomer_models_have_equal_lengths(
                    "ONE_MODEL", "A"
                )
            )


if __name__ == "__main__":
    unittest.main()
