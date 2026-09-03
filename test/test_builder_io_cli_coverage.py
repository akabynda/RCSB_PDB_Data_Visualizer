"""Additional coverage for builder serialization, filtering, and CLI wiring."""

from __future__ import annotations

import argparse
import csv
import sys
import tempfile
import unittest
from contextlib import ExitStack
from datetime import timezone
from pathlib import Path
from unittest.mock import Mock, patch

from src import pdb_dataset_builder as builder


class BuilderUtilityTests(unittest.TestCase):
    """Exercise deterministic helpers without contacting RCSB."""

    def tearDown(self) -> None:
        """Reset module-level report routing between tests."""
        builder._set_active_dataset_warning_logs(())
        builder._configure_dataset_warning_logs({})
        builder._configure_dataset_filtered_csvs({})

    def test_general_parsers_cover_valid_and_invalid_inputs(self) -> None:
        """Parse batching, dates, experiments, and dataset selections."""
        self.assertEqual(
            list(builder.chunked(["A", "B", "C", "D", "E"], 2)),
            [["A", "B"], ["C", "D"], ["E"]],
        )
        self.assertEqual(builder.extract_year("2024-09-30"), 2024)
        self.assertIsNone(builder.extract_year(None))
        self.assertIsNone(builder.extract_year("not-a-date"))

        parsed = builder.parse_rcsb_datetime("2024-09-30T12:15:00Z")
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.tzinfo, timezone.utc)
        self.assertIsNone(builder.parse_rcsb_datetime(""))
        self.assertIsNone(builder.parse_rcsb_datetime("invalid"))

        self.assertTrue(builder.contains_noesy_experiment(["HSQC", "3D noesy"]))
        self.assertFalse(builder.contains_noesy_experiment(["HSQC", "COSY"]))
        self.assertEqual(
            builder.parse_dataset_kinds("method_counts, solution_nmr_weights"),
            [
                builder.DatasetKind.METHOD_COUNTS,
                builder.DatasetKind.SOLUTION_NMR_WEIGHTS,
            ],
        )
        self.assertEqual(
            set(builder.parse_dataset_kinds(" ALL ")), set(builder.DatasetKind)
        )
        with self.assertRaises(argparse.ArgumentTypeError):
            builder.parse_dataset_kinds("unknown")
        with self.assertRaises(argparse.ArgumentTypeError):
            builder.parse_dataset_kinds(" , ")

        self.assertEqual(
            builder._normalize_refinement_program_name(" amber: molecular dynamics "),
            "AMBER",
        )
        self.assertEqual(
            builder._normalize_refinement_program_name("CNS VERSION"), "CNS"
        )
        self.assertEqual(
            builder._normalize_refinement_program_name("version 2"), "VERSION"
        )
        self.assertIsNone(builder._normalize_refinement_program_name("123"))

    def test_quality_and_assignment_csv_round_trips(self) -> None:
        """Preserve typed records and support the legacy assignment schema."""
        quality = builder.SolutionNMRMonomerQualityRecord(
            entry_id="1ABC",
            year=2001,
            clashscore=1.25,
            ramachandran_outliers_percent=2.5,
            sidechain_outliers_percent=3.75,
        )
        assignment = builder.SolutionNMRMonomerProgramClusterAssignmentRecord(
            entry_id="1ABC",
            year=2001,
            cluster_id="CLUSTER1",
            cluster_name="AMBER",
            cluster_score=0.5,
            has_program_text=True,
            program_text="AMBER; CNS",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            quality_path = root / "quality.csv"
            assignment_path = root / "assignments.csv"
            builder.write_solution_nmr_monomer_quality_csv([quality], quality_path)
            builder.write_solution_nmr_monomer_program_cluster_assignments_csv(
                [assignment], assignment_path
            )
            self.assertEqual(
                builder.read_solution_nmr_monomer_quality_csv(quality_path), [quality]
            )
            self.assertEqual(
                builder.read_solution_nmr_monomer_program_cluster_assignments_csv(
                    assignment_path
                ),
                [assignment],
            )
            self.assertEqual(
                builder.read_solution_nmr_monomer_quality_csv(root / "missing.csv"),
                [],
            )

            legacy_path = root / "legacy.csv"
            legacy_path.write_text(
                "entry_id,year,cluster_id,cluster_name,has_program_text,program_text\n"
                "2DEF,2002,CLUSTER9,OTHER,0,\n",
                encoding="utf-8",
            )
            legacy = builder.read_solution_nmr_monomer_program_cluster_assignments_csv(
                legacy_path
            )
            self.assertEqual(len(legacy), 1)
            self.assertEqual(legacy[0].cluster_score, 1.0)
            self.assertFalse(legacy[0].has_program_text)
            self.assertEqual(
                builder.read_solution_nmr_monomer_program_cluster_assignments_csv(
                    root / "absent.csv"
                ),
                [],
            )

    def test_program_cluster_summaries_are_unique_and_weighted(self) -> None:
        """Count structures once by year and weight per-cluster quality scores."""
        assignments = [
            builder.SolutionNMRMonomerProgramClusterAssignmentRecord(
                "entry-a", 2000, "CLUSTER1", "AMBER", 0.5, True, "AMBER/CNS"
            ),
            builder.SolutionNMRMonomerProgramClusterAssignmentRecord(
                "entry-a", 2000, "CLUSTER3", "CNS", 0.5, True, "AMBER/CNS"
            ),
            builder.SolutionNMRMonomerProgramClusterAssignmentRecord(
                "ENTRY-B", 2000, "CLUSTER1", "AMBER", 1.0, True, "AMBER"
            ),
            builder.SolutionNMRMonomerProgramClusterAssignmentRecord(
                "MISSING", 2001, "CLUSTER2", "ARIA", 1.0, True, "ARIA"
            ),
        ]
        qualities = [
            builder.SolutionNMRMonomerQualityRecord("ENTRY-A", 2000, 10.0, 2.0, 4.0),
            builder.SolutionNMRMonomerQualityRecord("entry-b", 2000, 20.0, 6.0, 8.0),
            builder.SolutionNMRMonomerQualityRecord("UNUSED", 2002, 99.0, 99.0, 99.0),
        ]

        yearly = builder.summarize_solution_nmr_monomer_program_cluster_quality_by_year(
            assignments, qualities
        )
        self.assertEqual(len(yearly), 1)
        self.assertEqual(yearly[0].structure_count, 2)
        self.assertEqual(yearly[0].avg_clashscore, 15.0)
        self.assertEqual(yearly[0].avg_ramachandran_outliers_percent, 4.0)
        self.assertEqual(yearly[0].avg_sidechain_outliers_percent, 6.0)

        totals = builder.summarize_solution_nmr_monomer_program_cluster_quality_total(
            assignments, qualities
        )
        self.assertEqual(len(totals), len(builder.PROGRAM_CLUSTER_DEFINITIONS))
        by_name = {record.cluster_name: record for record in totals}
        self.assertEqual(by_name["AMBER"].structure_count, 1.5)
        self.assertAlmostEqual(by_name["AMBER"].avg_clashscore or 0.0, 50.0 / 3.0)
        self.assertEqual(by_name["CNS"].structure_count, 0.5)
        self.assertEqual(by_name["CNS"].avg_clashscore, 10.0)
        self.assertEqual(by_name["OTHER"].structure_count, 0.0)
        self.assertIsNone(by_name["OTHER"].avg_clashscore)
        self.assertEqual(
            builder.summarize_solution_nmr_monomer_program_cluster_quality_by_year(
                [], qualities
            ),
            [],
        )
        self.assertEqual(
            builder.summarize_solution_nmr_monomer_program_cluster_quality_total(
                assignments, []
            ),
            [],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            yearly_path = root / "yearly.csv"
            total_path = root / "total.csv"
            builder.write_solution_nmr_monomer_program_cluster_yearly_summary_csv(
                yearly, yearly_path
            )
            builder.write_solution_nmr_monomer_program_cluster_total_csv(
                totals, total_path
            )
            with yearly_path.open(newline="", encoding="utf-8") as handle:
                self.assertEqual(
                    next(csv.DictReader(handle))["avg_clashscore"], "15.0000"
                )
            with total_path.open(newline="", encoding="utf-8") as handle:
                rows = {row["cluster_name"]: row for row in csv.DictReader(handle)}
            self.assertEqual(rows["AMBER"]["structure_count"], "1.5")
            self.assertEqual(rows["OTHER"]["avg_clashscore"], "")

    def test_precision_reader_supports_legacy_columns_and_skips_bad_rows(self) -> None:
        """Read old n_ca_core data while ignoring rows without any CA count."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "precision.csv"
            path.write_text(
                "entry_id,year,chain_id,core_start_seq_id,core_end_seq_id,n_models,n_ca_core,mean_rmsd_angstrom\n"
                "1ABC,2001,A,10,20,3,11,1.25\n"
                "SKIP,2002,A,1,2,2,,2.0\n",
                encoding="utf-8",
            )
            records = builder.read_solution_nmr_monomer_precision_csv(path)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].n_ca_core_used, 11)
            self.assertEqual(records[0].n_ca_core_raw, 11)
            self.assertEqual(
                builder.read_solution_nmr_monomer_precision_csv(
                    Path(tmpdir) / "missing.csv"
                ),
                [],
            )

    @staticmethod
    def _homolog_record(
        entry_id: str,
        entity_ids: tuple[str, ...] = (),
        *,
        query_length: int = 11,
        core_start: int | None = 1,
        core_end: int | None = 11,
    ) -> builder.SolutionNMRMonomerXrayHomologRecord:
        return builder.SolutionNMRMonomerXrayHomologRecord(
            entry_id=entry_id,
            year=2020,
            sequence_identity_percent=95,
            nmr_core_start_seq_id=core_start,
            nmr_core_end_seq_id=core_end,
            nmr_query_sequence_length=query_length,
            xray_homolog_entry_ids=tuple(item.split("_", 1)[0] for item in entity_ids),
            xray_homolog_entity_ids=entity_ids,
            has_xray_homolog=bool(entity_ids),
        )

    def test_homolog_csv_and_resume_checkpoint_round_trip(self) -> None:
        """Serialize optional core values and retain only valid checkpoint lines."""
        records = [
            self._homolog_record("EMPTY", core_start=None, core_end=None),
            self._homolog_record("HIT", ("1ABC_1", "2DEF_2")),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            path = root / "homologs.csv"
            builder.write_solution_nmr_monomer_xray_homolog_csv(records, path)
            self.assertEqual(
                builder.read_solution_nmr_monomer_xray_homolog_csv(path), records
            )
            self.assertEqual(
                builder.read_solution_nmr_monomer_xray_homolog_csv(root / "none.csv"),
                [],
            )

            checkpoint = builder._xray_homolog_resume_checkpoint_path(path)
            self.assertEqual(checkpoint, path.with_suffix(".resume.tsv"))
            checkpoint.write_text(
                "A\tsuccess\nB\tfailed\ninvalid\nA\tineligible\nC\tsuccess\textra\n",
                encoding="utf-8",
            )
            self.assertEqual(
                builder._read_xray_homolog_resume_checkpoint(checkpoint),
                {"A": "ineligible"},
            )
            self.assertEqual(
                builder._read_xray_homolog_resume_checkpoint(root / "absent.tsv"),
                {},
            )

    def test_historical_filter_handles_all_date_outcomes(self) -> None:
        """Reject malformed records and keep only homologs available in time."""
        empty = self._homolog_record("EMPTY")
        config = builder.DatasetBuildConfig(graphql_batch_size=2, max_workers=1)
        client = Mock()
        client.fetch_accession_dates_by_entry_id_for_ids.side_effect = lambda ids: {
            entry_id: dates
            for entry_id, dates in {
                "EMPTY": ("2020-01-01", None),
                "MISSING_NMR": (None, None),
                "MISSING_XRAY": ("2020-06-01", None),
                "MIXED": ("2020-06-01", None),
                "OLD": ("2010-01-01", "2019-01-01"),
                "NEW": ("2010-01-01", "2021-01-01"),
            }.items()
            if entry_id in ids
        }

        early = builder.filter_xray_homolog_records_by_deposit_date(
            [empty], client, config
        )
        self.assertEqual(early, [empty])

        filtered = builder.filter_xray_homolog_records_by_deposit_date(
            [
                self._homolog_record("SHORT", query_length=10),
                self._homolog_record("NO_CORE", core_start=None),
                empty,
                self._homolog_record("MISSING_NMR", ("OLD_1",)),
                self._homolog_record("MISSING_XRAY", ("UNKNOWN_1",)),
                self._homolog_record("MIXED", ("OLD_1", "NEW_1")),
            ],
            client,
            config,
        )
        self.assertEqual([record.entry_id for record in filtered], ["EMPTY", "MIXED"])
        mixed = filtered[1]
        self.assertEqual(mixed.xray_homolog_entity_ids, ("OLD_1",))
        self.assertEqual(mixed.xray_homolog_entry_ids, ("OLD",))
        self.assertTrue(mixed.has_xray_homolog)

    def test_stride_wrappers_handle_missing_state_and_parse_exceptions(self) -> None:
        """Return documented sentinels when cached STRIDE state is unavailable."""
        expected = {state: -1.0 for state in builder.STRIDE_STATE_CODES}
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch.object(
                builder, "download_pdb_if_needed", return_value=Path(tmpdir) / "x.pdb"
            ),
            patch.object(builder, "load_cached_chain_id_map", return_value={}),
            patch.object(
                builder,
                "load_first_model_stride_state_by_chain",
                return_value=(None, 3),
            ),
        ):
            self.assertEqual(
                builder.compute_stride_state_coverages_for_chain_modeled_first_model(
                    session=Mock(),
                    config=builder.DatasetBuildConfig(),
                    cache_dir=Path(tmpdir),
                    stride_cache_dir=Path(tmpdir) / "stride",
                    entry_id="1ABC",
                    chain_id="A",
                    modeled_sequence_length=1,
                    modeled_auth_seq_ids={1},
                    stride_executable="stride",
                ),
                (expected, 3, 0),
            )

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch.object(
                builder, "download_pdb_if_needed", return_value=Path(tmpdir) / "x.pdb"
            ),
            patch.object(builder, "load_cached_chain_id_map", return_value={}),
            patch.object(
                builder,
                "load_first_model_stride_state_by_chain",
                side_effect=RuntimeError("corrupt"),
            ),
        ):
            self.assertEqual(
                builder.compute_stride_state_coverages_for_chain_modeled_first_model(
                    Mock(),
                    builder.DatasetBuildConfig(),
                    Path(tmpdir),
                    Path(tmpdir) / "stride",
                    "1ABC",
                    "A",
                    1,
                    {1},
                    "stride",
                ),
                (expected, 0, 0),
            )

        with patch.object(
            builder,
            "load_first_model_stride_state_by_chain",
            side_effect=[(None, 1), ({"B": {1: "H"}, "C": {1: "E"}}, 1)],
        ):
            self.assertIsNone(
                builder.compute_stride_core_range_for_modeled_auth_seq_ids_in_first_model(
                    Path("x.pdb"), "1ABC", "A", {1}, "stride", Path("cache")
                )
            )
            self.assertIsNone(
                builder.compute_stride_core_range_for_modeled_auth_seq_ids_in_first_model(
                    Path("x.pdb"), "1ABC", "A", {1}, "stride", Path("cache")
                )
            )


class BuilderMainDispatchTests(unittest.TestCase):
    """Smoke-test every CLI dataset branch with deterministic collaborators."""

    def tearDown(self) -> None:
        builder._set_active_dataset_warning_logs(())
        builder._configure_dataset_warning_logs({})
        builder._configure_dataset_filtered_csvs({})

    def test_main_dispatches_all_dataset_builders_without_network(self) -> None:
        """Keep CLI argument and builder APIs wired together across all datasets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with patch.object(
                sys, "argv", ["pdb_dataset_builder.py", "--datasets", "all"]
            ):
                args = builder.parse_args()
            for name, value in vars(args).items():
                if isinstance(value, Path):
                    suffix = value.suffix or ""
                    setattr(args, name, root / f"{name}{suffix}")

            quality = builder.SolutionNMRMonomerQualityRecord(
                "QUALITY", 2020, 1.0, 2.0, 3.0
            )
            homolog = BuilderUtilityTests._homolog_record("NMR", ("XRAY_1",))
            class_names = [
                "PDBMethodYearlyBuilder",
                "MembraneProteinYearlyBuilder",
                "SolutionNMRProgramYearlyBuilder",
                "SolutionNMRWeightBuilder",
                "SolutionNMRMonomerExperimentsBuilder",
                "SolutionNMRMonomerStrideModeledFirstModelBuilder",
                "SolutionNMRMonomerPrecisionStrideModeledFirstModelBuilder",
                "SolutionNMRMonomerQualityBuilder",
                "SolutionNMRMonomerProgramClusterBuilder",
                "SolutionNMRMonomerXrayHomologBuilder",
            ]

            with ExitStack() as stack:
                stack.enter_context(
                    patch.object(builder, "parse_args", return_value=args)
                )
                stack.enter_context(patch.object(builder, "RCSBClient"))
                ensure_stride = stack.enter_context(
                    patch.object(
                        builder, "ensure_stride_executable", return_value="/bin/true"
                    )
                )
                stack.enter_context(
                    patch.object(builder, "_import_filtered_structures")
                )
                stack.enter_context(
                    patch.object(
                        builder,
                        "read_solution_nmr_monomer_quality_csv",
                        return_value=[quality],
                    )
                )
                stack.enter_context(
                    patch.object(
                        builder,
                        "read_solution_nmr_monomer_xray_homolog_csv",
                        return_value=[homolog],
                    )
                )
                stack.enter_context(
                    patch.object(
                        builder,
                        "filter_xray_homolog_records_by_deposit_date",
                        return_value=[homolog],
                    )
                )
                rmsd_build = stack.enter_context(
                    patch.object(
                        builder, "build_solution_nmr_monomer_xray_rmsd_outputs_to_csv"
                    )
                )
                class_mocks = {
                    name: stack.enter_context(patch.object(builder, name))
                    for name in class_names
                }
                for class_mock in class_mocks.values():
                    class_mock.return_value.build.return_value = []
                class_mocks[
                    "MembraneProteinYearlyBuilder"
                ].return_value.build_by_method.return_value = []
                class_mocks[
                    "SolutionNMRMonomerStrideModeledFirstModelBuilder"
                ].return_value.iter_records.return_value = []
                class_mocks[
                    "SolutionNMRMonomerProgramClusterBuilder"
                ].return_value.build.return_value = ([], [])
                class_mocks[
                    "SolutionNMRMonomerXrayHomologBuilder"
                ].return_value.build.return_value = ([], [])

                builder.main()

            for class_mock in class_mocks.values():
                class_mock.assert_called()
            self.assertEqual(ensure_stride.call_count, 3)
            ensure_stride.assert_called_with("", Path(args.stride_install_dir))
            rmsd_build.assert_called_once()


if __name__ == "__main__":
    unittest.main()
