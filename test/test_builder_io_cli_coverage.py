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

    @staticmethod
    def _homolog_cli_args(root: Path, *, resume: bool = False):
        """Build isolated CLI arguments for the X-ray homolog dataset."""
        arguments = [
            "pdb_dataset_builder.py",
            "--datasets",
            "solution_nmr_monomer_xray_homologs",
            "--solution-nmr-monomer-xray-homolog-95-output",
            str(root / "homologs_95.csv"),
            "--solution-nmr-monomer-xray-homolog-100-output",
            str(root / "homologs_100.csv"),
        ]
        if resume:
            arguments.append("--resume")
        with patch.object(sys, "argv", arguments):
            args = builder.parse_args()
        args.solution_nmr_monomer_cache_dir = root / "pdb-cache"
        args.stride_cache_dir = root / "stride-cache"
        args.stride_install_dir = root / "stride-install"
        return args

    @staticmethod
    def _rejected_homolog(
        nmr_entry_id: str,
        sequence_identity_percent: int,
        xray_entity_id: str,
    ) -> builder.RejectedXrayHomologRecord:
        """Create one compact rejected-candidate fixture."""
        return builder.RejectedXrayHomologRecord(
            nmr_entry_id=nmr_entry_id,
            nmr_year=2020,
            nmr_chain_id="N",
            sequence_identity_percent=sequence_identity_percent,
            nmr_core_start_seq_id=1,
            nmr_core_end_seq_id=11,
            nmr_query_sequence_length=11,
            xray_entry_id=xray_entity_id.split("_", 1)[0],
            xray_entity_id=xray_entity_id,
            xray_chain_ids=("A", "B"),
            reason=builder.XRAY_HOMOLOG_REJECTION_REASON,
        )

    @staticmethod
    def _write_homolog_checkpoint(output_95: Path, entry_id: str, status: str) -> None:
        """Write one completion marker beside a homolog output pair."""
        builder._xray_homolog_resume_checkpoint_path(output_95).write_text(
            f"{entry_id}\t{status}\n",
            encoding="utf-8",
        )

    @staticmethod
    def _homolog_with_rejections(
        entry_id: str,
        sequence_identity_percent: int,
        rejected: tuple[builder.RejectedXrayHomologRecord, ...] = (),
    ) -> builder.SolutionNMRMonomerXrayHomologRecord:
        """Create one completed homolog row carrying callback-only rejections."""
        return builder.SolutionNMRMonomerXrayHomologRecord(
            entry_id=entry_id,
            year=2020,
            sequence_identity_percent=sequence_identity_percent,
            nmr_core_start_seq_id=1,
            nmr_core_end_seq_id=11,
            nmr_query_sequence_length=11,
            xray_homolog_entry_ids=("1KEEP",),
            xray_homolog_entity_ids=("1KEEP_1",),
            has_xray_homolog=True,
            rejected_xray_homologs=rejected,
        )

    def test_main_writes_rejected_homolog_callbacks_without_changing_main_headers(
        self,
    ) -> None:
        """Persist separate 95/100 rejection reports from completed record pairs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            args = self._homolog_cli_args(root)
            rejected_95 = self._rejected_homolog("NMR1", 95, "2DROP_1")
            rejected_100 = self._rejected_homolog("NMR1", 100, "3DROP_2")
            record_95 = self._homolog_with_rejections("NMR1", 95, (rejected_95,))
            record_100 = self._homolog_with_rejections("NMR1", 100, (rejected_100,))

            def build_homologs(*, on_record_pair, on_entry_complete, **_kwargs):
                on_record_pair(record_95, record_100)
                on_entry_complete("NMR1", "success")
                return [record_95], [record_100]

            with (
                patch.object(builder, "parse_args", return_value=args),
                patch.object(builder, "RCSBClient"),
                patch.object(
                    builder, "ensure_stride_executable", return_value="/bin/true"
                ),
                patch.object(builder, "_configure_dataset_warning_logs"),
                patch.object(
                    builder, "SolutionNMRMonomerXrayHomologBuilder"
                ) as builder_class,
            ):
                builder_class.return_value.build.side_effect = build_homologs
                builder.main()

            output_95 = Path(args.solution_nmr_monomer_xray_homolog_95_output)
            output_100 = Path(args.solution_nmr_monomer_xray_homolog_100_output)
            for output_path in (output_95, output_100):
                with output_path.open(newline="", encoding="utf-8") as csvfile:
                    reader = csv.reader(csvfile)
                    self.assertEqual(
                        next(reader), builder.SOLUTION_NMR_MONOMER_XRAY_HOMOLOG_HEADER
                    )
                    self.assertEqual(len(list(reader)), 1)

            self.assertEqual(
                builder.read_rejected_xray_homolog_csv(
                    builder.rejected_xray_homologs_csv_path(output_95)
                ),
                [rejected_95],
            )
            self.assertEqual(
                builder.read_rejected_xray_homolog_csv(
                    builder.rejected_xray_homologs_csv_path(output_100)
                ),
                [rejected_100],
            )
            self.assertEqual(
                builder._read_xray_homolog_resume_checkpoint(
                    builder._xray_homolog_resume_checkpoint_path(output_95)
                )["NMR1"],
                "success_with_rejected_audit",
            )

    def test_main_resume_keeps_only_paired_rejections_and_deduplicates(
        self,
    ) -> None:
        """Retain audited completed pairs and rewrite stale or duplicate rows."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            args = self._homolog_cli_args(root, resume=True)
            output_95 = Path(args.solution_nmr_monomer_xray_homolog_95_output)
            output_100 = Path(args.solution_nmr_monomer_xray_homolog_100_output)
            paired_95 = self._homolog_with_rejections("PAIRED", 95)
            paired_100 = self._homolog_with_rejections("PAIRED", 100)
            unpaired_95 = self._homolog_with_rejections("UNPAIRED", 95)
            builder.write_solution_nmr_monomer_xray_homolog_csv(
                [paired_95, unpaired_95], output_95
            )
            builder.write_solution_nmr_monomer_xray_homolog_csv(
                [paired_100], output_100
            )
            self._write_homolog_checkpoint(
                output_95, "PAIRED", "success_with_rejected_audit"
            )

            paired_rejected_95 = self._rejected_homolog("PAIRED", 95, "2OLD_1")
            unpaired_rejected_95 = self._rejected_homolog("UNPAIRED", 95, "3STALE_1")
            paired_rejected_100 = self._rejected_homolog("PAIRED", 100, "4OLD_1")
            builder.write_rejected_xray_homolog_csv(
                [paired_rejected_95, paired_rejected_95, unpaired_rejected_95],
                builder.rejected_xray_homologs_csv_path(output_95),
            )
            builder.write_rejected_xray_homolog_csv(
                [paired_rejected_100, paired_rejected_100],
                builder.rejected_xray_homologs_csv_path(output_100),
            )

            new_rejected_95 = self._rejected_homolog("NEW", 95, "5NEW_1")
            new_rejected_100 = self._rejected_homolog("NEW", 100, "6NEW_1")
            new_95 = self._homolog_with_rejections(
                "NEW", 95, (new_rejected_95, new_rejected_95)
            )
            new_100 = self._homolog_with_rejections(
                "NEW", 100, (new_rejected_100, new_rejected_100)
            )

            def resume_homologs(
                *, skip_entry_ids, on_record_pair, on_entry_complete, **_kwargs
            ):
                self.assertEqual(skip_entry_ids, {"PAIRED"})
                on_record_pair(new_95, new_100)
                on_entry_complete("NEW", "success")
                return [new_95], [new_100]

            with (
                patch.object(builder, "parse_args", return_value=args),
                patch.object(builder, "RCSBClient"),
                patch.object(
                    builder, "ensure_stride_executable", return_value="/bin/true"
                ),
                patch.object(builder, "_configure_dataset_warning_logs"),
                patch.object(
                    builder, "SolutionNMRMonomerXrayHomologBuilder"
                ) as builder_class,
            ):
                builder_class.return_value.build.side_effect = resume_homologs
                builder.main()

            self.assertEqual(
                builder.read_rejected_xray_homolog_csv(
                    builder.rejected_xray_homologs_csv_path(output_95)
                ),
                [paired_rejected_95, new_rejected_95],
            )
            self.assertEqual(
                builder.read_rejected_xray_homolog_csv(
                    builder.rejected_xray_homologs_csv_path(output_100)
                ),
                [paired_rejected_100, new_rejected_100],
            )
            self.assertEqual(
                {
                    record.entry_id
                    for record in builder.read_solution_nmr_monomer_xray_homolog_csv(
                        output_95
                    )
                },
                {"PAIRED", "NEW"},
            )
            with output_95.open(newline="", encoding="utf-8") as csvfile:
                self.assertEqual(
                    next(csv.reader(csvfile)),
                    builder.SOLUTION_NMR_MONOMER_XRAY_HOMOLOG_HEADER,
                )

    def test_main_resume_recomputes_paired_rows_when_rejected_reports_are_missing(
        self,
    ) -> None:
        """Do not trust legacy paired rows that have no rejection audit sidecars."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            args = self._homolog_cli_args(root, resume=True)
            output_95 = Path(args.solution_nmr_monomer_xray_homolog_95_output)
            output_100 = Path(args.solution_nmr_monomer_xray_homolog_100_output)
            builder.write_solution_nmr_monomer_xray_homolog_csv(
                [self._homolog_with_rejections("PAIRED", 95)], output_95
            )
            builder.write_solution_nmr_monomer_xray_homolog_csv(
                [self._homolog_with_rejections("PAIRED", 100)], output_100
            )
            self._write_homolog_checkpoint(
                output_95, "PAIRED", "success_with_rejected_audit"
            )
            rejected_95_path = builder.rejected_xray_homologs_csv_path(output_95)
            rejected_100_path = builder.rejected_xray_homologs_csv_path(output_100)
            self.assertFalse(rejected_95_path.exists())
            self.assertFalse(rejected_100_path.exists())

            audit_95 = self._rejected_homolog("PAIRED", 95, "2AUDIT_1")
            audit_100 = self._rejected_homolog("PAIRED", 100, "3AUDIT_1")
            rebuilt_95 = self._homolog_with_rejections("PAIRED", 95, (audit_95,))
            rebuilt_100 = self._homolog_with_rejections("PAIRED", 100, (audit_100,))

            def resume_homologs(
                *, skip_entry_ids, on_record_pair, on_entry_complete, **_kwargs
            ):
                self.assertEqual(skip_entry_ids, set())
                on_record_pair(rebuilt_95, rebuilt_100)
                on_entry_complete("PAIRED", "success")
                return [rebuilt_95], [rebuilt_100]

            with (
                patch.object(builder, "parse_args", return_value=args),
                patch.object(builder, "RCSBClient"),
                patch.object(
                    builder, "ensure_stride_executable", return_value="/bin/true"
                ),
                patch.object(builder, "_configure_dataset_warning_logs"),
                patch.object(
                    builder, "SolutionNMRMonomerXrayHomologBuilder"
                ) as builder_class,
            ):
                builder_class.return_value.build.side_effect = resume_homologs
                builder.main()

            self.assertEqual(
                builder.read_rejected_xray_homolog_csv(rejected_95_path),
                [audit_95],
            )
            self.assertEqual(
                builder.read_rejected_xray_homolog_csv(rejected_100_path),
                [audit_100],
            )
            self.assertEqual(
                [
                    record.entry_id
                    for record in builder.read_solution_nmr_monomer_xray_homolog_csv(
                        output_95
                    )
                ],
                ["PAIRED"],
            )

    def test_malformed_rejected_tail_is_read_partially_and_forces_resume_rebuild(
        self,
    ) -> None:
        """Return valid audit rows publicly but distrust their damaged report."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            args = self._homolog_cli_args(root, resume=True)
            output_95 = Path(args.solution_nmr_monomer_xray_homolog_95_output)
            output_100 = Path(args.solution_nmr_monomer_xray_homolog_100_output)
            builder.write_solution_nmr_monomer_xray_homolog_csv(
                [self._homolog_with_rejections("PAIRED", 95)], output_95
            )
            builder.write_solution_nmr_monomer_xray_homolog_csv(
                [self._homolog_with_rejections("PAIRED", 100)], output_100
            )
            self._write_homolog_checkpoint(
                output_95, "PAIRED", "success_with_rejected_audit"
            )

            rejected_95_path = builder.rejected_xray_homologs_csv_path(output_95)
            rejected_100_path = builder.rejected_xray_homologs_csv_path(output_100)
            valid_95 = self._rejected_homolog("PAIRED", 95, "2VALID_1")
            valid_100 = self._rejected_homolog("PAIRED", 100, "3VALID_1")
            builder.write_rejected_xray_homolog_csv([valid_95], rejected_95_path)
            builder.write_rejected_xray_homolog_csv([valid_100], rejected_100_path)
            with rejected_95_path.open("a", encoding="utf-8") as csvfile:
                csvfile.write("BROKEN,not-a-year\n")

            self.assertEqual(
                builder.read_rejected_xray_homolog_csv(rejected_95_path),
                [valid_95],
            )

            audit_95 = self._rejected_homolog("PAIRED", 95, "4REBUILT_1")
            audit_100 = self._rejected_homolog("PAIRED", 100, "5REBUILT_1")
            rebuilt_95 = self._homolog_with_rejections("PAIRED", 95, (audit_95,))
            rebuilt_100 = self._homolog_with_rejections("PAIRED", 100, (audit_100,))

            def resume_homologs(
                *, skip_entry_ids, on_record_pair, on_entry_complete, **_kwargs
            ):
                self.assertEqual(skip_entry_ids, set())
                on_record_pair(rebuilt_95, rebuilt_100)
                on_entry_complete("PAIRED", "success")
                return [rebuilt_95], [rebuilt_100]

            with (
                patch.object(builder, "parse_args", return_value=args),
                patch.object(builder, "RCSBClient"),
                patch.object(
                    builder, "ensure_stride_executable", return_value="/bin/true"
                ),
                patch.object(builder, "_configure_dataset_warning_logs"),
                patch.object(
                    builder, "SolutionNMRMonomerXrayHomologBuilder"
                ) as builder_class,
            ):
                builder_class.return_value.build.side_effect = resume_homologs
                builder.main()

            self.assertEqual(
                builder.read_rejected_xray_homolog_csv(rejected_95_path),
                [audit_95],
            )
            self.assertEqual(
                builder.read_rejected_xray_homolog_csv(rejected_100_path),
                [audit_100],
            )

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
