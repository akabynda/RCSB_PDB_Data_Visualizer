import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.pdb_dataset_builder import (
    DatasetBuildConfig,
    NMRCoreContainsHetatmError,
    NMRHomologyQueryIneligibleError,
    RCSBClient,
    SolutionNMRMonomerXrayHomologRecord,
    SolutionNMRMonomerXrayHomologBuilder,
    SolutionNMRMonomerXrayHomologSeedRecord,
    XrayPolymerEntityCandidateRecord,
    _read_xray_homolog_resume_checkpoint,
)


class _NullResultSetResponse:
    status_code = 200
    text = ""

    def json(self) -> dict:
        return {"total_count": 0, "result_set": None}

    def raise_for_status(self) -> None:
        return None


class _FakeSession:
    def post(self, *args, **kwargs) -> _NullResultSetResponse:
        return _NullResultSetResponse()


class _NullGraphqlListClient(RCSBClient):
    def _post_json(self, url: str, payload: dict) -> dict:
        return {"data": {"polymer_entities": None}}


class SequenceSearchTests(unittest.TestCase):
    def test_reads_latest_valid_homolog_resume_checkpoint_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "homologs.resume.tsv"
            checkpoint_path.write_text(
                "1AAA\tsuccess\n"
                "2BBB\tineligible\n"
                "1AAA\tineligible\n"
                "3CCC\tfailed\n"
                "malformed\n",
                encoding="utf-8",
            )

            self.assertEqual(
                _read_xray_homolog_resume_checkpoint(checkpoint_path),
                {"1AAA": "ineligible", "2BBB": "ineligible"},
            )

    def test_homolog_build_skips_completed_entries_and_checkpoints_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = DatasetBuildConfig(graphql_batch_size=10, max_workers=1)
            client = RCSBClient(config)
            seeds = [
                SolutionNMRMonomerXrayHomologSeedRecord("1AAA", 2000, "A"),
                SolutionNMRMonomerXrayHomologSeedRecord("2BBB", 2001, "A"),
            ]
            client.fetch_solution_nmr_monomer_xray_homolog_seed_records_for_ids = (
                Mock(return_value=seeds)
            )
            builder = SolutionNMRMonomerXrayHomologBuilder(
                client=client,
                config=config,
                stride_executable="stride",
                cache_dir=root,
                stride_cache_dir=root / "stride_cache",
            )

            def record_pair(seed):
                def record(identity: int) -> SolutionNMRMonomerXrayHomologRecord:
                    return SolutionNMRMonomerXrayHomologRecord(
                        entry_id=seed.entry_id,
                        year=seed.year,
                        sequence_identity_percent=identity,
                        nmr_core_start_seq_id=1,
                        nmr_core_end_seq_id=20,
                        nmr_query_sequence_length=20,
                        xray_homolog_entry_ids=(),
                        xray_homolog_entity_ids=(),
                        has_xray_homolog=False,
                    )

                return record(95), record(100)

            completed: list[tuple[str, str]] = []
            with (
                patch(
                    "src.pdb_dataset_builder.fetch_solution_nmr_entry_ids",
                    return_value=["1AAA", "2BBB"],
                ),
                patch.object(builder, "_build_record_pair", side_effect=record_pair),
            ):
                records_95, records_100 = builder.build(
                    skip_entry_ids={"1AAA"},
                    on_entry_complete=lambda entry_id, status: completed.append(
                        (entry_id, status)
                    ),
                )

            self.assertEqual([record.entry_id for record in records_95], ["2BBB"])
            self.assertEqual([record.entry_id for record in records_100], ["2BBB"])
            self.assertEqual(completed, [("2BBB", "success")])

    def test_treats_null_result_set_as_empty_search_result(self) -> None:
        client = RCSBClient(DatasetBuildConfig(retries=1))
        client.session = _FakeSession()

        result = client.fetch_xray_polymer_entity_ids_by_sequence(
            sequence="ACDEFGHIKLMNPQRSTVWY",
            sequence_identity_percent=100,
        )

        self.assertEqual(result, [])

    def test_treats_null_candidate_entity_list_as_empty(self) -> None:
        client = _NullGraphqlListClient(DatasetBuildConfig())

        result = client.fetch_xray_polymer_entity_candidates_for_ids(["1ABC_1"])

        self.assertEqual(result, [])

    def test_excludes_nmr_entry_when_stride_core_contains_hetatm(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pdb_path = root / "1ABC.pdb"
            lines = ["MODEL        1\n"]
            for resid in range(1, 12):
                lines.append(
                    f"ATOM  {resid:5d}  CA  ALA A{resid:4d}    "
                    f"{0.0:8.3f}{0.0:8.3f}{0.0:8.3f}{1.0:6.2f}{20.0:6.2f}"
                    "           C\n"
                )
            lines.append(
                f"HETATM{12:5d}  CA  MSE A{12:4d}    "
                f"{0.0:8.3f}{0.0:8.3f}{0.0:8.3f}{1.0:6.2f}{20.0:6.2f}"
                "           C\n"
            )
            lines.append("ENDMDL\n")
            pdb_path.write_text("".join(lines), encoding="utf-8")

            config = DatasetBuildConfig()
            client = RCSBClient(config)
            builder = SolutionNMRMonomerXrayHomologBuilder(
                client=client,
                config=config,
                stride_executable="stride",
                cache_dir=root,
                stride_cache_dir=root / "stride_cache",
            )
            seed = SolutionNMRMonomerXrayHomologSeedRecord(
                entry_id="1ABC",
                year=2000,
                chain_id="A",
            )

            with (
                patch(
                    "src.pdb_dataset_builder.download_pdb_if_needed",
                    return_value=pdb_path,
                ),
                patch(
                    "src.pdb_dataset_builder.compute_stride_core_range_for_modeled_auth_seq_ids_in_first_model",
                    return_value=(1, 12),
                ),
            ):
                with self.assertRaises(NMRCoreContainsHetatmError):
                    builder._build_stride_core_query_sequence(seed)

    def test_excludes_nmr_entry_when_no_modeled_query_can_be_built(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pdb_path = root / "1ABC.pdb"
            pdb_path.write_text("MODEL        1\nENDMDL\n", encoding="utf-8")
            config = DatasetBuildConfig()
            builder = SolutionNMRMonomerXrayHomologBuilder(
                client=RCSBClient(config),
                config=config,
                stride_executable="stride",
                cache_dir=root,
                stride_cache_dir=root / "stride_cache",
            )
            seed = SolutionNMRMonomerXrayHomologSeedRecord("1ABC", 2000, "A")

            with patch(
                "src.pdb_dataset_builder.download_pdb_if_needed",
                return_value=pdb_path,
            ):
                with self.assertRaisesRegex(
                    NMRHomologyQueryIneligibleError,
                    "no usable first-model modeled CA residues",
                ):
                    builder._build_stride_core_query_sequence(seed)

    def test_excludes_short_core_instead_of_recording_no_homolog(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pdb_path = root / "1ABC.pdb"
            lines = ["MODEL        1\n"]
            for resid in range(1, 11):
                lines.append(
                    f"ATOM  {resid:5d}  CA  ALA A{resid:4d}    "
                    f"{0.0:8.3f}{0.0:8.3f}{0.0:8.3f}{1.0:6.2f}{20.0:6.2f}"
                    "           C\n"
                )
            lines.append("ENDMDL\n")
            pdb_path.write_text("".join(lines), encoding="utf-8")
            config = DatasetBuildConfig()
            builder = SolutionNMRMonomerXrayHomologBuilder(
                client=RCSBClient(config),
                config=config,
                stride_executable="stride",
                cache_dir=root,
                stride_cache_dir=root / "stride_cache",
            )
            seed = SolutionNMRMonomerXrayHomologSeedRecord("1ABC", 2000, "A")

            with (
                patch(
                    "src.pdb_dataset_builder.download_pdb_if_needed",
                    return_value=pdb_path,
                ),
                patch(
                    "src.pdb_dataset_builder.compute_stride_core_range_for_modeled_auth_seq_ids_in_first_model",
                    return_value=(1, 10),
                ),
            ):
                with self.assertRaisesRegex(
                    NMRHomologyQueryIneligibleError,
                    "STRIDE core is too short",
                ):
                    builder._build_stride_core_query_sequence(seed)

    def test_fetches_large_xray_candidate_sets_in_graphql_batches(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = DatasetBuildConfig(graphql_batch_size=2)
            client = RCSBClient(config)

            def candidates_for_ids(
                entity_ids: list[str],
            ) -> list[XrayPolymerEntityCandidateRecord]:
                return [
                    XrayPolymerEntityCandidateRecord(
                        polymer_entity_id=entity_id,
                        entry_id=entity_id.split("_", 1)[0],
                        chain_ids=("A",),
                        resolution_angstrom=2.0,
                    )
                    for entity_id in entity_ids
                ]

            client.fetch_xray_polymer_entity_candidates_for_ids = Mock(
                side_effect=candidates_for_ids
            )
            builder = SolutionNMRMonomerXrayHomologBuilder(
                client=client,
                config=config,
                stride_executable="stride",
                cache_dir=root,
                stride_cache_dir=root / "stride_cache",
            )
            entity_ids = ("1AAA_1", "2BBB_1", "3CCC_1", "4DDD_1", "5EEE_1")

            with patch.object(
                builder,
                "_xray_candidate_has_modeled_core_match",
                return_value=True,
            ):
                result = builder._filter_modeled_xray_homolog_entity_ids(
                    xray_entity_ids=entity_ids,
                    nmr_core_residues=[],
                    sequence_identity_percent=100,
                )

            self.assertEqual(result, entity_ids)
            self.assertEqual(
                client.fetch_xray_polymer_entity_candidates_for_ids.call_count,
                3,
            )


if __name__ == "__main__":
    unittest.main()
