"""Tests for sequence-based X-ray homolog searches and resume handling."""

import tempfile
import unittest
import math
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
    """Represent a successful RCSB response with a null result set."""

    status_code = 200
    text = ""

    def json(self) -> dict:
        """Return a payload whose result set is explicitly null."""
        return {"total_count": 0, "result_set": None}

    def raise_for_status(self) -> None:
        """Accept the fake response as successful without returning a value."""
        return None


class _FakeSession:
    """Return null-result responses for sequence-search POST requests."""

    def post(self, *args, **kwargs) -> _NullResultSetResponse:
        """Return a deterministic null-result response for any request."""
        return _NullResultSetResponse()


class _NullGraphqlListClient(RCSBClient):
    """Return null GraphQL list fields for robustness tests."""

    def _post_json(self, url: str, payload: dict) -> dict:
        """Return a GraphQL payload with a null polymer-entity list."""
        return {"data": {"polymer_entities": None}}


class SequenceSearchTests(unittest.TestCase):
    """Verify homolog discovery, eligibility, batching, and checkpointing."""

    def test_reads_latest_valid_homolog_resume_checkpoint_status(self) -> None:
        """Use the latest valid status for each entry in a resume checkpoint."""
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
        """Skip completed entries and checkpoint newly successful searches."""
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
                """Build paired 95 and 100 percent records for ``seed``."""
                def record(identity: int) -> SolutionNMRMonomerXrayHomologRecord:
                    """Build one successful homolog record at ``identity``."""
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
        """Treat a null REST result set as an empty sequence-search result."""
        client = RCSBClient(DatasetBuildConfig(retries=1))
        client.session = _FakeSession()

        result = client.fetch_xray_polymer_entity_ids_by_sequence(
            sequence="ACDEFGHIKLMNPQRSTVWY",
            sequence_identity_percent=100,
        )

        self.assertEqual(result, [])

    def test_treats_null_candidate_entity_list_as_empty(self) -> None:
        """Treat a null GraphQL candidate list as an empty result."""
        client = _NullGraphqlListClient(DatasetBuildConfig())

        result = client.fetch_xray_polymer_entity_candidates_for_ids(["1ABC_1"])

        self.assertEqual(result, [])

    def test_candidates_include_entries_without_resolution(self) -> None:
        """Retain X-ray candidates even when resolution is unavailable."""
        client = RCSBClient(DatasetBuildConfig())
        entity_response = {
            "data": {
                "polymer_entities": [
                    {
                        "rcsb_id": "1MCD_1",
                        "entity_poly": {"pdbx_strand_id": "A,B"},
                        "rcsb_polymer_entity_container_identifiers": {
                            "entry_id": "1MCD"
                        },
                    }
                ]
            }
        }
        empty_resolution_response = {"data": {"entries": []}}
        client._post_json = Mock(
            side_effect=[entity_response, empty_resolution_response]
        )

        result = client.fetch_xray_polymer_entity_candidates_for_ids(
            ["1MCD_1"]
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].polymer_entity_id, "1MCD_1")
        self.assertTrue(math.isnan(result[0].resolution_angstrom))

    def test_excludes_nmr_entry_when_stride_core_contains_hetatm(self) -> None:
        """Exclude an NMR query whose STRIDE core includes a HETATM residue."""
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
        """Exclude an entry when no modeled amino-acid query can be built."""
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
        """Mark short cores ineligible instead of recording a negative search."""
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
        """Fetch large X-ray candidate sets in bounded GraphQL batches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = DatasetBuildConfig(graphql_batch_size=2)
            client = RCSBClient(config)

            def candidates_for_ids(
                entity_ids: list[str],
            ) -> list[XrayPolymerEntityCandidateRecord]:
                """Return one deterministic candidate for each requested ID."""
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

    def test_evaluates_large_candidate_chain_sets_one_chain_at_a_time(self) -> None:
        """Evaluate large candidate sets without materializing all chain files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = DatasetBuildConfig()
            builder = SolutionNMRMonomerXrayHomologBuilder(
                client=RCSBClient(config),
                config=config,
                stride_executable="stride",
                cache_dir=root,
                stride_cache_dir=root / "stride_cache",
            )
            candidate = XrayPolymerEntityCandidateRecord(
                polymer_entity_id="4WIZ_1",
                entry_id="4WIZ",
                chain_ids=tuple(f"chain_{index}" for index in range(70)),
                resolution_angstrom=3.0,
            )

            with (
                patch(
                    "src.pdb_dataset_builder.download_pdb_chain_subset_if_needed",
                    return_value=(root / "subset.pdb", {}),
                ) as download_subset,
                patch(
                    "src.pdb_dataset_builder.parse_first_model_ca_residues",
                    return_value=[],
                ),
                patch(
                    "src.pdb_dataset_builder.find_modeled_ca_core_identity_matches",
                    side_effect=[False, True],
                ),
            ):
                matched = builder._xray_candidate_has_modeled_core_match(
                    nmr_core_residues=[],
                    candidate=candidate,
                    sequence_identity_percent=100,
                )

            self.assertTrue(matched)
            self.assertEqual(download_subset.call_count, 2)
            self.assertEqual(
                download_subset.call_args_list[0].kwargs["chain_ids"],
                ("chain_0",),
            )
            self.assertEqual(
                download_subset.call_args_list[1].kwargs["chain_ids"],
                ("chain_1",),
            )


if __name__ == "__main__":
    unittest.main()
