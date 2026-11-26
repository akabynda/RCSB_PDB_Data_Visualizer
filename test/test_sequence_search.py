"""Tests for sequence-based X-ray homolog searches and resume handling."""

import json
import tempfile
import unittest
import math
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch

import requests

from src.pdb_dataset_builder import (
    CAResidueRecord,
    DatasetBuildConfig,
    NMRCoreContainsHetatmError,
    NMRHomologyQueryIneligibleError,
    RCSBClient,
    RejectedXrayHomologRecord,
    ResidueId,
    SolutionNMRMonomerXrayHomologRecord,
    SolutionNMRMonomerXrayHomologBuilder,
    SolutionNMRMonomerXrayHomologSeedRecord,
    XrayHomologEvaluationError,
    XrayPolymerEntityCandidateRecord,
    XRAY_HOMOLOG_METHOD_REJECTION_REASON,
    _read_xray_homolog_resume_checkpoint,
    find_modeled_ca_core_identity_matches,
    read_rejected_xray_homolog_csv,
    rejected_xray_homologs_csv_path,
    write_rejected_xray_homolog_csv,
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


def _http_response(status_code: int, payload: object = None) -> requests.Response:
    """Return a real requests response with an optional JSON body."""
    response = requests.Response()
    response.status_code = status_code
    response.url = "https://example.test"
    response._content = json.dumps(payload).encode("utf-8")
    return response


class _HttpFailureSession:
    """Answer RCSB requests and fail one request kind with HTTP status codes.

    ``failures`` holds the status codes returned, in order, for the failing
    request kind; once it is empty that kind succeeds like the others.
    """

    def __init__(self, failing_kind: str, failures: list[int]) -> None:
        self.failing_kind = failing_kind
        self.failures = list(failures)

    def _fail(self, kind: str) -> requests.Response | None:
        """Return the next failure for the failing kind, else ``None``.

        A failed ``requests.Response`` is falsy, so callers compare with None.
        """
        if kind == self.failing_kind and self.failures:
            return _http_response(self.failures.pop(0))
        return None

    def post(self, url: str, **kwargs) -> requests.Response:
        """Return search hits or GraphQL candidate metadata."""
        query = kwargs["json"]["query"]
        if not isinstance(query, str):
            failed = self._fail("sequence_search")
            if failed is not None:
                return failed
            return _http_response(200, {"total_count": 1, "result_set": ["9XYZ_1"]})
        failed = self._fail("graphql")
        if failed is not None:
            return failed
        if "polymer_entities" in query:
            return _http_response(
                200,
                {
                    "data": {
                        "polymer_entities": [
                            {
                                "rcsb_id": "9XYZ_1",
                                "entity_poly": {"pdbx_strand_id": "A"},
                                "rcsb_polymer_entity_container_identifiers": {
                                    "entry_id": "9XYZ"
                                },
                            }
                        ]
                    }
                },
            )
        if "exptl" in query:
            return _http_response(
                200,
                {
                    "data": {
                        "entries": [
                            {
                                "rcsb_id": "9XYZ",
                                "exptl": [{"method": "X-RAY DIFFRACTION"}],
                            }
                        ]
                    }
                },
            )
        return _http_response(200, {"data": {"entries": []}})

    def get(self, url: str, **kwargs) -> requests.Response:
        """Fail coordinate downloads; only the failing kind counts as a 5xx."""
        kind = "nmr_download" if "1aaa" in url.lower() else "xray_download"
        failed = self._fail(kind)
        return failed if failed is not None else _http_response(404)


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
                "4DDD\tsuccess_with_rejected_audit\n"
                "5EEE\tsuccess_with_rejected_audit\n"
                "5EEE\tpending_rejected_audit\n"
                "malformed\n",
                encoding="utf-8",
            )

            self.assertEqual(
                _read_xray_homolog_resume_checkpoint(checkpoint_path),
                {
                    "1AAA": "ineligible",
                    "2BBB": "ineligible",
                    "4DDD": "success_with_rejected_audit",
                    "5EEE": "pending_rejected_audit",
                },
            )

    def test_homolog_build_skips_completed_entries_and_checkpoints_success(
        self,
    ) -> None:
        """Skip completed entries and checkpoint newly successful searches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = DatasetBuildConfig(graphql_batch_size=10, max_workers=1)
            client = RCSBClient(config)
            seeds = [
                SolutionNMRMonomerXrayHomologSeedRecord("1AAA", 2000, "A"),
                SolutionNMRMonomerXrayHomologSeedRecord("2BBB", 2001, "A"),
            ]
            client.fetch_solution_nmr_monomer_xray_homolog_seed_records_for_ids = Mock(
                return_value=seeds
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
                    "src.dataset.builders.homologs.fetch_solution_nmr_entry_ids",
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

    def test_homolog_build_streams_rejections_before_releasing_returned_details(
        self,
    ) -> None:
        """Expose rejects to the callback without retaining them in result lists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = DatasetBuildConfig(graphql_batch_size=10, max_workers=1)
            client = RCSBClient(config)
            seed = SolutionNMRMonomerXrayHomologSeedRecord("1NMR", 2000, "N")
            client.fetch_solution_nmr_monomer_xray_homolog_seed_records_for_ids = Mock(
                return_value=[seed]
            )
            builder = SolutionNMRMonomerXrayHomologBuilder(
                client=client,
                config=config,
                stride_executable="stride",
                cache_dir=root,
                stride_cache_dir=root / "stride_cache",
            )

            def record(identity: int) -> SolutionNMRMonomerXrayHomologRecord:
                rejected = RejectedXrayHomologRecord(
                    nmr_entry_id=seed.entry_id,
                    nmr_year=seed.year,
                    nmr_chain_id=seed.chain_id,
                    sequence_identity_percent=identity,
                    nmr_core_start_seq_id=1,
                    nmr_core_end_seq_id=11,
                    nmr_query_sequence_length=11,
                    xray_entry_id="2DROP",
                    xray_entity_id="2DROP_1",
                    xray_chain_ids=("A",),
                    reason="no eligible modeled core match",
                )
                return SolutionNMRMonomerXrayHomologRecord(
                    entry_id=seed.entry_id,
                    year=seed.year,
                    sequence_identity_percent=identity,
                    nmr_core_start_seq_id=1,
                    nmr_core_end_seq_id=11,
                    nmr_query_sequence_length=11,
                    xray_homolog_entry_ids=(),
                    xray_homolog_entity_ids=(),
                    has_xray_homolog=False,
                    rejected_xray_homologs=(rejected,),
                )

            built_pair = (record(95), record(100))
            callback_pairs: list[
                tuple[
                    SolutionNMRMonomerXrayHomologRecord,
                    SolutionNMRMonomerXrayHomologRecord,
                ]
            ] = []
            with (
                patch(
                    "src.dataset.builders.homologs.fetch_solution_nmr_entry_ids",
                    return_value=[seed.entry_id],
                ),
                patch.object(builder, "_build_record_pair", return_value=built_pair),
            ):
                records_95, records_100 = builder.build(
                    on_record_pair=lambda current, historical: callback_pairs.append(
                        (current, historical)
                    ),
                    retain_rejected_xray_homologs=False,
                )

            self.assertEqual(callback_pairs, [built_pair])
            self.assertTrue(callback_pairs[0][0].rejected_xray_homologs)
            self.assertTrue(callback_pairs[0][1].rejected_xray_homologs)
            self.assertEqual(records_95[0].rejected_xray_homologs, ())
            self.assertEqual(records_100[0].rejected_xray_homologs, ())

    def test_homolog_build_requeues_http_500_failures_up_to_three_attempts(
        self,
    ) -> None:
        """Put 5xx failures at the queue end and accept a later success."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = DatasetBuildConfig(graphql_batch_size=10, max_workers=1)
            client = RCSBClient(config)
            seeds = [
                SolutionNMRMonomerXrayHomologSeedRecord("1AAA", 2000, "A"),
                SolutionNMRMonomerXrayHomologSeedRecord("2BBB", 2001, "A"),
            ]
            client.fetch_solution_nmr_monomer_xray_homolog_seed_records_for_ids = Mock(
                return_value=seeds
            )
            builder = SolutionNMRMonomerXrayHomologBuilder(
                client=client,
                config=config,
                stride_executable="stride",
                cache_dir=root,
                stride_cache_dir=root / "stride_cache",
            )
            calls: list[str] = []
            attempts: dict[str, int] = {"1AAA": 0, "2BBB": 0}

            def record_pair(seed):
                """Fail 1AAA twice with HTTP 500, then build both records."""
                calls.append(seed.entry_id)
                attempts[seed.entry_id] += 1
                if seed.entry_id == "1AAA" and attempts[seed.entry_id] < 3:
                    response = requests.Response()
                    response.status_code = 500
                    raise requests.HTTPError("500 Server Error", response=response)

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

            with (
                patch(
                    "src.dataset.builders.homologs.fetch_solution_nmr_entry_ids",
                    return_value=["1AAA", "2BBB"],
                ),
                patch.object(builder, "_build_record_pair", side_effect=record_pair),
            ):
                records_95, records_100 = builder.build()

            self.assertEqual(calls, ["1AAA", "2BBB", "1AAA", "1AAA"])
            self.assertEqual(
                [record.entry_id for record in records_95], ["1AAA", "2BBB"]
            )
            self.assertEqual(
                [record.entry_id for record in records_100], ["1AAA", "2BBB"]
            )

    def test_homolog_build_excludes_http_500_failure_after_three_attempts(
        self,
    ) -> None:
        """Exclude a seed whose sequence search returns 5xx three times."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = DatasetBuildConfig(graphql_batch_size=10, max_workers=1)
            client = RCSBClient(config)
            seed = SolutionNMRMonomerXrayHomologSeedRecord("1AAA", 2000, "A")
            client.fetch_solution_nmr_monomer_xray_homolog_seed_records_for_ids = Mock(
                return_value=[seed]
            )
            builder = SolutionNMRMonomerXrayHomologBuilder(
                client=client,
                config=config,
                stride_executable="stride",
                cache_dir=root,
                stride_cache_dir=root / "stride_cache",
            )
            server_error = requests.HTTPError(
                "500 Server Error",
                response=Mock(status_code=500),
            )

            with (
                patch(
                    "src.dataset.builders.homologs.fetch_solution_nmr_entry_ids",
                    return_value=["1AAA"],
                ),
                patch.object(
                    builder,
                    "_build_record_pair",
                    side_effect=[server_error, server_error, server_error],
                ) as build_record_pair,
            ):
                records_95, records_100 = builder.build()

            self.assertEqual(build_record_pair.call_count, 3)
            self.assertEqual(records_95, [])
            self.assertEqual(records_100, [])

    def _build_with_http_failures(
        self, failing_kind: str, failures: list[int]
    ) -> tuple[int, list, Mock]:
        """Run the real record builder with one failing request kind.

        STRIDE, coordinate parsing and core matching are replaced unless the
        failing request kind needs the real download path for that step.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = DatasetBuildConfig(
                graphql_batch_size=10,
                max_workers=1,
                retries=2,
                backoff_seconds=0.0,
            )
            client = RCSBClient(config)
            client.session = _HttpFailureSession(failing_kind, failures)
            seed = SolutionNMRMonomerXrayHomologSeedRecord("1AAA", 2000, "A")
            client.fetch_solution_nmr_monomer_xray_homolog_seed_records_for_ids = Mock(
                return_value=[seed]
            )
            builder = SolutionNMRMonomerXrayHomologBuilder(
                client=client,
                config=config,
                stride_executable="stride",
                cache_dir=root,
                stride_cache_dir=root / "stride_cache",
            )
            build_record_pair = builder._build_record_pair
            entry_attempts = 0

            def counted_record_pair(record_seed):
                nonlocal entry_attempts
                entry_attempts += 1
                return build_record_pair(record_seed)

            module = "src.dataset.builders.homologs"
            with ExitStack() as stack:
                stack.enter_context(
                    patch(
                        f"{module}.fetch_solution_nmr_entry_ids", return_value=["1AAA"]
                    )
                )
                record_filtered = stack.enter_context(
                    patch(f"{module}._record_filtered_structure")
                )
                stack.enter_context(
                    patch(
                        f"{module}.find_modeled_ca_core_identity_matches",
                        return_value=[object()],
                    )
                )
                stack.enter_context(
                    patch.object(
                        builder, "_build_record_pair", side_effect=counted_record_pair
                    )
                )
                if failing_kind != "nmr_download":
                    stack.enter_context(
                        patch.object(
                            builder,
                            "_build_stride_core_query_sequence",
                            return_value=(
                                "ACDEFGHIKLM",
                                ResidueId(1, ""),
                                ResidueId(11, ""),
                                [CAResidueRecord(1, "A", True)],
                            ),
                        )
                    )
                if failing_kind != "xray_download":
                    stack.enter_context(
                        patch(
                            f"{module}.download_pdb_chain_subset_if_needed",
                            return_value=(root / "9XYZ.pdb", {}),
                        )
                    )
                    stack.enter_context(
                        patch(
                            f"{module}.load_cached_first_model_ca_data",
                            return_value=((CAResidueRecord(1, "A", True),), None),
                        )
                    )
                records_95, _ = builder.build()
            return entry_attempts, records_95, record_filtered

    def test_homolog_build_requeues_persistent_http_5xx_from_every_request_kind(
        self,
    ) -> None:
        """Search, GraphQL and download 5xx errors get three entry attempts."""
        for failing_kind in (
            "sequence_search",
            "graphql",
            "nmr_download",
            "xray_download",
        ):
            with self.subTest(failing_kind=failing_kind):
                attempts, records, record_filtered = self._build_with_http_failures(
                    failing_kind, [503] * 100
                )
                self.assertEqual(attempts, 3)
                self.assertEqual(records, [])
                record_filtered.assert_called_once()
                self.assertIn("503", record_filtered.call_args.args[1])

    def test_homolog_build_recovers_after_transient_graphql_http_5xx(self) -> None:
        """A requeued entry is built once GraphQL answers again."""
        attempts, records, record_filtered = self._build_with_http_failures(
            "graphql", [503, 503]
        )
        self.assertEqual(attempts, 2)
        self.assertEqual(
            [(record.entry_id, record.xray_homolog_entity_ids) for record in records],
            [("1AAA", ("9XYZ_1",))],
        )
        record_filtered.assert_not_called()

    def test_homolog_build_does_not_requeue_coordinate_http_404(self) -> None:
        """A missing coordinate file is final and is not retried per entry."""
        attempts, records, record_filtered = self._build_with_http_failures(
            "nmr_download", []
        )
        self.assertEqual(attempts, 1)
        self.assertEqual(records, [])
        record_filtered.assert_called_once()
        self.assertIn("404", record_filtered.call_args.args[1])

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
        method_response = {
            "data": {
                "entries": [
                    {
                        "rcsb_id": "1MCD",
                        "exptl": [{"method": "X-RAY DIFFRACTION"}],
                    }
                ]
            }
        }
        client._post_json = Mock(
            side_effect=[
                entity_response,
                empty_resolution_response,
                method_response,
            ]
        )

        result = client.fetch_xray_polymer_entity_candidates_for_ids(["1MCD_1"])

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].polymer_entity_id, "1MCD_1")
        self.assertTrue(math.isnan(result[0].resolution_angstrom))
        self.assertEqual(result[0].experimental_methods, ("X-RAY DIFFRACTION",))

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
                    "src.dataset.builders.homologs.download_pdb_if_needed",
                    return_value=pdb_path,
                ),
                patch(
                    "src.dataset.builders.homologs.compute_stride_core_range_for_modeled_auth_seq_ids_in_first_model",
                    return_value=(1, 12),
                ),
            ):
                with self.assertRaises(NMRCoreContainsHetatmError):
                    builder._build_stride_core_query_sequence(seed)

    def test_rejects_xray_regions_containing_hetatm_at_all_cutoffs(self) -> None:
        """Reject an otherwise matching X-ray region at 95% and 100%."""
        sequence = "ACDEFGHIKLMNPQRSTVWY"
        nmr_residues = [
            CAResidueRecord(index, identity, True)
            for index, identity in enumerate(sequence, start=1)
        ]
        xray_residues = [
            CAResidueRecord(
                index,
                identity,
                index != 10,
                has_hetatm_ca=index == 10,
            )
            for index, identity in enumerate(sequence, start=1)
        ]

        for identity_percent in (95, 100):
            with self.subTest(identity_percent=identity_percent):
                self.assertEqual(
                    find_modeled_ca_core_identity_matches(
                        nmr_residues=nmr_residues,
                        xray_residues=xray_residues,
                        sequence_identity_percent=identity_percent,
                    ),
                    [],
                )

    def test_uses_clean_xray_repeat_when_another_repeat_has_hetatm(self) -> None:
        """Keep a candidate when a second matching region is HETATM-free."""
        sequence = "ACDEFGHIKLMNPQRSTVWY"
        nmr_residues = [
            CAResidueRecord(index, identity, True)
            for index, identity in enumerate(sequence, start=1)
        ]
        dirty_region = [
            CAResidueRecord(
                index,
                identity,
                index != 10,
                has_hetatm_ca=index == 10,
            )
            for index, identity in enumerate(sequence, start=1)
        ]
        separator = CAResidueRecord(
            50,
            "X",
            False,
            has_hetatm_ca=True,
        )
        clean_region = [
            CAResidueRecord(index, identity, True)
            for index, identity in enumerate(sequence, start=101)
        ]

        for identity_percent in (95, 100):
            with self.subTest(identity_percent=identity_percent):
                matches = find_modeled_ca_core_identity_matches(
                    nmr_residues=nmr_residues,
                    xray_residues=dirty_region + [separator] + clean_region,
                    sequence_identity_percent=identity_percent,
                )

                self.assertTrue(matches)
                self.assertTrue(
                    all(
                        xray_record.resid >= 101 and not xray_record.has_hetatm_ca
                        for match in matches
                        for _, xray_record in match
                    )
                )

    def test_rejects_95_percent_alignment_that_skips_internal_hetatm(self) -> None:
        """Treat a gapped-over HETATM CA as part of the X-ray region."""
        sequence = "ACDEFGHIKLMNPQRSTVWY"
        nmr_residues = [
            CAResidueRecord(index, identity, True)
            for index, identity in enumerate(sequence, start=1)
        ]
        xray_residues = [
            CAResidueRecord(index, identity, True)
            for index, identity in enumerate(sequence[:10], start=1)
        ]
        xray_residues.append(CAResidueRecord(11, "X", False, has_hetatm_ca=True))
        xray_residues.extend(
            CAResidueRecord(index, identity, True)
            for index, identity in enumerate(sequence[10:], start=12)
        )

        self.assertEqual(
            find_modeled_ca_core_identity_matches(
                nmr_residues=nmr_residues,
                xray_residues=xray_residues,
                sequence_identity_percent=95,
            ),
            [],
        )

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
                "src.dataset.builders.homologs.download_pdb_if_needed",
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
                    "src.dataset.builders.homologs.download_pdb_if_needed",
                    return_value=pdb_path,
                ),
                patch(
                    "src.dataset.builders.homologs.compute_stride_core_range_for_modeled_auth_seq_ids_in_first_model",
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
                        experimental_methods=("X-RAY DIFFRACTION",),
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

    def test_build_record_keeps_matches_and_reports_rejected_xray_homologs(
        self,
    ) -> None:
        """Retain modeled matches while describing every rejected X-ray entity."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = DatasetBuildConfig()
            client = RCSBClient(config)
            client.fetch_xray_polymer_entity_ids_by_sequence = Mock(
                return_value=["1KEEP_1", "2DROP_2"]
            )
            client.fetch_xray_polymer_entity_candidates_for_ids = Mock(
                return_value=[
                    XrayPolymerEntityCandidateRecord(
                        polymer_entity_id="1KEEP_1",
                        entry_id="1KEEP",
                        chain_ids=("A",),
                        resolution_angstrom=1.5,
                        experimental_methods=("X-RAY DIFFRACTION",),
                    ),
                    XrayPolymerEntityCandidateRecord(
                        polymer_entity_id="2DROP_2",
                        entry_id="2DROP",
                        chain_ids=("B", "C"),
                        resolution_angstrom=2.5,
                        experimental_methods=("X-RAY DIFFRACTION",),
                    ),
                ]
            )
            builder = SolutionNMRMonomerXrayHomologBuilder(
                client=client,
                config=config,
                stride_executable="stride",
                cache_dir=root,
                stride_cache_dir=root / "stride_cache",
            )
            sequence = "ACDEFGHIKLM"
            nmr_residues = [
                CAResidueRecord(index, identity, True)
                for index, identity in enumerate(sequence, start=7)
            ]
            seed = SolutionNMRMonomerXrayHomologSeedRecord("3NMR", 2003, "N")

            with patch.object(
                builder,
                "_xray_candidate_has_modeled_core_match",
                side_effect=lambda **kwargs: (
                    kwargs["candidate"].polymer_entity_id == "1KEEP_1"
                ),
            ):
                record = builder._build_record(
                    seed,
                    sequence_identity_percent=100,
                    core_query=(sequence, 7, 17, nmr_residues),
                )

            self.assertEqual(record.xray_homolog_entry_ids, ("1KEEP",))
            self.assertEqual(record.xray_homolog_entity_ids, ("1KEEP_1",))
            self.assertTrue(record.has_xray_homolog)
            self.assertEqual(len(record.rejected_xray_homologs), 1)
            rejected = record.rejected_xray_homologs[0]
            self.assertEqual(rejected.nmr_entry_id, "3NMR")
            self.assertEqual(rejected.nmr_year, 2003)
            self.assertEqual(rejected.nmr_chain_id, "N")
            self.assertEqual(rejected.sequence_identity_percent, 100)
            self.assertEqual(rejected.nmr_core_start_seq_id, 7)
            self.assertEqual(rejected.nmr_core_end_seq_id, 17)
            self.assertEqual(rejected.nmr_query_sequence_length, len(sequence))
            self.assertEqual(rejected.xray_entry_id, "2DROP")
            self.assertEqual(rejected.xray_entity_id, "2DROP_2")
            self.assertEqual(rejected.xray_chain_ids, ("B", "C"))
            self.assertIn("modeled core", rejected.reason.lower())

    def test_build_record_reports_hybrid_hit_as_method_rejection(self) -> None:
        """Audit a hybrid sequence hit without evaluating its coordinates."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = DatasetBuildConfig()
            client = RCSBClient(config)
            client.fetch_xray_polymer_entity_ids_by_sequence = Mock(
                return_value=["2HYB_1", "1PURE_1"]
            )
            client.fetch_xray_polymer_entity_candidates_for_ids = Mock(
                return_value=[
                    XrayPolymerEntityCandidateRecord(
                        polymer_entity_id="2HYB_1",
                        entry_id="2HYB",
                        chain_ids=("B", "C"),
                        resolution_angstrom=2.5,
                        experimental_methods=(
                            "X-RAY DIFFRACTION",
                            "NEUTRON DIFFRACTION",
                        ),
                    ),
                    XrayPolymerEntityCandidateRecord(
                        polymer_entity_id="1PURE_1",
                        entry_id="1PURE",
                        chain_ids=("A",),
                        resolution_angstrom=1.5,
                        experimental_methods=("X-RAY DIFFRACTION",),
                    ),
                ]
            )
            builder = SolutionNMRMonomerXrayHomologBuilder(
                client=client,
                config=config,
                stride_executable="stride",
                cache_dir=root,
                stride_cache_dir=root / "stride_cache",
            )
            sequence = "ACDEFGHIKLM"
            nmr_residues = [
                CAResidueRecord(index, identity, True)
                for index, identity in enumerate(sequence, start=7)
            ]
            seed = SolutionNMRMonomerXrayHomologSeedRecord("3NMR", 2003, "N")

            with patch.object(
                builder,
                "_xray_candidate_has_modeled_core_match",
                return_value=True,
            ) as modeled_core_check:
                record = builder._build_record(
                    seed,
                    sequence_identity_percent=100,
                    core_query=(sequence, 7, 17, nmr_residues),
                )

            self.assertEqual(record.xray_homolog_entry_ids, ("1PURE",))
            self.assertEqual(record.xray_homolog_entity_ids, ("1PURE_1",))
            self.assertTrue(record.has_xray_homolog)
            self.assertEqual(len(record.rejected_xray_homologs), 1)
            rejected = record.rejected_xray_homologs[0]
            self.assertEqual(rejected.xray_entry_id, "2HYB")
            self.assertEqual(rejected.xray_entity_id, "2HYB_1")
            self.assertEqual(rejected.xray_chain_ids, ("B", "C"))
            self.assertEqual(
                rejected.reason,
                XRAY_HOMOLOG_METHOD_REJECTION_REASON,
            )
            modeled_core_check.assert_called_once()
            self.assertEqual(
                modeled_core_check.call_args.kwargs["candidate"].polymer_entity_id,
                "1PURE_1",
            )

    @staticmethod
    def _candidate_metadata_client(
        exptl_entries: list[dict | None],
    ) -> RCSBClient:
        """Return a client whose GraphQL reports two hits and given methods."""
        client = RCSBClient(DatasetBuildConfig())
        client.fetch_xray_polymer_entity_ids_by_sequence = Mock(
            return_value=["9XYZ_1", "1PUR_1"]
        )

        def post_json(url: str, payload: dict) -> dict:
            query = payload["query"]
            if "polymer_entities" in query:
                return {
                    "data": {
                        "polymer_entities": [
                            {
                                "rcsb_id": entity_id,
                                "entity_poly": {"pdbx_strand_id": "A"},
                                "rcsb_polymer_entity_container_identifiers": {
                                    "entry_id": entity_id.split("_")[0]
                                },
                            }
                            for entity_id in payload["variables"]["ids"]
                        ]
                    }
                }
            if "exptl" in query:
                return {
                    "data": {
                        "entries": [
                            {
                                "rcsb_id": "1PUR",
                                "exptl": [{"method": "X-RAY DIFFRACTION"}],
                            },
                            *exptl_entries,
                        ]
                    }
                }
            return {"data": {"entries": []}}

        client._post_json = post_json
        return client

    def _build_record_with_candidate_methods(
        self, exptl_entries: list[dict | None]
    ) -> tuple[SolutionNMRMonomerXrayHomologRecord | Exception, Mock]:
        """Build one 95% record through the real candidate metadata parser."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            builder = SolutionNMRMonomerXrayHomologBuilder(
                client=self._candidate_metadata_client(exptl_entries),
                config=DatasetBuildConfig(),
                stride_executable="stride",
                cache_dir=root,
                stride_cache_dir=root / "stride_cache",
            )
            sequence = "ACDEFGHIKLM"
            nmr_residues = [
                CAResidueRecord(index, identity, True)
                for index, identity in enumerate(sequence, start=1)
            ]
            seed = SolutionNMRMonomerXrayHomologSeedRecord("3NMR", 2003, "N")
            with patch.object(
                builder,
                "_xray_candidate_has_modeled_core_match",
                return_value=True,
            ) as modeled_core_check:
                try:
                    result = builder._build_record(
                        seed,
                        sequence_identity_percent=95,
                        core_query=(sequence, 1, 11, nmr_residues),
                    )
                except Exception as exc:
                    result = exc
            return result, modeled_core_check

    def test_build_record_fails_entry_when_candidate_methods_are_missing(
        self,
    ) -> None:
        """Absent methods are a metadata error, not a method rejection."""
        cases = {
            "entry absent": [],
            "null entry": [None],
            "null exptl": [{"rcsb_id": "9XYZ", "exptl": None}],
            "empty exptl": [{"rcsb_id": "9XYZ", "exptl": []}],
            "null method": [{"rcsb_id": "9XYZ", "exptl": [{"method": None}]}],
        }
        for label, exptl_entries in cases.items():
            with self.subTest(case=label):
                result, modeled_core_check = self._build_record_with_candidate_methods(
                    exptl_entries
                )
                self.assertIsInstance(result, XrayHomologEvaluationError)
                self.assertIn("9XYZ_1", str(result))
                self.assertNotIn("1PUR_1", str(result))
                modeled_core_check.assert_not_called()

    def test_build_record_keeps_reported_hybrid_methods_as_rejections(
        self,
    ) -> None:
        """Reported non-X-ray methods still produce a normal rejection row."""
        result, modeled_core_check = self._build_record_with_candidate_methods(
            [
                {
                    "rcsb_id": "9XYZ",
                    "exptl": [
                        {"method": "X-RAY DIFFRACTION"},
                        {"method": "NEUTRON DIFFRACTION"},
                    ],
                }
            ]
        )
        self.assertIsInstance(result, SolutionNMRMonomerXrayHomologRecord)
        self.assertEqual(result.xray_homolog_entity_ids, ("1PUR_1",))
        self.assertEqual(
            [
                (rejected.xray_entity_id, rejected.reason)
                for rejected in result.rejected_xray_homologs
            ],
            [("9XYZ_1", XRAY_HOMOLOG_METHOD_REJECTION_REASON)],
        )
        modeled_core_check.assert_called_once()

    def test_homolog_build_excludes_entry_with_missing_candidate_methods(
        self,
    ) -> None:
        """Missing methods remove the entry instead of writing a false zero."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            client = self._candidate_metadata_client([])
            client.fetch_xray_polymer_entity_ids_by_sequence = Mock(
                return_value=["9XYZ_1"]
            )
            seed = SolutionNMRMonomerXrayHomologSeedRecord("3NMR", 2003, "N")
            client.fetch_solution_nmr_monomer_xray_homolog_seed_records_for_ids = Mock(
                return_value=[seed]
            )
            builder = SolutionNMRMonomerXrayHomologBuilder(
                client=client,
                config=DatasetBuildConfig(max_workers=1),
                stride_executable="stride",
                cache_dir=root,
                stride_cache_dir=root / "stride_cache",
            )
            sequence = "ACDEFGHIKLM"
            nmr_residues = [
                CAResidueRecord(index, identity, True)
                for index, identity in enumerate(sequence, start=1)
            ]
            completed: list[tuple[str, str]] = []
            with (
                patch(
                    "src.dataset.builders.homologs.fetch_solution_nmr_entry_ids",
                    return_value=["3NMR"],
                ),
                patch(
                    "src.dataset.builders.homologs._record_filtered_structure"
                ) as record_filtered,
                patch.object(
                    builder,
                    "_build_stride_core_query_sequence",
                    return_value=(sequence, 1, 11, nmr_residues),
                ) as core_query,
            ):
                records_95, records_100 = builder.build(
                    on_entry_complete=lambda entry_id, status: completed.append(
                        (entry_id, status)
                    )
                )

            self.assertEqual(records_95, [])
            self.assertEqual(records_100, [])
            core_query.assert_called_once()
            record_filtered.assert_called_once()
            self.assertEqual(record_filtered.call_args.args[0], "3NMR")
            self.assertIn("9XYZ_1", record_filtered.call_args.args[1])
            self.assertNotIn(("3NMR", "success"), completed)

    def test_rejected_xray_homolog_csv_path_round_trip_and_empty_header(
        self,
    ) -> None:
        """Derive the sibling path and preserve rejected-homolog CSV records."""
        record = RejectedXrayHomologRecord(
            nmr_entry_id="3NMR",
            nmr_year=2003,
            nmr_chain_id="N",
            sequence_identity_percent=95,
            nmr_core_start_seq_id=7,
            nmr_core_end_seq_id=17,
            nmr_query_sequence_length=11,
            xray_entry_id="2DROP",
            xray_entity_id="2DROP_2",
            xray_chain_ids=("B", "C"),
            reason="no eligible modeled core match",
        )
        expected_header = (
            "nmr_entry_id,nmr_year,nmr_chain_id,sequence_identity_percent,"
            "nmr_core_start_seq_id,nmr_core_end_seq_id,nmr_query_sequence_length,"
            "xray_entry_id,xray_entity_id,xray_chain_ids,reason\n"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            primary_path = root / "homologs.csv"
            rejected_path = rejected_xray_homologs_csv_path(primary_path)
            self.assertEqual(rejected_path, root / "homologs_rejected.csv")

            write_rejected_xray_homolog_csv([record], rejected_path)
            self.assertEqual(read_rejected_xray_homolog_csv(rejected_path), [record])

            empty_path = root / "empty.csv"
            write_rejected_xray_homolog_csv([], empty_path)
            self.assertEqual(empty_path.read_text(encoding="utf-8"), expected_header)
            self.assertEqual(read_rejected_xray_homolog_csv(empty_path), [])

    def test_record_pair_reuses_candidate_metadata_and_parsed_chain(self) -> None:
        """Avoid repeating common X-ray work for the 95% and 100% rows."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pdb_path = root / "1XYZ.pdb"
            sequence = "ACDEFGHIKLMNPQRSTVWY"
            pdb_path.write_text(
                "".join(
                    f"ATOM  {serial:5d}  CA  ALA A{serial:4d}    "
                    f"{0.0:8.3f}{0.0:8.3f}{0.0:8.3f}{1.0:6.2f}{20.0:6.2f}"
                    "           C\n"
                    for serial in range(1, len(sequence) + 1)
                ),
                encoding="utf-8",
            )
            config = DatasetBuildConfig()
            client = RCSBClient(config)
            client.fetch_xray_polymer_entity_ids_by_sequence = Mock(
                return_value=["1XYZ_1"]
            )
            candidate = XrayPolymerEntityCandidateRecord(
                polymer_entity_id="1XYZ_1",
                entry_id="1XYZ",
                chain_ids=("A",),
                resolution_angstrom=2.0,
                experimental_methods=("X-RAY DIFFRACTION",),
            )
            client.fetch_xray_polymer_entity_candidates_for_ids = Mock(
                return_value=[candidate]
            )
            builder = SolutionNMRMonomerXrayHomologBuilder(
                client=client,
                config=config,
                stride_executable="stride",
                cache_dir=root,
                stride_cache_dir=root / "stride_cache",
            )
            nmr_residues = [
                CAResidueRecord(index, identity, True)
                for index, identity in enumerate(sequence, start=1)
            ]
            seed = SolutionNMRMonomerXrayHomologSeedRecord("2NMR", 2000, "A")

            with (
                patch.object(
                    builder,
                    "_build_stride_core_query_sequence",
                    return_value=(sequence, 1, len(sequence), nmr_residues),
                ),
                patch(
                    "src.dataset.builders.homologs.download_pdb_chain_subset_if_needed",
                    return_value=(pdb_path, {}),
                ) as download_subset,
                patch(
                    "src.dataset.builders.homologs.load_cached_first_model_ca_data",
                    return_value=(tuple(nmr_residues), {}),
                ) as load_ca_data,
            ):
                record_95, record_100 = builder._build_record_pair(seed)

            self.assertTrue(record_95.has_xray_homolog)
            self.assertTrue(record_100.has_xray_homolog)
            self.assertEqual(
                client.fetch_xray_polymer_entity_candidates_for_ids.call_count,
                1,
            )
            self.assertEqual(download_subset.call_count, 1)
            self.assertEqual(load_ca_data.call_count, 1)

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
                    "src.dataset.builders.homologs.download_pdb_chain_subset_if_needed",
                    return_value=(root / "subset.pdb", {}),
                ) as download_subset,
                patch(
                    "src.dataset.builders.homologs.load_cached_first_model_ca_data",
                    return_value=(tuple(), {}),
                ),
                patch(
                    "src.dataset.builders.homologs.find_modeled_ca_core_identity_matches",
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
