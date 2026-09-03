"""Network-free coverage for homolog and X-ray RMSD builder orchestration."""

from concurrent.futures import wait as futures_wait
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from src import pdb_dataset_builder as builder_module
from src.pdb_dataset_builder import (
    DatasetBuildConfig,
    NMRHomologyQueryIneligibleError,
    RCSBClient,
    SolutionNMRMonomerXrayHomologBuilder,
    SolutionNMRMonomerXrayHomologRecord,
    SolutionNMRMonomerXrayHomologSeedRecord,
    SolutionNMRMonomerXrayRmsdBuilder,
    SolutionNMRMonomerXrayRmsdExtremesRecord,
    SolutionNMRMonomerXrayRmsdRecord,
    XrayHomologEvaluationError,
    XrayPolymerEntityCandidateRecord,
    build_solution_nmr_monomer_xray_rmsd_extremes_to_csv,
    build_solution_nmr_monomer_xray_rmsd_to_csv,
    read_solution_nmr_monomer_xray_rmsd_csv,
    read_solution_nmr_monomer_xray_rmsd_extremes_csv,
)


def _homolog(
    entry_id: str,
    *,
    year: int = 2000,
    identity: int = 100,
    core_start: int | None = 1,
    core_end: int | None = 20,
    entity_ids: tuple[str, ...] = ("1AAA_1",),
) -> SolutionNMRMonomerXrayHomologRecord:
    """Create a compact homolog fixture with internally consistent hit fields."""
    entry_ids = tuple(
        dict.fromkeys(entity_id.split("_", 1)[0] for entity_id in entity_ids)
    )
    return SolutionNMRMonomerXrayHomologRecord(
        entry_id=entry_id,
        year=year,
        sequence_identity_percent=identity,
        nmr_core_start_seq_id=core_start,
        nmr_core_end_seq_id=core_end,
        nmr_query_sequence_length=20,
        xray_homolog_entry_ids=entry_ids,
        xray_homolog_entity_ids=entity_ids,
        has_xray_homolog=bool(entity_ids),
    )


def _candidate(
    entity_id: str,
    resolution: float = 2.0,
    chain_ids: tuple[str, ...] = ("X",),
) -> XrayPolymerEntityCandidateRecord:
    """Create an X-ray metadata fixture."""
    return XrayPolymerEntityCandidateRecord(
        polymer_entity_id=entity_id,
        entry_id=entity_id.split("_", 1)[0],
        chain_ids=chain_ids,
        resolution_angstrom=resolution,
    )


def _rmsd(
    entry_id: str,
    entity_id: str,
    *,
    year: int = 2000,
    identity: int = 100,
    rmsd: float = 1.25,
    resolution: float = 1.8,
) -> SolutionNMRMonomerXrayRmsdRecord:
    """Create a successful RMSD fixture."""
    xray_entry_id = entity_id.split("_", 1)[0]
    return SolutionNMRMonomerXrayRmsdRecord(
        entry_id=entry_id,
        year=year,
        sequence_identity_percent=identity,
        nmr_chain_id="N",
        nmr_core_start_seq_id=1,
        nmr_core_end_seq_id=20,
        nmr_query_sequence_length=20,
        xray_homolog_entity_id=entity_id,
        xray_homolog_count=2,
        xray_entry_id=xray_entry_id,
        xray_chain_id="X",
        xray_core_start_seq_id=31,
        xray_core_end_seq_id=50,
        xray_resolution_angstrom=resolution,
        n_common_ca=20,
        rmsd_ca_angstrom=rmsd,
    )


def _extremes(
    entry_id: str,
    *,
    year: int = 2000,
    identity: int = 100,
    best_entity_id: str = "1AAA_1",
    worst_entity_id: str = "2BBB_1",
) -> SolutionNMRMonomerXrayRmsdExtremesRecord:
    """Create a successful RMSD-extremes fixture."""
    return SolutionNMRMonomerXrayRmsdExtremesRecord(
        entry_id=entry_id,
        year=year,
        sequence_identity_percent=identity,
        nmr_chain_id="N",
        nmr_core_start_seq_id=1,
        nmr_core_end_seq_id=20,
        nmr_query_sequence_length=20,
        xray_homolog_count=2,
        successful_xray_homolog_count=2,
        best_xray_homolog_entity_id=best_entity_id,
        best_xray_entry_id=best_entity_id.split("_", 1)[0],
        best_xray_chain_id="A",
        best_xray_resolution_angstrom=1.2,
        best_xray_core_start_seq_id=11,
        best_xray_core_end_seq_id=30,
        best_n_common_ca=20,
        best_rmsd_ca_angstrom=0.5,
        worst_xray_homolog_entity_id=worst_entity_id,
        worst_xray_entry_id=worst_entity_id.split("_", 1)[0],
        worst_xray_chain_id="B",
        worst_xray_resolution_angstrom=2.2,
        worst_xray_core_start_seq_id=21,
        worst_xray_core_end_seq_id=40,
        worst_n_common_ca=19,
        worst_rmsd_ca_angstrom=3.0,
        rmsd_delta_angstrom=2.5,
    )


def _rmsd_builder(
    tmp_path: Path,
    homolog_records: list[SolutionNMRMonomerXrayHomologRecord] | None = None,
    *,
    identity: int = 100,
    batch_size: int = 2,
) -> SolutionNMRMonomerXrayRmsdBuilder:
    """Create an RMSD builder whose client can be mocked without network I/O."""
    config = DatasetBuildConfig(
        graphql_batch_size=batch_size,
        max_workers=1,
    )
    return SolutionNMRMonomerXrayRmsdBuilder(
        client=RCSBClient(config),
        config=config,
        cache_dir=tmp_path / "rmsd-cache",
        rmsd_workers=1,
        homolog_records=homolog_records or [],
        sequence_identity_percent=identity,
    )


def test_homolog_build_handles_heartbeat_ineligible_and_non_5xx_failure(
    tmp_path: Path,
) -> None:
    """Finish independent seeds while reporting every terminal outcome correctly."""
    config = DatasetBuildConfig(graphql_batch_size=10, max_workers=1)
    client = RCSBClient(config)
    seeds = [
        SolutionNMRMonomerXrayHomologSeedRecord("LATE", 2004, "A"),
        SolutionNMRMonomerXrayHomologSeedRecord("INEL", 2001, "A"),
        SolutionNMRMonomerXrayHomologSeedRecord("FAIL", 2002, "A"),
        SolutionNMRMonomerXrayHomologSeedRecord("EARLY", 1999, "A"),
    ]
    client.fetch_solution_nmr_monomer_xray_homolog_seed_records_for_ids = Mock(
        return_value=seeds
    )
    homolog_builder = SolutionNMRMonomerXrayHomologBuilder(
        client=client,
        config=config,
        stride_executable="stride",
        cache_dir=tmp_path / "pdb",
        stride_cache_dir=tmp_path / "stride",
    )

    def build_pair(
        seed: SolutionNMRMonomerXrayHomologSeedRecord,
    ) -> tuple[
        SolutionNMRMonomerXrayHomologRecord,
        SolutionNMRMonomerXrayHomologRecord,
    ]:
        if seed.entry_id == "INEL":
            raise NMRHomologyQueryIneligibleError(seed.entry_id, "synthetic core issue")
        if seed.entry_id == "FAIL":
            raise RuntimeError("synthetic parse failure")
        return (
            _homolog(seed.entry_id, year=seed.year, identity=95),
            _homolog(seed.entry_id, year=seed.year, identity=100),
        )

    wait_count = 0

    def heartbeat_then_wait(pending, *, timeout, return_when):
        nonlocal wait_count
        wait_count += 1
        if wait_count == 1:
            return set(), pending
        return futures_wait(pending, timeout=timeout, return_when=return_when)

    completed: list[tuple[str, str]] = []
    record_pairs: list[tuple[str, str]] = []
    with (
        patch.object(
            builder_module,
            "fetch_solution_nmr_entry_ids",
            return_value=[seed.entry_id for seed in seeds],
        ),
        patch.object(homolog_builder, "_build_record_pair", side_effect=build_pair),
        patch.object(
            builder_module, "wait", side_effect=heartbeat_then_wait
        ) as wait_mock,
        patch.object(builder_module, "_record_filtered_structure") as record_filtered,
    ):
        records_95, records_100 = homolog_builder.build(
            on_record_pair=lambda current, historical: record_pairs.append(
                (current.entry_id, historical.entry_id)
            ),
            on_entry_complete=lambda entry_id, status: completed.append(
                (entry_id, status)
            ),
        )

    assert [record.entry_id for record in records_95] == ["EARLY", "LATE"]
    assert [record.entry_id for record in records_100] == ["EARLY", "LATE"]
    assert set(record_pairs) == {("EARLY", "EARLY"), ("LATE", "LATE")}
    assert set(completed) == {
        ("EARLY", "success"),
        ("LATE", "success"),
        ("INEL", "ineligible"),
    }
    record_filtered.assert_any_call("INEL", "synthetic core issue", year=2001)
    record_filtered.assert_any_call(
        "FAIL",
        "X-ray homology search failed: synthetic parse failure",
        year=2002,
    )
    assert wait_mock.call_count >= 2


def test_homolog_record_and_candidate_failure_branches_are_network_free(
    tmp_path: Path,
) -> None:
    """Exercise negative lookup and metadata/chain failures without coordinates."""
    config = DatasetBuildConfig(graphql_batch_size=2, max_workers=1)
    client = RCSBClient(config)
    homolog_builder = SolutionNMRMonomerXrayHomologBuilder(
        client=client,
        config=config,
        stride_executable="stride",
        cache_dir=tmp_path / "pdb",
        stride_cache_dir=tmp_path / "stride",
    )
    seed = SolutionNMRMonomerXrayHomologSeedRecord("NMR1", 2000, "N")
    core_query = ("ACDEFGHIKLM", 5, 15, [])
    client.fetch_xray_polymer_entity_ids_by_sequence = Mock(return_value=[])

    with patch.object(
        homolog_builder,
        "_build_stride_core_query_sequence",
        return_value=core_query,
    ) as build_core:
        record = homolog_builder._build_record(seed, sequence_identity_percent=100)

    assert record.xray_homolog_entity_ids == ()
    assert record.xray_homolog_entry_ids == ()
    assert record.has_xray_homolog is False
    build_core.assert_called_once_with(seed)

    assert (
        homolog_builder._filter_modeled_xray_homolog_entity_ids(
            xray_entity_ids=(),
            nmr_core_residues=[],
            sequence_identity_percent=100,
        )
        == ()
    )
    client.fetch_xray_polymer_entity_candidates_for_ids = Mock(return_value=[])
    with pytest.raises(XrayHomologEvaluationError, match="Missing X-ray candidate"):
        homolog_builder._filter_modeled_xray_homolog_entity_ids(
            xray_entity_ids=("MISS_1",),
            nmr_core_residues=[],
            sequence_identity_percent=100,
        )

    failing_candidate = _candidate("BROKEN_1", chain_ids=("A", "B"))
    with patch.object(
        builder_module,
        "download_pdb_chain_subset_if_needed",
        side_effect=OSError("offline fixture"),
    ):
        with pytest.raises(XrayHomologEvaluationError, match="2/2 chains failed"):
            homolog_builder._xray_candidate_has_modeled_core_match(
                nmr_core_residues=[],
                candidate=failing_candidate,
                sequence_identity_percent=100,
            )


def test_prepare_work_items_filters_batches_and_sorts_candidates(
    tmp_path: Path,
) -> None:
    """Keep only actionable homologs and order finite resolutions before NaN."""
    homologs = [
        _homolog("WRONG", identity=95),
        _homolog("SKIP"),
        _homolog("NOCORE", core_start=None),
        _homolog("NOHIT", entity_ids=()),
        _homolog("NOCHAIN", entity_ids=("CHAINLESS_1",)),
        _homolog("NOMETA", entity_ids=("MISSING_1",)),
        _homolog("BETA", year=2002, entity_ids=("NAN_1", "TIEB_1", "LOW_1", "TIEA_1")),
        _homolog("ALPHA", year=2001, entity_ids=("SHARED_1",)),
    ]
    rmsd_builder = _rmsd_builder(tmp_path, homologs)

    def seeds_for_ids(
        entry_ids: list[str],
    ) -> list[SolutionNMRMonomerXrayHomologSeedRecord]:
        return [
            SolutionNMRMonomerXrayHomologSeedRecord(
                entry_id,
                2000,
                f"{entry_id}-CHAIN",
            )
            for entry_id in entry_ids
            if entry_id != "NOCHAIN"
        ]

    candidates = {
        "CHAINLESS_1": _candidate("CHAINLESS_1", 1.0),
        "SHARED_1": _candidate("SHARED_1", 1.1),
        "LOW_1": _candidate("LOW_1", 0.9),
        "TIEA_1": _candidate("TIEA_1", 1.5),
        "TIEB_1": _candidate("TIEB_1", 1.5),
        "NAN_1": _candidate("NAN_1", np.nan),
    }

    def candidates_for_ids(
        entity_ids: list[str],
    ) -> list[XrayPolymerEntityCandidateRecord]:
        return [
            candidates[entity_id] for entity_id in entity_ids if entity_id in candidates
        ]

    seed_fetch = Mock(side_effect=seeds_for_ids)
    candidate_fetch = Mock(side_effect=candidates_for_ids)
    rmsd_builder.client.fetch_solution_nmr_monomer_xray_homolog_seed_records_for_ids = (
        seed_fetch
    )
    rmsd_builder.client.fetch_xray_polymer_entity_candidates_for_ids = candidate_fetch

    with patch.object(builder_module, "_record_filtered_structure") as record_filtered:
        work_items = rmsd_builder._prepare_work_items(
            skip_entry_ids={"SKIP"},
            progress_prefix="TEST RMSD",
        )

    assert [homolog.entry_id for homolog, _, _ in work_items] == ["ALPHA", "BETA"]
    assert [chain_id for _, chain_id, _ in work_items] == [
        "ALPHA-CHAIN",
        "BETA-CHAIN",
    ]
    assert [candidate.polymer_entity_id for candidate in work_items[1][2]] == [
        "LOW_1",
        "TIEA_1",
        "TIEB_1",
        "NAN_1",
    ]
    assert seed_fetch.call_count == 2
    assert candidate_fetch.call_count >= 3
    requested_seed_ids = {
        entry_id
        for batch_call in seed_fetch.call_args_list
        for entry_id in batch_call.args[0]
    }
    assert requested_seed_ids == {"ALPHA", "BETA", "NOCHAIN", "NOMETA"}
    record_filtered.assert_any_call("NOCORE", "NMR core range is missing", year=2000)
    record_filtered.assert_any_call(
        "NOHIT",
        "no X-ray homologs at 100% sequence identity",
        year=2000,
    )
    record_filtered.assert_any_call(
        "NOCHAIN",
        "NMR chain metadata is missing",
        year=2000,
    )
    record_filtered.assert_any_call(
        "NOMETA",
        "X-ray homolog candidate metadata is missing",
        year=2000,
    )


def test_rmsd_builder_rejects_unsupported_identity(tmp_path: Path) -> None:
    """Reject a cutoff for which no homolog view exists."""
    with pytest.raises(ValueError, match="must be 95 or 100"):
        _rmsd_builder(tmp_path, identity=90)


def test_compute_candidate_record_selects_best_usable_chain(tmp_path: Path) -> None:
    """Prefer more common CA atoms, then lower RMSD, across recoverable failures."""
    rmsd_builder = _rmsd_builder(tmp_path, identity=95)
    homolog = _homolog(
        "NMR1",
        identity=95,
        entity_ids=("XRAY_1",),
    )
    candidate = _candidate(
        "XRAY_1",
        1.4,
        chain_ids=("DOWNLOAD", "ERROR", "NONE", "FEW", "MANY", "BETTER"),
    )
    prepared_core = (tuple(), {})

    def download_subset(*, chain_ids, **_kwargs):
        chain_id = chain_ids[0]
        if chain_id == "DOWNLOAD":
            raise OSError("missing fixture")
        return tmp_path / f"{chain_id}.pdb", {chain_id: f"parsed-{chain_id}"}

    def compute_rmsd(*, xray_chain_id, **_kwargs):
        if xray_chain_id == "parsed-ERROR":
            raise ValueError("bad coordinates")
        if xray_chain_id == "parsed-NONE":
            return None
        if xray_chain_id == "parsed-FEW":
            return 10, 0.2, 1, 20, 31, 50
        if xray_chain_id == "parsed-MANY":
            return 11, 4.0, 1, 20, 41, 60
        return 11, 0.5, 1, 20, 51, 70

    with (
        patch.object(
            builder_module,
            "download_pdb_chain_subset_if_needed",
            side_effect=download_subset,
        ),
        patch.object(
            rmsd_builder,
            "_compute_ca_rmsd_to_xray",
            side_effect=compute_rmsd,
        ) as compute,
    ):
        record = rmsd_builder._compute_candidate_record(
            homolog=homolog,
            nmr_chain_id="N",
            nmr_pdb_path=tmp_path / "nmr.pdb",
            parsed_nmr_chain_id="parsed-N",
            prepared_nmr_core=prepared_core,
            candidate=candidate,
        )

    assert record is not None
    assert record.xray_chain_id == "BETTER"
    assert record.n_common_ca == 11
    assert record.rmsd_ca_angstrom == 0.5
    assert record.xray_core_start_seq_id == 51
    assert record.sequence_identity_percent == 95
    assert compute.call_count == 5
    assert all(
        call_kwargs.kwargs["prepared_nmr_core"] is prepared_core
        for call_kwargs in compute.call_args_list
    )

    with (
        patch.object(
            builder_module,
            "download_pdb_chain_subset_if_needed",
            return_value=(tmp_path / "xray.pdb", {}),
        ),
        patch.object(rmsd_builder, "_compute_ca_rmsd_to_xray", return_value=None),
    ):
        no_record = rmsd_builder._compute_candidate_record(
            homolog=homolog,
            nmr_chain_id="N",
            nmr_pdb_path=tmp_path / "nmr.pdb",
            parsed_nmr_chain_id="parsed-N",
            prepared_nmr_core=prepared_core,
            candidate=_candidate("NONE_1"),
        )

    assert no_record is None


def test_compute_candidate_records_reports_guard_and_failure_reasons(
    tmp_path: Path,
) -> None:
    """Return empty candidate sets for every recoverable preparation failure."""
    rmsd_builder = _rmsd_builder(tmp_path)
    valid = _homolog("VALID", entity_ids=("ONE_1", "TWO_1"))
    candidates = (_candidate("ONE_1"), _candidate("TWO_1"))
    successful = _rmsd("VALID", "TWO_1")

    with patch.object(builder_module, "_record_filtered_structure") as record_filtered:
        assert (
            rmsd_builder._compute_candidate_records(
                homolog=_homolog("NOCORE", core_end=None),
                nmr_chain_id="N",
                candidates=candidates,
            )
            == ()
        )
        assert (
            rmsd_builder._compute_candidate_records(
                homolog=valid,
                nmr_chain_id="N",
                candidates=(),
            )
            == ()
        )
        with patch.object(
            rmsd_builder,
            "_download_pdb_if_needed",
            side_effect=OSError("cache unavailable"),
        ):
            assert (
                rmsd_builder._compute_candidate_records(
                    homolog=valid,
                    nmr_chain_id="N",
                    candidates=candidates,
                )
                == ()
            )
        with (
            patch.object(
                rmsd_builder,
                "_download_pdb_if_needed",
                return_value=tmp_path / "nmr.pdb",
            ),
            patch.object(builder_module, "load_cached_chain_id_map", return_value={}),
            patch.object(rmsd_builder, "_prepare_nmr_core_data", return_value=None),
        ):
            assert (
                rmsd_builder._compute_candidate_records(
                    homolog=valid,
                    nmr_chain_id="N",
                    candidates=candidates,
                )
                == ()
            )
        with (
            patch.object(
                rmsd_builder,
                "_download_pdb_if_needed",
                return_value=tmp_path / "nmr.pdb",
            ),
            patch.object(builder_module, "load_cached_chain_id_map", return_value={}),
            patch.object(
                rmsd_builder,
                "_prepare_nmr_core_data",
                return_value=(tuple(), {}),
            ),
            patch.object(rmsd_builder, "_compute_candidate_record", return_value=None),
        ):
            assert (
                rmsd_builder._compute_candidate_records(
                    homolog=valid,
                    nmr_chain_id="N",
                    candidates=candidates,
                )
                == ()
            )
        with (
            patch.object(
                rmsd_builder,
                "_download_pdb_if_needed",
                return_value=tmp_path / "nmr.pdb",
            ),
            patch.object(
                builder_module,
                "load_cached_chain_id_map",
                return_value={"N": "parsed-N"},
            ),
            patch.object(
                rmsd_builder,
                "_prepare_nmr_core_data",
                return_value=(tuple(), {}),
            ),
            patch.object(
                rmsd_builder,
                "_compute_candidate_record",
                side_effect=[None, successful],
            ) as compute_candidate,
        ):
            assert rmsd_builder._compute_candidate_records(
                homolog=valid,
                nmr_chain_id="N",
                candidates=candidates,
            ) == (successful,)

    record_filtered.assert_any_call("NOCORE", "NMR core range is missing", year=2000)
    record_filtered.assert_any_call(
        "VALID",
        "no usable X-ray homolog candidates",
        year=2000,
    )
    record_filtered.assert_any_call(
        "VALID",
        "X-ray RMSD calculation failed: cache unavailable",
        year=2000,
    )
    record_filtered.assert_any_call(
        "VALID",
        "NMR core cannot be prepared for X-ray RMSD",
        year=2000,
    )
    record_filtered.assert_any_call(
        "VALID",
        "no X-ray homolog candidate produced a usable CA RMSD",
        year=2000,
    )
    assert compute_candidate.call_count == 2
    assert all(
        call_kwargs.kwargs["parsed_nmr_chain_id"] == "parsed-N"
        for call_kwargs in compute_candidate.call_args_list
    )


def _orchestration_work_items():
    """Return deliberately unsorted work items for concurrency projections."""
    late = _homolog("LATE", year=2004, entity_ids=("LATE_X_1", "LATE_Y_1"))
    empty = _homolog("EMPTY", year=2002, entity_ids=("EMPTY_X_1",))
    early = _homolog("EARLY", year=1999, entity_ids=("EARLY_X_1", "EARLY_Y_1"))
    return [
        (late, "N", (_candidate("LATE_X_1"),)),
        (empty, "N", (_candidate("EMPTY_X_1"),)),
        (early, "N", (_candidate("EARLY_X_1"),)),
    ]


def test_build_projects_records_sorts_and_calls_back(tmp_path: Path) -> None:
    """Project ordinary records, omitting empty candidate results."""
    rmsd_builder = _rmsd_builder(tmp_path)
    work_items = _orchestration_work_items()
    candidate_sets = {
        "LATE": (_rmsd("LATE", "LATE_X_1", year=2004),),
        "EMPTY": (),
        "EARLY": (_rmsd("EARLY", "EARLY_X_1", year=1999),),
    }

    with (
        patch.object(
            rmsd_builder,
            "_prepare_work_items",
            return_value=work_items,
        ) as prepare,
        patch.object(
            rmsd_builder,
            "_compute_candidate_records",
            side_effect=lambda **kwargs: candidate_sets[kwargs["homolog"].entry_id],
        ) as compute,
    ):
        callbacks: list[SolutionNMRMonomerXrayRmsdRecord] = []
        records = rmsd_builder.build(
            skip_entry_ids={"DONE"},
            on_record=callbacks.append,
        )

    assert [record.entry_id for record in records] == ["EARLY", "LATE"]
    assert {record.entry_id for record in callbacks} == {"EARLY", "LATE"}
    prepare.assert_called_once_with(
        skip_entry_ids={"DONE"},
        progress_prefix="SOLUTION NMR X-ray RMSD",
    )
    assert compute.call_count == 3


def test_build_extremes_projects_records_sorts_and_calls_back(tmp_path: Path) -> None:
    """Project extrema records, omitting entries without successful candidates."""
    rmsd_builder = _rmsd_builder(tmp_path)
    work_items = _orchestration_work_items()
    candidate_sets = {
        "LATE": (
            _rmsd("LATE", "LATE_X_1", year=2004, rmsd=3.0),
            _rmsd("LATE", "LATE_Y_1", year=2004, rmsd=0.5),
        ),
        "EMPTY": (),
        "EARLY": (
            _rmsd("EARLY", "EARLY_X_1", year=1999, rmsd=2.0),
            _rmsd("EARLY", "EARLY_Y_1", year=1999, rmsd=1.0),
        ),
    }

    with (
        patch.object(
            rmsd_builder,
            "_prepare_work_items",
            return_value=work_items,
        ) as prepare,
        patch.object(
            rmsd_builder,
            "_compute_candidate_records",
            side_effect=lambda **kwargs: candidate_sets[kwargs["homolog"].entry_id],
        ),
    ):
        callbacks: list[SolutionNMRMonomerXrayRmsdExtremesRecord] = []
        records = rmsd_builder.build_extremes(on_record=callbacks.append)

    assert [record.entry_id for record in records] == ["EARLY", "LATE"]
    assert {record.entry_id for record in callbacks} == {"EARLY", "LATE"}
    assert records[0].best_xray_homolog_entity_id == "EARLY_Y_1"
    assert records[1].worst_xray_homolog_entity_id == "LATE_X_1"
    prepare.assert_called_once_with(
        skip_entry_ids=set(),
        progress_prefix="SOLUTION NMR X-ray RMSD extremes",
    )


def test_build_candidate_sets_keeps_empty_sets_sorts_and_calls_back(
    tmp_path: Path,
) -> None:
    """Return one shared result per homolog, including unsuccessful computations."""
    rmsd_builder = _rmsd_builder(tmp_path)
    work_items = _orchestration_work_items()
    candidate_sets = {
        "LATE": (_rmsd("LATE", "LATE_X_1", year=2004),),
        "EMPTY": (),
        "EARLY": (_rmsd("EARLY", "EARLY_X_1", year=1999),),
    }

    with (
        patch.object(
            rmsd_builder,
            "_prepare_work_items",
            return_value=work_items,
        ) as prepare,
        patch.object(
            rmsd_builder,
            "_compute_candidate_records",
            side_effect=lambda **kwargs: candidate_sets[kwargs["homolog"].entry_id],
        ),
    ):
        callbacks: list[
            tuple[
                SolutionNMRMonomerXrayHomologRecord,
                tuple[SolutionNMRMonomerXrayRmsdRecord, ...],
            ]
        ] = []
        results = rmsd_builder.build_candidate_sets(
            on_candidate_set=lambda homolog, candidates: callbacks.append(
                (homolog, candidates)
            )
        )

    assert [homolog.entry_id for homolog, _ in results] == [
        "EARLY",
        "EMPTY",
        "LATE",
    ]
    assert {homolog.entry_id for homolog, _ in callbacks} == {
        "EARLY",
        "EMPTY",
        "LATE",
    }
    empty_result = next(
        candidates for homolog, candidates in results if homolog.entry_id == "EMPTY"
    )
    assert empty_result == ()
    prepare.assert_called_once_with(
        skip_entry_ids=set(),
        progress_prefix="SOLUTION NMR unified X-ray RMSD",
    )


def test_ordinary_csv_wrapper_resumes_only_complete_matching_rows(
    tmp_path: Path,
) -> None:
    """Retain valid rows, retry outdated rows, and stream new ordinary RMSDs."""
    homolog_path = tmp_path / "homologs.csv"
    output_path = tmp_path / "ordinary.csv"
    output_path.touch()
    config = DatasetBuildConfig(max_workers=1)
    homolog_records = [_homolog("NEW", year=2010)]
    valid_late = _rmsd("ZOLD", "ZED_1", year=2005)
    valid_early = _rmsd("AOLD", "ALPHA_1", year=1998)
    outdated = replace(
        _rmsd("STALE", "STALE_1", year=2001),
        xray_core_start_seq_id=None,
    )
    wrong_identity = _rmsd("OTHER", "OTHER_1", identity=95)
    new_record = _rmsd("NEW", "NEWX_1", year=2010)

    def emit_new(*, skip_entry_ids, on_record):
        assert skip_entry_ids == {"AOLD", "ZOLD"}
        on_record(new_record)
        return [new_record]

    with (
        patch.object(builder_module, "_import_filtered_structures") as imported,
        patch.object(
            builder_module,
            "read_solution_nmr_monomer_xray_homolog_csv",
            return_value=homolog_records,
        ),
        patch.object(
            builder_module,
            "read_solution_nmr_monomer_xray_rmsd_csv",
            return_value=[valid_late, outdated, wrong_identity, valid_early],
        ),
        patch.object(
            builder_module,
            "SolutionNMRMonomerXrayRmsdBuilder",
        ) as builder_class,
    ):
        builder_class.return_value.build.side_effect = emit_new
        build_solution_nmr_monomer_xray_rmsd_to_csv(
            client=Mock(),
            config=config,
            homolog_input_path=homolog_path,
            output_path=output_path,
            cache_dir=tmp_path / "cache",
            rmsd_workers=2,
            sequence_identity_percent=100,
            resume=True,
            log_label="ordinary test",
        )

    imported.assert_called_once_with(homolog_path)
    assert [
        record.entry_id
        for record in read_solution_nmr_monomer_xray_rmsd_csv(output_path)
    ] == ["AOLD", "ZOLD", "NEW"]
    assert builder_class.call_args.kwargs["homolog_records"] == homolog_records
    assert builder_class.call_args.kwargs["sequence_identity_percent"] == 100


def test_extremes_csv_wrapper_resumes_only_complete_matching_rows(
    tmp_path: Path,
) -> None:
    """Retain complete extrema rows while retrying stale and wrong-cutoff rows."""
    homolog_path = tmp_path / "homologs.csv"
    output_path = tmp_path / "extremes.csv"
    output_path.touch()
    config = DatasetBuildConfig(max_workers=1)
    homolog_records = [_homolog("NEW", year=2010)]
    valid_late = _extremes("ZOLD", year=2005)
    valid_early = _extremes("AOLD", year=1998)
    outdated = replace(_extremes("STALE", year=2001), best_xray_homolog_entity_id="")
    wrong_identity = _extremes("OTHER", identity=95)
    new_record = _extremes("NEW", year=2010)

    def emit_new(*, skip_entry_ids, on_record):
        assert skip_entry_ids == {"AOLD", "ZOLD"}
        on_record(new_record)
        return [new_record]

    with (
        patch.object(builder_module, "_import_filtered_structures") as imported,
        patch.object(
            builder_module,
            "read_solution_nmr_monomer_xray_homolog_csv",
            return_value=homolog_records,
        ),
        patch.object(
            builder_module,
            "read_solution_nmr_monomer_xray_rmsd_extremes_csv",
            return_value=[valid_late, outdated, wrong_identity, valid_early],
        ),
        patch.object(
            builder_module,
            "SolutionNMRMonomerXrayRmsdBuilder",
        ) as builder_class,
    ):
        builder_class.return_value.build_extremes.side_effect = emit_new
        build_solution_nmr_monomer_xray_rmsd_extremes_to_csv(
            client=Mock(),
            config=config,
            homolog_input_path=homolog_path,
            output_path=output_path,
            cache_dir=tmp_path / "cache",
            rmsd_workers=2,
            sequence_identity_percent=100,
            resume=True,
            log_label="extremes test",
        )

    imported.assert_called_once_with(homolog_path)
    assert [
        record.entry_id
        for record in read_solution_nmr_monomer_xray_rmsd_extremes_csv(output_path)
    ] == ["AOLD", "ZOLD", "NEW"]
    assert builder_class.call_args.kwargs["homolog_records"] == homolog_records
    assert builder_class.call_args.kwargs["sequence_identity_percent"] == 100


@pytest.mark.parametrize(
    ("build_function", "builder_method", "new_record"),
    [
        (
            build_solution_nmr_monomer_xray_rmsd_to_csv,
            "build",
            _rmsd("NEW", "NEWX_1", year=2010),
        ),
        (
            build_solution_nmr_monomer_xray_rmsd_extremes_to_csv,
            "build_extremes",
            _extremes("NEW", year=2010),
        ),
    ],
)
def test_csv_wrappers_rebuild_without_loading_existing_rows(
    tmp_path: Path,
    build_function,
    builder_method: str,
    new_record,
) -> None:
    """A fresh build replaces an existing file without invoking a resume reader."""
    homolog_path = tmp_path / f"{builder_method}-homologs.csv"
    output_path = tmp_path / f"{builder_method}.csv"
    output_path.write_text("old contents\n", encoding="utf-8")
    config = DatasetBuildConfig(max_workers=1)

    def emit_new(*, skip_entry_ids, on_record):
        assert skip_entry_ids == set()
        on_record(new_record)
        return [new_record]

    with (
        patch.object(builder_module, "_import_filtered_structures"),
        patch.object(
            builder_module,
            "read_solution_nmr_monomer_xray_homolog_csv",
            return_value=[_homolog("NEW", year=2010)],
        ),
        patch.object(
            builder_module,
            "read_solution_nmr_monomer_xray_rmsd_csv",
        ) as read_ordinary,
        patch.object(
            builder_module,
            "read_solution_nmr_monomer_xray_rmsd_extremes_csv",
        ) as read_extremes,
        patch.object(
            builder_module,
            "SolutionNMRMonomerXrayRmsdBuilder",
        ) as builder_class,
    ):
        getattr(builder_class.return_value, builder_method).side_effect = emit_new
        build_function(
            client=Mock(),
            config=config,
            homolog_input_path=homolog_path,
            output_path=output_path,
            cache_dir=tmp_path / "cache",
            rmsd_workers=1,
            sequence_identity_percent=100,
            resume=False,
            log_label="fresh test",
        )

    read_ordinary.assert_not_called()
    read_extremes.assert_not_called()
    text = output_path.read_text(encoding="utf-8")
    assert "old contents" not in text
    assert "NEW" in text


@pytest.mark.parametrize(
    "build_function",
    [
        build_solution_nmr_monomer_xray_rmsd_to_csv,
        build_solution_nmr_monomer_xray_rmsd_extremes_to_csv,
    ],
)
def test_csv_wrappers_require_homolog_records(
    tmp_path: Path,
    build_function,
) -> None:
    """Fail with an actionable message before constructing an RMSD builder."""
    homolog_path = tmp_path / "missing-homologs.csv"
    with (
        patch.object(builder_module, "_import_filtered_structures"),
        patch.object(
            builder_module,
            "read_solution_nmr_monomer_xray_homolog_csv",
            return_value=[],
        ),
        patch.object(
            builder_module,
            "SolutionNMRMonomerXrayRmsdBuilder",
        ) as builder_class,
    ):
        with pytest.raises(SystemExit, match="No X-ray homolog records found"):
            build_function(
                client=Mock(),
                config=DatasetBuildConfig(),
                homolog_input_path=homolog_path,
                output_path=tmp_path / "output.csv",
                cache_dir=tmp_path / "cache",
                rmsd_workers=1,
                sequence_identity_percent=100,
                resume=False,
                log_label="missing input test",
            )

    builder_class.assert_not_called()
