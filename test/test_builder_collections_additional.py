"""Network-free coverage for the high-level dataset builders."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, call, patch

import numpy as np
import pytest

from src import pdb_dataset_builder as module


def _config(*, batch_size: int = 2) -> module.DatasetBuildConfig:
    """Return a deterministic single-worker configuration for builder tests."""
    return module.DatasetBuildConfig(
        graphql_batch_size=batch_size,
        max_workers=1,
    )


def _stride_record(
    entry_id: str,
    year: int,
) -> module.SolutionNMRMonomerStrideModeledFirstModelRecord:
    """Create a compact STRIDE record suitable for ordering assertions."""
    return module.SolutionNMRMonomerStrideModeledFirstModelRecord(
        entry_id=entry_id,
        year=year,
        chain_id="A",
        modeled_start_seq_id=1,
        modeled_end_seq_id=3,
        modeled_sequence_length=3,
        stride_alpha_helix_fraction=0.0,
        stride_3_10_helix_fraction=0.0,
        stride_pi_helix_fraction=0.0,
        stride_beta_strand_fraction=0.0,
        stride_isolated_beta_bridge_fraction=0.0,
        stride_turn_fraction=0.0,
        stride_coil_fraction=1.0,
        stride_secondary_structure_percent=0.0,
    )


def _precision_record(
    entry_id: str,
    year: int,
) -> module.SolutionNMRMonomerPrecisionRecord:
    """Create a precision record suitable for orchestration assertions."""
    return module.SolutionNMRMonomerPrecisionRecord(
        entry_id=entry_id,
        year=year,
        chain_id="A",
        core_start_seq_id=1,
        core_end_seq_id=3,
        n_models=2,
        n_ca_core_used=3,
        n_ca_core_raw=3,
        mean_rmsd_angstrom=0.25,
    )


def test_method_yearly_builder_batches_filters_dates_and_sorts() -> None:
    """Aggregate multiple method categories while ignoring malformed dates."""
    client = Mock()
    ids_by_method = {
        module.ExperimentalMethod.NMR: ["N2", "N1"],
        module.ExperimentalMethod.X_RAY: ["X2", "X1", "BAD"],
    }
    dates_by_id = {
        "N1": "2020-01-01",
        "N2": "2021-01-01",
        "X1": "2020-02-01",
        "X2": "2020-03-01",
        "BAD": "not-a-date",
    }
    client.fetch_entry_ids_for_method_category.side_effect = (
        lambda *, method, require_protein_entities: ids_by_method[method]
    )
    client.fetch_deposit_dates_for_ids.side_effect = lambda entry_ids: [
        dates_by_id[entry_id] for entry_id in entry_ids
    ]

    records = module.PDBMethodYearlyBuilder(client, _config()).build(
        [module.ExperimentalMethod.NMR, module.ExperimentalMethod.X_RAY]
    )

    assert records == [
        module.YearlyCountRecord(2020, "NMR", 1),
        module.YearlyCountRecord(2020, "X-ray", 2),
        module.YearlyCountRecord(2021, "NMR", 1),
    ]
    assert client.fetch_deposit_dates_for_ids.call_args_list == [
        call(["N2", "N1"]),
        call(["X2", "X1"]),
        call(["BAD"]),
    ]


def test_program_year_builder_fetches_year_batches_and_loads_programs(
    tmp_path: Path,
) -> None:
    """Merge year batches and cover successful and failed cached-PDB loading."""
    client = Mock()
    client.session = object()
    client.fetch_deposit_year_by_entry_id_for_ids.side_effect = lambda entry_ids: {
        entry_id: 2000 + int(entry_id[1:]) for entry_id in entry_ids
    }
    cache_dir = tmp_path / "program-cache"
    builder = module.SolutionNMRProgramYearlyBuilder(
        client=client,
        config=_config(),
        cache_dir=cache_dir,
    )

    assert cache_dir.is_dir()
    assert builder._fetch_entry_years(["E1", "E2", "E3"]) == {
        "E1": 2001,
        "E2": 2002,
        "E3": 2003,
    }
    assert client.fetch_deposit_year_by_entry_id_for_ids.call_args_list == [
        call(["E1", "E2"]),
        call(["E3"]),
    ]

    pdb_path = tmp_path / "entry.pdb"
    with (
        patch.object(
            module,
            "download_pdb_if_needed",
            side_effect=[pdb_path, RuntimeError("offline")],
        ) as download,
        patch.object(
            module,
            "extract_refinement_programs_from_pdb",
            return_value={"AMBER", "CNS"},
        ) as extract,
    ):
        assert builder._load_programs_for_entry("GOOD") == {"AMBER", "CNS"}
        assert builder._load_programs_for_entry("BROKEN") is None

    assert download.call_count == 2
    extract.assert_called_once_with(pdb_path)


def test_program_year_builder_handles_every_missing_value_and_counts(
    tmp_path: Path,
) -> None:
    """Exclude missing years/files/program fields and count the remaining labels."""
    client = Mock()
    builder = module.SolutionNMRProgramYearlyBuilder(client, _config(), tmp_path)
    entry_ids = ["MISSING_YEAR", "NO_FILE", "NO_PROGRAM", "OLD", "NEW"]
    years = {
        "NO_FILE": 2020,
        "NO_PROGRAM": 2020,
        "OLD": 2020,
        "NEW": 2021,
    }
    programs = {
        "NO_FILE": None,
        "NO_PROGRAM": set(),
        "OLD": {"AMBER"},
        "NEW": {"CNS", "AMBER"},
    }

    with (
        patch.object(module, "fetch_solution_nmr_entry_ids", return_value=entry_ids),
        patch.object(builder, "_fetch_entry_years", return_value=years),
        patch.object(
            builder,
            "_load_programs_for_entry",
            side_effect=lambda entry_id: programs[entry_id],
        ),
        patch.object(module, "_record_filtered_structure") as filtered,
    ):
        records = builder.build()

    assert records == [
        module.SolutionNMRProgramYearlyCountRecord(2020, "AMBER", 1),
        module.SolutionNMRProgramYearlyCountRecord(2021, "AMBER", 1),
        module.SolutionNMRProgramYearlyCountRecord(2021, "CNS", 1),
    ]
    filtered.assert_has_calls(
        [
            call("MISSING_YEAR", "deposit year is missing or invalid"),
            call("NO_FILE", "PDB file could not be loaded", year=2020),
            call("NO_PROGRAM", "refinement program is missing", year=2020),
        ],
        any_order=True,
    )
    assert filtered.call_count == 3


def test_program_year_builder_returns_early_when_search_is_empty(
    tmp_path: Path,
) -> None:
    """Avoid year and PDB work when the search returns no entries."""
    builder = module.SolutionNMRProgramYearlyBuilder(Mock(), _config(), tmp_path)
    with (
        patch.object(module, "fetch_solution_nmr_entry_ids", return_value=[]),
        patch.object(builder, "_fetch_entry_years") as fetch_years,
    ):
        assert builder.build() == []
    fetch_years.assert_not_called()


def test_program_cluster_loader_supports_local_and_remote_sources(
    tmp_path: Path,
) -> None:
    """Read valid cache entries, reject empty ones, and contain download errors."""
    valid_path = tmp_path / "VALID.pdb"
    valid_path.write_text("REMARK   3   PROGRAM     : AMBER\n", encoding="utf-8")
    (tmp_path / "EMPTY.pdb").write_text("", encoding="utf-8")

    local = module.SolutionNMRMonomerProgramClusterBuilder([], tmp_path, 0)
    assert local.max_workers == 1
    with patch.object(
        module,
        "extract_raw_refinement_program_text_from_pdb",
        return_value="AMBER",
    ) as extract:
        assert local._load_program_text("VALID") == "AMBER"
        assert local._load_program_text("EMPTY") == ""
        assert local._load_program_text("MISSING") == ""
    extract.assert_called_once_with(valid_path)

    client = Mock()
    client.session = object()
    remote = module.SolutionNMRMonomerProgramClusterBuilder(
        [], tmp_path, 1, client=client, config=_config()
    )
    with (
        patch.object(
            module,
            "download_pdb_if_needed",
            side_effect=[valid_path, RuntimeError("offline")],
        ),
        patch.object(
            module,
            "extract_raw_refinement_program_text_from_pdb",
            return_value="CNS",
        ),
    ):
        assert remote._load_program_text("REMOTE") == "CNS"
        assert remote._load_program_text("BROKEN") == ""


def test_program_cluster_builder_handles_empty_and_missing_program_text(
    tmp_path: Path,
) -> None:
    """Assign OTHER to missing text and emit zero rows for absent clusters."""
    empty = module.SolutionNMRMonomerProgramClusterBuilder([], tmp_path, 0)
    assert empty.build() == ([], [])

    quality_records = [
        module.SolutionNMRMonomerQualityRecord("LATE", 2021, 4.0, 2.0, 3.0),
        module.SolutionNMRMonomerQualityRecord("EARLY", 2020, 8.0, 6.0, 7.0),
        module.SolutionNMRMonomerQualityRecord("MISSING", 2022, 12.0, 10.0, 11.0),
    ]
    # A present, non-empty PDB can legitimately contain no program remark.
    (tmp_path / "EARLY.pdb").write_text("END\n", encoding="utf-8")
    builder = module.SolutionNMRMonomerProgramClusterBuilder(
        quality_records, tmp_path, 1
    )
    with patch.object(
        builder,
        "_load_program_text",
        side_effect=lambda entry_id: "AMBER" if entry_id == "LATE" else "",
    ):
        assignments, summaries = builder.build()

    assert [record.entry_id for record in assignments] == ["EARLY", "LATE", "MISSING"]
    assert all(record.cluster_score == 1.0 for record in assignments)
    assert assignments[0].cluster_name == "OTHER"
    assert not assignments[0].has_program_text
    assert assignments[1].cluster_name == "AMBER"
    assert assignments[1].has_program_text
    assert assignments[2].cluster_name == "OTHER"
    assert not assignments[2].has_program_text
    assert len(summaries) == 3 * len(module.PROGRAM_CLUSTER_DEFINITIONS)

    by_year_and_name = {
        (record.year, record.cluster_name): record for record in summaries
    }
    assert by_year_and_name[(2020, "OTHER")].structure_count == 1.0
    assert by_year_and_name[(2020, "OTHER")].avg_clashscore == 8.0
    assert by_year_and_name[(2021, "AMBER")].avg_clashscore == 4.0
    assert by_year_and_name[(2022, "OTHER")].avg_clashscore == 12.0
    amber = by_year_and_name[(2020, "AMBER")]
    assert amber.structure_count == 0
    assert amber.avg_clashscore is None
    assert amber.avg_ramachandran_outliers_percent is None
    assert amber.avg_sidechain_outliers_percent is None


def test_membrane_builder_deduplicates_counts_and_filters_by_method() -> None:
    """Count unique membrane IDs and report excluded method entries with years."""
    client = Mock()
    client.fetch_entry_ids_for_membrane_annotations.return_value = [
        "M2",
        "M1",
        "M2",
        "M3",
        "M_BAD",
    ]
    date_by_id = {
        "M1": "2020-01-01",
        "M2": "2019-01-01",
        "M3": "2019-06-01",
        "M_BAD": "bad-date",
    }
    client.fetch_deposit_dates_for_ids.side_effect = lambda entry_ids: [
        date_by_id[entry_id] for entry_id in entry_ids
    ]
    ids_by_method = {
        module.ExperimentalMethod.X_RAY: ["M3", "OUT2", "M1", "OUT1"],
        module.ExperimentalMethod.NMR: ["M2", "OUT3"],
    }
    client.fetch_entry_ids_for_method_category.side_effect = (
        lambda *, method, require_protein_entities: ids_by_method[method]
    )
    excluded_years = {"OUT1": 2018, "OUT3": 2022}
    client.fetch_deposit_year_by_entry_id_for_ids.side_effect = lambda entry_ids: {
        entry_id: excluded_years[entry_id]
        for entry_id in entry_ids
        if entry_id in excluded_years
    }
    builder = module.MembraneProteinYearlyBuilder(client, _config())

    assert builder.build() == [
        module.MembraneYearlyCountRecord(2019, 2),
        module.MembraneYearlyCountRecord(2020, 1),
    ]

    with patch.object(module, "_record_filtered_structure") as filtered:
        by_method = builder.build_by_method(
            [module.ExperimentalMethod.X_RAY, module.ExperimentalMethod.NMR]
        )

    assert by_method == [
        module.YearlyCountRecord(2019, "NMR", 1),
        module.YearlyCountRecord(2019, "X-ray", 1),
        module.YearlyCountRecord(2020, "X-ray", 1),
    ]
    filtered.assert_has_calls(
        [
            call(
                "OUT1",
                "entry has no supported membrane-protein annotation for X-ray",
                year=2018,
            ),
            call(
                "OUT2",
                "entry has no supported membrane-protein annotation for X-ray",
                year=None,
            ),
            call(
                "OUT3",
                "entry has no supported membrane-protein annotation for NMR",
                year=2022,
            ),
        ]
    )
    assert client.fetch_entry_ids_for_membrane_annotations.call_count == 2
    client.fetch_entry_ids_for_membrane_annotations.assert_called_with(
        module.MEMBRANE_ANNOTATION_TYPES
    )


@pytest.mark.parametrize(
    ("builder_type", "fetch_name", "records", "require_protein_entities"),
    [
        (
            module.SolutionNMRWeightBuilder,
            "fetch_solution_nmr_weight_records_for_ids",
            {
                "A": module.SolutionNMRWeightRecord("A", 2022, 10.0),
                "B": module.SolutionNMRWeightRecord("B", 2020, 20.0),
            },
            True,
        ),
        (
            module.SolutionNMRMonomerExperimentsBuilder,
            "fetch_solution_nmr_monomer_experiment_records_for_ids",
            {
                "A": module.SolutionNMRMonomerExperimentsRecord("A", 2022, ("HSQC",)),
                "B": module.SolutionNMRMonomerExperimentsRecord("B", 2020, ("NOESY",)),
            },
            False,
        ),
        (
            module.SolutionNMRMonomerQualityBuilder,
            "fetch_solution_nmr_monomer_quality_records_for_ids",
            {
                "A": module.SolutionNMRMonomerQualityRecord("A", 2022, 1.0, 2.0, 3.0),
                "B": module.SolutionNMRMonomerQualityRecord("B", 2020, 4.0, 5.0, 6.0),
            },
            False,
        ),
    ],
    ids=lambda value: getattr(value, "__name__", None),
)
def test_simple_nmr_builders_batch_deduplicate_and_sort(
    builder_type: type,
    fetch_name: str,
    records: dict[str, object],
    require_protein_entities: bool,
) -> None:
    """Exercise the common batched collection contract of three builders."""
    client = Mock()
    client.fetch_entry_ids_for_method.return_value = ["B", "A", "C", "A"]
    fetch_records = getattr(client, fetch_name)
    fetch_records.side_effect = lambda entry_ids: [
        records[entry_id] for entry_id in reversed(entry_ids) if entry_id in records
    ]

    result = builder_type(client, _config()).build()

    assert result == [records["B"], records["A"]]
    client.fetch_entry_ids_for_method.assert_called_once_with(
        method_label="SOLUTION NMR",
        query_value="SOLUTION NMR",
        require_protein_entities=require_protein_entities,
    )
    assert fetch_records.call_args_list == [call(["A", "B"]), call(["C"])]


def test_stride_collection_builder_batches_sorts_and_handles_empty_search(
    tmp_path: Path,
) -> None:
    """Stream STRIDE records with the configured cache paths and stable ordering."""
    client = Mock()
    client.fetch_entry_ids_for_method.return_value = ["B", "A", "C", "A"]
    records = {
        "A": _stride_record("A", 2022),
        "B": _stride_record("B", 2020),
    }
    client.fetch_solution_nmr_monomer_stride_modeled_first_model_records_for_ids.side_effect = (
        lambda *, entry_ids, **kwargs: [
            records[entry_id] for entry_id in reversed(entry_ids) if entry_id in records
        ]
    )
    pdb_cache = tmp_path / "pdb"
    stride_cache = tmp_path / "stride"
    builder = module.SolutionNMRMonomerStrideModeledFirstModelBuilder(
        client=client,
        config=_config(),
        stride_executable="/local/stride",
        cache_dir=pdb_cache,
        stride_cache_dir=stride_cache,
    )

    assert builder.build() == [records["B"], records["A"]]
    assert pdb_cache.is_dir()
    assert stride_cache.is_dir()
    assert (
        client.fetch_solution_nmr_monomer_stride_modeled_first_model_records_for_ids.call_args_list
        == [
            call(
                entry_ids=["A", "B"],
                stride_executable="/local/stride",
                pdb_cache_dir=pdb_cache,
                stride_cache_dir=stride_cache,
            ),
            call(
                entry_ids=["C"],
                stride_executable="/local/stride",
                pdb_cache_dir=pdb_cache,
                stride_cache_dir=stride_cache,
            ),
        ]
    )

    empty_client = Mock()
    empty_client.fetch_entry_ids_for_method.return_value = []
    empty_builder = module.SolutionNMRMonomerStrideModeledFirstModelBuilder(
        client=empty_client,
        config=_config(),
        stride_executable="/local/stride",
        cache_dir=pdb_cache,
        stride_cache_dir=stride_cache,
    )
    assert list(empty_builder.iter_batches()) == []
    empty_client.fetch_solution_nmr_monomer_stride_modeled_first_model_records_for_ids.assert_not_called()


def test_precision_computation_rejects_small_intersection_and_counts_raw_fallback(
    tmp_path: Path,
) -> None:
    """Cover model-count/core-size rejection and missing raw-statistics fallback."""
    xyz = {
        1: np.array([0.0, 0.0, 0.0]),
        2: np.array([1.0, 0.0, 0.0]),
        3: np.array([0.0, 1.0, 0.0]),
        4: np.array([0.0, 0.0, 1.0]),
    }
    one_model = ([{1: xyz[1], 2: xyz[2], 3: xyz[3]}], [{}])
    too_small = (
        [
            {1: xyz[1], 2: xyz[2], 3: xyz[3]},
            {1: xyz[1], 2: xyz[2], 4: xyz[4]},
        ],
        [{}, {}],
    )
    complete = (
        [
            {1: xyz[1], 2: xyz[2], 3: xyz[3]},
            {1: xyz[1], 2: xyz[2], 3: xyz[3]},
        ],
        [],
    )

    with patch.object(
        module,
        "parse_models_ca_coords_with_stats",
        side_effect=[one_model, too_small, complete],
    ):
        one_result, one_reason = (
            module.SolutionNMRMonomerPrecisionBuilder._compute_mean_rmsd_to_average(
                tmp_path / "unused.pdb", "A", 1, 3
            )
        )
        result, reason = (
            module.SolutionNMRMonomerPrecisionBuilder._compute_mean_rmsd_to_average(
                tmp_path / "unused.pdb", "A", 1, 4
            )
        )
        complete_result, complete_reason = (
            module.SolutionNMRMonomerPrecisionBuilder._compute_mean_rmsd_to_average(
                tmp_path / "unused.pdb", "A", 1, 3
            )
        )

    assert one_result is None
    assert one_reason == "fewer than 2 coordinate models in core range (found 1)"
    assert result is None
    assert (
        reason
        == "fewer than 3 CA residues common to all models in core range (found 2)"
    )
    assert complete_reason is None
    assert complete_result is not None
    assert complete_result[:3] == (2, 3, 3)
    assert complete_result[3] == pytest.approx(0.0)


def test_precision_builder_delegates_download_and_maps_core_result(
    tmp_path: Path,
) -> None:
    """Forward cache settings, honor parsed chain IDs, and record rejected cores."""
    client = Mock()
    client.session = object()
    builder = module.SolutionNMRMonomerPrecisionBuilder(
        client=client,
        config=_config(),
        cache_dir=tmp_path / "cache",
        precision_workers=0,
    )
    assert builder.precision_workers == 1
    pdb_path = tmp_path / "cached.pdb"
    with patch.object(
        module, "download_pdb_if_needed", return_value=pdb_path
    ) as download:
        assert builder._download_pdb_if_needed("1ABC") == pdb_path
    download.assert_called_once_with(
        session=client.session,
        config=builder.config,
        cache_dir=builder.cache_dir,
        entry_id="1ABC",
    )

    with patch.object(
        builder,
        "_compute_mean_rmsd_to_average",
        return_value=((4, 8, 9, 1.25), None),
    ) as compute:
        record = builder._build_record_from_core_range(
            pdb_path=pdb_path,
            entry_id="1ABC",
            year=2020,
            chain_id="LONG",
            core_start_seq_id=10,
            core_end_seq_id=20,
            parsed_chain_id="A",
        )
    assert record == module.SolutionNMRMonomerPrecisionRecord(
        "1ABC", 2020, "LONG", 10, 20, 4, 8, 9, 1.25
    )
    compute.assert_called_once_with(
        pdb_path=pdb_path,
        chain_id="A",
        start_seq_id=10,
        end_seq_id=20,
    )

    with (
        patch.object(
            builder,
            "_compute_mean_rmsd_to_average",
            return_value=(None, "bad core"),
        ),
        patch.object(module, "_record_filtered_structure") as filtered,
    ):
        assert (
            builder._build_record_from_core_range(pdb_path, "BAD", 2021, "B", 1, 2)
            is None
        )
    filtered.assert_called_once_with(
        "BAD",
        "precision calculation rejected the structural core: bad core",
        year=2021,
    )


def test_precision_stride_seed_handles_ineligible_inputs_and_failures(
    tmp_path: Path,
) -> None:
    """Record distinct skip reasons for missing atoms, cores, and PDB failures."""
    client = Mock()
    builder = module.SolutionNMRMonomerPrecisionStrideModeledFirstModelBuilder(
        client=client,
        config=_config(),
        cache_dir=tmp_path / "pdb",
        precision_workers=1,
        stride_executable="stride",
        stride_cache_dir=tmp_path / "stride",
    )
    seed = module.SolutionNMRMonomerModeledFirstModelSeedRecord("ENTRY", 2020, "LONG")
    pdb_path = tmp_path / "entry.pdb"

    with (
        patch.object(builder, "_download_pdb_if_needed", return_value=pdb_path),
        patch.object(module, "load_cached_chain_id_map", return_value={"LONG": "A"}),
        patch.object(
            module, "parse_first_model_modeled_ca_auth_seq_ids", return_value=[]
        ),
        patch.object(module, "_record_filtered_structure") as filtered,
    ):
        assert builder._compute_record_from_seed(seed) is None
    filtered.assert_called_once_with(
        "ENTRY", "no usable first-model modeled CA residues", year=2020
    )

    with (
        patch.object(builder, "_download_pdb_if_needed", return_value=pdb_path),
        patch.object(module, "load_cached_chain_id_map", return_value={"LONG": "A"}),
        patch.object(
            module,
            "parse_first_model_modeled_ca_auth_seq_ids",
            return_value=[10, 11, 12],
        ),
        patch.object(
            module,
            "compute_stride_core_range_for_modeled_auth_seq_ids_in_first_model",
            return_value=None,
        ),
        patch.object(module, "_record_filtered_structure") as filtered,
    ):
        assert builder._compute_record_from_seed(seed) is None
    filtered.assert_called_once_with(
        "ENTRY", "STRIDE found no modeled core residues", year=2020
    )

    with (
        patch.object(
            builder, "_download_pdb_if_needed", side_effect=RuntimeError("offline")
        ),
        patch.object(module, "_record_filtered_structure") as filtered,
    ):
        assert builder._compute_record_from_seed(seed) is None
    filtered.assert_called_once_with(
        "ENTRY", "precision calculation failed: offline", year=2020
    )


def test_precision_stride_seed_uses_mapped_chain_and_stride_core(
    tmp_path: Path,
) -> None:
    """Pass the mapped PDB chain and computed STRIDE range into precision work."""
    builder = module.SolutionNMRMonomerPrecisionStrideModeledFirstModelBuilder(
        client=Mock(),
        config=_config(),
        cache_dir=tmp_path / "pdb",
        precision_workers=1,
        stride_executable="stride",
        stride_cache_dir=tmp_path / "stride",
    )
    seed = module.SolutionNMRMonomerModeledFirstModelSeedRecord("ENTRY", 2020, "LONG")
    pdb_path = tmp_path / "entry.pdb"
    expected = _precision_record("ENTRY", 2020)

    with (
        patch.object(builder, "_download_pdb_if_needed", return_value=pdb_path),
        patch.object(module, "load_cached_chain_id_map", return_value={"LONG": "A"}),
        patch.object(
            module,
            "parse_first_model_modeled_ca_auth_seq_ids",
            return_value=[10, 11, 12],
        ) as parse_modeled,
        patch.object(
            module,
            "compute_stride_core_range_for_modeled_auth_seq_ids_in_first_model",
            return_value=(10, 12),
        ) as compute_core,
        patch.object(
            builder, "_build_record_from_core_range", return_value=expected
        ) as build_record,
    ):
        assert builder._compute_record_from_seed(seed) == expected

    parse_modeled.assert_called_once_with(pdb_path=pdb_path, chain_id="A")
    compute_core.assert_called_once_with(
        pdb_path=pdb_path,
        entry_id="ENTRY",
        chain_id="A",
        modeled_auth_seq_ids=[10, 11, 12],
        stride_executable="stride",
        stride_cache_dir=builder.stride_cache_dir,
    )
    build_record.assert_called_once_with(
        pdb_path=pdb_path,
        entry_id="ENTRY",
        year=2020,
        chain_id="LONG",
        core_start_seq_id=10,
        core_end_seq_id=12,
        parsed_chain_id="A",
    )


def test_precision_stride_builder_batches_skips_callbacks_and_sorts(
    tmp_path: Path,
) -> None:
    """Batch seed metadata, skip resumed entries, and stream successful records."""
    client = Mock()
    client.fetch_entry_ids_for_method.return_value = ["SKIP", "C", "A", "B"]
    seeds = {
        "SKIP": module.SolutionNMRMonomerModeledFirstModelSeedRecord("SKIP", 2019, "A"),
        "B": module.SolutionNMRMonomerModeledFirstModelSeedRecord("B", 2020, "A"),
        "C": module.SolutionNMRMonomerModeledFirstModelSeedRecord("C", 2021, "A"),
        "A": module.SolutionNMRMonomerModeledFirstModelSeedRecord("A", 2022, "A"),
    }
    client.fetch_solution_nmr_monomer_modeled_first_model_seed_records_for_ids.side_effect = (
        lambda entry_ids: [seeds[entry_id] for entry_id in reversed(entry_ids)]
    )
    expected = {
        "B": _precision_record("B", 2020),
        "A": _precision_record("A", 2022),
    }
    callback = Mock()
    builder = module.SolutionNMRMonomerPrecisionStrideModeledFirstModelBuilder(
        client=client,
        config=_config(),
        cache_dir=tmp_path / "pdb",
        precision_workers=1,
        stride_executable="stride",
        stride_cache_dir=tmp_path / "stride",
    )
    with patch.object(
        builder,
        "_compute_record_from_seed",
        side_effect=lambda seed: expected.get(seed.entry_id),
    ) as compute:
        records = builder.build(skip_entry_ids={"SKIP"}, on_record=callback)

    assert records == [expected["B"], expected["A"]]
    assert {args.args[0].entry_id for args in compute.call_args_list} == {"A", "B", "C"}
    callback.assert_has_calls(
        [call(expected["B"]), call(expected["A"])], any_order=True
    )
    assert callback.call_count == 2
    assert (
        client.fetch_solution_nmr_monomer_modeled_first_model_seed_records_for_ids.call_args_list
        == [call(["A", "B"]), call(["C", "SKIP"])]
    )


def test_precision_stride_builder_accepts_default_skip_and_callback_options(
    tmp_path: Path,
) -> None:
    """Return successful records when resume filtering and callbacks are omitted."""
    client = Mock()
    client.fetch_entry_ids_for_method.return_value = ["ONLY"]
    seed = module.SolutionNMRMonomerModeledFirstModelSeedRecord("ONLY", 2020, "A")
    client.fetch_solution_nmr_monomer_modeled_first_model_seed_records_for_ids.return_value = [
        seed
    ]
    expected = _precision_record("ONLY", 2020)
    builder = module.SolutionNMRMonomerPrecisionStrideModeledFirstModelBuilder(
        client=client,
        config=_config(),
        cache_dir=tmp_path / "pdb",
        precision_workers=1,
        stride_executable="stride",
        stride_cache_dir=tmp_path / "stride",
    )

    with patch.object(builder, "_compute_record_from_seed", return_value=expected):
        assert builder.build() == [expected]
