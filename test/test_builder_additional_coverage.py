"""Additional unit coverage for low-level dataset-builder helpers."""

from __future__ import annotations

import gzip
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import requests

import src.pdb_dataset_builder as builder


def _ca_line(
    *,
    record: str = "ATOM",
    serial: int = 1,
    atom_name: str = "CA",
    alt_loc: str = "",
    resname: str = "ALA",
    chain_id: str = "A",
    resid: int = 1,
    insertion_code: str = "",
    x: float = 1.0,
    y: float = 2.0,
    z: float = 3.0,
    occupancy: float = 1.0,
) -> str:
    """Return one fixed-column coordinate record."""
    return (
        f"{record:<6}{serial:5d} {atom_name:>4}{alt_loc:1}{resname:>3} "
        f"{chain_id:1}{resid:4d}{insertion_code:1}   "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{occupancy:6.2f}{20.0:6.2f}"
        "           C\n"
    )


def _modres_line(
    *,
    resname: str = "MSE",
    chain_id: str = "A",
    resid: int = 5,
    insertion_code: str = "",
    standard_resname: str = "MET",
) -> str:
    """Return one fixed-column MODRES record."""
    chars = [" "] * 80
    chars[0:6] = "MODRES"
    chars[7:11] = "1ABC"
    chars[12:15] = f"{resname:>3}"
    chars[16] = chain_id
    chars[18:22] = f"{resid:4d}"
    chars[22] = insertion_code or " "
    chars[24:27] = f"{standard_resname:>3}"
    return "".join(chars) + "\n"


class _ChunkedResponse:
    """Small streaming-response double used without network access."""

    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks
        self.headers: dict[str, str] = {}

    def iter_content(self, chunk_size: int):
        assert chunk_size == 1024 * 1024
        yield from self.chunks


def test_http_server_error_finds_nested_5xx_and_rejects_other_errors() -> None:
    server_error = requests.HTTPError(
        "unavailable", response=SimpleNamespace(status_code=503)
    )
    wrapper = RuntimeError("wrapper")
    wrapper.__cause__ = server_error

    assert builder._is_http_server_error(wrapper)
    assert not builder._is_http_server_error(
        requests.HTTPError("missing", response=SimpleNamespace(status_code=404))
    )
    assert not builder._is_http_server_error(ValueError("not HTTP"))


def test_error_types_and_chain_subset_selection_preserve_context() -> None:
    ineligible = builder.NMRHomologyQueryIneligibleError("1abc", "no core")
    hetatm = builder.NMRCoreContainsHetatmError("2def")
    selected_chain = object()
    selector = builder.ChainSubsetSelect({id(selected_chain)})

    assert (ineligible.entry_id, ineligible.reason, str(ineligible)) == (
        "1abc",
        "no core",
        "1abc: no core",
    )
    assert hetatm.entry_id == "2def"
    assert "HETATM" in str(hetatm)
    assert selector.accept_chain(selected_chain)
    assert not selector.accept_chain(object())


def test_experimental_method_properties_include_combined_nmr_method_set() -> None:
    assert builder.ExperimentalMethod.X_RAY.label == "X-ray"
    assert builder.ExperimentalMethod.CRYO_EM.query_values == ("ELECTRON MICROSCOPY",)
    assert builder.ExperimentalMethod.X_RAY.exact_method_sets == (
        ("X-RAY DIFFRACTION",),
    )
    assert builder.ExperimentalMethod.NMR.exact_method_sets == (
        ("SOLUTION NMR",),
        ("SOLID-STATE NMR",),
        ("SOLID-STATE NMR", "SOLUTION NMR"),
    )


def test_batch_helpers_handle_empty_work_and_collect_all_results() -> None:
    assert list(builder.chunked(["a", "b", "c"], 2)) == [["a", "b"], ["c"]]
    assert list(builder.chunked([], 2)) == []
    fetch = Mock(side_effect=lambda batch: tuple(item.upper() for item in batch))

    results = builder.collect_batch_results([["a"], ["b", "c"]], 2, fetch, "fixture")

    assert sorted(results) == [("A",), ("B", "C")]
    assert fetch.call_count == 2
    assert builder.collect_batch_results([], 2, fetch, "empty") == []
    assert fetch.call_count == 2


def test_missing_response_entries_are_deduplicated_and_recorded() -> None:
    with patch.object(builder, "_record_filtered_structure") as record:
        builder._record_entries_missing_from_response(
            ["3CCC", "1AAA", "2BBB", "1AAA"],
            [None, {}, {"rcsb_id": "2BBB"}],
        )

    assert [call.args for call in record.call_args_list] == [
        ("1AAA", "entry metadata missing from RCSB GraphQL response"),
        ("3CCC", "entry metadata missing from RCSB GraphQL response"),
    ]


def test_fetch_solution_nmr_entry_ids_sorts_deduplicates_and_forwards_flag() -> None:
    client = SimpleNamespace(
        fetch_entry_ids_for_method=Mock(return_value=["2DEF", "1ABC", "2DEF"])
    )

    assert builder.fetch_solution_nmr_entry_ids(
        client, "test", require_protein_entities=True
    ) == ["1ABC", "2DEF"]
    client.fetch_entry_ids_for_method.assert_called_once_with(
        method_label="SOLUTION NMR",
        query_value="SOLUTION NMR",
        require_protein_entities=True,
    )


def test_resolve_stride_executable_obeys_explicit_path_and_fallbacks(
    tmp_path: Path,
) -> None:
    explicit = tmp_path / "stride-explicit"
    explicit.touch()
    explicit.chmod(0o755)
    local = tmp_path / "stride-local"
    local.touch()
    local.chmod(0o755)

    with patch.object(builder.shutil, "which") as which:
        assert builder.resolve_stride_executable(str(explicit)) == str(explicit)
        which.assert_not_called()

    with patch.object(builder.shutil, "which", return_value="/bin/stride"):
        assert builder.resolve_stride_executable("  ") == "/bin/stride"

    with (
        patch.object(builder.shutil, "which", return_value=None),
        patch.object(builder, "LOCAL_STRIDE_CANDIDATE", local),
    ):
        assert builder.resolve_stride_executable("") == str(local.resolve())

    with (
        patch.object(builder.shutil, "which", return_value=None),
        patch.object(builder, "LOCAL_STRIDE_CANDIDATE", tmp_path / "absent"),
    ):
        assert builder.resolve_stride_executable("") is None
        assert builder.resolve_stride_executable(str(tmp_path / "missing")) is None


def test_cache_metadata_loader_rejects_missing_corrupt_and_old_payloads(
    tmp_path: Path,
) -> None:
    pdb_path = tmp_path / "1ABC.pdb"
    sidecar = builder._pdb_cache_metadata_path(pdb_path)
    assert sidecar.name == "1ABC.pdb.cache.json"
    assert builder._load_pdb_cache_metadata(pdb_path) is None

    sidecar.write_text("not json", encoding="utf-8")
    assert builder._load_pdb_cache_metadata(pdb_path) is None

    sidecar.write_text("[]", encoding="utf-8")
    assert builder._load_pdb_cache_metadata(pdb_path) is None

    sidecar.write_text(json.dumps({"schema_version": 999}), encoding="utf-8")
    assert builder._load_pdb_cache_metadata(pdb_path) is None

    payload = {
        "schema_version": builder.PDB_CACHE_METADATA_SCHEMA_VERSION,
        "cache_revision": "rev-1",
    }
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    assert builder._load_pdb_cache_metadata(pdb_path) == payload
    assert builder._cache_revision(payload) == "rev-1"
    assert builder._cache_revision({"cache_revision": ""}) is None
    assert builder._cache_revision({"cache_revision": 1}) is None
    assert builder._cache_revision(None) is None


def test_cached_coordinate_metadata_must_match_nonempty_file_stats(
    tmp_path: Path,
) -> None:
    pdb_path = tmp_path / "1ABC.pdb"
    assert not builder._cached_pdb_matches_metadata(pdb_path, {})

    pdb_path.touch()
    assert not builder._cached_pdb_matches_metadata(pdb_path, {})

    pdb_path.write_bytes(b"ATOM\n")
    stat = pdb_path.stat()
    valid = {
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": "a" * 64,
    }
    assert builder._cached_pdb_matches_metadata(pdb_path, valid)
    assert not builder._cached_pdb_matches_metadata(
        pdb_path, {**valid, "mtime_ns": stat.st_mtime_ns + 1}
    )
    assert not builder._cached_pdb_matches_metadata(pdb_path, {**valid, "sha256": ""})


def test_cache_validation_freshness_handles_invalid_naive_and_future_dates() -> None:
    now = datetime.now(timezone.utc)
    assert not builder._pdb_cache_validation_is_fresh({}, 1)
    assert not builder._pdb_cache_validation_is_fresh(
        {"validated_at": now.isoformat()}, 0
    )
    assert not builder._pdb_cache_validation_is_fresh({"validated_at": "not-a-date"}, 1)
    assert builder._pdb_cache_validation_is_fresh(
        {"validated_at": (now - timedelta(minutes=1)).replace(tzinfo=None).isoformat()},
        1,
    )
    assert not builder._pdb_cache_validation_is_fresh(
        {"validated_at": (now + timedelta(hours=1)).isoformat()}, 24
    )
    assert not builder._pdb_cache_validation_is_fresh(
        {"validated_at": (now - timedelta(hours=2)).isoformat()}, 1
    )


def test_response_helpers_filter_empty_chunks_and_never_leak_close_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    response = _ChunkedResponse([b"", b"first", b"", b"second"])
    assert list(builder._response_chunks(response)) == [b"first", b"second"]

    closable = SimpleNamespace(close=Mock())
    builder._close_http_response(closable)
    closable.close.assert_called_once_with()
    builder._close_http_response(SimpleNamespace(close=None))
    builder._close_http_response(None)

    def fail_close() -> None:
        raise RuntimeError("close failed")

    caplog.set_level("DEBUG", logger=builder.__name__)
    builder._close_http_response(SimpleNamespace(close=fail_close))
    assert "Could not close an HTTP response" in caplog.text


@pytest.mark.parametrize("compressed", [False, True])
def test_atomic_coordinate_install_streams_and_hashes_payload(
    tmp_path: Path, compressed: bool
) -> None:
    body = b"HEADER fixture\nATOM fixture\n"
    wire_body = gzip.compress(body) if compressed else body
    response = _ChunkedResponse([wire_body[:5], b"", wire_body[5:]])
    pdb_path = tmp_path / "cache" / "1ABC.pdb"

    digest, size, mtime_ns = builder._atomic_install_pdb_response(
        response, pdb_path, compressed
    )

    assert pdb_path.read_bytes() == body
    assert digest == hashlib.sha256(body).hexdigest()
    assert size == len(body)
    assert mtime_ns == pdb_path.stat().st_mtime_ns
    assert not list(pdb_path.parent.glob("*.download"))
    assert not list(pdb_path.parent.glob("*.pdb.tmp"))


def test_atomic_coordinate_install_rejects_empty_payload_and_cleans_temps(
    tmp_path: Path,
) -> None:
    pdb_path = tmp_path / "1ABC.pdb"
    with pytest.raises(RuntimeError, match="Downloaded empty coordinate file"):
        builder._atomic_install_pdb_response(_ChunkedResponse([b""]), pdb_path, False)

    assert not pdb_path.exists()
    assert sorted(tmp_path.iterdir()) == []


def test_coordinate_download_sources_and_cached_source_priority_are_stable() -> None:
    pdb_sources = builder._pdb_download_sources("1AbC")
    cif_sources = builder._mmcif_download_sources("1AbC")
    assert pdb_sources[0] == ("https://files.rcsb.org/download/1ABC.pdb.gz", True)
    assert "/ab/pdb1abc.ent.gz" in pdb_sources[1][0]
    assert cif_sources[0] == ("https://files.rcsb.org/download/1ABC.cif.gz", True)
    assert "/ab/1abc.cif.gz" in cif_sources[1][0]

    assert builder._prioritize_cached_source(pdb_sources, None) is pdb_sources
    assert builder._prioritize_cached_source(pdb_sources, "unknown") is pdb_sources
    prioritized = builder._prioritize_cached_source(pdb_sources, pdb_sources[2][0])
    assert prioritized[0] == pdb_sources[2]
    assert len(prioritized) == len(pdb_sources)
    assert len(set(prioritized)) == len(prioritized)


def test_chain_id_coercion_handles_collisions_and_duplicate_models() -> None:
    first_model = [
        SimpleNamespace(id="A"),
        SimpleNamespace(id="Alpha"),
        SimpleNamespace(id="Beta"),
    ]
    second_model = [SimpleNamespace(id="A"), SimpleNamespace(id="Alpha")]
    structure = [first_model, second_model]

    changed = builder._coerce_structure_chain_ids_for_pdbio(structure)

    assert changed == {"Alpha": "B", "Beta": "C"}
    assert [[chain.id for chain in model] for model in structure] == [
        ["A", "B", "C"],
        ["A", "B"],
    ]


def test_chain_id_map_is_applied_without_swap_collisions() -> None:
    chains = [SimpleNamespace(id="A"), SimpleNamespace(id="B")]
    builder._apply_chain_id_map_without_transient_conflicts(
        [chains], {"A": "B", "B": "A"}
    )
    assert [chain.id for chain in chains] == ["B", "A"]


def test_selected_chain_coercion_leaves_unselected_chains_untouched() -> None:
    selected_a = SimpleNamespace(id="A")
    selected_long = SimpleNamespace(id="Alpha")
    unselected = SimpleNamespace(id="other")
    mapping, selected_object_ids = (
        builder._coerce_selected_structure_chain_ids_for_pdbio(
            [[selected_long, unselected, selected_a]], {"A", "Alpha", "missing"}
        )
    )

    assert mapping == {"A": "A", "Alpha": "B"}
    assert selected_object_ids == {id(selected_a), id(selected_long)}
    assert unselected.id == "other"
    assert selected_long.id == "B"


@pytest.mark.parametrize("selected_only", [False, True])
def test_chain_id_coercion_rejects_exhausted_pdb_id_pool(selected_only: bool) -> None:
    structure = [[SimpleNamespace(id="A"), SimpleNamespace(id="Alpha")]]
    with patch.object(builder, "PDB_CHAIN_ID_POOL", ""):
        with pytest.raises(RuntimeError, match="Too many"):
            if selected_only:
                builder._coerce_selected_structure_chain_ids_for_pdbio(
                    structure, {"A", "Alpha"}
                )
            else:
                builder._coerce_structure_chain_ids_for_pdbio(structure)


def test_chain_subset_cache_key_is_order_independent_and_empty_selection_fails(
    tmp_path: Path,
) -> None:
    first = builder._chain_subset_cache_stem("1ABC", ["B", "A"])
    second = builder._chain_subset_cache_stem("1ABC", ["A", "B"])
    assert first == second
    assert first.startswith("1ABC.chains_")

    with pytest.raises(RuntimeError, match="No chain IDs selected"):
        builder.download_pdb_chain_subset_if_needed(
            session=MagicMock(),
            config=builder.DatasetBuildConfig(),
            cache_dir=tmp_path,
            entry_id="1ABC",
            chain_ids=["", ""],
        )


def test_valid_cached_chain_subset_requires_complete_source_bound_mapping(
    tmp_path: Path,
) -> None:
    subset_path = tmp_path / "subset.pdb"
    cif_path = tmp_path / "source.cif"
    subset_metadata = {
        "source_cif_sha256": "source-sha",
        "chain_ids": ["A", "long"],
        "chain_id_map": {"A": "A", "long": "B"},
    }
    cif_metadata = {"sha256": "source-sha"}

    def metadata(path: Path):
        return subset_metadata if path == subset_path else cif_metadata

    with (
        patch.object(builder, "_load_pdb_cache_metadata", side_effect=metadata),
        patch.object(builder, "_cached_pdb_matches_metadata", return_value=True),
    ):
        assert builder._load_valid_cached_chain_subset(
            subset_path=subset_path,
            cif_path=cif_path,
            selected_chain_ids={"long", "A"},
        ) == (subset_path, {"A": "A", "long": "B"})

        subset_metadata["chain_id_map"] = {"A": "A"}
        assert (
            builder._load_valid_cached_chain_subset(
                subset_path=subset_path,
                cif_path=cif_path,
                selected_chain_ids={"long", "A"},
            )
            is None
        )


def test_chain_map_loaders_use_metadata_then_legacy_csv(tmp_path: Path) -> None:
    metadata = {"chain_id_map": {"long": "A", "": "B", "bad": ""}}
    with (
        patch.object(builder, "_load_pdb_cache_metadata", return_value=metadata),
        patch.object(builder, "_cached_pdb_matches_metadata", return_value=True),
    ):
        assert builder.load_cached_chain_id_map(tmp_path, "1abc") == {"long": "A"}

    map_path = tmp_path / "legacy.chain_map.csv"
    assert builder.load_chain_id_map(map_path) == {}
    map_path.write_text(
        "original_chain_id,mapped_chain_id\nlong,A\n,B\nbad,\nsecond,C\n",
        encoding="utf-8",
    )
    assert builder.load_chain_id_map(map_path) == {"long": "A", "second": "C"}

    legacy_cache_map = tmp_path / "1ABC.chain_map.csv"
    legacy_cache_map.write_text(map_path.read_text(encoding="utf-8"), encoding="utf-8")
    with patch.object(builder, "_load_pdb_cache_metadata", return_value=None):
        assert builder.load_cached_chain_id_map(tmp_path, "1abc") == {
            "long": "A",
            "second": "C",
        }


def test_locked_subset_builder_requires_valid_cif_metadata(tmp_path: Path) -> None:
    with patch.object(builder, "_load_pdb_cache_metadata", return_value=None):
        with pytest.raises(RuntimeError, match="Missing validated mmCIF metadata"):
            builder._download_pdb_chain_subset_if_needed_locked(
                cache_dir=tmp_path,
                entry_id="1ABC",
                selected_chain_ids={"A"},
                stem="subset",
                cif_path=tmp_path / "1ABC.cif",
            )


def test_locked_subset_builder_reports_selected_chains_missing_from_structure(
    tmp_path: Path,
) -> None:
    cif_path = tmp_path / "1ABC.cif"
    with (
        patch.object(
            builder,
            "_load_pdb_cache_metadata",
            return_value={"sha256": "source", "source_url": "fixture"},
        ),
        patch.object(builder, "_load_valid_cached_chain_subset", return_value=None),
        patch.object(builder, "parse_mmcif_structure", return_value=[[]]),
    ):
        with pytest.raises(RuntimeError, match="missing selected chains: A"):
            builder._download_pdb_chain_subset_if_needed_locked(
                cache_dir=tmp_path,
                entry_id="1ABC",
                selected_chain_ids={"A"},
                stem="subset",
                cif_path=cif_path,
            )


def test_extract_model_texts_supports_model_less_and_malformed_nested_models(
    tmp_path: Path,
) -> None:
    model_less = tmp_path / "model-less.pdb"
    atom = _ca_line(serial=1)
    hetatm = _ca_line(record="HETATM", serial=2, resname="MSE", resid=2)
    model_less.write_text(atom + hetatm + "TER\nEND\n", encoding="utf-8")
    assert builder.extract_model_pdb_texts(model_less) == [atom + hetatm + "TER\nEND\n"]

    nested = tmp_path / "nested.pdb"
    atom_two = _ca_line(serial=2, resid=2)
    nested.write_text(
        "MODEL        1\n"
        + atom
        + "MODEL        2\n"
        + atom_two
        + "ENDMDL\n"
        + _ca_line(serial=3, resid=3),
        encoding="utf-8",
    )
    assert builder.extract_model_pdb_texts(nested) == [
        atom + "END\n",
        atom_two + "END\n",
    ]


def test_stride_output_parser_ignores_bad_rows_normalizes_and_deduplicates() -> None:
    stdout = "\n".join(
        [
            "HEADER ignored",
            "ASG too short",
            "ASG ALA A 12A 12 H",
            "ASG ALA A 12A 12 E",
            "ASG ALA A invalid 0 E",
            "ASG ALA B 13 13 Z",
            "ASG ALA B 14 14 HELIX",
        ]
    )
    assert builder._parse_stride_state_by_chain(stdout) == {
        "A": {12: "H"},
        "B": {13: "C"},
    }


def test_stride_chain_selection_uses_exact_then_single_chain_fallback() -> None:
    exact = {"A": {1: "H"}, "B": {2: "E"}}
    assert builder._select_stride_chain_states(exact, "B") == {2: "E"}
    assert builder._select_stride_chain_states(exact, "missing") is None
    assert builder._select_stride_chain_states({"only": {3: "T"}}, "A") == {3: "T"}
    assert builder._select_stride_chain_states({}, "A") is None


def test_stride_runner_parses_success_and_returns_none_on_nonzero_exit() -> None:
    model_text = _ca_line()

    def successful_run(command, **kwargs):
        assert command[0] == "stride-fixture"
        assert Path(command[1]).read_text(encoding="utf-8") == model_text
        assert kwargs == {"check": False, "capture_output": True, "text": True}
        return SimpleNamespace(returncode=0, stdout="ASG ALA A 1 1 H\n")

    with patch.object(builder.subprocess, "run", side_effect=successful_run):
        assert builder._run_stride_for_model_text(model_text, "stride-fixture") == {
            "A": {1: "H"}
        }

    with patch.object(
        builder.subprocess,
        "run",
        return_value=SimpleNamespace(returncode=2, stdout="ignored"),
    ):
        assert builder._run_stride_for_model_text(model_text, "stride-fixture") is None


@pytest.mark.parametrize(
    "payload",
    [
        {"first_model_sha1": "other", "state_by_chain": {}},
        {"first_model_sha1": "sha", "state_by_chain": []},
        {"first_model_sha1": "sha", "state_by_chain": {"A": []}},
        {"first_model_sha1": "sha", "state_by_chain": {"A": {"bad": "H"}}},
        {"first_model_sha1": "sha", "state_by_chain": {"A": {"1": "Z"}}},
    ],
)
def test_stride_cache_loader_rejects_stale_or_malformed_fields(
    tmp_path: Path, payload: dict[str, object]
) -> None:
    cache_path = tmp_path / "stride.json"
    cache_path.write_text(json.dumps(payload), encoding="utf-8")
    assert builder._load_cached_stride_state_by_chain(cache_path, "sha") is None


def test_stride_cache_loader_handles_missing_bad_json_and_valid_state(
    tmp_path: Path,
) -> None:
    cache_path = tmp_path / "stride.json"
    assert builder._load_cached_stride_state_by_chain(cache_path, "sha") is None
    cache_path.write_text("{bad", encoding="utf-8")
    assert builder._load_cached_stride_state_by_chain(cache_path, "sha") is None
    cache_path.write_text(
        json.dumps(
            {
                "first_model_sha1": "sha",
                "state_by_chain": {"A": {"-1": "G", "2": "C"}},
            }
        ),
        encoding="utf-8",
    )
    assert builder._load_cached_stride_state_by_chain(cache_path, "sha") == {
        "A": {-1: "G", 2: "C"}
    }


def test_stride_cache_loader_rejects_non_object_json(tmp_path: Path) -> None:
    cache_path = tmp_path / "stride.json"
    cache_path.write_text("[]", encoding="utf-8")
    assert builder._load_cached_stride_state_by_chain(cache_path, "sha") is None


def test_stride_cache_writer_serializes_sorted_keys_and_uppercase_entry(
    tmp_path: Path,
) -> None:
    cache_path = tmp_path / "cache" / "stride.json"
    builder._write_cached_stride_state_by_chain(
        cache_path,
        "1abc",
        "sha",
        {"B": {4: "C"}, "A": {2: "E", 1: "H"}},
    )
    assert json.loads(cache_path.read_text(encoding="utf-8")) == {
        "entry_id": "1ABC",
        "first_model_sha1": "sha",
        "state_by_chain": {"A": {"1": "H", "2": "E"}, "B": {"4": "C"}},
    }


def test_stride_cache_writer_cleans_temporary_file_after_replace_error(
    tmp_path: Path,
) -> None:
    cache_path = tmp_path / "stride.json"
    with patch.object(Path, "replace", side_effect=OSError("disk failure")):
        builder._write_cached_stride_state_by_chain(
            cache_path, "1ABC", "sha", {"A": {1: "H"}}
        )
    assert not cache_path.exists()
    assert list(tmp_path.iterdir()) == []


def test_first_model_stride_loader_handles_no_coordinates_and_failed_stride(
    tmp_path: Path,
) -> None:
    empty_pdb = tmp_path / "empty.pdb"
    empty_pdb.write_text("HEADER fixture\nEND\n", encoding="utf-8")
    with patch.object(builder, "_run_stride_for_model_text") as run_stride:
        assert builder.load_first_model_stride_state_by_chain(
            empty_pdb, "1ABC", "stride", tmp_path / "cache"
        ) == (None, 0)
        run_stride.assert_not_called()

    pdb_path = tmp_path / "one-model.pdb"
    pdb_path.write_text(_ca_line(), encoding="utf-8")
    with (
        patch.object(builder, "_run_stride_for_model_text", return_value=None),
        patch.object(builder, "_write_cached_stride_state_by_chain") as write_cache,
    ):
        assert builder.load_first_model_stride_state_by_chain(
            pdb_path, "1ABC", "stride", tmp_path / "cache"
        ) == (None, 1)
        write_cache.assert_not_called()


def test_stride_coverages_fill_unassigned_modeled_length_as_coil(
    tmp_path: Path,
) -> None:
    with (
        patch.object(
            builder, "download_pdb_if_needed", return_value=tmp_path / "x.pdb"
        ),
        patch.object(builder, "load_cached_chain_id_map", return_value={"long": "A"}),
        patch.object(
            builder,
            "load_first_model_stride_state_by_chain",
            return_value=({"A": {1: "H", 2: "E"}}, 3),
        ),
    ):
        coverages, model_count, succeeded = (
            builder.compute_stride_state_coverages_for_chain_modeled_first_model(
                session=MagicMock(),
                config=builder.DatasetBuildConfig(),
                cache_dir=tmp_path,
                stride_cache_dir=tmp_path / "stride",
                entry_id="1ABC",
                chain_id="long",
                modeled_sequence_length=4,
                modeled_auth_seq_ids={1, 2, 3},
                stride_executable="stride",
            )
        )

    assert (model_count, succeeded) == (3, 1)
    assert coverages == {
        "H": 0.25,
        "G": 0.0,
        "I": 0.0,
        "E": 0.25,
        "B": 0.0,
        "T": 0.0,
        "C": 0.5,
    }


def test_stride_coverages_return_sentinels_for_input_download_and_parse_failures(
    tmp_path: Path,
) -> None:
    expected = {state: -1.0 for state in builder.STRIDE_STATE_CODES}
    with patch.object(builder, "download_pdb_if_needed") as download:
        assert builder.compute_stride_state_coverages_for_chain_modeled_first_model(
            MagicMock(),
            builder.DatasetBuildConfig(),
            tmp_path,
            tmp_path,
            "1ABC",
            "A",
            0,
            {1},
            "stride",
        ) == (expected, 0, 0)
        download.assert_not_called()

    with patch.object(
        builder, "download_pdb_if_needed", side_effect=RuntimeError("offline")
    ):
        assert builder.compute_stride_state_coverages_for_chain_modeled_first_model(
            MagicMock(),
            builder.DatasetBuildConfig(),
            tmp_path,
            tmp_path,
            "1ABC",
            "A",
            1,
            {1},
            "stride",
        ) == (expected, 0, 0)

    with (
        patch.object(builder, "download_pdb_if_needed", return_value=tmp_path / "x"),
        patch.object(builder, "load_cached_chain_id_map", return_value={}),
        patch.object(
            builder,
            "load_first_model_stride_state_by_chain",
            return_value=({"B": {1: "H"}, "C": {1: "E"}}, 2),
        ),
    ):
        assert builder.compute_stride_state_coverages_for_chain_modeled_first_model(
            MagicMock(),
            builder.DatasetBuildConfig(),
            tmp_path,
            tmp_path,
            "1ABC",
            "A",
            1,
            {1},
            "stride",
        ) == (expected, 2, 0)


def test_stride_core_wrapper_short_circuits_and_selects_single_chain_fallback(
    tmp_path: Path,
) -> None:
    with patch.object(builder, "load_first_model_stride_state_by_chain") as load:
        assert (
            builder.compute_stride_core_range_for_modeled_auth_seq_ids_in_first_model(
                tmp_path / "x.pdb", "1ABC", "A", set(), "stride", tmp_path
            )
            is None
        )
        load.assert_not_called()

    with patch.object(
        builder,
        "load_first_model_stride_state_by_chain",
        return_value=({"different": {1: "C", 2: "H", 3: "E"}}, 1),
    ):
        assert (
            builder.compute_stride_core_range_for_modeled_auth_seq_ids_in_first_model(
                tmp_path / "x.pdb", "1ABC", "A", {1, 2, 3}, "stride", tmp_path
            )
            == (2, 3)
        )


def test_ca_line_parser_handles_fixed_columns_long_names_and_bad_fields() -> None:
    fixed = builder._parse_first_model_ca_line_fields(
        _ca_line(resid=-3, insertion_code="A", alt_loc="B", occupancy=0.75)
    )
    assert fixed == ("A", -3, "A", "B", 0.75, "ALA")
    assert builder._parse_first_model_ca_line_fields(_ca_line(atom_name="N")) is None

    long_name = builder._parse_first_model_ca_line_fields(
        _ca_line(record="HETATM", resname="A1BEB", resid=30)
    )
    assert long_name == ("A", 30, "", "", 1.0, "A1BEB")

    bad = list(_ca_line())
    bad[22:26] = "oops"
    bad[54:60] = "broken"
    assert builder._parse_first_model_ca_line_fields("".join(bad)) is None


def test_ca_line_parser_rejects_missing_occupancy_in_standard_record() -> None:
    line = list(_ca_line(occupancy=1.0))
    line[54:60] = "      "
    assert builder._parse_first_model_ca_line_fields("".join(line)) is None


def test_occupancy_and_ca_candidate_tiebreak_helpers_cover_all_precedence() -> None:
    line = _ca_line(occupancy=0.25)
    assert builder._parse_pdb_occupancy(line) == 0.25
    assert builder._parse_pdb_occupancy(line[:54] + "      " + line[60:]) == float(
        "-inf"
    )
    assert builder._parse_pdb_occupancy(line[:54] + "bad   " + line[60:]) == float(
        "-inf"
    )
    assert builder._alt_loc_tiebreak_key("") < builder._alt_loc_tiebreak_key("A")
    assert builder._alt_loc_tiebreak_key("A") < builder._alt_loc_tiebreak_key("1")
    assert builder._alt_loc_tiebreak_key("B") > builder._alt_loc_tiebreak_key("1")
    assert builder._insertion_code_tiebreak_key("") < (
        builder._insertion_code_tiebreak_key("A")
    )

    assert builder._is_better_ca_candidate("", 0.1, "B", "A", 1.0, "A")
    assert not builder._is_better_ca_candidate("A", 1.0, "A", "", 0.1, "B")
    assert builder._is_better_ca_candidate("", 0.9, "B", "", 0.5, "A")
    assert not builder._is_better_ca_candidate("", 0.5, "A", "", 0.9, "B")
    assert builder._is_better_ca_candidate("", 0.5, "", "", 0.5, "A")
    assert not builder._is_better_ca_candidate("", 0.5, "B", "", 0.5, "A")


def test_modres_parser_supports_fixed_and_split_fallback_rows(tmp_path: Path) -> None:
    pdb_path = tmp_path / "modres.pdb"
    pdb_path.write_text(
        _modres_line(insertion_code="B")
        + "MODRES short MLY A 6 LYS\n"
        + "MODRES short BAD A nope MET\n"
        + _modres_line(resname="FOO", resid=7, standard_resname="UNK"),
        encoding="utf-8",
    )
    assert builder._parse_pdb_modres_identity_map(pdb_path) == {
        ("A", 5, "B", "MSE"): "M",
        ("A", 6, "", "MLY"): "K",
    }


def test_modres_parser_ignores_truncated_record(tmp_path: Path) -> None:
    pdb_path = tmp_path / "truncated-modres.pdb"
    pdb_path.write_text("MODRES too-short\n", encoding="utf-8")
    assert builder._parse_pdb_modres_identity_map(pdb_path) == {}


def test_first_model_residue_parser_applies_atom_insertion_and_altloc_priority(
    tmp_path: Path,
) -> None:
    pdb_path = tmp_path / "selection.pdb"
    pdb_path.write_text(
        "MODEL        1\n"
        + _ca_line(record="HETATM", serial=1, resname="MSE", resid=5)
        + _ca_line(serial=2, resid=5, occupancy=0.2)
        + _ca_line(serial=3, resid=6, insertion_code="A", occupancy=1.0)
        + _ca_line(serial=4, resid=6, insertion_code="", occupancy=0.5)
        + _ca_line(serial=5, resid=7, alt_loc="B", occupancy=0.5, resname="GLY")
        + _ca_line(serial=6, resid=7, alt_loc="A", occupancy=0.5, resname="SER")
        + _ca_line(serial=7, resid=8, occupancy=0.0)
        + _ca_line(serial=8, chain_id="B", resid=6)
        + "ENDMDL\n"
        + "MODEL        2\n"
        + _ca_line(serial=9, resid=9)
        + "ENDMDL\n",
        encoding="utf-8",
    )

    records = builder.parse_first_model_ca_residues(
        pdb_path, "A", start_seq_id=5, end_seq_id=7, include_hetatm=True
    )
    assert [(record.resid, record.identity) for record in records] == [
        (5, "A"),
        (6, "A"),
        (7, "S"),
    ]
    assert records[0].is_standard_atom
    assert records[0].has_hetatm_ca

    atom_only = builder.parse_first_model_ca_residue_sequence(
        pdb_path, "A", include_hetatm=False
    )
    assert atom_only == [(5, "A"), (6, "A"), (7, "S")]


def test_model_coordinate_parser_finalizes_model_less_altloc_data(
    tmp_path: Path,
) -> None:
    pdb_path = tmp_path / "coords.pdb"
    pdb_path.write_text(
        _ca_line(serial=1, resid=1, alt_loc="B", occupancy=0.5, x=1.0)
        + _ca_line(serial=2, resid=1, alt_loc="A", occupancy=0.5, x=2.0)
        + _ca_line(serial=3, chain_id="B", resid=1, x=9.0),
        encoding="utf-8",
    )
    models, raw_counts = builder.parse_models_ca_coords_with_stats(pdb_path, "A")
    assert raw_counts == [{1: 2}]
    np.testing.assert_allclose(models[0][1], [2.0, 2.0, 3.0])
    wrapped = builder.parse_models_ca_coords(pdb_path, "A")
    np.testing.assert_allclose(wrapped[0][1], [2.0, 2.0, 3.0])


def test_hetatm_split_and_identity_match_input_validation() -> None:
    def standard(resid: int, identity: str) -> builder.CAResidueRecord:
        return builder.CAResidueRecord(resid, identity, True)

    hetatm = builder.CAResidueRecord(2, "M", True, has_hetatm_ca=True)
    regions = builder._split_xray_ca_residues_at_hetatm(
        [hetatm, standard(3, "A"), standard(4, "C"), hetatm, standard(6, "D")]
    )
    assert [[record.resid for record in region] for region in regions] == [
        [3, 4],
        [6],
    ]
    assert builder._ca_residue_has_hetatm(hetatm)
    assert builder._ca_residue_has_hetatm(builder.CAResidueRecord(7, "HET:X", False))
    assert not builder._ca_residue_has_hetatm(standard(8, "G"))

    with pytest.raises(ValueError, match="between 0 and 100"):
        builder.find_modeled_ca_core_identity_matches(
            [standard(1, "A")], [standard(1, "A")], 101
        )
    assert (
        builder.find_modeled_ca_core_identity_matches([], [standard(1, "A")], 101) == []
    )
    assert (
        builder._find_gapped_modeled_ca_core_identity_match(
            [standard(1, "A")], [standard(1, "A")], 0
        )
        is None
    )


def test_coordinate_hash_prefers_valid_metadata_and_falls_back_to_file(
    tmp_path: Path,
) -> None:
    pdb_path = tmp_path / "1ABC.pdb"
    pdb_path.write_bytes(b"fixture")
    trusted_sha = "a" * 64
    with (
        patch.object(
            builder, "_load_pdb_cache_metadata", return_value={"sha256": trusted_sha}
        ),
        patch.object(builder, "_cached_pdb_matches_metadata", return_value=True),
        patch.object(builder, "_sha256_file") as hash_file,
    ):
        assert builder._coordinate_source_sha256(pdb_path) == trusted_sha
        hash_file.assert_not_called()

    with patch.object(builder, "_load_pdb_cache_metadata", return_value=None):
        assert (
            builder._coordinate_source_sha256(pdb_path)
            == hashlib.sha256(b"fixture").hexdigest()
        )


def test_first_model_ca_cache_roundtrip_supports_empty_and_coordinate_less_chains(
    tmp_path: Path,
) -> None:
    cache_path = tmp_path / "ca.npz"
    records = (builder.CAResidueRecord(5, "M", False, True),)
    parsed = {"empty": (tuple(), {}), "A": (records, {})}
    builder._write_first_model_ca_cache(cache_path, "source", parsed)

    loaded = builder._read_first_model_ca_cache(cache_path, "source")
    assert loaded is not None
    assert loaded["empty"] == (tuple(), {})
    assert loaded["A"][0] == records
    assert loaded["A"][1] == {}
    assert builder._read_first_model_ca_cache(cache_path, "other") is None


def _write_raw_ca_cache(path: Path, **overrides: np.ndarray) -> None:
    payload = {
        "schema_version": np.asarray(builder.XRAY_CA_CACHE_SCHEMA_VERSION),
        "parser_revision": np.asarray(builder.XRAY_CA_PARSER_REVISION),
        "source_sha256": np.asarray("source"),
        "chain_ids": np.asarray(["A"]),
        "chain_offsets": np.asarray([0, 1], dtype=np.int64),
        "resids": np.asarray([1], dtype=np.int64),
        "identities": np.asarray(["A"]),
        "flags": np.asarray([1], dtype=np.uint8),
        "coords": np.asarray([[1.0, 2.0, 3.0]]),
        "coord_present": np.asarray([True]),
    }
    payload.update(overrides)
    with path.open("wb") as handle:
        np.savez_compressed(handle, **payload)


@pytest.mark.parametrize(
    "overrides",
    [
        {"chain_ids": np.asarray(["A", "A"]), "chain_offsets": np.asarray([0, 0, 1])},
        {"chain_offsets": np.asarray([1, 1])},
        {"chain_offsets": np.asarray([0, 2])},
        {"flags": np.asarray([4], dtype=np.uint8)},
        {"coords": np.asarray([[np.nan, 2.0, 3.0]])},
        {
            "chain_offsets": np.asarray([0, 2]),
            "resids": np.asarray([1, 1]),
            "identities": np.asarray(["A", "A"]),
            "flags": np.asarray([1, 1], dtype=np.uint8),
            "coords": np.asarray([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]),
            "coord_present": np.asarray([True, True]),
        },
    ],
)
def test_first_model_ca_cache_rejects_semantically_corrupt_arrays(
    tmp_path: Path, overrides: dict[str, np.ndarray]
) -> None:
    cache_path = tmp_path / "bad.npz"
    _write_raw_ca_cache(cache_path, **overrides)
    assert builder._read_first_model_ca_cache(cache_path, "source") is None


def test_cached_first_model_ca_data_reparses_if_source_changes_mid_transaction(
    tmp_path: Path,
) -> None:
    pdb_path = tmp_path / "1ABC.pdb"
    pdb_path.write_bytes(b"fixture")
    first = {"A": ((builder.CAResidueRecord(1, "A", True),), {1: np.zeros(3)})}
    second = {"A": ((builder.CAResidueRecord(2, "G", True),), {2: np.ones(3)})}
    with (
        patch.object(builder, "_coordinate_source_sha256", side_effect=["old", "new"]),
        patch.object(builder, "_read_first_model_ca_cache", return_value=None),
        patch.object(
            builder, "_parse_first_model_ca_data_by_chain", side_effect=[first, second]
        ) as parse,
        patch.object(builder, "_write_first_model_ca_cache") as write,
    ):
        records, coords = builder.load_cached_first_model_ca_data(pdb_path, "A")

    assert [record.resid for record in records] == [2]
    np.testing.assert_allclose(coords[2], np.ones(3))
    assert parse.call_count == 2
    assert write.call_args.kwargs["source_sha256"] == "new"
    assert write.call_args.kwargs["parsed_by_chain"] is second


def test_alignment_and_rmsd_helpers_validate_and_transform_coordinates() -> None:
    reference = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    mobile = reference + np.asarray([10.0, -4.0, 2.0])
    aligned = builder._aligned_coordinates_to_reference(mobile, reference)
    np.testing.assert_allclose(aligned, reference, atol=1e-7)
    assert builder._superposed_rmsd(mobile, reference) == pytest.approx(0.0, abs=1e-7)

    stacked = np.stack([reference, mobile])
    aligned_models = builder._coordinates_aligned_to_first_model(stacked)
    np.testing.assert_allclose(aligned_models[0], reference, atol=1e-7)
    np.testing.assert_allclose(aligned_models[1], reference, atol=1e-7)

    for invalid in (np.asarray([]), np.empty((0, 2, 3))):
        with pytest.raises(ValueError, match="coords must have shape"):
            builder._coordinates_aligned_to_first_model(invalid)
        with pytest.raises(ValueError, match="coords must have shape"):
            builder._ca_rmsd_to_mean_structure(invalid)


def test_thread_local_session_is_reused_and_delegates_get_and_post() -> None:
    session = MagicMock()
    session.get.return_value = "get-response"
    session.post.return_value = "post-response"
    session.headers = {}
    with patch.object(builder.requests, "Session", return_value=session) as factory:
        client = builder.ThreadLocalRequestsSession("fixture-agent")
        assert client.get("https://example.test/a", timeout=1) == "get-response"
        assert client.post("https://example.test/b", json={"x": 1}) == "post-response"
        assert client._session() is session

    factory.assert_called_once_with()
    assert session.headers == {"User-Agent": "fixture-agent"}
    session.get.assert_called_once_with("https://example.test/a", timeout=1)
    session.post.assert_called_once_with("https://example.test/b", json={"x": 1})


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("95", 95), (94.6, 95), (None, None), ("bad", None), (float("nan"), None)],
)
def test_similarity_cutoff_normalization(raw: object, expected: int | None) -> None:
    assert builder.RCSBClient._normalize_similarity_cutoff(raw) == expected


def test_similarity_cutoff_rejects_infinity() -> None:
    assert builder.RCSBClient._normalize_similarity_cutoff(float("inf")) is None


def test_sequence_identity_group_extraction_filters_memberships() -> None:
    memberships = [
        None,
        {"aggregation_method": "other", "similarity_cutoff": 95, "group_id": "x"},
        {
            "aggregation_method": builder.SEQUENCE_IDENTITY_AGGREGATION_METHOD,
            "similarity_cutoff": "94.6",
            "group_id": "g95",
        },
        {
            "aggregation_method": builder.SEQUENCE_IDENTITY_AGGREGATION_METHOD,
            "similarity_cutoff": 100,
            "group_id": "g100",
        },
        {
            "aggregation_method": builder.SEQUENCE_IDENTITY_AGGREGATION_METHOD,
            "similarity_cutoff": "bad",
            "group_id": "bad",
        },
        {
            "aggregation_method": builder.SEQUENCE_IDENTITY_AGGREGATION_METHOD,
            "similarity_cutoff": 90,
            "group_id": "",
        },
    ]
    assert builder.RCSBClient._extract_sequence_identity_groups(memberships) == {
        95: "g95",
        100: "g100",
    }
    assert builder.RCSBClient._extract_sequence_identity_groups(
        memberships, allowed_cutoffs={95}
    ) == {95: "g95"}


def test_client_pagination_advances_by_returned_items_and_stops_on_empty_page() -> None:
    client = builder.RCSBClient(builder.DatasetBuildConfig(page_size=2))
    client._post_json = Mock(
        side_effect=[
            {
                "total_count": 4,
                "result_set": [
                    {"identifier": "1ABC"},
                    {"missing": "ignored"},
                    {"identifier": "2DEF"},
                ],
            },
            {"total_count": 4, "result_set": []},
        ]
    )
    query = {"type": "terminal"}
    assert client._fetch_paginated_identifiers(query, "entry", "fixture") == [
        "1ABC",
        "2DEF",
    ]
    calls = client._post_json.call_args_list
    assert calls[0].args[1]["request_options"]["paginate"] == {
        "start": 0,
        "rows": 2,
    }
    assert calls[1].args[1]["request_options"]["paginate"] == {
        "start": 2,
        "rows": 2,
    }


def test_client_post_json_retries_request_and_json_errors_then_succeeds() -> None:
    client = builder.RCSBClient(
        builder.DatasetBuildConfig(retries=3, backoff_seconds=0.25, timeout_seconds=7)
    )
    request_failure = requests.ConnectionError("offline")
    bad_json_response = MagicMock()
    bad_json_response.raise_for_status.return_value = None
    bad_json_response.json.side_effect = ValueError("bad json")
    success_response = MagicMock()
    success_response.raise_for_status.return_value = None
    success_response.json.return_value = {"ok": True}
    client.session = SimpleNamespace(
        post=Mock(side_effect=[request_failure, bad_json_response, success_response])
    )
    with patch.object(builder.time, "sleep") as sleep:
        assert client._post_json("https://example.test", {"x": 1}) == {"ok": True}

    assert client.session.post.call_count == 3
    assert sleep.call_args_list[0].args == (0.25,)
    assert sleep.call_args_list[1].args == (0.5,)


def test_client_post_json_raises_after_last_attempt_without_final_sleep() -> None:
    client = builder.RCSBClient(
        builder.DatasetBuildConfig(retries=2, backoff_seconds=1.0)
    )
    client.session = SimpleNamespace(
        post=Mock(side_effect=requests.Timeout("always offline"))
    )
    with patch.object(builder.time, "sleep") as sleep:
        with pytest.raises(RuntimeError, match="after 2 attempts: always offline"):
            client._post_json("https://example.test", {})
    sleep.assert_called_once_with(1.0)


def test_fetch_entry_ids_for_method_set_builds_group_query_and_batches_filters() -> (
    None
):
    client = builder.RCSBClient(builder.DatasetBuildConfig(graphql_batch_size=2))
    client._fetch_paginated_identifiers = Mock(return_value=["1", "2", "3"])
    client._filter_entry_ids_by_exact_methods = Mock(
        side_effect=lambda entry_ids, **kwargs: entry_ids[:1]
    )

    result = client.fetch_entry_ids_for_method_set(
        "SOLUTION NMR",
        ("SOLUTION NMR", "SOLID-STATE NMR"),
        require_protein_entities=True,
    )

    assert result == ["1", "3"]
    query = client._fetch_paginated_identifiers.call_args.kwargs["query"]
    assert query["logical_operator"] == "and"
    assert [node["parameters"]["value"] for node in query["nodes"]] == [
        "SOLUTION NMR",
        "SOLID-STATE NMR",
    ]
    assert [
        call.kwargs["entry_ids"]
        for call in client._filter_entry_ids_by_exact_methods.call_args_list
    ] == [
        ["1", "2"],
        ["3"],
    ]
    assert all(
        call.kwargs["record_exclusions"]
        for call in client._filter_entry_ids_by_exact_methods.call_args_list
    )

    with pytest.raises(ValueError, match="must not be empty"):
        client.fetch_entry_ids_for_method_set("empty", tuple())
    client._fetch_paginated_identifiers.return_value = []
    assert client.fetch_entry_ids_for_method_set("X-ray", ("X-RAY DIFFRACTION",)) == []
