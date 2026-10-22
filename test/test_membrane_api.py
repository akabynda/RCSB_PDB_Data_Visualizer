"""Network-free coverage for membrane metadata retrieval and source caching."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from src.dataset.api_cache import CachedJSONRequests
from src.dataset.client.membrane import MembraneEntriesMixin, membrane_search_query
from src.dataset.config import DatasetBuildConfig, MEMBRANE_ANNOTATION_TYPES


class MembraneClient(MembraneEntriesMixin):
    def __init__(self, page_size: int = 2) -> None:
        self.config = DatasetBuildConfig(page_size=page_size)


def test_cache_hit_preserves_exact_source_and_avoids_network(tmp_path: Path) -> None:
    network = Mock(return_value={"data": {"entries": [{"rcsb_id": "1ABC"}]}})
    request = {"query": "query { entries { rcsb_id } }"}
    cache = CachedJSONRequests(network, tmp_path)
    expected = cache.post_json("https://data.rcsb.org/graphql", request)
    source = cache.sources[0]
    body = Path(source["response_path"])
    assert hashlib.sha256(body.read_bytes()).hexdigest() == source["response_sha256"]
    assert source["request"]["payload"] == request
    second = CachedJSONRequests(network, tmp_path)
    assert second.post_json("https://data.rcsb.org/graphql", request) == expected
    assert second.sources == cache.sources
    network.assert_called_once()


def test_cache_refresh_retains_previous_response_body(tmp_path: Path) -> None:
    network = Mock(side_effect=[{"value": 1}, {"value": 2}])
    cache = CachedJSONRequests(network, tmp_path, validation_hours=0)
    assert cache.post_json("api", {"selection": "one"}) == {"value": 1}
    original = cache.sources[0]["response_path"]
    assert cache.post_json("api", {"selection": "one"}) == {"value": 2}
    assert json.loads(Path(original).read_text()) == {"value": 1}
    assert len(list((tmp_path / "responses").glob("*.json"))) == 2
    assert len(cache.sources) == 2


def test_cache_key_includes_url_request_and_schema(tmp_path: Path) -> None:
    network = Mock(return_value={"value": 1})
    cache = CachedJSONRequests(network, tmp_path)
    cache.post_json("api-one", {"selection": "one"})
    cache.post_json("api-two", {"selection": "one"})
    cache.post_json("api-one", {"selection": "two"})
    changed_schema = CachedJSONRequests(network, tmp_path, schema_version="membrane-v2")
    changed_schema.post_json("api-one", {"selection": "one"})
    assert network.call_count == 4
    assert len(list((tmp_path / "requests").glob("*.json"))) == 4


@pytest.mark.parametrize("damaged", ["response", "pointer"])
def test_corrupt_cache_is_refetched(tmp_path: Path, caplog, damaged: str) -> None:
    network = Mock(return_value={"value": 1})
    cache = CachedJSONRequests(network, tmp_path)
    cache.post_json("api", {"selection": "one"})
    source = cache.sources[0]
    target = source["response_path" if damaged == "response" else "request_path"]
    Path(target).write_text("{}", encoding="utf-8")
    assert cache.post_json("api", {"selection": "one"}) == {"value": 1}
    assert network.call_count == 2
    assert "Refreshing invalid API cache" in caplog.text


def test_expired_cache_does_not_hide_network_failure(tmp_path: Path) -> None:
    network = Mock(return_value={"value": 1})
    cache = CachedJSONRequests(network, tmp_path)
    cache.post_json("api", {})
    pointer_path = Path(cache.sources[0]["request_path"])
    pointer = json.loads(pointer_path.read_text())
    pointer["retrieved_at"] = "2000-01-01T00:00:00Z"
    pointer_path.write_text(json.dumps(pointer), encoding="utf-8")
    previous = pointer_path.read_bytes()
    network.side_effect = RuntimeError("offline")
    with pytest.raises(RuntimeError, match="offline"):
        cache.post_json("api", {})
    assert pointer_path.read_bytes() == previous


@pytest.mark.parametrize("hours", [-1, float("nan"), float("inf")])
def test_cache_rejects_invalid_ttl(tmp_path: Path, hours: float) -> None:
    with pytest.raises(ValueError, match="finite and nonnegative"):
        CachedJSONRequests(Mock(), tmp_path, validation_hours=hours)


@pytest.mark.parametrize("response", [[], {"errors": [{"message": "bad query"}]}, {}])
def test_invalid_graphql_response_is_not_cached(tmp_path: Path, response) -> None:
    cache = CachedJSONRequests(Mock(return_value=response), tmp_path)
    with pytest.raises(ValueError):
        cache.post_json("api", {"query": "query { entries { rcsb_id } }"})
    assert not list((tmp_path / "requests").glob("*.json"))


def test_search_query_uses_only_three_base_criteria() -> None:
    parameters = {
        node["parameters"]["attribute"]: node["parameters"]
        for node in membrane_search_query()["nodes"]
    }
    assert set(parameters) == {
        "rcsb_entry_info.polymer_entity_count_protein",
        "exptl.method",
        "rcsb_polymer_entity_annotation.type",
    }
    assert parameters["rcsb_entry_info.polymer_entity_count_protein"]["value"] == 1
    assert parameters["exptl.method"]["value"] == "SOLUTION NMR"
    assert parameters["rcsb_polymer_entity_annotation.type"]["value"] == list(
        MEMBRANE_ANNOTATION_TYPES
    )


def test_search_paginates_and_sorts_complete_identifier_set() -> None:
    cache = Mock()
    cache.post_json.side_effect = [
        {
            "total_count": 3,
            "result_set": [{"identifier": "2DEF"}, {"identifier": "1ABC"}],
        },
        {"total_count": 3, "result_set": [{"identifier": "3GHI"}]},
    ]
    assert MembraneClient().fetch_solution_nmr_membrane_entry_ids(cache) == [
        "1ABC",
        "2DEF",
        "3GHI",
    ]
    calls = cache.post_json.call_args_list
    assert [call.args[1]["request_options"]["paginate"]["start"] for call in calls] == [
        0,
        2,
    ]


@pytest.mark.parametrize(
    "pages",
    [
        [{"result_set": []}],
        [{"total_count": True}],
        [{"total_count": 2, "result_set": []}],
        [
            {
                "total_count": 1,
                "result_set": [{"identifier": "1ABC"}, {"identifier": "2DEF"}],
            }
        ],
        [
            {
                "total_count": 2,
                "result_set": [{"identifier": "1ABC"}, {"identifier": "1ABC"}],
            }
        ],
        [{"total_count": 1, "result_set": [{}]}],
        [{"total_count": 1, "result_set": None}],
        [
            {"total_count": 2, "result_set": [{"identifier": "1ABC"}]},
            {"total_count": 3, "result_set": [{"identifier": "2DEF"}]},
        ],
        [
            {"total_count": 2, "result_set": [{"identifier": "1ABC"}]},
            {"total_count": 2, "result_set": [{"identifier": "1ABC"}]},
        ],
    ],
)
def test_search_rejects_incomplete_or_inconsistent_pages(pages) -> None:
    cache = Mock()
    cache.post_json.side_effect = pages
    with pytest.raises(RuntimeError):
        MembraneClient().fetch_solution_nmr_membrane_entry_ids(cache)


def test_search_accepts_explicitly_empty_cohort() -> None:
    cache = Mock()
    cache.post_json.return_value = {"total_count": 0, "result_set": []}
    assert MembraneClient().fetch_solution_nmr_membrane_entry_ids(cache) == []


def test_metadata_returns_raw_entries_without_monomer_or_model_filter() -> None:
    entry = {
        "rcsb_id": "1ABC",
        "rcsb_entry_info": {"deposited_model_count": 1},
        "polymer_entities": [{"rcsb_id": "1ABC_1"}, {"rcsb_id": "1ABC_2"}],
    }
    cache = Mock()
    cache.post_json.return_value = {"data": {"entries": [entry]}}
    assert MembraneClient().fetch_membrane_entries_for_ids(["1ABC"], cache) == [entry]


@pytest.mark.parametrize(
    "entries",
    [
        None,
        [],
        [None],
        [{}],
        [{"rcsb_id": "OTHER"}],
        [{"rcsb_id": "1ABC"}, {"rcsb_id": "1ABC"}],
    ],
)
def test_metadata_rejects_missing_unexpected_or_repeated_entries(entries) -> None:
    cache = Mock()
    cache.post_json.return_value = {"data": {"entries": entries}}
    with pytest.raises(RuntimeError):
        MembraneClient().fetch_membrane_entries_for_ids(["1ABC"], cache)


def test_metadata_empty_input_makes_no_request() -> None:
    cache = Mock()
    assert MembraneClient().fetch_membrane_entries_for_ids([], cache) == []
    cache.post_json.assert_not_called()
