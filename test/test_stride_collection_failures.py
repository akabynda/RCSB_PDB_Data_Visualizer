"""Collection failures must not look like completed or scientifically filtered data."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import requests

from src.dataset.client import RCSBClient
from src.dataset.config import DatasetBuildConfig


def _client(tmp_path: Path) -> RCSBClient:
    return RCSBClient(
        DatasetBuildConfig(retries=3, backoff_seconds=0, page_size=2),
        solution_nmr_monomer_cache_dir=tmp_path,
    )


def _response(payload: object) -> Mock:
    response = Mock()
    response.json.return_value = payload
    return response


def _monomer_entry() -> dict:
    return {
        "rcsb_id": "TEST",
        "rcsb_accession_info": {"deposit_date": "2020-01-01"},
        "rcsb_entry_info": {"deposited_model_count": 2},
        "polymer_entities": [
            {
                "entity_poly": {
                    "type": "polypeptide(L)",
                    "rcsb_entity_polymer_type": "Protein",
                    "pdbx_strand_id": "A",
                },
            },
        ],
    }


def _models(tmp_path: Path, chains_and_numbers: list[tuple[str, list[int]]]) -> Path:
    lines = []
    for model, (chain, numbers) in enumerate(chains_and_numbers, 1):
        lines.append(f"MODEL     {model:4d}\n")
        for serial, number in enumerate(numbers, 1):
            lines.append(
                f"ATOM  {serial:5d}  CA  ALA {chain}{number:4d}    "
                f"{float(number):8.3f}{0.0:8.3f}{0.0:8.3f}{1.0:6.2f}{20.0:6.2f}           C\n"
            )
        lines.append("ENDMDL\n")
    path = tmp_path / "models.pdb"
    path.write_text("".join(lines))
    return path


def test_http_200_graphql_errors_retry_instead_of_returning_partial_data(tmp_path):
    client = _client(tmp_path)
    partial = {
        "data": {"entries": [{"rcsb_id": "ONE"}, None]},
        "errors": [{"message": "upstream database timeout"}],
    }
    complete = {"data": {"entries": [{"rcsb_id": "ONE"}, {"rcsb_id": "TWO"}]}}
    client.session = Mock()
    client.session.post.side_effect = [_response(partial), _response(complete)]
    result = client._post_json(
        client.config.graphql_url, {"query": "query { entries { rcsb_id } }"}
    )
    assert result == complete
    assert client.session.post.call_count == 2


def test_persistent_graphql_error_aborts_batch_without_filtering_requested_ids(
    tmp_path,
):
    client = _client(tmp_path)
    client.session = Mock()
    client.session.post.return_value = _response(
        {"data": {"entries": None}, "errors": [{"message": "database unavailable"}]}
    )
    with patch("src.dataset.utils._record_filtered_structure") as filtered:
        with pytest.raises(
            RuntimeError, match="after 3 attempts.*database unavailable"
        ):
            client.fetch_solution_nmr_monomer_stride_modeled_first_model_records_for_ids(
                ["ONE", "TWO"], "unused-stride", tmp_path, tmp_path / "stride"
            )
    filtered.assert_not_called()
    assert client.session.post.call_count == 3


@pytest.mark.parametrize("payload", [None, [], {"data": None}, {}])
def test_malformed_graphql_response_cannot_become_an_empty_success(tmp_path, payload):
    client = _client(tmp_path)
    client.session = Mock()
    client.session.post.return_value = _response(payload)
    with pytest.raises(RuntimeError, match="after 3 attempts"):
        client._post_json(
            client.config.graphql_url, {"query": "query { entries { rcsb_id } }"}
        )
    assert client.session.post.call_count == 3


def test_search_requires_every_reported_identifier(tmp_path):
    client = _client(tmp_path)
    client._post_json = Mock(
        side_effect=[
            {
                "total_count": 3,
                "result_set": [{"identifier": "ONE"}, {"identifier": "TWO"}],
            },
            {"total_count": 3, "result_set": []},
        ]
    )
    with pytest.raises(RuntimeError, match="empty page before completion.*2/3"):
        client._fetch_paginated_identifiers({}, "entry")


@pytest.mark.parametrize("repeated", ["ONE", "TWO"])
def test_search_rejects_repeated_identifiers_between_pages(tmp_path, repeated):
    client = _client(tmp_path)
    client._post_json = Mock(
        side_effect=[
            {
                "total_count": 3,
                "result_set": [{"identifier": "ONE"}, {"identifier": "TWO"}],
            },
            {"total_count": 3, "result_set": [{"identifier": repeated}]},
        ]
    )
    with pytest.raises(RuntimeError, match="repeated identifiers"):
        client._fetch_paginated_identifiers({}, "entry")


def test_search_rejects_duplicates_within_one_page(tmp_path):
    client = _client(tmp_path)
    client._post_json = Mock(
        return_value={
            "total_count": 2,
            "result_set": [{"identifier": "ONE"}, {"identifier": "ONE"}],
        }
    )
    with pytest.raises(RuntimeError, match="repeated identifiers"):
        client._fetch_paginated_identifiers({}, "entry")


def test_search_rejects_total_count_changing_between_pages(tmp_path):
    client = _client(tmp_path)
    client._post_json = Mock(
        side_effect=[
            {
                "total_count": 3,
                "result_set": [{"identifier": "ONE"}, {"identifier": "TWO"}],
            },
            {"total_count": 2, "result_set": []},
        ]
    )
    with pytest.raises(RuntimeError, match="total_count changed"):
        client._fetch_paginated_identifiers({}, "entry")


def test_search_accepts_complete_pages_and_a_genuinely_empty_result(tmp_path):
    client = _client(tmp_path)
    client._post_json = Mock(
        side_effect=[
            {
                "total_count": 3,
                "result_set": [{"identifier": "ONE"}, {"identifier": "TWO"}],
            },
            {"total_count": 3, "result_set": [{"identifier": "THREE"}]},
            {"total_count": 0, "result_set": []},
        ]
    )
    assert client._fetch_paginated_identifiers({}, "entry") == ["ONE", "TWO", "THREE"]
    assert client._fetch_paginated_identifiers({}, "entry") == []


def test_network_failure_is_not_reported_as_unequal_model_lengths(tmp_path):
    client = _client(tmp_path)
    with patch.object(
        client,
        "_download_solution_nmr_monomer_pdb_if_needed",
        side_effect=requests.Timeout("network outage"),
    ):
        with patch("src.dataset.client.nmr._record_filtered_structure") as filtered:
            assert (
                client._extract_solution_nmr_monomer_context(_monomer_entry()) is None
            )
        filtered.assert_called_once_with(
            "TEST",
            "coordinate preparation failed: Timeout: network outage",
            year=2020,
        )
        assert not client._solution_nmr_monomer_coordinates_are_eligible("TEST", "A")


@pytest.mark.parametrize(
    "models,reason",
    [
        ([("A", [1, 2, 3])], "fewer than 2 coordinate models (found 1)"),
        (
            [("B", [1, 2, 3]), ("B", [1, 2, 3])],
            "no usable modeled CA residues for chain A in coordinate model(s): 1, 2",
        ),
        (
            [("A", [1, 2, 3]), ("B", [1, 2, 3])],
            "no usable modeled CA residues for chain A in coordinate model(s): 2",
        ),
        (
            [("A", [1, 2, 3]), ("A", [1, 2])],
            "coordinate models do not have equal full-chain lengths",
        ),
    ],
)
def test_model_filter_records_the_actual_failure_reason(tmp_path, models, reason):
    client = _client(tmp_path)
    path = _models(tmp_path, models)
    with patch.object(
        client, "_download_solution_nmr_monomer_pdb_if_needed", return_value=path
    ):
        with patch("src.dataset.client.nmr._record_filtered_structure") as filtered:
            assert (
                client._extract_solution_nmr_monomer_context(_monomer_entry()) is None
            )
        filtered.assert_called_once_with("TEST", reason, year=2020)
        assert not client._solution_nmr_monomer_coordinates_are_eligible("TEST", "A")


def test_equal_length_policy_still_allows_different_residue_sets(tmp_path):
    client = _client(tmp_path)
    path = _models(tmp_path, [("A", [-2, -1, 0]), ("A", [1, 2, 3])])
    with patch.object(
        client, "_download_solution_nmr_monomer_pdb_if_needed", return_value=path
    ):
        assert client._solution_nmr_monomer_coordinates_are_eligible("TEST", "A")
        assert (
            client._extract_solution_nmr_monomer_context(_monomer_entry()) is not None
        )
