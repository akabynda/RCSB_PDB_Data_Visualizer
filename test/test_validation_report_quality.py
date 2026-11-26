"""Network-free tests for quality metrics read from wwPDB validation reports."""

from __future__ import annotations

import gzip
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import requests

from src import pdb_dataset_builder as builder
from src.dataset.client import nmr as nmr_client


def _report(**attributes: str) -> bytes:
    """Return a gzipped validation report whose Entry has the given attributes."""
    rendered = " ".join(
        f'{name.replace("_", "-")}="{value}"' for name, value in attributes.items()
    )
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f"<wwPDB-validation-information><Entry {rendered}/>"
        '<ModelledSubgroup model="1" chain="A" resnum="1"/>'
        "</wwPDB-validation-information>\n"
    )
    return gzip.compress(xml.encode("utf-8"))


# files.rcsb.org answers a request for an absent report with S3 AccessDenied.
S3_ACCESS_DENIED = (
    b'<?xml version="1.0" encoding="UTF-8"?>\n'
    b"<Error><Code>AccessDenied</Code><Message>Access Denied</Message></Error>"
)


def _response(url: str, status_code: int, content: bytes) -> requests.Response:
    response = requests.Response()
    response.url = url
    response.status_code = status_code
    response._content = content
    return response


def _client(
    tmp_path: Path, reports: dict[str, bytes | requests.Response | None]
) -> builder.RCSBClient:
    """Build a client whose GraphQL lists every entry and whose GETs serve reports."""
    client = builder.RCSBClient(
        builder.DatasetBuildConfig(retries=1, backoff_seconds=0)
    )
    client.validation_report_cache_dir = tmp_path
    client._post_json = Mock(
        return_value={
            "data": {
                "entries": [
                    {
                        "rcsb_id": entry_id,
                        # RCSB GraphQL reports NMR zeros as null.
                        "pdbx_vrpt_summary_geometry": [
                            {
                                "clashscore": None,
                                "percent_ramachandran_outliers": None,
                                "percent_rotamer_outliers": None,
                            }
                        ],
                    }
                    for entry_id in reports
                ]
            }
        }
    )

    def get(url: str, **kwargs: object) -> Mock:
        for entry_id, content in reports.items():
            if url.endswith(f"/{entry_id.lower()}_validation.xml.gz"):
                if content is None:
                    return _response(url, 403, S3_ACCESS_DENIED)
                if isinstance(content, requests.Response):
                    return content
                return _response(url, 200, content)
        raise AssertionError(url)

    client.session = Mock()
    client.session.get = Mock(side_effect=get)
    return client


def _fetch(client: builder.RCSBClient, entry_ids: list[str]) -> list:
    with patch.object(
        client,
        "_extract_solution_nmr_monomer_context",
        side_effect=lambda entry: (entry["rcsb_id"], 2020, 2, {}, "A"),
    ):
        return client.fetch_solution_nmr_monomer_quality_records_for_ids(entry_ids)


def test_zero_metrics_from_the_validation_report_are_kept(tmp_path: Path) -> None:
    """A metric written as 0.00 in the report is a zero, not a missing value."""
    client = _client(
        tmp_path,
        {
            "1D3Z": _report(
                clashscore="3.07",
                percent_rama_outliers="0.00",
                percent_rota_outliers="1.67",
            )
        },
    )

    records = _fetch(client, ["1D3Z"])

    assert records == [
        builder.SolutionNMRMonomerQualityRecord("1D3Z", 2020, 3.07, 0.0, 1.67)
    ]
    client.session.get.assert_called_once()
    assert client.session.get.call_args.args[0] == (
        "https://files.rcsb.org/pub/pdb/validation_reports/d3/1d3z/"
        "1d3z_validation.xml.gz"
    )


def test_uncomputed_metric_or_missing_report_excludes_the_entry(
    tmp_path: Path,
) -> None:
    """An absent attribute means the metric was not computed; S3 denial, no report."""
    client = _client(
        tmp_path,
        {
            "2V93": _report(percent_rama_outliers="0.00"),
            "1BAD": _report(
                clashscore="x", percent_rama_outliers="1", percent_rota_outliers="2"
            ),
            "9NOR": None,
            "1HTM": gzip.compress(b"<html><body>Service unavailable</html>"),
            "1ZIP": b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\x03\xff\xff",
        },
    )

    with patch.object(nmr_client, "_record_filtered_structure") as record_filtered:
        records = _fetch(client, ["2V93", "1BAD", "9NOR", "1HTM", "1ZIP"])

    assert records == []
    reasons = {
        call.args[0]: call.args[1] for call in record_filtered.call_args_list
    }
    assert {key: reasons.pop(key) for key in ("2V93", "1BAD", "9NOR")} == {
        "2V93": "one or more validation quality metrics were not computed",
        "1BAD": "one or more validation quality metrics are invalid",
        "9NOR": "validation report is missing",
    }
    assert sorted(reasons) == ["1HTM", "1ZIP"]
    assert all(
        reason.startswith("validation report is unreadable: ")
        for reason in reasons.values()
    )
    # An unreadable download is not cached as a verified report.
    assert sorted(path.name for path in tmp_path.glob("*.gz*")) == [
        "1BAD_validation.xml.gz",
        "1BAD_validation.xml.gz.json",
        "2V93_validation.xml.gz",
        "2V93_validation.xml.gz.json",
    ]


def test_other_http_errors_stop_the_build(tmp_path: Path) -> None:
    """Only the S3 denial for an absent file counts as a missing report."""
    url = "https://files.rcsb.org/pub/pdb/validation_reports/bl/1blk/x"
    client = _client(
        tmp_path,
        {"1BLK": _response(url, 403, b"<html>Request blocked</html>")},
    )

    with pytest.raises(RuntimeError, match="403"):
        _fetch(client, ["1BLK"])


def test_cached_report_is_reused_and_refetched_when_its_checksum_differs(
    tmp_path: Path,
) -> None:
    """Every access verifies the cached report against its recorded SHA-256."""
    content = _report(
        clashscore="0.00", percent_rama_outliers="0.00", percent_rota_outliers="0.00"
    )
    client = _client(tmp_path, {"8T0G": content})

    first = _fetch(client, ["8T0G"])
    second = _fetch(client, ["8T0G"])
    assert first == second == [
        builder.SolutionNMRMonomerQualityRecord("8T0G", 2020, 0.0, 0.0, 0.0)
    ]
    assert client.session.get.call_count == 1

    cached = tmp_path / "8T0G_validation.xml.gz"
    cached.write_bytes(
        _report(clashscore="9", percent_rama_outliers="9", percent_rota_outliers="9")
    )
    assert _fetch(client, ["8T0G"]) == first
    assert client.session.get.call_count == 2
    assert cached.read_bytes() == content
