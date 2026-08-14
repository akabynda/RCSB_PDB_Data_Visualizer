import gzip
import json
import tempfile
import unittest
from pathlib import Path

import requests

from src.pdb_dataset_builder import DatasetBuildConfig, download_pdb_if_needed


class _FakeResponse:
    def __init__(
        self,
        status_code: int,
        body: bytes = b"",
        headers: dict[str, str] | None = None,
        fail_after_first_chunk: bool = False,
    ) -> None:
        self.status_code = status_code
        self.body = body
        self.headers = headers or {}
        self.fail_after_first_chunk = fail_after_first_chunk

    def iter_content(self, chunk_size: int) -> object:
        midpoint = max(1, len(self.body) // 2)
        yield self.body[:midpoint]
        if self.fail_after_first_chunk:
            raise OSError("simulated interrupted download")
        yield self.body[midpoint:]

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class _FakeSession:
    def __init__(self, responses: list[_FakeResponse | Exception]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def get(self, url: str, **kwargs: object) -> _FakeResponse:
        self.calls.append({"url": url, **kwargs})
        if not self.responses:
            raise AssertionError("Unexpected cache HTTP request")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _compressed_pdb(program: str) -> bytes:
    return gzip.compress(
        (
            "HEADER    CACHE TEST\n"
            f"REMARK   3   PROGRAM     : {program}\n"
            "ATOM      1  CA  ALA A   1       0.000   0.000   0.000\n"
            "END\n"
        ).encode("utf-8")
    )


class SafePDBCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.cache_dir = Path(self.temp_dir.name)

    def test_initial_download_writes_file_and_versioned_metadata(self) -> None:
        session = _FakeSession(
            [
                _FakeResponse(
                    200,
                    _compressed_pdb("CNS"),
                    {"ETag": '"v1"', "Last-Modified": "Wed, 01 Jan 2025 00:00:00 GMT"},
                )
            ]
        )

        path = download_pdb_if_needed(
            session=session,  # type: ignore[arg-type]
            config=DatasetBuildConfig(retries=1),
            cache_dir=self.cache_dir,
            entry_id="1abc",
        )

        self.assertIn("PROGRAM     : CNS", path.read_text(encoding="utf-8"))
        metadata = json.loads(
            path.with_suffix(".pdb.cache.json").read_text(encoding="utf-8")
        )
        self.assertEqual(metadata["entry_id"], "1ABC")
        self.assertEqual(metadata["etag"], '"v1"')
        self.assertEqual(metadata["size_bytes"], path.stat().st_size)
        self.assertEqual(len(metadata["sha256"]), 64)

    def test_fresh_metadata_avoids_another_remote_request(self) -> None:
        session = _FakeSession([_FakeResponse(200, _compressed_pdb("CNS"))])
        config = DatasetBuildConfig(retries=1, pdb_cache_validation_hours=24)
        first = download_pdb_if_needed(
            session=session,  # type: ignore[arg-type]
            config=config,
            cache_dir=self.cache_dir,
            entry_id="1ABC",
        )
        second = download_pdb_if_needed(
            session=session,  # type: ignore[arg-type]
            config=config,
            cache_dir=self.cache_dir,
            entry_id="1ABC",
        )

        self.assertEqual(first, second)
        self.assertEqual(len(session.calls), 1)

    def test_stale_metadata_uses_conditional_get_and_accepts_304(self) -> None:
        session = _FakeSession(
            [
                _FakeResponse(200, _compressed_pdb("CNS"), {"ETag": '"v1"'}),
                _FakeResponse(304),
            ]
        )
        config = DatasetBuildConfig(retries=1, pdb_cache_validation_hours=0)
        path = download_pdb_if_needed(
            session=session,  # type: ignore[arg-type]
            config=config,
            cache_dir=self.cache_dir,
            entry_id="1ABC",
        )
        original_bytes = path.read_bytes()

        download_pdb_if_needed(
            session=session,  # type: ignore[arg-type]
            config=config,
            cache_dir=self.cache_dir,
            entry_id="1ABC",
        )

        self.assertEqual(path.read_bytes(), original_bytes)
        self.assertEqual(session.calls[1]["headers"], {"If-None-Match": '"v1"'})

    def test_changed_remote_file_atomically_replaces_cached_file(self) -> None:
        session = _FakeSession(
            [
                _FakeResponse(200, _compressed_pdb("CNS"), {"ETag": '"v1"'}),
                _FakeResponse(200, _compressed_pdb("CYANA"), {"ETag": '"v2"'}),
            ]
        )
        config = DatasetBuildConfig(retries=1, pdb_cache_validation_hours=0)
        path = download_pdb_if_needed(
            session=session,  # type: ignore[arg-type]
            config=config,
            cache_dir=self.cache_dir,
            entry_id="1ABC",
        )
        download_pdb_if_needed(
            session=session,  # type: ignore[arg-type]
            config=config,
            cache_dir=self.cache_dir,
            entry_id="1ABC",
        )

        self.assertIn("PROGRAM     : CYANA", path.read_text(encoding="utf-8"))
        metadata = json.loads(
            path.with_suffix(".pdb.cache.json").read_text(encoding="utf-8")
        )
        self.assertEqual(metadata["etag"], '"v2"')

    def test_timeout_switches_to_another_download_route_immediately(self) -> None:
        session = _FakeSession(
            [
                requests.ReadTimeout("primary route timed out"),
                _FakeResponse(200, _compressed_pdb("CNS"), {"ETag": '"mirror"'}),
            ]
        )

        path = download_pdb_if_needed(
            session=session,  # type: ignore[arg-type]
            config=DatasetBuildConfig(retries=1),
            cache_dir=self.cache_dir,
            entry_id="11TJ",
        )

        self.assertIn("PROGRAM     : CNS", path.read_text(encoding="utf-8"))
        self.assertIn("files.rcsb.org", str(session.calls[0]["url"]))
        self.assertIn("files.wwpdb.org", str(session.calls[1]["url"]))
        metadata = json.loads(
            path.with_suffix(".pdb.cache.json").read_text(encoding="utf-8")
        )
        self.assertEqual(metadata["source_url"], session.calls[1]["url"])

    def test_interrupted_refresh_keeps_previous_file_intact(self) -> None:
        session = _FakeSession(
            [
                _FakeResponse(200, _compressed_pdb("CNS"), {"ETag": '"v1"'}),
                _FakeResponse(
                    200,
                    _compressed_pdb("CYANA"),
                    {"ETag": '"v2"'},
                    fail_after_first_chunk=True,
                ),
                requests.ConnectionError("wwPDB unavailable"),
                requests.ConnectionError("PDBe unavailable"),
                requests.ConnectionError("EBI archive unavailable"),
            ]
        )
        config = DatasetBuildConfig(retries=1, pdb_cache_validation_hours=0)
        path = download_pdb_if_needed(
            session=session,  # type: ignore[arg-type]
            config=config,
            cache_dir=self.cache_dir,
            entry_id="1ABC",
        )
        original_bytes = path.read_bytes()

        with self.assertRaises(RuntimeError):
            download_pdb_if_needed(
                session=session,  # type: ignore[arg-type]
                config=config,
                cache_dir=self.cache_dir,
                entry_id="1ABC",
            )

        self.assertEqual(path.read_bytes(), original_bytes)


if __name__ == "__main__":
    unittest.main()
