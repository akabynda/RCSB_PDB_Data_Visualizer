"""Tests for safe PDB and mmCIF coordinate-file caching."""

import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import requests

from src.pdb_dataset_builder import (
    DatasetBuildConfig,
    _download_mmcif_if_needed,
    download_pdb_chain_subset_if_needed,
    download_pdb_if_needed,
    parse_pdb_structure,
)


class _FakeResponse:
    """Provide the subset of an HTTP response needed by cache tests."""

    def __init__(
        self,
        status_code: int,
        body: bytes = b"",
        headers: dict[str, str] | None = None,
        fail_after_first_chunk: bool = False,
    ) -> None:
        """Initialize response bytes, status, headers, and failure behavior."""
        self.status_code = status_code
        self.body = body
        self.headers = headers or {}
        self.fail_after_first_chunk = fail_after_first_chunk

    def iter_content(self, chunk_size: int) -> object:
        """Yield response bytes or raise the configured stream failure."""
        midpoint = max(1, len(self.body) // 2)
        yield self.body[:midpoint]
        if self.fail_after_first_chunk:
            raise OSError("simulated interrupted download")
        yield self.body[midpoint:]

    def raise_for_status(self) -> None:
        """Raise an HTTP error for status codes of 400 or greater."""
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class _FakeSession:
    """Return queued fake responses for deterministic cache requests."""

    def __init__(self, responses: list[_FakeResponse | Exception]) -> None:
        """Store the ordered responses or exceptions for subsequent requests."""
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def get(self, url: str, **kwargs: object) -> _FakeResponse:
        """Record a GET request and return the next configured response."""
        self.calls.append({"url": url, **kwargs})
        if not self.responses:
            raise AssertionError("Unexpected cache HTTP request")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _compressed_pdb(program: str) -> bytes:
    """Return gzipped minimal PDB content containing ``program`` metadata."""
    return gzip.compress(
        (
            "HEADER    CACHE TEST\n"
            f"REMARK   3   PROGRAM     : {program}\n"
            "ATOM      1  CA  ALA A   1       0.000   0.000   0.000\n"
            "END\n"
        ).encode("utf-8")
    )


def _compressed_mmcif(entry_id: str = "1ABC") -> bytes:
    """Return gzipped minimal mmCIF content for ``entry_id``."""
    return gzip.compress(
        (
            f"data_{entry_id}\n"
            "#\n"
            "_entry.id 1ABC\n"
            "#\n"
        ).encode("utf-8")
    )


class SafePDBCacheTests(unittest.TestCase):
    """Verify download, validation, fallback, and atomic cache behavior."""

    def setUp(self) -> None:
        """Create a temporary cache directory for each test."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.cache_dir = Path(self.temp_dir.name)

    def test_initial_download_writes_file_and_versioned_metadata(self) -> None:
        """Write coordinate data and schema-versioned metadata initially."""
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
        """Reuse a cache entry whose validation metadata is still fresh."""
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
        """Revalidate stale data conditionally and accept not-modified responses."""
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
        """Replace stale cached content when the remote resource changed."""
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
        """Try the next PDB download route immediately after a timeout."""
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

    def test_mmcif_timeout_switches_to_another_route_immediately(self) -> None:
        """Try the next mmCIF route immediately after a timeout."""
        session = _FakeSession(
            [
                requests.ReadTimeout("RCSB route timed out"),
                _FakeResponse(
                    200, _compressed_mmcif(), {"ETag": '"wwpdb-mirror"'}
                ),
            ]
        )

        path = _download_mmcif_if_needed(
            session=session,  # type: ignore[arg-type]
            config=DatasetBuildConfig(retries=1),
            cache_dir=self.cache_dir,
            entry_id="1ABC",
        )

        self.assertIn("data_1ABC", path.read_text(encoding="utf-8"))
        self.assertIn("files.rcsb.org", str(session.calls[0]["url"]))
        self.assertIn("files.wwpdb.org", str(session.calls[1]["url"]))
        metadata = json.loads(
            path.with_suffix(".cif.cache.json").read_text(encoding="utf-8")
        )
        self.assertEqual(metadata["source_url"], session.calls[1]["url"])

    def test_mmcif_fallback_accepts_uncompressed_pdbe_response(self) -> None:
        """Accept plain-text mmCIF content from the PDBe fallback route."""
        session = _FakeSession(
            [
                requests.ConnectionError("RCSB unavailable"),
                requests.ConnectionError("wwPDB unavailable"),
                _FakeResponse(200, gzip.decompress(_compressed_mmcif())),
            ]
        )

        path = _download_mmcif_if_needed(
            session=session,  # type: ignore[arg-type]
            config=DatasetBuildConfig(retries=1),
            cache_dir=self.cache_dir,
            entry_id="1ABC",
        )

        self.assertIn("data_1ABC", path.read_text(encoding="utf-8"))
        self.assertIn("www.ebi.ac.uk/pdbe", str(session.calls[2]["url"]))

    def test_single_character_chain_falls_back_to_mmcif_subset(self) -> None:
        """Create a PDB-compatible subset from mmCIF for one-character chains."""
        source_pdb = self.cache_dir / "source.pdb"
        source_pdb.write_text(
            "ATOM      1  CA  ALA a   1       0.000   0.000   0.000  1.00 20.00           C  \n"
            "END\n",
            encoding="utf-8",
        )
        structure = parse_pdb_structure("4W2E", source_pdb)
        cif_path = self.cache_dir / "4W2E.cif"
        cif_path.write_text("test fixture", encoding="utf-8")

        def metadata(path: Path) -> dict[str, object] | None:
            """Read cache metadata from ``path`` when the file exists."""
            if path == cif_path:
                return {
                    "sha256": "cif-sha256",
                    "source_url": "https://files.rcsb.org/download/4W2E.cif.gz",
                }
            return None

        with (
            patch(
                "src.pdb_dataset_builder.download_pdb_if_needed",
                side_effect=RuntimeError(
                    "Atom serial number ('100000') exceeds PDB format limit"
                ),
            ),
            patch(
                "src.pdb_dataset_builder._download_mmcif_if_needed",
                return_value=cif_path,
            ),
            patch(
                "src.pdb_dataset_builder._load_pdb_cache_metadata",
                side_effect=metadata,
            ),
            patch(
                "src.pdb_dataset_builder.parse_mmcif_structure",
                return_value=structure,
            ),
        ):
            subset_path, chain_map = download_pdb_chain_subset_if_needed(
                session=_FakeSession([]),  # type: ignore[arg-type]
                config=DatasetBuildConfig(retries=1),
                cache_dir=self.cache_dir,
                entry_id="4W2E",
                chain_ids=("a",),
            )

        self.assertTrue(subset_path.exists())
        self.assertEqual(chain_map, {"a": "a"})
        self.assertIn(" ALA a   1", subset_path.read_text(encoding="utf-8"))

    def test_interrupted_refresh_keeps_previous_file_intact(self) -> None:
        """Keep valid cached data intact when a refresh stream is interrupted."""
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
