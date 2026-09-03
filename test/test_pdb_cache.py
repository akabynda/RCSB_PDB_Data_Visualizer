"""Tests for safe PDB and mmCIF coordinate-file caching."""

import gzip
import json
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
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


class _BlockingSameEntrySession:
    """Block concurrent requests so single-flight behavior can be observed."""

    def __init__(self, body: bytes) -> None:
        """Store one reusable response body and synchronization primitives."""
        self.body = body
        self.calls: list[dict[str, object]] = []
        self._calls_lock = threading.Lock()
        self.first_get_entered = threading.Event()
        self.release_gets = threading.Event()

    @property
    def call_count(self) -> int:
        """Return the number of GET calls recorded so far."""
        with self._calls_lock:
            return len(self.calls)

    def get(self, url: str, **kwargs: object) -> _FakeResponse:
        """Record and block a GET until the test permits it to complete."""
        with self._calls_lock:
            self.calls.append({"url": url, **kwargs})
            self.first_get_entered.set()
        if not self.release_gets.wait(timeout=5):
            raise AssertionError("Timed out waiting to release cache GET")
        return _FakeResponse(200, self.body, {"ETag": '"single-flight"'})


class _DifferentEntrySession:
    """Require two different PDB downloads to overlap in the HTTP layer."""

    def __init__(self) -> None:
        """Initialize entry-specific synchronization events and call storage."""
        self.calls: list[dict[str, object]] = []
        self._calls_lock = threading.Lock()
        self.first_entry_get_entered = threading.Event()
        self.second_entry_get_entered = threading.Event()

    def get(self, url: str, **kwargs: object) -> _FakeResponse:
        """Let 1ABC finish only after 2DEF has entered its own GET."""
        with self._calls_lock:
            self.calls.append({"url": url, **kwargs})
        if "1ABC" in url:
            self.first_entry_get_entered.set()
            if not self.second_entry_get_entered.wait(timeout=5):
                raise AssertionError(
                    "Different PDB entries were serialized by one global lock"
                )
            body = _compressed_pdb("CNS")
        elif "2DEF" in url:
            self.second_entry_get_entered.set()
            body = _compressed_pdb("CYANA")
        else:
            raise AssertionError(f"Unexpected PDB URL: {url}")
        return _FakeResponse(200, body)


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
    return gzip.compress((f"data_{entry_id}\n#\n_entry.id 1ABC\n#\n").encode("utf-8"))


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

    def test_legacy_pdb_install_removes_stale_converted_chain_map(self) -> None:
        """Do not pair a native legacy PDB with an obsolete mmCIF chain map."""
        map_path = self.cache_dir / "1ABC.chain_map.csv"
        map_path.write_text(
            "original_chain_id,mapped_chain_id\nlong_chain,A\n",
            encoding="utf-8",
        )
        session = _FakeSession([_FakeResponse(200, _compressed_pdb("CNS"))])

        download_pdb_if_needed(
            session=session,  # type: ignore[arg-type]
            config=DatasetBuildConfig(retries=1),
            cache_dir=self.cache_dir,
            entry_id="1ABC",
        )

        self.assertFalse(map_path.exists())

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

    def test_same_pdb_concurrent_download_is_single_flight_at_zero_hours(
        self,
    ) -> None:
        """Coalesce overlapping downloads without weakening zero-hour validation."""
        session = _BlockingSameEntrySession(_compressed_pdb("CNS"))
        config = DatasetBuildConfig(retries=1, pdb_cache_validation_hours=0)
        start = threading.Barrier(3)

        def download() -> Path:
            """Start both cache calls at the same synchronization barrier."""
            start.wait(timeout=5)
            return download_pdb_if_needed(
                session=session,  # type: ignore[arg-type]
                config=config,
                cache_dir=self.cache_dir,
                entry_id="1ABC",
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(download) for _ in range(2)]
            start.wait(timeout=5)
            self.assertTrue(session.first_get_entered.wait(timeout=5))

            # Give an unlocked duplicate enough time to reach its own GET. A
            # correctly keyed single-flight call remains behind the entry lock.
            duplicate_deadline = time.monotonic() + 0.5
            while session.call_count < 2 and time.monotonic() < duplicate_deadline:
                time.sleep(0.005)
            session.release_gets.set()
            paths = [future.result(timeout=5) for future in futures]

        self.assertEqual(paths, [self.cache_dir / "1ABC.pdb"] * 2)
        self.assertEqual(session.call_count, 1)
        self.assertIn("PROGRAM     : CNS", paths[0].read_text(encoding="utf-8"))

        metadata = json.loads(
            paths[0].with_suffix(".pdb.cache.json").read_text(encoding="utf-8")
        )
        self.assertEqual(metadata["size_bytes"], paths[0].stat().st_size)
        self.assertEqual(metadata["mtime_ns"], paths[0].stat().st_mtime_ns)

        # Zero hours still means a later, non-overlapping access revalidates.
        download_pdb_if_needed(
            session=session,  # type: ignore[arg-type]
            config=config,
            cache_dir=self.cache_dir,
            entry_id="1ABC",
        )
        self.assertEqual(session.call_count, 2)

    def test_different_pdb_downloads_are_not_globally_serialized(self) -> None:
        """Allow independent PDB IDs to download concurrently."""
        session = _DifferentEntrySession()
        config = DatasetBuildConfig(retries=1, pdb_cache_validation_hours=24)

        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(
                download_pdb_if_needed,
                session,  # type: ignore[arg-type]
                config,
                self.cache_dir,
                "1ABC",
            )
            self.assertTrue(session.first_entry_get_entered.wait(timeout=5))
            second = executor.submit(
                download_pdb_if_needed,
                session,  # type: ignore[arg-type]
                config,
                self.cache_dir,
                "2DEF",
            )
            first_path = first.result(timeout=5)
            second_path = second.result(timeout=5)

        self.assertTrue(session.second_entry_get_entered.is_set())
        self.assertEqual(len(session.calls), 2)
        self.assertIn("PROGRAM     : CNS", first_path.read_text(encoding="utf-8"))
        self.assertIn("PROGRAM     : CYANA", second_path.read_text(encoding="utf-8"))

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
                _FakeResponse(200, _compressed_mmcif(), {"ETag": '"wwpdb-mirror"'}),
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

    def test_same_chain_subset_build_is_single_flight_with_complete_map(
        self,
    ) -> None:
        """Publish one complete subset PDB, chain map, and metadata bundle."""
        source_pdb = self.cache_dir / "source.pdb"
        source_pdb.write_text(
            "ATOM      1  CA  ALA a   1       0.000   0.000   0.000  1.00 20.00           C  \n"
            "END\n",
            encoding="utf-8",
        )
        cif_path = self.cache_dir / "4W2E.cif"
        cif_path.write_text("test fixture", encoding="utf-8")
        parser_calls = 0
        parser_calls_lock = threading.Lock()
        first_parser_entered = threading.Event()
        release_parser = threading.Event()
        start = threading.Barrier(3)

        def metadata(path: Path) -> dict[str, object] | None:
            """Return source metadata or read a generated atomic sidecar."""
            if path == cif_path:
                return {
                    "sha256": "cif-sha256",
                    "source_url": "https://files.rcsb.org/download/4W2E.cif.gz",
                }
            metadata_path = path.with_suffix(path.suffix + ".cache.json")
            if not metadata_path.exists():
                return None
            return json.loads(metadata_path.read_text(encoding="utf-8"))

        def parse_structure(entry_id: str, path: Path) -> object:
            """Block the first subset conversion while a duplicate starts."""
            nonlocal parser_calls
            self.assertEqual(entry_id, "4W2E")
            self.assertEqual(path, cif_path)
            with parser_calls_lock:
                parser_calls += 1
                first_parser_entered.set()
            if not release_parser.wait(timeout=5):
                raise AssertionError("Timed out waiting to release subset parser")
            return parse_pdb_structure(entry_id, source_pdb)

        def build_subset() -> tuple[Path, dict[str, str]]:
            """Start both subset calls at the same synchronization barrier."""
            start.wait(timeout=5)
            return download_pdb_chain_subset_if_needed(
                session=_FakeSession([]),  # type: ignore[arg-type]
                config=DatasetBuildConfig(retries=1),
                cache_dir=self.cache_dir,
                entry_id="4W2E",
                chain_ids=("a",),
            )

        with (
            patch(
                "src.pdb_dataset_builder._download_pdb_if_needed_locked",
                side_effect=RuntimeError("legacy PDB unavailable"),
            ),
            patch(
                "src.pdb_dataset_builder._download_mmcif_if_needed_locked",
                return_value=cif_path,
            ),
            patch(
                "src.pdb_dataset_builder._load_pdb_cache_metadata",
                side_effect=metadata,
            ),
            patch(
                "src.pdb_dataset_builder.parse_mmcif_structure",
                side_effect=parse_structure,
            ),
            ThreadPoolExecutor(max_workers=2) as executor,
        ):
            futures = [executor.submit(build_subset) for _ in range(2)]
            start.wait(timeout=5)
            self.assertTrue(first_parser_entered.wait(timeout=5))
            duplicate_deadline = time.monotonic() + 0.5
            while parser_calls < 2 and time.monotonic() < duplicate_deadline:
                time.sleep(0.005)
            release_parser.set()
            results = [future.result(timeout=5) for future in futures]

        subset_path = results[0][0]
        self.assertEqual(results, [(subset_path, {"a": "a"})] * 2)
        self.assertEqual(parser_calls, 1)
        self.assertIn(" ALA a   1", subset_path.read_text(encoding="utf-8"))
        map_path = subset_path.with_name(
            subset_path.name.removesuffix(".pdb") + ".chain_map.csv"
        )
        self.assertEqual(
            map_path.read_text(encoding="utf-8"),
            "original_chain_id,mapped_chain_id\na,a\n",
        )
        subset_metadata = json.loads(
            subset_path.with_suffix(".pdb.cache.json").read_text(encoding="utf-8")
        )
        self.assertEqual(subset_metadata["source_cif_sha256"], "cif-sha256")
        self.assertEqual(subset_metadata["chain_ids"], ["a"])
        self.assertEqual(subset_metadata["size_bytes"], subset_path.stat().st_size)
        self.assertEqual(subset_metadata["mtime_ns"], subset_path.stat().st_mtime_ns)

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
