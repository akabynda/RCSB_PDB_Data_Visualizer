"""Atomic coordinate-cache transactions, validators, and per-entry locks."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from threading import Lock, RLock
from typing import Any

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback for library users.
    fcntl = None  # type: ignore[assignment]
import gzip
import hashlib
import json
import os
import tempfile
import uuid
from datetime import timezone
from threading import local

import requests

from src.dataset.config import LOGGER
from src.dataset.pdb_normalization import ensure_normalized_pdb

_PDB_CACHE_ENTRY_LOCKS_GUARD = Lock()


_PDB_CACHE_ENTRY_LOCKS: dict[tuple[str, str], tuple[Any, int]] = {}


_PDB_CACHE_ENTRY_LOCK_DEPTH = local()


def _pdb_cache_metadata_path(pdb_path: Path) -> Path:
    """Return the sidecar metadata path for a cached coordinate file."""
    return pdb_path.with_suffix(pdb_path.suffix + ".cache.json")


def _cache_revision(metadata: dict[str, Any] | None) -> str | None:
    """Return the transaction revision recorded by a coordinate sidecar."""
    if metadata is None:
        return None
    value = metadata.get("cache_revision")
    return value if isinstance(value, str) and value else None


@contextmanager
def _pdb_cache_entry_lock(cache_dir: Path, entry_id: str) -> Iterator[None]:
    """Serialize every cache artifact belonging to one PDB entry.

    The process-local lock prevents duplicate work between worker threads.  The
    persistent flock also protects a shared cache when two builder processes run
    at the same time.  Lock files are deliberately retained: unlinking a locked
    file can create two independently locked inodes.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    normalized_entry_id = entry_id.upper()
    resolved_cache_dir = str(cache_dir.resolve())
    key = (resolved_cache_dir, normalized_entry_id)
    with _PDB_CACHE_ENTRY_LOCKS_GUARD:
        state = _PDB_CACHE_ENTRY_LOCKS.get(key)
        if state is None:
            thread_lock = RLock()
            waiter_count = 0
        else:
            thread_lock, waiter_count = state
        _PDB_CACHE_ENTRY_LOCKS[key] = (thread_lock, waiter_count + 1)

    thread_lock.acquire()
    lock_handle: Any | None = None
    depths = getattr(_PDB_CACHE_ENTRY_LOCK_DEPTH, "depths", None)
    if depths is None:
        depths = {}
        _PDB_CACHE_ENTRY_LOCK_DEPTH.depths = depths
    previous_depth = int(depths.get(key, 0))
    depths[key] = previous_depth + 1
    is_outermost = previous_depth == 0
    try:
        if is_outermost:
            lock_dir = cache_dir / ".locks"
            lock_dir.mkdir(parents=True, exist_ok=True)
            lock_path = lock_dir / f"{normalized_entry_id}.lock"
            lock_handle = lock_path.open("a+b")
            if fcntl is not None:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            if lock_handle is not None and fcntl is not None:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        finally:
            if lock_handle is not None:
                lock_handle.close()
            if previous_depth == 0:
                depths.pop(key, None)
            else:
                depths[key] = previous_depth
            thread_lock.release()
            with _PDB_CACHE_ENTRY_LOCKS_GUARD:
                current_lock, current_count = _PDB_CACHE_ENTRY_LOCKS[key]
                if current_count <= 1:
                    del _PDB_CACHE_ENTRY_LOCKS[key]
                else:
                    _PDB_CACHE_ENTRY_LOCKS[key] = (
                        current_lock,
                        current_count - 1,
                    )


def _utc_now_iso() -> str:
    """Return a stable UTC timestamp for cache metadata."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_pdb_cache_metadata(pdb_path: Path) -> dict[str, Any] | None:
    """Load cache metadata, returning None for missing or damaged sidecars."""
    metadata_path = _pdb_cache_metadata_path(pdb_path)
    if not metadata_path.exists():
        return None
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _cached_pdb_matches_metadata(pdb_path: Path, metadata: dict[str, Any]) -> bool:
    """Return whether the local file still matches its recorded size and mtime."""
    if not pdb_path.exists() or pdb_path.stat().st_size <= 0:
        return False
    stat = pdb_path.stat()
    return (
        metadata.get("size_bytes") == stat.st_size
        and metadata.get("mtime_ns") == stat.st_mtime_ns
        and isinstance(metadata.get("sha256"), str)
        and bool(metadata["sha256"])
    )


def _pdb_cache_validation_is_fresh(
    metadata: dict[str, Any], validation_hours: float
) -> bool:
    """Return whether remote validation is still inside the configured window."""
    if validation_hours <= 0:
        return False
    raw_value = metadata.get("validated_at")
    if not isinstance(raw_value, str):
        return False
    try:
        validated_at = datetime.fromisoformat(raw_value.replace("Z", "+00:00"))
    except ValueError:
        return False
    if validated_at.tzinfo is None:
        validated_at = validated_at.replace(tzinfo=timezone.utc)
    age_seconds = (datetime.now(timezone.utc) - validated_at).total_seconds()
    return 0 <= age_seconds <= validation_hours * 3600.0


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write one JSON object using an atomic same-directory replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            suffix=".json.tmp",
            prefix=f"{path.name}.",
            dir=str(path.parent),
            encoding="utf-8",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temp_path.replace(path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _response_chunks(response: requests.Response) -> Iterator[bytes]:
    """Yield non-empty response body chunks without loading the whole file."""
    for chunk in response.iter_content(chunk_size=1024 * 1024):
        if chunk:
            yield chunk


def _close_http_response(response: Any | None) -> None:
    """Release a streaming HTTP response without masking the primary result."""
    close = getattr(response, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            LOGGER.debug("Could not close an HTTP response", exc_info=True)


def _atomic_install_pdb_response(
    response: requests.Response,
    pdb_path: Path,
    compressed: bool,
) -> tuple[str, int, int]:
    """Stream, validate, and atomically install a downloaded PDB response."""
    pdb_path.parent.mkdir(parents=True, exist_ok=True)
    download_path: Path | None = None
    output_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb",
            suffix=".download",
            prefix=f"{pdb_path.name}.",
            dir=str(pdb_path.parent),
            delete=False,
        ) as download_handle:
            download_path = Path(download_handle.name)
            for chunk in _response_chunks(response):
                download_handle.write(chunk)
            download_handle.flush()
            os.fsync(download_handle.fileno())

        digest = hashlib.sha256()
        size_bytes = 0
        with tempfile.NamedTemporaryFile(
            "wb",
            suffix=".pdb.tmp",
            prefix=f"{pdb_path.name}.",
            dir=str(pdb_path.parent),
            delete=False,
        ) as output_handle:
            output_path = Path(output_handle.name)
            source_handle: Any
            if compressed:
                source_handle = gzip.open(download_path, "rb")
            else:
                source_handle = download_path.open("rb")
            with source_handle:
                while True:
                    chunk = source_handle.read(1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
                    size_bytes += len(chunk)
                    output_handle.write(chunk)
            output_handle.flush()
            os.fsync(output_handle.fileno())

        if size_bytes <= 0:
            raise RuntimeError(f"Downloaded empty coordinate file for {pdb_path.stem}")
        if pdb_path.suffix.lower() == ".pdb":
            ensure_normalized_pdb(output_path)
            # The cache hash describes the coordinates all consumers receive,
            # including the conformer selection, rather than the HTTP payload.
            digest_hex = _sha256_file(output_path)
        else:
            digest_hex = digest.hexdigest()
        output_path.replace(pdb_path)
        stat = pdb_path.stat()
        return digest_hex, stat.st_size, stat.st_mtime_ns
    finally:
        if download_path is not None:
            download_path.unlink(missing_ok=True)
        if output_path is not None:
            output_path.unlink(missing_ok=True)


def _cache_metadata_from_response(
    entry_id: str,
    source_url: str,
    response: requests.Response,
    sha256: str,
    size_bytes: int,
    mtime_ns: int,
) -> dict[str, Any]:
    """Build the durable sidecar for one successfully installed PDB file."""
    return {
        "cache_revision": uuid.uuid4().hex,
        "entry_id": entry_id.upper(),
        "source_url": source_url,
        "etag": response.headers.get("ETag"),
        "last_modified": response.headers.get("Last-Modified"),
        "sha256": sha256,
        "size_bytes": size_bytes,
        "mtime_ns": mtime_ns,
        "validated_at": _utc_now_iso(),
    }


def _pdb_download_sources(entry_id: str) -> tuple[tuple[str, bool], ...]:
    """Return independent legacy-PDB download routes and compression flags."""
    normalized_entry_id = entry_id.upper()
    lower_entry_id = normalized_entry_id.lower()
    divided_directory = lower_entry_id[1:3]
    return (
        (
            f"https://files.rcsb.org/download/{normalized_entry_id}.pdb.gz",
            True,
        ),
        (
            "https://files.wwpdb.org/pub/pdb/data/structures/divided/pdb/"
            f"{divided_directory}/pdb{lower_entry_id}.ent.gz",
            True,
        ),
        (
            f"https://www.ebi.ac.uk/pdbe/entry-files/download/pdb{lower_entry_id}.ent",
            False,
        ),
        (
            "https://ftp.ebi.ac.uk/pub/databases/pdb/data/structures/divided/pdb/"
            f"{divided_directory}/pdb{lower_entry_id}.ent.gz",
            True,
        ),
    )


def _mmcif_download_sources(entry_id: str) -> tuple[tuple[str, bool], ...]:
    """Return independent mmCIF download routes and compression flags."""
    normalized_entry_id = entry_id.upper()
    lower_entry_id = normalized_entry_id.lower()
    divided_directory = lower_entry_id[1:3]
    return (
        (
            f"https://files.rcsb.org/download/{normalized_entry_id}.cif.gz",
            True,
        ),
        (
            "https://files.wwpdb.org/pub/pdb/data/structures/divided/mmCIF/"
            f"{divided_directory}/{lower_entry_id}.cif.gz",
            True,
        ),
        (
            f"https://www.ebi.ac.uk/pdbe/entry-files/download/{lower_entry_id}.cif",
            False,
        ),
        (
            "https://ftp.ebi.ac.uk/pub/databases/pdb/data/structures/divided/mmCIF/"
            f"{divided_directory}/{lower_entry_id}.cif.gz",
            True,
        ),
    )


def _prioritize_cached_source(
    sources: tuple[tuple[str, bool], ...],
    cached_source_url: str | None,
) -> tuple[tuple[str, bool], ...]:
    """Try the previously validated source first without duplicating a route."""
    if not cached_source_url:
        return sources
    cached_source = next(
        (source for source in sources if source[0] == cached_source_url), None
    )
    if cached_source is None:
        return sources
    return (cached_source,) + tuple(
        source for source in sources if source != cached_source
    )


def _sha256_file(path: Path) -> str:
    """Hash one file without loading it wholly into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()
