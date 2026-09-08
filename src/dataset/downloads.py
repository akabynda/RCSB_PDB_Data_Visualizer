"""Download and convert cached PDB/mmCIF coordinates and chain subsets."""

from __future__ import annotations

import gzip
import hashlib
import os
import tempfile
import time
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import requests
from Bio.PDB import PDBIO

from src.dataset.cache import (
    _atomic_install_pdb_response,
    _atomic_write_json,
    _cache_metadata_from_response,
    _cache_revision,
    _cached_pdb_matches_metadata,
    _close_http_response,
    _load_pdb_cache_metadata,
    _mmcif_download_sources,
    _pdb_cache_entry_lock,
    _pdb_cache_metadata_path,
    _pdb_cache_validation_is_fresh,
    _pdb_download_sources,
    _prioritize_cached_source,
    _response_chunks,
    _sha256_file,
    _utc_now_iso,
)
from src.dataset.config import LOGGER
from src.dataset.io.common import (
    _atomic_write_csv_rows,
)
from src.dataset.programs import (
    _prepend_mmcif_software_remarks,
)
from src.dataset.pdb_normalization import ensure_normalized_pdb
from src.dataset.structures import (
    ChainSubsetSelect,
    _chain_subset_cache_stem,
    _coerce_selected_structure_chain_ids_for_pdbio,
    _coerce_structure_chain_ids_for_pdbio,
    _prepend_mmcif_polymer_remarks,
    load_cached_chain_id_map,
    parse_mmcif_structure,
)

if TYPE_CHECKING:
    from src.dataset.config import (
        DatasetBuildConfig,
    )


def download_pdb_if_needed(
    session: requests.Session,
    config: DatasetBuildConfig,
    cache_dir: Path,
    entry_id: str,
) -> Path:
    """Download or remotely revalidate one single-flight cached PDB file."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    normalized_entry_id = entry_id.upper()
    path = cache_dir / f"{normalized_entry_id}.pdb"
    observed_revision = _cache_revision(_load_pdb_cache_metadata(path))
    with _pdb_cache_entry_lock(cache_dir, normalized_entry_id):
        return _download_pdb_if_needed_locked(
            session=session,
            config=config,
            cache_dir=cache_dir,
            entry_id=normalized_entry_id,
            observed_revision=observed_revision,
        )


def _download_pdb_if_needed_locked(
    session: requests.Session,
    config: DatasetBuildConfig,
    cache_dir: Path,
    entry_id: str,
    observed_revision: str | None,
) -> Path:
    """Download a PDB while the entry bundle lock is already held."""
    normalized_entry_id = entry_id.upper()
    path = cache_dir / f"{normalized_entry_id}.pdb"
    metadata_path = _pdb_cache_metadata_path(path)
    metadata = _load_pdb_cache_metadata(path)
    cache_is_valid = metadata is not None and _cached_pdb_matches_metadata(
        path, metadata
    )
    current_revision = _cache_revision(metadata)
    if cache_is_valid and metadata is not None:
        metadata = _normalize_cached_pdb(path, metadata)
    if (
        cache_is_valid
        and current_revision is not None
        and current_revision != observed_revision
    ):
        return path
    if cache_is_valid and _pdb_cache_validation_is_fresh(
        metadata, config.pdb_cache_validation_hours
    ):
        return path

    cached_source_url = (
        str(metadata.get("source_url"))
        if cache_is_valid and metadata and metadata.get("source_url")
        else None
    )
    sources = _prioritize_cached_source(
        _pdb_download_sources(normalized_entry_id), cached_source_url
    )
    conditional_headers: dict[str, str] = {}
    if cache_is_valid and metadata is not None:
        if metadata.get("etag"):
            conditional_headers["If-None-Match"] = str(metadata["etag"])
        if metadata.get("last_modified"):
            conditional_headers["If-Modified-Since"] = str(metadata["last_modified"])

    last_error: Exception | None = None
    saw_not_found = False
    unavailable_sources: set[str] = set()
    for attempt in range(1, config.retries + 1):
        for source_url, compressed in sources:
            if source_url in unavailable_sources:
                continue
            request_headers = (
                conditional_headers if source_url == cached_source_url else {}
            )
            response: Any | None = None
            try:
                response = session.get(
                    source_url,
                    headers=request_headers,
                    stream=True,
                    timeout=config.timeout_seconds,
                )
                if (
                    response.status_code == 304
                    and cache_is_valid
                    and metadata is not None
                    and source_url == cached_source_url
                ):
                    refreshed_metadata = dict(metadata)
                    refreshed_metadata["validated_at"] = _utc_now_iso()
                    refreshed_metadata["cache_revision"] = uuid.uuid4().hex
                    _atomic_write_json(metadata_path, refreshed_metadata)
                    _close_http_response(response)
                    return path
                if response.status_code == 404:
                    saw_not_found = True
                    unavailable_sources.add(source_url)
                    last_error = requests.HTTPError(
                        f"404 Client Error: Not Found for url: {source_url}"
                    )
                    _close_http_response(response)
                    continue
                response.raise_for_status()
                sha256, size_bytes, mtime_ns = _atomic_install_pdb_response(
                    response=response,
                    pdb_path=path,
                    compressed=compressed,
                )
                (cache_dir / f"{normalized_entry_id}.chain_map.csv").unlink(
                    missing_ok=True
                )
                _atomic_write_json(
                    metadata_path,
                    _cache_metadata_from_response(
                        entry_id=normalized_entry_id,
                        source_url=source_url,
                        response=response,
                        sha256=sha256,
                        size_bytes=size_bytes,
                        mtime_ns=mtime_ns,
                    ),
                )
                _close_http_response(response)
                return path
            except (
                requests.RequestException,
                OSError,
                EOFError,
                gzip.BadGzipFile,
                RuntimeError,
            ) as exc:
                _close_http_response(response)
                last_error = exc
                LOGGER.debug(
                    "PDB download route failed for %s (%s): %s",
                    normalized_entry_id,
                    source_url,
                    exc,
                )
        if attempt < config.retries:
            time.sleep(config.backoff_seconds * attempt)

    if saw_not_found:
        cif_path = cache_dir / f"{normalized_entry_id}.cif"
        cif_url = f"https://files.rcsb.org/download/{normalized_entry_id}.cif"
        LOGGER.info(
            "PDB file is unavailable for %s; trying mmCIF fallback",
            normalized_entry_id,
        )
        for attempt in range(1, config.retries + 1):
            response: Any | None = None
            try:
                response = session.get(
                    cif_url,
                    stream=True,
                    timeout=config.timeout_seconds,
                )
                response.raise_for_status()
                temp_cif_path: Path | None = None
                try:
                    with tempfile.NamedTemporaryFile(
                        "wb",
                        suffix=".cif.tmp",
                        prefix=f"{normalized_entry_id}.",
                        dir=str(cache_dir),
                        delete=False,
                    ) as handle:
                        temp_cif_path = Path(handle.name)
                        for chunk in _response_chunks(response):
                            handle.write(chunk)
                        handle.flush()
                        os.fsync(handle.fileno())
                    temp_cif_path.replace(cif_path)
                finally:
                    if temp_cif_path is not None:
                        temp_cif_path.unlink(missing_ok=True)
                structure = parse_mmcif_structure(normalized_entry_id, cif_path)
                chain_id_map = _coerce_structure_chain_ids_for_pdbio(structure)
                io = PDBIO()
                io.set_structure(structure)
                temp_pdb_path: Path | None = None
                try:
                    with tempfile.NamedTemporaryFile(
                        "wb",
                        suffix=".pdb.tmp",
                        prefix=f".{normalized_entry_id}.",
                        dir=str(cache_dir),
                        delete=False,
                    ) as pdb_handle:
                        temp_pdb_path = Path(pdb_handle.name)
                    io.save(str(temp_pdb_path))
                    _prepend_mmcif_software_remarks(temp_pdb_path, cif_path)
                    _prepend_mmcif_polymer_remarks(
                        temp_pdb_path, cif_path, chain_id_map
                    )
                    ensure_normalized_pdb(temp_pdb_path)
                    temp_pdb_path.replace(path)
                finally:
                    if temp_pdb_path is not None:
                        temp_pdb_path.unlink(missing_ok=True)
                if path.exists() and path.stat().st_size > 0:
                    if chain_id_map:
                        map_path = cache_dir / f"{normalized_entry_id}.chain_map.csv"
                        _atomic_write_csv_rows(
                            output_path=map_path,
                            header=["original_chain_id", "mapped_chain_id"],
                            rows=sorted(chain_id_map.items()),
                        )
                    else:
                        (cache_dir / f"{normalized_entry_id}.chain_map.csv").unlink(
                            missing_ok=True
                        )
                    LOGGER.info(
                        "Converted mmCIF fallback to cached PDB for %s",
                        normalized_entry_id,
                    )
                    pdb_bytes = path.read_bytes()
                    stat = path.stat()
                    fallback_metadata = _cache_metadata_from_response(
                        entry_id=normalized_entry_id,
                        source_url=cif_url,
                        response=response,
                        sha256=hashlib.sha256(pdb_bytes).hexdigest(),
                        size_bytes=stat.st_size,
                        mtime_ns=stat.st_mtime_ns,
                    )
                    fallback_metadata["chain_id_map"] = chain_id_map
                    _atomic_write_json(metadata_path, fallback_metadata)
                    _close_http_response(response)
                    return path
                _close_http_response(response)
            except Exception as exc:
                _close_http_response(response)
                last_error = exc
                wait_seconds = config.backoff_seconds * attempt
                if attempt < config.retries:
                    time.sleep(wait_seconds)
    raise RuntimeError(f"Failed to download {normalized_entry_id}: {last_error}")


def _download_mmcif_if_needed(
    session: requests.Session,
    config: DatasetBuildConfig,
    cache_dir: Path,
    entry_id: str,
) -> Path:
    """Download or remotely revalidate one single-flight cached mmCIF file."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    normalized_entry_id = entry_id.upper()
    cif_path = cache_dir / f"{normalized_entry_id}.cif"
    observed_revision = _cache_revision(_load_pdb_cache_metadata(cif_path))
    with _pdb_cache_entry_lock(cache_dir, normalized_entry_id):
        return _download_mmcif_if_needed_locked(
            session=session,
            config=config,
            cache_dir=cache_dir,
            entry_id=normalized_entry_id,
            observed_revision=observed_revision,
        )


def _download_mmcif_if_needed_locked(
    session: requests.Session,
    config: DatasetBuildConfig,
    cache_dir: Path,
    entry_id: str,
    observed_revision: str | None,
) -> Path:
    """Download an mmCIF while the entry bundle lock is already held."""
    normalized_entry_id = entry_id.upper()
    cif_path = cache_dir / f"{normalized_entry_id}.cif"
    metadata_path = _pdb_cache_metadata_path(cif_path)
    metadata = _load_pdb_cache_metadata(cif_path)
    cache_is_valid = metadata is not None and _cached_pdb_matches_metadata(
        cif_path, metadata
    )
    current_revision = _cache_revision(metadata)
    if (
        cache_is_valid
        and current_revision is not None
        and current_revision != observed_revision
    ):
        return cif_path
    if cache_is_valid and _pdb_cache_validation_is_fresh(
        metadata, config.pdb_cache_validation_hours
    ):
        return cif_path

    cached_source_url = (
        str(metadata.get("source_url"))
        if cache_is_valid and metadata and metadata.get("source_url")
        else None
    )
    sources = _prioritize_cached_source(
        _mmcif_download_sources(normalized_entry_id), cached_source_url
    )
    conditional_headers: dict[str, str] = {}
    if cache_is_valid and metadata is not None:
        if metadata.get("etag"):
            conditional_headers["If-None-Match"] = str(metadata["etag"])
        if metadata.get("last_modified"):
            conditional_headers["If-Modified-Since"] = str(metadata["last_modified"])

    last_error: Exception | None = None
    unavailable_sources: set[str] = set()
    for attempt in range(1, config.retries + 1):
        for source_url, compressed in sources:
            if source_url in unavailable_sources:
                continue
            request_headers = (
                conditional_headers if source_url == cached_source_url else {}
            )
            response: Any | None = None
            try:
                response = session.get(
                    source_url,
                    headers=request_headers,
                    stream=True,
                    timeout=config.timeout_seconds,
                )
                if (
                    response.status_code == 304
                    and cache_is_valid
                    and metadata is not None
                    and source_url == cached_source_url
                ):
                    refreshed_metadata = dict(metadata)
                    refreshed_metadata["validated_at"] = _utc_now_iso()
                    refreshed_metadata["cache_revision"] = uuid.uuid4().hex
                    _atomic_write_json(metadata_path, refreshed_metadata)
                    _close_http_response(response)
                    return cif_path
                if response.status_code == 404:
                    unavailable_sources.add(source_url)
                    last_error = requests.HTTPError(
                        f"404 Client Error: Not Found for url: {source_url}"
                    )
                    _close_http_response(response)
                    continue
                response.raise_for_status()
                sha256, size_bytes, mtime_ns = _atomic_install_pdb_response(
                    response=response,
                    pdb_path=cif_path,
                    compressed=compressed,
                )
                _atomic_write_json(
                    metadata_path,
                    _cache_metadata_from_response(
                        entry_id=normalized_entry_id,
                        source_url=source_url,
                        response=response,
                        sha256=sha256,
                        size_bytes=size_bytes,
                        mtime_ns=mtime_ns,
                    ),
                )
                _close_http_response(response)
                return cif_path
            except (
                requests.RequestException,
                OSError,
                EOFError,
                gzip.BadGzipFile,
                RuntimeError,
            ) as exc:
                _close_http_response(response)
                last_error = exc
                LOGGER.debug(
                    "mmCIF download route failed for %s (%s): %s",
                    normalized_entry_id,
                    source_url,
                    exc,
                )
        if attempt < config.retries:
            time.sleep(config.backoff_seconds * attempt)
    raise RuntimeError(f"Failed to download {normalized_entry_id} mmCIF: {last_error}")


def download_pdb_chain_subset_if_needed(
    session: requests.Session,
    config: DatasetBuildConfig,
    cache_dir: Path,
    entry_id: str,
    chain_ids: Sequence[str],
) -> tuple[Path, dict[str, str]]:
    """Create a single-flight, source-hash-bound selected-chain PDB cache."""
    selected_chain_ids = {str(chain_id) for chain_id in chain_ids if str(chain_id)}
    if not selected_chain_ids:
        raise RuntimeError(f"No chain IDs selected for {entry_id}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    normalized_entry_id = entry_id.upper()
    stem = _chain_subset_cache_stem(normalized_entry_id, sorted(selected_chain_ids))
    subset_path = cache_dir / f"{stem}.pdb"
    observed_subset_revision = _cache_revision(_load_pdb_cache_metadata(subset_path))
    with _pdb_cache_entry_lock(cache_dir, normalized_entry_id):
        cif_path = cache_dir / f"{normalized_entry_id}.cif"
        current_subset_metadata = _load_pdb_cache_metadata(subset_path)
        current_subset_revision = _cache_revision(current_subset_metadata)
        cached_subset = _load_valid_cached_chain_subset(
            subset_path=subset_path,
            cif_path=cif_path,
            selected_chain_ids=selected_chain_ids,
        )
        cif_metadata = _load_pdb_cache_metadata(cif_path)
        if cached_subset is not None and (
            (
                current_subset_revision is not None
                and current_subset_revision != observed_subset_revision
            )
            or (
                cif_metadata is not None
                and _pdb_cache_validation_is_fresh(
                    cif_metadata, config.pdb_cache_validation_hours
                )
            )
        ):
            return cached_subset

        if all(len(chain_id) == 1 for chain_id in selected_chain_ids):
            try:
                full_pdb_path = download_pdb_if_needed(
                    session=session,
                    config=config,
                    cache_dir=cache_dir,
                    entry_id=normalized_entry_id,
                )
            except RuntimeError as exc:
                LOGGER.info(
                    "Full PDB is unavailable for %s (%s); creating an mmCIF chain subset",
                    normalized_entry_id,
                    exc,
                )
            else:
                return full_pdb_path, load_cached_chain_id_map(
                    cache_dir, normalized_entry_id
                )

        cif_path = _download_mmcif_if_needed(
            session=session,
            config=config,
            cache_dir=cache_dir,
            entry_id=normalized_entry_id,
        )
        return _download_pdb_chain_subset_if_needed_locked(
            cache_dir=cache_dir,
            entry_id=normalized_entry_id,
            selected_chain_ids=selected_chain_ids,
            stem=stem,
            cif_path=cif_path,
        )


def _load_valid_cached_chain_subset(
    *,
    subset_path: Path,
    cif_path: Path,
    selected_chain_ids: set[str],
) -> tuple[Path, dict[str, str]] | None:
    """Load a complete subset bundle bound to the current mmCIF snapshot."""
    subset_metadata = _load_pdb_cache_metadata(subset_path)
    cif_metadata = _load_pdb_cache_metadata(cif_path)
    if subset_metadata is None or cif_metadata is None:
        return None
    cached_chain_map = subset_metadata.get("chain_id_map")
    if not (
        _cached_pdb_matches_metadata(subset_path, subset_metadata)
        and subset_metadata.get("source_cif_sha256") == cif_metadata.get("sha256")
        and subset_metadata.get("chain_ids") == sorted(selected_chain_ids)
        and isinstance(cached_chain_map, dict)
        and set(cached_chain_map) == selected_chain_ids
        and all(str(value) for value in cached_chain_map.values())
    ):
        return None
    _normalize_cached_pdb(subset_path, subset_metadata)
    return subset_path, {
        str(original): str(mapped) for original, mapped in cached_chain_map.items()
    }


def _download_pdb_chain_subset_if_needed_locked(
    cache_dir: Path,
    entry_id: str,
    selected_chain_ids: set[str],
    stem: str,
    cif_path: Path,
) -> tuple[Path, dict[str, str]]:
    """Build a selected-chain PDB while the entry bundle lock is held."""
    path = cache_dir / f"{stem}.pdb"
    map_path = cache_dir / f"{stem}.chain_map.csv"
    metadata_path = _pdb_cache_metadata_path(path)
    cif_metadata = _load_pdb_cache_metadata(cif_path)
    if cif_metadata is None:
        raise RuntimeError(f"Missing validated mmCIF metadata for {entry_id}")
    cached_subset = _load_valid_cached_chain_subset(
        subset_path=path,
        cif_path=cif_path,
        selected_chain_ids=selected_chain_ids,
    )
    if cached_subset is not None:
        return cached_subset

    structure = parse_mmcif_structure(entry_id, cif_path)
    chain_id_map, selected_chain_object_ids = (
        _coerce_selected_structure_chain_ids_for_pdbio(
            structure=structure,
            selected_chain_ids=selected_chain_ids,
        )
    )
    missing_chain_ids = selected_chain_ids - set(chain_id_map)
    if missing_chain_ids:
        raise RuntimeError(
            f"{entry_id} mmCIF is missing selected chains: "
            + ",".join(sorted(missing_chain_ids))
        )
    io = PDBIO()
    io.set_structure(structure)
    temp_subset_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb",
            suffix=".pdb.tmp",
            prefix=f".{stem}.",
            dir=str(cache_dir),
            delete=False,
        ) as handle:
            temp_subset_path = Path(handle.name)
        io.save(
            str(temp_subset_path),
            select=ChainSubsetSelect(selected_chain_object_ids),
        )
        _prepend_mmcif_software_remarks(temp_subset_path, cif_path)
        _prepend_mmcif_polymer_remarks(
            temp_subset_path, cif_path, chain_id_map, selected_chain_ids
        )
        ensure_normalized_pdb(temp_subset_path)
        temp_subset_path.replace(path)
    finally:
        if temp_subset_path is not None:
            temp_subset_path.unlink(missing_ok=True)
    _atomic_write_csv_rows(
        output_path=map_path,
        header=["original_chain_id", "mapped_chain_id"],
        rows=sorted(chain_id_map.items()),
    )
    stat = path.stat()
    _atomic_write_json(
        metadata_path,
        {
            "cache_revision": uuid.uuid4().hex,
            "entry_id": entry_id.upper(),
            "source_url": str(cif_metadata.get("source_url") or ""),
            "source_cif_sha256": cif_metadata["sha256"],
            "chain_ids": sorted(selected_chain_ids),
            "chain_id_map": chain_id_map,
            "etag": None,
            "last_modified": None,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "validated_at": _utc_now_iso(),
        },
    )
    return path, chain_id_map


def _normalize_cached_pdb(path: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    """Ensure cache hits satisfy the same input policy as fresh downloads."""
    if not path.is_file():
        return metadata
    ensure_normalized_pdb(path)
    stat = path.stat()
    if (
        metadata.get("size_bytes") == stat.st_size
        and metadata.get("mtime_ns") == stat.st_mtime_ns
    ):
        return metadata
    normalized_metadata = dict(metadata)
    normalized_metadata.update(
        sha256=_sha256_file(path),
        size_bytes=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        cache_revision=uuid.uuid4().hex,
    )
    _atomic_write_json(_pdb_cache_metadata_path(path), normalized_metadata)
    return normalized_metadata
