"""Hash-verified wwPDB validation reports and their entry-level quality metrics."""

from __future__ import annotations

import gzip
import io
import json
import time
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path
from typing import Any

import requests

from src.dataset.cache import (
    _atomic_write_json,
    _pdb_cache_entry_lock,
    _sha256_file,
    _utc_now_iso,
)
from src.dataset.config import LOGGER

VALIDATION_REPORT_URL = (
    "https://files.rcsb.org/pub/pdb/validation_reports/"
    "{middle}/{entry_id}/{entry_id}_validation.xml.gz"
)

# Entry attributes for well-defined (core) residues over all models.
QUALITY_METRIC_ATTRIBUTES = (
    "clashscore",
    "percent-rama-outliers",
    "percent-rota-outliers",
)

# ParseError is a SyntaxError; a corrupt deflate stream raises zlib.error.
REPORT_READ_ERRORS = (OSError, EOFError, ValueError, SyntaxError, zlib.error)


def download_validation_report(
    session: Any,
    cache_dir: Path,
    entry_id: str,
    timeout: float,
    retries: int = 1,
    backoff_seconds: float = 0.0,
) -> Path | None:
    """Return a cached report verified by SHA-256, or None when wwPDB has none.

    A downloaded report is parsed before it is cached; REPORT_READ_ERRORS from
    an unreadable download propagate and nothing is written.
    """
    entry_id = entry_id.upper()
    cache_dir = Path(cache_dir)
    path = cache_dir / f"{entry_id}_validation.xml.gz"
    meta_path = path.with_name(path.name + ".json")
    with _pdb_cache_entry_lock(cache_dir, entry_id):
        if path.is_file() and meta_path.is_file():
            try:
                metadata = json.loads(meta_path.read_text(encoding="utf-8"))
                if metadata["sha256"] == _sha256_file(path):
                    return path
            except (OSError, ValueError, KeyError, TypeError):
                pass
            LOGGER.warning("Refreshing invalid validation report cache for %s", entry_id)
        lower = entry_id.lower()
        url = VALIDATION_REPORT_URL.format(middle=lower[1:3], entry_id=lower)
        content = _get_report(session, url, timeout, retries, backoff_seconds)
        if content is None:
            return None
        _quality_metrics_from_bytes(content)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_bytes(content)
        temporary.replace(path)
        _atomic_write_json(
            meta_path,
            {"url": url, "retrieved_at": _utc_now_iso(), "sha256": _sha256_file(path)},
        )
    return path


def _get_report(
    session: Any, url: str, timeout: float, retries: int, backoff_seconds: float
) -> bytes | None:
    """GET one report with retries; None when wwPDB published no report.

    files.rcsb.org serves the archive from S3, which answers a request for an
    absent file with 403 AccessDenied (404 only for an absent directory).
    """
    last_error: Exception | None = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            response = session.get(url, timeout=timeout)
            if response.status_code == 404 or (
                response.status_code == 403
                and b"<Code>AccessDenied</Code>" in response.content
            ):
                return None
            response.raise_for_status()
            return response.content
        except requests.RequestException as exc:
            last_error = exc
            LOGGER.warning(
                "Validation report request failed (attempt %d/%d): %s",
                attempt,
                retries,
                exc,
            )
            if attempt < retries:
                time.sleep(backoff_seconds * attempt)
    raise RuntimeError(
        f"Validation report request failed after {retries} attempts: {last_error}"
    ) from last_error


def read_quality_metrics(path: Path) -> dict[str, str | None]:
    """Read the core quality attributes of the report's Entry element.

    wwPDB writes a computed zero as "0.00" and omits an attribute that was not
    computed, so None here means "not computed", never zero.
    """
    return _quality_metrics_from_bytes(Path(path).read_bytes())


def _quality_metrics_from_bytes(content: bytes) -> dict[str, str | None]:
    """Decompress a whole report (checking its CRC) and read its Entry element."""
    xml = gzip.decompress(content)
    for _, element in ET.iterparse(io.BytesIO(xml), events=("start",)):
        if element.tag == "Entry":
            return {name: element.get(name) for name in QUALITY_METRIC_ATTRIBUTES}
    raise ValueError("validation report has no Entry element")
