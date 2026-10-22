"""Content-addressed JSON responses for reproducible RCSB metadata collection."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from threading import Lock
from typing import Any

from src.dataset.cache import (
    _atomic_write_json,
    _pdb_cache_entry_lock,
    _sha256_file,
    _utc_now_iso,
)
from src.dataset.config import LOGGER


class CachedJSONRequests:
    """Cache exact JSON requests with a TTL in hours; zero always refreshes.

    Request pointers are replaced atomically. Response bodies are retained by
    SHA-256 so a later refresh does not overwrite a previous run's source data.
    """

    def __init__(
        self,
        post_json: Callable[[str, dict[str, Any]], dict[str, Any]],
        cache_dir: Path,
        validation_hours: float = 24.0,
        schema_version: str = "membrane-v1",
    ) -> None:
        if not math.isfinite(validation_hours) or validation_hours < 0:
            raise ValueError(
                "API cache validation hours must be finite and nonnegative"
            )
        self._post_json = post_json
        self.cache_dir = Path(cache_dir)
        self.validation_hours = validation_hours
        self.schema_version = schema_version
        self._sources: dict[tuple[str, str, str], dict[str, Any]] = {}
        self._source_lock = Lock()

    @property
    def sources(self) -> list[dict[str, Any]]:
        """Return the sources actually read during this run, ordered by request."""
        with self._source_lock:
            return [dict(self._sources[key]) for key in sorted(self._sources)]

    @staticmethod
    def _digest(value: dict[str, Any]) -> str:
        content = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _read_cached(
        self, pointer_path: Path, request: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        if not pointer_path.exists():
            return None
        try:
            source = json.loads(pointer_path.read_text(encoding="utf-8"))
            if not isinstance(source, dict) or source.get("request") != request:
                raise ValueError("request or schema differs")
            if source.get("request_sha256") != self._digest(request):
                raise ValueError("request checksum differs")
            retrieved = datetime.fromisoformat(
                source["retrieved_at"].replace("Z", "+00:00")
            )
            if retrieved.tzinfo is None:
                raise ValueError("retrieval time lacks timezone")
            age = (datetime.now(timezone.utc) - retrieved).total_seconds()
            if not 0 <= age < self.validation_hours * 3600:
                return None
            digest = source["response_sha256"]
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError("invalid response hash")
            body_path = self.cache_dir / "responses" / f"{digest}.json"
            if _sha256_file(body_path) != digest:
                raise ValueError("response checksum differs")
            body = json.loads(body_path.read_text(encoding="utf-8"))
            self._validate_response(body, request["payload"])
            source["response_path"] = str(body_path.resolve())
            source["request_path"] = str(pointer_path.resolve())
            return body, source
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            LOGGER.warning("Refreshing invalid API cache %s: %s", pointer_path, exc)
            return None

    @staticmethod
    def _validate_response(body: Any, payload: dict[str, Any]) -> None:
        if not isinstance(body, dict):
            raise ValueError("RCSB response is not a JSON object")
        if body.get("errors"):
            raise ValueError(f"RCSB response contains errors: {body['errors']}")
        if isinstance(payload.get("query"), str) and not isinstance(
            body.get("data"), dict
        ):
            raise ValueError("GraphQL response does not contain a data object")

    def post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Return a verified fresh response or fetch it; never use stale fallback."""
        request = {
            "schema_version": self.schema_version,
            "url": url,
            "payload": payload,
        }
        request_hash = self._digest(request)
        pointer_path = self.cache_dir / "requests" / f"{request_hash}.json"
        with _pdb_cache_entry_lock(self.cache_dir, request_hash):
            cached = self._read_cached(pointer_path, request)
            if cached is not None:
                body, source = cached
            else:
                body = self._post_json(url, payload)
                self._validate_response(body, payload)
                digest = self._digest(body)
                body_path = self.cache_dir / "responses" / f"{digest}.json"
                if not body_path.exists() or _sha256_file(body_path) != digest:
                    _atomic_write_json(body_path, body)
                source = {
                    "request": request,
                    "request_sha256": request_hash,
                    "request_path": str(pointer_path.resolve()),
                    "response_path": str(body_path.resolve()),
                    "response_sha256": digest,
                    "retrieved_at": _utc_now_iso(),
                }
                _atomic_write_json(pointer_path, source)
            with self._source_lock:
                source_key = (
                    request_hash,
                    source["response_sha256"],
                    source["retrieved_at"],
                )
                self._sources[source_key] = source
            return body
