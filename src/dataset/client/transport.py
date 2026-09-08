"""Thread-local HTTP sessions and retry handling for RCSB requests."""

from __future__ import annotations

import time
from pathlib import Path
from threading import local
from typing import Any

import requests

from ..config import LOGGER, DatasetBuildConfig


class ThreadLocalRequestsSession:
    """Provide one requests.Session per worker thread without serializing I/O."""

    def __init__(self, user_agent: str) -> None:
        """Store immutable session defaults and initialize thread-local state."""
        self.user_agent = user_agent
        self._local = local()

    def _session(self) -> requests.Session:
        """Create or reuse the calling thread's HTTP connection pool."""
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            session.headers.update({"User-Agent": self.user_agent})
            self._local.session = session
        return session

    def get(self, url: str, **kwargs: Any) -> requests.Response:
        """Issue GET through the calling thread's session."""
        return self._session().get(url, **kwargs)

    def post(self, url: str, **kwargs: Any) -> requests.Response:
        """Issue POST through the calling thread's session."""
        return self._session().post(url, **kwargs)


class RCSBTransport:
    """Store client configuration and provide the shared HTTP transport."""

    def __init__(
        self,
        config: DatasetBuildConfig,
        solution_nmr_monomer_cache_dir: Path | None = None,
    ) -> None:
        """Initialize the RCSB API client session and configuration."""
        self.config = config
        self.solution_nmr_monomer_cache_dir = (
            Path(solution_nmr_monomer_cache_dir)
            if solution_nmr_monomer_cache_dir is not None
            else Path("data/pdb_cache")
        )
        self.session = ThreadLocalRequestsSession("pdb-extensible-builder/1.0")

    def _post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST JSON to an RCSB endpoint with retry and backoff handling."""
        last_error: Exception | None = None
        for attempt in range(1, self.config.retries + 1):
            try:
                response = self.session.post(
                    url, json=payload, timeout=self.config.timeout_seconds
                )
                response.raise_for_status()
                data = response.json()
                if not isinstance(data, dict):
                    raise ValueError("RCSB response is not a JSON object")
                if data.get("errors"):
                    errors = data["errors"]
                    if isinstance(errors, list):
                        detail = "; ".join(
                            str(error.get("message", error))
                            if isinstance(error, dict)
                            else str(error)
                            for error in errors
                        )
                    else:
                        detail = str(errors)
                    raise ValueError(f"RCSB response contains errors: {detail}")
                if isinstance(payload.get("query"), str) and not isinstance(
                    data.get("data"), dict
                ):
                    raise ValueError("GraphQL response does not contain a data object")
                return data
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                wait_seconds = self.config.backoff_seconds * attempt
                LOGGER.warning(
                    "Request failed (attempt %d/%d): %s",
                    attempt,
                    self.config.retries,
                    exc,
                )
                if attempt < self.config.retries:
                    time.sleep(wait_seconds)
        raise RuntimeError(
            f"Request failed after {self.config.retries} attempts: {last_error}"
        )
