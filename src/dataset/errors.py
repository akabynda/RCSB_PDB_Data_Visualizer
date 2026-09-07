"""Domain errors raised during homology searches and remote requests."""

from __future__ import annotations

import requests


class NMRHomologyQueryIneligibleError(Exception):
    """Signal that no homology search was performed for an NMR entry."""

    def __init__(self, entry_id: str, reason: str) -> None:
        """Store the ineligible entry identifier and explanatory reason."""
        self.entry_id = entry_id
        self.reason = reason
        super().__init__(f"{entry_id}: {reason}")


class NMRCoreContainsHetatmError(NMRHomologyQueryIneligibleError):
    """Signal that an NMR core is ineligible because it contains HETATM CA."""

    def __init__(self, entry_id: str) -> None:
        """Create an error for an entry whose core contains HETATM CA atoms."""
        super().__init__(entry_id, "STRIDE core contains HETATM CA residues")


class XrayHomologEvaluationError(Exception):
    """Signal that X-ray candidates could not be evaluated conclusively."""


def _is_http_server_error(error: BaseException) -> bool:
    """Return whether an exception chain contains an HTTP 5xx response."""
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, requests.HTTPError):
            response = current.response
            status_code = getattr(response, "status_code", None)
            if isinstance(status_code, int) and 500 <= status_code < 600:
                return True
        current = current.__cause__ or current.__context__
    return False
