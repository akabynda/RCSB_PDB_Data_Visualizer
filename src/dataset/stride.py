"""Run STRIDE and cache first-model secondary-structure assignments."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

import requests

from src.dataset.config import (
    STRIDE_CORE_STATE_CODES,
    STRIDE_STATE_CODES,
)
from src.dataset.coordinates import (
    extract_model_pdb_texts,
)
from src.dataset.downloads import (
    download_pdb_if_needed,
)
from src.dataset.structures import (
    load_cached_chain_id_map,
)

if TYPE_CHECKING:
    from src.dataset.config import (
        DatasetBuildConfig,
    )


def _parse_stride_state_by_chain(stdout: str) -> dict[str, dict[int, str]]:
    """Parse STRIDE output into residue state codes grouped by chain."""
    state_by_chain: dict[str, dict[int, str]] = {}
    for line in stdout.splitlines():
        if not line.startswith("ASG"):
            continue
        parts = line.split()
        if len(parts) < 6:
            continue
        chain_label = str(parts[2]).strip()
        seq_id_raw = str(parts[3]).strip()
        if not seq_id_raw:
            continue
        if seq_id_raw[-1:].isalpha():
            seq_id_raw = seq_id_raw[:-1]
        try:
            auth_seq_id = int(seq_id_raw)
        except ValueError:
            continue
        state_raw = str(parts[5]).strip()
        if len(state_raw) != 1:
            continue
        state = state_raw if state_raw in STRIDE_STATE_CODES else "C"
        chain_states = state_by_chain.setdefault(chain_label, {})
        chain_states.setdefault(auth_seq_id, state)
    return state_by_chain


def _select_stride_chain_states(
    state_by_chain: dict[str, dict[int, str]],
    chain_id: str,
) -> dict[int, str] | None:
    """Select STRIDE residue states for the best matching chain identifier."""
    chain_states = state_by_chain.get(chain_id)
    if not chain_states and len(state_by_chain) == 1:
        chain_states = next(iter(state_by_chain.values()))
    return chain_states


def _run_stride_for_model_text(
    model_text: str,
    stride_executable: str,
) -> dict[str, dict[int, str]] | None:
    """Run STRIDE on a single MODEL text block and return parsed states."""
    with tempfile.NamedTemporaryFile(
        "w", suffix=".pdb", encoding="utf-8", delete=True
    ) as handle:
        handle.write(model_text)
        handle.flush()

        process = subprocess.run(
            [stride_executable, handle.name],
            check=False,
            capture_output=True,
            text=True,
        )
        if process.returncode != 0:
            return None
        return _parse_stride_state_by_chain(process.stdout)


def _stride_state_cache_path(stride_cache_dir: Path, entry_id: str) -> Path:
    """Return the first-model STRIDE state cache path for one structure."""
    return stride_cache_dir / f"{entry_id.upper()}.json"


def _load_cached_stride_state_by_chain(
    cache_path: Path,
    first_model_sha1: str,
) -> dict[str, dict[int, str]] | None:
    """Load cached first-model STRIDE states if they match the model text."""
    if not cache_path.exists():
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("first_model_sha1") != first_model_sha1:
        return None
    raw_state_by_chain = payload.get("state_by_chain")
    if not isinstance(raw_state_by_chain, dict):
        return None

    state_by_chain: dict[str, dict[int, str]] = {}
    for chain_id_raw, raw_chain_states in raw_state_by_chain.items():
        if not isinstance(raw_chain_states, dict):
            return None
        chain_states: dict[int, str] = {}
        for auth_seq_id_raw, state_raw in raw_chain_states.items():
            try:
                auth_seq_id = int(auth_seq_id_raw)
            except (TypeError, ValueError):
                return None
            state = str(state_raw)
            if state not in STRIDE_STATE_CODES:
                return None
            chain_states[auth_seq_id] = state
        state_by_chain[str(chain_id_raw)] = chain_states
    return state_by_chain


def _write_cached_stride_state_by_chain(
    cache_path: Path,
    entry_id: str,
    first_model_sha1: str,
    state_by_chain: dict[str, dict[int, str]],
) -> None:
    """Persist first-model STRIDE states using an atomic replace."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "entry_id": entry_id.upper(),
        "first_model_sha1": first_model_sha1,
        "state_by_chain": {
            chain_id: {
                str(auth_seq_id): state
                for auth_seq_id, state in sorted(chain_states.items())
            }
            for chain_id, chain_states in sorted(state_by_chain.items())
        },
    }
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            suffix=".json",
            prefix=f"{entry_id.upper()}.",
            dir=str(cache_path.parent),
            encoding="utf-8",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
        temp_path.replace(cache_path)
    except OSError:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def load_first_model_stride_state_by_chain(
    pdb_path: Path,
    entry_id: str,
    stride_executable: str,
    stride_cache_dir: Path,
) -> tuple[dict[str, dict[int, str]] | None, int]:
    """Load or compute cached STRIDE states for the first coordinate model."""
    model_texts = extract_model_pdb_texts(pdb_path)
    if not model_texts:
        return None, 0

    first_model_text = model_texts[0]
    first_model_sha1 = hashlib.sha1(first_model_text.encode("utf-8")).hexdigest()
    cache_path = _stride_state_cache_path(stride_cache_dir, entry_id)
    cached_states = _load_cached_stride_state_by_chain(
        cache_path=cache_path,
        first_model_sha1=first_model_sha1,
    )
    if cached_states is not None:
        return cached_states, len(model_texts)

    state_by_chain = _run_stride_for_model_text(
        model_text=first_model_text,
        stride_executable=stride_executable,
    )
    if state_by_chain is not None:
        _write_cached_stride_state_by_chain(
            cache_path=cache_path,
            entry_id=entry_id,
            first_model_sha1=first_model_sha1,
            state_by_chain=state_by_chain,
        )
    return state_by_chain, len(model_texts)


def _extract_stride_core_range_for_modeled_auth_seq_ids(
    chain_states: dict[int, str],
    modeled_auth_seq_ids: set[int],
) -> tuple[int, int] | None:
    """Find the outer modeled residue range covered by STRIDE core states."""
    structured_auth_seq_ids = sorted(
        auth_seq_id
        for auth_seq_id in modeled_auth_seq_ids
        if chain_states.get(auth_seq_id) in STRIDE_CORE_STATE_CODES
    )
    if not structured_auth_seq_ids:
        return None
    return structured_auth_seq_ids[0], structured_auth_seq_ids[-1]


def compute_stride_state_coverages_for_chain_modeled_first_model(
    session: requests.Session,
    config: DatasetBuildConfig,
    cache_dir: Path,
    stride_cache_dir: Path,
    entry_id: str,
    chain_id: str,
    modeled_sequence_length: int,
    modeled_auth_seq_ids: set[int],
    stride_executable: str,
) -> tuple[dict[str, float], int, int]:
    """Compute first-model STRIDE secondary-structure coverage for one chain."""
    default_coverages = {state: -1.0 for state in STRIDE_STATE_CODES}
    if modeled_sequence_length <= 0 or not modeled_auth_seq_ids:
        return default_coverages, 0, 0

    try:
        pdb_path = download_pdb_if_needed(
            session=session,
            config=config,
            cache_dir=cache_dir,
            entry_id=entry_id,
        )
    except Exception:
        return default_coverages, 0, 0

    chain_map = load_cached_chain_id_map(cache_dir, entry_id)
    parsed_chain_id = chain_map.get(chain_id, chain_id)
    model_count = 0
    try:
        state_by_chain, model_count = load_first_model_stride_state_by_chain(
            pdb_path=pdb_path,
            entry_id=entry_id,
            stride_executable=stride_executable,
            stride_cache_dir=stride_cache_dir,
        )
        if state_by_chain is None:
            return default_coverages, model_count, 0

        chain_states = _select_stride_chain_states(state_by_chain, parsed_chain_id)
        if not chain_states:
            return default_coverages, model_count, 0

        filtered_states = [
            chain_states.get(auth_seq_id, "C")
            for auth_seq_id in sorted(modeled_auth_seq_ids)
        ]
        missing_count = max(0, modeled_sequence_length - len(filtered_states))
        if missing_count > 0:
            filtered_states.extend(["C"] * missing_count)
        if not filtered_states:
            return default_coverages, model_count, 0

        state_counts = Counter(filtered_states)
        denominator = float(modeled_sequence_length)
        coverages = {
            state: min(1.0, max(0.0, state_counts.get(state, 0) / denominator))
            for state in STRIDE_STATE_CODES
        }
        return coverages, model_count, 1
    except Exception:
        return default_coverages, model_count, 0


def compute_stride_core_range_for_modeled_auth_seq_ids_in_first_model(
    pdb_path: Path,
    entry_id: str,
    chain_id: str,
    modeled_auth_seq_ids: set[int],
    stride_executable: str,
    stride_cache_dir: Path,
) -> tuple[int, int] | None:
    """Return the modeled first-model residue range supported by STRIDE core states."""
    if not modeled_auth_seq_ids:
        return None

    state_by_chain, _ = load_first_model_stride_state_by_chain(
        pdb_path=pdb_path,
        entry_id=entry_id,
        stride_executable=stride_executable,
        stride_cache_dir=stride_cache_dir,
    )
    if state_by_chain is None:
        return None

    chain_states = _select_stride_chain_states(state_by_chain, chain_id)
    if not chain_states:
        return None

    return _extract_stride_core_range_for_modeled_auth_seq_ids(
        chain_states=chain_states,
        modeled_auth_seq_ids=modeled_auth_seq_ids,
    )
