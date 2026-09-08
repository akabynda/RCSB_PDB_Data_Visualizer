"""Run STRIDE and cache first-model secondary-structure assignments."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from collections import Counter
from collections.abc import Iterable, Set
from pathlib import Path
from typing import TYPE_CHECKING

import requests

from src.dataset.config import (
    LOGGER,
    STRIDE_CORE_STATE_CODES,
    STRIDE_STATE_CODES,
)
from src.dataset.coordinates import (
    extract_model_pdb_texts,
)
from src.dataset.downloads import (
    download_pdb_if_needed,
)
from src.dataset.pdb_normalization import (
    normalize_pdb_altloc_text,
    pdb_atom_field_offset,
)
from src.dataset.records import ResidueId, parse_residue_id
from src.dataset.structures import (
    load_cached_chain_id_map,
)

if TYPE_CHECKING:
    from src.dataset.config import (
        DatasetBuildConfig,
    )


def _parse_stride_state_by_chain(stdout: str) -> dict[str, dict[ResidueId, str]]:
    """Parse STRIDE output into residue state codes grouped by chain."""
    state_by_chain: dict[str, dict[ResidueId, str]] = {}
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
        try:
            auth_seq_id = parse_residue_id(seq_id_raw)
        except ValueError:
            continue
        state_raw = str(parts[5]).strip()
        # STRIDE documents both B and b as isolated bridges. The dataset uses
        # one bridge class; treating b as unknown loses real assignments.
        state = "B" if state_raw == "b" else state_raw
        if state not in STRIDE_STATE_CODES:
            continue
        chain_states = state_by_chain.setdefault(chain_label, {})
        chain_states.setdefault(auth_seq_id, state)
    return state_by_chain


def _select_stride_chain_states(
    state_by_chain: dict[str, dict[ResidueId, str]],
    chain_id: str,
) -> dict[ResidueId, str] | None:
    """Select only the requested author chain; another chain cannot substitute."""
    return state_by_chain.get(chain_id)


def _prepare_stride_residue_numbering(
    model_text: str,
) -> tuple[str, dict[str, dict[ResidueId, ResidueId]]]:
    """Disambiguate digit insertion codes in STRIDE's whitespace author labels.

    STRIDE concatenates the four-column author number and insertion code. For
    example, (56, "1") and (561, "") both become "561". Only such inputs need
    temporary consecutive residue numbers; the source PDB remains unchanged.
    """
    lines = model_text.splitlines(keepends=True)
    needs_numbering = False
    for line in lines:
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        insertion_index = 26 + pdb_atom_field_offset(line)
        if len(line) > insertion_index:
            insertion_code = line[insertion_index].strip()
            if insertion_code and not insertion_code.isalpha():
                needs_numbering = True
                break
    if not needs_numbering:
        return model_text, {}

    number_by_chain: dict[str, dict[ResidueId, int]] = {}
    original_by_chain: dict[str, dict[ResidueId, ResidueId]] = {}
    prepared_lines: list[str] = []
    for line in lines:
        if not line.startswith(("ATOM  ", "HETATM", "TER   ")) or len(line) < 27:
            prepared_lines.append(line)
            continue
        offset = pdb_atom_field_offset(line)
        if len(line) < 27 + offset:
            prepared_lines.append(line)
            continue
        try:
            original_id = ResidueId(
                int(line[22 + offset : 26 + offset]), line[26 + offset].strip()
            )
        except ValueError:
            prepared_lines.append(line)
            continue
        chain_id = line[21 + offset].strip()
        chain_numbers = number_by_chain.setdefault(chain_id, {})
        number = chain_numbers.setdefault(original_id, len(chain_numbers) + 1)
        if number > 9999:
            raise ValueError("Too many residues for temporary STRIDE PDB numbering")
        original_by_chain.setdefault(chain_id, {})[ResidueId(number)] = original_id
        prepared_lines.append(f"{line[: 22 + offset]}{number:4d} {line[27 + offset :]}")
    return "".join(prepared_lines), original_by_chain


def _run_stride_for_model_text(
    model_text: str,
    stride_executable: str,
) -> dict[str, dict[ResidueId, str]] | None:
    """Run STRIDE on a single MODEL text block and return parsed states."""
    stride_text, original_by_chain = _prepare_stride_residue_numbering(
        normalize_pdb_altloc_text(model_text)
    )
    with tempfile.NamedTemporaryFile(
        "w", suffix=".pdb", encoding="utf-8", delete=True
    ) as handle:
        handle.write(stride_text)
        handle.flush()

        process = subprocess.run(
            [stride_executable, handle.name],
            check=False,
            capture_output=True,
            text=True,
        )
        diagnostic = " ".join((getattr(process, "stderr", "") or "").split())
        if process.returncode != 0:
            diagnostic = diagnostic or " ".join(process.stdout.split())
            diagnostic = diagnostic.replace(handle.name, "<input>")
            raise RuntimeError(
                f"STRIDE exited with status {process.returncode}: "
                f"{diagnostic[:1000] or 'no diagnostic output'}"
            )
        state_by_chain = _parse_stride_state_by_chain(process.stdout)
        if not state_by_chain:
            raise RuntimeError(
                "STRIDE returned no usable ASG assignments"
                + (f": {diagnostic[:1000]}" if diagnostic else "")
            )
        input_chains = {
            line[21 + pdb_atom_field_offset(line)].strip()
            for line in stride_text.splitlines()
            if line.startswith(("ATOM  ", "HETATM")) and len(line) >= 27
        }
        if "" in input_chains and "-" in input_chains:
            raise RuntimeError("STRIDE cannot distinguish blank and '-' chain IDs")
        if "" in input_chains and "-" in state_by_chain:
            state_by_chain[""] = state_by_chain.pop("-")
        if not original_by_chain:
            return state_by_chain
        restored: dict[str, dict[ResidueId, str]] = {}
        for reported_chain, chain_states in state_by_chain.items():
            chain_id = reported_chain
            if (
                chain_id == "-"
                and "" in original_by_chain
                and "-" not in original_by_chain
            ):
                chain_id = ""
            originals = original_by_chain.get(chain_id, {})
            if any(residue_id not in originals for residue_id in chain_states):
                LOGGER.warning(
                    "STRIDE returned an unknown temporary residue identifier"
                )
                return None
            restored[chain_id] = {
                originals[residue_id]: state
                for residue_id, state in chain_states.items()
            }
        return restored


def _stride_state_cache_path(stride_cache_dir: Path, entry_id: str) -> Path:
    """Return the first-model STRIDE state cache path for one structure."""
    return stride_cache_dir / f"{entry_id.upper()}.json"


def _load_cached_stride_state_by_chain(
    cache_path: Path,
    first_model_sha1: str,
) -> dict[str, dict[ResidueId, str]] | None:
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
    if not raw_state_by_chain or not any(raw_state_by_chain.values()):
        return None

    state_by_chain: dict[str, dict[ResidueId, str]] = {}
    for chain_id_raw, raw_chain_states in raw_state_by_chain.items():
        if not isinstance(raw_chain_states, dict):
            return None
        chain_states: dict[ResidueId, str] = {}
        for auth_seq_id_raw, state_raw in raw_chain_states.items():
            try:
                auth_seq_id = parse_residue_id(auth_seq_id_raw)
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
    state_by_chain: dict[str, dict[ResidueId, str]],
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
) -> tuple[dict[str, dict[ResidueId, str]] | None, int]:
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
    chain_states: dict[ResidueId, str],
    modeled_auth_seq_ids: Iterable[ResidueId],
) -> tuple[ResidueId, ResidueId] | None:
    """Find the outer modeled residue range covered by STRIDE core states."""
    modeled_order = (
        sorted(modeled_auth_seq_ids)
        if isinstance(modeled_auth_seq_ids, Set)
        else list(modeled_auth_seq_ids)
    )
    structured_auth_seq_ids = [
        auth_seq_id
        for auth_seq_id in modeled_order
        if chain_states.get(auth_seq_id) in STRIDE_CORE_STATE_CODES
    ]
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
    modeled_auth_seq_ids: Iterable[ResidueId],
    stride_executable: str,
    *,
    raise_on_failure: bool = False,
) -> tuple[dict[str, float], int, int]:
    """Compute coverage; builders request exceptions to retain failure reasons."""
    default_coverages = {state: -1.0 for state in STRIDE_STATE_CODES}
    modeled_auth_seq_ids = set(modeled_auth_seq_ids)
    if modeled_sequence_length <= 0 or not modeled_auth_seq_ids:
        return default_coverages, 0, 0

    model_count = 0
    try:
        pdb_path = download_pdb_if_needed(
            session=session,
            config=config,
            cache_dir=cache_dir,
            entry_id=entry_id,
        )
        chain_map = load_cached_chain_id_map(cache_dir, entry_id)
        parsed_chain_id = chain_map.get(chain_id, chain_id)
        state_by_chain, model_count = load_first_model_stride_state_by_chain(
            pdb_path=pdb_path,
            entry_id=entry_id,
            stride_executable=stride_executable,
            stride_cache_dir=stride_cache_dir,
        )
        if state_by_chain is None:
            raise RuntimeError("STRIDE returned no usable assignments")
        chain_states = _select_stride_chain_states(state_by_chain, parsed_chain_id)
        if not chain_states:
            reported_chains = ", ".join(repr(key) for key in state_by_chain) or "none"
            raise RuntimeError(
                f"STRIDE did not report requested chain {parsed_chain_id!r} "
                f"(reported chains: {reported_chains})"
            )
        missing_ids = modeled_auth_seq_ids.difference(chain_states)
        if missing_ids or modeled_sequence_length != len(modeled_auth_seq_ids):
            missing_text = ", ".join(str(key) for key in sorted(missing_ids)[:20])
            if len(missing_ids) > 20:
                missing_text += f", ... ({len(missing_ids)} total)"
            raise RuntimeError(
                f"incomplete STRIDE assignment "
                f"({len(modeled_auth_seq_ids) - len(missing_ids)}/"
                f"{len(modeled_auth_seq_ids)} modeled residues; "
                f"reported modeled length {modeled_sequence_length}); "
                f"missing residue IDs: {missing_text or 'none'}"
            )
        state_counts = Counter(chain_states[key] for key in modeled_auth_seq_ids)
        denominator = float(modeled_sequence_length)
        coverages = {
            state: min(1.0, max(0.0, state_counts.get(state, 0) / denominator))
            for state in STRIDE_STATE_CODES
        }
        return coverages, model_count, 1
    except Exception as exc:
        if raise_on_failure:
            raise
        LOGGER.warning("%s chain %s: %s", entry_id, chain_id, exc)
        return default_coverages, model_count, 0


def compute_stride_core_range_for_modeled_auth_seq_ids_in_first_model(
    pdb_path: Path,
    entry_id: str,
    chain_id: str,
    modeled_auth_seq_ids: Iterable[ResidueId],
    stride_executable: str,
    stride_cache_dir: Path,
) -> tuple[ResidueId, ResidueId] | None:
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
