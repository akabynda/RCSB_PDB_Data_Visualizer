"""Persist validated first-model alpha-carbon data in pickle-free NPZ files."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from typing import Any

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback for library users.
    fcntl = None  # type: ignore[assignment]
import os
import tempfile
from typing import TYPE_CHECKING

import numpy as np
from Bio.SeqUtils import seq1

from src.dataset.cache import (
    _cached_pdb_matches_metadata,
    _load_pdb_cache_metadata,
    _sha256_file,
)
from src.dataset.config import (
    XRAY_CA_CACHE_SCHEMA_VERSION,
    XRAY_CA_PARSER_REVISION,
)
from src.dataset.coordinates import (
    _is_better_ca_candidate,
    _parse_first_model_ca_line_fields,
    _parse_pdb_modres_identity_map,
)
from src.dataset.records import (
    CAResidueRecord,
)

if TYPE_CHECKING:
    from src.dataset.records import (
        PreparedXrayCAData,
    )


_FIRST_MODEL_CA_CACHE_LOCKS_GUARD = Lock()


_FIRST_MODEL_CA_CACHE_LOCKS: dict[str, tuple[Any, int]] = {}


def _first_model_ca_cache_path(pdb_path: Path) -> Path:
    """Return the versioned durable first-model CA cache path."""
    return pdb_path.with_name(
        f"{pdb_path.name}.first_model_ca.v{XRAY_CA_CACHE_SCHEMA_VERSION}.npz"
    )


@contextmanager
def _first_model_ca_cache_lock(cache_path: Path) -> Iterator[None]:
    """Serialize a parsed-CA cache transaction across threads and processes."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    key = str(cache_path.resolve())
    with _FIRST_MODEL_CA_CACHE_LOCKS_GUARD:
        state = _FIRST_MODEL_CA_CACHE_LOCKS.get(key)
        if state is None:
            thread_lock = Lock()
            waiter_count = 0
        else:
            thread_lock, waiter_count = state
        _FIRST_MODEL_CA_CACHE_LOCKS[key] = (thread_lock, waiter_count + 1)

    thread_lock.acquire()
    lock_handle: Any | None = None
    try:
        lock_dir = cache_path.parent / ".locks"
        lock_dir.mkdir(parents=True, exist_ok=True)
        lock_handle = (lock_dir / f"{cache_path.name}.lock").open("a+b")
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
            thread_lock.release()
            with _FIRST_MODEL_CA_CACHE_LOCKS_GUARD:
                current_lock, current_count = _FIRST_MODEL_CA_CACHE_LOCKS[key]
                if current_count <= 1:
                    del _FIRST_MODEL_CA_CACHE_LOCKS[key]
                else:
                    _FIRST_MODEL_CA_CACHE_LOCKS[key] = (
                        current_lock,
                        current_count - 1,
                    )


def _coordinate_source_sha256(pdb_path: Path) -> str:
    """Return a trusted coordinate SHA, hashing local fixtures when necessary."""
    metadata = _load_pdb_cache_metadata(pdb_path)
    if metadata is not None and _cached_pdb_matches_metadata(pdb_path, metadata):
        value = metadata.get("sha256")
        if isinstance(value, str) and len(value) == 64:
            return value
    return _sha256_file(pdb_path)


def _parse_first_model_ca_data_by_chain(
    pdb_path: Path,
) -> dict[str, PreparedXrayCAData]:
    """Parse residue identities, HETATM evidence, and coordinates in one pass."""
    residue_order_by_chain: dict[str, list[int]] = {}
    residue_candidates_by_chain: dict[
        str, dict[int, tuple[str, float, str, CAResidueRecord]]
    ] = {}
    coordinate_candidates_by_chain: dict[
        str, dict[int, tuple[str, float, str, bool, np.ndarray]]
    ] = {}
    hetatm_ca_resids_by_chain: dict[str, set[int]] = {}
    modres_identity_by_key = _parse_pdb_modres_identity_map(pdb_path)
    has_model_records = False
    in_model = False

    with pdb_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            record_type = line[:6]
            if record_type.startswith("MODEL"):
                if has_model_records:
                    break
                has_model_records = True
                in_model = True
                continue
            if record_type.startswith("ENDMDL"):
                if in_model:
                    break
                continue
            if has_model_records and not in_model:
                continue
            is_standard_atom = record_type.startswith("ATOM")
            is_hetero_atom = record_type.startswith("HETATM")
            if not is_standard_atom and not is_hetero_atom:
                continue
            parsed_fields = _parse_first_model_ca_line_fields(line)
            if parsed_fields is None:
                continue
            (
                chain_id,
                resid,
                insertion_code,
                alt_loc,
                occupancy,
                resname,
            ) = parsed_fields
            if occupancy <= 0.0:
                continue

            if is_hetero_atom:
                hetatm_ca_resids_by_chain.setdefault(chain_id, set()).add(resid)
            if is_standard_atom:
                identity = seq1(resname, custom_map={"MSE": "M"}, undef_code="X")
            else:
                identity = modres_identity_by_key.get(
                    (chain_id, resid, insertion_code, resname),
                    modres_identity_by_key.get(
                        (chain_id, resid, "", resname),
                        f"HET:{resname}",
                    ),
                )
            ca_record = CAResidueRecord(
                resid=resid,
                identity=identity,
                is_standard_atom=is_standard_atom,
                has_hetatm_ca=is_hetero_atom,
            )
            residue_candidates = residue_candidates_by_chain.setdefault(chain_id, {})
            existing_residue = residue_candidates.get(resid)
            if existing_residue is None:
                residue_order_by_chain.setdefault(chain_id, []).append(resid)
                residue_candidates[resid] = (
                    insertion_code,
                    occupancy,
                    alt_loc,
                    ca_record,
                )
            else:
                (
                    existing_insertion_code,
                    existing_occupancy,
                    existing_alt_loc,
                    existing_record,
                ) = existing_residue
                should_replace = False
                if ca_record.is_standard_atom != existing_record.is_standard_atom:
                    should_replace = ca_record.is_standard_atom
                else:
                    should_replace = _is_better_ca_candidate(
                        new_insertion_code=insertion_code,
                        new_occupancy=occupancy,
                        new_alt_loc=alt_loc,
                        current_insertion_code=existing_insertion_code,
                        current_occupancy=existing_occupancy,
                        current_alt_loc=existing_alt_loc,
                    )
                if should_replace:
                    residue_candidates[resid] = (
                        insertion_code,
                        occupancy,
                        alt_loc,
                        ca_record,
                    )

            try:
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
            except ValueError:
                parts = line.split()
                if len(parts) < 9:
                    continue
                try:
                    x, y, z = (float(value) for value in parts[6:9])
                except ValueError:
                    continue
            coords = np.asarray([x, y, z], dtype=float)
            coordinate_candidates = coordinate_candidates_by_chain.setdefault(
                chain_id, {}
            )
            existing_coordinate = coordinate_candidates.get(resid)
            if existing_coordinate is None:
                coordinate_candidates[resid] = (
                    insertion_code,
                    occupancy,
                    alt_loc,
                    is_standard_atom,
                    coords,
                )
                continue
            (
                existing_insertion_code,
                existing_occupancy,
                existing_alt_loc,
                existing_is_standard_atom,
                _,
            ) = existing_coordinate
            should_replace_coordinate = False
            if is_standard_atom != existing_is_standard_atom:
                should_replace_coordinate = is_standard_atom
            else:
                should_replace_coordinate = _is_better_ca_candidate(
                    new_insertion_code=insertion_code,
                    new_occupancy=occupancy,
                    new_alt_loc=alt_loc,
                    current_insertion_code=existing_insertion_code,
                    current_occupancy=existing_occupancy,
                    current_alt_loc=existing_alt_loc,
                )
            if should_replace_coordinate:
                coordinate_candidates[resid] = (
                    insertion_code,
                    occupancy,
                    alt_loc,
                    is_standard_atom,
                    coords,
                )

    parsed_by_chain: dict[str, PreparedXrayCAData] = {}
    for chain_id, residue_order in residue_order_by_chain.items():
        residue_candidates = residue_candidates_by_chain[chain_id]
        hetatm_resids = hetatm_ca_resids_by_chain.get(chain_id, set())
        records = tuple(
            CAResidueRecord(
                resid=residue_candidates[resid][3].resid,
                identity=residue_candidates[resid][3].identity,
                is_standard_atom=residue_candidates[resid][3].is_standard_atom,
                has_hetatm_ca=resid in hetatm_resids,
            )
            for resid in residue_order
        )
        coords = {
            resid: candidate[4]
            for resid, candidate in coordinate_candidates_by_chain.get(
                chain_id, {}
            ).items()
        }
        parsed_by_chain[chain_id] = records, coords
    return parsed_by_chain


def _write_first_model_ca_cache(
    cache_path: Path,
    source_sha256: str,
    parsed_by_chain: dict[str, PreparedXrayCAData],
) -> None:
    """Atomically write a pickle-free, flattened NPZ CA cache."""
    chain_ids = list(parsed_by_chain)
    offsets = [0]
    resids: list[int] = []
    identities: list[str] = []
    flags: list[int] = []
    coordinates: list[np.ndarray] = []
    coordinate_present: list[bool] = []
    for chain_id in chain_ids:
        records, coords_by_resid = parsed_by_chain[chain_id]
        for record in records:
            resids.append(record.resid)
            identities.append(record.identity)
            flags.append(
                int(record.is_standard_atom) | (int(record.has_hetatm_ca) << 1)
            )
            coords = coords_by_resid.get(record.resid)
            coordinate_present.append(coords is not None)
            coordinates.append(
                np.asarray(coords, dtype=np.float64)
                if coords is not None
                else np.full(3, np.nan, dtype=np.float64)
            )
        offsets.append(len(resids))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb",
            suffix=".npz.tmp",
            prefix=f".{cache_path.name}.",
            dir=str(cache_path.parent),
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            np.savez_compressed(
                handle,
                schema_version=np.asarray(XRAY_CA_CACHE_SCHEMA_VERSION, dtype=np.int64),
                parser_revision=np.asarray(XRAY_CA_PARSER_REVISION, dtype=np.int64),
                source_sha256=np.asarray(source_sha256),
                chain_ids=np.asarray(chain_ids, dtype=np.str_),
                chain_offsets=np.asarray(offsets, dtype=np.int64),
                resids=np.asarray(resids, dtype=np.int64),
                identities=np.asarray(identities, dtype=np.str_),
                flags=np.asarray(flags, dtype=np.uint8),
                coords=(
                    np.asarray(coordinates, dtype=np.float64).reshape((-1, 3))
                    if coordinates
                    else np.empty((0, 3), dtype=np.float64)
                ),
                coord_present=np.asarray(coordinate_present, dtype=np.bool_),
            )
            handle.flush()
            os.fsync(handle.fileno())
        temp_path.replace(cache_path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _read_first_model_ca_cache(
    cache_path: Path,
    source_sha256: str,
) -> dict[str, PreparedXrayCAData] | None:
    """Read and fully validate a first-model CA cache payload."""
    if not cache_path.is_file():
        return None
    try:
        with np.load(cache_path, allow_pickle=False) as payload:
            if int(payload["schema_version"].item()) != XRAY_CA_CACHE_SCHEMA_VERSION:
                return None
            if int(payload["parser_revision"].item()) != XRAY_CA_PARSER_REVISION:
                return None
            if str(payload["source_sha256"].item()) != source_sha256:
                return None
            chain_ids = np.asarray(payload["chain_ids"]).astype(str).tolist()
            offsets = np.asarray(payload["chain_offsets"], dtype=np.int64)
            resids = np.asarray(payload["resids"], dtype=np.int64)
            identities = np.asarray(payload["identities"]).astype(str)
            flags = np.asarray(payload["flags"], dtype=np.uint8)
            coordinates = np.asarray(payload["coords"], dtype=np.float64)
            coordinate_present = np.asarray(payload["coord_present"], dtype=np.bool_)
    except (OSError, ValueError, KeyError, EOFError):
        return None

    item_count = len(resids)
    if len(set(chain_ids)) != len(chain_ids):
        return None
    if offsets.shape != (len(chain_ids) + 1,):
        return None
    if len(offsets) == 0 or offsets[0] != 0 or offsets[-1] != item_count:
        return None
    if np.any(offsets[1:] < offsets[:-1]):
        return None
    if (
        identities.shape != (item_count,)
        or flags.shape != (item_count,)
        or coordinates.shape != (item_count, 3)
        or coordinate_present.shape != (item_count,)
        or np.any(flags & np.uint8(~0b11 & 0xFF))
    ):
        return None

    parsed_by_chain: dict[str, PreparedXrayCAData] = {}
    for chain_index, chain_id in enumerate(chain_ids):
        start = int(offsets[chain_index])
        end = int(offsets[chain_index + 1])
        chain_records = tuple(
            CAResidueRecord(
                resid=int(resids[index]),
                identity=str(identities[index]),
                is_standard_atom=bool(flags[index] & 0b01),
                has_hetatm_ca=bool(flags[index] & 0b10),
            )
            for index in range(start, end)
        )
        if len({record.resid for record in chain_records}) != len(chain_records):
            return None
        chain_coords = {
            int(resids[index]): np.asarray(coordinates[index], dtype=float).copy()
            for index in range(start, end)
            if coordinate_present[index]
        }
        if any(not np.all(np.isfinite(coords)) for coords in chain_coords.values()):
            return None
        parsed_by_chain[chain_id] = chain_records, chain_coords
    return parsed_by_chain


def load_cached_first_model_ca_data(
    pdb_path: Path,
    chain_id: str,
) -> PreparedXrayCAData:
    """Load versioned first-model X-ray CA data, rebuilding safe cache misses."""
    pdb_path = Path(pdb_path)
    cache_path = _first_model_ca_cache_path(pdb_path)
    with _first_model_ca_cache_lock(cache_path):
        source_sha256 = _coordinate_source_sha256(pdb_path)
        parsed_by_chain = _read_first_model_ca_cache(cache_path, source_sha256)
        if parsed_by_chain is None:
            parsed_by_chain = _parse_first_model_ca_data_by_chain(pdb_path)
            final_source_sha256 = _coordinate_source_sha256(pdb_path)
            if final_source_sha256 != source_sha256:
                source_sha256 = final_source_sha256
                parsed_by_chain = _parse_first_model_ca_data_by_chain(pdb_path)
            _write_first_model_ca_cache(
                cache_path=cache_path,
                source_sha256=source_sha256,
                parsed_by_chain=parsed_by_chain,
            )
        records, coords = parsed_by_chain.get(chain_id, (tuple(), {}))
        return records, {resid: value.copy() for resid, value in coords.items()}
