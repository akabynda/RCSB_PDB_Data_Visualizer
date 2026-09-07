"""Biopython parsing, chain selection, and legacy PDB chain-ID mapping."""

from __future__ import annotations

import csv
import hashlib
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from Bio import BiopythonWarning
from Bio.PDB import MMCIFParser, PDBParser, Select
from Bio.PDB.PDBExceptions import PDBConstructionWarning

from src.dataset.cache import (
    _cached_pdb_matches_metadata,
    _load_pdb_cache_metadata,
)
from src.dataset.config import (
    PDB_CHAIN_ID_POOL,
)


class ChainSubsetSelect(Select):
    """Select Biopython chain objects by identity for PDB serialization."""

    def __init__(self, chain_object_ids: set[int]) -> None:
        """Store the selected Biopython chain object identities."""
        self.chain_object_ids = chain_object_ids

    def accept_chain(self, chain: Any) -> bool:
        """Return whether a chain should be written by PDBIO."""
        return id(chain) in self.chain_object_ids


def parse_mmcif_structure(entry_id: str, cif_path: Path) -> Any:
    """Parse an mmCIF coordinate file into a Biopython structure."""
    parser = MMCIFParser(QUIET=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PDBConstructionWarning)
        return parser.get_structure(entry_id, str(cif_path))


def parse_pdb_structure(entry_id: str, pdb_path: str | Path) -> Any:
    """Parse a PDB coordinate file into a Biopython structure."""
    parser = PDBParser(QUIET=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PDBConstructionWarning)
        return parser.get_structure(entry_id, str(pdb_path))


def _coerce_structure_chain_ids_for_pdbio(structure: Any) -> dict[str, str]:
    """Assign temporary PDB-compatible chain IDs before PDBIO output."""
    original_chain_ids: list[str] = []
    seen_original_ids: set[str] = set()
    for model in structure:
        for chain in model:
            chain_id = str(chain.id)
            if chain_id in seen_original_ids:
                continue
            seen_original_ids.add(chain_id)
            original_chain_ids.append(chain_id)

    chain_id_map: dict[str, str] = {}
    used_ids: set[str] = set()
    for original_id in original_chain_ids:
        if len(original_id) == 1 and original_id not in used_ids:
            mapped_id = original_id
        elif original_id and original_id[0] not in used_ids:
            mapped_id = original_id[0]
        else:
            mapped_id = next(
                (
                    candidate
                    for candidate in PDB_CHAIN_ID_POOL
                    if candidate not in used_ids
                ),
                None,
            )
            if mapped_id is None:
                raise RuntimeError("Too many chains to convert mmCIF to PDB")
        chain_id_map[original_id] = mapped_id
        used_ids.add(mapped_id)

    _apply_chain_id_map_without_transient_conflicts(structure, chain_id_map)
    return {
        original_id: mapped_id
        for original_id, mapped_id in chain_id_map.items()
        if original_id != mapped_id
    }


def _coerce_selected_structure_chain_ids_for_pdbio(
    structure: Any,
    selected_chain_ids: set[str],
) -> tuple[dict[str, str], set[int]]:
    """Assign temporary PDB-compatible IDs only for selected chains."""
    selected_chain_object_ids: set[int] = set()
    existing_chain_ids: set[str] = set()
    for model in structure:
        for chain in model:
            chain_id = str(chain.id)
            if chain_id in selected_chain_ids:
                existing_chain_ids.add(chain_id)
                selected_chain_object_ids.add(id(chain))
    chain_id_map: dict[str, str] = {}
    used_ids: set[str] = set()
    for original_id in sorted(existing_chain_ids):
        if len(original_id) == 1 and original_id not in used_ids:
            mapped_id = original_id
        elif original_id and original_id[0] not in used_ids:
            mapped_id = original_id[0]
        else:
            mapped_id = next(
                (
                    candidate
                    for candidate in PDB_CHAIN_ID_POOL
                    if candidate not in used_ids
                ),
                None,
            )
            if mapped_id is None:
                raise RuntimeError("Too many selected chains to convert mmCIF to PDB")
        chain_id_map[original_id] = mapped_id
        used_ids.add(mapped_id)

    _apply_chain_id_map_without_transient_conflicts(structure, chain_id_map)
    return chain_id_map, selected_chain_object_ids


def _apply_chain_id_map_without_transient_conflicts(
    structure: Any,
    chain_id_map: dict[str, str],
) -> None:
    """Apply a chain-ID mapping without creating temporary ID collisions."""
    chain_objects: list[tuple[Any, str]] = []
    for model in structure:
        for chain in model:
            original_id = str(chain.id)
            if original_id in chain_id_map:
                chain_objects.append((chain, original_id))

    temporary_ids: dict[int, str] = {}
    for index, (chain, _) in enumerate(chain_objects):
        temporary_id = f"__tmp_chain_{index}__"
        temporary_ids[id(chain)] = temporary_id
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", BiopythonWarning)
            chain.id = temporary_id

    for chain, original_id in chain_objects:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", BiopythonWarning)
            chain.id = chain_id_map[original_id]


def load_cached_chain_id_map(cache_dir: Path, entry_id: str) -> dict[str, str]:
    """Load the cached original-to-PDB chain ID mapping for an entry."""
    normalized_entry_id = entry_id.upper()
    pdb_path = cache_dir / f"{normalized_entry_id}.pdb"
    metadata = _load_pdb_cache_metadata(pdb_path)
    if metadata is not None and _cached_pdb_matches_metadata(pdb_path, metadata):
        raw_mapping = metadata.get("chain_id_map")
        if isinstance(raw_mapping, dict):
            return {
                str(original): str(mapped)
                for original, mapped in raw_mapping.items()
                if str(original) and str(mapped)
            }
    map_path = cache_dir / f"{normalized_entry_id}.chain_map.csv"
    if not map_path.exists():
        return {}
    chain_id_map: dict[str, str] = {}
    with map_path.open("r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            original = str(row.get("original_chain_id") or "")
            mapped = str(row.get("mapped_chain_id") or "")
            if original and mapped:
                chain_id_map[original] = mapped
    return chain_id_map


def _chain_subset_cache_stem(entry_id: str, chain_ids: Sequence[str]) -> str:
    """Build a deterministic cache key for a chain subset file."""
    normalized = ",".join(sorted(str(chain_id) for chain_id in chain_ids))
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]
    return f"{entry_id}.chains_{digest}"


def load_chain_id_map(map_path: Path) -> dict[str, str]:
    """Read a JSON chain-ID mapping file from disk."""
    if not map_path.exists():
        return {}
    chain_id_map: dict[str, str] = {}
    with map_path.open("r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            original = str(row.get("original_chain_id") or "")
            mapped = str(row.get("mapped_chain_id") or "")
            if original and mapped:
                chain_id_map[original] = mapped
    return chain_id_map
