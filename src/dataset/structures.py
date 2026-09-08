"""Biopython parsing, chain selection, and legacy PDB chain-ID mapping."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from Bio import BiopythonWarning
from Bio.PDB import MMCIFParser, PDBParser, Select
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB.PDBExceptions import PDBConstructionWarning

from src.dataset.cache import (
    _cached_pdb_matches_metadata,
    _load_pdb_cache_metadata,
)
from src.dataset.config import (
    PDB_CHAIN_ID_POOL,
    PROTEIN_MONOMER_ENTITY_TYPES,
)
from src.dataset.pdb_normalization import (
    POLYPEPTIDE_REMARK_PREFIX,
    iter_normalized_pdb_lines,
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
    """Parse the shared residue conformer selection into a Biopython structure."""
    parser = PDBParser(QUIET=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PDBConstructionWarning)
        return parser.get_structure(
            entry_id, io.StringIO("".join(iter_normalized_pdb_lines(pdb_path)))
        )


def _prepend_mmcif_polymer_remarks(
    pdb_path: Path,
    cif_path: Path,
    chain_id_map: dict[str, str],
    selected_chain_ids: set[str] | None = None,
) -> None:
    """Preserve exact polypeptide membership in a temporary converted PDB.

    PDBIO drops polymer metadata, and a component name alone cannot distinguish
    a modified residue from a free ligand. The versioned remarks retain author
    IDs from the original atom_site rows, including D/rare amino acids. The
    caller atomically publishes this temporary file after all conversion steps.
    """
    metadata = MMCIF2Dict(str(cif_path))
    entity_ids = metadata.get("_entity_poly.entity_id", [])
    entity_types = metadata.get("_entity_poly.type", [])
    if not entity_ids and not entity_types:
        # Minimal standalone coordinate mmCIF can lack entity metadata entirely.
        # In that case the PDB parser retains only recognized amino acids.
        return
    if len(entity_ids) != len(entity_types):
        raise ValueError("Incomplete mmCIF polymer entity metadata")
    peptide_entities = {
        entity_id
        for entity_id, entity_type in zip(entity_ids, entity_types, strict=True)
        if entity_type in PROTEIN_MONOMER_ENTITY_TYPES
    }
    atom_entities = metadata.get("_atom_site.label_entity_id", [])
    atom_chains = metadata.get("_atom_site.auth_asym_id", [])
    atom_numbers = metadata.get("_atom_site.auth_seq_id", [])
    atom_components = metadata.get("_atom_site.label_comp_id", [])
    atom_insertions = metadata.get(
        "_atom_site.pdbx_PDB_ins_code", ["?"] * len(atom_entities)
    )
    if (
        not atom_entities
        or len(
            {
                len(values)
                for values in (
                    atom_entities,
                    atom_chains,
                    atom_numbers,
                    atom_components,
                    atom_insertions,
                )
            }
        )
        != 1
    ):
        raise ValueError("Incomplete mmCIF atom-site polymer membership metadata")

    residues: dict[tuple[str, int, str, str], None] = {}
    for entity, chain, number, component, insertion in zip(
        atom_entities,
        atom_chains,
        atom_numbers,
        atom_components,
        atom_insertions,
        strict=True,
    ):
        if entity not in peptide_entities:
            continue
        if selected_chain_ids is not None and chain not in selected_chain_ids:
            continue
        # entity_poly membership is authoritative even when label_seq_id is
        # absent: author IDs are the identifiers written by MMCIFParser/PDBIO.
        residue_key = (
            chain_id_map.get(chain, chain).strip(),
            int(number),
            "" if insertion in {"?", "."} else insertion.strip(),
            component,
        )
        residues[residue_key] = None
    remarks = POLYPEPTIDE_REMARK_PREFIX + "COMPLETE\n"
    remarks += "".join(
        POLYPEPTIDE_REMARK_PREFIX + json.dumps(residue, separators=(",", ":")) + "\n"
        for residue in residues
    )
    pdb_path.write_bytes(remarks.encode("ascii") + pdb_path.read_bytes())


def _coerce_structure_chain_ids_for_pdbio(structure: Any) -> dict[str, str]:
    """Assign temporary PDB-compatible chain IDs before PDBIO output."""
    original_chain_ids: list[str] = []
    seen_original_ids: set[str] = set()
    for model in structure:
        for chain in model:
            _validate_pdb_residue_numbers(chain)
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
                _validate_pdb_residue_numbers(chain)
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


def _validate_pdb_residue_numbers(chain: Any) -> None:
    """Reject author numbers that overflow the four-column legacy PDB field."""
    for residue in chain:
        author_number = residue.id[1]
        if not -999 <= author_number <= 9999:
            raise RuntimeError(
                f"Unsupported legacy PDB residue numbering: chain {chain.id!r} "
                f"has author residue number {author_number}; supported range is "
                "-999 through 9999"
            )


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
