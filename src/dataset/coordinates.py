"""Parse modeled alpha-carbon residues and coordinates from PDB records."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np
from Bio.SeqUtils import seq1

from src.dataset.records import (
    CAResidueRecord,
)


def extract_model_pdb_texts(pdb_path: Path) -> list[str]:
    """Split a PDB file into separate text blocks for each model."""
    model_texts: list[str] = []
    model_lines: list[str] = []
    saw_model = False
    in_model = False

    with pdb_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            record = line[:6]
            if record.startswith("MODEL"):
                if in_model and model_lines:
                    model_lines.append("END\n")
                    model_texts.append("".join(model_lines))
                    model_lines = []
                saw_model = True
                in_model = True
                continue
            if record.startswith("ENDMDL"):
                if in_model and model_lines:
                    model_lines.append("END\n")
                    model_texts.append("".join(model_lines))
                    model_lines = []
                in_model = False
                continue
            if saw_model and not in_model:
                continue
            if (
                record.startswith("ATOM")
                or record.startswith("HETATM")
                or record.startswith("TER")
            ):
                model_lines.append(line)

    if model_lines:
        model_lines.append("END\n")
        model_texts.append("".join(model_lines))

    return model_texts


def parse_models_ca_coords(
    pdb_path: Path,
    chain_id: str,
    start_seq_id: int | None = None,
    end_seq_id: int | None = None,
) -> list[dict[int, np.ndarray]]:
    """Parse CA coordinates from every model in a PDB file."""
    model_maps, _ = parse_models_ca_coords_with_stats(
        pdb_path=pdb_path,
        chain_id=chain_id,
        start_seq_id=start_seq_id,
        end_seq_id=end_seq_id,
    )
    return model_maps


def parse_first_model_ca_residue_sequence(
    pdb_path: Path,
    chain_id: str,
    start_seq_id: int | None = None,
    end_seq_id: int | None = None,
    include_hetatm: bool = True,
) -> list[tuple[int, str]]:
    """Return residue identities for CA atoms in the first model."""
    return [
        (record.resid, record.identity)
        for record in parse_first_model_ca_residues(
            pdb_path=pdb_path,
            chain_id=chain_id,
            start_seq_id=start_seq_id,
            end_seq_id=end_seq_id,
            include_hetatm=include_hetatm,
        )
    ]


def _parse_first_model_ca_line_fields(
    line: str,
) -> tuple[str, int, str, str, float, str] | None:
    """Parse carbon CA fields, including nonstandard long component IDs."""
    atom_name = line[12:16].strip()
    if atom_name != "CA":
        return None

    atom_chain = line[21].strip()
    resid_text = line[22:26].strip()
    insertion_code = line[26].strip()
    alt_loc = line[16].strip()
    occupancy = _parse_pdb_occupancy(line)
    resname = line[17:20].strip()
    try:
        resid = int(resid_text)
    except ValueError:
        resid = None
    if resid is not None:
        if occupancy == float("-inf"):
            return None
    else:
        parts = line.split()
        if len(parts) < 10 or parts[2] != "CA":
            return None
        match = re.fullmatch(r"(-?\d+)([A-Za-z]?)", parts[5])
        if match is None:
            return None
        try:
            resid = int(match.group(1))
            occupancy = float(parts[9])
        except ValueError:
            return None
        atom_chain = parts[4]
        insertion_code = match.group(2)
        alt_loc = ""
        resname = parts[3]

    # Long component IDs shift the element field along with the other columns.
    # Keep carbon HETATM records (modified amino acids), but never calcium ions,
    # including legacy records that omit the element field entirely.
    element_column = 76 + max(0, len(resname) - 3)
    element = line[element_column : element_column + 2].strip().upper()
    if resname.upper() == "CA" or (element and element != "C"):
        return None
    return atom_chain, resid, insertion_code, alt_loc, occupancy, resname


def parse_first_model_ca_residues(
    pdb_path: Path,
    chain_id: str,
    start_seq_id: int | None = None,
    end_seq_id: int | None = None,
    include_hetatm: bool = True,
) -> list[CAResidueRecord]:
    """Return first-model CA residues and whether each position has HETATM CA."""
    residue_order: list[int] = []
    candidates: dict[int, tuple[str, float, str, CAResidueRecord]] = {}
    hetatm_ca_resids: set[int] = set()
    modres_identity_by_key = _parse_pdb_modres_identity_map(pdb_path)
    has_model_records = False
    in_model = False

    with pdb_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            record = line[:6]
            if record.startswith("MODEL"):
                if has_model_records:
                    break
                has_model_records = True
                in_model = True
                continue
            if record.startswith("ENDMDL"):
                if in_model:
                    break
                continue
            if has_model_records and not in_model:
                continue
            is_standard_atom = record.startswith("ATOM")
            is_hetero_atom = record.startswith("HETATM")
            if not is_standard_atom and not (include_hetatm and is_hetero_atom):
                continue
            parsed_fields = _parse_first_model_ca_line_fields(line)
            if parsed_fields is None:
                continue
            (
                atom_chain,
                resid,
                insertion_code,
                alt_loc,
                occupancy,
                resname,
            ) = parsed_fields
            if atom_chain != chain_id:
                continue
            if start_seq_id is not None and resid < start_seq_id:
                continue
            if end_seq_id is not None and resid > end_seq_id:
                continue

            if occupancy <= 0.0:
                continue
            if is_hetero_atom:
                hetatm_ca_resids.add(resid)
            if is_standard_atom:
                identity = seq1(resname, custom_map={"MSE": "M"}, undef_code="X")
            else:
                identity = modres_identity_by_key.get(
                    (atom_chain, resid, insertion_code, resname),
                    modres_identity_by_key.get(
                        (atom_chain, resid, "", resname),
                        f"HET:{resname}",
                    ),
                )
            ca_record = CAResidueRecord(
                resid=resid,
                identity=identity,
                is_standard_atom=is_standard_atom,
                has_hetatm_ca=is_hetero_atom,
            )

            if resid not in candidates:
                residue_order.append(resid)
                candidates[resid] = (insertion_code, occupancy, alt_loc, ca_record)
                continue

            (
                existing_insertion_code,
                existing_occupancy,
                existing_alt_loc,
                existing_record,
            ) = candidates[resid]
            if ca_record.is_standard_atom != existing_record.is_standard_atom:
                if ca_record.is_standard_atom:
                    candidates[resid] = (
                        insertion_code,
                        occupancy,
                        alt_loc,
                        ca_record,
                    )
                continue
            if _is_better_ca_candidate(
                new_insertion_code=insertion_code,
                new_occupancy=occupancy,
                new_alt_loc=alt_loc,
                current_insertion_code=existing_insertion_code,
                current_occupancy=existing_occupancy,
                current_alt_loc=existing_alt_loc,
            ):
                candidates[resid] = (insertion_code, occupancy, alt_loc, ca_record)

    records: list[CAResidueRecord] = []
    for resid in residue_order:
        candidate = candidates.get(resid)
        if candidate is None:
            continue
        selected = candidate[3]
        records.append(
            CAResidueRecord(
                resid=selected.resid,
                identity=selected.identity,
                is_standard_atom=selected.is_standard_atom,
                has_hetatm_ca=resid in hetatm_ca_resids,
            )
        )
    return records


def parse_first_model_modeled_ca_auth_seq_ids(
    pdb_path: Path,
    chain_id: str,
) -> set[int]:
    """Return author residue IDs with positive-occupancy first-model CA atoms."""
    return {
        record.resid
        for record in parse_first_model_ca_residues(
            pdb_path=pdb_path,
            chain_id=chain_id,
            include_hetatm=True,
        )
    }


def _parse_pdb_modres_identity_map(
    pdb_path: Path,
) -> dict[tuple[str, int, str, str], str]:
    """Parse MODRES records into modified-to-standard residue identity mappings."""
    identity_by_key: dict[tuple[str, int, str, str], str] = {}
    with pdb_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("MODRES"):
                continue
            fixed_line = line.rstrip("\r\n").ljust(27)
            resname = fixed_line[12:15].strip()
            chain_id = fixed_line[16].strip()
            seq_num_text = fixed_line[18:22].strip()
            insertion_code = fixed_line[22].strip()
            standard_resname = fixed_line[24:27].strip()
            if not (resname and chain_id and seq_num_text and standard_resname):
                parts = line.split()
                if len(parts) < 6:
                    continue
                resname = parts[2].strip()
                chain_id = parts[3].strip()
                seq_num_text = parts[4].strip()
                insertion_code = ""
                standard_resname = parts[5].strip()
            try:
                seq_num = int(seq_num_text)
            except ValueError:
                continue
            identity = seq1(
                standard_resname,
                custom_map={"MSE": "M"},
                undef_code="X",
            )
            if identity == "X":
                continue
            identity_by_key[(chain_id, seq_num, insertion_code, resname)] = identity
    return identity_by_key


def _alt_loc_tiebreak_key(alt_loc: str) -> tuple[int, str]:
    # Prefer blank altLoc, then A, then 1; keep deterministic order for others.
    """Rank alternate atom locations for deterministic CA selection."""
    if alt_loc == "":
        return (0, "")
    if alt_loc == "A":
        return (1, "")
    if alt_loc == "1":
        return (2, "")
    return (3, alt_loc)


def _insertion_code_tiebreak_key(insertion_code: str) -> tuple[int, str]:
    # Prefer residue numbers without insertion codes (e.g., 102 over 102A).
    """Rank insertion codes for deterministic residue ordering."""
    if insertion_code == "":
        return (0, "")
    return (1, insertion_code)


def _parse_pdb_occupancy(line: str) -> float:
    """Parse the occupancy value from a PDB ATOM record."""
    occ_text = line[54:60].strip()
    if not occ_text:
        return float("-inf")
    try:
        return float(occ_text)
    except ValueError:
        return float("-inf")


def _is_better_ca_candidate(
    new_insertion_code: str,
    new_occupancy: float,
    new_alt_loc: str,
    current_insertion_code: str,
    current_occupancy: float,
    current_alt_loc: str,
) -> bool:
    """Decide whether a CA atom candidate should replace the current one."""
    new_i_code_key = _insertion_code_tiebreak_key(new_insertion_code)
    current_i_code_key = _insertion_code_tiebreak_key(current_insertion_code)
    if new_i_code_key < current_i_code_key:
        return True
    if new_i_code_key > current_i_code_key:
        return False

    if new_occupancy > current_occupancy + 1e-9:
        return True
    if current_occupancy > new_occupancy + 1e-9:
        return False
    return _alt_loc_tiebreak_key(new_alt_loc) < _alt_loc_tiebreak_key(current_alt_loc)


def parse_models_ca_coords_with_stats(
    pdb_path: Path,
    chain_id: str,
    start_seq_id: int | None = None,
    end_seq_id: int | None = None,
) -> tuple[list[dict[int, np.ndarray]], list[dict[int, int]]]:
    # Select one positive-occupancy CA per residue by max occupancy
    # (altLoc-aware) and keep raw per-residue counts so callers can report how
    # many modeled CA atoms were present before altLoc collapsing.
    """Parse model CA coordinates and report residue-selection statistics."""
    models: list[dict[int, np.ndarray]] = []
    raw_ca_counts_per_model: list[dict[int, int]] = []
    current_candidates: dict[int, tuple[str, float, str, bool, np.ndarray]] = {}
    current_raw_counts: Counter[int] = Counter()
    has_model_records = False
    in_model = False

    def finalize_model() -> None:
        """Finalize one parsed model and reset per-model parsing buffers."""
        models.append(
            {resid: candidate[4] for resid, candidate in current_candidates.items()}
        )
        raw_ca_counts_per_model.append(dict(current_raw_counts))

    with pdb_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            record = line[:6]
            if record.startswith("MODEL"):
                if in_model:
                    finalize_model()
                    current_candidates = {}
                    current_raw_counts = Counter()
                has_model_records = True
                in_model = True
                continue
            if record.startswith("ENDMDL"):
                if in_model:
                    finalize_model()
                    current_candidates = {}
                    current_raw_counts = Counter()
                    in_model = False
                continue
            is_standard_atom = record.startswith("ATOM")
            is_hetero_atom = record.startswith("HETATM")
            if not is_standard_atom and not is_hetero_atom:
                continue

            parsed_fields = _parse_first_model_ca_line_fields(line)
            if parsed_fields is None:
                continue
            (
                atom_chain,
                resid,
                insertion_code,
                alt_loc,
                occupancy,
                _resname,
            ) = parsed_fields
            if atom_chain != chain_id:
                continue
            if start_seq_id is not None and resid < start_seq_id:
                continue
            if end_seq_id is not None and resid > end_seq_id:
                continue

            if occupancy <= 0.0:
                continue
            current_raw_counts[resid] += 1

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
            coords = np.array([x, y, z], dtype=float)
            existing = current_candidates.get(resid)
            if existing is None:
                current_candidates[resid] = (
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
            ) = existing
            if is_standard_atom != existing_is_standard_atom:
                if is_standard_atom:
                    current_candidates[resid] = (
                        insertion_code,
                        occupancy,
                        alt_loc,
                        is_standard_atom,
                        coords,
                    )
                continue
            if _is_better_ca_candidate(
                new_insertion_code=insertion_code,
                new_occupancy=occupancy,
                new_alt_loc=alt_loc,
                current_insertion_code=existing_insertion_code,
                current_occupancy=existing_occupancy,
                current_alt_loc=existing_alt_loc,
            ):
                current_candidates[resid] = (
                    insertion_code,
                    occupancy,
                    alt_loc,
                    is_standard_atom,
                    coords,
                )

    if has_model_records:
        if in_model or current_candidates or current_raw_counts:
            finalize_model()
    elif current_candidates or current_raw_counts:
        finalize_model()
    return models, raw_ca_counts_per_model
