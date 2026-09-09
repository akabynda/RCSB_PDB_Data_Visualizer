"""Parse modeled alpha-carbon residues and coordinates from PDB records."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
from Bio.SeqUtils import seq1

from src.dataset.records import (
    CAResidueRecord,
    ResidueId,
    parse_residue_id,
)

from src.dataset.pdb_normalization import (
    PDBPolymerMetadata,
    iter_normalized_pdb_lines,
    load_pdb_polymer_metadata,
    pdb_atom_field_offset,
)


def _residue_keys_in_range(
    keys: list[ResidueId],
    start_seq_id: ResidueId | int | None,
    end_seq_id: ResidueId | int | None,
) -> list[ResidueId]:
    """Slice in coordinate order, including reverse and nonmonotonic numbering."""
    start = parse_residue_id(start_seq_id) if start_seq_id is not None else None
    end = parse_residue_id(end_seq_id) if end_seq_id is not None else None
    if (start is None or start in keys) and (end is None or end in keys):
        first = keys.index(start) if start is not None else 0
        last = keys.index(end) + 1 if end is not None else len(keys)
        return keys[first:last]
    # Legacy callers can give unmodeled numeric bounds (e.g. a numbering gap).
    return [
        key
        for key in keys
        if (start is None or key >= start) and (end is None or key <= end)
    ]


def extract_model_pdb_texts(pdb_path: Path) -> list[str]:
    """Split a PDB file into separate text blocks for each model."""
    model_texts: list[str] = []
    model_lines: list[str] = []
    saw_model = False
    in_model = False

    for line in iter_normalized_pdb_lines(pdb_path):
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
    start_seq_id: ResidueId | int | None = None,
    end_seq_id: ResidueId | int | None = None,
) -> list[dict[ResidueId, np.ndarray]]:
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
    start_seq_id: ResidueId | int | None = None,
    end_seq_id: ResidueId | int | None = None,
    include_hetatm: bool = True,
) -> list[tuple[ResidueId, str]]:
    """Return residue identities for CA atoms in the first model."""
    return [
        (record.key, record.identity)
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
    polymer_metadata: PDBPolymerMetadata | None = None,
) -> tuple[str, int, str, str, float, str] | None:
    """Parse carbon CA fields, including nonstandard long component IDs."""
    if len(line) < 27 or line[12:16].strip() != "CA":
        return None
    offset = pdb_atom_field_offset(line)
    atom_chain = line[21 + offset : 22 + offset].strip()
    resid_text = line[22 + offset : 26 + offset].strip()
    insertion_code = line[26 + offset : 27 + offset].strip()
    alt_loc = line[16].strip()
    resname = line[17 : 20 + offset].strip()
    try:
        resid = int(resid_text)
        occupancy = float(line[54 + offset : 60 + offset])
    except ValueError:
        return None
    element = line[76 + offset : 78 + offset].strip().upper()
    if resname.upper() == "CA" or (element and element != "C"):
        return None
    if line.startswith("HETATM") and not (
        polymer_metadata or PDBPolymerMetadata()
    ).includes(atom_chain, resid, insertion_code, resname):
        return None
    return atom_chain, resid, insertion_code, alt_loc, occupancy, resname


def parse_first_model_ca_residues(
    pdb_path: Path,
    chain_id: str,
    start_seq_id: ResidueId | int | None = None,
    end_seq_id: ResidueId | int | None = None,
    include_hetatm: bool = True,
) -> list[CAResidueRecord]:
    """Return first-model CA residues and whether each position has HETATM CA."""
    residue_order: list[ResidueId] = []
    candidates: dict[ResidueId, tuple[str, float, str, CAResidueRecord]] = {}
    hetatm_ca_resids: set[ResidueId] = set()
    modres_identity_by_key = _parse_pdb_modres_identity_map(pdb_path)
    polymer_metadata = load_pdb_polymer_metadata(pdb_path)
    has_model_records = False
    in_model = False

    for line in iter_normalized_pdb_lines(pdb_path):
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
        parsed_fields = _parse_first_model_ca_line_fields(line, polymer_metadata)
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
        residue_key = ResidueId(resid, insertion_code)

        if occupancy <= 0.0:
            continue
        if is_hetero_atom:
            hetatm_ca_resids.add(residue_key)
        if is_standard_atom:
            identity = seq1(resname, custom_map={"MSE": "M"}, undef_code="X")
        else:
            identity = modres_identity_by_key.get(
                (atom_chain, resid, insertion_code, resname),
                f"HET:{resname}",
            )
        ca_record = CAResidueRecord(
            resid=resid,
            insertion_code=insertion_code,
            identity=identity,
            is_standard_atom=is_standard_atom,
            has_hetatm_ca=is_hetero_atom,
        )

        if residue_key not in candidates:
            residue_order.append(residue_key)
            candidates[residue_key] = (insertion_code, occupancy, alt_loc, ca_record)
            continue

        (
            existing_insertion_code,
            existing_occupancy,
            existing_alt_loc,
            existing_record,
        ) = candidates[residue_key]
        if ca_record.is_standard_atom != existing_record.is_standard_atom:
            if ca_record.is_standard_atom:
                candidates[residue_key] = (
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
            candidates[residue_key] = (insertion_code, occupancy, alt_loc, ca_record)

    records: list[CAResidueRecord] = []
    for resid in residue_order:
        candidate = candidates.get(resid)
        if candidate is None:
            continue
        selected = candidate[3]
        records.append(
            CAResidueRecord(
                resid=selected.resid,
                insertion_code=selected.insertion_code,
                identity=selected.identity,
                is_standard_atom=selected.is_standard_atom,
                has_hetatm_ca=resid in hetatm_ca_resids,
            )
        )
    selected_keys = set(
        _residue_keys_in_range(
            [record.key for record in records], start_seq_id, end_seq_id
        )
    )
    return [record for record in records if record.key in selected_keys]


def parse_first_model_modeled_ca_auth_seq_ids(
    pdb_path: Path,
    chain_id: str,
) -> list[ResidueId]:
    """Return full author IDs in coordinate order for positive-occupancy CA."""
    return [
        record.key
        for record in parse_first_model_ca_residues(
            pdb_path=pdb_path,
            chain_id=chain_id,
            include_hetatm=True,
        )
    ]


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
    start_seq_id: ResidueId | int | None = None,
    end_seq_id: ResidueId | int | None = None,
) -> tuple[list[dict[ResidueId, np.ndarray]], list[dict[ResidueId, int]]]:
    """Parse normalized CA coordinates and count retained records per full ID."""
    models, raw_counts, _ = parse_models_ca_data(
        pdb_path, chain_id, start_seq_id, end_seq_id
    )
    return models, raw_counts


def parse_models_ca_data(
    pdb_path: Path,
    chain_id: str,
    start_seq_id: ResidueId | int | None = None,
    end_seq_id: ResidueId | int | None = None,
) -> tuple[
    list[dict[ResidueId, np.ndarray]],
    list[dict[ResidueId, int]],
    list[set[ResidueId]],
]:
    """Parse coordinates, record counts, and modeled HETATM evidence per model."""
    models: list[dict[ResidueId, np.ndarray]] = []
    raw_ca_counts_per_model: list[dict[ResidueId, int]] = []
    hetatm_ids_per_model: list[set[ResidueId]] = []
    current_candidates: dict[ResidueId, tuple[str, float, str, bool, np.ndarray]] = {}
    current_raw_counts: Counter[ResidueId] = Counter()
    current_hetatm_ids: set[ResidueId] = set()
    polymer_metadata = load_pdb_polymer_metadata(pdb_path)
    has_model_records = False
    in_model = False

    def finalize_model() -> None:
        """Finalize one parsed model and reset per-model parsing buffers."""
        keys = list(current_candidates)
        models.append({key: current_candidates[key][4] for key in keys})
        raw_ca_counts_per_model.append({key: current_raw_counts[key] for key in keys})
        hetatm_ids_per_model.append(current_hetatm_ids.intersection(keys))

    for line in iter_normalized_pdb_lines(pdb_path):
        record = line[:6]
        if record.startswith("MODEL"):
            if in_model:
                finalize_model()
                current_candidates = {}
                current_raw_counts = Counter()
                current_hetatm_ids = set()
            has_model_records = True
            in_model = True
            continue
        if record.startswith("ENDMDL"):
            if in_model:
                finalize_model()
                current_candidates = {}
                current_raw_counts = Counter()
                current_hetatm_ids = set()
                in_model = False
            continue
        is_standard_atom = record.startswith("ATOM")
        is_hetero_atom = record.startswith("HETATM")
        if not is_standard_atom and not is_hetero_atom:
            continue

        parsed_fields = _parse_first_model_ca_line_fields(line, polymer_metadata)
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
        residue_key = ResidueId(resid, insertion_code)

        if occupancy <= 0.0:
            continue
        current_raw_counts[residue_key] += 1
        if is_hetero_atom:
            current_hetatm_ids.add(residue_key)

        offset = pdb_atom_field_offset(line)
        try:
            x = float(line[30 + offset : 38 + offset].strip())
            y = float(line[38 + offset : 46 + offset].strip())
            z = float(line[46 + offset : 54 + offset].strip())
        except ValueError:
            continue
        coords = np.array([x, y, z], dtype=float)
        if not np.all(np.isfinite(coords)):
            continue
        existing = current_candidates.get(residue_key)
        if existing is None:
            current_candidates[residue_key] = (
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
                current_candidates[residue_key] = (
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
            current_candidates[residue_key] = (
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
    if models and (start_seq_id is not None or end_seq_id is not None):
        # Core boundaries refer to the first model's physical sequence. A missing
        # endpoint in a later model must not reinterpret reverse author numbering.
        selected_keys = set(
            _residue_keys_in_range(list(models[0]), start_seq_id, end_seq_id)
        )
        models = [
            {key: coords for key, coords in model.items() if key in selected_keys}
            for model in models
        ]
        raw_ca_counts_per_model = [
            {key: count for key, count in counts.items() if key in selected_keys}
            for counts in raw_ca_counts_per_model
        ]
        hetatm_ids_per_model = [ids & selected_keys for ids in hetatm_ids_per_model]
    return models, raw_ca_counts_per_model, hetatm_ids_per_model
