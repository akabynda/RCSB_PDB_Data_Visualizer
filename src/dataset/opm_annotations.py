"""Count first-model residues inside the hydrophobic boundaries of OPM coordinates."""

from collections import defaultdict
from fractions import Fraction

from src.dataset.pdb_normalization import pdb_atom_field_offset

OPM_SUMMARY_COLUMNS = (
    "opm_modeled_residue_count", "opm_membrane_residue_count", "opm_other_residue_count",
    "opm_membrane_fraction", "opm_absent_residue_count",
)
OPM_COPY_COLUMNS = (
    "opm_copy_mean_modeled_residue_count", "opm_copy_mean_membrane_residue_count",
    "opm_copy_mean_other_residue_count", "opm_copy_min_membrane_residue_count",
    "opm_copy_max_membrane_residue_count",
)
OPM_ENTRY_COLUMNS = (
    "opm_status", "opm_error", "opm_type", "opm_membrane_name", "opm_half_thickness_angstrom",
    "opm_reported_thickness_angstrom", "opm_tilt_degrees", "opm_transfer_energy_kcal_per_mol",
    *OPM_SUMMARY_COLUMNS, "opm_unmatched_residue_count", "opm_residue_name_conflict_count",
    *OPM_COPY_COLUMNS,
)
OPM_RESIDUE_COLUMNS = (
    "entry_id", "entity_id", "asym_id", "auth_asym_id", "pdb_chain_id", "auth_seq_id",
    "insertion_code", "residue_name", "is_membrane", "opm_residue_name", "opm_mean_z_angstrom",
    "in_opm_file", "in_membrane_region",
)
# PDB coordinates have three decimals, so paired OPM planes agree within this bound.
OPM_PLANE_ATOL = 1e-3


def collect_modeled_residues(model_text, chain_rows, polymer_metadata, chain_id_map):
    """List each deposited protein residue with coordinates once, preserving author keys."""
    chain_lookup = {}
    for chain in chain_rows:
        auth_id = chain["auth_asym_id"]
        if auth_id is None:
            continue
        pdb_chain = chain_id_map.get(auth_id, auth_id)
        if pdb_chain in chain_lookup:
            raise ValueError(f"Ambiguous PDB chain mapping: {pdb_chain}")
        chain_lookup[pdb_chain] = chain
    residues = {}
    for line in model_text.splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        offset = pdb_atom_field_offset(line)
        chain_id = line[21 + offset:22 + offset].strip()
        if chain_id not in chain_lookup:
            continue
        number = int(line[22 + offset:26 + offset])
        insertion = line[26 + offset:27 + offset].strip()
        name = line[17:20 + offset].strip()
        if line.startswith("HETATM") and not polymer_metadata.includes(chain_id, number, insertion, name):
            continue
        key = chain_id, number, insertion
        chain = chain_lookup[chain_id]
        row = residues.setdefault(key, {
            **{field: chain[field] for field in ("entry_id", "entity_id", "asym_id", "auth_asym_id", "is_membrane")},
            "pdb_chain_id": chain_id, "auth_seq_id": number, "insertion_code": insertion, "residue_name": name,
        })
        if row["residue_name"] != name:
            raise ValueError(f"Conflicting residue names at {key}")
    return list(residues.values())


def _is_heavy_atom(line, atom_name):
    element = line[76:78].strip().upper()
    if not element.isalpha():
        element = atom_name.lstrip("0123456789")[:1].upper()
    return element not in {"H", "D"}


def parse_opm_coordinates(text):
    """Return the half-thickness D and heavy-atom mean z of each first-model residue.

    OPM places the membrane normal along z with the bilayer center at z = 0 and
    marks the hydrophobic boundaries with DUM atoms, on one or both sides. These
    atoms are used instead of the REMARK value, which is rounded, sometimes
    repeated and absent from some files.
    """
    planes, atoms, names = set(), defaultdict(dict), {}
    first_model_ended = False
    for line in text.splitlines():
        first_model_ended = first_model_ended or line.startswith("ENDMDL")
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        name, z = line[17:20].strip(), float(line[46:54])
        if name == "DUM":
            planes.add(z)
            continue
        atom_name = line[12:16].strip()
        if first_model_ended or not _is_heavy_atom(line, atom_name):
            continue
        key = line[21].strip(), int(line[22:26]), line[26].strip()
        # The first alternate position of an atom represents it.
        atoms[key].setdefault(atom_name, z)
        names.setdefault(key, name)
    if len(planes) not in {1, 2} or (len(planes) == 2 and abs(sum(planes)) > OPM_PLANE_ATOL):
        raise ValueError(f"Unexpected OPM membrane planes: {sorted(planes)}")
    half_thickness = max(abs(z) for z in planes)
    if half_thickness <= 0:
        raise ValueError("OPM membrane half-thickness must be positive")
    if not atoms:
        raise ValueError("OPM file has no first-model residues")
    return half_thickness, {
        key: {"name": names[key], "mean_z": sum(values.values()) / len(values)}
        for key, values in atoms.items()
    }


def _summarize(rows):
    """Count residues positioned by OPM; deposited residues absent from its file stay separate."""
    positioned = [row for row in rows if row["in_opm_file"]]
    membrane = sum(row["in_membrane_region"] for row in positioned)
    return {
        "opm_modeled_residue_count": len(positioned),
        "opm_membrane_residue_count": membrane,
        "opm_other_residue_count": len(positioned) - membrane,
        "opm_membrane_fraction": membrane / len(positioned) if positioned else None,
        "opm_absent_residue_count": len(rows) - len(positioned),
    }


def count_membrane_residues(deposited, positions, half_thickness, chain_rows):
    """Match deposited residues to OPM positions; |mean z| <= D places one in the membrane.

    Matching by chain, author number and insertion code ignores OPM ligands, ions
    and solvent. A blank OPM chain ID is accepted only for one deposited chain.
    Isolated name conflicts (mutations, modified residues) are only recorded; when
    they are the majority, the file holds another sequence and yields no counts.
    """
    chain_ids = {row["pdb_chain_id"] for row in deposited}
    if "" in {key[0] for key in positions} and len(chain_ids) == 1:
        only = next(iter(chain_ids))
        positions = {(key[0] or only, *key[1:]): value for key, value in positions.items()}
    rows, conflicts, matched = [], 0, set()
    for residue in deposited:
        key = residue["pdb_chain_id"], residue["auth_seq_id"], residue["insertion_code"]
        position = positions.get(key)
        if position is not None:
            matched.add(key)
            conflicts += position["name"] != residue["residue_name"]
        rows.append({
            **residue,
            "opm_residue_name": position["name"] if position else None,
            "opm_mean_z_angstrom": position["mean_z"] if position else None,
            "in_opm_file": position is not None,
            "in_membrane_region": abs(position["mean_z"]) <= half_thickness if position else None,
        })
    if not matched:
        raise ValueError("No OPM residue could be mapped to the deposited protein chains")
    if 2 * conflicts > len(matched):
        message = f"{conflicts} of {len(matched)} matched OPM residue names differ from the deposited residues"
        return {"opm_status": "structure_mismatch", "opm_error": message,
                "opm_residue_name_conflict_count": conflicts}, {}, []
    chains = {
        chain["asym_id"]: _summarize([row for row in rows if row["asym_id"] == chain["asym_id"]])
        for chain in chain_rows
    }
    return {
        "opm_status": "counted", "opm_error": "", "opm_half_thickness_angstrom": half_thickness,
        **_summarize(rows), "opm_unmatched_residue_count": len(positions.keys() - matched),
        "opm_residue_name_conflict_count": conflicts,
    }, chains, rows


def summarize_entity_copies(chain_rows):
    """Average chain counts over copies of each entity, then sum the entities.

    A chain absent from the OPM file drops out, as does an entity without
    remaining copies. Min/max sum each entity's fewest/most membrane residues
    among its copies. Counts are blank without a positioned chain.
    """
    chain_rows = [chain for chain in chain_rows if chain.get("opm_modeled_residue_count")]
    if not chain_rows:
        return dict.fromkeys(OPM_COPY_COLUMNS)
    copies = defaultdict(list)
    for chain in chain_rows:
        copies[chain["entity_id"]].append(chain)
    modeled, membrane, low, high = Fraction(0), Fraction(0), 0, 0
    for rows in copies.values():
        counts = [row["opm_membrane_residue_count"] for row in rows]
        modeled += Fraction(sum(row["opm_modeled_residue_count"] for row in rows), len(rows))
        membrane += Fraction(sum(counts), len(rows))
        low, high = low + min(counts), high + max(counts)
    # Exact rationals keep mean membrane + mean other equal to the mean modeled count.
    values = [value.numerator if value.denominator == 1 else float(value)
              for value in (modeled, membrane, modeled - membrane)]
    return dict(zip(OPM_COPY_COLUMNS, (*values, low, high)))
