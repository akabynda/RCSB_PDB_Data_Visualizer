"""Map TmDet XML regions onto actual first-model PDB residues."""

from collections import Counter, defaultdict
from fractions import Fraction
import json
import math
import xml.etree.ElementTree as ET

from src.dataset.pdb_normalization import pdb_atom_field_offset

TMDET_SUMMARY_COLUMNS = (
    "tmdet_status", "tmdet_error", "tmdet_version", "tmdet_protein_type", "tmdet_q_value",
    "tmdet_modeled_residue_count", "tmdet_assigned_residue_count", "tmdet_unassigned_residue_count",
    "tmdet_membrane_residue_count", "tmdet_other_residue_count", "tmdet_membrane_fraction",
    "tmdet_region_counts_json",
)
TMDET_EXCLUSION_COLUMNS = (
    "tmdet_counted_residue_count", "tmdet_excluded_chain_count", "tmdet_excluded_residue_count",
)
TMDET_COPY_COLUMNS = (
    "tmdet_copy_mean_modeled_residue_count", "tmdet_copy_mean_membrane_residue_count",
    "tmdet_copy_mean_other_residue_count", "tmdet_copy_min_membrane_residue_count",
    "tmdet_copy_max_membrane_residue_count",
)
TMDET_ENTRY_COLUMNS = (
    *TMDET_SUMMARY_COLUMNS, *TMDET_EXCLUSION_COLUMNS, *TMDET_COPY_COLUMNS,
    "tmdet_model_index", "tmdet_result_directory",
)
COMPLETE_TMDET_STATUSES = frozenset({"membrane_detected", "no_membrane_detected"})
COUNTED_TMDET_STATUSES = COMPLETE_TMDET_STATUSES | {"partial_annotation"}
# TmDet 4 Region::isAnnotatedMembraneType: TM helix/strand, re-entrant loop,
# interfacial helix and beta-barrel inside element.
MEMBRANE_REGION_CODES = frozenset("HBLFN")
TMDET_RESIDUE_COLUMNS = (
    "entry_id", "entity_id", "asym_id", "auth_asym_id", "pdb_chain_id", "auth_seq_id",
    "insertion_code", "residue_name", "is_membrane", "has_ca", "region_type", "assignment_status",
)
TMDET_REGION_COLUMNS = (
    "entry_id", "pdb_chain_id", "region_index", "region_type", "start_auth_seq_id",
    "start_insertion_code", "end_auth_seq_id", "end_insertion_code", "modeled_residue_count",
    "raw_attributes_json",
)


def collect_modeled_residues(model_text, chain_rows, polymer_metadata, chain_id_map):
    """Count each protein residue with coordinates once, preserving author keys."""
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
            "pdb_chain_id": chain_id, "auth_seq_id": number, "insertion_code": insertion,
            "residue_name": name, "has_ca": False, "region_type": None, "assignment_status": "unassigned",
        })
        if row["residue_name"] != name:
            raise ValueError(f"Conflicting residue names at {key}")
        if line[12:16].strip() == "CA" and line[76 + offset:78 + offset].strip() in {"", "C"}:
            row["has_ca"] = True
    return list(residues.values())


def _insertion_code(value):
    """TmDet 4 serializes a nonblank C++ char as its integer ASCII value."""
    if value is None or value == "":
        return ""
    if value.isdigit() and 33 <= int(value) <= 126:
        return chr(int(value))
    if len(value) == 1:
        return value
    raise ValueError(f"Invalid TmDet insertion code: {value}")


def _summarize(rows, metadata, status, excluded=frozenset()):
    """Count residues of chains not in `excluded`; an unassigned counted residue blanks other/fraction."""
    counts = Counter(row["region_type"] for row in rows if row["region_type"] is not None)
    assigned = sum(counts.values())
    n = len(rows)
    valid_negative = status == "no_membrane_detected"
    unusable = status in {"no_analyzable_protein", "not_in_model"}
    kept = [row for row in rows if row["asym_id"] not in excluded]
    membrane = None if unusable or not kept else sum(row["region_type"] in MEMBRANE_REGION_CODES for row in kept)
    complete = not unusable and bool(kept) and (
        valid_negative or all(row["region_type"] is not None for row in kept))
    return {
        "tmdet_status": status, "tmdet_error": "", **metadata,
        "tmdet_modeled_residue_count": n, "tmdet_assigned_residue_count": assigned,
        "tmdet_unassigned_residue_count": n - assigned,
        "tmdet_membrane_residue_count": membrane,
        "tmdet_other_residue_count": len(kept) - membrane if complete else None,
        "tmdet_membrane_fraction": membrane / len(kept) if complete else None,
        "tmdet_region_counts_json": json.dumps(dict(sorted(counts.items())), separators=(",", ":")),
    }


def summarize_entity_copies(entry_status, chain_rows):
    """Average complete chain counts over copies of each entity, then sum the entities.

    Partially annotated chains are excluded, as in the entry totals; an entity
    without remaining copies drops out. Min/max sum each entity's fewest/most
    membrane residues among its copies. Counts are blank unless every other
    protein chain has a complete TmDet result.
    """
    chain_rows = [chain for chain in chain_rows if chain.get("tmdet_status") != "partial_annotation"]
    if entry_status not in COUNTED_TMDET_STATUSES or not chain_rows or any(
        chain.get("tmdet_status") not in COMPLETE_TMDET_STATUSES
        or chain.get("tmdet_other_residue_count") is None for chain in chain_rows
    ):
        return dict.fromkeys(TMDET_COPY_COLUMNS)
    copies = defaultdict(list)
    for chain in chain_rows:
        copies[chain["entity_id"]].append(chain)
    modeled, membrane, low, high = Fraction(0), Fraction(0), 0, 0
    for rows in copies.values():
        counts = [row["tmdet_membrane_residue_count"] for row in rows]
        modeled += Fraction(sum(row["tmdet_modeled_residue_count"] for row in rows), len(rows))
        membrane += Fraction(sum(counts), len(rows))
        low, high = low + min(counts), high + max(counts)
    # Exact rationals keep mean membrane + mean other equal to the mean modeled count.
    values = [value.numerator if value.denominator == 1 else float(value)
              for value in (modeled, membrane, modeled - membrane)]
    return dict(zip(TMDET_COPY_COLUMNS, (*values, low, high)))


def parse_tmdet_annotations(xml_text, modeled_residues, chain_rows):
    """Retain every region code; use observed coordinate order, never span length."""
    root = ET.fromstring(xml_text)
    for element in root.iter():
        element.tag = element.tag.rsplit("}", 1)[-1]
    if root.tag != "pdbtm" or root.get("transmembrane") not in {"yes", "no"}:
        raise ValueError("Invalid TmDet XML root or transmembrane status")
    version = root.findtext("rawData/tmdetVersion") or ""
    protein_type = root.findtext("rawData/tmType") or ""
    if not version.startswith("4.") or not protein_type:
        raise ValueError("Expected TmDet 4 XML with version and protein type")
    q_value = float(root.findtext("rawData/qValue"))
    if not math.isfinite(q_value):
        raise ValueError("Nonfinite TmDet qValue")
    metadata = {"tmdet_version": version, "tmdet_protein_type": protein_type, "tmdet_q_value": q_value}
    residues = [dict(row) for row in modeled_residues]
    by_chain = defaultdict(list)
    for row in residues:
        by_chain[row["pdb_chain_id"]].append(row)
    regions = []
    for chain in root.findall("chains/chain"):
        chain_id = chain.get("authId", "")
        observed = by_chain.get(chain_id, [])
        keys = [(r["auth_seq_id"], r["insertion_code"]) for r in observed]
        for index, region in enumerate(chain.findall("regions/region"), 1):
            start = int(region.attrib["startAuthId"]), _insertion_code(region.get("startAuthIcode"))
            end = int(region.attrib["endAuthId"]), _insertion_code(region.get("endAuthIcode"))
            code = region.get("type")
            if not code or start not in keys or end not in keys:
                raise ValueError(f"Cannot map TmDet region onto modeled chain {chain_id}: {start}–{end}")
            first, last = keys.index(start), keys.index(end)
            if first > last:
                raise ValueError("TmDet region runs backwards in coordinate order")
            for residue in observed[first:last + 1]:
                if residue["region_type"] is not None:
                    raise ValueError("Overlapping TmDet regions")
                residue["region_type"] = code
                residue["assignment_status"] = "assigned"
            regions.append({
                "entry_id": observed[0]["entry_id"], "pdb_chain_id": chain_id,
                "region_index": index, "region_type": code,
                "start_auth_seq_id": start[0], "start_insertion_code": start[1],
                "end_auth_seq_id": end[0], "end_insertion_code": end[1],
                "modeled_residue_count": last - first + 1,
                "raw_attributes_json": json.dumps(region.attrib, sort_keys=True, separators=(",", ":")),
            })
    negative = root.get("transmembrane") == "no"
    if negative and regions:
        raise ValueError("Negative TmDet result unexpectedly contains regions")
    if protein_type == "Noprotein":
        base_status = "no_analyzable_protein"
    elif negative:
        base_status = "no_membrane_detected"
    else:
        base_status = "membrane_detected"
    if negative:
        for row in residues:
            row["assignment_status"] = base_status
    summaries = {}
    for chain in chain_rows:
        rows = [r for r in residues if r["asym_id"] == chain["asym_id"]]
        chain_status = base_status if rows else "not_in_model"
        if rows and not negative and any(r["region_type"] is None for r in rows):
            chain_status = "partial_annotation"
        summaries[chain["asym_id"]] = _summarize(rows, metadata, chain_status)
    # A chain with any unassigned residue is excluded whole from the entry totals.
    excluded = {asym for asym, row in summaries.items() if row["tmdet_status"] == "partial_annotation"}
    status = "partial_annotation" if excluded else base_status
    summary = _summarize(residues, metadata, status, excluded)
    excluded_count = sum(r["asym_id"] in excluded for r in residues)
    summary.update({
        "tmdet_counted_residue_count": (
            None if summary["tmdet_other_residue_count"] is None else len(residues) - excluded_count),
        "tmdet_excluded_chain_count": len(excluded), "tmdet_excluded_residue_count": excluded_count,
    })
    return summary, summaries, regions, residues
