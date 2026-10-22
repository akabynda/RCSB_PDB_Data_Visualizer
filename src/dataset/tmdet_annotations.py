"""Map TmDet XML regions onto actual first-model PDB residues."""

from collections import Counter, defaultdict
import json
import math
import xml.etree.ElementTree as ET

from src.dataset.pdb_normalization import pdb_atom_field_offset

TMDET_SUMMARY_COLUMNS = (
    "tmdet_status", "tmdet_error", "tmdet_version", "tmdet_protein_type", "tmdet_q_value",
    "tmdet_modeled_residue_count", "tmdet_assigned_residue_count", "tmdet_unassigned_residue_count",
    "tmdet_tm_residue_count", "tmdet_other_residue_count", "tmdet_tm_fraction", "tmdet_region_counts_json",
)
TMDET_ENTRY_COLUMNS = (*TMDET_SUMMARY_COLUMNS, "tmdet_model_index", "tmdet_result_directory")
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


def _summarize(rows, metadata, status):
    counts = Counter(row["region_type"] for row in rows if row["region_type"] is not None)
    assigned = sum(counts.values())
    n = len(rows)
    valid_negative = status == "no_membrane_detected"
    unusable = status in {"no_analyzable_protein", "not_in_model"}
    tm_count = None if unusable else counts["H"] + counts["B"]
    complete = not unusable and n > 0 and (assigned == n or valid_negative)
    return {
        "tmdet_status": status, "tmdet_error": "", **metadata,
        "tmdet_modeled_residue_count": n, "tmdet_assigned_residue_count": assigned,
        "tmdet_unassigned_residue_count": n - assigned,
        "tmdet_tm_residue_count": tm_count,
        "tmdet_other_residue_count": n - tm_count if complete else None,
        "tmdet_tm_fraction": tm_count / n if complete else None,
        "tmdet_region_counts_json": json.dumps(dict(sorted(counts.items())), separators=(",", ":")),
    }


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
    status = base_status
    if not negative and any(r["region_type"] is None for r in residues):
        status = "partial_annotation"
    summary = _summarize(residues, metadata, status)
    summaries = {}
    for chain in chain_rows:
        rows = [r for r in residues if r["asym_id"] == chain["asym_id"]]
        chain_status = base_status if rows else "not_in_model"
        if rows and not negative and any(r["region_type"] is None for r in rows):
            chain_status = "partial_annotation"
        summaries[chain["asym_id"]] = _summarize(rows, metadata, chain_status)
    return summary, summaries, regions, residues
