"""Extract a solution-NMR membrane cohort and audit deposited PDBTM coverage."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from hashlib import sha256
import json
from typing import Any

from src.dataset.config import (
    MEMBRANE_ANNOTATION_TYPES,
    PROTEIN_MONOMER_ENTITY_TYPES,
    SOLUTION_NMR_METHOD,
)
from src.dataset.utils import extract_year


ENTRY_COLUMNS = (
    "entry_id", "year", "deposit_date", "initial_release_date", "revision_date",
    "methods", "deposited_model_count", "model_count_status", "protein_entity_count",
    "protein_chain_count", "max_protein_entity_chain_count", "membrane_entity_count", "membrane_chain_count",
    "membrane_annotation_sources", "has_pdbtm_annotation", "pdbtm_annotated_entity_count",
    "pdbtm_segment_chain_count", "pdbtm_segment_status", "title", "primary_citation_doi",
    "primary_citation_pubmed_id", "primary_citation_title", "related_pdb_ids",
    "related_database_records_json", "nmr_ensemble_json", "nmr_representative_json",
    "nmr_sample_details_json", "sequence_multiset_sha256", "assembly_rcsb_id",
    "assembly_global_symmetry", "assembly_global_stoichiometry", "assembly_oligomeric_state",
    "assembly_symmetry_json",
)
ENTITY_COLUMNS = (
    "entry_id", "entity_id", "description", "polymer_type", "is_membrane",
    "membrane_annotation_sources", "has_pdbtm_annotation", "annotation_records_json",
    "sequence", "sequence_length", "sequence_sha256", "uniprot_ids", "chain_count",
    "pdbtm_segment_chain_count", "pdbtm_segment_status",
)
CHAIN_COLUMNS = (
    "entry_id", "entity_id", "asym_id", "auth_asym_id", "instance_metadata_present",
    "is_membrane", "membrane_annotation_sources", "has_pdbtm_annotation",
    "modeled_residue_count", "pdbtm_valid_segment_count", "pdbtm_invalid_segment_count",
    "pdbtm_segment_status",
)
SEGMENT_COLUMNS = (
    "entry_id", "entity_id", "asym_id", "auth_asym_id", "is_membrane",
    "feature_name", "feature_description", "assignment_version", "provenance_source",
    "start_label_seq_id", "end_label_seq_id", "raw_position_json", "valid", "validation_reason",
)
DUPLICATE_CANDIDATE_COLUMNS = (
    "single_model_entry_id", "ensemble_entry_id", "sequence_multiset_sha256",
    "shared_doi", "shared_pubmed_id", "explicit_related_pdb_link", "evidence",
)


@dataclass
class MembraneBuildResult:
    """CSV-ready tables; entries, entities, chains and segments are distinct units."""

    entry_rows: list[dict[str, Any]] = field(default_factory=list)
    entity_rows: list[dict[str, Any]] = field(default_factory=list)
    chain_rows: list[dict[str, Any]] = field(default_factory=list)
    segment_rows: list[dict[str, Any]] = field(default_factory=list)
    tmdet_region_rows: list[dict[str, Any]] = field(default_factory=list)
    tmdet_residue_rows: list[dict[str, Any]] = field(default_factory=list)
    duplicate_candidate_rows: list[dict[str, Any]] = field(default_factory=list)
    excluded: list[dict[str, Any]] = field(default_factory=list)
    filter_counts: dict[str, int] = field(default_factory=dict)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if number > 0 and str(value).strip() in {str(number), f"{number}.0"} else None


def _sources(entity: dict[str, Any]) -> list[str]:
    found = {annotation.get("type") for annotation in entity.get("rcsb_polymer_entity_annotation") or []}
    return [source for source in MEMBRANE_ANNOTATION_TYPES if source in found]


def _instances(entity: dict[str, Any]) -> list[dict[str, Any]]:
    """Include declared chains whose instance metadata is absent in the response."""
    instances = entity.get("polymer_entity_instances") or []
    by_asym = {}
    for instance in instances:
        identifiers = instance.get("rcsb_polymer_entity_instance_container_identifiers") or {}
        asym_id = identifiers.get("asym_id")
        if not asym_id or asym_id in by_asym:
            raise ValueError("Missing or duplicate polymer instance asym_id")
        by_asym[asym_id] = instance
    identifiers = entity.get("rcsb_polymer_entity_container_identifiers") or {}
    for asym_id in identifiers.get("asym_ids") or []:
        by_asym.setdefault(asym_id, {
            "rcsb_polymer_entity_instance_container_identifiers": {"asym_id": asym_id},
            "_missing_instance_metadata": True,
        })
    return [by_asym[key] for key in sorted(by_asym)]


def _assembly_symmetry(entry: dict[str, Any], protein_asym_ids: set[str]) -> dict[str, Any]:
    """Describe the one RCSB assembly that contains each deposited protein chain once."""
    matches = []
    for assembly in entry.get("assemblies") or []:
        listed = {asym_id for generator in assembly.get("pdbx_struct_assembly_gen") or []
                  for asym_id in generator.get("asym_id_list") or []}
        protein_count = (assembly.get("rcsb_assembly_info") or {}).get("polymer_entity_instance_count_protein")
        # Equal protein counts exclude assemblies that repeat a chain under another operator.
        if protein_asym_ids <= listed and protein_count == len(protein_asym_ids):
            matches.append(assembly)
    if len(matches) != 1:
        return {"assembly_rcsb_id": None, "assembly_global_symmetry": None, "assembly_global_stoichiometry": None,
                "assembly_oligomeric_state": None, "assembly_symmetry_json": None}
    records = matches[0].get("rcsb_struct_symmetry") or []
    global_records = [record for record in records if record.get("kind") == "Global Symmetry"]
    symmetry = global_records[0] if len(global_records) == 1 else {}
    return {
        "assembly_rcsb_id": matches[0].get("rcsb_id"),
        "assembly_global_symmetry": symmetry.get("symbol"),
        "assembly_global_stoichiometry": ";".join(symmetry.get("stoichiometry") or []) or None,
        "assembly_oligomeric_state": symmetry.get("oligomeric_state"),
        "assembly_symmetry_json": _json(records),
    }


def _coverage_status(chains: list[dict[str, Any]]) -> str:
    if chains and all(chain["pdbtm_segment_status"] == "complete" for chain in chains):
        return "complete"
    if any(chain["pdbtm_valid_segment_count"] > 0 for chain in chains):
        return "partial"
    return "missing"


def _chain_segments(
    instance: dict[str, Any], base: dict[str, Any], sequence_length: int | None,
) -> list[dict[str, Any]]:
    rows = []
    for feature in instance.get("rcsb_polymer_instance_feature") or []:
        if feature.get("type") != "MEMBRANE_SEGMENT" or feature.get("provenance_source") != "PDBTM":
            continue
        for position in feature.get("feature_positions") or [None]:
            raw = position or {}
            start = _positive_int(raw.get("beg_seq_id"))
            end = _positive_int(raw.get("end_seq_id"))
            reason = ""
            if start is None or end is None:
                reason = "missing or invalid residue boundary"
            elif start > end:
                reason = "start exceeds end"
            elif sequence_length is not None and end > sequence_length:
                reason = "boundary exceeds deposited sequence length"
            rows.append({
                **base,
                "feature_name": feature.get("name"),
                "feature_description": feature.get("description"),
                "assignment_version": feature.get("assignment_version"),
                "provenance_source": feature.get("provenance_source"),
                "start_label_seq_id": start, "end_label_seq_id": end,
                "raw_position_json": _json(position), "valid": not reason,
                "validation_reason": reason,
            })
    return rows


def find_duplicate_candidates(entry_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flag pairs for review, requiring sequence/copy equality and a citation or link."""
    by_sequence: dict[str, list[dict[str, Any]]] = {}
    for row in entry_rows:
        fingerprint = row["sequence_multiset_sha256"]
        if fingerprint:
            by_sequence.setdefault(fingerprint, []).append(row)
    candidates = []
    for fingerprint, rows in sorted(by_sequence.items()):
        singles = [row for row in rows if row["deposited_model_count"] == 1]
        ensembles = [row for row in rows if (row["deposited_model_count"] or 0) > 1]
        for single in singles:
            for ensemble in ensembles:
                doi = single["primary_citation_doi"]
                doi = doi if doi and doi == ensemble["primary_citation_doi"] else None
                pmid = single["primary_citation_pubmed_id"]
                pmid = pmid if pmid and pmid == ensemble["primary_citation_pubmed_id"] else None
                linked = (
                    ensemble["entry_id"] in single["related_pdb_ids"].split(";")
                    or single["entry_id"] in ensemble["related_pdb_ids"].split(";")
                )
                if not (doi or pmid or linked):
                    continue
                evidence = ["identical deposited protein sequences and chain multiplicities"]
                evidence.extend(label for flag, label in (
                    (doi, "shared primary DOI"), (pmid, "shared primary PubMed ID"),
                    (linked, "explicit related PDB link"),
                ) if flag)
                candidates.append({
                    "single_model_entry_id": single["entry_id"],
                    "ensemble_entry_id": ensemble["entry_id"],
                    "sequence_multiset_sha256": fingerprint, "shared_doi": doi,
                    "shared_pubmed_id": pmid, "explicit_related_pdb_link": linked,
                    "evidence": "; ".join(evidence),
                })
    return sorted(candidates, key=lambda row: (row["single_model_entry_id"], row["ensemble_entry_id"]))


class SolutionNMRMembraneBuilder:
    """Apply the cohort rules to raw GraphQL records, without network or file IO."""

    def build_from_entries(
        self, entries: list[dict[str, Any]], require_multiple_models: bool = True,
    ) -> MembraneBuildResult:
        result = MembraneBuildResult()
        counts: Counter[str] = Counter(
            input_entries=len(entries), after_polypeptide_filter=0,
            after_exact_method_filter=0, after_membrane_annotation_filter=0,
            after_model_filter=0,
        )
        seen = set()
        for entry in sorted(entries, key=lambda item: item.get("rcsb_id") or ""):
            entry_id = entry.get("rcsb_id")
            if not isinstance(entry_id, str) or not entry_id or entry_id in seen:
                raise ValueError("Missing or duplicate PDB entry identifier")
            seen.add(entry_id)
            accession = entry.get("rcsb_accession_info") or {}
            year = extract_year(accession.get("deposit_date"))
            proteins = [entity for entity in entry.get("polymer_entities") or []
                        if (entity.get("entity_poly") or {}).get("type") in PROTEIN_MONOMER_ENTITY_TYPES]
            methods = {item.get("method") for item in entry.get("exptl") or []}
            model_count = _positive_int((entry.get("rcsb_entry_info") or {}).get("deposited_model_count"))
            reason = None
            checks = (
                ("after_polypeptide_filter", bool(proteins) and any(_instances(entity) for entity in proteins),
                 "no polypeptide chain"),
                ("after_exact_method_filter", methods == {SOLUTION_NMR_METHOD},
                 "experimental method set is not exactly [SOLUTION NMR]"),
                ("after_membrane_annotation_filter", any(_sources(entity) for entity in proteins),
                 "no supported membrane annotation on a protein entity"),
                ("after_model_filter", not require_multiple_models or (model_count is not None and model_count > 1),
                 "deposited model count is missing or invalid" if model_count is None else "not more than one deposited model"),
            )
            for stage, passes, rejection_reason in checks:
                if not passes:
                    reason = rejection_reason
                    break
                counts[stage] += 1
            if reason:
                result.excluded.append({"entry_id": entry_id, "year": year, "reason": reason})
                counts[reason] += 1
                continue

            entities, chains, segments, sequence_copies = [], [], [], []
            complete_sequences = True
            for entity in proteins:
                identifiers = entity.get("rcsb_polymer_entity_container_identifiers") or {}
                entity_id = identifiers.get("entity_id")
                if entity_id is None:
                    raise ValueError(f"Missing protein entity identifier for {entry_id}")
                sources = _sources(entity)
                is_membrane = bool(sources)
                polymer = entity.get("entity_poly") or {}
                sequence = "".join((polymer.get("pdbx_seq_one_letter_code_can") or "").split()).upper()
                length = len(sequence) if sequence else _positive_int(polymer.get("rcsb_sample_sequence_length"))
                instances = _instances(entity)
                complete_sequences &= bool(sequence) and bool(instances)
                sequence_copies.extend([sequence] * len(instances))
                entity_chains = []
                for instance in instances:
                    chain_ids = instance.get("rcsb_polymer_entity_instance_container_identifiers") or {}
                    base = {"entry_id": entry_id, "entity_id": entity_id,
                            "asym_id": chain_ids.get("asym_id"), "auth_asym_id": chain_ids.get("auth_asym_id"),
                            "is_membrane": is_membrane}
                    chain_segments = _chain_segments(instance, base, length)
                    valid = sum(row["valid"] for row in chain_segments)
                    invalid = len(chain_segments) - valid
                    modeled = (instance.get("rcsb_polymer_instance_info") or {}).get("modeled_residue_count")
                    chain = {
                        **base, "instance_metadata_present": not instance.get("_missing_instance_metadata", False),
                        "membrane_annotation_sources": ";".join(sources), "has_pdbtm_annotation": "PDBTM" in sources,
                        "modeled_residue_count": 0 if modeled == 0 and not isinstance(modeled, bool) else _positive_int(modeled),
                        "pdbtm_valid_segment_count": valid, "pdbtm_invalid_segment_count": invalid,
                        "pdbtm_segment_status": "complete" if valid and not invalid else "partial" if valid else "missing",
                    }
                    entity_chains.append(chain)
                    segments.extend(chain_segments)
                chains.extend(entity_chains)
                entities.append({
                    "entry_id": entry_id, "entity_id": entity_id,
                    "description": (entity.get("rcsb_polymer_entity") or {}).get("pdbx_description"),
                    "polymer_type": polymer.get("type"), "is_membrane": is_membrane,
                    "membrane_annotation_sources": ";".join(sources), "has_pdbtm_annotation": "PDBTM" in sources,
                    "annotation_records_json": _json(entity.get("rcsb_polymer_entity_annotation") or []),
                    "sequence": sequence or None, "sequence_length": length,
                    "sequence_sha256": sha256(sequence.encode()).hexdigest() if sequence else None,
                    "uniprot_ids": ";".join(sorted(identifiers.get("uniprot_ids") or [])),
                    "chain_count": len(entity_chains),
                    "pdbtm_segment_chain_count": sum(chain["pdbtm_valid_segment_count"] > 0 for chain in entity_chains),
                    "pdbtm_segment_status": _coverage_status(entity_chains),
                })
            membrane_entities = [entity for entity in entities if entity["is_membrane"]]
            membrane_chains = [chain for chain in chains if chain["is_membrane"]]
            citation = entry.get("rcsb_primary_citation") or {}
            pubmed_id = _positive_int(citation.get("pdbx_database_id_PubMed"))
            related = entry.get("pdbx_database_related") or []
            sources = [source for source in MEMBRANE_ANNOTATION_TYPES
                       if any(source in entity["membrane_annotation_sources"].split(";") for entity in entities)]
            coverage = _coverage_status(membrane_chains)
            if coverage == "complete" and any(entity["chain_count"] == 0 for entity in membrane_entities):
                coverage = "partial"
            result.entry_rows.append({
                "entry_id": entry_id, "year": year, "deposit_date": accession.get("deposit_date"),
                "initial_release_date": accession.get("initial_release_date"), "revision_date": accession.get("revision_date"),
                "methods": ";".join(sorted(methods)), "deposited_model_count": model_count,
                "model_count_status": "unknown" if model_count is None else "single" if model_count == 1 else "multiple",
                "protein_entity_count": len(entities), "protein_chain_count": len(chains),
                "max_protein_entity_chain_count": max(entity["chain_count"] for entity in entities),
                "membrane_entity_count": len(membrane_entities), "membrane_chain_count": len(membrane_chains),
                "membrane_annotation_sources": ";".join(sources), "has_pdbtm_annotation": "PDBTM" in sources,
                "pdbtm_annotated_entity_count": sum(entity["has_pdbtm_annotation"] for entity in membrane_entities),
                "pdbtm_segment_chain_count": sum(chain["pdbtm_valid_segment_count"] > 0 for chain in membrane_chains),
                "pdbtm_segment_status": coverage, "title": (entry.get("struct") or {}).get("title"),
                "primary_citation_doi": str(citation.get("pdbx_database_id_DOI") or "").strip().lower() or None,
                "primary_citation_pubmed_id": str(pubmed_id) if pubmed_id is not None else None,
                "primary_citation_title": citation.get("title"),
                "related_pdb_ids": ";".join(sorted({str(item["db_id"]).upper() for item in related
                                                  if str(item.get("db_name", "")).upper() == "PDB" and item.get("db_id")})),
                "related_database_records_json": _json(related),
                "nmr_ensemble_json": _json(entry.get("pdbx_nmr_ensemble")),
                "nmr_representative_json": _json(entry.get("pdbx_nmr_representative")),
                "nmr_sample_details_json": _json(entry.get("pdbx_nmr_sample_details")),
                "sequence_multiset_sha256": sha256(_json(sorted(sequence_copies)).encode()).hexdigest() if complete_sequences else None,
                **_assembly_symmetry(entry, {chain["asym_id"] for chain in chains}),
            })
            result.entity_rows.extend(entities)
            result.chain_rows.extend(chains)
            result.segment_rows.extend(segments)
        result.duplicate_candidate_rows = find_duplicate_candidates(result.entry_rows)
        counts["retained_entries"] = len(result.entry_rows)
        counts["excluded_entries"] = len(result.excluded)
        result.filter_counts = dict(counts)
        return result
