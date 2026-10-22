"""Network-free tests of membrane cohort selection and annotation completeness."""

from copy import deepcopy

import pytest

from src.dataset.builders.membrane import (
    CHAIN_COLUMNS,
    DUPLICATE_CANDIDATE_COLUMNS,
    ENTITY_COLUMNS,
    ENTRY_COLUMNS,
    SEGMENT_COLUMNS,
    SolutionNMRMembraneBuilder,
)
from src.dataset.config import MEMBRANE_ANNOTATION_TYPES


def _feature(start=3, end=20, *, source="PDBTM"):
    return {
        "type": "MEMBRANE_SEGMENT", "name": "transmembrane region",
        "provenance_source": source, "feature_positions": [{"beg_seq_id": start, "end_seq_id": end}],
    }


def _entity(entity_id="1", *, source="PDBTM", chains=("A",), sequence="A" * 30):
    return {
        "entity_poly": {"type": "polypeptide(L)", "pdbx_seq_one_letter_code_can": sequence,
                        "rcsb_sample_sequence_length": len(sequence)},
        "rcsb_polymer_entity": {"pdbx_description": "Protein"},
        "rcsb_polymer_entity_container_identifiers": {
            "entity_id": entity_id, "asym_ids": list(chains), "uniprot_ids": ["P12345"],
        },
        "rcsb_polymer_entity_annotation": [{"type": source}] if source else [],
        "polymer_entity_instances": [{
            "rcsb_polymer_entity_instance_container_identifiers": {
                "entity_id": entity_id, "asym_id": chain, "auth_asym_id": chain.lower(),
            },
            "rcsb_polymer_instance_info": {"modeled_residue_count": 28},
            "rcsb_polymer_instance_feature": [_feature()] if source == "PDBTM" else [],
        } for chain in chains],
    }


def _entry(entry_id="1ABC", *, models=20, entities=None):
    return {
        "rcsb_id": entry_id,
        "struct": {"title": "NMR membrane structure"},
        "rcsb_accession_info": {"deposit_date": "2026-01-01"},
        "exptl": [{"method": "SOLUTION NMR"}],
        "rcsb_entry_info": {"deposited_model_count": models, "polymer_entity_count_protein": 1},
        "polymer_entities": entities if entities is not None else [_entity()],
        "rcsb_primary_citation": {"pdbx_database_id_DOI": "10.1000/example", "pdbx_database_id_PubMed": 123},
    }


def _build(entries, **kwargs):
    return SolutionNMRMembraneBuilder().build_from_entries(entries, **kwargs)


@pytest.mark.parametrize("source", MEMBRANE_ANNOTATION_TYPES)
def test_each_membrane_source_qualifies_without_requiring_pdbtm(source):
    result = _build([_entry(entities=[_entity(source=source)])])
    assert len(result.entry_rows) == 1
    assert result.entry_rows[0]["has_pdbtm_annotation"] is (source == "PDBTM")
    assert result.entry_rows[0]["pdbtm_segment_status"] == ("complete" if source == "PDBTM" else "missing")
    assert result.entry_rows[0]["year"] == 2026


def test_multiple_chains_and_unannotated_partners_are_retained_separately():
    membrane = _entity(chains=("A", "B"))
    partner = _entity("2", source=None, chains=("C",), sequence="G" * 50)
    result = _build([_entry(entities=[membrane, partner])])
    row = result.entry_rows[0]
    assert (row["protein_entity_count"], row["protein_chain_count"]) == (2, 3)
    assert (row["membrane_entity_count"], row["membrane_chain_count"]) == (1, 2)
    assert row["pdbtm_segment_status"] == "complete"
    assert row["pdbtm_segment_chain_count"] == 2
    assert result.entity_rows[1]["is_membrane"] is False
    assert result.chain_rows[2]["is_membrane"] is False
    assert result.chain_rows[0]["auth_asym_id"] == "a"


@pytest.mark.parametrize("methods", [["SOLID-STATE NMR"], ["SOLUTION NMR", "SOLID-STATE NMR"], ["SOLUTION NMR", "X-RAY DIFFRACTION"], []])
def test_only_exact_solution_nmr_method_set_is_retained(methods):
    entry = _entry()
    entry["exptl"] = [{"method": method} for method in methods]
    result = _build([entry])
    assert not result.entry_rows
    assert result.excluded[0]["reason"] == "experimental method set is not exactly [SOLUTION NMR]"


def test_only_membrane_annotations_on_protein_entities_qualify():
    dna = _entity("2", source="PDBTM", chains=("B",))
    dna["entity_poly"]["type"] = "polydeoxyribonucleotide"
    result = _build([_entry(entities=[_entity(source=None), dna])])
    assert result.excluded[0]["reason"] == "no supported membrane annotation on a protein entity"
    assert not result.entity_rows


def test_polypeptide_d_is_allowed_and_missing_protein_chain_is_excluded():
    d_protein = _entity()
    d_protein["entity_poly"]["type"] = "polypeptide(D)"
    assert len(_build([_entry(entities=[d_protein])]).entry_rows) == 1
    for entities in ([], [_entity(chains=())]):
        result = _build([_entry(entities=entities)])
        assert result.excluded[0]["reason"] == "no polypeptide chain"


@pytest.mark.parametrize("models", [None, 0, -1, 1.5, True, "unknown"])
def test_unknown_model_count_is_excluded_by_default(models):
    entry = _entry(models=models)
    result = _build([entry], require_multiple_models=False)
    assert result.entry_rows[0]["deposited_model_count"] is None
    assert result.entry_rows[0]["model_count_status"] == "unknown"
    filtered = _build([entry], require_multiple_models=True)
    assert not filtered.entry_rows
    assert filtered.excluded[0]["reason"] == "deposited model count is missing or invalid"


def test_multiple_models_filter_is_on_by_default_and_counts_exclusions():
    entries = [_entry("2ABC", models=2), _entry("1ABC", models=1)]
    assert [row["entry_id"] for row in _build(entries, require_multiple_models=False).entry_rows] == ["1ABC", "2ABC"]
    result = _build(entries)
    assert [row["entry_id"] for row in result.entry_rows] == ["2ABC"]
    assert result.excluded == [{"entry_id": "1ABC", "year": 2026, "reason": "not more than one deposited model"}]
    assert result.filter_counts["retained_entries"] == 1
    assert result.filter_counts["excluded_entries"] == 1


def test_pdbtm_annotation_alone_does_not_imply_available_chain_segments():
    entity = _entity()
    entity["polymer_entity_instances"][0]["rcsb_polymer_instance_feature"] = [_feature(source="OPM")]
    result = _build([_entry(entities=[entity])])
    assert result.entry_rows[0]["has_pdbtm_annotation"] is True
    assert result.entry_rows[0]["pdbtm_segment_status"] == "missing"
    assert not result.segment_rows


@pytest.mark.parametrize("start,end", [(None, 20), (2, None), (0, 20), (2.5, 20), (20, 2), (2, 31)])
def test_invalid_pdbtm_boundaries_are_preserved_but_never_counted_valid(start, end):
    entity = _entity()
    entity["polymer_entity_instances"][0]["rcsb_polymer_instance_feature"] = [_feature(start, end)]
    result = _build([_entry(entities=[entity])])
    assert result.entry_rows[0]["pdbtm_segment_status"] == "missing"
    assert result.chain_rows[0]["pdbtm_invalid_segment_count"] == 1
    assert result.segment_rows[0]["valid"] is False
    assert result.segment_rows[0]["validation_reason"]


def test_partial_coverage_includes_missing_chains_and_invalid_feature_positions():
    entity = _entity(chains=("A", "B"))
    entity["polymer_entity_instances"].pop()
    result = _build([_entry(entities=[entity])])
    assert result.entry_rows[0]["pdbtm_segment_status"] == "partial"
    assert result.chain_rows[1]["instance_metadata_present"] is False
    entity = _entity()
    entity["polymer_entity_instances"][0]["rcsb_polymer_instance_feature"].append(_feature(None, None))
    result = _build([_entry(entities=[entity])])
    assert result.entry_rows[0]["pdbtm_segment_status"] == "partial"
    assert result.chain_rows[0]["pdbtm_valid_segment_count"] == 1


def test_missing_positions_date_and_sequence_are_not_converted_to_zero():
    entry = _entry()
    entry["rcsb_accession_info"] = None
    entity = entry["polymer_entities"][0]
    entity["entity_poly"]["pdbx_seq_one_letter_code_can"] = None
    entity["polymer_entity_instances"][0]["rcsb_polymer_instance_feature"][0]["feature_positions"] = None
    result = _build([entry])
    assert result.entry_rows[0]["year"] is None
    assert result.entry_rows[0]["sequence_multiset_sha256"] is None
    assert result.entity_rows[0]["sequence"] is None
    assert result.entity_rows[0]["sequence_length"] == 30
    assert result.segment_rows[0]["start_label_seq_id"] is None
    assert result.segment_rows[0]["raw_position_json"] == "null"


def test_duplicate_candidates_require_whole_protein_copy_multiset_and_citation_or_link():
    single = _entry("1ABC", models=1)
    same = _entry("2ABC", models=20)
    other_sequence = _entry("3ABC", models=20, entities=[_entity(sequence="G" * 30)])
    other_copies = _entry("4ABC", models=20, entities=[_entity(chains=("A", "B"))])
    no_citation = _entry("5ABC", models=20)
    no_citation["rcsb_primary_citation"] = None
    linked = _entry("6ABC", models=20)
    linked["rcsb_primary_citation"] = None
    linked["pdbx_database_related"] = [{"db_name": "PDB", "db_id": "1abc", "details": "average structure"}]
    result = _build([single, same, other_sequence, other_copies, no_citation, linked], require_multiple_models=False)
    assert len(result.entry_rows) == 6
    assert [(row["single_model_entry_id"], row["ensemble_entry_id"]) for row in result.duplicate_candidate_rows] == [("1ABC", "2ABC"), ("1ABC", "6ABC")]
    assert result.duplicate_candidate_rows[0]["shared_doi"] == "10.1000/example"
    assert result.duplicate_candidate_rows[1]["explicit_related_pdb_link"] is True


def test_missing_pubmed_sentinel_cannot_support_a_duplicate_candidate():
    entries = [_entry("1ABC", models=1), _entry("2ABC", models=20)]
    for entry in entries:
        entry["rcsb_primary_citation"] = {"pdbx_database_id_PubMed": -1}
    result = _build(entries)
    assert all(row["primary_citation_pubmed_id"] is None for row in result.entry_rows)
    assert not result.duplicate_candidate_rows


def test_results_match_csv_headers_and_input_is_not_modified():
    entries = [_entry("1ABC", models=1), _entry("2ABC", models=20)]
    original = deepcopy(entries)
    result = _build(entries, require_multiple_models=False)
    for rows, columns in ((result.entry_rows, ENTRY_COLUMNS), (result.entity_rows, ENTITY_COLUMNS),
                          (result.chain_rows, CHAIN_COLUMNS), (result.segment_rows, SEGMENT_COLUMNS),
                          (result.duplicate_candidate_rows, DUPLICATE_CANDIDATE_COLUMNS)):
        assert rows
        assert all(set(row) == set(columns) for row in rows)
    assert entries == original


def test_duplicate_entry_ids_raise_instead_of_double_counting():
    with pytest.raises(ValueError, match="duplicate PDB entry"):
        _build([_entry(), _entry()])


def test_empty_collection_has_explicit_zero_counts():
    result = _build([])
    assert result.filter_counts == {
        "input_entries": 0, "retained_entries": 0, "excluded_entries": 0,
        "after_polypeptide_filter": 0, "after_exact_method_filter": 0,
        "after_membrane_annotation_filter": 0, "after_model_filter": 0,
    }


def test_filter_counts_record_each_sequential_stage():
    no_protein = _entry("1ABC", entities=[])
    wrong_method = _entry("2ABC")
    wrong_method["exptl"] = [{"method": "SOLID-STATE NMR"}]
    no_annotation = _entry("3ABC", entities=[_entity(source=None)])
    single = _entry("4ABC", models=1)
    valid = _entry("5ABC")
    result = _build([no_protein, wrong_method, no_annotation, single, valid], require_multiple_models=True)
    assert result.filter_counts["input_entries"] == 5
    assert result.filter_counts["after_polypeptide_filter"] == 4
    assert result.filter_counts["after_exact_method_filter"] == 3
    assert result.filter_counts["after_membrane_annotation_filter"] == 2
    assert result.filter_counts["after_model_filter"] == 1
