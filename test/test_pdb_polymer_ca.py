"""Keep carbon atoms named CA in ligands out of protein coordinate sets."""

import json
from unittest.mock import Mock, patch

from Bio.PDB import PDBIO

from src.dataset.ca_cache import _parse_first_model_ca_data_by_chain
from src.dataset.coordinates import (
    parse_first_model_ca_residues,
    parse_models_ca_coords_with_stats,
)
from src.dataset.pdb_normalization import (
    POLYPEPTIDE_REMARK_PREFIX,
    load_pdb_polymer_metadata,
    normalize_pdb_altloc_text,
)
from src.dataset.records import ResidueId
from src.dataset.config import DatasetBuildConfig
from src.dataset.downloads import (
    _download_pdb_chain_subset_if_needed_locked,
    download_pdb_if_needed,
)
from src.dataset.structures import (
    _coerce_structure_chain_ids_for_pdbio,
    _prepend_mmcif_polymer_remarks,
    parse_mmcif_structure,
)


def _ca(
    serial, name, number, *, record="HETATM", insertion="", altloc="", occupancy=1.0
):
    return (
        f"{record:<6}{serial:5d}  CA {altloc or ' '}{name:>3} A{number:4d}{insertion or ' '}   "
        f"{float(serial):8.3f}{0.0:8.3f}{0.0:8.3f}{occupancy:6.2f}{0.0:6.2f}           C  \n"
    )


def _modres(name, number, insertion="", chain="A"):
    line = list(" " * 80)
    line[:6] = "MODRES"
    line[12:15] = name
    line[16] = chain
    line[18:22] = f"{number:4d}"
    line[22] = insertion or " "
    line[24:27] = "ALA"
    return "".join(line) + "\n"


def _all_keys(path):
    records = parse_first_model_ca_residues(path, "A")
    models, counts = parse_models_ca_coords_with_stats(path, "A")
    cached = _parse_first_model_ca_data_by_chain(path)["A"]
    expected = [record.key for record in records]
    assert list(models[0]) == expected
    assert list(counts[0]) == expected
    assert [record.key for record in cached[0]] == expected
    assert list(cached[1]) == expected
    return records


def test_ligand_ca_excluded_but_modified_residue_evidence_is_preserved(tmp_path):
    path = tmp_path / "polymer.pdb"
    path.write_text(
        "SEQRES   1 A    4  GLY HYP DAL ZZZ\n"
        + _modres("Z9Z", 4, "A")
        + _ca(1, "GLY", 1, record="ATOM")
        + _ca(2, "HYP", 2)
        + _ca(3, "DAL", 3)
        + _ca(4, "Z9Z", 4, insertion="A")
        + _ca(5, "Z9Z", 4, insertion="B")
        + _ca(6, "PLM", 88)
        + _ca(7, "MTX", 89)
        + "END\n"
    )
    records = _all_keys(path)
    assert [record.key for record in records] == [1, 2, 3, ResidueId(4, "A")]
    assert [record.has_hetatm_ca for record in records] == [False, True, True, True]


def test_unknown_legacy_het_ca_requires_polymer_evidence(tmp_path):
    path = tmp_path / "unknown.pdb"
    path.write_text(_ca(1, "GLY", 1, record="ATOM") + _ca(2, "Z9Z", 2))
    assert [r.key for r in _all_keys(path)] == [1]
    path.write_text("SEQRES   1 A    2  GLY Z9Z\n" + path.read_text())
    assert [r.key for r in _all_keys(path)] == [1, 2]


def test_native_seqres_overrides_amino_acid_dictionary_for_free_ligand(tmp_path):
    # Real 2LYB: 8SP A300 is a non-polymer despite Bio.is_aa("8SP") being true.
    path = tmp_path / "native_dictionary_ligand.pdb"
    path.write_text(
        "SEQRES   1 A    2  GLY HYP\n"
        + _ca(1, "GLY", 1, record="ATOM")
        + _ca(2, "HYP", 2)
        + _ca(3, "8SP", 300)
    )
    records = _all_keys(path)
    assert [record.key for record in records] == [1, 2]
    assert [record.has_hetatm_ca for record in records] == [False, True]


def test_dictionary_ligand_does_not_replace_polymer_altloc(tmp_path):
    path = tmp_path / "independent_chemical_residues.pdb"
    path.write_text(
        "SEQRES   1 A    1  GLY\n"
        + _ca(1, "GLY", 1, record="ATOM", altloc="A", occupancy=0.2)
        + _ca(2, "8SP", 1, altloc="B", occupancy=0.8)
    )
    records = _all_keys(path)
    assert len(records) == 1
    assert records[0].is_standard_atom
    assert not records[0].has_hetatm_ca


def test_ligand_ca_reusing_polymer_number_after_ter_is_not_duplicate():
    text = _ca(1, "GLY", 1, record="ATOM") + "TER\n" + _ca(2, "PLM", 1)
    assert normalize_pdb_altloc_text(text) == text


def test_seqres_recognizes_rare_hetero_alternate_of_standard_residue(tmp_path):
    path = tmp_path / "rare_altloc.pdb"
    path.write_text(
        "SEQRES   1 A    1  Z9Z\n"
        + _ca(1, "ALA", 1, record="ATOM", altloc="A", occupancy=0.2)
        + _ca(2, "Z9Z", 1, altloc="B", occupancy=0.8)
    )
    records = _all_keys(path)
    assert len(records) == 1
    assert records[0].identity == "HET:Z9Z"
    assert not records[0].is_standard_atom


def _mmcif(entity_type="polypeptide(D)"):
    return f"""data_polymer
loop_
_entity_poly.entity_id
_entity_poly.type
1 '{entity_type}'
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.auth_seq_id
_atom_site.auth_asym_id
_atom_site.pdbx_PDB_model_num
ATOM 1 C CA . GLY A 1 1 ? 0 0 0 1 0 1 LONG 1
HETATM 2 C CA . Z9Z A 1 . A 1 0 0 1 0 2 LONG 1
HETATM 3 C CA . DAL A 1 3 ? 2 0 0 1 0 3 LONG 1
HETATM 4 C CA . MSE B 2 . ? 3 0 0 1 0 4 LONG 1
HETATM 5 C CA . PLM B 2 . ? 4 0 0 1 0 5 LONG 1
#
"""


def test_conversion_preserves_exact_membership_and_author_ids(tmp_path):
    cif_path = tmp_path / "polymer.cif"
    pdb_path = tmp_path / "polymer.pdb"
    cif_path.write_text(_mmcif())
    structure = parse_mmcif_structure("test", cif_path)
    chain_map = _coerce_structure_chain_ids_for_pdbio(structure)
    assert chain_map == {"LONG": "L"}
    writer = PDBIO()
    writer.set_structure(structure)
    writer.save(str(pdb_path))
    coordinate_bytes = pdb_path.read_bytes()
    _prepend_mmcif_polymer_remarks(pdb_path, cif_path, chain_map)
    assert pdb_path.read_bytes().endswith(coordinate_bytes)
    assert (POLYPEPTIDE_REMARK_PREFIX + "COMPLETE") in pdb_path.read_text()
    metadata = load_pdb_polymer_metadata(pdb_path)
    assert metadata.includes("L", 2, "A", "Z9Z")  # Missing label_seq_id is allowed.
    assert metadata.includes("L", 3, "", "DAL")
    assert not metadata.includes("L", 4, "", "MSE")  # Known amino acid, free ligand.
    assert not metadata.includes("L", 5, "", "PLM")
    assert [r.key for r in parse_first_model_ca_residues(pdb_path, "L")] == [
        1,
        ResidueId(2, "A"),
        3,
    ]


def test_conversion_complete_empty_membership_disables_amino_acid_fallback(tmp_path):
    cif_path = tmp_path / "nonprotein.cif"
    pdb_path = tmp_path / "nonprotein.pdb"
    cif_path.write_text(_mmcif("polydeoxyribonucleotide"))
    pdb_path.write_text(_ca(1, "HYP", 1))
    _prepend_mmcif_polymer_remarks(pdb_path, cif_path, {"LONG": "A"})
    assert load_pdb_polymer_metadata(pdb_path).exact_membership_complete
    assert parse_first_model_ca_residues(pdb_path, "A") == []


def test_conversion_subset_only_records_selected_author_chains(tmp_path):
    cif_path = tmp_path / "subset.cif"
    pdb_path = tmp_path / "subset.pdb"
    cif_path.write_text(_mmcif())
    pdb_path.write_text(_ca(1, "DAL", 3))
    _prepend_mmcif_polymer_remarks(pdb_path, cif_path, {"LONG": "A"}, {"OTHER"})
    assert load_pdb_polymer_metadata(pdb_path).exact_keys == set()
    assert parse_first_model_ca_residues(pdb_path, "A") == []


def test_full_download_fallback_publishes_polymer_membership(tmp_path):
    missing = Mock(status_code=404, headers={})
    cif_response = Mock(status_code=200, headers={})
    cif_response.iter_content.return_value = iter([_mmcif().encode()])
    session = Mock()
    session.get.side_effect = [missing, cif_response]
    with patch(
        "src.dataset.downloads._pdb_download_sources",
        return_value=[("https://example.test/test.pdb", False)],
    ):
        pdb_path = download_pdb_if_needed(
            session, DatasetBuildConfig(retries=1), tmp_path, "1ABC"
        )
    assert load_pdb_polymer_metadata(pdb_path).exact_membership_complete
    assert [r.key for r in parse_first_model_ca_residues(pdb_path, "L")] == [
        1,
        ResidueId(2, "A"),
        3,
    ]


def test_chain_subset_conversion_publishes_polymer_membership(tmp_path):
    cif_path = tmp_path / "1ABC.cif"
    cif_path.write_text(_mmcif())
    cif_path.with_suffix(".cif.cache.json").write_text(
        json.dumps(
            {
                "sha256": "source-hash",
            }
        )
    )
    pdb_path, chain_map = _download_pdb_chain_subset_if_needed_locked(
        tmp_path, "1ABC", {"LONG"}, "1ABC.subset", cif_path
    )
    assert chain_map == {"LONG": "L"}
    assert load_pdb_polymer_metadata(pdb_path).exact_membership_complete
    assert [r.key for r in parse_first_model_ca_residues(pdb_path, "L")] == [
        1,
        ResidueId(2, "A"),
        3,
    ]
