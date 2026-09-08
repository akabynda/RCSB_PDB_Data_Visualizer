"""Shared upstream conformer selection for PDB and mmCIF coordinate routes."""

import gzip
import hashlib
import json
from pathlib import Path

import pytest
import requests

from src.dataset.config import DatasetBuildConfig
from src.dataset.downloads import (
    download_pdb_chain_subset_if_needed,
    download_pdb_if_needed,
)
from src.dataset.pdb_normalization import (
    ensure_normalized_pdb,
    iter_normalized_pdb_lines,
    normalize_pdb_altloc_text,
    pdb_atom_field_offset,
)
from src.dataset.structures import (
    _coerce_selected_structure_chain_ids_for_pdbio,
    _coerce_structure_chain_ids_for_pdbio,
    parse_mmcif_structure,
    parse_pdb_structure,
)


def _atom(
    serial: int,
    name: str = "CA",
    alt: str = "",
    occupancy: float = 1,
    resid: int = 56,
    icode: str = "",
    chain: str = "A",
    x: float = 0,
    resname: str = "ALA",
    record: str = "ATOM",
    element: str = "C",
) -> str:
    return (
        f"{record:6}{serial:5d} {name:^4}{alt:1}{resname:>3} {chain}{resid:4d}{icode:1}   "
        f"{x:8.3f}{0:8.3f}{0:8.3f}{occupancy:6.2f}{10:6.2f}          {element:>2}  \n"
    )


def _atoms(text: str) -> list[str]:
    return [line for line in text.splitlines() if line[:6] in {"ATOM  ", "HETATM"}]


def test_ca_selects_one_complete_conformer_and_retains_shared_atoms() -> None:
    original = (
        "REMARK   3   PROGRAM     : CYANA\nREMARK 210  SOFTWARE USED: Rosetta\n"
        + _atom(1, "N", element="N")
        + _atom(2, alt="C", occupancy=0.05, x=2)
        + _atom(3, alt="O", occupancy=0.95, x=3)
        + _atom(4, "C", alt="C", occupancy=0.99, x=4)
        + _atom(5, "C", alt="O", occupancy=0.01, x=5)
        + _atom(6, "CB", alt="C", occupancy=0.95, x=6)
        + "TER\nEND\n"
    )
    normalized = normalize_pdb_altloc_text(original)
    atoms = _atoms(normalized)
    assert [int(line[6:11]) for line in atoms] == [1, 3, 5]
    assert all(line[16] == " " for line in atoms)
    assert [float(line[30:38]) for line in atoms] == [0, 3, 5]
    assert normalized.startswith(original[: original.index("ATOM")])
    assert normalized.endswith("TER\nEND\n")
    assert normalize_pdb_altloc_text(normalized) == normalized


@pytest.mark.parametrize(
    "insertion_codes",
    [["", "A", "B", "C", "D", "E"], ["R", "B", "A", ""], ["1", "2", ""]],
)
def test_insertion_codes_preserve_distinct_residues_and_file_order(
    insertion_codes: list[str],
) -> None:
    original = "".join(
        _atom(2 * index + 1, alt="C", occupancy=0.05, icode=icode)
        + _atom(2 * index + 2, alt="O", occupancy=0.95, icode=icode, x=index)
        for index, icode in enumerate(insertion_codes)
    )
    atoms = _atoms(normalize_pdb_altloc_text(original))
    assert [line[26].strip() for line in atoms] == insertion_codes
    assert [int(line[22:26]) for line in atoms] == [56] * len(insertion_codes)


def test_signed_gapped_numbers_models_chains_and_ter_are_separate() -> None:
    original = "MODEL        1\n"
    for index, (resid, chain, icode) in enumerate(
        [(-2, "A", ""), (0, "A", ""), (56, "A", "1"), (561, "A", ""), (56, "B", "1")]
    ):
        original += _atom(
            2 * index + 1, alt="A", occupancy=0.3, resid=resid, chain=chain, icode=icode
        )
        original += _atom(
            2 * index + 2, alt="B", occupancy=0.7, resid=resid, chain=chain, icode=icode
        )
    original += (
        "TER\n"
        + _atom(11, alt="C", occupancy=0.8, resid=90)
        + _atom(12, alt="D", occupancy=0.2, resid=90)
    )
    original += (
        "ENDMDL\nMODEL        2\n"
        + _atom(1, alt="A", occupancy=0.9, resid=-2)
        + _atom(2, alt="B", occupancy=0.1, resid=-2)
        + "ENDMDL\nEND\n"
    )
    normalized = normalize_pdb_altloc_text(original)
    assert [int(line[6:11]) for line in _atoms(normalized)] == [2, 4, 6, 8, 10, 11, 1]
    assert normalized.count("MODEL ") == 2
    assert normalized.count("ENDMDL") == 2


@pytest.mark.parametrize("altloc", ["", "A"])
def test_repeated_author_identifier_across_ter_segments_is_rejected(
    altloc: str,
) -> None:
    original = (
        _atom(1, alt=altloc, icode="A") + "TER\n" + _atom(2, alt=altloc, icode="A")
    )
    with pytest.raises(ValueError, match="Repeated Cα residue identifier across TER"):
        normalize_pdb_altloc_text(original)


def test_calcium_and_zero_occupancy_do_not_trigger_ter_collision() -> None:
    original = _atom(1) + "TER\n"
    original += _atom(2, resname="CA", record="HETATM", element="CA")
    original += _atom(3, occupancy=0)
    assert normalize_pdb_altloc_text(original) == original


def test_number_field_padding_does_not_create_another_residue() -> None:
    minor = _atom(1, alt="C", occupancy=0.05)
    original = minor[:22] + "0056" + minor[26:] + _atom(2, alt="O", occupancy=0.95)
    assert [
        int(line[6:11]) for line in _atoms(normalize_pdb_altloc_text(original))
    ] == [2]


def test_long_component_names_retain_numeric_insertion_boundary() -> None:
    original = (
        _atom(
            1,
            resname="LONG",
            record="HETATM",
            resid=56,
            icode="1",
            alt="C",
            occupancy=0.05,
        )
        + _atom(
            2,
            resname="LONG",
            record="HETATM",
            resid=56,
            icode="1",
            alt="O",
            occupancy=0.95,
        )
        + _atom(3, resname="LONG", record="HETATM", resid=561, alt="C", occupancy=0.95)
        + _atom(4, resname="LONG", record="HETATM", resid=561, alt="O", occupancy=0.05)
    )
    atoms = _atoms(normalize_pdb_altloc_text(original))
    assert [int(line[6:11]) for line in atoms] == [2, 3]
    assert all(pdb_atom_field_offset(line) == 1 for line in atoms)
    assert [line[23:28] for line in atoms] == ["  561", " 561 "]


@pytest.mark.parametrize(
    "labels,expected",
    [(["B", "A"], "A"), (["B", "1"], "1"), (["Z", "C"], "C"), (["A", ""], "")],
)
def test_equal_occupancy_uses_deterministic_label_preference(
    labels: list[str], expected: str
) -> None:
    original = "".join(
        _atom(index + 1, alt=label, occupancy=0.5, x=index)
        for index, label in enumerate(labels)
    )
    atoms = _atoms(normalize_pdb_altloc_text(original))
    assert len(atoms) == 1
    assert float(atoms[0][30:38]) == labels.index(expected)


def test_shared_ca_uses_one_sidechain_conformer_and_ignores_zero_occupancy() -> None:
    original = (
        _atom(1)
        + _atom(2, "CB", alt="A", occupancy=0.1)
        + _atom(3, "CB", alt="B", occupancy=0.9)
    )
    original += _atom(4, "CG", alt="B", occupancy=0)
    original += _atom(5, "CD", alt="A", occupancy=0.1)
    assert [
        int(line[6:11]) for line in _atoms(normalize_pdb_altloc_text(original))
    ] == [1, 3]


def test_alternate_residue_names_are_one_conformer_not_two_residues() -> None:
    original = _atom(1, resname="SER", alt="A", occupancy=0.2) + _atom(
        2, resname="CYS", alt="B", occupancy=0.8
    )
    original += _atom(3, "OG", resname="SER", alt="A", occupancy=0.2)
    original += _atom(4, "SG", resname="CYS", alt="B", occupancy=0.8)
    atoms = _atoms(normalize_pdb_altloc_text(original))
    assert [int(line[6:11]) for line in atoms] == [2, 4]
    assert {line[17:20] for line in atoms} == {"CYS"}


def test_hetero_component_reusing_author_id_is_not_a_polymer_altloc() -> None:
    original = _atom(1, alt="C", occupancy=0.05) + _atom(2, alt="O", occupancy=0.95)
    original += _atom(3, resname="CA", record="HETATM", element="CA")
    atoms = _atoms(normalize_pdb_altloc_text(original))
    assert [int(line[6:11]) for line in atoms] == [2, 3]


def test_modified_amino_acid_can_alternate_with_standard_amino_acid() -> None:
    original = _atom(1, resname="MSE", record="HETATM", alt="A", occupancy=0.2)
    original += _atom(2, resname="MET", alt="B", occupancy=0.8)
    original += _atom(3, "SE", resname="MSE", record="HETATM", alt="A", occupancy=0.2)
    original += _atom(4, "SD", resname="MET", alt="B", occupancy=0.8)
    assert [
        int(line[6:11]) for line in _atoms(normalize_pdb_altloc_text(original))
    ] == [2, 4]


def test_companion_records_and_conect_do_not_reference_removed_atoms() -> None:
    minor = _atom(1, alt="C", occupancy=0.05)
    major = _atom(2, alt="O", occupancy=0.95)
    shared = _atom(3, "N", element="N")
    original = minor + "ANISOU" + minor[6:] + major + "ANISOU" + major[6:] + shared
    original += "CONECT    1    3\nCONECT    3    1    2\nEND\n"
    normalized = normalize_pdb_altloc_text(original)
    companions = [line for line in normalized.splitlines() if line.startswith("ANISOU")]
    assert len(companions) == 1
    assert int(companions[0][6:11]) == 2 and companions[0][16] == " "
    assert "CONECT    1" not in normalized
    assert "CONECT    3    2\n" in normalized


def test_direct_parser_and_atomic_normalization_share_selection(tmp_path: Path) -> None:
    path = tmp_path / "input.pdb"
    original = (
        _atom(1, alt="C", occupancy=0.05, x=1)
        + _atom(2, alt="O", occupancy=0.95, x=2)
        + "END\n"
    )
    path.write_text(original)
    structure = parse_pdb_structure("TEST", path)
    ca = next(structure.get_atoms())
    assert ca.get_altloc() == " " and ca.coord[0] == 2
    assert path.read_text() == original
    expected = "".join(iter_normalized_pdb_lines(path))
    ensure_normalized_pdb(path)
    assert path.read_text() == expected
    mtime = path.stat().st_mtime_ns
    ensure_normalized_pdb(path)
    assert path.stat().st_mtime_ns == mtime
    assert sorted(item.name for item in tmp_path.iterdir()) == ["input.pdb"]


class _Response:
    def __init__(self, status: int, body: bytes = b""):
        self.status_code = status
        self.body = body
        self.headers = {"ETag": '"test"'}

    def iter_content(self, chunk_size: int):
        yield self.body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))


class _Session:
    def __init__(self, responses: list[_Response]):
        self.responses = responses

    def get(self, url: str, **kwargs):
        return self.responses.pop(0)


def _mmcif(chain: str) -> bytes:
    columns = "group_PDB id type_symbol label_atom_id label_alt_id label_comp_id label_asym_id label_entity_id label_seq_id pdbx_PDB_ins_code Cartn_x Cartn_y Cartn_z occupancy B_iso_or_equiv pdbx_formal_charge auth_seq_id auth_comp_id auth_asym_id auth_atom_id pdbx_PDB_model_num".split()
    return (
        "data_TEST\n#\nloop_\n_pdbx_nmr_software.name\n_pdbx_nmr_software.version\nCYANA 3.98\n#\nloop_\n"
        + "".join(f"_atom_site.{column}\n" for column in columns)
        + f"ATOM 1 C CA C ALA {chain} 1 1 ? 1 0 0 0.05 10 ? 56 ALA {chain} CA 1\n"
        + f"ATOM 2 C CA O ALA {chain} 1 1 ? 2 0 0 0.95 10 ? 56 ALA {chain} CA 1\n#\n"
    ).encode()


@pytest.mark.parametrize("route", ["pdb", "cif", "subset"])
def test_download_routes_normalize_before_cache_hash_and_preserve_program(
    tmp_path: Path, route: str
) -> None:
    config = DatasetBuildConfig(retries=1, pdb_cache_validation_hours=24)
    if route == "pdb":
        body = (
            "REMARK 210  SOFTWARE USED: CYANA 3.98\n"
            + _atom(1, alt="C", occupancy=0.05, x=1)
            + _atom(2, alt="O", occupancy=0.95, x=2)
            + "END\n"
        )
        session = _Session([_Response(200, gzip.compress(body.encode()))])
    elif route == "cif":
        session = _Session(
            [_Response(404) for _ in range(4)] + [_Response(200, _mmcif("A"))]
        )
    else:
        session = _Session([_Response(200, gzip.compress(_mmcif("LONG")))])
    if route == "subset":
        path, chain_map = download_pdb_chain_subset_if_needed(
            session, config, tmp_path, "TEST", ["LONG"]
        )
        assert chain_map == {"LONG": "L"}
    else:
        path = download_pdb_if_needed(session, config, tmp_path, "TEST")
    atoms = _atoms(path.read_text())
    assert len(atoms) == 1 and atoms[0][16] == " "
    assert float(atoms[0][30:38]) == 2
    assert "CYANA 3.98" in path.read_text()
    metadata = json.loads(path.with_suffix(".pdb.cache.json").read_text())
    assert metadata["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert metadata["size_bytes"] == path.stat().st_size
    assert metadata["mtime_ns"] == path.stat().st_mtime_ns
    assert not session.responses
    if route == "subset":
        assert (
            download_pdb_chain_subset_if_needed(
                session, config, tmp_path, "TEST", ["LONG"]
            )[0]
            == path
        )
    else:
        assert download_pdb_if_needed(session, config, tmp_path, "TEST") == path


def test_cache_hit_normalizes_and_updates_metadata_before_return(
    tmp_path: Path,
) -> None:
    body = (
        _atom(1, alt="C", occupancy=0.05) + _atom(2, alt="O", occupancy=0.95) + "END\n"
    )
    config = DatasetBuildConfig(retries=1, pdb_cache_validation_hours=24)
    session = _Session([_Response(200, gzip.compress(body.encode()))])
    path = download_pdb_if_needed(session, config, tmp_path, "TEST")
    path.write_text(body)
    metadata_path = path.with_suffix(".pdb.cache.json")
    metadata = json.loads(metadata_path.read_text())
    metadata.update(
        size_bytes=path.stat().st_size,
        mtime_ns=path.stat().st_mtime_ns,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    metadata_path.write_text(json.dumps(metadata))
    download_pdb_if_needed(session, config, tmp_path, "TEST")
    assert len(_atoms(path.read_text())) == 1
    assert (
        json.loads(metadata_path.read_text())["sha256"]
        == hashlib.sha256(path.read_bytes()).hexdigest()
    )


@pytest.mark.parametrize("author_number", [-1000, 10000])
@pytest.mark.parametrize("selected_only", [False, True])
def test_mmcif_conversion_rejects_overflowing_author_numbers(
    tmp_path: Path, author_number: int, selected_only: bool
) -> None:
    path = tmp_path / "TEST.cif"
    path.write_bytes(_mmcif("LONG").replace(b"56 ALA", f"{author_number} ALA".encode()))
    structure = parse_mmcif_structure("TEST", path)
    with pytest.raises(RuntimeError, match="Unsupported legacy PDB residue numbering"):
        if selected_only:
            _coerce_selected_structure_chain_ids_for_pdbio(structure, {"LONG"})
        else:
            _coerce_structure_chain_ids_for_pdbio(structure)


def test_unselected_chain_numbering_does_not_prevent_subset_conversion(
    tmp_path: Path,
) -> None:
    path = tmp_path / "TEST.cif"
    path.write_bytes(_mmcif("LONG").replace(b"56 ALA", b"-1000 ALA"))
    structure = parse_mmcif_structure("TEST", path)
    assert _coerce_selected_structure_chain_ids_for_pdbio(structure, {"OTHER"}) == (
        {},
        set(),
    )
