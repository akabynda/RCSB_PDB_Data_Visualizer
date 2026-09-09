"""Apply the same modeled-chain HETATM exclusion to every NMR monomer cohort."""

from copy import deepcopy
from unittest.mock import Mock, patch

import pytest

from src.dataset.client import RCSBClient
from src.dataset.config import DatasetBuildConfig
from src.dataset.coordinates import parse_models_ca_data
from src.dataset.records import ResidueId


def _atom(
    serial,
    number,
    *,
    record="ATOM",
    name="ALA",
    chain="A",
    insertion="",
    occupancy=1.0,
    altloc="",
    element="C",
):
    return (
        f"{record:<6}{serial:5d}  CA {altloc or ' '}{name:>3} {chain}{number:4d}{insertion or ' '}   "
        f"{float(serial):8.3f}{0.0:8.3f}{0.0:8.3f}{occupancy:6.2f}{20.0:6.2f}          {element:>2}\n"
    )


def _pdb(tmp_path, tails):
    path = tmp_path / "TEST.pdb"
    lines = ["SEQRES   1 A    4  ALA ALA ALA HYP\n", "SEQRES   1 B    1  HYP\n"]
    for index, tail in enumerate(tails, 1):
        lines += [f"MODEL     {index:4d}\n", _atom(1, 1), _atom(2, 2), tail, "ENDMDL\n"]
    path.write_text("".join(lines))
    return path


def _entry():
    return {
        "rcsb_id": "TEST",
        "rcsb_entry_info": {"deposited_model_count": 2},
        "rcsb_accession_info": {"deposit_date": "2000-01-01"},
        "polymer_entities": [
            {
                "entity_poly": {
                    "type": "polypeptide(L)",
                    "rcsb_entity_polymer_type": "Protein",
                    "pdbx_strand_id": "A",
                },
                "polymer_entity_instances": [{"rcsb_id": "TEST.A"}],
            }
        ],
        "pdbx_nmr_exptl": [{"type": "NOESY"}],
        "pdbx_vrpt_summary_geometry": [
            {
                "clashscore": 1,
                "percent_ramachandran_outliers": 0,
                "percent_rotamer_outliers": 0,
            }
        ],
    }


def _client(tmp_path, path):
    client = RCSBClient(DatasetBuildConfig(), solution_nmr_monomer_cache_dir=tmp_path)
    client._download_solution_nmr_monomer_pdb_if_needed = Mock(return_value=path)
    return client


@pytest.mark.parametrize("model_index", [0, 1])
@pytest.mark.parametrize("insertion", ["", "A", "1"])
def test_rejects_hetatm_at_modeled_tail_in_any_model(tmp_path, model_index, insertion):
    tails = [_atom(3, 56, insertion=insertion)] * 2
    tails[model_index] = _atom(3, 56, insertion=insertion, record="HETATM", name="HYP")
    path = _pdb(tmp_path, tails)
    client = _client(tmp_path, path)
    models, _, hetatm = parse_models_ca_data(path, "A")
    assert list(map(len, models)) == [3, 3]
    assert hetatm[model_index] == {ResidueId(56, insertion)}
    assert hetatm[1 - model_index] == set()
    with patch("src.dataset.client.nmr._record_filtered_structure") as filtered:
        assert client._extract_solution_nmr_monomer_context(_entry()) is None
    filtered.assert_called_once_with(
        "TEST",
        "modeled protein chain contains HETATM CA residues "
        f"(chain A, model {model_index + 1}; residue IDs: {ResidueId(56, insertion)})",
        year=2000,
    )


@pytest.mark.parametrize(
    "extra",
    [
        _atom(4, 99, record="HETATM", name="CA", element="CA"),
        _atom(4, 99, record="HETATM", name="PLM"),
        _atom(4, 99, record="HETATM", name="HYP", occupancy=0),
        _atom(4, 99, record="HETATM", name="HYP", chain="B"),
        "",  # HYP in SEQRES without coordinates is not a modeled position.
    ],
)
def test_nonmodeled_or_nonpolymer_hetero_atoms_do_not_exclude_monomer(tmp_path, extra):
    path = _pdb(tmp_path, [_atom(3, 56) + extra] * 2)
    client = _client(tmp_path, path)
    assert client._extract_solution_nmr_monomer_context(_entry()) is not None


@pytest.mark.parametrize("hetero_occupancy,rejected", [(0.1, False), (0.9, True)])
def test_filter_uses_selected_conformer(tmp_path, hetero_occupancy, rejected):
    tail = _atom(3, 56, altloc="A", occupancy=1 - hetero_occupancy)
    tail += _atom(
        4, 56, record="HETATM", name="HYP", altloc="B", occupancy=hetero_occupancy
    )
    path = _pdb(tmp_path, [tail] * 2)
    result = _client(tmp_path, path)._extract_solution_nmr_monomer_context(_entry())
    assert (result is None) == rejected


def test_retained_hetatm_duplicate_is_rejected_even_when_atom_wins(tmp_path):
    tail = _atom(3, 56) + _atom(4, 56, record="HETATM", name="HYP")
    path = _pdb(tmp_path, [tail] * 2)
    assert (
        _client(tmp_path, path)._extract_solution_nmr_monomer_context(_entry()) is None
    )


@pytest.mark.parametrize(
    "method",
    [
        "fetch_solution_nmr_monomer_experiment_records_for_ids",
        "fetch_solution_nmr_monomer_quality_records_for_ids",
        "fetch_solution_nmr_monomer_modeled_first_model_seed_records_for_ids",
        "fetch_solution_nmr_monomer_xray_homolog_seed_records_for_ids",
        "fetch_solution_nmr_monomer_stride_modeled_first_model_records_for_ids",
    ],
)
def test_all_direct_monomer_collectors_reject_before_stride(tmp_path, method):
    path = _pdb(tmp_path, [_atom(3, 56, record="HETATM", name="HYP")] * 2)
    client = _client(tmp_path, path)
    client._post_json = Mock(return_value={"data": {"entries": [deepcopy(_entry())]}})
    kwargs = {}
    if "stride" in method:
        kwargs = {
            "stride_executable": "unused",
            "pdb_cache_dir": tmp_path,
            "stride_cache_dir": tmp_path / "stride",
        }
    with (
        patch("src.dataset.client.nmr._record_filtered_structure") as filtered,
        patch("src.dataset.stride._run_stride_for_model_text") as run_stride,
    ):
        assert getattr(client, method)(["TEST"], **kwargs) == []
    assert "modeled protein chain contains HETATM" in filtered.call_args.args[1]
    run_stride.assert_not_called()
