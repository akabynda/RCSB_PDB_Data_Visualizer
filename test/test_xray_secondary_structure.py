"""Check secondary-structure masks without changing homolog sequence matching."""

import csv
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest

from src.dataset.builders.rmsd_outputs import (
    build_solution_nmr_monomer_xray_rmsd_outputs_to_csv,
)
from src.dataset.builders.xray_rmsd import (
    SolutionNMRMonomerXrayRmsdBuilder,
    _build_xray_rmsd_extremes_record_from_candidates,
)
from src.dataset.client import RCSBClient
from src.dataset.config import DatasetBuildConfig
from src.dataset.io.homologs import write_solution_nmr_monomer_xray_homolog_csv
from src.dataset.io.rmsd import (
    read_solution_nmr_monomer_xray_rmsd_csv,
    read_solution_nmr_monomer_xray_rmsd_extremes_csv,
    write_solution_nmr_monomer_xray_rmsd_csv,
    write_solution_nmr_monomer_xray_rmsd_extremes_csv,
)
from src.dataset.records import (
    CAResidueRecord,
    ResidueId,
    SolutionNMRMonomerXrayHomologRecord,
    SolutionNMRMonomerXrayRmsdRecord,
    XrayPolymerEntityCandidateRecord,
)
from src.dataset.reporting import _set_active_dataset_filtered_csvs


def _coordinates():
    """Provide a full 11-residue core, including 5 and 5A as separate atoms."""
    ids = [ResidueId(i) for i in range(1, 6)]
    ids += [ResidueId(5, "A")]
    ids += [ResidueId(i) for i in range(6, 11)]
    residues = tuple(
        CAResidueRecord(r.seq_id, aa, True, insertion_code=r.insertion_code)
        for r, aa in zip(ids, "ACDEFGHIKLM")
    )
    coords = {
        r: np.array([float(i), float(i * i % 7), float(i % 3)])
        for i, r in enumerate(ids)
    }
    selected = {ResidueId(1), ResidueId(2), ResidueId(3), ResidueId(5, "A")}
    return residues, coords, selected


def _score(tmp_path, residues, nmr_coords, xray_coords, selected):
    with patch(
        "src.dataset.builders.xray_rmsd.load_cached_first_model_ca_data",
        return_value=(residues, xray_coords),
    ):
        return SolutionNMRMonomerXrayRmsdBuilder._compute_ca_rmsd_to_xray(
            nmr_pdb_path=tmp_path / "nmr.pdb",
            nmr_chain_id="A",
            nmr_core_start_seq_id=ResidueId(1),
            nmr_core_end_seq_id=ResidueId(10),
            xray_pdb_path=tmp_path / "xray.pdb",
            xray_chain_id="A",
            sequence_identity_percent=100,
            prepared_nmr_core=(_coordinates()[0], nmr_coords),
            allowed_nmr_residue_ids=selected,
        )


def test_secondary_fit_excludes_loops_and_distinguishes_insertion_codes(tmp_path):
    residues, coords, selected = _coordinates()
    xray = {
        key: point + [10.0, -4.0, 2.0] + ([0, 0, 0] if key in selected else [0, 8, 0])
        for key, point in coords.items()
    }
    full = _score(tmp_path, residues, coords, xray, None)
    secondary = _score(tmp_path, residues, coords, xray, selected)
    assert full is not None and full[1] > 1.0
    assert secondary is not None
    assert secondary[0] == 4
    assert secondary[1] == pytest.approx(0, abs=1e-12)
    # Full matched range metadata stays intact, including excluded terminal loops.
    assert secondary[4:] == (ResidueId(1), ResidueId(10))


def test_secondary_selection_does_not_relax_full_core_sequence_match(tmp_path):
    residues, coords, selected = _coordinates()
    changed = list(residues)
    changed[4] = replace(changed[4], identity="W")  # A loop excluded from RMSD.
    assert _score(tmp_path, tuple(changed), coords, coords, selected) is None


@pytest.mark.parametrize(
    "selected", [set(), {ResidueId(1)}, {ResidueId(1), ResidueId(2)}]
)
def test_secondary_fit_requires_three_matching_atoms(tmp_path, selected):
    residues, coords, _ = _coordinates()
    assert _score(tmp_path, residues, coords, coords, selected) is None


def _homolog():
    return SolutionNMRMonomerXrayHomologRecord(
        entry_id="1NMR",
        year=2001,
        sequence_identity_percent=100,
        nmr_core_start_seq_id=1,
        nmr_core_end_seq_id=10,
        nmr_query_sequence_length=11,
        xray_homolog_entry_ids=("1AAA", "2BBB"),
        xray_homolog_entity_ids=("1AAA_1", "2BBB_1"),
        has_xray_homolog=True,
    )


def test_secondary_recalculates_all_candidates_and_can_change_best_match(tmp_path):
    residues, coords, selected = _coordinates()
    first = {key: point.copy() for key, point in coords.items()}
    first[ResidueId(1)] += [0, 2, 0]
    second = {
        key: point + ([0, 0, 0] if key in selected else [0, 8, 0])
        for key, point in coords.items()
    }
    homolog = _homolog()
    candidates = tuple(
        XrayPolymerEntityCandidateRecord(f"{entry}_1", entry, ("A",), float(i))
        for i, entry in enumerate(("1AAA", "2BBB"), 1)
    )
    config = DatasetBuildConfig()
    builders = [
        SolutionNMRMonomerXrayRmsdBuilder(
            RCSBClient(config),
            config,
            tmp_path,
            1,
            [homolog],
            residue_selection=selection,
        )
        for selection in ("core", "secondary-structure")
    ]
    with (
        patch.object(
            SolutionNMRMonomerXrayRmsdBuilder,
            "_download_pdb_if_needed",
            return_value=tmp_path / "nmr.pdb",
        ),
        patch.object(
            SolutionNMRMonomerXrayRmsdBuilder,
            "_prepare_nmr_core_data",
            return_value=(residues, coords),
        ),
        patch(
            "src.dataset.builders.xray_rmsd.load_cached_chain_id_map", return_value={}
        ),
        patch(
            "src.dataset.builders.xray_rmsd.download_pdb_chain_subset_if_needed",
            side_effect=lambda **kw: (tmp_path / (kw["entry_id"] + ".pdb"), {}),
        ),
        patch(
            "src.dataset.builders.xray_rmsd.load_cached_first_model_ca_data",
            side_effect=lambda **kw: (
                residues,
                first if kw["pdb_path"].stem == "1AAA" else second,
            ),
        ),
        patch(
            "src.dataset.builders.xray_rmsd.compute_stride_secondary_residue_ids_in_first_model",
            return_value=selected,
        ) as stride_mask,
    ):
        core = builders[0]._compute_extremes_record(homolog, "A", candidates)
        secondary = builders[1]._compute_extremes_record(homolog, "A", candidates)
    assert core is not None and secondary is not None
    assert core.best_xray_entry_id == "1AAA"
    assert secondary.best_xray_entry_id == "2BBB"
    assert secondary.successful_xray_homolog_count == 2
    assert secondary.best_n_common_ca == 4
    assert secondary.residue_selection == "secondary-structure"
    stride_mask.assert_called_once()
    assert stride_mask.call_args.kwargs["modeled_auth_seq_ids"] == [
        r.key for r in residues
    ]


def _record(selection="core"):
    return SolutionNMRMonomerXrayRmsdRecord(
        entry_id="1NMR",
        year=2001,
        sequence_identity_percent=100,
        nmr_chain_id="A",
        nmr_core_start_seq_id=1,
        nmr_core_end_seq_id=10,
        nmr_query_sequence_length=11,
        xray_homolog_entity_id="1AAA_1",
        xray_homolog_count=2,
        xray_entry_id="1AAA",
        xray_chain_id="A",
        xray_core_start_seq_id=1,
        xray_core_end_seq_id=10,
        xray_resolution_angstrom=1.0,
        n_common_ca=4,
        rmsd_ca_angstrom=0.5,
        residue_selection=selection,
    )


@pytest.mark.parametrize("is_extremes", [False, True])
@pytest.mark.parametrize("legacy", [False, True])
def test_resume_rejects_core_rows_without_modifying_output(
    tmp_path, is_extremes, legacy
):
    source = tmp_path / "homologs.csv"
    output = tmp_path / "rmsd.csv"
    write_solution_nmr_monomer_xray_homolog_csv([_homolog()], source)
    if is_extremes:
        record = _build_xray_rmsd_extremes_record_from_candidates(
            homolog=_homolog(),
            candidate_records=[_record()],
        )
        write_solution_nmr_monomer_xray_rmsd_extremes_csv([record], output)
    else:
        write_solution_nmr_monomer_xray_rmsd_csv([_record()], output)
    if legacy:
        with output.open(newline="") as stream:
            rows = list(csv.reader(stream))
        with output.open("w", newline="") as stream:
            csv.writer(stream).writerows(row[:-1] for row in rows)
    before = output.read_bytes()
    config = DatasetBuildConfig()
    try:
        with pytest.raises(
            SystemExit, match="Cannot resume.*requested secondary-structure"
        ):
            build_solution_nmr_monomer_xray_rmsd_outputs_to_csv(
                client=RCSBClient(config),
                config=config,
                current_homolog_input_path=source,
                historical_homolog_input_path=None,
                ordinary_output_path=None if is_extremes else output,
                historical_ordinary_output_path=None,
                extremes_output_path=output if is_extremes else None,
                historical_extremes_output_path=None,
                cache_dir=tmp_path / "cache",
                rmsd_workers=1,
                sequence_identity_percent=100,
                resume=True,
                residue_selection="secondary-structure",
            )
    finally:
        _set_active_dataset_filtered_csvs(())
    assert output.read_bytes() == before


def test_secondary_selection_survives_csv_roundtrip(tmp_path):
    ordinary_path, extremes_path = tmp_path / "ordinary.csv", tmp_path / "extremes.csv"
    record = _record("secondary-structure")
    extremes = _build_xray_rmsd_extremes_record_from_candidates(
        homolog=_homolog(),
        candidate_records=[record],
    )
    write_solution_nmr_monomer_xray_rmsd_csv([record], ordinary_path)
    write_solution_nmr_monomer_xray_rmsd_extremes_csv([extremes], extremes_path)
    assert read_solution_nmr_monomer_xray_rmsd_csv(ordinary_path)[0] == record
    assert (
        read_solution_nmr_monomer_xray_rmsd_extremes_csv(extremes_path)[0] == extremes
    )
