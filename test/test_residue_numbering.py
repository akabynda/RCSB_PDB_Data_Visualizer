"""Residue numbering must remain distinct from alternate atom locations."""

from pathlib import Path

import numpy as np
import pytest

from src.dataset.builders.xray_rmsd import SolutionNMRMonomerXrayRmsdBuilder
from src.dataset.builders.precision import SolutionNMRMonomerPrecisionBuilder
from src.dataset.ca_cache import load_cached_first_model_ca_data
from src.dataset.coordinates import (
    parse_first_model_ca_residues,
    parse_first_model_modeled_ca_auth_seq_ids,
    parse_models_ca_coords_with_stats,
)
from src.dataset.geometry import _superposed_rmsd
from src.dataset.io.rmsd import (
    read_solution_nmr_monomer_xray_rmsd_csv,
    write_solution_nmr_monomer_xray_rmsd_csv,
)
from src.dataset.records import (
    ResidueId,
    SolutionNMRMonomerXrayRmsdRecord,
    parse_residue_id,
)
from src.dataset.stride import _extract_stride_core_range_for_modeled_auth_seq_ids


def _ca(
    serial: int,
    key: ResidueId,
    resname: str = "ALA",
    *,
    chain: str = "A",
    altloc: str = " ",
    occupancy: float = 1.0,
    xyz: tuple[float, float, float] | None = None,
    record: str = "ATOM  ",
) -> str:
    if xyz is None:
        xyz = (float(serial), float(serial % 3), float(serial % 5))
    return (
        f"{record}{serial:5d}  CA {altloc}{resname:>3} {chain}"
        f"{key.seq_id:4d}{key.insertion_code or ' '}   "
        f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}"
        f"{occupancy:6.2f}{20.0:6.2f}           C\n"
    )


def _keys_and_coords(path: Path, expected: list[ResidueId]) -> None:
    """Compare independent residue, ensemble, and disk-cache consumers."""
    records = parse_first_model_ca_residues(path, "A")
    assert [record.key for record in records] == expected
    assert list(parse_first_model_modeled_ca_auth_seq_ids(path, "A")) == expected
    models, counts = parse_models_ca_coords_with_stats(path, "A")
    assert list(models[0]) == expected
    assert counts[0] == dict.fromkeys(expected, 1)
    for _ in range(2):
        cached_records, cached_coords = load_cached_first_model_ca_data(path, "A")
        assert [record.key for record in cached_records] == expected
        assert list(cached_coords) == expected
        for key in expected:
            np.testing.assert_array_equal(cached_coords[key], models[0][key])


def test_negative_zero_gaps_and_nonmonotonic_numbers_are_not_altloc(tmp_path):
    keys = [ResidueId(n) for n in (-2, -1, 0, 4, 10, 7, 9999)]
    path = tmp_path / "author_numbering.pdb"
    path.write_text("".join(_ca(i, key) for i, key in enumerate(keys, 1)))
    _keys_and_coords(path, keys)


def test_2orc_insertion_sequence_survives_all_coordinate_consumers(tmp_path):
    keys = [ResidueId(56)] + [ResidueId(56, c) for c in "ABCDE"] + [ResidueId(57)]
    names = ["LYS", "ASP", "GLY", "GLU", "VAL", "LYS", "PRO"]
    path = tmp_path / "2orc_numbering.pdb"
    path.write_text(
        "".join(_ca(i, key, name) for i, (key, name) in enumerate(zip(keys, names), 1))
    )
    _keys_and_coords(path, keys)
    assert (
        "".join(record.identity for record in parse_first_model_ca_residues(path, "A"))
        == "KDGEVKP"
    )


def test_reverse_insertion_order_defines_core_and_coordinate_interval(tmp_path):
    # 3B9F's light chain begins 1R, 1Q, ..., 1A, 1 rather than numeric order.
    keys = [ResidueId(1, c) for c in "RQPONMLKJIHGFEDCBA"]
    keys += [ResidueId(n) for n in range(1, 15)]
    keys += [ResidueId(14, c) for c in "ABCDEFGHIJKL"]
    path = tmp_path / "reverse_insertions.pdb"
    path.write_text("".join(_ca(i, key) for i, key in enumerate(keys, 1)))
    _keys_and_coords(path, keys)
    states = dict.fromkeys(keys, "C")
    for key in [ResidueId(1, c) for c in "JIHGCBA"] + [
        ResidueId(14, c) for c in "BCDEFGHIJ"
    ]:
        states[key] = "H"
    bounds = _extract_stride_core_range_for_modeled_auth_seq_ids(states, keys)
    assert bounds == (ResidueId(1, "J"), ResidueId(14, "J"))
    expected = keys[8:42]
    records = parse_first_model_ca_residues(path, "A", *bounds)
    models, _ = parse_models_ca_coords_with_stats(path, "A", *bounds)
    assert len(records) == 34
    assert [record.key for record in records] == expected
    assert list(models[0]) == expected


def test_numeric_and_lowercase_insertions_remain_distinct(tmp_path):
    keys = [
        ResidueId(56),
        ResidueId(56, "1"),
        ResidueId(561),
        ResidueId(56, "a"),
        ResidueId(56, "A"),
    ]
    path = tmp_path / "numeric_insertion.pdb"
    path.write_text("".join(_ca(i, key) for i, key in enumerate(keys, 1)))
    _keys_and_coords(path, keys)
    assert str(keys[1]) == "56^1"
    assert all(parse_residue_id(str(key)) == key for key in keys)


def test_long_component_preserves_numeric_insertion_and_four_digit_number(tmp_path):
    keys = [ResidueId(56, "1"), ResidueId(561), ResidueId(9999)]
    path = tmp_path / "long_component.pdb"
    path.write_text(
        "SEQRES   1 A    3  A1BEB A1BEB A1BEB\n"
        + "".join(
            _ca(i, key, "A1BEB", record="HETATM") for i, key in enumerate(keys, 1)
        )
    )
    _keys_and_coords(path, keys)


def test_altloc_selection_is_independent_for_each_insertion_and_model(tmp_path):
    keys = [ResidueId(56), ResidueId(56, "A")]
    lines = []
    for model in (1, 2):
        lines.append(f"MODEL     {model:4d}\n")
        for i, key in enumerate(keys, 1):
            major = "O" if i == model else "C"
            for altloc in "CO":
                lines.append(
                    _ca(
                        i,
                        key,
                        altloc=altloc,
                        occupancy=0.95 if altloc == major else 0.05,
                        xyz=(float(model * 10 + i), float(altloc == major), 0.0),
                    )
                )
        lines.append(_ca(10, ResidueId(56), chain="B", xyz=(99.0, 99.0, 99.0)))
        lines.append("ENDMDL\n")
    path = tmp_path / "independent_conformers.pdb"
    path.write_text("".join(lines))
    models, _ = parse_models_ca_coords_with_stats(path, "A")
    assert len(models) == 2
    for model, coords in enumerate(models, 1):
        assert list(coords) == keys
        for i, key in enumerate(keys, 1):
            np.testing.assert_array_equal(coords[key], [model * 10 + i, 1.0, 0.0])


def test_missing_later_model_boundary_does_not_change_reverse_numbered_core(tmp_path):
    keys = [ResidueId(1, c) for c in "KJIHGF"] + [ResidueId(2), ResidueId(3)]
    start, end = ResidueId(1, "J"), ResidueId(2)
    first_core = keys[1:-1]
    path = tmp_path / "missing_core_boundary.pdb"
    lines = []
    for model, missing in [(1, None), (2, start), (3, end)]:
        lines.append(f"MODEL     {model:4d}\n")
        lines.extend(_ca(i, key) for i, key in enumerate(keys, 1) if key != missing)
        lines.append("ENDMDL\n")
    path.write_text("".join(lines))
    models, counts = parse_models_ca_coords_with_stats(path, "A", start, end)
    for index, missing in enumerate([None, start, end]):
        expected = [key for key in first_core if key != missing]
        assert list(models[index]) == expected
        assert list(counts[index]) == expected
    result, reason = SolutionNMRMonomerPrecisionBuilder._compute_mean_rmsd_to_average(
        path, "A", start, end
    )
    assert reason is None
    assert result is not None
    assert result[:3] == (3, 4, 4)
    assert result[3] == pytest.approx(0.0, abs=1e-7)


@pytest.mark.parametrize("end", [ResidueId(56, "E"), ResidueId(56, "1")])
def test_rmsd_csv_roundtrip_keeps_full_core_endpoints(tmp_path, end):
    record = SolutionNMRMonomerXrayRmsdRecord(
        entry_id="2ORC",
        year=1998,
        sequence_identity_percent=100,
        nmr_chain_id="A",
        nmr_core_start_seq_id=ResidueId(-2),
        nmr_core_end_seq_id=end,
        nmr_query_sequence_length=11,
        xray_homolog_entity_id="TEST_1",
        xray_homolog_count=1,
        xray_entry_id="TEST",
        xray_chain_id="A",
        xray_core_start_seq_id=ResidueId(0),
        xray_core_end_seq_id=end,
        xray_resolution_angstrom=1.5,
        n_common_ca=11,
        rmsd_ca_angstrom=1.25,
    )
    path = tmp_path / "rmsd.csv"
    write_solution_nmr_monomer_xray_rmsd_csv([record], path)
    assert read_solution_nmr_monomer_xray_rmsd_csv(path) == [record]


def test_xray_rmsd_pairs_use_each_inserted_residues_own_coordinates(tmp_path):
    keys = [ResidueId(n) for n in range(51, 57)] + [ResidueId(56, c) for c in "ABCDE"]
    nmr_coords = np.array([(float(i), float(i % 3), float(i % 5)) for i in range(11)])
    xray_coords = nmr_coords.copy()
    xray_coords[8, 2] += 3.0  # Only the independently numbered 56C moves.
    nmr_path = tmp_path / "nmr.pdb"
    xray_path = tmp_path / "xray.pdb"
    for path, coords in [(nmr_path, nmr_coords), (xray_path, xray_coords)]:
        path.write_text(
            "".join(
                _ca(i + 1, key, xyz=tuple(xyz))
                for i, (key, xyz) in enumerate(zip(keys, coords))
            )
        )
    result = SolutionNMRMonomerXrayRmsdBuilder._compute_ca_rmsd_to_xray(
        nmr_pdb_path=nmr_path,
        nmr_chain_id="A",
        nmr_core_start_seq_id=keys[0],
        nmr_core_end_seq_id=keys[-1],
        xray_pdb_path=xray_path,
        xray_chain_id="A",
        sequence_identity_percent=100,
    )
    assert result is not None
    assert result[0] == 11
    assert result[1] == pytest.approx(
        _superposed_rmsd(nmr_coords, xray_coords), abs=1e-8
    )
    assert result[2:] == (keys[0], keys[-1], keys[0], keys[-1])
