"""Tests for caching STRIDE assignments for first PDB models."""

import tempfile
from types import SimpleNamespace
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import src.dataset.stride as stride
from src.dataset.config import DatasetBuildConfig, STRIDE_STATE_CODES
from src.dataset.records import ResidueId
from src.pdb_dataset_builder import load_first_model_stride_state_by_chain


def _ca_line(serial: int, resid: int) -> str:
    """Return a minimal alpha-carbon PDB record for ``resid``."""
    return (
        f"ATOM  {serial:5d}  CA  ALA A{resid:4d}    "
        f"{0.0:8.3f}{0.0:8.3f}{0.0:8.3f}{1.0:6.2f}{20.0:6.2f}"
        "           C\n"
    )


class StrideCacheTests(unittest.TestCase):
    """Verify that identical STRIDE inputs reuse cached assignments."""

    def test_reuses_cached_first_model_stride_states(self) -> None:
        """Avoid rerunning STRIDE when a matching cache entry exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pdb_path = root / "1ABC.pdb"
            cache_dir = root / "stride_cache"
            pdb_path.write_text(
                "".join(
                    [
                        "MODEL        1\n",
                        _ca_line(1, 1),
                        "ENDMDL\n",
                        "MODEL        2\n",
                        _ca_line(2, 1),
                        "ENDMDL\n",
                    ]
                ),
                encoding="utf-8",
            )

            expected_states = {"A": {1: "H"}}
            with patch(
                "src.dataset.stride._run_stride_for_model_text",
                return_value=expected_states,
            ) as run_stride:
                first_states, first_model_count = (
                    load_first_model_stride_state_by_chain(
                        pdb_path=pdb_path,
                        entry_id="1ABC",
                        stride_executable="stride",
                        stride_cache_dir=cache_dir,
                    )
                )
                second_states, second_model_count = (
                    load_first_model_stride_state_by_chain(
                        pdb_path=pdb_path,
                        entry_id="1ABC",
                        stride_executable="stride",
                        stride_cache_dir=cache_dir,
                    )
                )

            self.assertEqual(first_states, expected_states)
            self.assertEqual(second_states, expected_states)
            self.assertEqual(first_model_count, 2)
            self.assertEqual(second_model_count, 2)
            self.assertEqual(run_stride.call_count, 1)
            self.assertTrue((cache_dir / "1ABC.json").exists())


if __name__ == "__main__":
    unittest.main()


def test_stride_preserves_independent_insertions_and_signed_numbering(
    tmp_path: Path,
) -> None:
    states = stride._parse_stride_state_by_chain(
        "\n".join(
            f"ASG ALA A {residue_id} {ordinal} {state}"
            for ordinal, (residue_id, state) in enumerate(
                [
                    ("-3B", "G"),
                    ("0", "C"),
                    ("56", "H"),
                    ("56A", "T"),
                    ("56a", "E"),
                    ("57", "C"),
                ],
                start=1,
            )
        )
    )
    expected = {
        "A": {
            ResidueId(-3, "B"): "G",
            ResidueId(0): "C",
            ResidueId(56): "H",
            ResidueId(56, "A"): "T",
            ResidueId(56, "a"): "E",
            ResidueId(57): "C",
        }
    }
    assert states == expected
    cache_path = tmp_path / "states.json"
    stride._write_cached_stride_state_by_chain(cache_path, "entry", "sha", states)
    assert stride._load_cached_stride_state_by_chain(cache_path, "sha") == expected


def test_stride_restores_numeric_insertion_code_without_author_number_collision() -> (
    None
):
    normal = _ca_line(1, 56)
    inserted = _ca_line(2, 56)
    inserted = inserted[:26] + "1" + inserted[27:]
    separate_number = _ca_line(3, 561)
    model_text = normal + inserted + separate_number

    def run(command, **kwargs):
        received = Path(command[1]).read_text(encoding="utf-8").splitlines()
        assert [line[22:27] for line in received] == ["   1 ", "   2 ", "   3 "]
        assert [line[30:] for line in received] == [
            line[30:] for line in model_text.splitlines()
        ]
        return SimpleNamespace(
            returncode=0, stdout=("ASG ALA A 1 1 H\nASG ALA A 2 2 E\nASG ALA A 3 3 C\n")
        )

    with patch.object(stride.subprocess, "run", side_effect=run):
        assert stride._run_stride_for_model_text(model_text, "stride") == {
            "A": {ResidueId(56): "H", ResidueId(56, "1"): "E", ResidueId(561): "C"}
        }


def test_stride_numeric_insertion_round_trips_unambiguous_cache_keys(
    tmp_path: Path,
) -> None:
    expected = {"A": {ResidueId(56, "1"): "E", ResidueId(561): "C"}}
    cache_path = tmp_path / "states.json"
    stride._write_cached_stride_state_by_chain(cache_path, "entry", "sha", expected)
    assert stride._load_cached_stride_state_by_chain(cache_path, "sha") == expected


def test_stride_incomplete_assignment_is_unavailable_even_with_successful_exit(
    tmp_path: Path,
) -> None:
    ids = [ResidueId(56), ResidueId(56, "A"), ResidueId(57)]
    with (
        patch.object(stride, "download_pdb_if_needed", return_value=tmp_path / "x.pdb"),
        patch.object(stride, "load_cached_chain_id_map", return_value={}),
        patch.object(
            stride,
            "load_first_model_stride_state_by_chain",
            return_value=({"A": {ids[0]: "H", ids[2]: "C"}}, 1),
        ),
        patch.object(stride.LOGGER, "warning") as warning,
    ):
        result = stride.compute_stride_state_coverages_for_chain_modeled_first_model(
            MagicMock(),
            DatasetBuildConfig(),
            tmp_path,
            tmp_path,
            "entry",
            "A",
            len(ids),
            ids,
            "stride",
        )
    assert result == ({state: -1.0 for state in STRIDE_STATE_CODES}, 1, 0)
    warning.assert_called_once()


def test_stride_counts_only_explicit_coil_and_preserves_inserted_residues(
    tmp_path: Path,
) -> None:
    ids = [ResidueId(56), ResidueId(56, "A"), ResidueId(57)]
    with (
        patch.object(stride, "download_pdb_if_needed", return_value=tmp_path / "x.pdb"),
        patch.object(stride, "load_cached_chain_id_map", return_value={}),
        patch.object(
            stride,
            "load_first_model_stride_state_by_chain",
            return_value=({"A": dict(zip(ids, ["H", "E", "C"]))}, 1),
        ),
    ):
        coverages, count, succeeded = (
            stride.compute_stride_state_coverages_for_chain_modeled_first_model(
                MagicMock(),
                DatasetBuildConfig(),
                tmp_path,
                tmp_path,
                "entry",
                "A",
                len(ids),
                ids,
                "stride",
            )
        )
    assert (count, succeeded) == (1, 1)
    assert coverages["H"] == coverages["E"] == coverages["C"] == 1 / 3
    assert sum(coverages.values()) == 1.0


def test_lowercase_bridge_assignments_are_counted_and_define_core(
    tmp_path: Path,
) -> None:
    # Actual ASG records for the three falsely missing residues of 1TIU.
    stdout = (
        "ASG  LYS A    6    6    b        Bridge   -120.79    105.33     115.1\n"
        "ASG  GLU A   24   24    b        Bridge   -101.08    131.88      80.7\n"
        "ASG  LEU A   36   36    b        Bridge    -75.25   -131.45      47.3\n"
        "ASG  ALA A   37   37    C          Coil     0.00      0.00       0.0\n"
    )
    ids = [ResidueId(number) for number in (6, 24, 36, 37)]
    states = stride._parse_stride_state_by_chain(stdout)
    assert states == {"A": dict(zip(ids, ["B", "B", "B", "C"]))}
    assert stride._extract_stride_core_range_for_modeled_auth_seq_ids(
        states["A"], ids
    ) == (ResidueId(6), ResidueId(36))
    with (
        patch.object(stride, "download_pdb_if_needed", return_value=tmp_path / "x.pdb"),
        patch.object(stride, "load_cached_chain_id_map", return_value={}),
        patch.object(
            stride, "load_first_model_stride_state_by_chain", return_value=(states, 1)
        ),
    ):
        coverages, count, succeeded = (
            stride.compute_stride_state_coverages_for_chain_modeled_first_model(
                MagicMock(),
                DatasetBuildConfig(),
                tmp_path,
                tmp_path,
                "1TIU",
                "A",
                len(ids),
                ids,
                "stride",
            )
        )
    assert (count, succeeded) == (1, 1)
    assert coverages["B"] == 0.75
    assert coverages["C"] == 0.25
    assert sum(coverages.values()) == 1.0


def test_stride_temporary_numbering_handles_shifted_long_component_fields() -> None:
    protein = _ca_line(1, 56)
    component = _ca_line(2, 56)
    component = component[:26] + "1" + component[27:]
    component = "HETATM" + component[6:17] + "A1A2" + component[20:]
    prepared, mapping = stride._prepare_stride_residue_numbering(protein + component)
    lines = prepared.splitlines()
    assert lines[0][22:27] == "   1 "
    assert lines[1][23:28] == "   2 "
    assert lines[1][31:] == component.splitlines()[0][31:]
    assert mapping == {
        "A": {ResidueId(1): ResidueId(56), ResidueId(2): ResidueId(56, "1")}
    }
