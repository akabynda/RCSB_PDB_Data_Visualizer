"""Numerical and resume checks for secondary-structure ensemble precision."""

import csv
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from src.dataset.arguments import parse_args
from src.dataset.builders.precision import (
    SolutionNMRMonomerPrecisionBuilder,
    SolutionNMRMonomerPrecisionStrideModeledFirstModelBuilder,
)
from src.dataset.config import DatasetBuildConfig
from src.dataset.io.nmr import (
    SOLUTION_NMR_MONOMER_PRECISION_HEADER,
    _solution_nmr_monomer_precision_csv_row,
    read_solution_nmr_monomer_precision_csv,
)
from src.dataset.records import ResidueId, SolutionNMRMonomerPrecisionRecord
from src.dataset.stride import compute_stride_secondary_residue_ids_in_first_model
from src.dataset.workflows.stride import run_precision


def _args(*arguments):
    with patch.object(sys, "argv", ["builder", *arguments]):
        return parse_args()


def _record():
    return SolutionNMRMonomerPrecisionRecord("TEST", 2020, "A", 1, 5, 2, 5, 5, 2.0)


def test_mask_excludes_mobile_loop_before_alignment_and_preserves_insertions():
    ids = [ResidueId(1), ResidueId(2), ResidueId(2, "A"), ResidueId(3), ResidueId(5)]
    coordinates = np.array(
        [[0, 0, 0], [2, 0, 0], [0, 3, 0], [0, 0, 4], [3, 3, 3]], float
    )
    shifted = coordinates + [5, -2, 1]
    shifted[-1] += [20, 10, -8]  # Only the coil moves internally.
    maps = [dict(zip(ids, model)) for model in (coordinates, shifted)]
    counts = [{resid: 2 if resid == ResidueId(2, "A") else 1 for resid in ids}] * 2
    with patch(
        "src.dataset.builders.precision.parse_models_ca_coords_with_stats",
        return_value=(maps, counts),
    ):
        full, _ = SolutionNMRMonomerPrecisionBuilder._compute_mean_rmsd_to_average(
            Path("unused"), "A", 1, 5
        )
        selected, reason = (
            SolutionNMRMonomerPrecisionBuilder._compute_mean_rmsd_to_average(
                Path("unused"),
                "A",
                1,
                5,
                allowed_residue_ids=set(ids[:-1]),
            )
        )
    assert reason is None
    assert full[3] > 1
    assert selected[:3] == (2, 4, 5)
    assert selected[3] == pytest.approx(0, abs=1e-7)


def test_mask_requires_three_residues_shared_by_all_models():
    maps = [{1: (0, 0, 0), 2: (2, 0, 0), 3: (0, 3, 0)}, {1: (0, 0, 0), 2: (2, 0, 0)}]
    with patch(
        "src.dataset.builders.precision.parse_models_ca_coords_with_stats",
        return_value=(maps, [{}, {}]),
    ):
        result, reason = (
            SolutionNMRMonomerPrecisionBuilder._compute_mean_rmsd_to_average(
                Path("unused"),
                "A",
                1,
                3,
                allowed_residue_ids={1, 2, 3},
            )
        )
    assert result is None
    assert "found 2" in reason


def test_stride_mask_excludes_turns_coil_and_outside_core_and_requires_complete_states():
    ids = [ResidueId(2), ResidueId(2, "A"), ResidueId(3), ResidueId(4), ResidueId(5)]
    states = dict(zip(ids, ["H", "E", "T", "C", "B"]))
    with patch(
        "src.dataset.stride.load_first_model_stride_state_by_chain",
        return_value=({"A": states}, 2),
    ):
        result = compute_stride_secondary_residue_ids_in_first_model(
            Path("unused"),
            "TEST",
            "A",
            ids[:-1],
            "stride",
            Path("unused"),
        )
        assert result == {ResidueId(2), ResidueId(2, "A")}
        with pytest.raises(ValueError, match="Incomplete STRIDE"):
            compute_stride_secondary_residue_ids_in_first_model(
                Path("unused"),
                "TEST",
                "A",
                [ResidueId(99)],
                "stride",
                Path("unused"),
            )


def test_existing_precision_metadata_skips_api_discovery_and_supports_resume(tmp_path):
    client = Mock()
    builder = SolutionNMRMonomerPrecisionStrideModeledFirstModelBuilder(
        client,
        DatasetBuildConfig(),
        tmp_path,
        1,
        "stride",
        tmp_path,
        residue_selection="secondary-structure",
    )
    record = _record()
    with patch.object(builder, "_recompute_record", return_value=record) as compute:
        assert builder.build(source_records=[record]) == [record]
        assert builder.build(source_records=[record], skip_entry_ids={"TEST"}) == []
    compute.assert_called_once_with(record)
    assert not client.mock_calls


def test_secondary_defaults_are_separate_and_explicit_path_is_honored():
    args = _args("--rmsd-residue-selection", "secondary-structure")
    assert args.solution_nmr_monomer_precision_stride_modeled_first_model_output.stem.endswith(
        "_secondary_structure"
    )
    assert args.solution_nmr_monomer_xray_rmsd_extremes_output.stem.endswith(
        "_secondary_structure"
    )
    assert args.solution_nmr_monomer_xray_homolog_100_output == Path(
        "data/solution_nmr_monomer_xray_homologs_100.csv"
    )
    path = "data/solution_nmr_monomer_xray_rmsd_extremes.csv"
    args = _args(
        "--rmsd-residue-selection",
        "secondary-structure",
        "--solution-nmr-monomer-xray-rmsd-extremes-output",
        path,
    )
    assert args.solution_nmr_monomer_xray_rmsd_extremes_output == Path(path)


def test_legacy_precision_resume_cannot_silently_reuse_core_values(tmp_path):
    output = tmp_path / "precision.csv"
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(SOLUTION_NMR_MONOMER_PRECISION_HEADER[:-1])
        writer.writerow(_solution_nmr_monomer_precision_csv_row(_record())[:-1])
    original = output.read_bytes()
    assert (
        read_solution_nmr_monomer_precision_csv(output)[0].residue_selection == "core"
    )
    args = _args(
        "--resume",
        "--rmsd-residue-selection",
        "secondary-structure",
        "--solution-nmr-monomer-precision-stride-modeled-first-model-output",
        str(output),
    )
    with patch(
        "src.dataset.workflows.stride.ensure_stride_executable", return_value="stride"
    ):
        with pytest.raises(ValueError, match="different residue selection"):
            run_precision(args, Mock(), DatasetBuildConfig())
    assert output.read_bytes() == original
