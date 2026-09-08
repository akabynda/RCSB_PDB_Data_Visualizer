"""STRIDE failures must keep their cause and must not borrow another chain."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.dataset import stride
from src.dataset.client import RCSBClient
from src.dataset.client import nmr_stride as client_stride
from src.dataset.config import DatasetBuildConfig


def _ca(chain: str) -> str:
    return (
        f"ATOM      1  CA  ALA {chain}   1       0.000   0.000   0.000"
        "  1.00 20.00           C\n"
    )


def test_runner_restores_blank_chain_without_borrowing_named_chain():
    with patch.object(
        stride.subprocess,
        "run",
        return_value=SimpleNamespace(
            returncode=0, stderr="", stdout="ASG ALA - 1 1 H\nASG ALA A 1 1 E\n"
        ),
    ):
        states = stride._run_stride_for_model_text(_ca(" ") + _ca("A"), "stride")
    assert states == {"": {1: "H"}, "A": {1: "E"}}
    assert stride._select_stride_chain_states(states, "B") is None


def test_runner_rejects_ambiguous_blank_and_dash_chain_ids():
    with (
        patch.object(
            stride.subprocess,
            "run",
            return_value=SimpleNamespace(
                returncode=0, stderr="", stdout="ASG ALA - 1 1 H\n"
            ),
        ),
        pytest.raises(RuntimeError, match="cannot distinguish"),
    ):
        stride._run_stride_for_model_text(_ca(" ") + _ca("-"), "stride")


def test_runner_preserves_no_assignment_diagnostic_on_successful_exit():
    with (
        patch.object(
            stride.subprocess,
            "run",
            return_value=SimpleNamespace(
                returncode=0, stderr="IGNORED chain A (less than 5 residues)", stdout=""
            ),
        ),
        pytest.raises(RuntimeError, match="less than 5 residues"),
    ):
        stride._run_stride_for_model_text(_ca("A"), "stride")


@pytest.mark.parametrize(
    ("states", "error", "reason"),
    [
        ({"B": {1: "H", 2: "H", 3: "H"}}, None, "did not report requested chain 'A'"),
        ({"A": {1: "H", 3: "C"}}, None, "missing residue IDs: 2"),
        (None, OSError("Too many open files"), "Too many open files"),
    ],
)
def test_filtered_report_retains_actual_stride_failure(tmp_path, states, error, reason):
    client = RCSBClient(DatasetBuildConfig())
    entity = {"polymer_entity_instances": [{"rcsb_id": "TEST.A"}]}
    with (
        patch.object(
            client,
            "_extract_solution_nmr_monomer_context",
            return_value=("TEST", 2020, 2, entity, "A"),
        ),
        patch.object(
            client_stride, "download_pdb_if_needed", return_value=tmp_path / "TEST.pdb"
        ),
        patch.object(client_stride, "load_cached_chain_id_map", return_value={}),
        patch.object(
            client_stride,
            "parse_first_model_modeled_ca_auth_seq_ids",
            return_value=[1, 2, 3],
        ),
        patch.object(
            stride, "download_pdb_if_needed", return_value=tmp_path / "TEST.pdb"
        ),
        patch.object(stride, "load_cached_chain_id_map", return_value={}),
        patch.object(
            stride,
            "load_first_model_stride_state_by_chain",
            return_value=(states, 2),
            side_effect=error,
        ),
        patch.object(client_stride, "_record_filtered_structure") as filtered,
    ):
        result = (
            client._compute_solution_nmr_monomer_stride_modeled_first_model_for_entry(
                {"rcsb_id": "TEST"}, "stride", tmp_path, tmp_path / "states"
            )
        )
    assert result is None
    filtered.assert_called_once()
    assert filtered.call_args.args[0] == "TEST"
    assert reason in filtered.call_args.args[1]
    assert filtered.call_args.kwargs == {"year": 2020}
