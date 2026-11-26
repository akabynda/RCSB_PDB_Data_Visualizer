"""Refuse to resume from outputs whose last row was cut off mid-write."""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.dataset.arguments import parse_args
from src.dataset.builders.rmsd_outputs import (
    build_solution_nmr_monomer_xray_rmsd_outputs_to_csv,
)
from src.dataset.builders.xray_rmsd import (
    _build_xray_rmsd_extremes_record_from_candidates,
)
from src.dataset.client import RCSBClient
from src.dataset.config import (
    XRAY_HOMOLOG_HETATM_REJECTION_REASON,
    DatasetBuildConfig,
)
from src.dataset.io.homologs import (
    write_rejected_xray_homolog_csv,
    write_solution_nmr_monomer_xray_homolog_csv,
)
from src.dataset.io.nmr import (
    SOLUTION_NMR_MONOMER_PRECISION_HEADER,
    _solution_nmr_monomer_precision_csv_row,
)
from src.dataset.io.rmsd import (
    write_solution_nmr_monomer_xray_rmsd_csv,
    write_solution_nmr_monomer_xray_rmsd_extremes_csv,
)
from src.dataset.records import (
    RejectedXrayHomologRecord,
    ResidueId,
    SolutionNMRMonomerPrecisionRecord,
    SolutionNMRMonomerXrayHomologRecord,
    SolutionNMRMonomerXrayRmsdRecord,
)
from src.dataset.reporting import _set_active_dataset_filtered_csvs
from src.dataset.workflows.homolog_resume import (
    HomologOutputPaths,
    load_homolog_resume_state,
)
from src.dataset.workflows.stride import run_precision


def _truncate_inside_last_value(path: Path) -> bytes:
    """Cut the file inside the next-to-last field of its last row."""
    data = path.read_bytes()
    truncated = data[: data.rstrip().rfind(b",") - 2]
    path.write_bytes(truncated)
    return truncated


def _homolog(identity: int) -> SolutionNMRMonomerXrayHomologRecord:
    return SolutionNMRMonomerXrayHomologRecord(
        entry_id="1NMR",
        year=2001,
        sequence_identity_percent=identity,
        nmr_core_start_seq_id=1,
        nmr_core_end_seq_id=11,
        nmr_query_sequence_length=11,
        xray_homolog_entry_ids=("1AAA",),
        xray_homolog_entity_ids=("1AAA_1",),
        has_xray_homolog=True,
    )


def _rejected(identity: int) -> RejectedXrayHomologRecord:
    return RejectedXrayHomologRecord(
        nmr_entry_id="1NMR",
        nmr_year=2001,
        nmr_chain_id="A",
        sequence_identity_percent=identity,
        nmr_core_start_seq_id=1,
        nmr_core_end_seq_id=11,
        nmr_query_sequence_length=11,
        xray_entry_id="2BBB",
        xray_entity_id="2BBB_1",
        xray_chain_ids=("A",),
        reason=XRAY_HOMOLOG_HETATM_REJECTION_REASON,
    )


def _rmsd() -> SolutionNMRMonomerXrayRmsdRecord:
    return SolutionNMRMonomerXrayRmsdRecord(
        entry_id="1NMR",
        year=2001,
        sequence_identity_percent=100,
        nmr_chain_id="A",
        nmr_core_start_seq_id=1,
        nmr_core_end_seq_id=11,
        nmr_query_sequence_length=11,
        xray_homolog_entity_id="1AAA_1",
        xray_homolog_count=1,
        xray_entry_id="1AAA",
        xray_chain_id="A",
        xray_core_start_seq_id=1,
        xray_core_end_seq_id=11,
        xray_resolution_angstrom=1.0,
        n_common_ca=11,
        rmsd_ca_angstrom=12.345678,
        residue_selection="core",
    )


def test_precision_resume_rejects_truncated_last_row(tmp_path: Path) -> None:
    output = tmp_path / "precision.csv"
    record = SolutionNMRMonomerPrecisionRecord(
        entry_id="1NMR",
        year=2001,
        chain_id="A",
        core_start_seq_id=ResidueId(1),
        core_end_seq_id=ResidueId(11),
        n_models=20,
        n_ca_core_used=11,
        n_ca_core_raw=11,
        mean_rmsd_angstrom=12.345678,
    )
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(SOLUTION_NMR_MONOMER_PRECISION_HEADER)
        writer.writerow(_solution_nmr_monomer_precision_csv_row(record))
    truncated = _truncate_inside_last_value(output)
    with patch.object(
        sys,
        "argv",
        [
            "builder",
            "--resume",
            "--solution-nmr-monomer-precision-stride-modeled-first-model-output",
            str(output),
        ],
    ):
        args = parse_args()

    with (
        patch(
            "src.dataset.workflows.stride.ensure_stride_executable",
            return_value="stride",
        ),
        pytest.raises(SystemExit, match="incomplete last row"),
    ):
        run_precision(args, Mock(), DatasetBuildConfig())
    assert output.read_bytes() == truncated


@pytest.mark.parametrize(
    "damaged", ["homolog_95", "homolog_100", "rejected_95", "rejected_100"]
)
def test_homolog_resume_rejects_truncated_last_row(
    tmp_path: Path, damaged: str
) -> None:
    paths = HomologOutputPaths(
        homolog_95=tmp_path / "homologs_95.csv",
        homolog_100=tmp_path / "homologs_100.csv",
        rejected_95=tmp_path / "homologs_95_rejected.csv",
        rejected_100=tmp_path / "homologs_100_rejected.csv",
        checkpoint=tmp_path / "homologs_95.resume.tsv",
    )
    write_solution_nmr_monomer_xray_homolog_csv([_homolog(95)], paths.homolog_95)
    write_solution_nmr_monomer_xray_homolog_csv([_homolog(100)], paths.homolog_100)
    write_rejected_xray_homolog_csv([_rejected(95)], paths.rejected_95)
    write_rejected_xray_homolog_csv([_rejected(100)], paths.rejected_100)
    paths.checkpoint.write_text("1NMR\tsuccess_with_rejected_audit\n")
    _truncate_inside_last_value(getattr(paths, damaged))

    with pytest.raises(SystemExit, match="incomplete last row"):
        load_homolog_resume_state(True, paths)


@pytest.mark.parametrize("is_extremes", [False, True])
def test_rmsd_resume_rejects_truncated_last_row(
    tmp_path: Path, is_extremes: bool
) -> None:
    source = tmp_path / "homologs.csv"
    output = tmp_path / "rmsd.csv"
    write_solution_nmr_monomer_xray_homolog_csv([_homolog(100)], source)
    if is_extremes:
        extremes = _build_xray_rmsd_extremes_record_from_candidates(
            homolog=_homolog(100), candidate_records=[_rmsd()]
        )
        write_solution_nmr_monomer_xray_rmsd_extremes_csv([extremes], output)
    else:
        write_solution_nmr_monomer_xray_rmsd_csv([_rmsd()], output)
    truncated = _truncate_inside_last_value(output)
    config = DatasetBuildConfig()
    try:
        with pytest.raises(SystemExit, match="incomplete last row"):
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
                residue_selection="core",
            )
    finally:
        _set_active_dataset_filtered_csvs(())
    assert output.read_bytes() == truncated
