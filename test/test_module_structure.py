"""Regression coverage for public entry points and lightweight module imports."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("script_name", "expected_option"),
    [
        ("pdb_dataset_builder", "--datasets"),
        ("pdb_plot", "--plots"),
    ],
)
def test_direct_and_module_entry_points_have_matching_help(
    script_name: str, expected_option: str, tmp_path: Path
) -> None:
    """Allow absolute script execution outside the repo and normal module use."""
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    direct = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "src" / f"{script_name}.py"), "--help"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    module = subprocess.run(
        [sys.executable, "-m", f"src.{script_name}", "--help"],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert expected_option in direct.stdout
    assert direct.stdout == module.stdout


def test_legacy_exports_retain_the_canonical_objects() -> None:
    """Keep existing scripts, enum comparisons, and record type checks compatible."""
    from src import pdb_dataset_builder, pdb_plot
    from src.dataset.builders.counts import PDBMethodYearlyBuilder
    from src.dataset.client import RCSBClient, ThreadLocalRequestsSession
    from src.dataset.config import DatasetBuildConfig, DatasetKind
    from src.dataset.geometry import _superposed_rmsd
    from src.dataset.io.nmr import write_solution_nmr_weights_csv
    from src.dataset.records import SolutionNMRWeightRecord
    from src.plotting import PDBScientificPlotter, PlotConfig, PlotKind

    builder_exports = (
        PDBMethodYearlyBuilder,
        RCSBClient,
        ThreadLocalRequestsSession,
        DatasetBuildConfig,
        DatasetKind,
        _superposed_rmsd,
        write_solution_nmr_weights_csv,
        SolutionNMRWeightRecord,
    )
    for exported in builder_exports:
        assert getattr(pdb_dataset_builder, exported.__name__) is exported
    for exported in (PDBScientificPlotter, PlotConfig, PlotKind):
        assert getattr(pdb_plot, exported.__name__) is exported


@pytest.mark.parametrize("module_name", ["src.dataset.config", "src.dataset.records"])
def test_data_types_do_not_import_services_or_plotting(module_name: str) -> None:
    """Basic types remain usable without initializing API or plotting modules."""
    program = """
import importlib
import sys

importlib.import_module(sys.argv[1])
forbidden = ("src.plotting", "src.pdb_plot", "src.dataset.client", "MDAnalysis")
loaded = sorted(
    name for name in sys.modules
    if any(name == prefix or name.startswith(prefix + ".") for prefix in forbidden)
)
assert not loaded, f"Low-level import loaded unrelated dependencies: {loaded}"
"""
    result = subprocess.run(
        [sys.executable, "-c", program, module_name],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
