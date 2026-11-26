"""Check that both command-line tools list their defaults in --help."""

from __future__ import annotations

import contextlib
import io
import os
import sys
from unittest.mock import patch

import pytest

from src.dataset.arguments import parse_args as parse_builder_args
from src.plotting.cli import parse_args as parse_plot_args


def _help_text(parse_args, *arguments: str) -> str:
    """Return unwrapped --help output for an isolated command line."""
    output = io.StringIO()
    with (
        patch.dict(os.environ, {"COLUMNS": "10000"}),
        patch.object(sys, "argv", ["prog", *arguments, "--help"]),
        contextlib.redirect_stdout(output),
        pytest.raises(SystemExit),
    ):
        parse_args()
    return output.getvalue()


def test_builder_help_lists_defaults() -> None:
    text = _help_text(parse_builder_args)

    assert "(default: method_counts, solution_nmr_weights, " in text
    assert "(default: data/pdb_cache)" in text
    assert "(default: 100)" in text
    assert "(default: core)" in text
    assert "default: None" not in text
    assert "default: False" not in text
    assert "default: )" not in text
    assert "DatasetKind" not in text


def test_builder_help_lists_selected_secondary_structure_defaults() -> None:
    text = _help_text(
        parse_builder_args, "--rmsd-residue-selection", "secondary-structure"
    )

    assert (
        "(default: data/solution_nmr_monomer_xray_rmsd_secondary_structure.csv)"
        in text
    )


def test_plot_help_lists_defaults() -> None:
    text = _help_text(parse_plot_args)

    assert "(default: data/pdb_method_counts_by_year.csv)" in text
    assert "(default: figures/solution_nmr_monomer_xray_rmsd_by_year.png)" in text
    assert "default: None" not in text
    assert "default: False" not in text
    assert "PlotKind" not in text
    assert text.count("(default: all)") == 1
