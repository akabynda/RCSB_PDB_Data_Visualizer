"""Offline integration checks for membrane collection, cache reuse and publication."""

import csv
import json
from pathlib import Path
import sys
from unittest.mock import Mock

import pytest

from src.dataset import cli
from src.dataset.arguments import parse_args, parse_dataset_kinds
from src.dataset.cache import _sha256_file
from src.dataset.client import RCSBClient
from src.dataset.config import DatasetBuildConfig, DatasetKind
from src.dataset.reporting import (
    _configure_dataset_filtered_csvs,
    _set_active_dataset_filtered_csvs,
)
from src.dataset.workflows.membrane import run_solution_nmr_membrane


def _args(monkeypatch, tmp_path, *extra):
    monkeypatch.setattr(sys, "argv", [
        "pdb_dataset_builder.py", "--datasets", "solution_nmr_membrane_proteins",
        "--solution-nmr-membrane-output", str(tmp_path / "cohort.csv"),
        "--solution-nmr-membrane-cache-dir", str(tmp_path / "cache"), *extra,
    ])
    return parse_args()


def _entry():
    return {
        "rcsb_id": "1ABC", "exptl": [{"method": "SOLUTION NMR"}],
        "rcsb_entry_info": {"deposited_model_count": 1},
        "rcsb_accession_info": {"deposit_date": "2026-01-01"},
        "polymer_entities": [{
            "entity_poly": {"type": "polypeptide(L)", "pdbx_seq_one_letter_code_can": "AAAA"},
            "rcsb_polymer_entity_container_identifiers": {"entity_id": "1", "asym_ids": ["A"]},
            "rcsb_polymer_entity_annotation": [{"type": "OPM"}],
            "polymer_entity_instances": [{
                "rcsb_polymer_entity_instance_container_identifiers": {"asym_id": "A", "auth_asym_id": "X"},
            }],
        }],
    }


def _network(url, payload):
    if isinstance(payload["query"], dict):
        return {"total_count": 1, "result_set": [{"identifier": "1ABC"}]}
    return {"data": {"entries": [_entry()]}}


def test_cohort_workflow_cache_replay_and_optional_filter(monkeypatch, tmp_path):
    args = _args(monkeypatch, tmp_path)
    config = DatasetBuildConfig(max_workers=1)
    client = RCSBClient(config)
    request = Mock(side_effect=_network)
    monkeypatch.setattr(client, "_post_json", request)
    _configure_dataset_filtered_csvs({DatasetKind.SOLUTION_NMR_MEMBRANE_PROTEINS: (args.solution_nmr_membrane_output,)})
    try:
        run_solution_nmr_membrane(args, client, config)
        assert request.call_count == 2
        rows = list(csv.DictReader(args.solution_nmr_membrane_output.open()))
        assert len(rows) == 1
        assert rows[0]["year"] == "2026"  # collection has no plotting cutoff
        assert rows[0]["deposited_model_count"] == "1"
        assert rows[0]["has_pdbtm_annotation"] == "false"
        manifest = json.loads((tmp_path / "cohort_manifest.json").read_text())
        assert manifest["filters"]["require_multiple_models"] is False
        assert manifest["candidate_entry_ids"] == ["1ABC"]
        for output in manifest["outputs"].values():
            assert _sha256_file(Path(output["path"])) == output["sha256"]
        for source in manifest["sources"]:
            assert _sha256_file(Path(source["response_path"])) == source["response_sha256"]
        previous_csv = args.solution_nmr_membrane_output.read_bytes()
        request.side_effect = AssertionError("cache replay must not contact network")
        run_solution_nmr_membrane(args, client, config)
        assert args.solution_nmr_membrane_output.read_bytes() == previous_csv
        assert len(list((tmp_path / "cache" / "manifests").glob("*.json"))) == 2

        args.solution_nmr_membrane_require_multiple_models = True
        run_solution_nmr_membrane(args, client, config)
        assert list(csv.DictReader(args.solution_nmr_membrane_output.open())) == []
        rejected = list(csv.DictReader((tmp_path / "cohort_filtered.csv").open()))
        assert rejected == [{"entry_id": "1ABC", "year": "2026", "reason": "not more than one deposited model"}]
    finally:
        _set_active_dataset_filtered_csvs(())
        _configure_dataset_filtered_csvs({})


def test_failed_fetch_does_not_publish_empty_success(monkeypatch, tmp_path):
    args = _args(monkeypatch, tmp_path)
    args.solution_nmr_membrane_output.write_text("previous result\n")
    client = RCSBClient(DatasetBuildConfig())
    monkeypatch.setattr(client, "_post_json", Mock(side_effect=RuntimeError("unavailable")))
    with pytest.raises(RuntimeError, match="unavailable"):
        run_solution_nmr_membrane(args, client, client.config)
    assert args.solution_nmr_membrane_output.read_text() == "previous result\n"
    assert not (tmp_path / "cohort_manifest.json").exists()
    _set_active_dataset_filtered_csvs(())


def test_membrane_kind_dispatch_and_model_filter_defaults(monkeypatch, tmp_path):
    args = _args(monkeypatch, tmp_path)
    assert not args.solution_nmr_membrane_require_multiple_models
    assert DatasetKind.SOLUTION_NMR_MEMBRANE_PROTEINS in parse_dataset_kinds("all")
    runner = Mock()
    monkeypatch.setattr(cli, "parse_args", lambda: args)
    monkeypatch.setattr(cli, "RCSBClient", Mock())
    monkeypatch.setattr(cli, "run_solution_nmr_membrane", runner)
    monkeypatch.setattr(cli, "_configure_dataset_warning_logs", Mock())
    monkeypatch.setattr(cli, "_configure_dataset_filtered_csvs", Mock())
    monkeypatch.setattr(cli, "_set_active_dataset_warning_logs", Mock())
    monkeypatch.setattr(cli, "run_xray_rmsd", Mock())
    cli.main()
    runner.assert_called_once()
    assert _args(monkeypatch, tmp_path, "--solution-nmr-membrane-require-multiple-models").solution_nmr_membrane_require_multiple_models
