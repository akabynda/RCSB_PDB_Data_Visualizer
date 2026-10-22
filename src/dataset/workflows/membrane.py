"""Collect, audit, and publish the solution-NMR membrane cohort."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
import platform
import shlex
import subprocess
import sys

from src.dataset.api_cache import CachedJSONRequests
from src.dataset.builders.membrane import SolutionNMRMembraneBuilder
from src.dataset.cache import _atomic_write_json, _sha256_file, _utc_now_iso
from src.dataset.client import RCSBClient
from src.dataset.config import LOGGER, MEMBRANE_ANNOTATION_TYPES, DatasetBuildConfig
from src.dataset.io.membrane import write_solution_nmr_membrane_csvs
from src.dataset.reporting import _record_filtered_structure, _set_active_dataset_filtered_csvs
from src.dataset.utils import chunked, collect_batch_results
from src.dataset.tmdet import annotate_membrane_cohort


def _code_provenance() -> dict:
    """Identify the checked-out revision and exact source files used locally."""
    root = Path(__file__).resolve().parents[3]
    hashes = {
        str(path.relative_to(root)): _sha256_file(path)
        for path in sorted((root / "src").rglob("*.py"))
    }
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        revision, status = None, None
    return {"git_revision": revision, "git_status": status, "source_sha256": hashes}


def run_solution_nmr_membrane(
    args: argparse.Namespace, client: RCSBClient, config: DatasetBuildConfig
) -> None:
    """Collect the cohort and annotate its first coordinate models with TmDet."""
    if config.graphql_batch_size < 1 or config.max_workers < 1:
        raise ValueError("Membrane batch size and worker count must be positive")
    output_path = Path(args.solution_nmr_membrane_output)
    _set_active_dataset_filtered_csvs((output_path,))
    cache = CachedJSONRequests(
        client._post_json,
        Path(args.solution_nmr_membrane_cache_dir),
        validation_hours=args.solution_nmr_membrane_cache_validation_hours,
    )
    started_at = _utc_now_iso()
    entry_ids = client.fetch_solution_nmr_membrane_entry_ids(cache)
    batches = collect_batch_results(
        list(chunked(entry_ids, config.graphql_batch_size)),
        max_workers=config.max_workers,
        fetch_fn=lambda batch: client.fetch_membrane_entries_for_ids(batch, cache),
        progress_label="SOLUTION NMR membrane metadata",
    )
    result = SolutionNMRMembraneBuilder().build_from_entries(
        [entry for batch in batches for entry in batch],
        require_multiple_models=args.solution_nmr_membrane_require_multiple_models,
    )
    for excluded in result.excluded:
        _record_filtered_structure(excluded["entry_id"], excluded["reason"], excluded["year"])
    tmdet_sources = annotate_membrane_cohort(result, args, client, config)
    paths = write_solution_nmr_membrane_csvs(result, output_path)
    rows = result.entry_rows
    summary = {
        "entries": len(rows),
        "protein_entities": len(result.entity_rows),
        "protein_chains": len(result.chain_rows),
        "pdbtm_annotation_present": sum(row["has_pdbtm_annotation"] for row in rows),
        "pdbtm_segment_status": dict(Counter(row["pdbtm_segment_status"] for row in rows)),
        "model_count_groups": dict(Counter(
            "unknown" if row["deposited_model_count"] is None
            else "one" if row["deposited_model_count"] == 1 else "multiple"
            for row in rows
        )),
        "missing_deposition_year": sum(row["year"] is None for row in rows),
        "duplicate_candidate_pairs": len(result.duplicate_candidate_rows),
        "tmdet_status": dict(Counter(row["tmdet_status"] for row in rows)),
        "tmdet_region_residue_counts": dict(Counter(
            row["region_type"] for row in result.tmdet_residue_rows if row["region_type"] is not None
        )),
    }
    manifest = {
        "schema_version": "solution-nmr-membrane-v2",
        "started_at": started_at,
        "completed_at": _utc_now_iso(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "working_directory": str(Path.cwd()),
        "python": platform.python_version(),
        "libraries": {name: version(name) for name in (
            "biopython", "MDAnalysis", "matplotlib", "numpy", "pandas", "requests",
        )},
        "code": _code_provenance(),
        "config": asdict(config),
        "filters": {
            "polypeptide_chain": "at least one polypeptide(L) or polypeptide(D) chain",
            "experimental_methods": ["SOLUTION NMR"],
            "membrane_field": "rcsb_polymer_entity_annotation.type",
            "membrane_any_of": list(MEMBRANE_ANNOTATION_TYPES),
            "require_multiple_models": args.solution_nmr_membrane_require_multiple_models,
            "deposition_year_limit": None,
        },
        "api_cache": {
            "directory": str(cache.cache_dir.resolve()),
            "validation_hours": cache.validation_hours,
        },
        "candidate_entry_ids": sorted(entry_ids),
        "selected_entry_ids": [row["entry_id"] for row in rows],
        "filter_counts": result.filter_counts,
        "excluded": result.excluded,
        "summary": summary,
        "sources": cache.sources,
        "tmdet": {
            "model_index": 0, "input": "first PDB coordinate model, no core crop",
            "count_unit": "modeled polypeptide residues across all protein chains in each PDB entry",
            "tm_region_codes": ["H", "B"], "preserved_region_codes": "all",
            "cache_directory": str(Path(args.tmdet_cache_dir).resolve()),
            "timeout_seconds": args.tmdet_timeout_seconds,
            "sources": tmdet_sources,
        },
        "outputs": {
            name: {"path": str(path.resolve()), "sha256": _sha256_file(path)}
            for name, path in paths.items()
        },
    }
    manifest_path = output_path.with_name(f"{output_path.stem}_manifest.json")
    # Publish the manifest last: its hashes detect incomplete multi-file updates.
    _atomic_write_json(manifest_path, manifest)
    manifest_hash = _sha256_file(manifest_path)
    _atomic_write_json(cache.cache_dir / "manifests" / f"{manifest_hash}.json", manifest)
    LOGGER.info("Membrane cohort summary: %s", summary)
    LOGGER.info("Saved membrane CSVs to %s; manifest %s", output_path, manifest_path)
    if summary["tmdet_status"].get("error", 0):
        raise RuntimeError("Some TmDet calculations failed; rows and raw logs were saved. Rerun to retry failed entries.")
