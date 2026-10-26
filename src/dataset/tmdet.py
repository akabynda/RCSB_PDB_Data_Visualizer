"""Run TmDet on first-model coordinates and retain verifiable raw results."""

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET

import requests

from src.dataset.cache import _atomic_write_json, _pdb_cache_entry_lock, _sha256_file, _utc_now_iso
from src.dataset.config import LOGGER
from src.dataset.coordinates import extract_model_pdb_texts
from src.dataset.downloads import download_pdb_if_needed
from src.dataset.pdb_normalization import load_pdb_polymer_metadata, pdb_atom_field_offset
from src.dataset.structures import load_cached_chain_id_map
from src.dataset.tmdet_annotations import (
    collect_modeled_residues, parse_tmdet_annotations, summarize_entity_copies,
)
from src.dataset.tmdet_install import ensure_tmdet_executable

TMDET_CACHE_SCHEMA = "first-model-tmdet-v1"
TMDET_ARGUMENTS = ("-e", "tmdet.env", "-pi", "input.pdb", "-x", "result.xml",
                   "-po", "transformed.cif", "-m", "0", "-nc")


def prepare_ccd(session, cache_dir, model_text, timeout):
    """Fetch only required CCD components, plus TmDet's real ZZZ check record."""
    names = {"ZZZ"}
    for line in model_text.splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            names.add(line[17:20 + pdb_atom_field_offset(line)].strip())
    sources = {}
    for name in sorted(names):
        if not re.fullmatch(r"[A-Z0-9]{1,5}", name):
            raise ValueError(f"Unsupported CCD component ID: {name!r}")
        path = cache_dir / name[0]
        if len(name) > 1:
            path /= name[1]
        path /= f"{name}.cif"
        meta_path = path.with_suffix(".json")
        with _pdb_cache_entry_lock(cache_dir, name):
            try:
                metadata = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
            except (OSError, ValueError):
                metadata = {}
            if not isinstance(metadata, dict):
                metadata = {}
            if not path.is_file() or metadata.get("sha256") != _sha256_file(path):
                url = f"https://files.rcsb.org/ligands/download/{name}.cif"
                response = session.get(url, timeout=timeout)
                response.raise_for_status()
                if not response.text.lstrip().startswith(f"data_{name}"):
                    raise ValueError(f"Invalid CCD response for {name}")
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(".tmp")
                temporary.write_bytes(response.content)
                temporary.replace(path)
                metadata = {"url": url, "retrieved_at": _utc_now_iso(), "sha256": _sha256_file(path)}
                _atomic_write_json(meta_path, metadata)
            sources[name] = {**metadata, "path": str(path.resolve())}
    return sources


def _valid_result(path, inputs):
    if not path.is_file():
        return False
    try:
        manifest = json.loads(path.read_text())
        return (manifest["inputs"] == inputs and manifest["outputs"]
                and all((path.parent / name).is_file()
                        and _sha256_file(path.parent / name) == checksum
                        for name, checksum in manifest["outputs"].items()))
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


def run_tmdet_entry(entry_id, chain_rows, client, executable, binary_sha256, cache_dir, timeout):
    """Cache successes by inputs; retain failed attempts without converting them to zero."""
    cache_dir = Path(cache_dir).resolve()
    entry_dir = cache_dir / entry_id
    entry_dir.mkdir(parents=True, exist_ok=True)
    residues, directory = [], entry_dir
    source = {"entry_id": entry_id}
    try:
        pdb_path = download_pdb_if_needed(client.session, client.config,
                                          client.solution_nmr_monomer_cache_dir, entry_id)
        models = extract_model_pdb_texts(pdb_path)
        if not models:
            raise ValueError("No coordinate model in PDB file")
        if len(models) <= 1:
            raise ValueError("Coordinate file does not contain more than one model")
        model_text = models[0]
        chain_map = load_cached_chain_id_map(pdb_path.parent, entry_id)
        residues = collect_modeled_residues(model_text, chain_rows, load_pdb_polymer_metadata(pdb_path), chain_map)
        if not residues:
            raise ValueError("No modeled polypeptide residues could be mapped to RCSB chains")
        ccd = prepare_ccd(client.session, cache_dir / "ccd", model_text, client.config.timeout_seconds)
        inputs = {
            "schema": TMDET_CACHE_SCHEMA, "model_index": 0,
            "pdb_sha256": _sha256_file(pdb_path),
            "first_model_sha256": sha256(model_text.encode()).hexdigest(),
            "executable_sha256": binary_sha256, "arguments": list(TMDET_ARGUMENTS),
            "ccd_sha256": {name: item["sha256"] for name, item in ccd.items()},
        }
        key = sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
        target = entry_dir / key
        source.update({"inputs": inputs, "pdb_path": str(pdb_path.resolve()), "chain_id_map": chain_map,
                       "coordinate_model_count": len(models), "ccd": ccd, "executable": str(executable)})
        with _pdb_cache_entry_lock(cache_dir, entry_id):
            manifest_path = target / "manifest.json"
            if _valid_result(manifest_path, inputs):
                directory = target
                parsed = parse_tmdet_annotations((target / "result.xml").read_text(), residues, chain_rows)
            else:
                directory = Path(tempfile.mkdtemp(prefix=f"{key[:12]}-attempt-", dir=entry_dir))
                (directory / "input.pdb").write_text(model_text)
                (directory / "tmdet.env").write_text(
                    f"TMDET_CC_DIR={cache_dir / 'ccd'}\nTMDET_LOG_FILE=tmdet.log\nTMDET_CACHE_ROOT=cache\n")
                command = [str(executable), *TMDET_ARGUMENTS]
                _atomic_write_json(directory / "request.json", {**source, "command": command, "started_at": _utc_now_iso()})
                environment = {key: value for key, value in os.environ.items() if not key.startswith("TMDET_")}
                with (directory / "stdout.log").open("w") as stdout, (directory / "stderr.log").open("w") as stderr:
                    subprocess.run(command, cwd=directory, env=environment, stdout=stdout, stderr=stderr,
                                   timeout=timeout, check=True)
                parsed = parse_tmdet_annotations((directory / "result.xml").read_text(), residues, chain_rows)
                outputs = {str(path.relative_to(directory)): _sha256_file(path)
                           for path in directory.rglob("*") if path.is_file()}
                _atomic_write_json(directory / "manifest.json", {
                    "inputs": inputs, "outputs": outputs, "completed_at": _utc_now_iso(),
                })
                if target.exists():
                    # Retain the corrupt/incompatible bundle for inspection.
                    target.rename(entry_dir / f"{key}-invalid-{directory.name}")
                directory.rename(target)
                directory = target
        summary, chains, regions, residues = parsed
        source["manifest_path"] = str(directory / "manifest.json")
        source["manifest_sha256"] = _sha256_file(directory / "manifest.json")
    except (OSError, ValueError, KeyError, TypeError, ET.ParseError,
            subprocess.SubprocessError, requests.RequestException) as exc:
        # Network, native-program and XML failures are visible in the published table.
        error = f"{type(exc).__name__}: {exc}"
        LOGGER.warning("TmDet %s: %s", entry_id, error)
        summary = {"tmdet_status": "error", "tmdet_error": error,
                   "tmdet_modeled_residue_count": len(residues) if residues else None}
        chains, regions = {}, []
        for row in residues:
            row["assignment_status"] = "error"
        source["error"] = error
        _atomic_write_json(directory / "failure.json", {**source, "failed_at": _utc_now_iso()})
    summary.update({"tmdet_model_index": 0, "tmdet_result_directory": str(directory)})
    return summary, chains, regions, residues, source


def annotate_membrane_cohort(result, args, client, config):
    """Enrich the existing cohort tables and return raw-run provenance."""
    timeout = args.tmdet_timeout_seconds
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("TmDet timeout must be positive and finite")
    if not result.entry_rows:
        return []
    executable = ensure_tmdet_executable(args.tmdet_executable, args.tmdet_install_dir)
    binary_hash = _sha256_file(executable)
    chains = defaultdict(list)
    for row in result.chain_rows:
        chains[row["entry_id"]].append(row)
    sources = []
    with ThreadPoolExecutor(max_workers=config.max_workers) as pool:
        futures = {
            pool.submit(run_tmdet_entry, row["entry_id"], chains[row["entry_id"]], client,
                        executable, binary_hash, args.tmdet_cache_dir, timeout): row
            for row in result.entry_rows
        }
        for index, future in enumerate(as_completed(futures), 1):
            row = futures[future]
            summary, chain_summaries, regions, residues, source = future.result()
            row.update(summary)
            for chain in chains[row["entry_id"]]:
                chain.update(chain_summaries.get(chain["asym_id"], {
                    "tmdet_status": "error", "tmdet_error": summary.get("tmdet_error", ""),
                }))
            row.update(summarize_entity_copies(row["tmdet_status"], chains[row["entry_id"]]))
            result.tmdet_region_rows.extend(regions)
            result.tmdet_residue_rows.extend(residues)
            sources.append(source)
            LOGGER.info("TmDet first model: %d/%d %s %s", index, len(futures), row["entry_id"], summary["tmdet_status"])
    result.tmdet_region_rows.sort(key=lambda row: (row["entry_id"], row["pdb_chain_id"], row["region_index"]))
    # Stable entry sorting keeps the original coordinate order within each entry.
    result.tmdet_residue_rows.sort(key=lambda row: row["entry_id"])
    return sorted(sources, key=lambda row: row["entry_id"])
