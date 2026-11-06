"""Retrieve OPM positions and count deposited residues inside its membrane boundaries."""

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path

import requests

from src.dataset.api_cache import CachedJSONRequests
from src.dataset.cache import _atomic_write_json, _pdb_cache_entry_lock, _sha256_file, _utc_now_iso
from src.dataset.config import LOGGER
from src.dataset.coordinates import extract_model_pdb_texts
from src.dataset.downloads import download_pdb_if_needed
from src.dataset.opm_annotations import (
    collect_modeled_residues, count_membrane_residues, parse_opm_coordinates, summarize_entity_copies,
)
from src.dataset.pdb_normalization import load_pdb_polymer_metadata
from src.dataset.structures import load_cached_chain_id_map

OPM_API_URL = "https://opm-back.cc.lehigh.edu/opm-backend"
OPM_COORDINATES_URL = "https://opm-assets.storage.googleapis.com/pdb/{entry_id}.pdb"
# One request returns every record when the page exceeds the database size; checked on receipt.
OPM_PAGE_SIZE = 100000
OPM_API_SCHEMA = "opm-api-v1"


def fetch_opm_metadata(cache):
    """Return primary OPM records by uppercase PDB ID and OPM type names by ID."""
    structures = cache.post_json(f"{OPM_API_URL}/primary_structures", {"pageSize": OPM_PAGE_SIZE})
    types = cache.post_json(f"{OPM_API_URL}/types", {"pageSize": OPM_PAGE_SIZE})
    for body in (structures, types):
        if body.get("total_objects") != len(body.get("objects") or []):
            raise RuntimeError("OPM listing is incomplete")
    return (
        {record["pdbid"].upper(): record for record in structures["objects"]},
        {record["id"]: record["name"] for record in types["objects"]},
    )


def download_opm_coordinates(session, cache_dir, entry_id, timeout):
    """Keep a hash-verified OPM file; return its source record, or None if OPM has no file."""
    path = Path(cache_dir) / "pdb" / f"{entry_id}.pdb"
    meta_path = path.with_suffix(".json")
    with _pdb_cache_entry_lock(Path(cache_dir), entry_id):
        if path.is_file() and meta_path.is_file():
            try:
                metadata = json.loads(meta_path.read_text())
                if metadata["sha256"] == _sha256_file(path):
                    return {**metadata, "path": str(path.resolve())}
            except (OSError, ValueError, KeyError, TypeError):
                LOGGER.warning("Refreshing invalid OPM cache for %s", entry_id)
        url = OPM_COORDINATES_URL.format(entry_id=entry_id.lower())
        response = session.get(url, timeout=timeout)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes(response.content)
        temporary.replace(path)
        metadata = {"url": url, "retrieved_at": _utc_now_iso(), "sha256": _sha256_file(path)}
        _atomic_write_json(meta_path, metadata)
    return {**metadata, "path": str(path.resolve())}


def count_opm_entry(entry_id, chain_rows, client, cache_dir):
    """Keep unavailable files and failures visible instead of converting them to zero."""
    source = {"entry_id": entry_id}
    try:
        opm_source = download_opm_coordinates(client.session, cache_dir, entry_id, client.config.timeout_seconds)
        if opm_source is None:
            message = "OPM coordinate file not found"
            source["error"] = message
            return {"opm_status": "coordinates_unavailable", "opm_error": message}, {}, [], source
        source.update(opm_source)
        pdb_path = download_pdb_if_needed(client.session, client.config,
                                          client.solution_nmr_monomer_cache_dir, entry_id)
        models = extract_model_pdb_texts(pdb_path)
        if not models:
            raise ValueError("No coordinate model in PDB file")
        chain_map = load_cached_chain_id_map(pdb_path.parent, entry_id)
        deposited = collect_modeled_residues(models[0], chain_rows, load_pdb_polymer_metadata(pdb_path), chain_map)
        if not deposited:
            raise ValueError("No modeled polypeptide residues could be mapped to RCSB chains")
        half_thickness, positions = parse_opm_coordinates(Path(opm_source["path"]).read_text())
        source.update({"pdb_path": str(pdb_path.resolve()), "pdb_sha256": _sha256_file(pdb_path), "chain_id_map": chain_map})
        summary, chains, residues = count_membrane_residues(deposited, positions, half_thickness, chain_rows)
        if summary["opm_status"] != "counted":
            source["error"] = summary["opm_error"]
        return summary, chains, residues, source
    except (OSError, ValueError, KeyError, TypeError, requests.RequestException) as exc:
        error = f"{type(exc).__name__}: {exc}"
        LOGGER.warning("OPM %s: %s", entry_id, error)
        source["error"] = error
        return {"opm_status": "error", "opm_error": error}, {}, [], source


def annotate_membrane_cohort(result, args, client, config):
    """Enrich OPM-annotated cohort rows and return raw-file and API provenance."""
    cache_dir = Path(args.opm_cache_dir)

    def get_json(url, params):
        response = client.session.get(url, params=params, timeout=client.config.timeout_seconds)
        response.raise_for_status()
        return response.json()

    cache = CachedJSONRequests(
        get_json, cache_dir / "api",
        validation_hours=args.solution_nmr_membrane_cache_validation_hours,
        schema_version=OPM_API_SCHEMA,
    )
    annotated = [row for row in result.entry_rows if row["has_opm_annotation"]]
    for row in result.entry_rows:
        if not row["has_opm_annotation"]:
            row.update(opm_status="no_opm_annotation", opm_error="")
    if not annotated:
        return [], cache.sources
    records, type_names = fetch_opm_metadata(cache)
    chains = defaultdict(list)
    for row in result.chain_rows:
        chains[row["entry_id"]].append(row)
    sources = []
    with ThreadPoolExecutor(max_workers=config.max_workers) as pool:
        futures = {
            pool.submit(count_opm_entry, row["entry_id"], chains[row["entry_id"]], client, cache_dir): row
            for row in annotated
        }
        for index, future in enumerate(as_completed(futures), 1):
            row = futures[future]
            summary, chain_summaries, residues, source = future.result()
            record = records.get(row["entry_id"], {})
            row.update(summary)
            row.update(
                opm_type=type_names.get(record.get("type_id")),
                opm_membrane_name=record.get("membrane_name_cache"),
                opm_reported_thickness_angstrom=record.get("thickness"),
                opm_tilt_degrees=record.get("tilt"),
                opm_transfer_energy_kcal_per_mol=record.get("gibbs"),
            )
            for chain in chains[row["entry_id"]]:
                chain.update(chain_summaries.get(chain["asym_id"], {}))
            row.update(summarize_entity_copies(chains[row["entry_id"]]))
            result.opm_residue_rows.extend(residues)
            sources.append(source)
            LOGGER.info("OPM positions: %d/%d %s %s", index, len(futures), row["entry_id"], summary["opm_status"])
    # Stable entry sorting keeps the original coordinate order within each entry.
    result.opm_residue_rows.sort(key=lambda row: row["entry_id"])
    return sorted(sources, key=lambda row: row["entry_id"]), cache.sources
