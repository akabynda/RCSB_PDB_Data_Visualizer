from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
import warnings
import requests
import numpy as np
from Bio import BiopythonWarning
from Bio.PDB import MMCIFParser, PDBIO, PDBParser, Select
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.SeqUtils import seq1
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar
from MDAnalysis.analysis.align import rotation_matrix as mda_rotation_matrix
from MDAnalysis.analysis.rms import rmsd as mda_rmsd

LOGGER = logging.getLogger(__name__)
T = TypeVar("T")
_MISSING = object()
PDB_CHAIN_ID_POOL = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


class ChainSubsetSelect(Select):
    def __init__(self, chain_object_ids: set[int]) -> None:
        """Store the selected Biopython chain object identities."""
        self.chain_object_ids = chain_object_ids

    def accept_chain(self, chain: Any) -> bool:
        """Return whether a chain should be written by PDBIO."""
        return id(chain) in self.chain_object_ids

SOLUTION_NMR_METHOD = "SOLUTION NMR"
PROTEIN_MONOMER_ENTITY_TYPES: frozenset[str] = frozenset(
    {"polypeptide(L)", "polypeptide(D)"}
)
PROTEIN_POLYMER_TYPE = "Protein"
SEQUENCE_IDENTITY_AGGREGATION_METHOD = "sequence_identity"
DEFAULT_PDB_CACHE_DIR = Path("data/pdb_cache")
LOCAL_STRIDE_CANDIDATE = Path("/tmp/stride_src/src/stride")
STRIDE_STATE_CODES: tuple[str, ...] = ("H", "G", "I", "E", "B", "T", "C")
STRIDE_CORE_STATE_CODES: frozenset[str] = frozenset({"H", "G", "I", "E", "B"})
DEFAULT_MAX_WORKERS = max(1, os.cpu_count() or 1)
print(f"Using up to {DEFAULT_MAX_WORKERS} worker threads for concurrent tasks")


class ExperimentalMethod(Enum):
    X_RAY = ("X-ray", ("X-RAY DIFFRACTION",))
    CRYO_EM = ("cryo-EM", ("ELECTRON MICROSCOPY",))
    NMR = ("NMR", ("SOLUTION NMR", "SOLID-STATE NMR"))

    @property
    def label(self) -> str:
        """Return the display label for the experimental method."""
        return self.value[0]

    @property
    def query_values(self) -> tuple[str, ...]:
        """Return the RCSB method values used in search queries."""
        return self.value[1]


class DatasetKind(str, Enum):
    METHOD_COUNTS = "method_counts"
    MEMBRANE_PROTEIN_COUNTS = "membrane_protein_counts"
    SOLUTION_NMR_PROGRAM_COUNTS = "solution_nmr_program_counts"
    SOLUTION_NMR_MONOMER_PROGRAM_CLUSTERS = "solution_nmr_monomer_program_clusters"
    SOLUTION_NMR_WEIGHTS = "solution_nmr_weights"
    SOLUTION_NMR_MONOMER_STRIDE_MODELED_FIRST_MODEL = (
        "solution_nmr_monomer_stride_modeled_first_model"
    )
    SOLUTION_NMR_MONOMER_PRECISION_STRIDE_MODELED_FIRST_MODEL = (
        "solution_nmr_monomer_precision_stride_modeled_first_model"
    )
    SOLUTION_NMR_MONOMER_QUALITY = "solution_nmr_monomer_quality"
    SOLUTION_NMR_MONOMER_XRAY_HOMOLOGS = "solution_nmr_monomer_xray_homologs"
    SOLUTION_NMR_MONOMER_XRAY_HOMOLOGS_HISTORICAL = (
        "solution_nmr_monomer_xray_homologs_historical"
    )
    SOLUTION_NMR_MONOMER_XRAY_RMSD = "solution_nmr_monomer_xray_rmsd"
    SOLUTION_NMR_MONOMER_XRAY_RMSD_HISTORICAL = (
        "solution_nmr_monomer_xray_rmsd_historical"
    )
    SOLUTION_NMR_MONOMER_XRAY_RMSD_EXTREMES = (
        "solution_nmr_monomer_xray_rmsd_extremes"
    )
    SOLUTION_NMR_MONOMER_XRAY_RMSD_EXTREMES_HISTORICAL = (
        "solution_nmr_monomer_xray_rmsd_extremes_historical"
    )


@dataclass(frozen=True)
class CollectorConfig:
    search_url: str = "https://search.rcsb.org/rcsbsearch/v2/query"
    graphql_url: str = "https://data.rcsb.org/graphql"
    page_size: int = 10000
    graphql_batch_size: int = 300
    max_workers: int = DEFAULT_MAX_WORKERS
    timeout_seconds: int = 60
    retries: int = 4
    backoff_seconds: float = 1.3


@dataclass(frozen=True)
class CAResidueRecord:
    resid: int
    identity: str
    is_standard_atom: bool


@dataclass(frozen=True)
class YearlyCountRecord:
    year: int
    method: str
    count: int


@dataclass(frozen=True)
class MembraneYearlyCountRecord:
    year: int
    count: int


@dataclass(frozen=True)
class SolutionNMRProgramYearlyCountRecord:
    year: int
    program: str
    count: int


@dataclass(frozen=True)
class SolutionNMRMonomerProgramClusterAssignmentRecord:
    entry_id: str
    year: int
    cluster_id: str
    cluster_name: str
    has_program_text: bool
    program_text: str


@dataclass(frozen=True)
class SolutionNMRMonomerProgramClusterSummaryRecord:
    year: int
    cluster_id: str
    cluster_name: str
    structure_count: int
    avg_ramachandran_outliers_percent: float | None
    avg_sidechain_outliers_percent: float | None
    avg_clashscore: float | None


@dataclass(frozen=True)
class SolutionNMRMonomerProgramClusterYearlySummaryRecord:
    year: int
    structure_count: int
    avg_ramachandran_outliers_percent: float | None
    avg_sidechain_outliers_percent: float | None
    avg_clashscore: float | None


@dataclass(frozen=True)
class SolutionNMRMonomerProgramClusterTotalRecord:
    cluster_name: str
    structure_count: int
    avg_ramachandran_outliers_percent: float | None
    avg_sidechain_outliers_percent: float | None
    avg_clashscore: float | None


@dataclass(frozen=True)
class SolutionNMRWeightRecord:
    entry_id: str
    year: int
    molecular_weight_kda: float


@dataclass(frozen=True)
class SolutionNMRMonomerStrideModeledFirstModelRecord:
    entry_id: str
    year: int
    chain_id: str
    modeled_start_seq_id: int
    modeled_end_seq_id: int
    modeled_sequence_length: int
    secondary_structure_percent: float
    helix_fraction: float
    sheet_fraction: float
    stride_alpha_helix_fraction: float
    stride_3_10_helix_fraction: float
    stride_pi_helix_fraction: float
    stride_beta_strand_fraction: float
    stride_isolated_beta_bridge_fraction: float
    stride_turn_fraction: float
    stride_coil_fraction: float
    stride_secondary_structure_percent: float


@dataclass(frozen=True)
class SolutionNMRMonomerModeledFirstModelSeedRecord:
    entry_id: str
    year: int
    chain_id: str


@dataclass(frozen=True)
class SolutionNMRMonomerPrecisionRecord:
    entry_id: str
    year: int
    chain_id: str
    core_start_seq_id: int
    core_end_seq_id: int
    n_models: int
    n_ca_core_used: int
    n_ca_core_raw: int
    mean_rmsd_angstrom: float


@dataclass(frozen=True)
class SolutionNMRMonomerQualityRecord:
    entry_id: str
    year: int
    clashscore: float
    ramachandran_outliers_percent: float
    sidechain_outliers_percent: float


@dataclass(frozen=True)
class SolutionNMRMonomerXrayHomologRecord:
    entry_id: str
    year: int
    sequence_identity_percent: int
    nmr_core_start_seq_id: int | None
    nmr_core_end_seq_id: int | None
    nmr_query_sequence_length: int
    xray_homolog_entry_ids: tuple[str, ...]
    xray_homolog_entity_ids: tuple[str, ...]
    has_xray_homolog: bool


@dataclass(frozen=True)
class SolutionNMRMonomerXrayHomologSeedRecord:
    entry_id: str
    year: int
    chain_id: str


@dataclass(frozen=True)
class XrayEntityGroupMappingRecord:
    polymer_entity_id: str
    entry_id: str
    chain_ids: tuple[str, ...]
    group_id: str


@dataclass(frozen=True)
class SolutionNMRMonomerXrayRmsdRecord:
    entry_id: str
    year: int
    sequence_identity_percent: int
    nmr_chain_id: str
    nmr_core_start_seq_id: int | None
    nmr_core_end_seq_id: int | None
    nmr_query_sequence_length: int
    xray_homolog_entity_id: str
    xray_homolog_count: int
    xray_entry_id: str
    xray_chain_id: str
    xray_core_start_seq_id: int | None
    xray_core_end_seq_id: int | None
    xray_resolution_angstrom: float
    n_common_ca: int
    rmsd_ca_angstrom: float


@dataclass(frozen=True)
class SolutionNMRMonomerXrayRmsdExtremesRecord:
    entry_id: str
    year: int
    sequence_identity_percent: int
    nmr_chain_id: str
    nmr_core_start_seq_id: int | None
    nmr_core_end_seq_id: int | None
    nmr_query_sequence_length: int
    xray_homolog_count: int
    successful_xray_homolog_count: int
    best_xray_homolog_entity_id: str
    best_xray_entry_id: str
    best_xray_chain_id: str
    best_xray_resolution_angstrom: float
    best_xray_core_start_seq_id: int | None
    best_xray_core_end_seq_id: int | None
    best_n_common_ca: int
    best_rmsd_ca_angstrom: float
    worst_xray_homolog_entity_id: str
    worst_xray_entry_id: str
    worst_xray_chain_id: str
    worst_xray_resolution_angstrom: float
    worst_xray_core_start_seq_id: int | None
    worst_xray_core_end_seq_id: int | None
    worst_n_common_ca: int
    worst_rmsd_ca_angstrom: float
    rmsd_delta_angstrom: float


@dataclass(frozen=True)
class XrayPolymerEntityCandidateRecord:
    polymer_entity_id: str
    entry_id: str
    chain_ids: tuple[str, ...]
    resolution_angstrom: float


def chunked(items: list[str], size: int) -> Iterator[list[str]]:
    """Yield fixed-size batches from a list of item identifiers."""
    for i in range(0, len(items), size):
        yield items[i : i + size]


def extract_year(deposit_date: str | None) -> int | None:
    """Extract a deposit year from an RCSB date string."""
    if not deposit_date:
        return None
    try:
        return int(deposit_date[:4])
    except (TypeError, ValueError):
        return None


def parse_rcsb_datetime(value: str | None) -> datetime | None:
    """Parse an RCSB date or datetime string into a datetime object."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def collect_batch_results(
    batches: list[list[str]],
    max_workers: int,
    fetch_fn: Callable[[list[str]], T],
    progress_label: str,
) -> list[T]:
    """Run batched collection work in parallel and collect successful records."""
    if not batches:
        return []
    results: list[T] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(fetch_fn, batch): idx
            for idx, batch in enumerate(batches, start=1)
        }
        for future in as_completed(future_map):
            results.append(future.result())
            batch_idx = future_map[future]
            LOGGER.info(
                "%s: processed batch %d/%d", progress_label, batch_idx, len(batches)
            )
    return results


def fetch_solution_nmr_entry_ids(client: RCSBClient, log_label: str) -> list[str]:
    """Fetch all entry IDs assigned to the SOLUTION NMR method."""
    entry_ids = sorted(
        set(
            client.fetch_entry_ids_for_method(
                method_label=SOLUTION_NMR_METHOD,
                query_value=SOLUTION_NMR_METHOD,
            )
        )
    )
    LOGGER.info("%s: total unique IDs collected: %d", log_label, len(entry_ids))
    return entry_ids


def resolve_stride_executable(explicit_value: str) -> str | None:
    """Resolve the STRIDE executable path from arguments, PATH, or local builds."""
    explicit_path = explicit_value.strip()
    if explicit_path:
        path = Path(explicit_path).expanduser()
        if path.exists():
            return str(path)
        return None

    resolved = shutil.which("stride")
    if resolved:
        return resolved

    if LOCAL_STRIDE_CANDIDATE.exists():
        return str(LOCAL_STRIDE_CANDIDATE.resolve())

    return None


def download_pdb_if_needed(
    session: requests.Session,
    config: CollectorConfig,
    cache_dir: Path,
    entry_id: str,
) -> Path:
    """Download and cache a PDB-format coordinate file if it is missing."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{entry_id}.pdb"
    if path.exists() and path.stat().st_size > 0:
        return path

    url = f"https://files.rcsb.org/download/{entry_id}.pdb"
    last_error: Exception | None = None
    saw_not_found = False
    for attempt in range(1, config.retries + 1):
        try:
            response = session.get(url, timeout=config.timeout_seconds)
            if response.status_code == 404:
                saw_not_found = True
                last_error = requests.HTTPError(
                    f"404 Client Error: Not Found for url: {url}"
                )
                break
            response.raise_for_status()
            path.write_text(response.text, encoding="utf-8")
            return path
        except (requests.RequestException, OSError) as exc:
            last_error = exc
            wait_seconds = config.backoff_seconds * attempt
            if attempt < config.retries:
                time.sleep(wait_seconds)
    if saw_not_found:
        cif_path = cache_dir / f"{entry_id}.cif"
        cif_url = f"https://files.rcsb.org/download/{entry_id}.cif"
        LOGGER.info(
            "PDB file is unavailable for %s; trying mmCIF fallback",
            entry_id,
        )
        for attempt in range(1, config.retries + 1):
            try:
                response = session.get(cif_url, timeout=config.timeout_seconds)
                response.raise_for_status()
                cif_path.write_text(response.text, encoding="utf-8")
                structure = parse_mmcif_structure(entry_id, cif_path)
                chain_id_map = _coerce_structure_chain_ids_for_pdbio(structure)
                io = PDBIO()
                io.set_structure(structure)
                io.save(str(path))
                if path.exists() and path.stat().st_size > 0:
                    if chain_id_map:
                        map_path = cache_dir / f"{entry_id}.chain_map.csv"
                        write_csv_rows(
                            output_path=map_path,
                            header=["original_chain_id", "mapped_chain_id"],
                            rows=sorted(chain_id_map.items()),
                        )
                    LOGGER.info(
                        "Converted mmCIF fallback to cached PDB for %s",
                        entry_id,
                    )
                    return path
            except Exception as exc:
                last_error = exc
                wait_seconds = config.backoff_seconds * attempt
                if attempt < config.retries:
                    time.sleep(wait_seconds)
    raise RuntimeError(f"Failed to download {entry_id}: {last_error}")


def parse_mmcif_structure(entry_id: str, cif_path: Path) -> Any:
    """Parse an mmCIF coordinate file into a Biopython structure."""
    parser = MMCIFParser(QUIET=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PDBConstructionWarning)
        return parser.get_structure(entry_id, str(cif_path))


def parse_pdb_structure(entry_id: str, pdb_path: str | Path) -> Any:
    """Parse a PDB coordinate file into a Biopython structure."""
    parser = PDBParser(QUIET=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PDBConstructionWarning)
        return parser.get_structure(entry_id, str(pdb_path))


def _coerce_structure_chain_ids_for_pdbio(structure: Any) -> dict[str, str]:
    """Assign temporary PDB-compatible chain IDs before PDBIO output."""
    original_chain_ids: list[str] = []
    seen_original_ids: set[str] = set()
    for model in structure:
        for chain in model:
            chain_id = str(chain.id)
            if chain_id in seen_original_ids:
                continue
            seen_original_ids.add(chain_id)
            original_chain_ids.append(chain_id)

    chain_id_map: dict[str, str] = {}
    used_ids: set[str] = set()
    for original_id in original_chain_ids:
        if len(original_id) == 1 and original_id not in used_ids:
            mapped_id = original_id
        elif original_id and original_id[0] not in used_ids:
            mapped_id = original_id[0]
        else:
            mapped_id = next(
                (
                    candidate
                    for candidate in PDB_CHAIN_ID_POOL
                    if candidate not in used_ids
                ),
                None,
            )
            if mapped_id is None:
                raise RuntimeError("Too many chains to convert mmCIF to PDB")
        chain_id_map[original_id] = mapped_id
        used_ids.add(mapped_id)

    _apply_chain_id_map_without_transient_conflicts(structure, chain_id_map)
    return {
        original_id: mapped_id
        for original_id, mapped_id in chain_id_map.items()
        if original_id != mapped_id
    }


def _coerce_selected_structure_chain_ids_for_pdbio(
    structure: Any,
    selected_chain_ids: set[str],
) -> tuple[dict[str, str], set[int]]:
    """Assign temporary PDB-compatible IDs only for selected chains."""
    selected_chain_object_ids: set[int] = set()
    existing_chain_ids: set[str] = set()
    for model in structure:
        for chain in model:
            chain_id = str(chain.id)
            if chain_id in selected_chain_ids:
                existing_chain_ids.add(chain_id)
                selected_chain_object_ids.add(id(chain))
    chain_id_map: dict[str, str] = {}
    used_ids: set[str] = set()
    for original_id in sorted(existing_chain_ids):
        if len(original_id) == 1 and original_id not in used_ids:
            mapped_id = original_id
        elif original_id and original_id[0] not in used_ids:
            mapped_id = original_id[0]
        else:
            mapped_id = next(
                (
                    candidate
                    for candidate in PDB_CHAIN_ID_POOL
                    if candidate not in used_ids
                ),
                None,
            )
            if mapped_id is None:
                raise RuntimeError("Too many selected chains to convert mmCIF to PDB")
        chain_id_map[original_id] = mapped_id
        used_ids.add(mapped_id)

    _apply_chain_id_map_without_transient_conflicts(structure, chain_id_map)
    return chain_id_map, selected_chain_object_ids


def _apply_chain_id_map_without_transient_conflicts(
    structure: Any,
    chain_id_map: dict[str, str],
) -> None:
    """Apply a chain-ID mapping without creating temporary ID collisions."""
    chain_objects: list[tuple[Any, str]] = []
    for model in structure:
        for chain in model:
            original_id = str(chain.id)
            if original_id in chain_id_map:
                chain_objects.append((chain, original_id))

    temporary_ids: dict[int, str] = {}
    for index, (chain, _) in enumerate(chain_objects):
        temporary_id = f"__tmp_chain_{index}__"
        temporary_ids[id(chain)] = temporary_id
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", BiopythonWarning)
            chain.id = temporary_id

    for chain, original_id in chain_objects:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", BiopythonWarning)
            chain.id = chain_id_map[original_id]


def load_cached_chain_id_map(cache_dir: Path, entry_id: str) -> dict[str, str]:
    """Load the cached original-to-PDB chain ID mapping for an entry."""
    map_path = cache_dir / f"{entry_id}.chain_map.csv"
    if not map_path.exists():
        return {}
    chain_id_map: dict[str, str] = {}
    with map_path.open("r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            original = str(row.get("original_chain_id") or "")
            mapped = str(row.get("mapped_chain_id") or "")
            if original and mapped:
                chain_id_map[original] = mapped
    return chain_id_map


def _chain_subset_cache_stem(entry_id: str, chain_ids: Sequence[str]) -> str:
    """Build a deterministic cache key for a chain subset file."""
    normalized = ",".join(sorted(str(chain_id) for chain_id in chain_ids))
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]
    return f"{entry_id}.chains_{digest}"


def download_pdb_chain_subset_if_needed(
    session: requests.Session,
    config: CollectorConfig,
    cache_dir: Path,
    entry_id: str,
    chain_ids: Sequence[str],
) -> tuple[Path, dict[str, str]]:
    """Create or reuse a cached PDB file containing only selected chains."""
    selected_chain_ids = {str(chain_id) for chain_id in chain_ids if str(chain_id)}
    if not selected_chain_ids:
        raise RuntimeError(f"No chain IDs selected for {entry_id}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = _chain_subset_cache_stem(entry_id, sorted(selected_chain_ids))
    path = cache_dir / f"{stem}.pdb"
    map_path = cache_dir / f"{stem}.chain_map.csv"
    if path.exists() and path.stat().st_size > 0:
        return path, load_chain_id_map(map_path)

    full_pdb_path = cache_dir / f"{entry_id}.pdb"
    if (
        full_pdb_path.exists()
        and full_pdb_path.stat().st_size > 0
        and all(len(chain_id) == 1 for chain_id in selected_chain_ids)
    ):
        return full_pdb_path, {}

    cif_path = cache_dir / f"{entry_id}.cif"
    if not cif_path.exists() or cif_path.stat().st_size <= 0:
        cif_url = f"https://files.rcsb.org/download/{entry_id}.cif"
        last_error: Exception | None = None
        for attempt in range(1, config.retries + 1):
            try:
                response = session.get(cif_url, timeout=config.timeout_seconds)
                response.raise_for_status()
                cif_path.write_text(response.text, encoding="utf-8")
                break
            except (requests.RequestException, OSError) as exc:
                last_error = exc
                if attempt < config.retries:
                    time.sleep(config.backoff_seconds * attempt)
        if not cif_path.exists() or cif_path.stat().st_size <= 0:
            raise RuntimeError(f"Failed to download {entry_id} mmCIF: {last_error}")

    structure = parse_mmcif_structure(entry_id, cif_path)
    chain_id_map, selected_chain_object_ids = _coerce_selected_structure_chain_ids_for_pdbio(
        structure=structure,
        selected_chain_ids=selected_chain_ids,
    )
    missing_chain_ids = selected_chain_ids - set(chain_id_map)
    if missing_chain_ids:
        raise RuntimeError(
            f"{entry_id} mmCIF is missing selected chains: "
            + ",".join(sorted(missing_chain_ids))
        )
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(path), select=ChainSubsetSelect(selected_chain_object_ids))
    write_csv_rows(
        output_path=map_path,
        header=["original_chain_id", "mapped_chain_id"],
        rows=sorted(chain_id_map.items()),
    )
    return path, chain_id_map


def load_chain_id_map(map_path: Path) -> dict[str, str]:
    """Read a JSON chain-ID mapping file from disk."""
    if not map_path.exists():
        return {}
    chain_id_map: dict[str, str] = {}
    with map_path.open("r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            original = str(row.get("original_chain_id") or "")
            mapped = str(row.get("mapped_chain_id") or "")
            if original and mapped:
                chain_id_map[original] = mapped
    return chain_id_map


def write_csv_rows(
    output_path: Path,
    header: Sequence[str],
    rows: Iterable[Sequence[Any]],
) -> None:
    """Write dataclass records to CSV with the requested field order."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(rows)


PROGRAM_REMARK_PATTERN = re.compile(r"^REMARK\s+3\s+PROGRAM\s*:\s*(.*)$")
PROGRAM_SPLIT_PATTERN = re.compile(
    r"\s*(?:,|;|/|\+|\|\||\bAND\b)\s*",
    re.IGNORECASE,
)
PROGRAM_TRAILING_VERSION_PATTERN = re.compile(
    r"\s+(?:V(?:ERSION)?\.?\s*)?\d[\w.\-_:]*$",
    re.IGNORECASE,
)
PROGRAM_PARENTHESIS_PATTERN = re.compile(r"\([^)]*\)")
PROGRAM_HAS_LETTER_PATTERN = re.compile(r"[A-Z]")
PROGRAM_EMPTY_VALUES: frozenset[str] = frozenset(
    {
        "",
        "NULL",
        "NONE",
        "N/A",
        "NA",
        "NOT APPLICABLE",
        "NOT PROVIDED",
        "UNKNOWN",
        "?",
    }
)
PROGRAM_CLUSTER_DEFINITIONS: tuple[tuple[str, str], ...] = (
    ("CLUSTER1", "AMBER"),
    ("CLUSTER2", "ARIA"),
    ("CLUSTER3", "CNS"),
    ("CLUSTER4", "CYANA"),
    ("CLUSTER5", "DISCOVER"),
    ("CLUSTER6", "DIANA_DYANA"),
    ("CLUSTER7", "XPLOR"),
    ("CLUSTER8", "XPLOR_NIH"),
    ("CLUSTER9", "OTHER"),
)
PROGRAM_CLUSTER_NAME_BY_ID: dict[str, str] = dict(PROGRAM_CLUSTER_DEFINITIONS)


def _normalize_refinement_program_name(raw_value: str) -> str | None:
    """Normalize raw refinement program text to a canonical program label."""
    token = raw_value.strip().upper()
    if not token:
        return None
    token = PROGRAM_PARENTHESIS_PATTERN.sub(" ", token)
    token = " ".join(token.split()).strip(".,:; ")
    if not token or token in PROGRAM_EMPTY_VALUES:
        return None
    if ":" in token:
        token = token.split(":", 1)[0].strip(".,:; ")
    while token:
        trimmed = PROGRAM_TRAILING_VERSION_PATTERN.sub("", token).strip(".,:; ")
        if trimmed == token:
            break
        token = trimmed
    if token.endswith(" VERSION"):
        token = token[: -len(" VERSION")].strip(".,:; ")
    token = " ".join(token.split()).strip(".,:; ")
    if not token or token in PROGRAM_EMPTY_VALUES:
        return None
    if PROGRAM_HAS_LETTER_PATTERN.search(token) is None:
        return None
    return token


def extract_raw_refinement_program_text_from_pdb(pdb_path: Path) -> str:
    """Extract the raw refinement program remark text from a PDB file."""
    values: list[str] = []
    with pdb_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            match = PROGRAM_REMARK_PATTERN.match(line)
            if not match:
                continue
            value = match.group(1).strip()
            if value:
                values.append(value)
    return " || ".join(values)


def extract_refinement_programs_from_pdb(pdb_path: Path) -> set[str]:
    """Extract canonical refinement program names from PDB remarks."""
    programs: set[str] = set()
    with pdb_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            match = PROGRAM_REMARK_PATTERN.match(line)
            if not match:
                continue
            value = match.group(1).strip()
            if not value:
                continue
            for raw_token in PROGRAM_SPLIT_PATTERN.split(value):
                normalized = _normalize_refinement_program_name(raw_token)
                if normalized is not None:
                    programs.add(normalized)
    return programs


def _classify_normalized_program_cluster(
    program: str,
) -> tuple[str, str] | None:
    """Assign one normalized refinement program name to a broad cluster."""
    text = program.upper()
    if "AMBER" in text:
        return "CLUSTER1", PROGRAM_CLUSTER_NAME_BY_ID["CLUSTER1"]
    if "ARIA" in text:
        return "CLUSTER2", PROGRAM_CLUSTER_NAME_BY_ID["CLUSTER2"]
    if "CNS" in text:
        return "CLUSTER3", PROGRAM_CLUSTER_NAME_BY_ID["CLUSTER3"]
    if "CYANA" in text:
        return "CLUSTER4", PROGRAM_CLUSTER_NAME_BY_ID["CLUSTER4"]
    if "DISCOVER" in text:
        return "CLUSTER5", PROGRAM_CLUSTER_NAME_BY_ID["CLUSTER5"]
    if "DIANA" in text or "DYANA" in text:
        return "CLUSTER6", PROGRAM_CLUSTER_NAME_BY_ID["CLUSTER6"]
    if ("X-PLOR" in text or "XPLOR" in text) and "NIH" not in text:
        return "CLUSTER7", PROGRAM_CLUSTER_NAME_BY_ID["CLUSTER7"]
    if (
        "X-PLOR NIH" in text
        or "XPLOR NIH" in text
        or "X-PLOR-NIH" in text
        or "XPLOR-NIH" in text
    ):
        return "CLUSTER8", PROGRAM_CLUSTER_NAME_BY_ID["CLUSTER8"]
    return None


def extract_solution_nmr_program_clusters(
    program_text: str | None,
) -> list[tuple[str, str]]:
    """Extract unique program clusters from refinement program text in text order."""
    clusters: list[tuple[str, str]] = []
    seen_cluster_ids: set[str] = set()
    text = (program_text or "").strip()
    for raw_token in PROGRAM_SPLIT_PATTERN.split(text):
        normalized = _normalize_refinement_program_name(raw_token)
        if normalized is None:
            continue
        cluster = _classify_normalized_program_cluster(normalized)
        if cluster is None:
            continue
        cluster_id, _ = cluster
        if cluster_id in seen_cluster_ids:
            continue
        seen_cluster_ids.add(cluster_id)
        clusters.append(cluster)

    if clusters:
        return clusters
    return [("CLUSTER9", PROGRAM_CLUSTER_NAME_BY_ID["CLUSTER9"])]


def extract_model_pdb_texts(pdb_path: Path) -> list[str]:
    """Split a PDB file into separate text blocks for each model."""
    model_texts: list[str] = []
    model_lines: list[str] = []
    saw_model = False
    in_model = False

    with pdb_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            record = line[:6]
            if record.startswith("MODEL"):
                if in_model and model_lines:
                    model_lines.append("END\n")
                    model_texts.append("".join(model_lines))
                    model_lines = []
                saw_model = True
                in_model = True
                continue
            if record.startswith("ENDMDL"):
                if in_model and model_lines:
                    model_lines.append("END\n")
                    model_texts.append("".join(model_lines))
                    model_lines = []
                in_model = False
                continue
            if saw_model and not in_model:
                continue
            if (
                record.startswith("ATOM")
                or record.startswith("HETATM")
                or record.startswith("TER")
            ):
                model_lines.append(line)

    if model_lines:
        model_lines.append("END\n")
        model_texts.append("".join(model_lines))

    return model_texts


def _parse_stride_state_by_chain(stdout: str) -> dict[str, dict[int, str]]:
    """Parse STRIDE output into residue state codes grouped by chain."""
    state_by_chain: dict[str, dict[int, str]] = {}
    for line in stdout.splitlines():
        if not line.startswith("ASG"):
            continue
        parts = line.split()
        if len(parts) < 6:
            continue
        chain_label = str(parts[2]).strip()
        seq_id_raw = str(parts[3]).strip()
        if not seq_id_raw:
            continue
        if seq_id_raw[-1:].isalpha():
            seq_id_raw = seq_id_raw[:-1]
        try:
            auth_seq_id = int(seq_id_raw)
        except ValueError:
            continue
        state_raw = str(parts[5]).strip()
        if len(state_raw) != 1:
            continue
        state = state_raw if state_raw in STRIDE_STATE_CODES else "C"
        chain_states = state_by_chain.setdefault(chain_label, {})
        chain_states.setdefault(auth_seq_id, state)
    return state_by_chain


def _select_stride_chain_states(
    state_by_chain: dict[str, dict[int, str]],
    chain_id: str,
) -> dict[int, str] | None:
    """Select STRIDE residue states for the best matching chain identifier."""
    chain_states = state_by_chain.get(chain_id)
    if not chain_states and len(state_by_chain) == 1:
        chain_states = next(iter(state_by_chain.values()))
    return chain_states


def _run_stride_for_model_text(
    model_text: str,
    stride_executable: str,
) -> dict[str, dict[int, str]] | None:
    """Run STRIDE on a single MODEL text block and return parsed states."""
    with tempfile.NamedTemporaryFile(
        "w", suffix=".pdb", encoding="utf-8", delete=True
    ) as handle:
        handle.write(model_text)
        handle.flush()

        process = subprocess.run(
            [stride_executable, handle.name],
            check=False,
            capture_output=True,
            text=True,
        )
        if process.returncode != 0:
            return None
        return _parse_stride_state_by_chain(process.stdout)


def _extract_stride_core_range_for_modeled_auth_seq_ids(
    chain_states: dict[int, str],
    modeled_auth_seq_ids: set[int],
) -> tuple[int, int] | None:
    """Find the outer modeled residue range covered by STRIDE core states."""
    structured_auth_seq_ids = sorted(
        auth_seq_id
        for auth_seq_id in modeled_auth_seq_ids
        if chain_states.get(auth_seq_id) in STRIDE_CORE_STATE_CODES
    )
    if not structured_auth_seq_ids:
        return None
    return structured_auth_seq_ids[0], structured_auth_seq_ids[-1]


def compute_stride_state_coverages_for_chain_modeled_first_model(
    session: requests.Session,
    config: CollectorConfig,
    cache_dir: Path,
    entry_id: str,
    chain_id: str,
    modeled_sequence_length: int,
    modeled_auth_seq_ids: set[int],
    stride_executable: str,
) -> tuple[dict[str, float], int, int]:
    """Compute first-model STRIDE secondary-structure coverage for one chain."""
    default_coverages = {state: -1.0 for state in STRIDE_STATE_CODES}
    if modeled_sequence_length <= 0 or not modeled_auth_seq_ids:
        return default_coverages, 0, 0

    try:
        pdb_path = download_pdb_if_needed(
            session=session,
            config=config,
            cache_dir=cache_dir,
            entry_id=entry_id,
        )
    except Exception:
        return default_coverages, 0, 0

    chain_map = load_cached_chain_id_map(cache_dir, entry_id)
    parsed_chain_id = chain_map.get(chain_id, chain_id)
    model_texts = extract_model_pdb_texts(pdb_path)
    if not model_texts:
        return default_coverages, 0, 0

    try:
        state_by_chain = _run_stride_for_model_text(
            model_text=model_texts[0],
            stride_executable=stride_executable,
        )
        if state_by_chain is None:
            return default_coverages, len(model_texts), 0

        chain_states = _select_stride_chain_states(state_by_chain, parsed_chain_id)
        if not chain_states:
            return default_coverages, len(model_texts), 0

        filtered_states = [
            chain_states.get(auth_seq_id, "C")
            for auth_seq_id in sorted(modeled_auth_seq_ids)
        ]
        missing_count = max(0, modeled_sequence_length - len(filtered_states))
        if missing_count > 0:
            filtered_states.extend(["C"] * missing_count)
        if not filtered_states:
            return default_coverages, len(model_texts), 0

        state_counts = Counter(filtered_states)
        denominator = float(modeled_sequence_length)
        coverages = {
            state: min(1.0, max(0.0, state_counts.get(state, 0) / denominator))
            for state in STRIDE_STATE_CODES
        }
        return coverages, len(model_texts), 1
    except Exception:
        return default_coverages, len(model_texts), 0


def compute_stride_core_range_for_modeled_auth_seq_ids_in_first_model(
    pdb_path: Path,
    chain_id: str,
    modeled_auth_seq_ids: set[int],
    stride_executable: str,
) -> tuple[int, int] | None:
    """Return the modeled first-model residue range supported by STRIDE core states."""
    if not modeled_auth_seq_ids:
        return None

    model_texts = extract_model_pdb_texts(pdb_path)
    if not model_texts:
        return None

    state_by_chain = _run_stride_for_model_text(
        model_text=model_texts[0],
        stride_executable=stride_executable,
    )
    if state_by_chain is None:
        return None

    chain_states = _select_stride_chain_states(state_by_chain, chain_id)
    if not chain_states:
        return None

    return _extract_stride_core_range_for_modeled_auth_seq_ids(
        chain_states=chain_states,
        modeled_auth_seq_ids=modeled_auth_seq_ids,
    )


def parse_models_ca_coords(
    pdb_path: Path,
    chain_id: str,
    start_seq_id: int | None = None,
    end_seq_id: int | None = None,
) -> list[dict[int, np.ndarray]]:
    """Parse CA coordinates from every model in a PDB file."""
    model_maps, _ = parse_models_ca_coords_with_stats(
        pdb_path=pdb_path,
        chain_id=chain_id,
        start_seq_id=start_seq_id,
        end_seq_id=end_seq_id,
    )
    return model_maps


def parse_first_model_ca_residue_sequence(
    pdb_path: Path,
    chain_id: str,
    start_seq_id: int | None = None,
    end_seq_id: int | None = None,
    include_hetatm: bool = True,
) -> list[tuple[int, str]]:
    """Return residue identities for CA atoms in the first model."""
    return [
        (record.resid, record.identity)
        for record in parse_first_model_ca_residues(
            pdb_path=pdb_path,
            chain_id=chain_id,
            start_seq_id=start_seq_id,
            end_seq_id=end_seq_id,
            include_hetatm=include_hetatm,
        )
    ]


def _parse_first_model_ca_line_fields(
    line: str,
) -> tuple[str, int, str, str, float, str] | None:
    """Parse CA fields, with a fallback for nonstandard long component IDs."""
    atom_name = line[12:16].strip()
    if atom_name != "CA":
        return None

    atom_chain = line[21].strip()
    resid_text = line[22:26].strip()
    insertion_code = line[26].strip()
    alt_loc = line[16].strip()
    occupancy = _parse_pdb_occupancy(line)
    resname = line[17:20].strip()
    try:
        resid = int(resid_text)
    except ValueError:
        resid = None
    if resid is not None and occupancy != float("-inf"):
        return atom_chain, resid, insertion_code, alt_loc, occupancy, resname

    parts = line.split()
    if len(parts) < 10 or parts[2] != "CA":
        return None
    match = re.fullmatch(r"(-?\d+)([A-Za-z]?)", parts[5])
    if match is None:
        return None
    try:
        fallback_resid = int(match.group(1))
        fallback_occupancy = float(parts[9])
    except ValueError:
        return None
    return (
        parts[4],
        fallback_resid,
        match.group(2),
        "",
        fallback_occupancy,
        parts[3],
    )


def parse_first_model_ca_residues(
    pdb_path: Path,
    chain_id: str,
    start_seq_id: int | None = None,
    end_seq_id: int | None = None,
    include_hetatm: bool = True,
) -> list[CAResidueRecord]:
    """Return first-model CA residue records with identity and atom-standard flags."""
    residue_order: list[int] = []
    candidates: dict[int, tuple[str, float, str, CAResidueRecord]] = {}
    modres_identity_by_key = _parse_pdb_modres_identity_map(pdb_path)
    has_model_records = False
    in_model = False

    with pdb_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            record = line[:6]
            if record.startswith("MODEL"):
                if has_model_records:
                    break
                has_model_records = True
                in_model = True
                continue
            if record.startswith("ENDMDL"):
                if in_model:
                    break
                continue
            if has_model_records and not in_model:
                continue
            is_standard_atom = record.startswith("ATOM")
            is_hetero_atom = record.startswith("HETATM")
            if not is_standard_atom and not (include_hetatm and is_hetero_atom):
                continue
            parsed_fields = _parse_first_model_ca_line_fields(line)
            if parsed_fields is None:
                continue
            (
                atom_chain,
                resid,
                insertion_code,
                alt_loc,
                occupancy,
                resname,
            ) = parsed_fields
            if atom_chain != chain_id:
                continue
            if start_seq_id is not None and resid < start_seq_id:
                continue
            if end_seq_id is not None and resid > end_seq_id:
                continue

            if occupancy <= 0.0:
                continue
            if is_standard_atom:
                identity = seq1(resname, custom_map={"MSE": "M"}, undef_code="X")
            else:
                identity = modres_identity_by_key.get(
                    (atom_chain, resid, insertion_code, resname),
                    modres_identity_by_key.get(
                        (atom_chain, resid, "", resname),
                        f"HET:{resname}",
                    ),
                )
            ca_record = CAResidueRecord(
                resid=resid,
                identity=identity,
                is_standard_atom=is_standard_atom,
            )

            if resid not in candidates:
                residue_order.append(resid)
                candidates[resid] = (insertion_code, occupancy, alt_loc, ca_record)
                continue

            (
                existing_insertion_code,
                existing_occupancy,
                existing_alt_loc,
                existing_record,
            ) = candidates[resid]
            if ca_record.is_standard_atom != existing_record.is_standard_atom:
                if ca_record.is_standard_atom:
                    candidates[resid] = (
                        insertion_code,
                        occupancy,
                        alt_loc,
                        ca_record,
                    )
                continue
            if _is_better_ca_candidate(
                new_insertion_code=insertion_code,
                new_occupancy=occupancy,
                new_alt_loc=alt_loc,
                current_insertion_code=existing_insertion_code,
                current_occupancy=existing_occupancy,
                current_alt_loc=existing_alt_loc,
            ):
                candidates[resid] = (insertion_code, occupancy, alt_loc, ca_record)

    return [candidates[resid][3] for resid in residue_order if resid in candidates]


def parse_first_model_modeled_ca_auth_seq_ids(
    pdb_path: Path,
    chain_id: str,
) -> set[int]:
    """Return author residue IDs with positive-occupancy first-model CA atoms."""
    return {
        record.resid
        for record in parse_first_model_ca_residues(
            pdb_path=pdb_path,
            chain_id=chain_id,
            include_hetatm=False,
        )
    }


def _parse_pdb_modres_identity_map(
    pdb_path: Path,
) -> dict[tuple[str, int, str, str], str]:
    """Parse MODRES records into modified-to-standard residue identity mappings."""
    identity_by_key: dict[tuple[str, int, str, str], str] = {}
    with pdb_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("MODRES"):
                continue
            resname = line[12:15].strip()
            chain_id = line[16].strip()
            seq_num_text = line[18:22].strip()
            insertion_code = line[22].strip()
            standard_resname = line[24:27].strip()
            if not (resname and chain_id and seq_num_text and standard_resname):
                parts = line.split()
                if len(parts) < 6:
                    continue
                resname = parts[2].strip()
                chain_id = parts[3].strip()
                seq_num_text = parts[4].strip()
                insertion_code = ""
                standard_resname = parts[5].strip()
            try:
                seq_num = int(seq_num_text)
            except ValueError:
                continue
            identity = seq1(
                standard_resname,
                custom_map={"MSE": "M"},
                undef_code="X",
            )
            if identity == "X":
                continue
            identity_by_key[(chain_id, seq_num, insertion_code, resname)] = identity
    return identity_by_key


def find_modeled_ca_core_identity_matches(
    nmr_residues: list[CAResidueRecord],
    xray_residues: list[CAResidueRecord],
    sequence_identity_percent: int,
) -> list[list[tuple[CAResidueRecord, CAResidueRecord]]]:
    """Find matching first-model CA ranges that align to a modeled core sequence."""
    if not nmr_residues or not xray_residues:
        return []
    if not 0 <= sequence_identity_percent <= 100:
        raise ValueError("sequence_identity_percent must be between 0 and 100")
    nmr_identities = [record.identity for record in nmr_residues]
    xray_identities = [record.identity for record in xray_residues]
    query_length = len(nmr_identities)
    if query_length > len(xray_identities):
        if sequence_identity_percent == 100:
            return []
    if sequence_identity_percent < 100:
        match = _find_gapped_modeled_ca_core_identity_match(
            nmr_residues=nmr_residues,
            xray_residues=xray_residues,
            sequence_identity_percent=sequence_identity_percent,
        )
        return [match] if match else []
    if query_length > len(xray_identities):
        return []

    matches: list[list[tuple[CAResidueRecord, CAResidueRecord]]] = []
    for start_idx in range(0, len(xray_identities) - query_length + 1):
        end_idx = start_idx + query_length
        match_count = sum(
            1
            for nmr_identity, xray_identity in zip(
                nmr_identities,
                xray_identities[start_idx:end_idx],
            )
            if nmr_identity == xray_identity
        )
        if match_count * 100 < sequence_identity_percent * query_length:
            continue
        matches.append(list(zip(nmr_residues, xray_residues[start_idx:end_idx])))
    return matches


def _find_gapped_modeled_ca_core_identity_match(
    nmr_residues: list[CAResidueRecord],
    xray_residues: list[CAResidueRecord],
    sequence_identity_percent: int,
) -> list[tuple[CAResidueRecord, CAResidueRecord]] | None:
    """Find a modeled core match allowing residue gaps at CA positions."""
    nmr_len = len(nmr_residues)
    min_count = (nmr_len * sequence_identity_percent + 99) // 100
    if min_count <= 0:
        return None

    nmr_identities = [record.identity for record in nmr_residues]
    xray_identities = [record.identity for record in xray_residues]
    rows = nmr_len + 1
    cols = len(xray_residues) + 1
    scores = np.zeros((rows, cols), dtype=float)
    pointers = np.zeros((rows, cols), dtype=np.int8)
    best_score = 0.0
    best_pos: tuple[int, int] | None = None

    for i in range(1, rows):
        nmr_identity = nmr_identities[i - 1]
        for j in range(1, cols):
            xray_identity = xray_identities[j - 1]
            diag_score = scores[i - 1, j - 1] + (
                2.0 if nmr_identity == xray_identity else -1.0
            )
            up_score = scores[i - 1, j] - 1.0
            left_score = scores[i, j - 1] - 1.0
            cell_score = max(0.0, diag_score, up_score, left_score)
            scores[i, j] = cell_score
            if cell_score == 0.0:
                continue
            if cell_score == diag_score:
                pointers[i, j] = 1
            elif cell_score == up_score:
                pointers[i, j] = 2
            else:
                pointers[i, j] = 3
            if cell_score > best_score:
                best_score = cell_score
                best_pos = (i, j)

    if best_pos is None:
        return None

    i, j = best_pos
    pairs: list[tuple[CAResidueRecord, CAResidueRecord]] = []
    identity_count = 0
    while i > 0 and j > 0 and scores[i, j] > 0.0:
        pointer = pointers[i, j]
        if pointer == 1:
            nmr_record = nmr_residues[i - 1]
            xray_record = xray_residues[j - 1]
            pairs.append((nmr_record, xray_record))
            if nmr_record.identity == xray_record.identity:
                identity_count += 1
            i -= 1
            j -= 1
        elif pointer == 2:
            i -= 1
        elif pointer == 3:
            j -= 1
        else:
            break

    pairs.reverse()
    if len(pairs) < min_count:
        return None
    if identity_count < min_count:
        return None
    return pairs


def _alt_loc_tiebreak_key(alt_loc: str) -> tuple[int, str]:
    # Prefer blank altLoc, then A, then 1; keep deterministic order for others.
    """Rank alternate atom locations for deterministic CA selection."""
    if alt_loc == "":
        return (0, "")
    if alt_loc == "A":
        return (1, "")
    if alt_loc == "1":
        return (2, "")
    return (3, alt_loc)


def _insertion_code_tiebreak_key(insertion_code: str) -> tuple[int, str]:
    # Prefer residue numbers without insertion codes (e.g., 102 over 102A).
    """Rank insertion codes for deterministic residue ordering."""
    if insertion_code == "":
        return (0, "")
    return (1, insertion_code)


def _parse_pdb_occupancy(line: str) -> float:
    """Parse the occupancy value from a PDB ATOM record."""
    occ_text = line[54:60].strip()
    if not occ_text:
        return float("-inf")
    try:
        return float(occ_text)
    except ValueError:
        return float("-inf")


def _is_better_ca_candidate(
    new_insertion_code: str,
    new_occupancy: float,
    new_alt_loc: str,
    current_insertion_code: str,
    current_occupancy: float,
    current_alt_loc: str,
) -> bool:
    """Decide whether a CA atom candidate should replace the current one."""
    new_i_code_key = _insertion_code_tiebreak_key(new_insertion_code)
    current_i_code_key = _insertion_code_tiebreak_key(current_insertion_code)
    if new_i_code_key < current_i_code_key:
        return True
    if new_i_code_key > current_i_code_key:
        return False

    if new_occupancy > current_occupancy + 1e-9:
        return True
    if current_occupancy > new_occupancy + 1e-9:
        return False
    return _alt_loc_tiebreak_key(new_alt_loc) < _alt_loc_tiebreak_key(current_alt_loc)


def parse_models_ca_coords_with_stats(
    pdb_path: Path,
    chain_id: str,
    start_seq_id: int | None = None,
    end_seq_id: int | None = None,
) -> tuple[list[dict[int, np.ndarray]], list[dict[int, int]]]:
    # Select one positive-occupancy CA per residue by max occupancy
    # (altLoc-aware) and keep raw per-residue counts so callers can report how
    # many modeled CA atoms were present before altLoc collapsing.
    """Parse model CA coordinates and report residue-selection statistics."""
    models: list[dict[int, np.ndarray]] = []
    raw_ca_counts_per_model: list[dict[int, int]] = []
    current_candidates: dict[int, tuple[str, float, str, np.ndarray]] = {}
    current_raw_counts: Counter[int] = Counter()
    has_model_records = False
    in_model = False

    def finalize_model() -> None:
        """Finalize one parsed model and reset per-model parsing buffers."""
        models.append(
            {
                resid: candidate[3]
                for resid, candidate in current_candidates.items()
            }
        )
        raw_ca_counts_per_model.append(dict(current_raw_counts))

    with pdb_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            record = line[:6]
            if record.startswith("MODEL"):
                if in_model:
                    finalize_model()
                    current_candidates = {}
                    current_raw_counts = Counter()
                has_model_records = True
                in_model = True
                continue
            if record.startswith("ENDMDL"):
                if in_model:
                    finalize_model()
                    current_candidates = {}
                    current_raw_counts = Counter()
                    in_model = False
                continue
            if not record.startswith("ATOM"):
                continue

            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue
            atom_chain = line[21].strip()
            if atom_chain != chain_id:
                continue
            resid_text = line[22:26].strip()
            try:
                resid = int(resid_text)
            except ValueError:
                continue
            if start_seq_id is not None and resid < start_seq_id:
                continue
            if end_seq_id is not None and resid > end_seq_id:
                continue

            insertion_code = line[26].strip()
            alt_loc = line[16].strip()
            occupancy = _parse_pdb_occupancy(line)
            if occupancy <= 0.0:
                continue
            current_raw_counts[resid] += 1

            try:
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
            except ValueError:
                continue
            coords = np.array([x, y, z], dtype=float)
            existing = current_candidates.get(resid)
            if existing is None:
                current_candidates[resid] = (insertion_code, occupancy, alt_loc, coords)
                continue

            (
                existing_insertion_code,
                existing_occupancy,
                existing_alt_loc,
                _,
            ) = existing
            if _is_better_ca_candidate(
                new_insertion_code=insertion_code,
                new_occupancy=occupancy,
                new_alt_loc=alt_loc,
                current_insertion_code=existing_insertion_code,
                current_occupancy=existing_occupancy,
                current_alt_loc=existing_alt_loc,
            ):
                current_candidates[resid] = (insertion_code, occupancy, alt_loc, coords)

    if has_model_records:
        if in_model or current_candidates or current_raw_counts:
            finalize_model()
    elif current_candidates or current_raw_counts:
        finalize_model()
    return models, raw_ca_counts_per_model


def _superposed_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    """Compute RMSD after optimal rigid-body superposition."""
    return float(mda_rmsd(a, b, center=True, superposition=True))


def _aligned_coordinates_to_reference(
    mobile: np.ndarray, reference: np.ndarray
) -> np.ndarray:
    """Align coordinates onto a reference coordinate set."""
    mobile_center = np.mean(mobile, axis=0)
    reference_center = np.mean(reference, axis=0)
    mobile_centered = mobile - mobile_center
    reference_centered = reference - reference_center

    rotation, _ = mda_rotation_matrix(mobile_centered, reference_centered)
    return mobile_centered @ rotation.T + reference_center


def _coordinates_aligned_to_first_model(coords: np.ndarray) -> np.ndarray:
    """Align every model coordinate set onto the first model."""
    if coords.ndim != 3 or coords.shape[0] == 0:
        raise ValueError("coords must have shape (n_models, n_atoms, 3)")
    reference = np.asarray(coords[0], dtype=float)
    return np.asarray(
        [
            _aligned_coordinates_to_reference(model, reference)
            for model in coords
        ],
        dtype=float,
    )


def _ca_rmsd_to_mean_structure(coords: np.ndarray) -> float:
    """Compute sqrt(1 / (N*n) * sum_i sum_j |r_ij - r_mean,j|^2)."""
    if coords.ndim != 3 or coords.shape[0] == 0:
        raise ValueError("coords must have shape (n_models, n_atoms, 3)")
    mean_coords = np.mean(coords, axis=0)
    squared_distances = np.sum(np.square(coords - mean_coords), axis=2)
    return float(np.sqrt(np.mean(squared_distances)))


def _average_structure_aligned_to_first_model(coords: np.ndarray) -> np.ndarray:
    """Build an average structure after aligning models to the first model."""
    aligned = _coordinates_aligned_to_first_model(coords)
    return np.mean(aligned, axis=0)


MEMBRANE_ANNOTATION_TYPES: tuple[str, ...] = ("OPM", "PDBTM", "MemProtMD", "mpstruc")


class RCSBClient:
    def __init__(self, config: CollectorConfig) -> None:
        """Initialize the RCSB API client session and configuration."""
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "pdb-extensible-collector/1.0"})

    @staticmethod
    def _normalize_similarity_cutoff(raw_cutoff: Any) -> int | None:
        """Normalize raw sequence-identity cutoff values to integer percentages."""
        try:
            return int(round(float(raw_cutoff)))
        except (TypeError, ValueError):
            return None

    @classmethod
    def _extract_sequence_identity_groups(
        cls,
        memberships: Iterable[dict[str, Any] | None],
        allowed_cutoffs: set[int] | None = None,
    ) -> dict[int, str]:
        """Extract matching sequence-identity group IDs from GraphQL entity data."""
        groups: dict[int, str] = {}
        for membership in memberships:
            if not membership:
                continue
            if (
                membership.get("aggregation_method")
                != SEQUENCE_IDENTITY_AGGREGATION_METHOD
            ):
                continue
            cutoff = cls._normalize_similarity_cutoff(
                membership.get("similarity_cutoff")
            )
            group_id = membership.get("group_id")
            if cutoff is None or not group_id:
                continue
            if allowed_cutoffs is not None and cutoff not in allowed_cutoffs:
                continue
            groups[cutoff] = str(group_id)
        return groups

    @staticmethod
    def _extract_solution_nmr_monomer_context(
        entry: dict[str, Any],
    ) -> tuple[str, int, int, dict[str, Any], str] | None:
        """Extract monomer chain and modeled-sequence context from entry data."""
        entry_id = entry.get("rcsb_id")
        if not entry_id:
            return None

        model_count_raw = entry.get("rcsb_entry_info", {}).get("deposited_model_count")
        if model_count_raw is None:
            return None
        try:
            model_count = int(model_count_raw)
        except (TypeError, ValueError):
            return None
        if model_count <= 1:
            return None

        year = extract_year(entry.get("rcsb_accession_info", {}).get("deposit_date"))
        if year is None:
            return None

        polymer_entities = entry.get("polymer_entities") or []
        if len(polymer_entities) != 1:
            return None
        polymer_entity = polymer_entities[0] or {}
        entity_poly = polymer_entity.get("entity_poly") or {}

        if entity_poly.get("type") not in PROTEIN_MONOMER_ENTITY_TYPES:
            return None
        if entity_poly.get("rcsb_entity_polymer_type") != PROTEIN_POLYMER_TYPE:
            return None

        chain_id = str(entity_poly.get("pdbx_strand_id") or "").strip()
        if not chain_id or "," in chain_id:
            return None

        return str(entry_id), year, model_count, polymer_entity, chain_id

    def _fetch_paginated_identifiers(
        self,
        query: dict[str, Any],
        return_type: str,
        progress_label: str | None = None,
    ) -> list[str]:
        """Run a paginated RCSB search query and return all identifiers."""
        all_ids: list[str] = []
        start = 0
        total_count: int | None = None
        while total_count is None or start < total_count:
            payload = {
                "query": query,
                "return_type": return_type,
                "request_options": {
                    "paginate": {"start": start, "rows": self.config.page_size}
                },
            }
            data = self._post_json(self.config.search_url, payload)
            total_count = int(data.get("total_count", 0))
            batch_ids = [
                item["identifier"]
                for item in data.get("result_set") or []
                if "identifier" in item
            ]
            all_ids.extend(batch_ids)
            start += len(batch_ids)
            if not batch_ids:
                break
            if progress_label:
                LOGGER.info(
                    "%s: fetched %d/%d entry IDs",
                    progress_label,
                    len(all_ids),
                    total_count,
                )
        return all_ids

    def _post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST JSON to an RCSB endpoint with retry and backoff handling."""
        last_error: Exception | None = None
        for attempt in range(1, self.config.retries + 1):
            try:
                response = self.session.post(
                    url, json=payload, timeout=self.config.timeout_seconds
                )
                response.raise_for_status()
                return response.json()
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                wait_seconds = self.config.backoff_seconds * attempt
                LOGGER.warning(
                    "Request failed (attempt %d/%d): %s",
                    attempt,
                    self.config.retries,
                    exc,
                )
                if attempt < self.config.retries:
                    time.sleep(wait_seconds)
        raise RuntimeError(
            f"Request failed after {self.config.retries} attempts: {last_error}"
        )

    def fetch_entry_ids_for_method(
        self,
        method_label: str,
        query_value: str,
        require_protein_entities: bool = False,
    ) -> list[str]:
        """Fetch entry IDs for one experimental method."""
        method_query: dict[str, Any] = {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "exptl.method",
                "operator": "exact_match",
                "value": query_value,
            },
        }
        query: dict[str, Any] = method_query
        if require_protein_entities:
            query = {
                "type": "group",
                "logical_operator": "and",
                "nodes": [
                    method_query,
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "rcsb_entry_info.polymer_entity_count_protein",
                            "operator": "greater_or_equal",
                            "value": 1,
                        },
                    },
                ],
            }
        matched_entry_ids = self._fetch_paginated_identifiers(
            query=query,
            return_type="entry",
            progress_label=f"{method_label} ({query_value})",
        )
        if not matched_entry_ids:
            return []

        filtered_entry_ids: list[str] = []
        for batch in chunked(matched_entry_ids, self.config.graphql_batch_size):
            filtered_entry_ids.extend(
                self._filter_entry_ids_by_exact_single_method(
                    entry_ids=batch,
                    method_value=query_value,
                )
            )

        LOGGER.info(
            "%s (%s): kept %d/%d entries with exactly one method",
            method_label,
            query_value,
            len(filtered_entry_ids),
            len(matched_entry_ids),
        )
        return filtered_entry_ids

    def _filter_entry_ids_by_exact_single_method(
        self, entry_ids: list[str], method_value: str
    ) -> list[str]:
        """Keep entries whose experimental method list is exactly the requested method."""
        if not entry_ids:
            return []

        query = """
        query($ids:[String!]!) {
          entries(entry_ids:$ids) {
            rcsb_id
            exptl {
              method
            }
          }
        }
        """
        payload = {"query": query, "variables": {"ids": entry_ids}}
        data = self._post_json(self.config.graphql_url, payload)
        entries = data.get("data", {}).get("entries") or []

        filtered: list[str] = []
        for entry in entries:
            if not entry:
                continue
            entry_id = entry.get("rcsb_id")
            if not entry_id:
                continue

            methods = []
            for exptl in entry.get("exptl") or []:
                if not exptl:
                    continue
                method = exptl.get("method")
                if method:
                    methods.append(str(method))

            unique_methods = set(methods)
            if len(unique_methods) == 1 and method_value in unique_methods:
                filtered.append(str(entry_id))

        return filtered

    def fetch_entry_ids_for_membrane_annotations(
        self, annotation_types: tuple[str, ...]
    ) -> list[str]:
        """Fetch entry IDs with membrane-protein annotations."""
        query = {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_polymer_entity_annotation.type",
                "operator": "in",
                "value": list(annotation_types),
            },
        }
        return self._fetch_paginated_identifiers(
            query=query,
            return_type="entry",
            progress_label=f"Membrane proteins ({','.join(annotation_types)})",
        )

    def fetch_deposit_dates_for_ids(self, entry_ids: list[str]) -> list[str]:
        """Fetch deposit date strings for entry IDs."""
        query = """
        query($ids:[String!]!) {
          entries(entry_ids:$ids) {
            rcsb_accession_info {
              deposit_date
            }
          }
        }
        """
        payload = {"query": query, "variables": {"ids": entry_ids}}
        data = self._post_json(self.config.graphql_url, payload)
        entries = data.get("data", {}).get("entries") or []
        return [
            entry.get("rcsb_accession_info", {}).get("deposit_date")
            for entry in entries
            if entry and entry.get("rcsb_accession_info", {}).get("deposit_date")
        ]

    def fetch_deposit_year_by_entry_id_for_ids(
        self, entry_ids: list[str]
    ) -> dict[str, int]:
        """Fetch deposit years keyed by entry ID."""
        if not entry_ids:
            return {}
        query = """
        query($ids:[String!]!) {
          entries(entry_ids:$ids) {
            rcsb_id
            rcsb_accession_info {
              deposit_date
            }
          }
        }
        """
        payload = {"query": query, "variables": {"ids": entry_ids}}
        data = self._post_json(self.config.graphql_url, payload)
        entries = data.get("data", {}).get("entries") or []
        entry_year_by_id: dict[str, int] = {}
        for entry in entries:
            if not entry:
                continue
            entry_id = entry.get("rcsb_id")
            if not entry_id:
                continue
            year = extract_year(entry.get("rcsb_accession_info", {}).get("deposit_date"))
            if year is None:
                continue
            entry_year_by_id[str(entry_id)] = year
        return entry_year_by_id

    def fetch_deposit_date_by_entry_id_for_ids(
        self, entry_ids: list[str]
    ) -> dict[str, str]:
        """Fetch deposit datetimes keyed by entry ID."""
        if not entry_ids:
            return {}
        query = """
        query($ids:[String!]!) {
          entries(entry_ids:$ids) {
            rcsb_id
            rcsb_accession_info {
              deposit_date
            }
          }
        }
        """
        payload = {"query": query, "variables": {"ids": entry_ids}}
        data = self._post_json(self.config.graphql_url, payload)
        entries = data.get("data", {}).get("entries") or []
        entry_date_by_id: dict[str, str] = {}
        for entry in entries:
            if not entry:
                continue
            entry_id = entry.get("rcsb_id")
            deposit_date = (
                entry.get("rcsb_accession_info", {}) or {}
            ).get("deposit_date")
            if not entry_id or not deposit_date:
                continue
            entry_date_by_id[str(entry_id)] = str(deposit_date)
        return entry_date_by_id

    def fetch_accession_dates_by_entry_id_for_ids(
        self, entry_ids: list[str]
    ) -> dict[str, tuple[str | None, str | None]]:
        """Fetch initial release datetimes keyed by entry ID."""
        if not entry_ids:
            return {}
        query = """
        query($ids:[String!]!) {
          entries(entry_ids:$ids) {
            rcsb_id
            rcsb_accession_info {
              deposit_date
              initial_release_date
            }
          }
        }
        """
        payload = {"query": query, "variables": {"ids": entry_ids}}
        data = self._post_json(self.config.graphql_url, payload)
        entries = data.get("data", {}).get("entries") or []
        entry_dates_by_id: dict[str, tuple[str | None, str | None]] = {}
        for entry in entries:
            if not entry:
                continue
            entry_id = entry.get("rcsb_id")
            accession_info = entry.get("rcsb_accession_info", {}) or {}
            if not entry_id:
                continue
            entry_dates_by_id[str(entry_id)] = (
                accession_info.get("deposit_date"),
                accession_info.get("initial_release_date"),
            )
        return entry_dates_by_id

    def fetch_entry_resolution_for_ids(self, entry_ids: list[str]) -> dict[str, float]:
        """Fetch crystallographic resolution values keyed by entry ID."""
        query = """
        query($ids:[String!]!) {
          entries(entry_ids:$ids) {
            rcsb_id
            rcsb_entry_info {
              resolution_combined
            }
          }
        }
        """
        payload = {"query": query, "variables": {"ids": entry_ids}}
        data = self._post_json(self.config.graphql_url, payload)
        entries = data.get("data", {}).get("entries") or []
        resolutions: dict[str, float] = {}
        for entry in entries:
            if not entry:
                continue
            entry_id = entry.get("rcsb_id")
            if not entry_id:
                continue
            combined = entry.get("rcsb_entry_info", {}).get("resolution_combined")
            if not combined:
                continue
            try:
                value = min(float(item) for item in combined if item is not None)
            except (TypeError, ValueError):
                continue
            resolutions[str(entry_id)] = value
        return resolutions

    def fetch_xray_polymer_entity_ids_for_group_ids(
        self, group_ids: list[str]
    ) -> list[str]:
        """Fetch X-ray polymer entity IDs for sequence-identity group IDs."""
        if not group_ids:
            return []
        query = {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "exptl.method",
                        "operator": "exact_match",
                        "value": "X-RAY DIFFRACTION",
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_polymer_entity_group_membership.group_id",
                        "operator": "in",
                        "value": group_ids,
                    },
                },
            ],
        }
        return self._fetch_paginated_identifiers(
            query=query,
            return_type="polymer_entity",
        )

    def fetch_polymer_entity_group_mapping_for_ids(
        self, entity_ids: list[str], similarity_cutoff: int
    ) -> list[XrayEntityGroupMappingRecord]:
        """Fetch sequence-identity group mappings for polymer entity IDs."""
        if not entity_ids:
            return []
        query = """
        query($ids:[String!]!) {
          polymer_entities(entity_ids:$ids) {
            rcsb_id
            entity_poly {
              pdbx_strand_id
            }
            rcsb_polymer_entity_container_identifiers {
              entry_id
            }
            rcsb_polymer_entity_group_membership {
              aggregation_method
              similarity_cutoff
              group_id
            }
          }
        }
        """
        payload = {"query": query, "variables": {"ids": entity_ids}}
        data = self._post_json(self.config.graphql_url, payload)
        entities = data.get("data", {}).get("polymer_entities") or []

        records: list[XrayEntityGroupMappingRecord] = []
        for entity in entities:
            if not entity:
                continue
            polymer_entity_id = entity.get("rcsb_id")
            entry_id = (
                entity.get("rcsb_polymer_entity_container_identifiers", {}) or {}
            ).get("entry_id")
            chain_ids = str(
                (entity.get("entity_poly", {}) or {}).get("pdbx_strand_id") or ""
            ).strip()
            chain_id_tuple = tuple(
                item.strip() for item in chain_ids.split(",") if item.strip()
            )
            if not polymer_entity_id or not entry_id or not chain_id_tuple:
                continue

            memberships = entity.get("rcsb_polymer_entity_group_membership") or []
            matched_group_id: str | None = None
            for membership in memberships:
                if not membership:
                    continue
                if (
                    membership.get("aggregation_method")
                    != SEQUENCE_IDENTITY_AGGREGATION_METHOD
                ):
                    continue
                group_id = membership.get("group_id")
                cutoff = self._normalize_similarity_cutoff(
                    membership.get("similarity_cutoff")
                )
                if cutoff is None or not group_id:
                    continue
                if cutoff == similarity_cutoff:
                    matched_group_id = str(group_id)
                    break
            if not matched_group_id:
                continue
            records.append(
                XrayEntityGroupMappingRecord(
                    polymer_entity_id=str(polymer_entity_id),
                    entry_id=str(entry_id),
                    chain_ids=chain_id_tuple,
                    group_id=matched_group_id,
                )
            )
        return records

    def fetch_xray_polymer_entity_candidates_for_ids(
        self, entity_ids: list[str]
    ) -> list[XrayPolymerEntityCandidateRecord]:
        """Fetch candidate X-ray polymer entities and coordinate metadata."""
        if not entity_ids:
            return []
        query = """
        query($ids:[String!]!) {
          polymer_entities(entity_ids:$ids) {
            rcsb_id
            entity_poly {
              pdbx_strand_id
            }
            rcsb_polymer_entity_container_identifiers {
              entry_id
            }
          }
        }
        """
        payload = {"query": query, "variables": {"ids": entity_ids}}
        data = self._post_json(self.config.graphql_url, payload)
        entities = data.get("data", {}).get("polymer_entities") or []

        entity_rows: list[tuple[str, str, tuple[str, ...]]] = []
        for entity in entities:
            if not entity:
                continue
            polymer_entity_id = entity.get("rcsb_id")
            entry_id = (
                entity.get("rcsb_polymer_entity_container_identifiers", {}) or {}
            ).get("entry_id")
            chain_ids = str(
                (entity.get("entity_poly", {}) or {}).get("pdbx_strand_id") or ""
            ).strip()
            chain_id_tuple = tuple(
                item.strip() for item in chain_ids.split(",") if item.strip()
            )
            if not polymer_entity_id or not entry_id or not chain_id_tuple:
                continue
            entity_rows.append(
                (str(polymer_entity_id), str(entry_id), chain_id_tuple)
            )

        resolution_by_entry_id: dict[str, float] = {}
        unknown_entries = sorted({entry_id for _, entry_id, _ in entity_rows})
        for entry_batch in chunked(unknown_entries, self.config.graphql_batch_size):
            resolution_by_entry_id.update(self.fetch_entry_resolution_for_ids(entry_batch))

        records: list[XrayPolymerEntityCandidateRecord] = []
        for polymer_entity_id, entry_id, chain_ids in entity_rows:
            resolution = resolution_by_entry_id.get(entry_id)
            if resolution is None:
                continue
            records.append(
                XrayPolymerEntityCandidateRecord(
                    polymer_entity_id=polymer_entity_id,
                    entry_id=entry_id,
                    chain_ids=chain_ids,
                    resolution_angstrom=resolution,
                )
            )
        return records

    def fetch_sequence_identity_group_ids_for_polymer_entity_ids(
        self, entity_ids: list[str], similarity_cutoff: int
    ) -> set[str]:
        """Fetch sequence-identity group IDs for polymer entity IDs."""
        if not entity_ids:
            return set()
        query = """
        query($ids:[String!]!) {
          polymer_entities(entity_ids:$ids) {
            rcsb_polymer_entity_group_membership {
              aggregation_method
              similarity_cutoff
              group_id
            }
          }
        }
        """
        payload = {"query": query, "variables": {"ids": entity_ids}}
        data = self._post_json(self.config.graphql_url, payload)
        entities = data.get("data", {}).get("polymer_entities") or []
        matching_group_ids: set[str] = set()
        for entity in entities:
            if not entity:
                continue
            memberships = entity.get("rcsb_polymer_entity_group_membership") or []
            group_ids = self._extract_sequence_identity_groups(
                memberships,
                allowed_cutoffs={similarity_cutoff},
            )
            matching_group_ids.update(group_ids.values())
        return matching_group_ids

    def fetch_xray_polymer_entity_ids_by_sequence(
        self,
        sequence: str,
        sequence_identity_percent: int,
    ) -> list[str]:
        """Search RCSB for X-ray polymer entities matching a query sequence."""
        if sequence_identity_percent not in {95, 100}:
            raise ValueError("sequence_identity_percent must be 95 or 100")
        sequence = "".join(sequence.split()).upper()
        if not sequence:
            return []
        if len(sequence) < 10:
            return []

        query = {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "sequence",
                    "parameters": {
                        "evalue_cutoff": 0.1,
                        "identity_cutoff": sequence_identity_percent / 100.0,
                        "sequence_type": "protein",
                        "target": "pdb_protein_sequence",
                        "value": sequence,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "exptl.method",
                        "operator": "exact_match",
                        "value": "X-RAY DIFFRACTION",
                    },
                },
            ],
        }

        entity_ids: list[str] = []
        start = 0
        total_count: int | None = None
        while total_count is None or start < total_count:
            payload = {
                "query": query,
                "return_type": "polymer_entity",
                "request_options": {
                    "paginate": {"start": start, "rows": self.config.page_size},
                    "results_verbosity": "compact",
                    "scoring_strategy": "sequence",
                },
            }
            response: requests.Response | None = None
            last_error: Exception | None = None
            for attempt in range(1, self.config.retries + 1):
                try:
                    response = self.session.post(
                        self.config.search_url,
                        json=payload,
                        timeout=self.config.timeout_seconds,
                    )
                    if response.status_code in {429} or response.status_code >= 500:
                        last_error = requests.HTTPError(
                            f"{response.status_code} Server Error for url: {self.config.search_url}"
                        )
                        LOGGER.warning(
                            "Sequence search request returned HTTP %d (attempt %d/%d)",
                            response.status_code,
                            attempt,
                            self.config.retries,
                        )
                        if attempt < self.config.retries:
                            time.sleep(self.config.backoff_seconds * attempt)
                            continue
                    break
                except requests.RequestException as exc:
                    last_error = exc
                    LOGGER.warning(
                        "Sequence search request failed (attempt %d/%d): %s",
                        attempt,
                        self.config.retries,
                        exc,
                    )
                    if attempt < self.config.retries:
                        time.sleep(self.config.backoff_seconds * attempt)
            if response is None:
                raise RuntimeError(
                    "Sequence search request failed after "
                    f"{self.config.retries} attempts: {last_error}"
                )
            if response.status_code == 204:
                break
            if response.status_code == 400 and "minimum length" in response.text:
                break
            response.raise_for_status()
            data = response.json()
            total_count = int(data.get("total_count", 0))
            batch_ids: list[str] = []
            for item in data.get("result_set") or []:
                if isinstance(item, str):
                    batch_ids.append(item)
                elif isinstance(item, dict) and item.get("identifier"):
                    batch_ids.append(str(item["identifier"]))
            entity_ids.extend(batch_ids)
            start += len(batch_ids)
            if not batch_ids:
                break
        return entity_ids

    def fetch_solution_nmr_weight_records_for_ids(
        self, entry_ids: list[str]
    ) -> list[SolutionNMRWeightRecord]:
        """Build molecular-weight records for SOLUTION NMR entries."""
        query = """
        query($ids:[String!]!) {
          entries(entry_ids:$ids) {
            rcsb_id
            rcsb_accession_info {
              deposit_date
            }
            rcsb_entry_info {
              molecular_weight
            }
          }
        }
        """
        payload = {"query": query, "variables": {"ids": entry_ids}}
        data = self._post_json(self.config.graphql_url, payload)
        entries = data.get("data", {}).get("entries") or []

        records: list[SolutionNMRWeightRecord] = []
        for entry in entries:
            if not entry:
                continue
            entry_id = entry.get("rcsb_id")
            year = extract_year(
                entry.get("rcsb_accession_info", {}).get("deposit_date")
            )
            if not entry_id or year is None:
                continue
            entry_mw_raw = (entry.get("rcsb_entry_info") or {}).get("molecular_weight")
            try:
                molecular_weight_kda = float(entry_mw_raw)
            except (TypeError, ValueError):
                continue

            records.append(
                SolutionNMRWeightRecord(
                    entry_id=entry_id,
                    year=year,
                    molecular_weight_kda=molecular_weight_kda,
                )
            )
        return records

    def iter_solution_nmr_monomer_stride_modeled_first_model_records_for_ids(
        self,
        entry_ids: list[str],
        stride_executable: str,
        pdb_cache_dir: Path,
    ) -> Iterator[SolutionNMRMonomerStrideModeledFirstModelRecord]:
        """Yield STRIDE first-model monomer records for SOLUTION NMR entries."""
        query = """
        query($ids:[String!]!) {
          entries(entry_ids:$ids) {
            rcsb_id
            rcsb_entry_info {
              deposited_model_count
            }
            rcsb_accession_info {
              deposit_date
            }
            polymer_entities {
              entity_poly {
                type
                rcsb_entity_polymer_type
                pdbx_strand_id
              }
              polymer_entity_instances {
                rcsb_id
                rcsb_polymer_instance_feature_summary {
                  type
                  coverage
                }
              }
            }
          }
        }
        """
        payload = {"query": query, "variables": {"ids": entry_ids}}
        data = self._post_json(self.config.graphql_url, payload)
        entries = data.get("data", {}).get("entries") or []

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_map = {
                executor.submit(
                    self._compute_solution_nmr_monomer_stride_modeled_first_model_for_entry,
                    entry=entry,
                    stride_executable=stride_executable,
                    pdb_cache_dir=pdb_cache_dir,
                ): idx
                for idx, entry in enumerate(entries, start=1)
            }
            for future in as_completed(future_map):
                record = future.result()
                if record is None:
                    continue
                yield record

    def _compute_solution_nmr_monomer_stride_modeled_first_model_for_entry(
        self,
        entry: dict[str, Any] | None,
        stride_executable: str,
        pdb_cache_dir: Path,
    ) -> SolutionNMRMonomerStrideModeledFirstModelRecord | None:
        """Compute one entry-level STRIDE modeled-first-model record."""
        if not entry:
            return None
        context = self._extract_solution_nmr_monomer_context(entry)
        if context is None:
            return None
        entry_id, year, _, polymer_entity, chain_id = context

        instances = polymer_entity.get("polymer_entity_instances") or []
        if len(instances) != 1:
            return None
        instance = instances[0] or {}
        try:
            pdb_path = download_pdb_if_needed(
                session=self.session,
                config=self.config,
                cache_dir=pdb_cache_dir,
                entry_id=entry_id,
            )
            chain_map = load_cached_chain_id_map(pdb_cache_dir, entry_id)
            parsed_chain_id = chain_map.get(chain_id, chain_id)
            modeled_auth_seq_ids = parse_first_model_modeled_ca_auth_seq_ids(
                pdb_path=pdb_path,
                chain_id=parsed_chain_id,
            )
        except Exception as exc:
            LOGGER.debug(
                "Skipping STRIDE modeled-first-model entry %s: %s",
                entry_id,
                exc,
            )
            return None

        modeled_sequence_length = len(modeled_auth_seq_ids)
        if modeled_sequence_length <= 0:
            return None
        modeled_start_seq_id = min(modeled_auth_seq_ids)
        modeled_end_seq_id = max(modeled_auth_seq_ids)

        feature_summary = instance.get("rcsb_polymer_instance_feature_summary") or []
        coverage_by_type: dict[str, float] = {}
        for item in feature_summary:
            if not item:
                continue
            feature_type = item.get("type")
            coverage = item.get("coverage")
            if feature_type and coverage is not None:
                coverage_by_type[str(feature_type)] = float(coverage)

        helix_fraction = coverage_by_type.get("HELIX_P", -1.0)
        sheet_fraction = coverage_by_type.get("SHEET", -1.0)
        unassigned_fraction = coverage_by_type.get("UNASSIGNED_SEC_STRUCT", -1.0)
        secondary_fraction = 1.0 - unassigned_fraction
        stride_coverages, _, _ = compute_stride_state_coverages_for_chain_modeled_first_model(
            session=self.session,
            config=self.config,
            cache_dir=pdb_cache_dir,
            entry_id=entry_id,
            chain_id=chain_id,
            modeled_sequence_length=modeled_sequence_length,
            modeled_auth_seq_ids=modeled_auth_seq_ids,
            stride_executable=stride_executable,
        )
        stride_coil_fraction = stride_coverages["C"]
        stride_secondary_percent = (1.0 - stride_coil_fraction) * 100.0

        return SolutionNMRMonomerStrideModeledFirstModelRecord(
            entry_id=entry_id,
            year=year,
            chain_id=chain_id,
            modeled_start_seq_id=modeled_start_seq_id,
            modeled_end_seq_id=modeled_end_seq_id,
            modeled_sequence_length=modeled_sequence_length,
            secondary_structure_percent=secondary_fraction * 100.0,
            helix_fraction=helix_fraction,
            sheet_fraction=sheet_fraction,
            stride_alpha_helix_fraction=stride_coverages["H"],
            stride_3_10_helix_fraction=stride_coverages["G"],
            stride_pi_helix_fraction=stride_coverages["I"],
            stride_beta_strand_fraction=stride_coverages["E"],
            stride_isolated_beta_bridge_fraction=stride_coverages["B"],
            stride_turn_fraction=stride_coverages["T"],
            stride_coil_fraction=stride_coil_fraction,
            stride_secondary_structure_percent=stride_secondary_percent,
        )

    def fetch_solution_nmr_monomer_stride_modeled_first_model_records_for_ids(
        self,
        entry_ids: list[str],
        stride_executable: str,
        pdb_cache_dir: Path,
    ) -> list[SolutionNMRMonomerStrideModeledFirstModelRecord]:
        """Fetch all STRIDE modeled-first-model records for SOLUTION NMR monomers."""
        return list(
            self.iter_solution_nmr_monomer_stride_modeled_first_model_records_for_ids(
                entry_ids=entry_ids,
                stride_executable=stride_executable,
                pdb_cache_dir=pdb_cache_dir,
            )
        )

    def fetch_solution_nmr_monomer_quality_records_for_ids(
        self, entry_ids: list[str]
    ) -> list[SolutionNMRMonomerQualityRecord]:
        """Fetch validation quality metrics for SOLUTION NMR monomers."""
        query = """
        query($ids:[String!]!) {
          entries(entry_ids:$ids) {
            rcsb_id
            rcsb_entry_info {
              deposited_model_count
            }
            rcsb_accession_info {
              deposit_date
            }
            pdbx_vrpt_summary_geometry {
              clashscore
              percent_ramachandran_outliers
              percent_rotamer_outliers
            }
            polymer_entities {
              entity_poly {
                type
                rcsb_entity_polymer_type
                pdbx_strand_id
              }
            }
          }
        }
        """
        payload = {"query": query, "variables": {"ids": entry_ids}}
        data = self._post_json(self.config.graphql_url, payload)
        entries = data.get("data", {}).get("entries") or []
        records: list[SolutionNMRMonomerQualityRecord] = []

        for entry in entries:
            if not entry:
                continue
            context = self._extract_solution_nmr_monomer_context(entry)
            if context is None:
                continue
            entry_id, year, _, _, _ = context
            quality_items = entry.get("pdbx_vrpt_summary_geometry") or []
            if not quality_items:
                continue
            quality = quality_items[0] or {}
            clashscore = quality.get("clashscore")
            rama = quality.get("percent_ramachandran_outliers")
            rotamer = quality.get("percent_rotamer_outliers")
            if clashscore is None or rama is None or rotamer is None:
                continue

            records.append(
                SolutionNMRMonomerQualityRecord(
                    entry_id=entry_id,
                    year=year,
                    clashscore=float(clashscore),
                    ramachandran_outliers_percent=float(rama),
                    sidechain_outliers_percent=float(rotamer),
                )
            )
        return records

    def fetch_solution_nmr_monomer_modeled_first_model_seed_records_for_ids(
        self, entry_ids: list[str]
    ) -> list[SolutionNMRMonomerModeledFirstModelSeedRecord]:
        """Fetch seed data needed for modeled-first-model precision analysis."""
        query = """
        query($ids:[String!]!) {
          entries(entry_ids:$ids) {
            rcsb_id
            rcsb_entry_info {
              deposited_model_count
            }
            rcsb_accession_info {
              deposit_date
            }
            polymer_entities {
              entity_poly {
                type
                rcsb_entity_polymer_type
                pdbx_strand_id
              }
            }
          }
        }
        """
        payload = {"query": query, "variables": {"ids": entry_ids}}
        data = self._post_json(self.config.graphql_url, payload)
        entries = data.get("data", {}).get("entries") or []
        records: list[SolutionNMRMonomerModeledFirstModelSeedRecord] = []

        for entry in entries:
            if not entry:
                continue
            context = self._extract_solution_nmr_monomer_context(entry)
            if context is None:
                continue
            entry_id, year, _, _, chain_id = context

            records.append(
                SolutionNMRMonomerModeledFirstModelSeedRecord(
                    entry_id=str(entry_id),
                    year=year,
                    chain_id=chain_id,
                )
            )
        return records

    def fetch_solution_nmr_monomer_xray_homolog_seed_records_for_ids(
        self, entry_ids: list[str]
    ) -> list[SolutionNMRMonomerXrayHomologSeedRecord]:
        """Fetch seed data needed to search X-ray homologs for NMR monomers."""
        query = """
        query($ids:[String!]!) {
          entries(entry_ids:$ids) {
            rcsb_id
            rcsb_entry_info {
              deposited_model_count
            }
            rcsb_accession_info {
              deposit_date
            }
            polymer_entities {
              entity_poly {
                type
                rcsb_entity_polymer_type
                pdbx_strand_id
              }
            }
          }
        }
        """
        payload = {"query": query, "variables": {"ids": entry_ids}}
        data = self._post_json(self.config.graphql_url, payload)
        entries = data.get("data", {}).get("entries") or []
        records: list[SolutionNMRMonomerXrayHomologSeedRecord] = []

        for entry in entries:
            if not entry:
                continue
            context = self._extract_solution_nmr_monomer_context(entry)
            if context is None:
                continue
            entry_id, year, _, _, chain_id = context
            records.append(
                SolutionNMRMonomerXrayHomologSeedRecord(
                    entry_id=str(entry_id),
                    year=year,
                    chain_id=chain_id,
                )
            )
        return records


class PDBMethodYearlyCollector:
    def __init__(self, client: RCSBClient, config: CollectorConfig) -> None:
        """Initialize a yearly experimental-method count collector."""
        self.client = client
        self.config = config

    def _fetch_method_records(
        self, method: ExperimentalMethod
    ) -> list[YearlyCountRecord]:
        """Fetch yearly count records for one experimental method."""
        entry_ids = sorted(
            {
                entry_id
                for query_value in method.query_values
                for entry_id in self.client.fetch_entry_ids_for_method(
                    method_label=method.label,
                    query_value=query_value,
                    require_protein_entities=True,
                )
            }
        )
        LOGGER.info("%s: total unique IDs collected: %d", method.label, len(entry_ids))
        year_counter: Counter[int] = Counter()

        batches = list(chunked(entry_ids, self.config.graphql_batch_size))
        for batch_dates in collect_batch_results(
            batches=batches,
            max_workers=self.config.max_workers,
            fetch_fn=self.client.fetch_deposit_dates_for_ids,
            progress_label=method.label,
        ):
            years = filter(
                None, (extract_year(date_value) for date_value in batch_dates)
            )
            year_counter.update(years)

        return [
            YearlyCountRecord(year=year, method=method.label, count=count)
            for year, count in sorted(year_counter.items())
        ]

    def collect(self, methods: Iterable[ExperimentalMethod]) -> list[YearlyCountRecord]:
        """Collect yearly counts for all requested experimental methods."""
        records: list[YearlyCountRecord] = []
        for method in methods:
            records.extend(self._fetch_method_records(method))
        return sorted(records, key=lambda record: (record.year, record.method))


class SolutionNMRProgramYearlyCollector:
    def __init__(
        self,
        client: RCSBClient,
        config: CollectorConfig,
        cache_dir: Path,
        download_missing: bool = True,
    ) -> None:
        """Initialize the SOLUTION NMR refinement-program trend collector."""
        self.client = client
        self.config = config
        self.cache_dir = cache_dir
        self.download_missing = download_missing
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _fetch_entry_years(self, entry_ids: list[str]) -> dict[str, int]:
        """Fetch deposit years for SOLUTION NMR entries."""
        entry_year_by_id: dict[str, int] = {}
        batches = list(chunked(entry_ids, self.config.graphql_batch_size))
        for batch_year_map in collect_batch_results(
            batches=batches,
            max_workers=self.config.max_workers,
            fetch_fn=self.client.fetch_deposit_year_by_entry_id_for_ids,
            progress_label="SOLUTION NMR programs years",
        ):
            entry_year_by_id.update(batch_year_map)
        return entry_year_by_id

    def _load_programs_for_entry(self, entry_id: str) -> set[str] | None:
        """Load refinement program labels from one cached PDB file."""
        cached_pdb_path = self.cache_dir / f"{entry_id}.pdb"
        if cached_pdb_path.exists() and cached_pdb_path.stat().st_size > 0:
            return extract_refinement_programs_from_pdb(cached_pdb_path)
        if not self.download_missing:
            return None
        try:
            downloaded_pdb_path = download_pdb_if_needed(
                session=self.client.session,
                config=self.config,
                cache_dir=self.cache_dir,
                entry_id=entry_id,
            )
        except Exception as exc:
            LOGGER.warning("Failed to get PDB for %s: %s", entry_id, exc)
            return None
        return extract_refinement_programs_from_pdb(downloaded_pdb_path)

    def collect(self) -> list[SolutionNMRProgramYearlyCountRecord]:
        """Collect yearly SOLUTION NMR refinement-program usage counts."""
        entry_ids = fetch_solution_nmr_entry_ids(
            client=self.client,
            log_label="SOLUTION NMR programs",
        )
        if not entry_ids:
            return []

        entry_year_by_id = self._fetch_entry_years(entry_ids)
        missing_year_count = 0
        missing_program_count = 0
        skipped_uncached_count = 0
        yearly_program_counter: Counter[tuple[int, str]] = Counter()

        entry_year_pairs = [
            (entry_id, entry_year_by_id.get(entry_id))
            for entry_id in entry_ids
        ]

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_map = {
                executor.submit(self._load_programs_for_entry, entry_id): (
                    entry_id,
                    year,
                )
                for entry_id, year in entry_year_pairs
                if year is not None
            }
            missing_year_count = len(entry_ids) - len(future_map)
            total = len(entry_ids)
            processed = 0
            for future in as_completed(future_map):
                entry_id, year = future_map[future]
                programs = future.result()
                if programs is None:
                    cached_pdb_path = self.cache_dir / f"{entry_id}.pdb"
                    if (
                        not self.download_missing
                        and (not cached_pdb_path.exists() or cached_pdb_path.stat().st_size <= 0)
                    ):
                        skipped_uncached_count += 1
                    else:
                        missing_program_count += 1
                elif not programs:
                    missing_program_count += 1
                else:
                    for program in programs:
                        yearly_program_counter[(year, program)] += 1

                processed += 1
                if processed % 500 == 0 or processed == len(future_map):
                    LOGGER.info(
                        "SOLUTION NMR programs: parsed %d/%d entries",
                        processed + missing_year_count,
                        total,
                    )

        LOGGER.info(
            (
                "SOLUTION NMR programs: entries=%d, missing_year=%d, "
                "without_program=%d, skipped_uncached=%d, unique_programs=%d"
            ),
            len(entry_ids),
            missing_year_count,
            missing_program_count,
            skipped_uncached_count,
            len({program for _, program in yearly_program_counter.keys()}),
        )
        return sorted(
            (
                SolutionNMRProgramYearlyCountRecord(
                    year=year,
                    program=program,
                    count=count,
                )
                for (year, program), count in yearly_program_counter.items()
            ),
            key=lambda record: (record.year, record.program),
        )


class SolutionNMRMonomerProgramClusterCollector:
    def __init__(
        self,
        quality_records: list[SolutionNMRMonomerQualityRecord],
        cache_dir: Path,
        max_workers: int,
    ) -> None:
        """Initialize the monomer refinement-program cluster collector."""
        self.quality_records = quality_records
        self.cache_dir = cache_dir
        self.max_workers = max(1, max_workers)

    def _load_program_text(self, entry_id: str) -> str:
        """Load raw refinement program text for one entry."""
        pdb_path = self.cache_dir / f"{entry_id}.pdb"
        if not pdb_path.exists() or pdb_path.stat().st_size <= 0:
            return ""
        return extract_raw_refinement_program_text_from_pdb(pdb_path)

    def collect(
        self,
    ) -> tuple[
        list[SolutionNMRMonomerProgramClusterAssignmentRecord],
        list[SolutionNMRMonomerProgramClusterSummaryRecord],
    ]:
        """Collect program-cluster assignments for SOLUTION NMR monomers."""
        if not self.quality_records:
            return [], []

        quality_by_entry_id = {
            record.entry_id: record for record in self.quality_records
        }
        assignments: list[SolutionNMRMonomerProgramClusterAssignmentRecord] = []
        summary_totals: dict[
            tuple[int, str],
            dict[str, float | int],
        ] = {}
        missing_cache_count = 0
        empty_program_count = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_map = {
                executor.submit(self._load_program_text, entry_id): entry_id
                for entry_id in quality_by_entry_id
            }
            total = len(future_map)
            processed = 0
            for future in as_completed(future_map):
                entry_id = future_map[future]
                quality_record = quality_by_entry_id[entry_id]
                program_text = future.result()
                if not program_text:
                    pdb_path = self.cache_dir / f"{entry_id}.pdb"
                    if not pdb_path.exists() or pdb_path.stat().st_size <= 0:
                        missing_cache_count += 1
                    empty_program_count += 1

                clusters = extract_solution_nmr_program_clusters(program_text)
                for cluster_id, cluster_name in clusters:
                    assignments.append(
                        SolutionNMRMonomerProgramClusterAssignmentRecord(
                            entry_id=quality_record.entry_id,
                            year=quality_record.year,
                            cluster_id=cluster_id,
                            cluster_name=cluster_name,
                            has_program_text=bool(program_text),
                            program_text=program_text,
                        )
                    )
                    key = (quality_record.year, cluster_id)
                    total_row = summary_totals.setdefault(
                        key,
                        {
                            "count": 0,
                            "rama_sum": 0.0,
                            "side_sum": 0.0,
                            "clash_sum": 0.0,
                        },
                    )
                    total_row["count"] += 1
                    total_row["rama_sum"] += (
                        quality_record.ramachandran_outliers_percent
                    )
                    total_row["side_sum"] += quality_record.sidechain_outliers_percent
                    total_row["clash_sum"] += quality_record.clashscore

                processed += 1
                if processed % 500 == 0 or processed == total:
                    LOGGER.info(
                        "SOLUTION NMR monomer program clusters: processed %d/%d entries",
                        processed,
                        total,
                    )

        LOGGER.info(
            (
                "SOLUTION NMR monomer program clusters: entries=%d, "
                "missing_cache=%d, empty_program=%d"
            ),
            len(self.quality_records),
            missing_cache_count,
            empty_program_count,
        )

        assignments = sorted(assignments, key=lambda r: (r.year, r.entry_id))
        years = sorted({record.year for record in self.quality_records})
        summaries: list[SolutionNMRMonomerProgramClusterSummaryRecord] = []
        for year in years:
            for cluster_id, cluster_name in PROGRAM_CLUSTER_DEFINITIONS:
                totals = summary_totals.get((year, cluster_id))
                if totals is None:
                    summaries.append(
                        SolutionNMRMonomerProgramClusterSummaryRecord(
                            year=year,
                            cluster_id=cluster_id,
                            cluster_name=cluster_name,
                            structure_count=0,
                            avg_ramachandran_outliers_percent=None,
                            avg_sidechain_outliers_percent=None,
                            avg_clashscore=None,
                        )
                    )
                    continue
                count = int(totals["count"])
                summaries.append(
                    SolutionNMRMonomerProgramClusterSummaryRecord(
                        year=year,
                        cluster_id=cluster_id,
                        cluster_name=cluster_name,
                        structure_count=count,
                        avg_ramachandran_outliers_percent=float(totals["rama_sum"])
                        / count,
                        avg_sidechain_outliers_percent=float(totals["side_sum"])
                        / count,
                        avg_clashscore=float(totals["clash_sum"]) / count,
                    )
                )
        return assignments, summaries


class MembraneProteinYearlyCollector:
    def __init__(self, client: RCSBClient, config: CollectorConfig) -> None:
        """Initialize the membrane-protein yearly count collector."""
        self.client = client
        self.config = config

    def _fetch_membrane_entry_ids(self) -> list[str]:
        """Fetch entry IDs annotated as membrane proteins."""
        entry_ids = sorted(
            set(
                self.client.fetch_entry_ids_for_membrane_annotations(
                    MEMBRANE_ANNOTATION_TYPES
                )
            )
        )
        LOGGER.info("Membrane proteins: total unique IDs collected: %d", len(entry_ids))
        return entry_ids

    def _count_entry_years(
        self,
        entry_ids: list[str],
        progress_label: str,
    ) -> Counter[int]:
        """Count entries by deposit year from an entry ID list."""
        year_counter: Counter[int] = Counter()

        batches = list(chunked(entry_ids, self.config.graphql_batch_size))
        for batch_dates in collect_batch_results(
            batches=batches,
            max_workers=self.config.max_workers,
            fetch_fn=self.client.fetch_deposit_dates_for_ids,
            progress_label=progress_label,
        ):
            years = filter(
                None, (extract_year(date_value) for date_value in batch_dates)
            )
            year_counter.update(years)
        return year_counter

    def collect(self) -> list[MembraneYearlyCountRecord]:
        """Collect yearly membrane-protein entry counts."""
        entry_ids = self._fetch_membrane_entry_ids()
        year_counter = self._count_entry_years(
            entry_ids=entry_ids,
            progress_label="Membrane proteins",
        )

        return [
            MembraneYearlyCountRecord(year=year, count=count)
            for year, count in sorted(year_counter.items())
        ]

    def collect_by_method(
        self,
        methods: Iterable[ExperimentalMethod],
    ) -> list[YearlyCountRecord]:
        """Collect yearly membrane-protein counts split by experimental method."""
        membrane_entry_ids = set(self._fetch_membrane_entry_ids())
        records: list[YearlyCountRecord] = []

        for method in methods:
            method_entry_ids = sorted(
                {
                    entry_id
                    for query_value in method.query_values
                    for entry_id in self.client.fetch_entry_ids_for_method(
                        method_label=method.label,
                        query_value=query_value,
                        require_protein_entities=True,
                    )
                }
            )
            membrane_method_entry_ids = [
                entry_id for entry_id in method_entry_ids if entry_id in membrane_entry_ids
            ]
            LOGGER.info(
                "Membrane proteins %s: kept %d/%d method entries",
                method.label,
                len(membrane_method_entry_ids),
                len(method_entry_ids),
            )
            year_counter = self._count_entry_years(
                entry_ids=membrane_method_entry_ids,
                progress_label=f"Membrane proteins {method.label}",
            )
            records.extend(
                YearlyCountRecord(year=year, method=method.label, count=count)
                for year, count in sorted(year_counter.items())
            )

        return sorted(records, key=lambda record: (record.year, record.method))


class SolutionNMRWeightCollector:
    def __init__(self, client: RCSBClient, config: CollectorConfig) -> None:
        """Initialize the SOLUTION NMR molecular-weight collector."""
        self.client = client
        self.config = config

    def collect(self) -> list[SolutionNMRWeightRecord]:
        """Collect molecular-weight records for SOLUTION NMR entries."""
        entry_ids = fetch_solution_nmr_entry_ids(
            client=self.client,
            log_label="SOLUTION NMR",
        )
        batches = list(chunked(entry_ids, self.config.graphql_batch_size))

        records: list[SolutionNMRWeightRecord] = []
        for batch_records in collect_batch_results(
            batches=batches,
            max_workers=self.config.max_workers,
            fetch_fn=self.client.fetch_solution_nmr_weight_records_for_ids,
            progress_label="SOLUTION NMR weights",
        ):
            records.extend(batch_records)
        return sorted(records, key=lambda record: (record.year, record.entry_id))


class SolutionNMRMonomerStrideModeledFirstModelCollector:
    def __init__(
        self,
        client: RCSBClient,
        config: CollectorConfig,
        stride_executable: str,
        cache_dir: Path,
    ) -> None:
        """Initialize streaming STRIDE modeled-first-model collection."""
        self.client = client
        self.config = config
        self.stride_executable = stride_executable
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def iter_batches(
        self,
    ) -> Iterator[list[SolutionNMRMonomerStrideModeledFirstModelRecord]]:
        """Yield STRIDE modeled-first-model records batch by batch."""
        entry_ids = fetch_solution_nmr_entry_ids(
            client=self.client,
            log_label="SOLUTION NMR monomer-stride-modeled-first-model",
        )
        batches = list(chunked(entry_ids, self.config.graphql_batch_size))
        if not batches:
            return

        for batch_idx, batch in enumerate(batches, start=1):
            batch_records = (
                self.client.fetch_solution_nmr_monomer_stride_modeled_first_model_records_for_ids(
                    entry_ids=batch,
                    stride_executable=self.stride_executable,
                    pdb_cache_dir=self.cache_dir,
                )
            )
            LOGGER.info(
                (
                    "SOLUTION NMR monomer-stride-modeled-first-model: "
                    "processed batch %d/%d (entries: %d)"
                ),
                batch_idx,
                len(batches),
                len(batch_records),
            )
            yield batch_records

    def iter_records(self) -> Iterator[SolutionNMRMonomerStrideModeledFirstModelRecord]:
        """Yield individual STRIDE modeled-first-model records."""
        for batch_records in self.iter_batches():
            for record in batch_records:
                yield record

    def collect(self) -> list[SolutionNMRMonomerStrideModeledFirstModelRecord]:
        """Collect all STRIDE modeled-first-model records into a list."""
        records = list(self.iter_records())
        return sorted(records, key=lambda record: (record.year, record.entry_id))


class SolutionNMRMonomerPrecisionCollector:
    def __init__(
        self,
        client: RCSBClient,
        config: CollectorConfig,
        cache_dir: Path,
        precision_workers: int,
    ) -> None:
        """Initialize shared state for monomer precision collectors."""
        self.client = client
        self.config = config
        self.cache_dir = cache_dir
        self.precision_workers = max(1, precision_workers)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _download_pdb_if_needed(self, entry_id: str) -> Path:
        """Download or reuse the PDB file needed for precision calculations."""
        return download_pdb_if_needed(
            session=self.client.session,
            config=self.config,
            cache_dir=self.cache_dir,
            entry_id=entry_id,
        )

    @staticmethod
    def _compute_mean_rmsd_to_average(
        pdb_path: Path,
        chain_id: str,
        start_seq_id: int,
        end_seq_id: int,
    ) -> tuple[tuple[int, int, int, float] | None, str | None]:
        """Compute ensemble CA RMSD to per-residue mean coordinates."""
        model_maps, raw_ca_counts_per_model = parse_models_ca_coords_with_stats(
            pdb_path=pdb_path,
            chain_id=chain_id,
            start_seq_id=start_seq_id,
            end_seq_id=end_seq_id,
        )
        if len(model_maps) < 2:
            return (
                None,
                f"fewer than 2 coordinate models in core range (found {len(model_maps)})",
            )

        common_resids = set(model_maps[0].keys())
        for model_map in model_maps[1:]:
            common_resids &= set(model_map.keys())
        if len(common_resids) < 3:
            return (
                None,
                (
                    "fewer than 3 CA residues common to all models in core range "
                    f"(found {len(common_resids)})"
                ),
            )
        sorted_resids = sorted(common_resids)

        coords = np.asarray(
            [[model_map[resid] for resid in sorted_resids] for model_map in model_maps],
            dtype=float,
        )
        aligned_coords = _coordinates_aligned_to_first_model(coords)
        ensemble_rmsd = _ca_rmsd_to_mean_structure(aligned_coords)

        raw_ca_counts_common = [
            sum(raw_counts.get(resid, 0) for resid in sorted_resids)
            for raw_counts in raw_ca_counts_per_model
        ]
        n_ca_core_raw = (
            min(raw_ca_counts_common)
            if raw_ca_counts_common
            else len(sorted_resids)
        )
        return (
            (
                len(model_maps),
                len(sorted_resids),
                int(n_ca_core_raw),
                ensemble_rmsd,
            ),
            None,
        )

    def _build_record_from_core_range(
        self,
        pdb_path: Path,
        entry_id: str,
        year: int,
        chain_id: str,
        core_start_seq_id: int,
        core_end_seq_id: int,
        parsed_chain_id: str | None = None,
    ) -> SolutionNMRMonomerPrecisionRecord | None:
        """Build a precision record from a modeled residue core range."""
        result, skip_reason = self._compute_mean_rmsd_to_average(
            pdb_path=pdb_path,
            chain_id=parsed_chain_id or chain_id,
            start_seq_id=core_start_seq_id,
            end_seq_id=core_end_seq_id,
        )
        if result is None:
            LOGGER.info(
                "Skipping precision entry %s chain %s: %s",
                entry_id,
                chain_id,
                skip_reason,
            )
            return None
        n_models, n_ca_core_used, n_ca_core_raw, mean_rmsd = result
        return SolutionNMRMonomerPrecisionRecord(
            entry_id=entry_id,
            year=year,
            chain_id=chain_id,
            core_start_seq_id=core_start_seq_id,
            core_end_seq_id=core_end_seq_id,
            n_models=n_models,
            n_ca_core_used=n_ca_core_used,
            n_ca_core_raw=n_ca_core_raw,
            mean_rmsd_angstrom=mean_rmsd,
        )

class SolutionNMRMonomerPrecisionStrideModeledFirstModelCollector(
    SolutionNMRMonomerPrecisionCollector
):
    def __init__(
        self,
        client: RCSBClient,
        config: CollectorConfig,
        cache_dir: Path,
        precision_workers: int,
        stride_executable: str,
    ) -> None:
        """Initialize STRIDE-core modeled-first-model precision collection."""
        super().__init__(
            client=client,
            config=config,
            cache_dir=cache_dir,
            precision_workers=precision_workers,
        )
        self.stride_executable = stride_executable

    def _compute_record_from_seed(
        self,
        seed: SolutionNMRMonomerModeledFirstModelSeedRecord,
    ) -> SolutionNMRMonomerPrecisionRecord | None:
        """Compute one STRIDE-core precision record from seed metadata."""
        try:
            pdb_path = self._download_pdb_if_needed(seed.entry_id)
            chain_map = load_cached_chain_id_map(self.cache_dir, seed.entry_id)
            parsed_chain_id = chain_map.get(seed.chain_id, seed.chain_id)
            modeled_auth_seq_ids = parse_first_model_modeled_ca_auth_seq_ids(
                pdb_path=pdb_path,
                chain_id=parsed_chain_id,
            )
            if not modeled_auth_seq_ids:
                LOGGER.info(
                    "Skipping precision entry %s chain %s: no usable first-model modeled CA residues",
                    seed.entry_id,
                    seed.chain_id,
                )
                return None
            core_range = compute_stride_core_range_for_modeled_auth_seq_ids_in_first_model(
                pdb_path=pdb_path,
                chain_id=parsed_chain_id,
                modeled_auth_seq_ids=modeled_auth_seq_ids,
                stride_executable=self.stride_executable,
            )
            if core_range is None:
                LOGGER.info(
                    "Skipping precision entry %s chain %s: STRIDE found no modeled core residues",
                    seed.entry_id,
                    seed.chain_id,
                )
                return None
            core_start, core_end = core_range
            return self._build_record_from_core_range(
                pdb_path=pdb_path,
                entry_id=seed.entry_id,
                year=seed.year,
                chain_id=seed.chain_id,
                core_start_seq_id=core_start,
                core_end_seq_id=core_end,
                parsed_chain_id=parsed_chain_id,
            )
        except Exception as exc:
            LOGGER.warning(
                "STRIDE modeled-first-model precision calculation failed for %s: %s",
                seed.entry_id,
                exc,
            )
            return None

    def collect(
        self,
        max_entries: int | None = None,
        skip_entry_ids: set[str] | None = None,
        on_record: Callable[[SolutionNMRMonomerPrecisionRecord], None] | None = None,
    ) -> list[SolutionNMRMonomerPrecisionRecord]:
        """Collect STRIDE-core precision records for all eligible seeds."""
        skip_entry_ids = skip_entry_ids or set()
        entry_ids = fetch_solution_nmr_entry_ids(
            client=self.client,
            log_label="SOLUTION NMR precision STRIDE modeled-first-model",
        )

        batches = list(chunked(entry_ids, self.config.graphql_batch_size))
        seed_records: list[SolutionNMRMonomerModeledFirstModelSeedRecord] = []
        for batch_seeds in collect_batch_results(
            batches=batches,
            max_workers=self.config.max_workers,
            fetch_fn=(
                self.client.fetch_solution_nmr_monomer_modeled_first_model_seed_records_for_ids
            ),
            progress_label="SOLUTION NMR precision STRIDE modeled-first-model seeds",
        ):
            seed_records.extend(batch_seeds)

        filtered_seeds = [
            record for record in seed_records if record.entry_id not in skip_entry_ids
        ]
        filtered_seeds = sorted(filtered_seeds, key=lambda r: (r.year, r.entry_id))
        if max_entries is not None:
            filtered_seeds = filtered_seeds[: max(0, max_entries)]
        LOGGER.info(
            "SOLUTION NMR precision STRIDE modeled-first-model: entries to process after filters: %d",
            len(filtered_seeds),
        )

        precision_records: list[SolutionNMRMonomerPrecisionRecord] = []
        with ThreadPoolExecutor(max_workers=self.precision_workers) as executor:
            future_map = {
                executor.submit(self._compute_record_from_seed, seed): idx
                for idx, seed in enumerate(filtered_seeds, start=1)
            }
            total = len(future_map)
            for future in as_completed(future_map):
                record = future.result()
                if record is not None:
                    precision_records.append(record)
                    if on_record is not None:
                        on_record(record)
                idx = future_map[future]
                if total > 0 and (idx % 50 == 0 or idx == total):
                    LOGGER.info(
                        "SOLUTION NMR precision STRIDE modeled-first-model RMSD: processed %d/%d entries",
                        idx,
                        total,
                    )

        return sorted(precision_records, key=lambda r: (r.year, r.entry_id))


class SolutionNMRMonomerQualityCollector:
    def __init__(self, client: RCSBClient, config: CollectorConfig) -> None:
        """Initialize the SOLUTION NMR monomer quality collector."""
        self.client = client
        self.config = config

    def collect(self) -> list[SolutionNMRMonomerQualityRecord]:
        """Collect validation quality records for SOLUTION NMR monomers."""
        entry_ids = fetch_solution_nmr_entry_ids(
            client=self.client,
            log_label="SOLUTION NMR quality",
        )
        batches = list(chunked(entry_ids, self.config.graphql_batch_size))
        records: list[SolutionNMRMonomerQualityRecord] = []

        for batch_records in collect_batch_results(
            batches=batches,
            max_workers=self.config.max_workers,
            fetch_fn=self.client.fetch_solution_nmr_monomer_quality_records_for_ids,
            progress_label="SOLUTION NMR quality",
        ):
            records.extend(batch_records)
        return sorted(records, key=lambda r: (r.year, r.entry_id))


class SolutionNMRMonomerXrayHomologCollector:
    def __init__(
        self,
        client: RCSBClient,
        config: CollectorConfig,
        stride_executable: str,
        cache_dir: Path,
    ) -> None:
        """Initialize X-ray homolog collection for SOLUTION NMR monomers."""
        self.client = client
        self.config = config
        self.stride_executable = stride_executable
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _entry_ids_from_polymer_entity_ids(
        entity_ids: Sequence[str],
    ) -> tuple[str, ...]:
        """Extract entry IDs from polymer entity identifiers."""
        seen: set[str] = set()
        entry_ids: list[str] = []
        for entity_id in entity_ids:
            entry_id = str(entity_id).split("_", 1)[0].strip()
            if not entry_id or entry_id in seen:
                continue
            seen.add(entry_id)
            entry_ids.append(entry_id)
        return tuple(entry_ids)

    def _build_stride_core_query_sequence(
        self,
        seed: SolutionNMRMonomerXrayHomologSeedRecord,
    ) -> tuple[str, int, int, list[CAResidueRecord]] | None:
        """Build the STRIDE-core query sequence used for homolog searches."""
        pdb_path = download_pdb_if_needed(
            session=self.client.session,
            config=self.config,
            cache_dir=self.cache_dir,
            entry_id=seed.entry_id,
        )
        chain_map = load_cached_chain_id_map(self.cache_dir, seed.entry_id)
        parsed_chain_id = chain_map.get(seed.chain_id, seed.chain_id)
        modeled_auth_seq_ids = parse_first_model_modeled_ca_auth_seq_ids(
            pdb_path=pdb_path,
            chain_id=parsed_chain_id,
        )
        if not modeled_auth_seq_ids:
            return None
        core_range = compute_stride_core_range_for_modeled_auth_seq_ids_in_first_model(
            pdb_path=pdb_path,
            chain_id=parsed_chain_id,
            modeled_auth_seq_ids=modeled_auth_seq_ids,
            stride_executable=self.stride_executable,
        )
        if core_range is None:
            return None
        core_start, core_end = core_range
        nmr_core_residues = parse_first_model_ca_residues(
            pdb_path=pdb_path,
            chain_id=parsed_chain_id,
            start_seq_id=core_start,
            end_seq_id=core_end,
            include_hetatm=False,
        )
        if len(nmr_core_residues) <= 10:
            return None
        identities = [record.identity for record in nmr_core_residues]
        if any(len(identity) != 1 or not identity.isalpha() for identity in identities):
            return None
        query_sequence = "".join(identities)
        if not query_sequence:
            return None
        return query_sequence, core_start, core_end, nmr_core_residues

    def _xray_candidate_has_modeled_core_match(
        self,
        nmr_core_residues: list[CAResidueRecord],
        candidate: XrayPolymerEntityCandidateRecord,
        sequence_identity_percent: int,
    ) -> bool:
        """Check whether an X-ray candidate models the NMR core sequence."""
        try:
            (
                xray_pdb_path,
                xray_chain_map,
            ) = download_pdb_chain_subset_if_needed(
                session=self.client.session,
                config=self.config,
                cache_dir=self.cache_dir,
                entry_id=candidate.entry_id,
                chain_ids=candidate.chain_ids,
            )
        except Exception as exc:
            LOGGER.debug(
                "Skipping X-ray homolog candidate %s while pruning modeled core: %s",
                candidate.polymer_entity_id,
                exc,
            )
            return False

        for xray_chain_id in candidate.chain_ids:
            parsed_xray_chain_id = xray_chain_map.get(xray_chain_id, xray_chain_id)
            xray_residues = parse_first_model_ca_residues(
                pdb_path=xray_pdb_path,
                chain_id=parsed_xray_chain_id,
                include_hetatm=True,
            )
            if find_modeled_ca_core_identity_matches(
                nmr_residues=nmr_core_residues,
                xray_residues=xray_residues,
                sequence_identity_percent=sequence_identity_percent,
            ):
                return True
        return False

    def _filter_modeled_xray_homolog_entity_ids(
        self,
        xray_entity_ids: tuple[str, ...],
        nmr_core_residues: list[CAResidueRecord],
        sequence_identity_percent: int,
    ) -> tuple[str, ...]:
        """Filter homolog candidates to those modeling the required core."""
        if not xray_entity_ids:
            return tuple()

        candidates = self.client.fetch_xray_polymer_entity_candidates_for_ids(
            list(xray_entity_ids)
        )
        candidate_by_entity_id = {
            candidate.polymer_entity_id: candidate for candidate in candidates
        }

        filtered_entity_ids: list[str] = []
        for entity_id in xray_entity_ids:
            candidate = candidate_by_entity_id.get(entity_id)
            if candidate is None:
                continue
            if self._xray_candidate_has_modeled_core_match(
                nmr_core_residues=nmr_core_residues,
                candidate=candidate,
                sequence_identity_percent=sequence_identity_percent,
            ):
                filtered_entity_ids.append(entity_id)
        return tuple(filtered_entity_ids)

    def _build_record(
        self,
        seed: SolutionNMRMonomerXrayHomologSeedRecord,
        sequence_identity_percent: int,
        core_query: (
            tuple[str, int, int, list[CAResidueRecord]] | None | object
        ) = _MISSING,
    ) -> SolutionNMRMonomerXrayHomologRecord:
        """Build one X-ray homolog summary record from a seed."""
        if core_query is _MISSING:
            core_query = self._build_stride_core_query_sequence(seed)
        if core_query is None:
            return SolutionNMRMonomerXrayHomologRecord(
                entry_id=seed.entry_id,
                year=seed.year,
                sequence_identity_percent=sequence_identity_percent,
                nmr_core_start_seq_id=None,
                nmr_core_end_seq_id=None,
                nmr_query_sequence_length=0,
                xray_homolog_entry_ids=tuple(),
                xray_homolog_entity_ids=tuple(),
                has_xray_homolog=False,
            )
        query_sequence, core_start, core_end, nmr_core_residues = core_query
        raw_xray_entity_ids = tuple(
            self.client.fetch_xray_polymer_entity_ids_by_sequence(
                sequence=query_sequence,
                sequence_identity_percent=sequence_identity_percent,
            )
        )
        xray_entity_ids = self._filter_modeled_xray_homolog_entity_ids(
            xray_entity_ids=raw_xray_entity_ids,
            nmr_core_residues=nmr_core_residues,
            sequence_identity_percent=sequence_identity_percent,
        )
        xray_entry_ids = self._entry_ids_from_polymer_entity_ids(xray_entity_ids)
        return SolutionNMRMonomerXrayHomologRecord(
            entry_id=seed.entry_id,
            year=seed.year,
            sequence_identity_percent=sequence_identity_percent,
            nmr_core_start_seq_id=core_start,
            nmr_core_end_seq_id=core_end,
            nmr_query_sequence_length=len(query_sequence),
            xray_homolog_entry_ids=xray_entry_ids,
            xray_homolog_entity_ids=xray_entity_ids,
            has_xray_homolog=bool(xray_entity_ids),
        )

    def _build_record_pair(
        self,
        seed: SolutionNMRMonomerXrayHomologSeedRecord,
    ) -> tuple[SolutionNMRMonomerXrayHomologRecord, SolutionNMRMonomerXrayHomologRecord]:
        """Build current and historical homolog records for one seed."""
        core_query = self._build_stride_core_query_sequence(seed)
        return (
            self._build_record(seed, sequence_identity_percent=95, core_query=core_query),
            self._build_record(seed, sequence_identity_percent=100, core_query=core_query),
        )

    def collect(
        self,
        on_record_pair: Callable[
            [SolutionNMRMonomerXrayHomologRecord, SolutionNMRMonomerXrayHomologRecord],
            None,
        ]
        | None = None,
    ) -> tuple[
        list[SolutionNMRMonomerXrayHomologRecord],
        list[SolutionNMRMonomerXrayHomologRecord],
    ]:
        """Collect X-ray homolog records for SOLUTION NMR monomer seeds."""
        entry_ids = fetch_solution_nmr_entry_ids(
            client=self.client,
            log_label="SOLUTION NMR monomer X-ray homologs",
        )
        batches = list(chunked(entry_ids, self.config.graphql_batch_size))
        seeds: list[SolutionNMRMonomerXrayHomologSeedRecord] = []

        for batch_seeds in collect_batch_results(
            batches=batches,
            max_workers=self.config.max_workers,
            fetch_fn=(
                self.client.fetch_solution_nmr_monomer_xray_homolog_seed_records_for_ids
            ),
            progress_label="SOLUTION NMR monomer X-ray homolog seeds",
        ):
            seeds.extend(batch_seeds)
        LOGGER.info("SOLUTION NMR monomer X-ray homolog seeds: %d", len(seeds))

        records_95: list[SolutionNMRMonomerXrayHomologRecord] = []
        records_100: list[SolutionNMRMonomerXrayHomologRecord] = []
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_map = {
                executor.submit(self._build_record_pair, seed): seed for seed in seeds
            }
            pending = set(future_map)
            total = len(pending)
            completed_count = 0
            error_count = 0
            last_progress_log = time.monotonic()

            while pending:
                done, pending = wait(
                    pending,
                    timeout=30.0,
                    return_when=FIRST_COMPLETED,
                )
                if not done:
                    LOGGER.info(
                        "SOLUTION NMR monomer X-ray homolog sequence searches: "
                        "processed %d/%d entries (95%% hits=%d, 100%% hits=%d, errors=%d)",
                        completed_count,
                        total,
                        sum(1 for record in records_95 if record.has_xray_homolog),
                        sum(1 for record in records_100 if record.has_xray_homolog),
                        error_count,
                    )
                    continue

                for future in done:
                    completed_count += 1
                    seed = future_map[future]
                    try:
                        record_95, record_100 = future.result()
                    except Exception as exc:
                        error_count += 1
                        LOGGER.warning(
                            "SOLUTION NMR monomer X-ray homolog sequence search failed for %s: %s",
                            seed.entry_id,
                            exc,
                        )
                        continue

                    records_95.append(record_95)
                    records_100.append(record_100)
                    if on_record_pair is not None:
                        on_record_pair(record_95, record_100)

                    now = time.monotonic()
                    if (
                        completed_count % 25 == 0
                        or completed_count == total
                        or now - last_progress_log >= 30.0
                    ):
                        LOGGER.info(
                            "SOLUTION NMR monomer X-ray homolog sequence searches: "
                            "processed %d/%d entries (95%% hits=%d, 100%% hits=%d, errors=%d)",
                            completed_count,
                            total,
                            sum(1 for record in records_95 if record.has_xray_homolog),
                            sum(1 for record in records_100 if record.has_xray_homolog),
                            error_count,
                        )
                        last_progress_log = now

        LOGGER.info(
            "SOLUTION NMR monomer X-ray homologs: found X-ray hits (%d%%=%d/%d, %d%%=%d/%d)",
            95,
            sum(1 for record in records_95 if record.has_xray_homolog),
            len(records_95),
            100,
            sum(1 for record in records_100 if record.has_xray_homolog),
            len(records_100),
        )

        def key_fn(record):
            return record.year, record.entry_id

        return sorted(records_95, key=key_fn), sorted(records_100, key=key_fn)


class SolutionNMRMonomerXrayRmsdCollector:
    def __init__(
        self,
        client: RCSBClient,
        config: CollectorConfig,
        cache_dir: Path,
        rmsd_workers: int,
        homolog_records: list[SolutionNMRMonomerXrayHomologRecord],
        sequence_identity_percent: int = 100,
    ) -> None:
        """Initialize NMR-to-X-ray RMSD collection."""
        if sequence_identity_percent not in {95, 100}:
            raise ValueError("sequence_identity_percent must be 95 or 100")
        self.client = client
        self.config = config
        self.cache_dir = cache_dir
        self.rmsd_workers = max(1, rmsd_workers)
        self.homolog_records = homolog_records
        self.sequence_identity_percent = sequence_identity_percent
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _download_pdb_if_needed(self, entry_id: str) -> Path:
        """Download or reuse a PDB coordinate file for RMSD work."""
        return download_pdb_if_needed(
            session=self.client.session,
            config=self.config,
            cache_dir=self.cache_dir,
            entry_id=entry_id,
        )

    def _compute_candidate_record(
        self,
        homolog: SolutionNMRMonomerXrayHomologRecord,
        nmr_chain_id: str,
        nmr_pdb_path: Path,
        parsed_nmr_chain_id: str,
        candidate: XrayPolymerEntityCandidateRecord,
    ) -> SolutionNMRMonomerXrayRmsdRecord | None:
        """Compute RMSD metrics for one NMR and X-ray candidate pair."""
        try:
            (
                xray_pdb_path,
                xray_chain_map,
            ) = download_pdb_chain_subset_if_needed(
                session=self.client.session,
                config=self.config,
                cache_dir=self.cache_dir,
                entry_id=candidate.entry_id,
                chain_ids=candidate.chain_ids,
            )
        except Exception as exc:
            LOGGER.debug(
                "Skipping X-ray RMSD candidate %s for %s: %s",
                candidate.polymer_entity_id,
                homolog.entry_id,
                exc,
            )
            return None

        best_chain_result: tuple[str, int, float, int, int, int, int] | None = None
        for xray_chain_id in candidate.chain_ids:
            parsed_xray_chain_id = xray_chain_map.get(
                xray_chain_id,
                xray_chain_id,
            )
            try:
                rmsd_result = self._compute_ca_rmsd_to_xray(
                    nmr_pdb_path=nmr_pdb_path,
                    nmr_chain_id=parsed_nmr_chain_id,
                    nmr_core_start_seq_id=homolog.nmr_core_start_seq_id,
                    nmr_core_end_seq_id=homolog.nmr_core_end_seq_id,
                    xray_pdb_path=xray_pdb_path,
                    xray_chain_id=parsed_xray_chain_id,
                    sequence_identity_percent=self.sequence_identity_percent,
                )
            except Exception as exc:
                LOGGER.debug(
                    "Skipping X-ray RMSD chain %s/%s for %s: %s",
                    candidate.entry_id,
                    xray_chain_id,
                    homolog.entry_id,
                    exc,
                )
                continue
            if rmsd_result is None:
                continue
            (
                n_common_ca,
                rmsd_ca,
                _nmr_core_start,
                _nmr_core_end,
                xray_core_start,
                xray_core_end,
            ) = rmsd_result
            chain_candidate = (
                xray_chain_id,
                n_common_ca,
                rmsd_ca,
                homolog.nmr_core_start_seq_id,
                homolog.nmr_core_end_seq_id,
                xray_core_start,
                xray_core_end,
            )
            if (
                best_chain_result is None
                or chain_candidate[1] > best_chain_result[1]
                or (
                    chain_candidate[1] == best_chain_result[1]
                    and chain_candidate[2] < best_chain_result[2]
                )
            ):
                best_chain_result = chain_candidate

        if best_chain_result is None:
            return None
        (
            xray_chain_id,
            n_common_ca,
            rmsd_ca,
            _nmr_core_start,
            _nmr_core_end,
            xray_core_start,
            xray_core_end,
        ) = best_chain_result
        return SolutionNMRMonomerXrayRmsdRecord(
            entry_id=homolog.entry_id,
            year=homolog.year,
            sequence_identity_percent=self.sequence_identity_percent,
            nmr_chain_id=nmr_chain_id,
            nmr_core_start_seq_id=homolog.nmr_core_start_seq_id,
            nmr_core_end_seq_id=homolog.nmr_core_end_seq_id,
            nmr_query_sequence_length=homolog.nmr_query_sequence_length,
            xray_homolog_entity_id=candidate.polymer_entity_id,
            xray_homolog_count=len(homolog.xray_homolog_entity_ids),
            xray_entry_id=candidate.entry_id,
            xray_chain_id=xray_chain_id,
            xray_core_start_seq_id=xray_core_start,
            xray_core_end_seq_id=xray_core_end,
            xray_resolution_angstrom=candidate.resolution_angstrom,
            n_common_ca=n_common_ca,
            rmsd_ca_angstrom=rmsd_ca,
        )

    @staticmethod
    def _compute_ca_rmsd_to_xray(
        nmr_pdb_path: Path,
        nmr_chain_id: str,
        nmr_core_start_seq_id: int,
        nmr_core_end_seq_id: int,
        xray_pdb_path: Path,
        xray_chain_id: str,
        sequence_identity_percent: int,
    ) -> tuple[int, float, int, int, int, int] | None:
        """Compute CA RMSD after aligning an NMR model core to an X-ray chain."""
        _ = sequence_identity_percent
        nmr_residues = parse_first_model_ca_residues(
            nmr_pdb_path,
            nmr_chain_id,
            start_seq_id=nmr_core_start_seq_id,
            end_seq_id=nmr_core_end_seq_id,
            include_hetatm=False,
        )
        if len(nmr_residues) <= 10:
            return None

        xray_residues = parse_first_model_ca_residues(
            xray_pdb_path,
            xray_chain_id,
            include_hetatm=True,
        )
        matched_pair_sets = find_modeled_ca_core_identity_matches(
            nmr_residues=nmr_residues,
            xray_residues=xray_residues,
            sequence_identity_percent=100,
        )
        if not matched_pair_sets:
            return None

        nmr_model_maps = parse_models_ca_coords(
            nmr_pdb_path,
            nmr_chain_id,
            start_seq_id=nmr_core_start_seq_id,
            end_seq_id=nmr_core_end_seq_id,
        )
        if not nmr_model_maps:
            return None
        nmr_first_model_map = nmr_model_maps[0]
        xray_models = parse_models_ca_coords(xray_pdb_path, xray_chain_id)
        if not xray_models:
            return None
        xray_first_model_map = xray_models[0]

        best_result: tuple[int, float, int, int, int, int] | None = None
        for matched_pairs in matched_pair_sets:
            rmsd_pairs = [
                (nmr_record.resid, xray_record.resid)
                for nmr_record, xray_record in matched_pairs
                if nmr_record.is_standard_atom and xray_record.is_standard_atom
            ]
            if len(rmsd_pairs) < 3:
                continue

            nmr_common_resids = [nmr_resid for nmr_resid, _ in rmsd_pairs]
            xray_common_resids = [xray_resid for _, xray_resid in rmsd_pairs]
            if any(resid not in nmr_first_model_map for resid in nmr_common_resids):
                continue
            if any(resid not in xray_first_model_map for resid in xray_common_resids):
                continue

            nmr_coords = np.asarray(
                [nmr_first_model_map[resid] for resid in nmr_common_resids],
                dtype=float,
            )
            xray_coords = np.asarray(
                [xray_first_model_map[resid] for resid in xray_common_resids],
                dtype=float,
            )
            rmsd_value = _superposed_rmsd(nmr_coords, xray_coords)
            xray_matched_resids = [
                xray_record.resid for _, xray_record in matched_pairs
            ]
            result = (
                len(rmsd_pairs),
                rmsd_value,
                nmr_core_start_seq_id,
                nmr_core_end_seq_id,
                min(xray_matched_resids),
                max(xray_matched_resids),
            )
            if best_result is None or result[1] < best_result[1]:
                best_result = result

        return best_result

    def _compute_record(
        self,
        homolog: SolutionNMRMonomerXrayHomologRecord,
        nmr_chain_id: str,
        candidates: tuple[XrayPolymerEntityCandidateRecord, ...],
    ) -> SolutionNMRMonomerXrayRmsdRecord | None:
        """Compute the best X-ray RMSD record for one NMR seed."""
        if (
            homolog.nmr_core_start_seq_id is None
            or homolog.nmr_core_end_seq_id is None
            or not candidates
        ):
            return None

        try:
            nmr_pdb_path = self._download_pdb_if_needed(homolog.entry_id)
            nmr_chain_map = load_cached_chain_id_map(self.cache_dir, homolog.entry_id)
            parsed_nmr_chain_id = nmr_chain_map.get(nmr_chain_id, nmr_chain_id)

            for candidate in candidates:
                record = self._compute_candidate_record(
                    homolog=homolog,
                    nmr_chain_id=nmr_chain_id,
                    nmr_pdb_path=nmr_pdb_path,
                    parsed_nmr_chain_id=parsed_nmr_chain_id,
                    candidate=candidate,
                )
                if record is not None:
                    return record
            return None
        except Exception as exc:
            LOGGER.warning(
                "X-ray RMSD calculation failed for %s: %s", homolog.entry_id, exc
            )
            return None

    def _compute_extremes_record(
        self,
        homolog: SolutionNMRMonomerXrayHomologRecord,
        nmr_chain_id: str,
        candidates: tuple[XrayPolymerEntityCandidateRecord, ...],
    ) -> SolutionNMRMonomerXrayRmsdExtremesRecord | None:
        """Compute minimum and maximum X-ray RMSD records for one NMR seed."""
        if (
            homolog.nmr_core_start_seq_id is None
            or homolog.nmr_core_end_seq_id is None
            or not candidates
        ):
            return None

        try:
            nmr_pdb_path = self._download_pdb_if_needed(homolog.entry_id)
            nmr_chain_map = load_cached_chain_id_map(self.cache_dir, homolog.entry_id)
            parsed_nmr_chain_id = nmr_chain_map.get(nmr_chain_id, nmr_chain_id)

            candidate_records: list[SolutionNMRMonomerXrayRmsdRecord] = []
            for candidate in candidates:
                record = self._compute_candidate_record(
                    homolog=homolog,
                    nmr_chain_id=nmr_chain_id,
                    nmr_pdb_path=nmr_pdb_path,
                    parsed_nmr_chain_id=parsed_nmr_chain_id,
                    candidate=candidate,
                )
                if record is not None:
                    candidate_records.append(record)
            if not candidate_records:
                return None

            best = min(
                candidate_records,
                key=lambda record: (
                    record.rmsd_ca_angstrom,
                    -record.n_common_ca,
                    record.xray_resolution_angstrom,
                    record.xray_entry_id,
                    record.xray_homolog_entity_id,
                ),
            )
            worst = max(
                candidate_records,
                key=lambda record: (
                    record.rmsd_ca_angstrom,
                    record.n_common_ca,
                    -record.xray_resolution_angstrom,
                    record.xray_entry_id,
                    record.xray_homolog_entity_id,
                ),
            )
            return SolutionNMRMonomerXrayRmsdExtremesRecord(
                entry_id=homolog.entry_id,
                year=homolog.year,
                sequence_identity_percent=self.sequence_identity_percent,
                nmr_chain_id=nmr_chain_id,
                nmr_core_start_seq_id=homolog.nmr_core_start_seq_id,
                nmr_core_end_seq_id=homolog.nmr_core_end_seq_id,
                nmr_query_sequence_length=homolog.nmr_query_sequence_length,
                xray_homolog_count=len(homolog.xray_homolog_entity_ids),
                successful_xray_homolog_count=len(candidate_records),
                best_xray_homolog_entity_id=best.xray_homolog_entity_id,
                best_xray_entry_id=best.xray_entry_id,
                best_xray_chain_id=best.xray_chain_id,
                best_xray_resolution_angstrom=best.xray_resolution_angstrom,
                best_xray_core_start_seq_id=best.xray_core_start_seq_id,
                best_xray_core_end_seq_id=best.xray_core_end_seq_id,
                best_n_common_ca=best.n_common_ca,
                best_rmsd_ca_angstrom=best.rmsd_ca_angstrom,
                worst_xray_homolog_entity_id=worst.xray_homolog_entity_id,
                worst_xray_entry_id=worst.xray_entry_id,
                worst_xray_chain_id=worst.xray_chain_id,
                worst_xray_resolution_angstrom=worst.xray_resolution_angstrom,
                worst_xray_core_start_seq_id=worst.xray_core_start_seq_id,
                worst_xray_core_end_seq_id=worst.xray_core_end_seq_id,
                worst_n_common_ca=worst.n_common_ca,
                worst_rmsd_ca_angstrom=worst.rmsd_ca_angstrom,
                rmsd_delta_angstrom=(
                    worst.rmsd_ca_angstrom - best.rmsd_ca_angstrom
                ),
            )
        except Exception as exc:
            LOGGER.warning(
                "X-ray RMSD extremes calculation failed for %s: %s",
                homolog.entry_id,
                exc,
            )
            return None

    def _prepare_work_items(
        self,
        max_entries: int | None,
        skip_entry_ids: set[str],
        progress_prefix: str,
    ) -> list[
        tuple[
            SolutionNMRMonomerXrayHomologRecord,
            str,
            tuple[XrayPolymerEntityCandidateRecord, ...],
        ]
    ]:
        """Prepare NMR seeds and candidate lists for RMSD collection."""
        filtered_homologs = [
            record
            for record in self.homolog_records
            if record.sequence_identity_percent == self.sequence_identity_percent
            and record.entry_id not in skip_entry_ids
            and record.nmr_core_start_seq_id is not None
            and record.nmr_core_end_seq_id is not None
            and record.xray_homolog_entity_ids
        ]
        filtered_homologs = sorted(
            filtered_homologs, key=lambda r: (r.year, r.entry_id)
        )
        if max_entries is not None:
            filtered_homologs = filtered_homologs[: max(0, max_entries)]

        entry_ids = sorted({record.entry_id for record in filtered_homologs})
        chain_by_entry_id: dict[str, str] = {}
        for batch_seeds in collect_batch_results(
            batches=list(chunked(entry_ids, self.config.graphql_batch_size)),
            max_workers=self.config.max_workers,
            fetch_fn=self.client.fetch_solution_nmr_monomer_xray_homolog_seed_records_for_ids,
            progress_label=f"{progress_prefix} NMR chain lookup",
        ):
            for seed in batch_seeds:
                chain_by_entry_id[seed.entry_id] = seed.chain_id

        xray_entity_ids = sorted(
            {
                entity_id
                for record in filtered_homologs
                for entity_id in record.xray_homolog_entity_ids
            }
        )
        candidate_by_entity_id: dict[str, XrayPolymerEntityCandidateRecord] = {}
        for batch_candidates in collect_batch_results(
            batches=list(chunked(xray_entity_ids, self.config.graphql_batch_size)),
            max_workers=self.config.max_workers,
            fetch_fn=self.client.fetch_xray_polymer_entity_candidates_for_ids,
            progress_label=f"{progress_prefix} X-ray candidate metadata",
        ):
            for candidate in batch_candidates:
                candidate_by_entity_id[candidate.polymer_entity_id] = candidate

        work_items: list[
            tuple[
                SolutionNMRMonomerXrayHomologRecord,
                str,
                tuple[XrayPolymerEntityCandidateRecord, ...],
            ]
        ] = []
        for homolog in filtered_homologs:
            nmr_chain_id = chain_by_entry_id.get(homolog.entry_id)
            if not nmr_chain_id:
                continue
            candidates = tuple(
                sorted(
                    (
                        candidate_by_entity_id[entity_id]
                        for entity_id in homolog.xray_homolog_entity_ids
                        if entity_id in candidate_by_entity_id
                    ),
                    key=lambda c: (
                        c.resolution_angstrom,
                        c.entry_id,
                        c.polymer_entity_id,
                    ),
                )
            )
            if not candidates:
                continue
            work_items.append((homolog, nmr_chain_id, candidates))

        LOGGER.info(
            "%s %d%%: entries to process=%d, unique X-ray entities=%d",
            progress_prefix,
            self.sequence_identity_percent,
            len(work_items),
            len(candidate_by_entity_id),
        )
        return work_items

    def collect(
        self,
        max_entries: int | None = None,
        skip_entry_ids: set[str] | None = None,
        on_record: Callable[[SolutionNMRMonomerXrayRmsdRecord], None] | None = None,
    ) -> list[SolutionNMRMonomerXrayRmsdRecord]:
        """Collect best-match NMR-to-X-ray RMSD records."""
        skip_entry_ids = skip_entry_ids or set()
        work_items = self._prepare_work_items(
            max_entries=max_entries,
            skip_entry_ids=skip_entry_ids,
            progress_prefix="SOLUTION NMR X-ray RMSD",
        )

        records: list[SolutionNMRMonomerXrayRmsdRecord] = []
        with ThreadPoolExecutor(max_workers=self.rmsd_workers) as executor:
            future_map = {
                executor.submit(
                    self._compute_record,
                    homolog=homolog,
                    nmr_chain_id=nmr_chain_id,
                    candidates=candidates,
                ): idx
                for idx, (homolog, nmr_chain_id, candidates) in enumerate(
                    work_items, start=1
                )
            }
            total = len(future_map)
            completed = 0
            for future in as_completed(future_map):
                record = future.result()
                if record is not None:
                    records.append(record)
                    if on_record is not None:
                        on_record(record)
                completed += 1
                if total > 0 and (completed % 50 == 0 or completed == total):
                    LOGGER.info(
                        "SOLUTION NMR X-ray RMSD %d%%: processed %d/%d entries",
                        self.sequence_identity_percent,
                        completed,
                        total,
                    )

        return sorted(records, key=lambda r: (r.year, r.entry_id))

    def collect_extremes(
        self,
        max_entries: int | None = None,
        skip_entry_ids: set[str] | None = None,
        on_record: Callable[[SolutionNMRMonomerXrayRmsdExtremesRecord], None]
        | None = None,
    ) -> list[SolutionNMRMonomerXrayRmsdExtremesRecord]:
        """Collect minimum and maximum NMR-to-X-ray RMSD records."""
        skip_entry_ids = skip_entry_ids or set()
        work_items = self._prepare_work_items(
            max_entries=max_entries,
            skip_entry_ids=skip_entry_ids,
            progress_prefix="SOLUTION NMR X-ray RMSD extremes",
        )

        records: list[SolutionNMRMonomerXrayRmsdExtremesRecord] = []
        with ThreadPoolExecutor(max_workers=self.rmsd_workers) as executor:
            future_map = {
                executor.submit(
                    self._compute_extremes_record,
                    homolog=homolog,
                    nmr_chain_id=nmr_chain_id,
                    candidates=candidates,
                ): idx
                for idx, (homolog, nmr_chain_id, candidates) in enumerate(
                    work_items, start=1
                )
            }
            total = len(future_map)
            completed = 0
            for future in as_completed(future_map):
                record = future.result()
                if record is not None:
                    records.append(record)
                    if on_record is not None:
                        on_record(record)
                completed += 1
                if total > 0 and (completed % 50 == 0 or completed == total):
                    LOGGER.info(
                        "SOLUTION NMR X-ray RMSD extremes %d%%: processed %d/%d entries",
                        self.sequence_identity_percent,
                        completed,
                        total,
                    )

        return sorted(records, key=lambda r: (r.year, r.entry_id))


def write_method_counts_csv(
    records: list[YearlyCountRecord], output_path: Path
) -> None:
    """Write yearly experimental-method counts to CSV."""
    write_csv_rows(
        output_path=output_path,
        header=["year", "method", "count"],
        rows=((r.year, r.method, r.count) for r in records),
    )


def write_solution_nmr_program_counts_csv(
    records: list[SolutionNMRProgramYearlyCountRecord], output_path: Path
) -> None:
    """Write yearly SOLUTION NMR program counts to CSV."""
    write_csv_rows(
        output_path=output_path,
        header=["year", "program", "count"],
        rows=((r.year, r.program, r.count) for r in records),
    )


def read_solution_nmr_monomer_quality_csv(
    input_path: Path,
) -> list[SolutionNMRMonomerQualityRecord]:
    """Read SOLUTION NMR monomer quality records from CSV."""
    if not input_path.exists():
        return []
    records: list[SolutionNMRMonomerQualityRecord] = []
    with input_path.open("r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if not row:
                continue
            records.append(
                SolutionNMRMonomerQualityRecord(
                    entry_id=str(row["entry_id"]),
                    year=int(row["year"]),
                    clashscore=float(row["clashscore"]),
                    ramachandran_outliers_percent=float(
                        row["ramachandran_outliers_percent"]
                    ),
                    sidechain_outliers_percent=float(
                        row["sidechain_outliers_percent"]
                    ),
                )
            )
    return records


def write_solution_nmr_monomer_program_cluster_assignments_csv(
    records: list[SolutionNMRMonomerProgramClusterAssignmentRecord],
    output_path: Path,
) -> None:
    """Write program-cluster assignment records to CSV."""
    write_csv_rows(
        output_path=output_path,
        header=[
            "entry_id",
            "year",
            "cluster_id",
            "cluster_name",
            "has_program_text",
            "program_text",
        ],
        rows=(
            (
                r.entry_id,
                r.year,
                r.cluster_id,
                r.cluster_name,
                int(r.has_program_text),
                r.program_text,
            )
            for r in records
        ),
    )


def read_solution_nmr_monomer_program_cluster_assignments_csv(
    input_path: Path,
) -> list[SolutionNMRMonomerProgramClusterAssignmentRecord]:
    """Read program-cluster assignment records from CSV."""
    if not input_path.exists():
        return []
    records: list[SolutionNMRMonomerProgramClusterAssignmentRecord] = []
    with input_path.open("r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if not row:
                continue
            records.append(
                SolutionNMRMonomerProgramClusterAssignmentRecord(
                    entry_id=str(row["entry_id"]),
                    year=int(row["year"]),
                    cluster_id=str(row["cluster_id"]),
                    cluster_name=str(row["cluster_name"]),
                    has_program_text=bool(int(row["has_program_text"])),
                    program_text=str(row["program_text"]),
                )
            )
    return records


def write_solution_nmr_monomer_program_cluster_summary_csv(
    records: list[SolutionNMRMonomerProgramClusterSummaryRecord],
    output_path: Path,
) -> None:
    """Write yearly program-cluster quality summary rows to CSV."""
    write_csv_rows(
        output_path=output_path,
        header=[
            "year",
            "cluster_id",
            "cluster_name",
            "structure_count",
            "avg_ramachandran_outliers_percent",
            "avg_sidechain_outliers_percent",
            "avg_clashscore",
        ],
        rows=(
            (
                r.year,
                r.cluster_id,
                r.cluster_name,
                r.structure_count,
                (
                    f"{r.avg_ramachandran_outliers_percent:.4f}"
                    if r.avg_ramachandran_outliers_percent is not None
                    else ""
                ),
                (
                    f"{r.avg_sidechain_outliers_percent:.4f}"
                    if r.avg_sidechain_outliers_percent is not None
                    else ""
                ),
                f"{r.avg_clashscore:.4f}" if r.avg_clashscore is not None else "",
            )
            for r in records
        ),
    )


def summarize_solution_nmr_monomer_program_cluster_quality_by_year(
    assignment_records: list[SolutionNMRMonomerProgramClusterAssignmentRecord],
    quality_records: list[SolutionNMRMonomerQualityRecord],
) -> list[SolutionNMRMonomerProgramClusterYearlySummaryRecord]:
    """Summarize monomer quality metrics by year and program cluster."""
    if not assignment_records or not quality_records:
        return []

    quality_by_key = {
        (record.entry_id.upper(), record.year): record for record in quality_records
    }
    yearly_totals: dict[int, dict[str, float | int]] = {}
    assignment_keys = {
        (record.entry_id.upper(), record.year) for record in assignment_records
    }
    matched_quality_keys: set[tuple[str, int]] = set()
    missing_quality_count = 0

    for key in sorted(assignment_keys):
        quality_record = quality_by_key.get(key)
        if quality_record is None:
            missing_quality_count += 1
            continue
        matched_quality_keys.add(key)
        total_row = yearly_totals.setdefault(
            quality_record.year,
            {
                "count": 0,
                "rama_sum": 0.0,
                "side_sum": 0.0,
                "clash_sum": 0.0,
            },
        )
        total_row["count"] += 1
        total_row["rama_sum"] += quality_record.ramachandran_outliers_percent
        total_row["side_sum"] += quality_record.sidechain_outliers_percent
        total_row["clash_sum"] += quality_record.clashscore

    unmatched_quality_count = len(quality_by_key) - len(matched_quality_keys)
    LOGGER.info(
        (
            "SOLUTION NMR monomer program-cluster yearly totals: years=%d, "
            "matched_entries=%d, missing_quality=%d, unmatched_quality=%d"
        ),
        len(yearly_totals),
        len(matched_quality_keys),
        missing_quality_count,
        unmatched_quality_count,
    )

    return [
        SolutionNMRMonomerProgramClusterYearlySummaryRecord(
            year=year,
            structure_count=int(total_row["count"]),
            avg_ramachandran_outliers_percent=(
                float(total_row["rama_sum"]) / int(total_row["count"])
                if int(total_row["count"]) > 0
                else None
            ),
            avg_sidechain_outliers_percent=(
                float(total_row["side_sum"]) / int(total_row["count"])
                if int(total_row["count"]) > 0
                else None
            ),
            avg_clashscore=(
                float(total_row["clash_sum"]) / int(total_row["count"])
                if int(total_row["count"]) > 0
                else None
            ),
        )
        for year, total_row in sorted(yearly_totals.items())
    ]


def write_solution_nmr_monomer_program_cluster_yearly_summary_csv(
    records: list[SolutionNMRMonomerProgramClusterYearlySummaryRecord],
    output_path: Path,
) -> None:
    """Write overall yearly program-cluster quality summaries to CSV."""
    write_csv_rows(
        output_path=output_path,
        header=[
            "year",
            "structure_count",
            "avg_ramachandran_outliers_percent",
            "avg_sidechain_outliers_percent",
            "avg_clashscore",
        ],
        rows=(
            (
                r.year,
                r.structure_count,
                (
                    f"{r.avg_ramachandran_outliers_percent:.4f}"
                    if r.avg_ramachandran_outliers_percent is not None
                    else ""
                ),
                (
                    f"{r.avg_sidechain_outliers_percent:.4f}"
                    if r.avg_sidechain_outliers_percent is not None
                    else ""
                ),
                f"{r.avg_clashscore:.4f}" if r.avg_clashscore is not None else "",
            )
            for r in records
        ),
    )


def summarize_solution_nmr_monomer_program_cluster_quality_total(
    assignment_records: list[SolutionNMRMonomerProgramClusterAssignmentRecord],
    quality_records: list[SolutionNMRMonomerQualityRecord],
) -> list[SolutionNMRMonomerProgramClusterTotalRecord]:
    """Summarize program-cluster quality metrics across all years."""
    if not assignment_records or not quality_records:
        return []

    quality_by_key = {
        (record.entry_id.upper(), record.year): record for record in quality_records
    }
    totals_by_cluster_id: dict[str, dict[str, float | int | str]] = {
        cluster_id: {
            "cluster_name": cluster_name,
            "count": 0,
            "rama_sum": 0.0,
            "side_sum": 0.0,
            "clash_sum": 0.0,
        }
        for cluster_id, cluster_name in PROGRAM_CLUSTER_DEFINITIONS
    }
    matched_quality_keys: set[tuple[str, int]] = set()
    missing_quality_count = 0

    for assignment_record in assignment_records:
        key = (assignment_record.entry_id.upper(), assignment_record.year)
        quality_record = quality_by_key.get(key)
        if quality_record is None:
            missing_quality_count += 1
            continue
        matched_quality_keys.add(key)
        total_row = totals_by_cluster_id.setdefault(
            assignment_record.cluster_id,
            {
                "cluster_name": assignment_record.cluster_name,
                "count": 0,
                "rama_sum": 0.0,
                "side_sum": 0.0,
                "clash_sum": 0.0,
            },
        )
        total_row["count"] += 1
        total_row["rama_sum"] += quality_record.ramachandran_outliers_percent
        total_row["side_sum"] += quality_record.sidechain_outliers_percent
        total_row["clash_sum"] += quality_record.clashscore

    unmatched_quality_count = len(quality_by_key) - len(matched_quality_keys)
    LOGGER.info(
        (
            "SOLUTION NMR monomer program-cluster totals: clusters=%d, "
            "matched_entries=%d, missing_quality=%d, unmatched_quality=%d"
        ),
        len(totals_by_cluster_id),
        len(matched_quality_keys),
        missing_quality_count,
        unmatched_quality_count,
    )

    ordered_records: list[SolutionNMRMonomerProgramClusterTotalRecord] = []
    for cluster_id, cluster_name in PROGRAM_CLUSTER_DEFINITIONS:
        total_row = totals_by_cluster_id.get(
            cluster_id,
            {
                "cluster_name": cluster_name,
                "count": 0,
                "rama_sum": 0.0,
                "side_sum": 0.0,
                "clash_sum": 0.0,
            },
        )
        count = int(total_row["count"])
        ordered_records.append(
            SolutionNMRMonomerProgramClusterTotalRecord(
                cluster_name=str(total_row["cluster_name"]),
                structure_count=count,
                avg_ramachandran_outliers_percent=(
                    float(total_row["rama_sum"]) / count if count > 0 else None
                ),
                avg_sidechain_outliers_percent=(
                    float(total_row["side_sum"]) / count if count > 0 else None
                ),
                avg_clashscore=(
                    float(total_row["clash_sum"]) / count if count > 0 else None
                ),
            )
        )
    return ordered_records


def write_solution_nmr_monomer_program_cluster_total_csv(
    records: list[SolutionNMRMonomerProgramClusterTotalRecord],
    output_path: Path,
) -> None:
    """Write total program-cluster quality summaries to CSV."""
    write_csv_rows(
        output_path=output_path,
        header=[
            "cluster_name",
            "structure_count",
            "avg_ramachandran_outliers_percent",
            "avg_sidechain_outliers_percent",
            "avg_clashscore",
        ],
        rows=(
            (
                r.cluster_name,
                r.structure_count,
                (
                    f"{r.avg_ramachandran_outliers_percent:.4f}"
                    if r.avg_ramachandran_outliers_percent is not None
                    else ""
                ),
                (
                    f"{r.avg_sidechain_outliers_percent:.4f}"
                    if r.avg_sidechain_outliers_percent is not None
                    else ""
                ),
                f"{r.avg_clashscore:.4f}" if r.avg_clashscore is not None else "",
            )
            for r in records
        ),
    )


def write_membrane_counts_csv(
    records: list[MembraneYearlyCountRecord], output_path: Path
) -> None:
    """Write membrane-protein yearly count records to CSV."""
    write_csv_rows(
        output_path=output_path,
        header=["year", "count"],
        rows=((r.year, r.count) for r in records),
    )


def write_solution_nmr_weights_csv(
    records: list[SolutionNMRWeightRecord], output_path: Path
) -> None:
    """Write SOLUTION NMR molecular-weight records to CSV."""
    write_csv_rows(
        output_path=output_path,
        header=["entry_id", "year", "molecular_weight_kda"],
        rows=(
            (
                r.entry_id,
                r.year,
                f"{r.molecular_weight_kda:.3f}",
            )
            for r in records
        ),
    )


def stream_solution_nmr_monomer_stride_modeled_first_model_csv(
    records: Iterator[SolutionNMRMonomerStrideModeledFirstModelRecord],
    output_path: Path,
) -> int:
    """Stream STRIDE modeled-first-model records directly to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "entry_id",
                "year",
                "chain_id",
                "modeled_start_seq_id",
                "modeled_end_seq_id",
                "modeled_sequence_length",
                "secondary_structure_percent",
                "helix_fraction",
                "sheet_fraction",
                "stride_alpha_helix_fraction",
                "stride_3_10_helix_fraction",
                "stride_pi_helix_fraction",
                "stride_beta_strand_fraction",
                "stride_isolated_beta_bridge_fraction",
                "stride_turn_fraction",
                "stride_coil_fraction",
                "stride_secondary_structure_percent",
            ]
        )
        csvfile.flush()
        for record in records:
            writer.writerow(
                (
                    record.entry_id,
                    record.year,
                    record.chain_id,
                    record.modeled_start_seq_id,
                    record.modeled_end_seq_id,
                    record.modeled_sequence_length,
                    f"{record.secondary_structure_percent:.3f}",
                    f"{record.helix_fraction:.6f}",
                    f"{record.sheet_fraction:.6f}",
                    f"{record.stride_alpha_helix_fraction:.6f}",
                    f"{record.stride_3_10_helix_fraction:.6f}",
                    f"{record.stride_pi_helix_fraction:.6f}",
                    f"{record.stride_beta_strand_fraction:.6f}",
                    f"{record.stride_isolated_beta_bridge_fraction:.6f}",
                    f"{record.stride_turn_fraction:.6f}",
                    f"{record.stride_coil_fraction:.6f}",
                    f"{record.stride_secondary_structure_percent:.3f}",
                )
            )
            csvfile.flush()
            count += 1
    return count


def read_solution_nmr_monomer_precision_csv(
    input_path: Path,
) -> list[SolutionNMRMonomerPrecisionRecord]:
    """Read SOLUTION NMR monomer precision records from CSV."""
    if not input_path.exists():
        return []
    records: list[SolutionNMRMonomerPrecisionRecord] = []
    with input_path.open("r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if not row:
                continue
            n_ca_core_used_raw = row.get("n_ca_core_used")
            if n_ca_core_used_raw in {None, ""}:
                n_ca_core_used_raw = row.get("n_ca_core")
            if n_ca_core_used_raw in {None, ""}:
                continue
            n_ca_core_raw_raw = row.get("n_ca_core_raw")
            if n_ca_core_raw_raw in {None, ""}:
                n_ca_core_raw_raw = n_ca_core_used_raw
            records.append(
                SolutionNMRMonomerPrecisionRecord(
                    entry_id=str(row["entry_id"]),
                    year=int(row["year"]),
                    chain_id=str(row["chain_id"]),
                    core_start_seq_id=int(row["core_start_seq_id"]),
                    core_end_seq_id=int(row["core_end_seq_id"]),
                    n_models=int(row["n_models"]),
                    n_ca_core_used=int(str(n_ca_core_used_raw)),
                    n_ca_core_raw=int(str(n_ca_core_raw_raw)),
                    mean_rmsd_angstrom=float(row["mean_rmsd_angstrom"]),
                )
            )
    return records


SOLUTION_NMR_MONOMER_PRECISION_HEADER: tuple[str, ...] = (
    "entry_id",
    "year",
    "chain_id",
    "core_start_seq_id",
    "core_end_seq_id",
    "n_models",
    "n_ca_core_used",
    "n_ca_core_raw",
    "mean_rmsd_angstrom",
)


def _solution_nmr_monomer_precision_csv_row(
    record: SolutionNMRMonomerPrecisionRecord,
) -> tuple[Any, ...]:
    """Convert a precision record into a CSV row dictionary."""
    return (
        record.entry_id,
        record.year,
        record.chain_id,
        record.core_start_seq_id,
        record.core_end_seq_id,
        record.n_models,
        record.n_ca_core_used,
        record.n_ca_core_raw,
        f"{record.mean_rmsd_angstrom:.6f}",
    )


def write_solution_nmr_monomer_quality_csv(
    records: list[SolutionNMRMonomerQualityRecord], output_path: Path
) -> None:
    """Write SOLUTION NMR monomer quality records to CSV."""
    write_csv_rows(
        output_path=output_path,
        header=[
            "entry_id",
            "year",
            "clashscore",
            "ramachandran_outliers_percent",
            "sidechain_outliers_percent",
        ],
        rows=(
            (
                r.entry_id,
                r.year,
                f"{r.clashscore:.4f}",
                f"{r.ramachandran_outliers_percent:.4f}",
                f"{r.sidechain_outliers_percent:.4f}",
            )
            for r in records
        ),
    )


SOLUTION_NMR_MONOMER_XRAY_HOMOLOG_HEADER = [
    "entry_id",
    "year",
    "sequence_identity_percent",
    "nmr_core_start_seq_id",
    "nmr_core_end_seq_id",
    "nmr_query_sequence_length",
    "has_xray_homolog",
    "xray_homolog_count",
    "xray_homolog_entry_ids",
    "xray_homolog_entity_ids",
]


def _solution_nmr_monomer_xray_homolog_csv_row(
    record: SolutionNMRMonomerXrayHomologRecord,
) -> tuple[Any, ...]:
    """Convert an X-ray homolog record into a CSV row dictionary."""
    return (
        record.entry_id,
        record.year,
        record.sequence_identity_percent,
        (
            record.nmr_core_start_seq_id
            if record.nmr_core_start_seq_id is not None
            else ""
        ),
        record.nmr_core_end_seq_id
        if record.nmr_core_end_seq_id is not None
        else "",
        record.nmr_query_sequence_length,
        int(record.has_xray_homolog),
        len(record.xray_homolog_entity_ids),
        ";".join(record.xray_homolog_entry_ids),
        ";".join(record.xray_homolog_entity_ids),
    )


def write_solution_nmr_monomer_xray_homolog_csv(
    records: list[SolutionNMRMonomerXrayHomologRecord], output_path: Path
) -> None:
    """Write X-ray homolog records to CSV."""
    write_csv_rows(
        output_path=output_path,
        header=SOLUTION_NMR_MONOMER_XRAY_HOMOLOG_HEADER,
        rows=(_solution_nmr_monomer_xray_homolog_csv_row(r) for r in records),
    )


def read_solution_nmr_monomer_xray_homolog_csv(
    input_path: Path,
) -> list[SolutionNMRMonomerXrayHomologRecord]:
    """Read X-ray homolog records from CSV."""
    if not input_path.exists():
        return []
    records: list[SolutionNMRMonomerXrayHomologRecord] = []
    with input_path.open("r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if not row:
                continue
            nmr_core_start_raw = row.get("nmr_core_start_seq_id")
            nmr_core_end_raw = row.get("nmr_core_end_seq_id")
            xray_entry_ids = tuple(
                item.strip()
                for item in str(row.get("xray_homolog_entry_ids") or "").split(";")
                if item.strip()
            )
            xray_entity_ids = tuple(
                item.strip()
                for item in str(row.get("xray_homolog_entity_ids") or "").split(";")
                if item.strip()
            )
            records.append(
                SolutionNMRMonomerXrayHomologRecord(
                    entry_id=str(row["entry_id"]),
                    year=int(row["year"]),
                    sequence_identity_percent=int(row["sequence_identity_percent"]),
                    nmr_core_start_seq_id=(
                        int(nmr_core_start_raw)
                        if nmr_core_start_raw not in {None, ""}
                        else None
                    ),
                    nmr_core_end_seq_id=(
                        int(nmr_core_end_raw)
                        if nmr_core_end_raw not in {None, ""}
                        else None
                    ),
                    nmr_query_sequence_length=int(
                        row.get("nmr_query_sequence_length") or 0
                    ),
                    xray_homolog_entry_ids=xray_entry_ids,
                    xray_homolog_entity_ids=xray_entity_ids,
                    has_xray_homolog=bool(int(row.get("has_xray_homolog") or 0)),
                )
            )
    return records


def filter_xray_homolog_records_by_deposit_date(
    records: list[SolutionNMRMonomerXrayHomologRecord],
    client: RCSBClient,
    config: CollectorConfig,
) -> list[SolutionNMRMonomerXrayHomologRecord]:
    """Keep homolog records whose X-ray release timing matches the mode."""
    xray_entity_ids = sorted(
        {
            entity_id
            for record in records
            for entity_id in record.xray_homolog_entity_ids
        }
    )
    if not xray_entity_ids:
        return [
            SolutionNMRMonomerXrayHomologRecord(
                entry_id=record.entry_id,
                year=record.year,
                sequence_identity_percent=record.sequence_identity_percent,
                nmr_core_start_seq_id=record.nmr_core_start_seq_id,
                nmr_core_end_seq_id=record.nmr_core_end_seq_id,
                nmr_query_sequence_length=record.nmr_query_sequence_length,
                xray_homolog_entry_ids=tuple(),
                xray_homolog_entity_ids=tuple(),
                has_xray_homolog=False,
            )
            for record in records
        ]

    xray_entry_id_by_entity_id = {
        entity_id: str(entity_id).split("_", 1)[0].strip()
        for entity_id in xray_entity_ids
    }
    entry_ids = sorted(
        {
            record.entry_id
            for record in records
        }
        | {
            entry_id
            for entry_id in xray_entry_id_by_entity_id.values()
            if entry_id
        }
    )
    accession_dates_by_entry_id: dict[str, tuple[str | None, str | None]] = {}
    for batch_dates in collect_batch_results(
        batches=list(chunked(entry_ids, config.graphql_batch_size)),
        max_workers=config.max_workers,
        fetch_fn=client.fetch_accession_dates_by_entry_id_for_ids,
        progress_label="Historical homolog accession dates",
    ):
        accession_dates_by_entry_id.update(batch_dates)

    historical_records: list[SolutionNMRMonomerXrayHomologRecord] = []
    for record in records:
        nmr_deposit_date = parse_rcsb_datetime(
            (accession_dates_by_entry_id.get(record.entry_id) or (None, None))[0]
        )
        kept_entity_ids_list: list[str] = []
        if nmr_deposit_date is not None:
            for entity_id in record.xray_homolog_entity_ids:
                xray_entry_id = xray_entry_id_by_entity_id.get(entity_id)
                if not xray_entry_id:
                    continue
                xray_release_date = (
                    accession_dates_by_entry_id.get(xray_entry_id) or (None, None)
                )[1]
                parsed_xray_release_date = parse_rcsb_datetime(xray_release_date)
                if parsed_xray_release_date is None:
                    continue
                if parsed_xray_release_date <= nmr_deposit_date:
                    kept_entity_ids_list.append(entity_id)
        kept_entity_ids = tuple(kept_entity_ids_list)
        kept_entry_ids = SolutionNMRMonomerXrayHomologCollector._entry_ids_from_polymer_entity_ids(
            kept_entity_ids
        )
        historical_records.append(
            SolutionNMRMonomerXrayHomologRecord(
                entry_id=record.entry_id,
                year=record.year,
                sequence_identity_percent=record.sequence_identity_percent,
                nmr_core_start_seq_id=record.nmr_core_start_seq_id,
                nmr_core_end_seq_id=record.nmr_core_end_seq_id,
                nmr_query_sequence_length=record.nmr_query_sequence_length,
                xray_homolog_entry_ids=kept_entry_ids,
                xray_homolog_entity_ids=kept_entity_ids,
                has_xray_homolog=bool(kept_entity_ids),
            )
        )
    return historical_records


def read_solution_nmr_monomer_xray_rmsd_csv(
    input_path: Path,
) -> list[SolutionNMRMonomerXrayRmsdRecord]:
    """Read NMR-to-X-ray RMSD records from CSV."""
    if not input_path.exists():
        return []
    records: list[SolutionNMRMonomerXrayRmsdRecord] = []
    with input_path.open("r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if not row:
                continue
            nmr_core_start_raw = row.get("nmr_core_start_seq_id")
            nmr_core_end_raw = row.get("nmr_core_end_seq_id")
            xray_core_start_raw = row.get("xray_core_start_seq_id")
            xray_core_end_raw = row.get("xray_core_end_seq_id")
            records.append(
                SolutionNMRMonomerXrayRmsdRecord(
                    entry_id=str(row["entry_id"]),
                    year=int(row["year"]),
                    sequence_identity_percent=int(row["sequence_identity_percent"]),
                    nmr_chain_id=str(row["nmr_chain_id"]),
                    nmr_core_start_seq_id=(
                        int(nmr_core_start_raw)
                        if nmr_core_start_raw not in {None, ""}
                        else None
                    ),
                    nmr_core_end_seq_id=(
                        int(nmr_core_end_raw)
                        if nmr_core_end_raw not in {None, ""}
                        else None
                    ),
                    nmr_query_sequence_length=int(
                        row.get("nmr_query_sequence_length") or 0
                    ),
                    xray_homolog_entity_id=str(
                        row.get("xray_homolog_entity_id") or ""
                    ),
                    xray_homolog_count=int(row.get("xray_homolog_count") or 0),
                    xray_entry_id=str(row["xray_entry_id"]),
                    xray_chain_id=str(row["xray_chain_id"]),
                    xray_core_start_seq_id=(
                        int(xray_core_start_raw)
                        if xray_core_start_raw not in {None, ""}
                        else None
                    ),
                    xray_core_end_seq_id=(
                        int(xray_core_end_raw)
                        if xray_core_end_raw not in {None, ""}
                        else None
                    ),
                    xray_resolution_angstrom=float(row["xray_resolution_angstrom"]),
                    n_common_ca=int(row["n_common_ca"]),
                    rmsd_ca_angstrom=float(row["rmsd_ca_angstrom"]),
                )
            )
    return records


SOLUTION_NMR_MONOMER_XRAY_RMSD_HEADER: tuple[str, ...] = (
    "entry_id",
    "year",
    "sequence_identity_percent",
    "nmr_chain_id",
    "nmr_core_start_seq_id",
    "nmr_core_end_seq_id",
    "nmr_query_sequence_length",
    "xray_homolog_entity_id",
    "xray_homolog_count",
    "xray_entry_id",
    "xray_chain_id",
    "xray_core_start_seq_id",
    "xray_core_end_seq_id",
    "xray_resolution_angstrom",
    "n_common_ca",
    "rmsd_ca_angstrom",
)


def _solution_nmr_monomer_xray_rmsd_csv_row(
    record: SolutionNMRMonomerXrayRmsdRecord,
) -> tuple[Any, ...]:
    """Convert an NMR-to-X-ray RMSD record into a CSV row dictionary."""
    return (
        record.entry_id,
        record.year,
        record.sequence_identity_percent,
        record.nmr_chain_id,
        record.nmr_core_start_seq_id if record.nmr_core_start_seq_id is not None else "",
        record.nmr_core_end_seq_id if record.nmr_core_end_seq_id is not None else "",
        record.nmr_query_sequence_length,
        record.xray_homolog_entity_id,
        record.xray_homolog_count,
        record.xray_entry_id,
        record.xray_chain_id,
        (
            record.xray_core_start_seq_id
            if record.xray_core_start_seq_id is not None
            else ""
        ),
        record.xray_core_end_seq_id if record.xray_core_end_seq_id is not None else "",
        f"{record.xray_resolution_angstrom:.4f}",
        record.n_common_ca,
        f"{record.rmsd_ca_angstrom:.4f}",
    )


def write_solution_nmr_monomer_xray_rmsd_csv(
    records: list[SolutionNMRMonomerXrayRmsdRecord], output_path: Path
) -> None:
    """Write NMR-to-X-ray RMSD records to CSV."""
    write_csv_rows(
        output_path=output_path,
        header=list(SOLUTION_NMR_MONOMER_XRAY_RMSD_HEADER),
        rows=(_solution_nmr_monomer_xray_rmsd_csv_row(r) for r in records),
    )


SOLUTION_NMR_MONOMER_XRAY_RMSD_EXTREMES_HEADER: tuple[str, ...] = (
    "entry_id",
    "year",
    "sequence_identity_percent",
    "nmr_chain_id",
    "nmr_core_start_seq_id",
    "nmr_core_end_seq_id",
    "nmr_query_sequence_length",
    "xray_homolog_count",
    "successful_xray_homolog_count",
    "best_xray_homolog_entity_id",
    "best_xray_entry_id",
    "best_xray_chain_id",
    "best_xray_resolution_angstrom",
    "best_xray_core_start_seq_id",
    "best_xray_core_end_seq_id",
    "best_n_common_ca",
    "best_rmsd_ca_angstrom",
    "worst_xray_homolog_entity_id",
    "worst_xray_entry_id",
    "worst_xray_chain_id",
    "worst_xray_resolution_angstrom",
    "worst_xray_core_start_seq_id",
    "worst_xray_core_end_seq_id",
    "worst_n_common_ca",
    "worst_rmsd_ca_angstrom",
    "rmsd_delta_angstrom",
)


def _solution_nmr_monomer_xray_rmsd_extremes_csv_row(
    record: SolutionNMRMonomerXrayRmsdExtremesRecord,
) -> tuple[Any, ...]:
    """Convert an RMSD extremes record into a CSV row dictionary."""
    return (
        record.entry_id,
        record.year,
        record.sequence_identity_percent,
        record.nmr_chain_id,
        record.nmr_core_start_seq_id if record.nmr_core_start_seq_id is not None else "",
        record.nmr_core_end_seq_id if record.nmr_core_end_seq_id is not None else "",
        record.nmr_query_sequence_length,
        record.xray_homolog_count,
        record.successful_xray_homolog_count,
        record.best_xray_homolog_entity_id,
        record.best_xray_entry_id,
        record.best_xray_chain_id,
        f"{record.best_xray_resolution_angstrom:.4f}",
        (
            record.best_xray_core_start_seq_id
            if record.best_xray_core_start_seq_id is not None
            else ""
        ),
        (
            record.best_xray_core_end_seq_id
            if record.best_xray_core_end_seq_id is not None
            else ""
        ),
        record.best_n_common_ca,
        f"{record.best_rmsd_ca_angstrom:.4f}",
        record.worst_xray_homolog_entity_id,
        record.worst_xray_entry_id,
        record.worst_xray_chain_id,
        f"{record.worst_xray_resolution_angstrom:.4f}",
        (
            record.worst_xray_core_start_seq_id
            if record.worst_xray_core_start_seq_id is not None
            else ""
        ),
        (
            record.worst_xray_core_end_seq_id
            if record.worst_xray_core_end_seq_id is not None
            else ""
        ),
        record.worst_n_common_ca,
        f"{record.worst_rmsd_ca_angstrom:.4f}",
        f"{record.rmsd_delta_angstrom:.4f}",
    )


def read_solution_nmr_monomer_xray_rmsd_extremes_csv(
    input_path: Path,
) -> list[SolutionNMRMonomerXrayRmsdExtremesRecord]:
    """Read NMR-to-X-ray RMSD extremes records from CSV."""
    if not input_path.exists():
        return []
    records: list[SolutionNMRMonomerXrayRmsdExtremesRecord] = []
    with input_path.open("r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if not row:
                continue
            nmr_core_start_raw = row.get("nmr_core_start_seq_id")
            nmr_core_end_raw = row.get("nmr_core_end_seq_id")
            best_core_start_raw = row.get("best_xray_core_start_seq_id")
            best_core_end_raw = row.get("best_xray_core_end_seq_id")
            worst_core_start_raw = row.get("worst_xray_core_start_seq_id")
            worst_core_end_raw = row.get("worst_xray_core_end_seq_id")
            records.append(
                SolutionNMRMonomerXrayRmsdExtremesRecord(
                    entry_id=str(row["entry_id"]),
                    year=int(row["year"]),
                    sequence_identity_percent=int(row["sequence_identity_percent"]),
                    nmr_chain_id=str(row["nmr_chain_id"]),
                    nmr_core_start_seq_id=(
                        int(nmr_core_start_raw)
                        if nmr_core_start_raw not in {None, ""}
                        else None
                    ),
                    nmr_core_end_seq_id=(
                        int(nmr_core_end_raw)
                        if nmr_core_end_raw not in {None, ""}
                        else None
                    ),
                    nmr_query_sequence_length=int(
                        row.get("nmr_query_sequence_length") or 0
                    ),
                    xray_homolog_count=int(row.get("xray_homolog_count") or 0),
                    successful_xray_homolog_count=int(
                        row.get("successful_xray_homolog_count") or 0
                    ),
                    best_xray_homolog_entity_id=str(
                        row["best_xray_homolog_entity_id"]
                    ),
                    best_xray_entry_id=str(row["best_xray_entry_id"]),
                    best_xray_chain_id=str(row["best_xray_chain_id"]),
                    best_xray_resolution_angstrom=float(
                        row["best_xray_resolution_angstrom"]
                    ),
                    best_xray_core_start_seq_id=(
                        int(best_core_start_raw)
                        if best_core_start_raw not in {None, ""}
                        else None
                    ),
                    best_xray_core_end_seq_id=(
                        int(best_core_end_raw)
                        if best_core_end_raw not in {None, ""}
                        else None
                    ),
                    best_n_common_ca=int(row["best_n_common_ca"]),
                    best_rmsd_ca_angstrom=float(row["best_rmsd_ca_angstrom"]),
                    worst_xray_homolog_entity_id=str(
                        row["worst_xray_homolog_entity_id"]
                    ),
                    worst_xray_entry_id=str(row["worst_xray_entry_id"]),
                    worst_xray_chain_id=str(row["worst_xray_chain_id"]),
                    worst_xray_resolution_angstrom=float(
                        row["worst_xray_resolution_angstrom"]
                    ),
                    worst_xray_core_start_seq_id=(
                        int(worst_core_start_raw)
                        if worst_core_start_raw not in {None, ""}
                        else None
                    ),
                    worst_xray_core_end_seq_id=(
                        int(worst_core_end_raw)
                        if worst_core_end_raw not in {None, ""}
                        else None
                    ),
                    worst_n_common_ca=int(row["worst_n_common_ca"]),
                    worst_rmsd_ca_angstrom=float(row["worst_rmsd_ca_angstrom"]),
                    rmsd_delta_angstrom=float(row["rmsd_delta_angstrom"]),
                )
            )
    return records


def write_solution_nmr_monomer_xray_rmsd_extremes_csv(
    records: list[SolutionNMRMonomerXrayRmsdExtremesRecord],
    output_path: Path,
) -> None:
    """Write NMR-to-X-ray RMSD extremes records to CSV."""
    write_csv_rows(
        output_path=output_path,
        header=list(SOLUTION_NMR_MONOMER_XRAY_RMSD_EXTREMES_HEADER),
        rows=(
            _solution_nmr_monomer_xray_rmsd_extremes_csv_row(r) for r in records
        ),
    )


def collect_solution_nmr_monomer_xray_rmsd_to_csv(
    client: RCSBClient,
    config: CollectorConfig,
    homolog_input_path: Path,
    output_path: Path,
    cache_dir: Path,
    rmsd_workers: int,
    sequence_identity_percent: int,
    max_entries: int | None,
    overwrite: bool,
    log_label: str,
) -> None:
    """Collect best-match NMR-to-X-ray RMSD records and stream them to CSV."""
    existing_records: list[SolutionNMRMonomerXrayRmsdRecord] = []
    valid_existing_records: list[SolutionNMRMonomerXrayRmsdRecord] = []
    skip_entry_ids: set[str] = set()
    homolog_records = read_solution_nmr_monomer_xray_homolog_csv(
        homolog_input_path
    )
    if not homolog_records:
        raise SystemExit(
            f"No X-ray homolog records found for {log_label}. Run the matching "
            f"homolog dataset first or provide the expected CSV at {homolog_input_path}."
        )
    LOGGER.info(
        "%s %d%%: loaded %d homolog records from %s",
        log_label,
        sequence_identity_percent,
        len(homolog_records),
        homolog_input_path,
    )
    if not overwrite and output_path.exists():
        existing_records = read_solution_nmr_monomer_xray_rmsd_csv(output_path)
        existing_records = [
            record
            for record in existing_records
            if record.sequence_identity_percent == sequence_identity_percent
        ]
        valid_existing_records = [
            record
            for record in existing_records
            if record.nmr_core_start_seq_id is not None
            and record.nmr_core_end_seq_id is not None
            and record.xray_core_start_seq_id is not None
            and record.xray_core_end_seq_id is not None
            and record.xray_homolog_entity_id
        ]
        dropped_existing = len(existing_records) - len(valid_existing_records)
        skip_entry_ids = {record.entry_id for record in valid_existing_records}
        LOGGER.info(
            "%s %d%%: loaded %d existing records for resume (outdated=%d)",
            log_label,
            sequence_identity_percent,
            len(valid_existing_records),
            dropped_existing,
        )

    rmsd_collector = SolutionNMRMonomerXrayRmsdCollector(
        client=client,
        config=config,
        cache_dir=cache_dir,
        rmsd_workers=rmsd_workers,
        homolog_records=homolog_records,
        sequence_identity_percent=sequence_identity_percent,
    )
    valid_existing_records = sorted(
        valid_existing_records, key=lambda r: (r.year, r.entry_id)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(SOLUTION_NMR_MONOMER_XRAY_RMSD_HEADER)
        for record in valid_existing_records:
            writer.writerow(_solution_nmr_monomer_xray_rmsd_csv_row(record))
        csvfile.flush()

        def _on_xray_rmsd_record(record: SolutionNMRMonomerXrayRmsdRecord) -> None:
            """Persist one RMSD record while tracking processed seeds."""
            writer.writerow(_solution_nmr_monomer_xray_rmsd_csv_row(record))
            csvfile.flush()

        new_records = rmsd_collector.collect(
            max_entries=max_entries,
            skip_entry_ids=skip_entry_ids,
            on_record=_on_xray_rmsd_record,
        )

    LOGGER.info(
        "Saved %d records to %s (new: %d, identity=%d%%)",
        len(valid_existing_records) + len(new_records),
        output_path,
        len(new_records),
        sequence_identity_percent,
    )


def collect_solution_nmr_monomer_xray_rmsd_extremes_to_csv(
    client: RCSBClient,
    config: CollectorConfig,
    homolog_input_path: Path,
    output_path: Path,
    cache_dir: Path,
    rmsd_workers: int,
    sequence_identity_percent: int,
    max_entries: int | None,
    overwrite: bool,
    log_label: str,
) -> None:
    """Collect NMR-to-X-ray RMSD extremes and stream them to CSV."""
    existing_records: list[SolutionNMRMonomerXrayRmsdExtremesRecord] = []
    valid_existing_records: list[SolutionNMRMonomerXrayRmsdExtremesRecord] = []
    skip_entry_ids: set[str] = set()
    homolog_records = read_solution_nmr_monomer_xray_homolog_csv(
        homolog_input_path
    )
    if not homolog_records:
        raise SystemExit(
            f"No X-ray homolog records found for {log_label}. Run the matching "
            f"homolog dataset first or provide the expected CSV at {homolog_input_path}."
        )
    LOGGER.info(
        "%s %d%%: loaded %d homolog records from %s",
        log_label,
        sequence_identity_percent,
        len(homolog_records),
        homolog_input_path,
    )
    if not overwrite and output_path.exists():
        existing_records = read_solution_nmr_monomer_xray_rmsd_extremes_csv(
            output_path
        )
        existing_records = [
            record
            for record in existing_records
            if record.sequence_identity_percent == sequence_identity_percent
        ]
        valid_existing_records = [
            record
            for record in existing_records
            if record.best_xray_homolog_entity_id
            and record.worst_xray_homolog_entity_id
        ]
        dropped_existing = len(existing_records) - len(valid_existing_records)
        skip_entry_ids = {record.entry_id for record in valid_existing_records}
        LOGGER.info(
            "%s %d%%: loaded %d existing records for resume (outdated=%d)",
            log_label,
            sequence_identity_percent,
            len(valid_existing_records),
            dropped_existing,
        )

    rmsd_collector = SolutionNMRMonomerXrayRmsdCollector(
        client=client,
        config=config,
        cache_dir=cache_dir,
        rmsd_workers=rmsd_workers,
        homolog_records=homolog_records,
        sequence_identity_percent=sequence_identity_percent,
    )
    valid_existing_records = sorted(
        valid_existing_records, key=lambda r: (r.year, r.entry_id)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(SOLUTION_NMR_MONOMER_XRAY_RMSD_EXTREMES_HEADER)
        for record in valid_existing_records:
            writer.writerow(_solution_nmr_monomer_xray_rmsd_extremes_csv_row(record))
        csvfile.flush()

        def _on_xray_rmsd_extremes_record(
            record: SolutionNMRMonomerXrayRmsdExtremesRecord,
        ) -> None:
            """Persist one RMSD extremes record while tracking processed seeds."""
            writer.writerow(_solution_nmr_monomer_xray_rmsd_extremes_csv_row(record))
            csvfile.flush()

        new_records = rmsd_collector.collect_extremes(
            max_entries=max_entries,
            skip_entry_ids=skip_entry_ids,
            on_record=_on_xray_rmsd_extremes_record,
        )

    LOGGER.info(
        "Saved %d records to %s (new: %d, identity=%d%%)",
        len(valid_existing_records) + len(new_records),
        output_path,
        len(new_records),
        sequence_identity_percent,
    )


def parse_dataset_kinds(raw_value: str) -> list[DatasetKind]:
    """Parse comma-separated dataset names into DatasetKind values."""
    if raw_value.strip().lower() == "all":
        return [
            DatasetKind.METHOD_COUNTS,
            DatasetKind.MEMBRANE_PROTEIN_COUNTS,
            DatasetKind.SOLUTION_NMR_PROGRAM_COUNTS,
            DatasetKind.SOLUTION_NMR_MONOMER_PROGRAM_CLUSTERS,
            DatasetKind.SOLUTION_NMR_WEIGHTS,
            DatasetKind.SOLUTION_NMR_MONOMER_STRIDE_MODELED_FIRST_MODEL,
            DatasetKind.SOLUTION_NMR_MONOMER_PRECISION_STRIDE_MODELED_FIRST_MODEL,
            DatasetKind.SOLUTION_NMR_MONOMER_QUALITY,
            DatasetKind.SOLUTION_NMR_MONOMER_XRAY_HOMOLOGS,
            DatasetKind.SOLUTION_NMR_MONOMER_XRAY_HOMOLOGS_HISTORICAL,
            DatasetKind.SOLUTION_NMR_MONOMER_XRAY_RMSD,
            DatasetKind.SOLUTION_NMR_MONOMER_XRAY_RMSD_HISTORICAL,
            DatasetKind.SOLUTION_NMR_MONOMER_XRAY_RMSD_EXTREMES,
            DatasetKind.SOLUTION_NMR_MONOMER_XRAY_RMSD_EXTREMES_HISTORICAL,
        ]
    raw_items = [item.strip() for item in raw_value.split(",") if item.strip()]
    selected: list[DatasetKind] = []
    for item in raw_items:
        try:
            selected.append(DatasetKind(item))
        except ValueError as exc:
            valid = ", ".join(dataset.value for dataset in DatasetKind)
            raise argparse.ArgumentTypeError(
                f"Unknown dataset '{item}'. Use one of: {valid}, all."
            ) from exc
    if not selected:
        raise argparse.ArgumentTypeError("No datasets selected.")
    return selected


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the collector CLI."""
    parser = argparse.ArgumentParser(
        description="Collect extensible PDB datasets from RCSB APIs."
    )
    parser.add_argument(
        "--datasets",
        type=parse_dataset_kinds,
        default=[
            DatasetKind.METHOD_COUNTS,
            DatasetKind.SOLUTION_NMR_WEIGHTS,
            DatasetKind.SOLUTION_NMR_MONOMER_STRIDE_MODELED_FIRST_MODEL,
        ],
        help="Comma-separated dataset kinds or 'all'. "
        "Available: method_counts, membrane_protein_counts, solution_nmr_program_counts, solution_nmr_monomer_program_clusters, solution_nmr_weights, solution_nmr_monomer_stride_modeled_first_model, solution_nmr_monomer_precision_stride_modeled_first_model, solution_nmr_monomer_quality, solution_nmr_monomer_xray_homologs, solution_nmr_monomer_xray_homologs_historical, solution_nmr_monomer_xray_rmsd, solution_nmr_monomer_xray_rmsd_historical, solution_nmr_monomer_xray_rmsd_extremes, solution_nmr_monomer_xray_rmsd_extremes_historical.",
    )
    parser.add_argument(
        "--counts-output",
        type=Path,
        default=Path("data/pdb_method_counts_by_year.csv"),
        help="Output CSV path for method_counts dataset.",
    )
    parser.add_argument(
        "--membrane-counts-output",
        type=Path,
        default=Path("data/membrane_protein_counts_by_year.csv"),
        help="Output CSV path for membrane_protein_counts dataset.",
    )
    parser.add_argument(
        "--membrane-method-counts-output",
        type=Path,
        default=Path("data/membrane_protein_method_counts_by_year.csv"),
        help=(
            "Output CSV path for membrane protein counts split by experimental "
            "method."
        ),
    )
    parser.add_argument(
        "--solution-nmr-output",
        type=Path,
        default=Path("data/solution_nmr_structure_weights.csv"),
        help="Output CSV path for solution_nmr_weights dataset.",
    )
    parser.add_argument(
        "--solution-nmr-program-counts-output",
        type=Path,
        default=Path("data/solution_nmr_program_counts_by_year.csv"),
        help="Output CSV path for solution_nmr_program_counts dataset.",
    )
    parser.add_argument(
        "--solution-nmr-program-cache-dir",
        type=Path,
        default=Path("data/pdb_cache"),
        help="Directory to cache downloaded PDB files for solution_nmr_program_counts dataset.",
    )
    parser.add_argument(
        "--solution-nmr-program-cache-only",
        action="store_true",
        help=(
            "Use only already cached PDB files for solution_nmr_program_counts "
            "(skip downloading missing files)."
        ),
    )
    parser.add_argument(
        "--solution-nmr-monomer-stride-modeled-first-model-output",
        type=Path,
        default=Path("data/solution_nmr_monomer_stride_modeled_first_model.csv"),
        help=(
            "Output CSV path for solution_nmr_monomer_stride_modeled_first_model "
            "dataset (STRIDE for modeled residues of the first model only)."
        ),
    )
    parser.add_argument(
        "--solution-nmr-monomer-stride-cache-dir",
        type=Path,
        default=Path("data/pdb_cache"),
        help="Directory to cache downloaded PDB files for STRIDE in monomer-stride dataset.",
    )
    parser.add_argument(
        "--solution-nmr-monomer-stride-executable",
        type=str,
        default="",
        help=(
            "Path to STRIDE executable for monomer-stride dataset. "
            "If omitted, script tries stride and /tmp/stride_src/src/stride."
        ),
    )
    parser.add_argument(
        "--solution-nmr-monomer-precision-stride-modeled-first-model-output",
        type=Path,
        default=Path("data/solution_nmr_monomer_precision_stride_modeled_first_model.csv"),
        help=(
            "Output CSV path for solution_nmr_monomer_precision_stride_modeled_first_model "
            "dataset."
        ),
    )
    parser.add_argument(
        "--solution-nmr-monomer-quality-output",
        type=Path,
        default=Path("data/solution_nmr_monomer_quality_metrics.csv"),
        help="Output CSV path for solution_nmr_monomer_quality dataset.",
    )
    parser.add_argument(
        "--solution-nmr-monomer-program-cluster-input",
        type=Path,
        default=Path("data/solution_nmr_monomer_quality_metrics.csv"),
        help=(
            "Input CSV path for solution_nmr_monomer_program_clusters dataset. "
            "Expected format: solution_nmr_monomer_quality_metrics.csv."
        ),
    )
    parser.add_argument(
        "--solution-nmr-monomer-program-cluster-cache-dir",
        type=Path,
        default=Path("data/pdb_cache"),
        help=(
            "Directory with cached PDB files for solution_nmr_monomer_program_clusters "
            "dataset."
        ),
    )
    parser.add_argument(
        "--solution-nmr-monomer-program-cluster-assignment-output",
        type=Path,
        default=Path("data/solution_nmr_monomer_program_cluster_assignments.csv"),
        help=(
            "Output CSV path for per-entry assignments in "
            "solution_nmr_monomer_program_clusters dataset."
        ),
    )
    parser.add_argument(
        "--solution-nmr-monomer-program-cluster-summary-output",
        type=Path,
        default=Path("data/solution_nmr_monomer_program_cluster_quality_by_year.csv"),
        help=(
            "Output CSV path for yearly cluster summary in "
            "solution_nmr_monomer_program_clusters dataset."
        ),
    )
    parser.add_argument(
        "--solution-nmr-monomer-program-cluster-yearly-summary-output",
        type=Path,
        default=Path("data/solution_nmr_monomer_program_cluster_quality_total_by_year.csv"),
        help=(
            "Output CSV path for yearly totals across all program clusters in "
            "solution_nmr_monomer_program_clusters dataset."
        ),
    )
    parser.add_argument(
        "--solution-nmr-monomer-program-cluster-total-output",
        type=Path,
        default=Path("data/solution_nmr_monomer_program_cluster_quality_total.csv"),
        help=(
            "Output CSV path for totals across all years per program cluster in "
            "solution_nmr_monomer_program_clusters dataset."
        ),
    )
    parser.add_argument(
        "--solution-nmr-monomer-xray-homolog-95-output",
        type=Path,
        default=Path("data/solution_nmr_monomer_xray_homologs_95.csv"),
        help="Output CSV path for solution_nmr_monomer_xray_homologs dataset at 95%% sequence identity.",
    )
    parser.add_argument(
        "--solution-nmr-monomer-xray-homolog-100-output",
        type=Path,
        default=Path("data/solution_nmr_monomer_xray_homologs_100.csv"),
        help="Output CSV path for solution_nmr_monomer_xray_homologs dataset at 100%% sequence identity.",
    )
    parser.add_argument(
        "--solution-nmr-monomer-xray-homolog-95-historical-output",
        type=Path,
        default=Path("data/solution_nmr_monomer_xray_homologs_95_historical.csv"),
        help=(
            "Output CSV path for 95%% X-ray homologs released no later than "
            "the NMR entry deposit date."
        ),
    )
    parser.add_argument(
        "--solution-nmr-monomer-xray-homolog-100-historical-output",
        type=Path,
        default=Path("data/solution_nmr_monomer_xray_homologs_100_historical.csv"),
        help=(
            "Output CSV path for 100%% X-ray homologs released no later than "
            "the NMR entry deposit date."
        ),
    )
    parser.add_argument(
        "--solution-nmr-monomer-xray-rmsd-output",
        type=Path,
        default=Path("data/solution_nmr_monomer_xray_rmsd.csv"),
        help="Output CSV path for solution_nmr_monomer_xray_rmsd dataset.",
    )
    parser.add_argument(
        "--solution-nmr-monomer-xray-rmsd-historical-output",
        type=Path,
        default=Path("data/solution_nmr_monomer_xray_rmsd_historical.csv"),
        help=(
            "Output CSV path for X-ray RMSD calculated from already-released "
            "historical homologs."
        ),
    )
    parser.add_argument(
        "--solution-nmr-monomer-xray-rmsd-extremes-output",
        type=Path,
        default=Path("data/solution_nmr_monomer_xray_rmsd_extremes.csv"),
        help=(
            "Output CSV path for solution_nmr_monomer_xray_rmsd_extremes "
            "dataset."
        ),
    )
    parser.add_argument(
        "--solution-nmr-monomer-xray-rmsd-extremes-historical-output",
        type=Path,
        default=Path("data/solution_nmr_monomer_xray_rmsd_extremes_historical.csv"),
        help=(
            "Output CSV path for X-ray RMSD extremes calculated from "
            "already-released historical homologs."
        ),
    )
    parser.add_argument(
        "--precision-cache-dir",
        type=Path,
        default=Path("data/pdb_cache"),
        help="Directory to cache downloaded PDB files for precision calculation.",
    )
    parser.add_argument(
        "--precision-max-entries",
        type=int,
        default=None,
        help="Optional limit of entries to process for precision dataset.",
    )
    parser.add_argument(
        "--precision-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help="Parallel workers for RMSD precision computation.",
    )
    parser.add_argument(
        "--precision-overwrite",
        action="store_true",
        help="Recompute precision CSV from scratch (ignore existing rows).",
    )
    parser.add_argument(
        "--xray-rmsd-cache-dir",
        type=Path,
        default=Path("data/pdb_cache"),
        help="Directory to cache downloaded PDB files for X-ray RMSD calculation.",
    )
    parser.add_argument(
        "--xray-rmsd-max-entries",
        type=int,
        default=None,
        help="Optional limit of entries to process for X-ray RMSD dataset.",
    )
    parser.add_argument(
        "--xray-rmsd-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help="Parallel workers for X-ray RMSD computation.",
    )
    parser.add_argument(
        "--xray-rmsd-overwrite",
        action="store_true",
        help="Recompute X-ray RMSD CSV from scratch (ignore existing rows).",
    )
    parser.add_argument(
        "--xray-rmsd-sequence-identity",
        type=int,
        choices=(95, 100),
        default=100,
        help=(
            "Sequence identity cutoff for selecting X-ray homologs from the "
            "STRIDE-core homolog CSV (95 or 100)."
        ),
    )
    parser.add_argument(
        "--page-size", type=int, default=10000, help="Search API page size."
    )
    parser.add_argument(
        "--batch-size", type=int, default=300, help="GraphQL batch size."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help="Parallel workers for GraphQL calls.",
    )
    parser.add_argument(
        "--log-level", default="INFO", help="Logging level (DEBUG/INFO/WARNING/ERROR)."
    )
    return parser.parse_args()


def main() -> None:
    """Run the requested dataset collection workflow from the CLI."""
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    config = CollectorConfig(
        page_size=args.page_size,
        graphql_batch_size=args.batch_size,
        max_workers=args.workers,
    )
    client = RCSBClient(config=config)

    if DatasetKind.METHOD_COUNTS in args.datasets:
        method_collector = PDBMethodYearlyCollector(client=client, config=config)
        method_records = method_collector.collect(
            [
                ExperimentalMethod.X_RAY,
                ExperimentalMethod.CRYO_EM,
                ExperimentalMethod.NMR,
            ]
        )
        write_method_counts_csv(records=method_records, output_path=args.counts_output)
        LOGGER.info("Saved %d records to %s", len(method_records), args.counts_output)

    if DatasetKind.MEMBRANE_PROTEIN_COUNTS in args.datasets:
        membrane_collector = MembraneProteinYearlyCollector(
            client=client, config=config
        )
        membrane_records = membrane_collector.collect()
        write_membrane_counts_csv(
            records=membrane_records, output_path=args.membrane_counts_output
        )
        membrane_method_records = membrane_collector.collect_by_method(
            [
                ExperimentalMethod.X_RAY,
                ExperimentalMethod.CRYO_EM,
                ExperimentalMethod.NMR,
            ]
        )
        write_method_counts_csv(
            records=membrane_method_records,
            output_path=args.membrane_method_counts_output,
        )
        LOGGER.info(
            "Saved %d records to %s",
            len(membrane_records),
            args.membrane_counts_output,
        )
        LOGGER.info(
            "Saved %d records to %s",
            len(membrane_method_records),
            args.membrane_method_counts_output,
        )

    if DatasetKind.SOLUTION_NMR_PROGRAM_COUNTS in args.datasets:
        nmr_program_collector = SolutionNMRProgramYearlyCollector(
            client=client,
            config=config,
            cache_dir=Path(args.solution_nmr_program_cache_dir),
            download_missing=not args.solution_nmr_program_cache_only,
        )
        nmr_program_records = nmr_program_collector.collect()
        write_solution_nmr_program_counts_csv(
            records=nmr_program_records,
            output_path=args.solution_nmr_program_counts_output,
        )
        LOGGER.info(
            "Saved %d records to %s",
            len(nmr_program_records),
            args.solution_nmr_program_counts_output,
        )

    if DatasetKind.SOLUTION_NMR_WEIGHTS in args.datasets:
        nmr_weight_collector = SolutionNMRWeightCollector(client=client, config=config)
        nmr_weight_records = nmr_weight_collector.collect()
        write_solution_nmr_weights_csv(
            records=nmr_weight_records, output_path=args.solution_nmr_output
        )
        LOGGER.info(
            "Saved %d records to %s",
            len(nmr_weight_records),
            args.solution_nmr_output,
        )

    if DatasetKind.SOLUTION_NMR_MONOMER_STRIDE_MODELED_FIRST_MODEL in args.datasets:
        modeled_first_stride_executable = resolve_stride_executable(
            args.solution_nmr_monomer_stride_executable
        )
        if modeled_first_stride_executable is None:
            raise RuntimeError(
                "STRIDE executable not found for solution_nmr_monomer_stride_modeled_first_model. "
                "Provide --solution-nmr-monomer-stride-executable or install stride."
            )
        LOGGER.info(
            (
                "SOLUTION NMR monomer-stride-modeled-first-model: "
                "using STRIDE executable %s"
            ),
            modeled_first_stride_executable,
        )
        modeled_stride_collector = SolutionNMRMonomerStrideModeledFirstModelCollector(
            client=client,
            config=config,
            stride_executable=modeled_first_stride_executable,
            cache_dir=Path(args.solution_nmr_monomer_stride_cache_dir),
        )
        modeled_stride_count = (
            stream_solution_nmr_monomer_stride_modeled_first_model_csv(
                records=modeled_stride_collector.iter_records(),
                output_path=Path(
                    args.solution_nmr_monomer_stride_modeled_first_model_output
                ),
            )
        )
        LOGGER.info(
            "Saved %d records to %s",
            modeled_stride_count,
            args.solution_nmr_monomer_stride_modeled_first_model_output,
        )

    if (
        DatasetKind.SOLUTION_NMR_MONOMER_PRECISION_STRIDE_MODELED_FIRST_MODEL
        in args.datasets
    ):
        precision_stride_executable = resolve_stride_executable(
            args.solution_nmr_monomer_stride_executable
        )
        if precision_stride_executable is None:
            raise RuntimeError(
                "STRIDE executable not found for "
                "solution_nmr_monomer_precision_stride_modeled_first_model. "
                "Provide --solution-nmr-monomer-stride-executable or install stride."
            )

        existing_records = []
        skip_entry_ids = set()
        precision_stride_output_path = Path(
            args.solution_nmr_monomer_precision_stride_modeled_first_model_output
        )
        if not args.precision_overwrite and precision_stride_output_path.exists():
            existing_records = read_solution_nmr_monomer_precision_csv(
                precision_stride_output_path
            )
            skip_entry_ids = {record.entry_id for record in existing_records}
            LOGGER.info(
                "SOLUTION NMR precision STRIDE modeled-first-model: loaded %d existing records for resume",
                len(existing_records),
            )

        precision_stride_collector = (
            SolutionNMRMonomerPrecisionStrideModeledFirstModelCollector(
                client=client,
                config=config,
                cache_dir=Path(args.precision_cache_dir),
                precision_workers=args.precision_workers,
                stride_executable=precision_stride_executable,
            )
        )
        existing_records = sorted(existing_records, key=lambda r: (r.year, r.entry_id))
        precision_stride_output_path.parent.mkdir(parents=True, exist_ok=True)
        with precision_stride_output_path.open(
            "w", newline="", encoding="utf-8"
        ) as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(SOLUTION_NMR_MONOMER_PRECISION_HEADER)
            for record in existing_records:
                writer.writerow(_solution_nmr_monomer_precision_csv_row(record))
            csvfile.flush()

            def _on_precision_stride_record(
                record: SolutionNMRMonomerPrecisionRecord,
            ) -> None:
                """Persist one STRIDE-core precision record while tracking processed seeds."""
                writer.writerow(_solution_nmr_monomer_precision_csv_row(record))
                csvfile.flush()

            new_records = precision_stride_collector.collect(
                max_entries=args.precision_max_entries,
                skip_entry_ids=skip_entry_ids,
                on_record=_on_precision_stride_record,
            )

        LOGGER.info(
            "Saved %d records to %s (new: %d)",
            len(existing_records) + len(new_records),
            args.solution_nmr_monomer_precision_stride_modeled_first_model_output,
            len(new_records),
        )

    if DatasetKind.SOLUTION_NMR_MONOMER_QUALITY in args.datasets:
        quality_collector = SolutionNMRMonomerQualityCollector(
            client=client, config=config
        )
        quality_records = quality_collector.collect()
        write_solution_nmr_monomer_quality_csv(
            records=quality_records,
            output_path=args.solution_nmr_monomer_quality_output,
        )
        LOGGER.info(
            "Saved %d records to %s",
            len(quality_records),
            args.solution_nmr_monomer_quality_output,
        )

    if DatasetKind.SOLUTION_NMR_MONOMER_PROGRAM_CLUSTERS in args.datasets:
        quality_input_path = Path(args.solution_nmr_monomer_program_cluster_input)
        quality_records = read_solution_nmr_monomer_quality_csv(quality_input_path)
        if not quality_records:
            raise RuntimeError(
                "solution_nmr_monomer_program_clusters requires a non-empty quality "
                f"CSV at {quality_input_path}"
            )
        cluster_collector = SolutionNMRMonomerProgramClusterCollector(
            quality_records=quality_records,
            cache_dir=Path(args.solution_nmr_monomer_program_cluster_cache_dir),
            max_workers=config.max_workers,
        )
        assignment_records, summary_records = cluster_collector.collect()
        write_solution_nmr_monomer_program_cluster_assignments_csv(
            records=assignment_records,
            output_path=Path(
                args.solution_nmr_monomer_program_cluster_assignment_output
            ),
        )
        write_solution_nmr_monomer_program_cluster_summary_csv(
            records=summary_records,
            output_path=Path(args.solution_nmr_monomer_program_cluster_summary_output),
        )
        yearly_summary_records = (
            summarize_solution_nmr_monomer_program_cluster_quality_by_year(
                assignment_records=assignment_records,
                quality_records=quality_records,
            )
        )
        write_solution_nmr_monomer_program_cluster_yearly_summary_csv(
            records=yearly_summary_records,
            output_path=Path(
                args.solution_nmr_monomer_program_cluster_yearly_summary_output
            ),
        )
        total_records = summarize_solution_nmr_monomer_program_cluster_quality_total(
            assignment_records=assignment_records,
            quality_records=quality_records,
        )
        write_solution_nmr_monomer_program_cluster_total_csv(
            records=total_records,
            output_path=Path(args.solution_nmr_monomer_program_cluster_total_output),
        )
        LOGGER.info(
            "Saved %d records to %s",
            len(assignment_records),
            args.solution_nmr_monomer_program_cluster_assignment_output,
        )
        LOGGER.info(
            "Saved %d records to %s",
            len(summary_records),
            args.solution_nmr_monomer_program_cluster_summary_output,
        )
        LOGGER.info(
            "Saved %d records to %s",
            len(yearly_summary_records),
            args.solution_nmr_monomer_program_cluster_yearly_summary_output,
        )
        LOGGER.info(
            "Saved %d records to %s",
            len(total_records),
            args.solution_nmr_monomer_program_cluster_total_output,
        )

    if DatasetKind.SOLUTION_NMR_MONOMER_XRAY_HOMOLOGS in args.datasets:
        homolog_stride_executable = resolve_stride_executable(
            args.solution_nmr_monomer_stride_executable
        )
        if homolog_stride_executable is None:
            raise SystemExit(
                "STRIDE executable not found for solution_nmr_monomer_xray_homologs. "
                "Provide --solution-nmr-monomer-stride-executable or install stride."
            )
        LOGGER.info(
            "SOLUTION NMR monomer X-ray homologs: using STRIDE executable %s",
            homolog_stride_executable,
        )
        homolog_collector = SolutionNMRMonomerXrayHomologCollector(
            client=client,
            config=config,
            stride_executable=homolog_stride_executable,
            cache_dir=Path(args.solution_nmr_monomer_stride_cache_dir),
        )
        homolog_95_output_path = Path(args.solution_nmr_monomer_xray_homolog_95_output)
        homolog_100_output_path = Path(args.solution_nmr_monomer_xray_homolog_100_output)
        homolog_95_output_path.parent.mkdir(parents=True, exist_ok=True)
        homolog_100_output_path.parent.mkdir(parents=True, exist_ok=True)

        with (
            homolog_95_output_path.open("w", newline="", encoding="utf-8") as file_95,
            homolog_100_output_path.open("w", newline="", encoding="utf-8") as file_100,
        ):
            writer_95 = csv.writer(file_95)
            writer_100 = csv.writer(file_100)
            writer_95.writerow(SOLUTION_NMR_MONOMER_XRAY_HOMOLOG_HEADER)
            writer_100.writerow(SOLUTION_NMR_MONOMER_XRAY_HOMOLOG_HEADER)
            file_95.flush()
            file_100.flush()

            def _on_homolog_record_pair(
                record_95: SolutionNMRMonomerXrayHomologRecord,
                record_100: SolutionNMRMonomerXrayHomologRecord,
            ) -> None:
                """Persist current and historical homolog records for one seed."""
                writer_95.writerow(
                    _solution_nmr_monomer_xray_homolog_csv_row(record_95)
                )
                writer_100.writerow(
                    _solution_nmr_monomer_xray_homolog_csv_row(record_100)
                )
                file_95.flush()
                file_100.flush()

            records_95, records_100 = homolog_collector.collect(
                on_record_pair=_on_homolog_record_pair
            )
        LOGGER.info(
            "Saved %d records to %s",
            len(records_95),
            homolog_95_output_path,
        )
        LOGGER.info(
            "Saved %d records to %s",
            len(records_100),
            homolog_100_output_path,
        )

    if DatasetKind.SOLUTION_NMR_MONOMER_XRAY_HOMOLOGS_HISTORICAL in args.datasets:
        homolog_95_input_path = Path(args.solution_nmr_monomer_xray_homolog_95_output)
        homolog_100_input_path = Path(args.solution_nmr_monomer_xray_homolog_100_output)
        records_95 = read_solution_nmr_monomer_xray_homolog_csv(
            homolog_95_input_path
        )
        records_100 = read_solution_nmr_monomer_xray_homolog_csv(
            homolog_100_input_path
        )
        if not records_95 or not records_100:
            raise SystemExit(
                "No X-ray homolog records found for historical filtering. Run "
                "solution_nmr_monomer_xray_homologs first or provide the expected "
                f"homolog CSVs at {homolog_95_input_path} and {homolog_100_input_path}."
            )
        historical_records_95 = filter_xray_homolog_records_by_deposit_date(
            records=records_95,
            client=client,
            config=config,
        )
        historical_records_100 = filter_xray_homolog_records_by_deposit_date(
            records=records_100,
            client=client,
            config=config,
        )
        homolog_95_historical_output_path = Path(
            args.solution_nmr_monomer_xray_homolog_95_historical_output
        )
        homolog_100_historical_output_path = Path(
            args.solution_nmr_monomer_xray_homolog_100_historical_output
        )
        write_solution_nmr_monomer_xray_homolog_csv(
            records=historical_records_95,
            output_path=homolog_95_historical_output_path,
        )
        write_solution_nmr_monomer_xray_homolog_csv(
            records=historical_records_100,
            output_path=homolog_100_historical_output_path,
        )
        LOGGER.info(
            "Saved %d historical records to %s",
            len(historical_records_95),
            homolog_95_historical_output_path,
        )
        LOGGER.info(
            "Saved %d historical records to %s",
            len(historical_records_100),
            homolog_100_historical_output_path,
        )

    if DatasetKind.SOLUTION_NMR_MONOMER_XRAY_RMSD in args.datasets:
        homolog_input_path = (
            Path(args.solution_nmr_monomer_xray_homolog_95_output)
            if args.xray_rmsd_sequence_identity == 95
            else Path(args.solution_nmr_monomer_xray_homolog_100_output)
        )
        collect_solution_nmr_monomer_xray_rmsd_to_csv(
            client=client,
            config=config,
            homolog_input_path=homolog_input_path,
            output_path=Path(args.solution_nmr_monomer_xray_rmsd_output),
            cache_dir=Path(args.xray_rmsd_cache_dir),
            rmsd_workers=args.xray_rmsd_workers,
            sequence_identity_percent=args.xray_rmsd_sequence_identity,
            max_entries=args.xray_rmsd_max_entries,
            overwrite=args.xray_rmsd_overwrite,
            log_label="SOLUTION NMR X-ray RMSD",
        )

    if DatasetKind.SOLUTION_NMR_MONOMER_XRAY_RMSD_HISTORICAL in args.datasets:
        homolog_input_path = (
            Path(args.solution_nmr_monomer_xray_homolog_95_historical_output)
            if args.xray_rmsd_sequence_identity == 95
            else Path(args.solution_nmr_monomer_xray_homolog_100_historical_output)
        )
        collect_solution_nmr_monomer_xray_rmsd_to_csv(
            client=client,
            config=config,
            homolog_input_path=homolog_input_path,
            output_path=Path(args.solution_nmr_monomer_xray_rmsd_historical_output),
            cache_dir=Path(args.xray_rmsd_cache_dir),
            rmsd_workers=args.xray_rmsd_workers,
            sequence_identity_percent=args.xray_rmsd_sequence_identity,
            max_entries=args.xray_rmsd_max_entries,
            overwrite=args.xray_rmsd_overwrite,
            log_label="SOLUTION NMR historical X-ray RMSD",
        )

    if DatasetKind.SOLUTION_NMR_MONOMER_XRAY_RMSD_EXTREMES in args.datasets:
        homolog_input_path = (
            Path(args.solution_nmr_monomer_xray_homolog_95_output)
            if args.xray_rmsd_sequence_identity == 95
            else Path(args.solution_nmr_monomer_xray_homolog_100_output)
        )
        collect_solution_nmr_monomer_xray_rmsd_extremes_to_csv(
            client=client,
            config=config,
            homolog_input_path=homolog_input_path,
            output_path=Path(args.solution_nmr_monomer_xray_rmsd_extremes_output),
            cache_dir=Path(args.xray_rmsd_cache_dir),
            rmsd_workers=args.xray_rmsd_workers,
            sequence_identity_percent=args.xray_rmsd_sequence_identity,
            max_entries=args.xray_rmsd_max_entries,
            overwrite=args.xray_rmsd_overwrite,
            log_label="SOLUTION NMR X-ray RMSD extremes",
        )

    if (
        DatasetKind.SOLUTION_NMR_MONOMER_XRAY_RMSD_EXTREMES_HISTORICAL
        in args.datasets
    ):
        homolog_input_path = (
            Path(args.solution_nmr_monomer_xray_homolog_95_historical_output)
            if args.xray_rmsd_sequence_identity == 95
            else Path(args.solution_nmr_monomer_xray_homolog_100_historical_output)
        )
        collect_solution_nmr_monomer_xray_rmsd_extremes_to_csv(
            client=client,
            config=config,
            homolog_input_path=homolog_input_path,
            output_path=Path(
                args.solution_nmr_monomer_xray_rmsd_extremes_historical_output
            ),
            cache_dir=Path(args.xray_rmsd_cache_dir),
            rmsd_workers=args.xray_rmsd_workers,
            sequence_identity_percent=args.xray_rmsd_sequence_identity,
            max_entries=args.xray_rmsd_max_entries,
            overwrite=args.xray_rmsd_overwrite,
            log_label="SOLUTION NMR historical X-ray RMSD extremes",
        )


if __name__ == "__main__":
    main()
