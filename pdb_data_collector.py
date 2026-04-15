from __future__ import annotations

import argparse
import csv
import logging
import os
import shutil
import subprocess
import tempfile
import time
import requests
import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP as BioDSSP
from Bio.SeqUtils import molecular_weight as sequence_molecular_weight
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence, TypeVar
from MDAnalysis.analysis.rms import rmsd as mda_rmsd

LOGGER = logging.getLogger(__name__)
T = TypeVar("T")

SOLUTION_NMR_METHOD = "SOLUTION NMR"
PROTEIN_MONOMER_ENTITY_TYPES: frozenset[str] = frozenset(
    {"polypeptide(L)", "polypeptide(D)"}
)
PROTEIN_POLYMER_TYPE = "Protein"
SEQUENCE_IDENTITY_AGGREGATION_METHOD = "sequence_identity"
UNMODELED_INSTANCE_FEATURE_TYPES: frozenset[str] = frozenset(
    {
        "UNOBSERVED_RESIDUE_XYZ",
        "ZERO_OCCUPANCY_RESIDUE_XYZ",
        "UNMODELED_RESIDUE_XYZ",
        "MISSING_RESIDUE",
    }
)
DEFAULT_PDB_CACHE_DIR = Path("data/pdb_cache")
LOCAL_DSSP_CANDIDATE = Path("dssp/build/mkdssp")
LOCAL_STRIDE_CANDIDATE = Path("/tmp/stride_src/src/stride")
DSSP_STATE_CODES: tuple[str, ...] = ("H", "B", "E", "G", "I", "T", "S", "-")
STRIDE_STATE_CODES: tuple[str, ...] = ("H", "G", "I", "E", "B", "T", "C")
DEFAULT_MAX_WORKERS = max(1, os.cpu_count() or 1)
print(f"Using up to {DEFAULT_MAX_WORKERS} worker threads for concurrent tasks")


class ExperimentalMethod(Enum):
    X_RAY = ("X-ray", ("X-RAY DIFFRACTION",))
    CRYO_EM = ("cryo-EM", ("ELECTRON MICROSCOPY",))
    NMR = ("NMR", ("SOLUTION NMR", "SOLID-STATE NMR"))

    @property
    def label(self) -> str:
        return self.value[0]

    @property
    def query_values(self) -> tuple[str, ...]:
        return self.value[1]


class DatasetKind(str, Enum):
    METHOD_COUNTS = "method_counts"
    MEMBRANE_PROTEIN_COUNTS = "membrane_protein_counts"
    SOLUTION_NMR_WEIGHTS = "solution_nmr_weights"
    SOLUTION_NMR_MONOMER_SECONDARY = "solution_nmr_monomer_secondary"
    SOLUTION_NMR_MONOMER_SECONDARY_MODELED_FIRST_MODEL = (
        "solution_nmr_monomer_secondary_modeled_first_model"
    )
    SOLUTION_NMR_MONOMER_STRIDE = "solution_nmr_monomer_stride"
    SOLUTION_NMR_MONOMER_STRIDE_MODELED_FIRST_MODEL = (
        "solution_nmr_monomer_stride_modeled_first_model"
    )
    SOLUTION_NMR_MONOMER_PRECISION = "solution_nmr_monomer_precision"
    SOLUTION_NMR_MONOMER_QUALITY = "solution_nmr_monomer_quality"
    SOLUTION_NMR_MONOMER_XRAY_HOMOLOGS = "solution_nmr_monomer_xray_homologs"
    SOLUTION_NMR_MONOMER_XRAY_RMSD = "solution_nmr_monomer_xray_rmsd"


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
class YearlyCountRecord:
    year: int
    method: str
    count: int


@dataclass(frozen=True)
class MembraneYearlyCountRecord:
    year: int
    count: int


@dataclass(frozen=True)
class SolutionNMRWeightRecord:
    entry_id: str
    year: int
    molecular_weight_kda: float
    rcsb_entry_molecular_weight_kda: float | None
    polymer_molecular_weight_maximum_kda: float | None
    modeled_molecular_weight_kda: float | None


@dataclass(frozen=True)
class SolutionNMRMonomerSecondaryRecord:
    entry_id: str
    year: int
    sequence_length: int
    secondary_structure_percent: float
    helix_fraction: float
    sheet_fraction: float
    deposited_model_count: int
    dssp_alpha_helix_fraction: float
    dssp_isolated_beta_bridge_fraction: float
    dssp_beta_strand_fraction: float
    dssp_3_10_helix_fraction: float
    dssp_pi_helix_fraction: float
    dssp_turn_fraction: float
    dssp_bend_fraction: float
    dssp_unassigned_percent: float
    dssp_secondary_structure_percent: float
    dssp_pdb_model_count: int
    dssp_analyzed_model_count: int


@dataclass(frozen=True)
class SolutionNMRMonomerSecondaryModeledFirstModelRecord:
    entry_id: str
    year: int
    chain_id: str
    modeled_start_seq_id: int
    modeled_end_seq_id: int
    modeled_sequence_length: int
    secondary_structure_percent: float
    helix_fraction: float
    sheet_fraction: float
    dssp_alpha_helix_fraction: float
    dssp_isolated_beta_bridge_fraction: float
    dssp_beta_strand_fraction: float
    dssp_3_10_helix_fraction: float
    dssp_pi_helix_fraction: float
    dssp_turn_fraction: float
    dssp_bend_fraction: float
    dssp_unassigned_percent: float
    dssp_secondary_structure_percent: float


@dataclass(frozen=True)
class SolutionNMRMonomerSecondaryByModelRecord:
    entry_id: str
    year: int
    chain_id: str
    sequence_length: int
    deposited_model_count: int
    pdb_model_count: int
    model_index: int
    assigned_residue_count: int
    secondary_structure_percent: float
    helix_fraction: float
    sheet_fraction: float
    dssp_alpha_helix_fraction: float
    dssp_isolated_beta_bridge_fraction: float
    dssp_beta_strand_fraction: float
    dssp_3_10_helix_fraction: float
    dssp_pi_helix_fraction: float
    dssp_turn_fraction: float
    dssp_bend_fraction: float
    dssp_unassigned_percent: float
    dssp_secondary_structure_percent: float


@dataclass(frozen=True)
class SolutionNMRMonomerStrideRecord:
    entry_id: str
    year: int
    sequence_length: int
    secondary_structure_percent: float
    helix_fraction: float
    sheet_fraction: float
    deposited_model_count: int
    stride_alpha_helix_fraction: float
    stride_3_10_helix_fraction: float
    stride_pi_helix_fraction: float
    stride_beta_strand_fraction: float
    stride_isolated_beta_bridge_fraction: float
    stride_turn_fraction: float
    stride_coil_fraction: float
    stride_secondary_structure_percent: float
    stride_pdb_model_count: int
    stride_analyzed_model_count: int


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
class SolutionNMRMonomerCoreRegionRecord:
    entry_id: str
    year: int
    chain_id: str
    core_start_seq_id: int
    core_end_seq_id: int
    deposited_model_count: int


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
    group_id: str | None
    has_xray_homolog: bool


@dataclass(frozen=True)
class SolutionNMRMonomerXraySeedRecord:
    entry_id: str
    year: int
    chain_id: str
    core_start_seq_id: int
    core_end_seq_id: int
    group_id_95: str | None
    group_id_100: str | None


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
    xray_entry_id: str
    xray_chain_id: str
    xray_core_start_seq_id: int | None
    xray_core_end_seq_id: int | None
    xray_resolution_angstrom: float
    n_common_ca: int
    rmsd_ca_angstrom: float


def chunked(items: list[str], size: int) -> Iterator[list[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def extract_year(deposit_date: str | None) -> int | None:
    if not deposit_date:
        return None
    try:
        return int(deposit_date[:4])
    except (TypeError, ValueError):
        return None


def collect_batch_results(
    batches: list[list[str]],
    max_workers: int,
    fetch_fn: Callable[[list[str]], T],
    progress_label: str,
) -> list[T]:
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


def fetch_solution_nmr_entry_ids(client: "RCSBClient", log_label: str) -> list[str]:
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


def resolve_dssp_executable(explicit_value: str) -> str | None:
    explicit_path = explicit_value.strip()
    if explicit_path:
        path = Path(explicit_path).expanduser()
        if path.exists():
            return str(path)
        return None

    for cmd in ("mkdssp", "dssp"):
        resolved = shutil.which(cmd)
        if resolved:
            return resolved

    if LOCAL_DSSP_CANDIDATE.exists():
        return str(LOCAL_DSSP_CANDIDATE.resolve())

    return None


def resolve_stride_executable(explicit_value: str) -> str | None:
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
    path = cache_dir / f"{entry_id}.pdb"
    if path.exists() and path.stat().st_size > 0:
        return path

    url = f"https://files.rcsb.org/download/{entry_id}.pdb"
    last_error: Exception | None = None
    for attempt in range(1, config.retries + 1):
        try:
            response = session.get(url, timeout=config.timeout_seconds)
            response.raise_for_status()
            path.write_text(response.text, encoding="utf-8")
            return path
        except (requests.RequestException, OSError) as exc:
            last_error = exc
            wait_seconds = config.backoff_seconds * attempt
            if attempt < config.retries:
                time.sleep(wait_seconds)
    raise RuntimeError(f"Failed to download {entry_id}: {last_error}")


def write_csv_rows(
    output_path: Path,
    header: Sequence[str],
    rows: Iterable[Sequence[Any]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(rows)


def extract_model_pdb_texts(pdb_path: Path) -> list[str]:
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


def compute_dssp_state_coverages_for_chain(
    session: requests.Session,
    config: CollectorConfig,
    cache_dir: Path,
    entry_id: str,
    chain_id: str,
    sequence_length: int,
    dssp_executable: str,
) -> tuple[dict[str, float], int, int, list[tuple[int, int, dict[str, float]]]]:
    default_coverages = {state: -1.0 for state in DSSP_STATE_CODES}
    if sequence_length <= 0:
        return default_coverages, 0, 0, []

    try:
        pdb_path = download_pdb_if_needed(
            session=session,
            config=config,
            cache_dir=cache_dir,
            entry_id=entry_id,
        )
    except Exception:
        return default_coverages, 0, 0, []

    model_texts = extract_model_pdb_texts(pdb_path)
    if not model_texts:
        return default_coverages, 0, 0, []

    parser = PDBParser(QUIET=True)
    state_counts: Counter[str] = Counter()
    analyzed_model_count = 0
    per_model_coverages: list[tuple[int, int, dict[str, float]]] = []

    for model_index, model_text in enumerate(model_texts, start=1):
        model_state_coverages = {state: -1.0 for state in DSSP_STATE_CODES}
        assigned_residue_count = 0
        try:
            with tempfile.NamedTemporaryFile(
                "w", suffix=".pdb", encoding="utf-8", delete=True
            ) as handle:
                handle.write(model_text)
                handle.flush()

                structure = parser.get_structure(entry_id, handle.name)
                model = next(structure.get_models(), None)
                if model is None:
                    continue

                dssp_result = BioDSSP(
                    model,
                    handle.name,
                    dssp=dssp_executable,
                    file_type="PDB",
                )

                state_by_chain: dict[str, list[str]] = {}
                for key in dssp_result.keys():
                    chain_label = str(key[0]).strip()
                    state_raw = str(dssp_result[key][2]).strip() or "-"
                    state = state_raw if state_raw in DSSP_STATE_CODES else "-"
                    state_by_chain.setdefault(chain_label, []).append(state)

                states = state_by_chain.get(chain_id, [])
                if not states and len(state_by_chain) == 1:
                    states = next(iter(state_by_chain.values()))
                if not states:
                    per_model_coverages.append(
                        (model_index, assigned_residue_count, model_state_coverages)
                    )
                    continue

                analyzed_model_count += 1
                assigned_residue_count = len(states)
                model_state_counts = Counter(states)
                model_denominator = float(sequence_length)
                for state in DSSP_STATE_CODES:
                    model_state_coverages[state] = min(
                        1.0,
                        max(0.0, model_state_counts.get(state, 0) / model_denominator),
                    )
                state_counts.update(states)
                per_model_coverages.append(
                    (model_index, assigned_residue_count, model_state_coverages)
                )
        except Exception:
            per_model_coverages.append(
                (model_index, assigned_residue_count, model_state_coverages)
            )
            continue

    if analyzed_model_count <= 0:
        return default_coverages, len(model_texts), 0, per_model_coverages

    denominator = float(sequence_length * analyzed_model_count)
    coverages = {
        state: min(1.0, max(0.0, state_counts.get(state, 0) / denominator))
        for state in DSSP_STATE_CODES
    }
    return coverages, len(model_texts), analyzed_model_count, per_model_coverages


def compute_stride_state_coverages_for_chain(
    session: requests.Session,
    config: CollectorConfig,
    cache_dir: Path,
    entry_id: str,
    chain_id: str,
    sequence_length: int,
    stride_executable: str,
) -> tuple[dict[str, float], int, int]:
    default_coverages = {state: -1.0 for state in STRIDE_STATE_CODES}
    if sequence_length <= 0:
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

    model_texts = extract_model_pdb_texts(pdb_path)
    if not model_texts:
        return default_coverages, 0, 0

    state_counts: Counter[str] = Counter()
    analyzed_model_count = 0

    for model_text in model_texts:
        try:
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
                    continue

                state_by_chain: dict[str, list[str]] = {}
                for line in process.stdout.splitlines():
                    if not line.startswith("ASG"):
                        continue
                    parts = line.split()
                    if len(parts) < 6:
                        continue
                    chain_label = str(parts[2]).strip()
                    state_raw = str(parts[5]).strip()
                    if len(state_raw) != 1:
                        continue
                    state = state_raw if state_raw in STRIDE_STATE_CODES else "C"
                    state_by_chain.setdefault(chain_label, []).append(state)

                states = state_by_chain.get(chain_id, [])
                if not states and len(state_by_chain) == 1:
                    states = next(iter(state_by_chain.values()))
                if not states:
                    continue

                analyzed_model_count += 1
                state_counts.update(states)
        except Exception:
            continue

    if analyzed_model_count <= 0:
        return default_coverages, len(model_texts), 0

    denominator = float(sequence_length * analyzed_model_count)
    coverages = {
        state: min(1.0, max(0.0, state_counts.get(state, 0) / denominator))
        for state in STRIDE_STATE_CODES
    }
    return coverages, len(model_texts), analyzed_model_count


def compute_dssp_state_coverages_for_chain_modeled_first_model(
    session: requests.Session,
    config: CollectorConfig,
    cache_dir: Path,
    entry_id: str,
    chain_id: str,
    modeled_sequence_length: int,
    modeled_auth_seq_ids: set[int],
    dssp_executable: str,
) -> tuple[dict[str, float], int, int]:
    default_coverages = {state: -1.0 for state in DSSP_STATE_CODES}
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

    model_texts = extract_model_pdb_texts(pdb_path)
    if not model_texts:
        return default_coverages, 0, 0

    parser = PDBParser(QUIET=True)
    try:
        with tempfile.NamedTemporaryFile(
            "w", suffix=".pdb", encoding="utf-8", delete=True
        ) as handle:
            handle.write(model_texts[0])
            handle.flush()

            structure = parser.get_structure(entry_id, handle.name)
            model = next(structure.get_models(), None)
            if model is None:
                return default_coverages, len(model_texts), 0

            dssp_result = BioDSSP(
                model,
                handle.name,
                dssp=dssp_executable,
                file_type="PDB",
            )

            state_by_chain: dict[str, dict[int, str]] = {}
            for key in dssp_result.keys():
                chain_label = str(key[0]).strip()
                residue_key = key[1]
                try:
                    auth_seq_id = int(residue_key[1])
                except (TypeError, ValueError, IndexError):
                    continue
                state_raw = str(dssp_result[key][2]).strip() or "-"
                state = state_raw if state_raw in DSSP_STATE_CODES else "-"
                chain_states = state_by_chain.setdefault(chain_label, {})
                chain_states.setdefault(auth_seq_id, state)

            chain_states = state_by_chain.get(chain_id)
            if not chain_states and len(state_by_chain) == 1:
                chain_states = next(iter(state_by_chain.values()))
            if not chain_states:
                return default_coverages, len(model_texts), 0

            filtered_states = [
                chain_states.get(auth_seq_id, "-")
                for auth_seq_id in sorted(modeled_auth_seq_ids)
            ]
            missing_count = max(0, modeled_sequence_length - len(filtered_states))
            if missing_count > 0:
                filtered_states.extend(["-"] * missing_count)
            if not filtered_states:
                return default_coverages, len(model_texts), 0

            state_counts = Counter(filtered_states)
            denominator = float(modeled_sequence_length)
            coverages = {
                state: min(1.0, max(0.0, state_counts.get(state, 0) / denominator))
                for state in DSSP_STATE_CODES
            }
            return coverages, len(model_texts), 1
    except Exception:
        return default_coverages, len(model_texts), 0


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

    model_texts = extract_model_pdb_texts(pdb_path)
    if not model_texts:
        return default_coverages, 0, 0

    try:
        with tempfile.NamedTemporaryFile(
            "w", suffix=".pdb", encoding="utf-8", delete=True
        ) as handle:
            handle.write(model_texts[0])
            handle.flush()

            process = subprocess.run(
                [stride_executable, handle.name],
                check=False,
                capture_output=True,
                text=True,
            )
            if process.returncode != 0:
                return default_coverages, len(model_texts), 0

            state_by_chain: dict[str, dict[int, str]] = {}
            for line in process.stdout.splitlines():
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

            chain_states = state_by_chain.get(chain_id)
            if not chain_states and len(state_by_chain) == 1:
                chain_states = next(iter(state_by_chain.values()))
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


def parse_models_ca_coords(
    pdb_path: Path,
    chain_id: str,
    start_seq_id: int | None = None,
    end_seq_id: int | None = None,
) -> list[dict[int, np.ndarray]]:
    model_maps, _ = parse_models_ca_coords_with_stats(
        pdb_path=pdb_path,
        chain_id=chain_id,
        start_seq_id=start_seq_id,
        end_seq_id=end_seq_id,
    )
    return model_maps


def _alt_loc_tiebreak_key(alt_loc: str) -> tuple[int, str]:
    # Prefer blank altLoc, then A, then 1; keep deterministic order for others.
    if alt_loc == "":
        return (0, "")
    if alt_loc == "A":
        return (1, "")
    if alt_loc == "1":
        return (2, "")
    return (3, alt_loc)


def _insertion_code_tiebreak_key(insertion_code: str) -> tuple[int, str]:
    # Prefer residue numbers without insertion codes (e.g., 102 over 102A).
    if insertion_code == "":
        return (0, "")
    return (1, insertion_code)


def _parse_pdb_occupancy(line: str) -> float:
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
    # Select one CA per residue by max occupancy (altLoc-aware) and keep raw
    # per-residue counts so callers can report how many CA atoms were present
    # before altLoc collapsing.
    models: list[dict[int, np.ndarray]] = []
    raw_ca_counts_per_model: list[dict[int, int]] = []
    current_candidates: dict[int, tuple[str, float, str, np.ndarray]] = {}
    current_raw_counts: Counter[int] = Counter()
    has_model_records = False
    in_model = False

    def finalize_model() -> None:
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

            current_raw_counts[resid] += 1

            insertion_code = line[26].strip()
            alt_loc = line[16].strip()
            occupancy = _parse_pdb_occupancy(line)
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
    return float(mda_rmsd(a, b, center=True, superposition=True))


def _normalize_polymer_sequence(
    raw_sequence: Any, expected_length: int | None = None
) -> str | None:
    if not isinstance(raw_sequence, str):
        return None
    compact = "".join(raw_sequence.split()).upper()
    if not compact:
        return None
    if (
        expected_length is not None
        and expected_length > 0
        and len(compact) != expected_length
    ):
        return None
    return compact


def _seq_type_for_polymer(polymer_type: str, sequence: str | None = None) -> str | None:
    mapping = {
        "Protein": "protein",
        "DNA": "DNA",
        "RNA": "RNA",
    }
    if polymer_type in mapping:
        return mapping[polymer_type]
    if polymer_type != "NA-hybrid":
        return None
    seq = (sequence or "").upper()
    has_u = "U" in seq
    has_t = "T" in seq
    if has_u and not has_t:
        return "RNA"
    return "DNA"


def _modeled_sequence_from_instance_features(
    sequence: str, instance_features: list[dict[str, Any] | None]
) -> str:
    if not sequence:
        return sequence
    modeled_mask = [True] * len(sequence)
    for feature in instance_features:
        if not feature:
            continue
        feature_type = str(feature.get("type") or "")
        if feature_type not in UNMODELED_INSTANCE_FEATURE_TYPES:
            continue
        for pos in feature.get("feature_positions") or []:
            if not pos:
                continue
            beg = pos.get("beg_seq_id")
            end = pos.get("end_seq_id")
            if beg is None:
                continue
            try:
                beg_i = int(beg)
                end_i = int(end) if end is not None else beg_i
            except (TypeError, ValueError):
                continue
            start = max(1, min(beg_i, end_i))
            stop = min(len(sequence), max(beg_i, end_i))
            if start > stop:
                continue
            for idx in range(start - 1, stop):
                modeled_mask[idx] = False
    return "".join(
        residue for residue, keep in zip(sequence, modeled_mask, strict=False) if keep
    )


def _modeled_label_seq_ids_from_instance_features(
    sequence_length: int, instance_features: list[dict[str, Any] | None]
) -> set[int]:
    if sequence_length <= 0:
        return set()
    modeled_seq_ids = set(range(1, sequence_length + 1))
    for feature in instance_features:
        if not feature:
            continue
        feature_type = str(feature.get("type") or "")
        if feature_type not in UNMODELED_INSTANCE_FEATURE_TYPES:
            continue
        for pos in feature.get("feature_positions") or []:
            if not pos:
                continue
            beg = pos.get("beg_seq_id")
            end = pos.get("end_seq_id")
            if beg is None:
                continue
            try:
                beg_i = int(beg)
                end_i = int(end) if end is not None else beg_i
            except (TypeError, ValueError):
                continue
            start = max(1, min(beg_i, end_i))
            stop = min(sequence_length, max(beg_i, end_i))
            if start > stop:
                continue
            for seq_id in range(start, stop + 1):
                modeled_seq_ids.discard(seq_id)
    return modeled_seq_ids


def _map_label_seq_ids_to_auth_seq_ids(
    label_seq_ids: set[int], auth_mapping_raw: Any
) -> set[int]:
    if not label_seq_ids:
        return set()

    mapped_auth_seq_ids: set[int] = set()
    if isinstance(auth_mapping_raw, list) and auth_mapping_raw:
        for label_seq_id in sorted(label_seq_ids):
            if label_seq_id <= 0 or label_seq_id > len(auth_mapping_raw):
                continue
            auth_seq_raw = auth_mapping_raw[label_seq_id - 1]
            try:
                mapped_auth_seq_ids.add(int(str(auth_seq_raw).strip()))
            except (TypeError, ValueError):
                continue
    if mapped_auth_seq_ids:
        return mapped_auth_seq_ids

    # Fallback for entries without mapping: treat label ids as auth ids.
    return {int(seq_id) for seq_id in label_seq_ids}


def _sequence_weight_kda(sequence: str, seq_type: str) -> float | None:
    if sequence_molecular_weight is None:
        return None
    sequence_for_weight = sequence
    if seq_type == "protein":
        allowed = set("ACDEFGHIKLMNPQRSTVWY")
        sequence_for_weight = "".join(aa for aa in sequence_for_weight if aa in allowed)
    elif seq_type == "DNA":
        allowed = set("ACGT")
        sequence_for_weight = "".join(nt for nt in sequence_for_weight if nt in allowed)
    elif seq_type == "RNA":
        allowed = set("ACGU")
        sequence_for_weight = "".join(nt for nt in sequence_for_weight if nt in allowed)
    if not sequence_for_weight:
        return 0.0
    try:
        daltons = float(
            sequence_molecular_weight(sequence_for_weight, seq_type=seq_type)
        )
    except Exception:
        return None
    return daltons / 1000.0


MEMBRANE_ANNOTATION_TYPES: tuple[str, ...] = ("OPM", "PDBTM", "MemProtMD", "mpstruc")


class RCSBClient:
    def __init__(self, config: CollectorConfig) -> None:
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "pdb-extensible-collector/1.0"})

    @staticmethod
    def _normalize_similarity_cutoff(raw_cutoff: Any) -> int | None:
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

    @staticmethod
    def _extract_modeled_residue_sets_for_instance(
        sequence_length: int,
        instance: dict[str, Any],
    ) -> tuple[set[int], set[int]]:
        instance_features = instance.get("rcsb_polymer_instance_feature") or []
        modeled_label_seq_ids = _modeled_label_seq_ids_from_instance_features(
            sequence_length=sequence_length,
            instance_features=instance_features,
        )
        identifiers = (
            instance.get("rcsb_polymer_entity_instance_container_identifiers") or {}
        )
        auth_mapping_raw = identifiers.get("auth_to_entity_poly_seq_mapping") or []
        modeled_auth_seq_ids = _map_label_seq_ids_to_auth_seq_ids(
            label_seq_ids=modeled_label_seq_ids,
            auth_mapping_raw=auth_mapping_raw,
        )
        return modeled_label_seq_ids, modeled_auth_seq_ids

    @staticmethod
    def _extract_secondary_label_ranges(
        instance_features: list[dict[str, Any] | None],
    ) -> list[tuple[int, int]]:
        sec_ranges: list[tuple[int, int]] = []
        for feature in instance_features:
            if not feature:
                continue
            feature_type = feature.get("type")
            if feature_type not in {"HELIX_P", "SHEET"}:
                continue
            for pos in feature.get("feature_positions") or []:
                if not pos:
                    continue
                beg = pos.get("beg_seq_id")
                end = pos.get("end_seq_id")
                if beg is None:
                    continue
                try:
                    beg_i = int(beg)
                    end_i = int(end) if end is not None else beg_i
                except (TypeError, ValueError):
                    continue
                sec_ranges.append((beg_i, end_i))
        return sec_ranges

    @staticmethod
    def _extract_secondary_core_range(
        polymer_entity: dict[str, Any],
    ) -> tuple[int, int] | None:
        instances = polymer_entity.get("polymer_entity_instances") or []
        if len(instances) != 1:
            return None

        instance = instances[0] or {}
        features = instance.get("rcsb_polymer_instance_feature") or []
        sec_ranges = RCSBClient._extract_secondary_label_ranges(features)

        if not sec_ranges:
            return None

        instance_identifiers = (
            instance.get("rcsb_polymer_entity_instance_container_identifiers") or {}
        )
        auth_mapping_raw = (
            instance_identifiers.get("auth_to_entity_poly_seq_mapping") or []
        )

        mapped_auth_positions: list[int] = []
        if isinstance(auth_mapping_raw, list) and auth_mapping_raw:
            for beg_i, end_i in sec_ranges:
                start = min(beg_i, end_i)
                stop = max(beg_i, end_i)
                for label_seq_id in range(start, stop + 1):
                    if label_seq_id <= 0 or label_seq_id > len(auth_mapping_raw):
                        continue
                    auth_seq_id_raw = auth_mapping_raw[label_seq_id - 1]
                    try:
                        mapped_auth_positions.append(int(str(auth_seq_id_raw).strip()))
                    except (TypeError, ValueError):
                        continue

        if mapped_auth_positions:
            core_start = min(mapped_auth_positions)
            core_end = max(mapped_auth_positions)
            if core_end >= core_start:
                return core_start, core_end

        core_start = min(beg for beg, _ in sec_ranges)
        core_end = max(end for _, end in sec_ranges)
        if core_end < core_start:
            return None
        return core_start, core_end

    def _fetch_paginated_identifiers(
        self,
        query: dict[str, Any],
        return_type: str,
        progress_label: str | None = None,
    ) -> list[str]:
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
                for item in data.get("result_set", [])
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
        entries = data.get("data", {}).get("entries", [])

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
        entries = data.get("data", {}).get("entries", [])
        return [
            entry.get("rcsb_accession_info", {}).get("deposit_date")
            for entry in entries
            if entry and entry.get("rcsb_accession_info", {}).get("deposit_date")
        ]

    def fetch_entry_resolution_for_ids(self, entry_ids: list[str]) -> dict[str, float]:
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
        entries = data.get("data", {}).get("entries", [])
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
        entities = data.get("data", {}).get("polymer_entities", [])

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

    def fetch_sequence_identity_group_ids_for_polymer_entity_ids(
        self, entity_ids: list[str], similarity_cutoff: int
    ) -> set[str]:
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
        entities = data.get("data", {}).get("polymer_entities", [])
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

    def fetch_solution_nmr_weight_records_for_ids(
        self, entry_ids: list[str]
    ) -> list[SolutionNMRWeightRecord]:
        query = """
        query($ids:[String!]!) {
          entries(entry_ids:$ids) {
            rcsb_id
            rcsb_accession_info {
              deposit_date
            }
            rcsb_entry_info {
              molecular_weight
              polymer_molecular_weight_maximum
            }
            polymer_entities {
              entity_poly {
                rcsb_entity_polymer_type
                pdbx_strand_id
                rcsb_sample_sequence_length
                pdbx_seq_one_letter_code_can
              }
              rcsb_polymer_entity {
                formula_weight
              }
              polymer_entity_instances {
                rcsb_id
                rcsb_polymer_instance_feature {
                  type
                  feature_positions {
                    beg_seq_id
                    end_seq_id
                  }
                }
              }
            }
          }
        }
        """
        payload = {"query": query, "variables": {"ids": entry_ids}}
        data = self._post_json(self.config.graphql_url, payload)
        entries = data.get("data", {}).get("entries", [])
        allowed_polymer_types = {"Protein", "RNA", "DNA", "NA-hybrid"}

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
                entry_mw_kda = float(entry_mw_raw) if entry_mw_raw is not None else None
            except (TypeError, ValueError):
                entry_mw_kda = None
            polymer_mw_max_raw = (entry.get("rcsb_entry_info") or {}).get(
                "polymer_molecular_weight_maximum"
            )
            try:
                polymer_mw_max_kda = (
                    float(polymer_mw_max_raw)
                    if polymer_mw_max_raw is not None
                    else None
                )
            except (TypeError, ValueError):
                polymer_mw_max_kda = None
            total_weight_kda = 0.0
            modeled_weight_total_kda = 0.0
            for polymer_entity in entry.get("polymer_entities") or []:
                if not polymer_entity:
                    continue
                polymer_type = polymer_entity.get("entity_poly", {}).get(
                    "rcsb_entity_polymer_type"
                )
                if polymer_type not in allowed_polymer_types:
                    continue
                entity_weight = polymer_entity.get("rcsb_polymer_entity", {}).get(
                    "formula_weight"
                )
                if entity_weight is not None:
                    entity_weight_f = float(entity_weight)
                    instances = polymer_entity.get("polymer_entity_instances") or []
                    instance_count = len(
                        [
                            instance
                            for instance in instances
                            if instance and instance.get("rcsb_id")
                        ]
                    )
                    if instance_count <= 0:
                        strand_ids = str(
                            polymer_entity.get("entity_poly", {}).get("pdbx_strand_id")
                            or ""
                        )
                        chain_ids = {
                            chain_id.strip()
                            for chain_id in strand_ids.split(",")
                            if chain_id.strip()
                        }
                        instance_count = len(chain_ids) if chain_ids else 1

                    total_weight_kda += entity_weight_f * instance_count

                    sequence_length_raw = polymer_entity.get("entity_poly", {}).get(
                        "rcsb_sample_sequence_length"
                    )
                    try:
                        sequence_length = int(sequence_length_raw)
                    except (TypeError, ValueError):
                        sequence_length = 0

                    if sequence_length > 0:
                        raw_sequence = polymer_entity.get("entity_poly", {}).get(
                            "pdbx_seq_one_letter_code_can"
                        )
                        normalized_sequence = _normalize_polymer_sequence(
                            raw_sequence=raw_sequence,
                            expected_length=sequence_length,
                        )
                        seq_type = _seq_type_for_polymer(
                            str(polymer_type), normalized_sequence
                        )
                        if normalized_sequence is None or seq_type is None:
                            continue

                        valid_instances = [
                            instance
                            for instance in instances
                            if instance and instance.get("rcsb_id")
                        ]
                        if valid_instances:
                            for instance in valid_instances:
                                modeled_sequence = (
                                    _modeled_sequence_from_instance_features(
                                        sequence=normalized_sequence,
                                        instance_features=instance.get(
                                            "rcsb_polymer_instance_feature"
                                        )
                                        or [],
                                    )
                                )
                                modeled_sequence_weight = _sequence_weight_kda(
                                    sequence=modeled_sequence,
                                    seq_type=seq_type,
                                )
                                if modeled_sequence_weight is not None:
                                    modeled_weight_total_kda += modeled_sequence_weight
                        else:
                            full_sequence_weight = _sequence_weight_kda(
                                sequence=normalized_sequence,
                                seq_type=seq_type,
                            )
                            if full_sequence_weight is not None:
                                modeled_weight_total_kda += (
                                    full_sequence_weight * instance_count
                                )

            if total_weight_kda <= 0.0:
                total_weight_kda = entry_mw_kda

            if modeled_weight_total_kda <= 0.0:
                modeled_weight_total_kda = total_weight_kda

            records.append(
                SolutionNMRWeightRecord(
                    entry_id=entry_id,
                    year=year,
                    molecular_weight_kda=total_weight_kda,
                    rcsb_entry_molecular_weight_kda=entry_mw_kda,
                    polymer_molecular_weight_maximum_kda=polymer_mw_max_kda,
                    modeled_molecular_weight_kda=modeled_weight_total_kda,
                )
            )
        return records

    def iter_solution_nmr_monomer_secondary_records_for_ids(
        self,
        entry_ids: list[str],
        dssp_executable: str,
        pdb_cache_dir: Path,
        skip_entry_ids: set[str] | None = None,
    ) -> Iterator[
        tuple[
            SolutionNMRMonomerSecondaryRecord,
            list[SolutionNMRMonomerSecondaryByModelRecord],
        ]
    ]:
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
                rcsb_sample_sequence_length
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
        entries = data.get("data", {}).get("entries", [])
        normalized_skip_ids = (
            {entry_id.strip().upper() for entry_id in skip_entry_ids}
            if skip_entry_ids
            else set()
        )
        entries_to_process: list[dict[str, Any] | None] = []
        for entry in entries:
            raw_entry_id = str((entry or {}).get("rcsb_id") or "").strip().upper()
            if raw_entry_id and raw_entry_id in normalized_skip_ids:
                continue
            entries_to_process.append(entry)

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_map = {
                executor.submit(
                    self._compute_solution_nmr_monomer_secondary_for_entry,
                    entry=entry,
                    dssp_executable=dssp_executable,
                    pdb_cache_dir=pdb_cache_dir,
                ): idx
                for idx, entry in enumerate(entries_to_process, start=1)
            }
            for future in as_completed(future_map):
                secondary_record, by_model_records = future.result()
                if secondary_record is None:
                    continue
                yield secondary_record, by_model_records

    def _compute_solution_nmr_monomer_secondary_for_entry(
        self,
        entry: dict[str, Any] | None,
        dssp_executable: str,
        pdb_cache_dir: Path,
    ) -> tuple[
        SolutionNMRMonomerSecondaryRecord | None,
        list[SolutionNMRMonomerSecondaryByModelRecord],
    ]:
        if not entry:
            return None, []
        context = self._extract_solution_nmr_monomer_context(entry)
        if context is None:
            return None, []
        entry_id, year, _model_count, polymer_entity, chain_id = context
        entity_poly = polymer_entity.get("entity_poly") or {}

        sequence_length = entity_poly.get("rcsb_sample_sequence_length")
        if sequence_length is None or int(sequence_length) <= 0:
            return None, []
        sequence_length = int(sequence_length)

        instances = polymer_entity.get("polymer_entity_instances") or []
        if len(instances) != 1:
            return None, []
        feature_summary = instances[0].get("rcsb_polymer_instance_feature_summary") or []
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
        (
            dssp_coverages,
            dssp_pdb_model_count,
            dssp_analyzed_model_count,
            dssp_per_model_coverages,
        ) = compute_dssp_state_coverages_for_chain(
            session=self.session,
            config=self.config,
            cache_dir=pdb_cache_dir,
            entry_id=entry_id,
            chain_id=chain_id,
            sequence_length=sequence_length,
            dssp_executable=dssp_executable,
        )
        dssp_unassigned_percent = dssp_coverages["-"]
        dssp_secondary_structure_percent = (1.0 - dssp_unassigned_percent) * 100.0

        secondary_record = SolutionNMRMonomerSecondaryRecord(
            entry_id=entry_id,
            year=year,
            sequence_length=sequence_length,
            secondary_structure_percent=secondary_fraction * 100.0,
            helix_fraction=helix_fraction,
            sheet_fraction=sheet_fraction,
            deposited_model_count=model_count,
            dssp_alpha_helix_fraction=dssp_coverages["H"],
            dssp_isolated_beta_bridge_fraction=dssp_coverages["B"],
            dssp_beta_strand_fraction=dssp_coverages["E"],
            dssp_3_10_helix_fraction=dssp_coverages["G"],
            dssp_pi_helix_fraction=dssp_coverages["I"],
            dssp_turn_fraction=dssp_coverages["T"],
            dssp_bend_fraction=dssp_coverages["S"],
            dssp_unassigned_percent=dssp_unassigned_percent,
            dssp_secondary_structure_percent=dssp_secondary_structure_percent,
            dssp_pdb_model_count=dssp_pdb_model_count,
            dssp_analyzed_model_count=dssp_analyzed_model_count,
        )

        by_model_records: list[SolutionNMRMonomerSecondaryByModelRecord] = []
        for model_index, assigned_residue_count, model_coverages in dssp_per_model_coverages:
            model_unassigned_percent = model_coverages["-"]
            by_model_records.append(
                SolutionNMRMonomerSecondaryByModelRecord(
                    entry_id=entry_id,
                    year=year,
                    chain_id=chain_id,
                    sequence_length=sequence_length,
                    deposited_model_count=model_count,
                    pdb_model_count=dssp_pdb_model_count,
                    model_index=model_index,
                    assigned_residue_count=assigned_residue_count,
                    secondary_structure_percent=secondary_fraction * 100.0,
                    helix_fraction=helix_fraction,
                    sheet_fraction=sheet_fraction,
                    dssp_alpha_helix_fraction=model_coverages["H"],
                    dssp_isolated_beta_bridge_fraction=model_coverages["B"],
                    dssp_beta_strand_fraction=model_coverages["E"],
                    dssp_3_10_helix_fraction=model_coverages["G"],
                    dssp_pi_helix_fraction=model_coverages["I"],
                    dssp_turn_fraction=model_coverages["T"],
                    dssp_bend_fraction=model_coverages["S"],
                    dssp_unassigned_percent=model_unassigned_percent,
                    dssp_secondary_structure_percent=(1.0 - model_unassigned_percent)
                    * 100.0,
                )
            )

        return secondary_record, by_model_records

    def fetch_solution_nmr_monomer_secondary_records_for_ids(
        self,
        entry_ids: list[str],
        dssp_executable: str,
        pdb_cache_dir: Path,
    ) -> tuple[
        list[SolutionNMRMonomerSecondaryRecord],
        list[SolutionNMRMonomerSecondaryByModelRecord],
    ]:
        records: list[SolutionNMRMonomerSecondaryRecord] = []
        by_model_records: list[SolutionNMRMonomerSecondaryByModelRecord] = []
        for (
            secondary_record,
            secondary_by_model_records,
        ) in self.iter_solution_nmr_monomer_secondary_records_for_ids(
            entry_ids=entry_ids,
            dssp_executable=dssp_executable,
            pdb_cache_dir=pdb_cache_dir,
        ):
            records.append(secondary_record)
            by_model_records.extend(secondary_by_model_records)
        return records, by_model_records

    def iter_solution_nmr_monomer_secondary_modeled_first_model_records_for_ids(
        self,
        entry_ids: list[str],
        dssp_executable: str,
        pdb_cache_dir: Path,
    ) -> Iterator[SolutionNMRMonomerSecondaryModeledFirstModelRecord]:
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
                rcsb_sample_sequence_length
              }
              polymer_entity_instances {
                rcsb_id
                rcsb_polymer_entity_instance_container_identifiers {
                  auth_to_entity_poly_seq_mapping
                }
                rcsb_polymer_instance_feature_summary {
                  type
                  coverage
                }
                rcsb_polymer_instance_feature {
                  type
                  feature_positions {
                    beg_seq_id
                    end_seq_id
                  }
                }
              }
            }
          }
        }
        """
        payload = {"query": query, "variables": {"ids": entry_ids}}
        data = self._post_json(self.config.graphql_url, payload)
        entries = data.get("data", {}).get("entries", [])

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_map = {
                executor.submit(
                    self._compute_solution_nmr_monomer_secondary_modeled_first_model_for_entry,
                    entry=entry,
                    dssp_executable=dssp_executable,
                    pdb_cache_dir=pdb_cache_dir,
                ): idx
                for idx, entry in enumerate(entries, start=1)
            }
            for future in as_completed(future_map):
                record = future.result()
                if record is None:
                    continue
                yield record

    def _compute_solution_nmr_monomer_secondary_modeled_first_model_for_entry(
        self,
        entry: dict[str, Any] | None,
        dssp_executable: str,
        pdb_cache_dir: Path,
    ) -> SolutionNMRMonomerSecondaryModeledFirstModelRecord | None:
        if not entry:
            return None
        context = self._extract_solution_nmr_monomer_context(entry)
        if context is None:
            return None
        entry_id, year, _model_count, polymer_entity, chain_id = context
        entity_poly = polymer_entity.get("entity_poly") or {}

        sequence_length_raw = entity_poly.get("rcsb_sample_sequence_length")
        if sequence_length_raw is None or int(sequence_length_raw) <= 0:
            return None
        sequence_length = int(sequence_length_raw)

        instances = polymer_entity.get("polymer_entity_instances") or []
        if len(instances) != 1:
            return None
        instance = instances[0] or {}
        (
            modeled_label_seq_ids,
            modeled_auth_seq_ids,
        ) = self._extract_modeled_residue_sets_for_instance(
            sequence_length=sequence_length,
            instance=instance,
        )
        modeled_sequence_length = len(modeled_label_seq_ids)
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
        dssp_coverages, _, _ = compute_dssp_state_coverages_for_chain_modeled_first_model(
            session=self.session,
            config=self.config,
            cache_dir=pdb_cache_dir,
            entry_id=entry_id,
            chain_id=chain_id,
            modeled_sequence_length=modeled_sequence_length,
            modeled_auth_seq_ids=modeled_auth_seq_ids,
            dssp_executable=dssp_executable,
        )
        dssp_unassigned_percent = dssp_coverages["-"]
        dssp_secondary_structure_percent = (1.0 - dssp_unassigned_percent) * 100.0

        return SolutionNMRMonomerSecondaryModeledFirstModelRecord(
            entry_id=entry_id,
            year=year,
            chain_id=chain_id,
            modeled_start_seq_id=modeled_start_seq_id,
            modeled_end_seq_id=modeled_end_seq_id,
            modeled_sequence_length=modeled_sequence_length,
            secondary_structure_percent=secondary_fraction * 100.0,
            helix_fraction=helix_fraction,
            sheet_fraction=sheet_fraction,
            dssp_alpha_helix_fraction=dssp_coverages["H"],
            dssp_isolated_beta_bridge_fraction=dssp_coverages["B"],
            dssp_beta_strand_fraction=dssp_coverages["E"],
            dssp_3_10_helix_fraction=dssp_coverages["G"],
            dssp_pi_helix_fraction=dssp_coverages["I"],
            dssp_turn_fraction=dssp_coverages["T"],
            dssp_bend_fraction=dssp_coverages["S"],
            dssp_unassigned_percent=dssp_unassigned_percent,
            dssp_secondary_structure_percent=dssp_secondary_structure_percent,
        )

    def fetch_solution_nmr_monomer_secondary_modeled_first_model_records_for_ids(
        self,
        entry_ids: list[str],
        dssp_executable: str,
        pdb_cache_dir: Path,
    ) -> list[SolutionNMRMonomerSecondaryModeledFirstModelRecord]:
        return list(
            self.iter_solution_nmr_monomer_secondary_modeled_first_model_records_for_ids(
                entry_ids=entry_ids,
                dssp_executable=dssp_executable,
                pdb_cache_dir=pdb_cache_dir,
            )
        )

    def iter_solution_nmr_monomer_stride_records_for_ids(
        self,
        entry_ids: list[str],
        stride_executable: str,
        pdb_cache_dir: Path,
    ) -> Iterator[SolutionNMRMonomerStrideRecord]:
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
                rcsb_sample_sequence_length
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
        entries = data.get("data", {}).get("entries", [])

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_map = {
                executor.submit(
                    self._compute_solution_nmr_monomer_stride_for_entry,
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

    def _compute_solution_nmr_monomer_stride_for_entry(
        self,
        entry: dict[str, Any] | None,
        stride_executable: str,
        pdb_cache_dir: Path,
    ) -> SolutionNMRMonomerStrideRecord | None:
        if not entry:
            return None
        context = self._extract_solution_nmr_monomer_context(entry)
        if context is None:
            return None
        entry_id, year, model_count, polymer_entity, chain_id = context
        entity_poly = polymer_entity.get("entity_poly") or {}

        sequence_length_raw = entity_poly.get("rcsb_sample_sequence_length")
        if sequence_length_raw is None or int(sequence_length_raw) <= 0:
            return None
        sequence_length = int(sequence_length_raw)

        instances = polymer_entity.get("polymer_entity_instances") or []
        if len(instances) != 1:
            return None
        feature_summary = instances[0].get("rcsb_polymer_instance_feature_summary") or []
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
        (
            stride_coverages,
            stride_pdb_model_count,
            stride_analyzed_model_count,
        ) = compute_stride_state_coverages_for_chain(
            session=self.session,
            config=self.config,
            cache_dir=pdb_cache_dir,
            entry_id=entry_id,
            chain_id=chain_id,
            sequence_length=sequence_length,
            stride_executable=stride_executable,
        )
        stride_coil_fraction = stride_coverages["C"]
        stride_secondary_percent = (1.0 - stride_coil_fraction) * 100.0

        return SolutionNMRMonomerStrideRecord(
            entry_id=entry_id,
            year=year,
            sequence_length=sequence_length,
            secondary_structure_percent=secondary_fraction * 100.0,
            helix_fraction=helix_fraction,
            sheet_fraction=sheet_fraction,
            deposited_model_count=model_count,
            stride_alpha_helix_fraction=stride_coverages["H"],
            stride_3_10_helix_fraction=stride_coverages["G"],
            stride_pi_helix_fraction=stride_coverages["I"],
            stride_beta_strand_fraction=stride_coverages["E"],
            stride_isolated_beta_bridge_fraction=stride_coverages["B"],
            stride_turn_fraction=stride_coverages["T"],
            stride_coil_fraction=stride_coil_fraction,
            stride_secondary_structure_percent=stride_secondary_percent,
            stride_pdb_model_count=stride_pdb_model_count,
            stride_analyzed_model_count=stride_analyzed_model_count,
        )

    def fetch_solution_nmr_monomer_stride_records_for_ids(
        self,
        entry_ids: list[str],
        stride_executable: str,
        pdb_cache_dir: Path,
    ) -> list[SolutionNMRMonomerStrideRecord]:
        return list(
            self.iter_solution_nmr_monomer_stride_records_for_ids(
                entry_ids=entry_ids,
                stride_executable=stride_executable,
                pdb_cache_dir=pdb_cache_dir,
            )
        )

    def iter_solution_nmr_monomer_stride_modeled_first_model_records_for_ids(
        self,
        entry_ids: list[str],
        stride_executable: str,
        pdb_cache_dir: Path,
    ) -> Iterator[SolutionNMRMonomerStrideModeledFirstModelRecord]:
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
                rcsb_sample_sequence_length
              }
              polymer_entity_instances {
                rcsb_id
                rcsb_polymer_entity_instance_container_identifiers {
                  auth_to_entity_poly_seq_mapping
                }
                rcsb_polymer_instance_feature_summary {
                  type
                  coverage
                }
                rcsb_polymer_instance_feature {
                  type
                  feature_positions {
                    beg_seq_id
                    end_seq_id
                  }
                }
              }
            }
          }
        }
        """
        payload = {"query": query, "variables": {"ids": entry_ids}}
        data = self._post_json(self.config.graphql_url, payload)
        entries = data.get("data", {}).get("entries", [])

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
        if not entry:
            return None
        context = self._extract_solution_nmr_monomer_context(entry)
        if context is None:
            return None
        entry_id, year, model_count, polymer_entity, chain_id = context
        entity_poly = polymer_entity.get("entity_poly") or {}

        sequence_length_raw = entity_poly.get("rcsb_sample_sequence_length")
        if sequence_length_raw is None or int(sequence_length_raw) <= 0:
            return None
        sequence_length = int(sequence_length_raw)

        instances = polymer_entity.get("polymer_entity_instances") or []
        if len(instances) != 1:
            return None
        instance = instances[0] or {}
        (
            modeled_label_seq_ids,
            modeled_auth_seq_ids,
        ) = self._extract_modeled_residue_sets_for_instance(
            sequence_length=sequence_length,
            instance=instance,
        )
        modeled_sequence_length = len(modeled_label_seq_ids)
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
        return list(
            self.iter_solution_nmr_monomer_stride_modeled_first_model_records_for_ids(
                entry_ids=entry_ids,
                stride_executable=stride_executable,
                pdb_cache_dir=pdb_cache_dir,
            )
        )

    def fetch_solution_nmr_monomer_core_region_records_for_ids(
        self, entry_ids: list[str]
    ) -> list[SolutionNMRMonomerCoreRegionRecord]:
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
                rcsb_polymer_entity_instance_container_identifiers {
                  auth_to_entity_poly_seq_mapping
                }
                rcsb_polymer_instance_feature {
                  type
                  feature_positions {
                    beg_seq_id
                    end_seq_id
                  }
                }
              }
            }
          }
        }
        """
        payload = {"query": query, "variables": {"ids": entry_ids}}
        data = self._post_json(self.config.graphql_url, payload)
        entries = data.get("data", {}).get("entries", [])
        records: list[SolutionNMRMonomerCoreRegionRecord] = []

        for entry in entries:
            if not entry:
                continue
            context = self._extract_solution_nmr_monomer_context(entry)
            if context is None:
                continue
            entry_id, year, model_count, polymer_entity, chain_id = context
            core_range = self._extract_secondary_core_range(polymer_entity)
            if core_range is None:
                continue
            core_start, core_end = core_range

            records.append(
                SolutionNMRMonomerCoreRegionRecord(
                    entry_id=entry_id,
                    year=year,
                    chain_id=chain_id,
                    core_start_seq_id=core_start,
                    core_end_seq_id=core_end,
                    deposited_model_count=model_count,
                )
            )
        return records

    def fetch_solution_nmr_monomer_quality_records_for_ids(
        self, entry_ids: list[str]
    ) -> list[SolutionNMRMonomerQualityRecord]:
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
        entries = data.get("data", {}).get("entries", [])
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

    def fetch_solution_nmr_monomer_xray_group_records_for_ids(
        self, entry_ids: list[str]
    ) -> list[tuple[str, int, str | None, str | None]]:
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
              rcsb_polymer_entity_group_membership {
                aggregation_method
                similarity_cutoff
                group_id
              }
            }
          }
        }
        """
        payload = {"query": query, "variables": {"ids": entry_ids}}
        data = self._post_json(self.config.graphql_url, payload)
        entries = data.get("data", {}).get("entries", [])
        records: list[tuple[str, int, str | None, str | None]] = []

        for entry in entries:
            if not entry:
                continue
            context = self._extract_solution_nmr_monomer_context(entry)
            if context is None:
                continue
            entry_id, year, _, polymer_entity, _ = context
            memberships = (
                polymer_entity.get("rcsb_polymer_entity_group_membership") or []
            )
            groups = self._extract_sequence_identity_groups(
                memberships,
                allowed_cutoffs={95, 100},
            )
            group_95 = groups.get(95)
            group_100 = groups.get(100)

            records.append((str(entry_id), year, group_95, group_100))
        return records

    def fetch_solution_nmr_monomer_xray_seed_records_for_ids(
        self, entry_ids: list[str]
    ) -> list[SolutionNMRMonomerXraySeedRecord]:
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
                rcsb_polymer_entity_instance_container_identifiers {
                  auth_to_entity_poly_seq_mapping
                }
                rcsb_polymer_instance_feature {
                  type
                  feature_positions {
                    beg_seq_id
                    end_seq_id
                  }
                }
              }
              rcsb_polymer_entity_group_membership {
                aggregation_method
                similarity_cutoff
                group_id
              }
            }
          }
        }
        """
        payload = {"query": query, "variables": {"ids": entry_ids}}
        data = self._post_json(self.config.graphql_url, payload)
        entries = data.get("data", {}).get("entries", [])
        records: list[SolutionNMRMonomerXraySeedRecord] = []

        for entry in entries:
            if not entry:
                continue
            context = self._extract_solution_nmr_monomer_context(entry)
            if context is None:
                continue
            entry_id, year, _, polymer_entity, chain_id = context
            memberships = (
                polymer_entity.get("rcsb_polymer_entity_group_membership") or []
            )
            groups = self._extract_sequence_identity_groups(
                memberships,
                allowed_cutoffs={95, 100},
            )
            group_95 = groups.get(95)
            group_100 = groups.get(100)
            core_range = self._extract_secondary_core_range(polymer_entity)
            if core_range is None:
                continue
            core_start, core_end = core_range

            records.append(
                SolutionNMRMonomerXraySeedRecord(
                    entry_id=str(entry_id),
                    year=year,
                    chain_id=chain_id,
                    core_start_seq_id=core_start,
                    core_end_seq_id=core_end,
                    group_id_95=group_95,
                    group_id_100=group_100,
                )
            )
        return records


class PDBMethodYearlyCollector:
    def __init__(self, client: RCSBClient, config: CollectorConfig) -> None:
        self.client = client
        self.config = config

    def _fetch_method_records(
        self, method: ExperimentalMethod
    ) -> list[YearlyCountRecord]:
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
        records: list[YearlyCountRecord] = []
        for method in methods:
            records.extend(self._fetch_method_records(method))
        return sorted(records, key=lambda record: (record.year, record.method))


class MembraneProteinYearlyCollector:
    def __init__(self, client: RCSBClient, config: CollectorConfig) -> None:
        self.client = client
        self.config = config

    def collect(self) -> list[MembraneYearlyCountRecord]:
        entry_ids = sorted(
            set(
                self.client.fetch_entry_ids_for_membrane_annotations(
                    MEMBRANE_ANNOTATION_TYPES
                )
            )
        )
        LOGGER.info("Membrane proteins: total unique IDs collected: %d", len(entry_ids))
        year_counter: Counter[int] = Counter()

        batches = list(chunked(entry_ids, self.config.graphql_batch_size))
        for batch_dates in collect_batch_results(
            batches=batches,
            max_workers=self.config.max_workers,
            fetch_fn=self.client.fetch_deposit_dates_for_ids,
            progress_label="Membrane proteins",
        ):
            years = filter(
                None, (extract_year(date_value) for date_value in batch_dates)
            )
            year_counter.update(years)

        return [
            MembraneYearlyCountRecord(year=year, count=count)
            for year, count in sorted(year_counter.items())
        ]


class SolutionNMRWeightCollector:
    def __init__(self, client: RCSBClient, config: CollectorConfig) -> None:
        self.client = client
        self.config = config

    def collect(self) -> list[SolutionNMRWeightRecord]:
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


class SolutionNMRMonomerSecondaryCollector:
    def __init__(
        self,
        client: RCSBClient,
        config: CollectorConfig,
        dssp_executable: str,
        cache_dir: Path,
    ) -> None:
        self.client = client
        self.config = config
        self.dssp_executable = dssp_executable
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def iter_batches(
        self,
        skip_entry_ids: set[str] | None = None,
    ) -> Iterator[
        tuple[
            list[SolutionNMRMonomerSecondaryRecord],
            list[SolutionNMRMonomerSecondaryByModelRecord],
        ]
    ]:
        entry_ids = fetch_solution_nmr_entry_ids(
            client=self.client,
            log_label="SOLUTION NMR monomer-secondary",
        )
        batches = list(chunked(entry_ids, self.config.graphql_batch_size))
        if not batches:
            return

        for batch_idx, batch in enumerate(batches, start=1):
            batch_entry_count = 0
            for secondary_record, by_model_records in (
                self.client.iter_solution_nmr_monomer_secondary_records_for_ids(
                    entry_ids=batch,
                    dssp_executable=self.dssp_executable,
                    pdb_cache_dir=self.cache_dir,
                    skip_entry_ids=skip_entry_ids,
                )
            ):
                batch_entry_count += 1
                yield [secondary_record], by_model_records
            LOGGER.info(
                "SOLUTION NMR monomer-secondary: processed batch %d/%d (entries: %d)",
                batch_idx,
                len(batches),
                batch_entry_count,
            )

    def iter_records(self) -> Iterator[SolutionNMRMonomerSecondaryRecord]:
        for batch_records, _ in self.iter_batches():
            for record in batch_records:
                yield record

    def iter_model_records(self) -> Iterator[SolutionNMRMonomerSecondaryByModelRecord]:
        for _, batch_model_records in self.iter_batches():
            for record in batch_model_records:
                yield record

    def collect(self) -> list[SolutionNMRMonomerSecondaryRecord]:
        records = list(self.iter_records())
        return sorted(records, key=lambda record: (record.year, record.entry_id))


class SolutionNMRMonomerStrideCollector:
    def __init__(
        self,
        client: RCSBClient,
        config: CollectorConfig,
        stride_executable: str,
        cache_dir: Path,
    ) -> None:
        self.client = client
        self.config = config
        self.stride_executable = stride_executable
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def iter_batches(self) -> Iterator[list[SolutionNMRMonomerStrideRecord]]:
        entry_ids = fetch_solution_nmr_entry_ids(
            client=self.client,
            log_label="SOLUTION NMR monomer-stride",
        )
        batches = list(chunked(entry_ids, self.config.graphql_batch_size))
        if not batches:
            return

        for batch_idx, batch in enumerate(batches, start=1):
            batch_records = self.client.fetch_solution_nmr_monomer_stride_records_for_ids(
                entry_ids=batch,
                stride_executable=self.stride_executable,
                pdb_cache_dir=self.cache_dir,
            )
            LOGGER.info(
                "SOLUTION NMR monomer-stride: processed batch %d/%d (entries: %d)",
                batch_idx,
                len(batches),
                len(batch_records),
            )
            yield batch_records

    def iter_records(self) -> Iterator[SolutionNMRMonomerStrideRecord]:
        for batch_records in self.iter_batches():
            for record in batch_records:
                yield record

    def collect(self) -> list[SolutionNMRMonomerStrideRecord]:
        records = list(self.iter_records())
        return sorted(records, key=lambda record: (record.year, record.entry_id))


class SolutionNMRMonomerSecondaryModeledFirstModelCollector:
    def __init__(
        self,
        client: RCSBClient,
        config: CollectorConfig,
        dssp_executable: str,
        cache_dir: Path,
    ) -> None:
        self.client = client
        self.config = config
        self.dssp_executable = dssp_executable
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def iter_batches(
        self,
    ) -> Iterator[list[SolutionNMRMonomerSecondaryModeledFirstModelRecord]]:
        entry_ids = fetch_solution_nmr_entry_ids(
            client=self.client,
            log_label="SOLUTION NMR monomer-secondary-modeled-first-model",
        )
        batches = list(chunked(entry_ids, self.config.graphql_batch_size))
        if not batches:
            return

        for batch_idx, batch in enumerate(batches, start=1):
            batch_records = (
                self.client.fetch_solution_nmr_monomer_secondary_modeled_first_model_records_for_ids(
                    entry_ids=batch,
                    dssp_executable=self.dssp_executable,
                    pdb_cache_dir=self.cache_dir,
                )
            )
            LOGGER.info(
                (
                    "SOLUTION NMR monomer-secondary-modeled-first-model: "
                    "processed batch %d/%d (entries: %d)"
                ),
                batch_idx,
                len(batches),
                len(batch_records),
            )
            yield batch_records

    def iter_records(self) -> Iterator[SolutionNMRMonomerSecondaryModeledFirstModelRecord]:
        for batch_records in self.iter_batches():
            for record in batch_records:
                yield record

    def collect(self) -> list[SolutionNMRMonomerSecondaryModeledFirstModelRecord]:
        records = list(self.iter_records())
        return sorted(records, key=lambda record: (record.year, record.entry_id))


class SolutionNMRMonomerStrideModeledFirstModelCollector:
    def __init__(
        self,
        client: RCSBClient,
        config: CollectorConfig,
        stride_executable: str,
        cache_dir: Path,
    ) -> None:
        self.client = client
        self.config = config
        self.stride_executable = stride_executable
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def iter_batches(
        self,
    ) -> Iterator[list[SolutionNMRMonomerStrideModeledFirstModelRecord]]:
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
        for batch_records in self.iter_batches():
            for record in batch_records:
                yield record

    def collect(self) -> list[SolutionNMRMonomerStrideModeledFirstModelRecord]:
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
        self.client = client
        self.config = config
        self.cache_dir = cache_dir
        self.precision_workers = max(1, precision_workers)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _download_pdb_if_needed(self, entry_id: str) -> Path:
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
    ) -> tuple[int, int, int, float] | None:
        model_maps, raw_ca_counts_per_model = parse_models_ca_coords_with_stats(
            pdb_path=pdb_path,
            chain_id=chain_id,
            start_seq_id=start_seq_id,
            end_seq_id=end_seq_id,
        )
        if len(model_maps) < 2:
            return None

        common_resids = set(model_maps[0].keys())
        for model_map in model_maps[1:]:
            common_resids &= set(model_map.keys())
        if len(common_resids) < 3:
            return None
        sorted_resids = sorted(common_resids)

        coords = np.asarray(
            [[model_map[resid] for resid in sorted_resids] for model_map in model_maps],
            dtype=float,
        )
        reference_coords = np.mean(coords, axis=0)
        per_model_rmsd = [
            _superposed_rmsd(model_coord, reference_coords) for model_coord in coords
        ]

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
            len(model_maps),
            len(sorted_resids),
            int(n_ca_core_raw),
            float(np.mean(per_model_rmsd)),
        )

    def _compute_record(
        self, core: SolutionNMRMonomerCoreRegionRecord
    ) -> SolutionNMRMonomerPrecisionRecord | None:
        try:
            pdb_path = self._download_pdb_if_needed(core.entry_id)
            result = self._compute_mean_rmsd_to_average(
                pdb_path=pdb_path,
                chain_id=core.chain_id,
                start_seq_id=core.core_start_seq_id,
                end_seq_id=core.core_end_seq_id,
            )
            if result is None:
                return None
            n_models, n_ca_core_used, n_ca_core_raw, mean_rmsd = result
            return SolutionNMRMonomerPrecisionRecord(
                entry_id=core.entry_id,
                year=core.year,
                chain_id=core.chain_id,
                core_start_seq_id=core.core_start_seq_id,
                core_end_seq_id=core.core_end_seq_id,
                n_models=n_models,
                n_ca_core_used=n_ca_core_used,
                n_ca_core_raw=n_ca_core_raw,
                mean_rmsd_angstrom=mean_rmsd,
            )
        except Exception as exc:
            LOGGER.warning(
                "Precision calculation failed for %s: %s", core.entry_id, exc
            )
            return None

    def collect(
        self,
        max_entries: int | None = None,
        skip_entry_ids: set[str] | None = None,
        on_record: Callable[[SolutionNMRMonomerPrecisionRecord], None] | None = None,
    ) -> list[SolutionNMRMonomerPrecisionRecord]:
        skip_entry_ids = skip_entry_ids or set()
        entry_ids = fetch_solution_nmr_entry_ids(
            client=self.client,
            log_label="SOLUTION NMR precision",
        )

        batches = list(chunked(entry_ids, self.config.graphql_batch_size))
        core_records: list[SolutionNMRMonomerCoreRegionRecord] = []
        for batch_core in collect_batch_results(
            batches=batches,
            max_workers=self.config.max_workers,
            fetch_fn=self.client.fetch_solution_nmr_monomer_core_region_records_for_ids,
            progress_label="SOLUTION NMR precision core ranges",
        ):
            core_records.extend(batch_core)

        filtered_core = [
            record for record in core_records if record.entry_id not in skip_entry_ids
        ]
        filtered_core = sorted(filtered_core, key=lambda r: (r.year, r.entry_id))
        if max_entries is not None:
            filtered_core = filtered_core[: max(0, max_entries)]
        LOGGER.info(
            "SOLUTION NMR precision: entries to process after filters: %d",
            len(filtered_core),
        )

        precision_records: list[SolutionNMRMonomerPrecisionRecord] = []
        with ThreadPoolExecutor(max_workers=self.precision_workers) as executor:
            future_map = {
                executor.submit(self._compute_record, core): idx
                for idx, core in enumerate(filtered_core, start=1)
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
                        "SOLUTION NMR precision RMSD: processed %d/%d entries",
                        idx,
                        total,
                    )

        return sorted(precision_records, key=lambda r: (r.year, r.entry_id))


class SolutionNMRMonomerQualityCollector:
    def __init__(self, client: RCSBClient, config: CollectorConfig) -> None:
        self.client = client
        self.config = config

    def collect(self) -> list[SolutionNMRMonomerQualityRecord]:
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
    def __init__(self, client: RCSBClient, config: CollectorConfig) -> None:
        self.client = client
        self.config = config
        self.group_query_batch_size = 200

    def _resolve_xray_groups(
        self, group_ids: set[str], similarity_cutoff: int
    ) -> set[str]:
        if not group_ids:
            return set()
        found_groups: set[str] = set()
        group_batches = list(chunked(sorted(group_ids), self.group_query_batch_size))
        for batch_idx, group_batch in enumerate(group_batches, start=1):
            entity_ids = self.client.fetch_xray_polymer_entity_ids_for_group_ids(
                group_batch
            )
            if entity_ids:
                for entity_batch in chunked(entity_ids, self.config.graphql_batch_size):
                    matched = self.client.fetch_sequence_identity_group_ids_for_polymer_entity_ids(
                        entity_batch, similarity_cutoff=similarity_cutoff
                    )
                    found_groups.update(gid for gid in matched if gid in group_ids)
            LOGGER.info(
                "SOLUTION NMR X-ray homologs %d%%: processed group batch %d/%d",
                similarity_cutoff,
                batch_idx,
                len(group_batches),
            )
        return found_groups

    def collect(
        self,
    ) -> tuple[
        list[SolutionNMRMonomerXrayHomologRecord],
        list[SolutionNMRMonomerXrayHomologRecord],
    ]:
        entry_ids = fetch_solution_nmr_entry_ids(
            client=self.client,
            log_label="SOLUTION NMR monomer X-ray homologs",
        )
        batches = list(chunked(entry_ids, self.config.graphql_batch_size))
        base_rows: list[tuple[str, int, str | None, str | None]] = []

        for batch_rows in collect_batch_results(
            batches=batches,
            max_workers=self.config.max_workers,
            fetch_fn=self.client.fetch_solution_nmr_monomer_xray_group_records_for_ids,
            progress_label="SOLUTION NMR monomer X-ray homolog base",
        ):
            base_rows.extend(batch_rows)

        group_ids_95 = {row[2] for row in base_rows if row[2]}
        group_ids_100 = {row[3] for row in base_rows if row[3]}
        LOGGER.info(
            "SOLUTION NMR monomer X-ray homologs: unique groups found (%d%%=%d, %d%%=%d)",
            95,
            len(group_ids_95),
            100,
            len(group_ids_100),
        )

        xray_groups_95 = self._resolve_xray_groups(group_ids_95, similarity_cutoff=95)
        xray_groups_100 = self._resolve_xray_groups(
            group_ids_100, similarity_cutoff=100
        )

        records_95: list[SolutionNMRMonomerXrayHomologRecord] = []
        records_100: list[SolutionNMRMonomerXrayHomologRecord] = []
        for entry_id, year, group_95, group_100 in base_rows:
            records_95.append(
                SolutionNMRMonomerXrayHomologRecord(
                    entry_id=entry_id,
                    year=year,
                    sequence_identity_percent=95,
                    group_id=group_95,
                    has_xray_homolog=bool(group_95 and group_95 in xray_groups_95),
                )
            )
            records_100.append(
                SolutionNMRMonomerXrayHomologRecord(
                    entry_id=entry_id,
                    year=year,
                    sequence_identity_percent=100,
                    group_id=group_100,
                    has_xray_homolog=bool(group_100 and group_100 in xray_groups_100),
                )
            )

        key_fn = lambda r: (r.year, r.entry_id)
        return sorted(records_95, key=key_fn), sorted(records_100, key=key_fn)


class SolutionNMRMonomerXrayRmsdCollector:
    def __init__(
        self,
        client: RCSBClient,
        config: CollectorConfig,
        cache_dir: Path,
        rmsd_workers: int,
        sequence_identity_percent: int = 100,
    ) -> None:
        if sequence_identity_percent not in {95, 100}:
            raise ValueError("sequence_identity_percent must be 95 or 100")
        self.client = client
        self.config = config
        self.cache_dir = cache_dir
        self.rmsd_workers = max(1, rmsd_workers)
        self.sequence_identity_percent = sequence_identity_percent
        self.group_query_batch_size = 200
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _download_pdb_if_needed(self, entry_id: str) -> Path:
        return download_pdb_if_needed(
            session=self.client.session,
            config=self.config,
            cache_dir=self.cache_dir,
            entry_id=entry_id,
        )

    @staticmethod
    def _compute_ca_rmsd_to_xray(
        nmr_pdb_path: Path,
        nmr_chain_id: str,
        nmr_core_start_seq_id: int,
        nmr_core_end_seq_id: int,
        xray_pdb_path: Path,
        xray_chain_id: str,
    ) -> tuple[int, float, int, int, int, int] | None:
        nmr_models = parse_models_ca_coords(
            nmr_pdb_path,
            nmr_chain_id,
            start_seq_id=nmr_core_start_seq_id,
            end_seq_id=nmr_core_end_seq_id,
        )
        if len(nmr_models) < 2:
            return None
        xray_models = parse_models_ca_coords(xray_pdb_path, xray_chain_id)
        if not xray_models:
            return None

        common_nmr_resids = set(nmr_models[0].keys())
        for model_map in nmr_models[1:]:
            common_nmr_resids &= set(model_map.keys())
        if len(common_nmr_resids) < 3:
            return None

        xray_resids = set(xray_models[0].keys())
        if len(xray_resids) < 3:
            return None

        nmr_min = min(common_nmr_resids)
        nmr_max = max(common_nmr_resids)
        xray_min = min(xray_resids)
        xray_max = max(xray_resids)

        max_overlap = 0
        overlap_candidates: list[tuple[int, list[int]]] = []
        for offset in range(xray_min - nmr_max, xray_max - nmr_min + 1):
            mapped_resids = sorted(
                resid for resid in common_nmr_resids if (resid + offset) in xray_resids
            )
            overlap = len(mapped_resids)
            if overlap < 3:
                continue
            if overlap > max_overlap:
                max_overlap = overlap
                overlap_candidates = [(offset, mapped_resids)]
            elif overlap == max_overlap:
                overlap_candidates.append((offset, mapped_resids))

        if not overlap_candidates:
            return None

        best_alignment: tuple[int, list[int], float] | None = None
        for offset, common_resids in overlap_candidates:
            nmr_coords = np.asarray(
                [
                    [model_map[resid] for resid in common_resids]
                    for model_map in nmr_models
                ],
                dtype=float,
            )
            xray_coords = np.asarray(
                [xray_models[0][resid + offset] for resid in common_resids],
                dtype=float,
            )
            per_model_rmsd = [
                _superposed_rmsd(model_coord, xray_coords) for model_coord in nmr_coords
            ]
            mean_rmsd = float(np.mean(per_model_rmsd))

            if best_alignment is None:
                best_alignment = (offset, common_resids, mean_rmsd)
                continue

            best_offset, _, best_rmsd = best_alignment
            if mean_rmsd < best_rmsd or (
                np.isclose(mean_rmsd, best_rmsd)
                and (abs(offset), offset) < (abs(best_offset), best_offset)
            ):
                best_alignment = (offset, common_resids, mean_rmsd)

        if best_alignment is None:
            return None
        best_offset, common_resids, rmsd_value = best_alignment
        xray_common_resids = [resid + best_offset for resid in common_resids]
        return (
            len(common_resids),
            rmsd_value,
            min(common_resids),
            max(common_resids),
            min(xray_common_resids),
            max(xray_common_resids),
        )

    def _resolve_best_xray_analog_by_group(
        self, group_ids: set[str]
    ) -> dict[str, tuple[str, tuple[str, ...], float]]:
        if not group_ids:
            return {}

        best_by_group: dict[str, tuple[str, tuple[str, ...], float]] = {}
        resolution_cache: dict[str, float] = {}
        group_batches = list(chunked(sorted(group_ids), self.group_query_batch_size))

        for batch_idx, group_batch in enumerate(group_batches, start=1):
            entity_ids = self.client.fetch_xray_polymer_entity_ids_for_group_ids(
                group_batch
            )
            for entity_batch in chunked(entity_ids, self.config.graphql_batch_size):
                mappings = self.client.fetch_polymer_entity_group_mapping_for_ids(
                    entity_batch, similarity_cutoff=self.sequence_identity_percent
                )
                if not mappings:
                    continue
                unknown_entries = sorted(
                    {m.entry_id for m in mappings if m.entry_id not in resolution_cache}
                )
                for entry_batch in chunked(
                    unknown_entries, self.config.graphql_batch_size
                ):
                    resolution_cache.update(
                        self.client.fetch_entry_resolution_for_ids(entry_batch)
                    )

                for mapping in mappings:
                    if mapping.group_id not in group_ids:
                        continue
                    resolution = resolution_cache.get(mapping.entry_id)
                    if resolution is None:
                        continue
                    candidate = (mapping.entry_id, mapping.chain_ids, resolution)
                    current = best_by_group.get(mapping.group_id)
                    if (
                        current is None
                        or candidate[2] < current[2]
                        or (
                            candidate[2] == current[2]
                            and (candidate[0], candidate[1]) < (current[0], current[1])
                        )
                    ):
                        best_by_group[mapping.group_id] = candidate

            LOGGER.info(
                "SOLUTION NMR X-ray RMSD %d%%: resolved best analog for group batch %d/%d",
                self.sequence_identity_percent,
                batch_idx,
                len(group_batches),
            )
        return best_by_group

    def _compute_record(
        self,
        seed: SolutionNMRMonomerXraySeedRecord,
        best_xray_by_group: dict[str, tuple[str, tuple[str, ...], float]],
    ) -> SolutionNMRMonomerXrayRmsdRecord | None:
        group_id = (
            seed.group_id_95
            if self.sequence_identity_percent == 95
            else seed.group_id_100
        )
        if not group_id:
            return None

        best = best_xray_by_group.get(group_id)
        if best is None:
            return None
        xray_entry_id, xray_chain_ids, xray_resolution = best

        try:
            nmr_pdb_path = self._download_pdb_if_needed(seed.entry_id)
            xray_pdb_path = self._download_pdb_if_needed(xray_entry_id)

            best_chain_result: tuple[str, int, float, int, int, int, int] | None = None
            for xray_chain_id in xray_chain_ids:
                rmsd_result = self._compute_ca_rmsd_to_xray(
                    nmr_pdb_path=nmr_pdb_path,
                    nmr_chain_id=seed.chain_id,
                    nmr_core_start_seq_id=seed.core_start_seq_id,
                    nmr_core_end_seq_id=seed.core_end_seq_id,
                    xray_pdb_path=xray_pdb_path,
                    xray_chain_id=xray_chain_id,
                )
                if rmsd_result is None:
                    continue
                (
                    n_common_ca,
                    rmsd_ca,
                    nmr_core_start,
                    nmr_core_end,
                    xray_core_start,
                    xray_core_end,
                ) = rmsd_result
                candidate = (
                    xray_chain_id,
                    n_common_ca,
                    rmsd_ca,
                    nmr_core_start,
                    nmr_core_end,
                    xray_core_start,
                    xray_core_end,
                )
                if (
                    best_chain_result is None
                    or candidate[1] > best_chain_result[1]
                    or (
                        candidate[1] == best_chain_result[1]
                        and candidate[2] < best_chain_result[2]
                    )
                ):
                    best_chain_result = candidate

            if best_chain_result is None:
                return None
            (
                xray_chain_id,
                n_common_ca,
                rmsd_ca,
                nmr_core_start,
                nmr_core_end,
                xray_core_start,
                xray_core_end,
            ) = best_chain_result
            return SolutionNMRMonomerXrayRmsdRecord(
                entry_id=seed.entry_id,
                year=seed.year,
                sequence_identity_percent=self.sequence_identity_percent,
                nmr_chain_id=seed.chain_id,
                nmr_core_start_seq_id=nmr_core_start,
                nmr_core_end_seq_id=nmr_core_end,
                xray_entry_id=xray_entry_id,
                xray_chain_id=xray_chain_id,
                xray_core_start_seq_id=xray_core_start,
                xray_core_end_seq_id=xray_core_end,
                xray_resolution_angstrom=xray_resolution,
                n_common_ca=n_common_ca,
                rmsd_ca_angstrom=rmsd_ca,
            )
        except Exception as exc:
            LOGGER.warning(
                "X-ray RMSD calculation failed for %s: %s", seed.entry_id, exc
            )
            return None

    def collect(
        self,
        max_entries: int | None = None,
        skip_entry_ids: set[str] | None = None,
        on_record: Callable[[SolutionNMRMonomerXrayRmsdRecord], None] | None = None,
    ) -> list[SolutionNMRMonomerXrayRmsdRecord]:
        skip_entry_ids = skip_entry_ids or set()
        entry_ids = fetch_solution_nmr_entry_ids(
            client=self.client,
            log_label=f"SOLUTION NMR X-ray RMSD {self.sequence_identity_percent}%",
        )

        batches = list(chunked(entry_ids, self.config.graphql_batch_size))
        seeds: list[SolutionNMRMonomerXraySeedRecord] = []
        for batch_seeds in collect_batch_results(
            batches=batches,
            max_workers=self.config.max_workers,
            fetch_fn=self.client.fetch_solution_nmr_monomer_xray_seed_records_for_ids,
            progress_label="SOLUTION NMR X-ray RMSD base",
        ):
            seeds.extend(batch_seeds)

        filtered_seeds = [
            seed
            for seed in seeds
            if seed.entry_id not in skip_entry_ids
            and (
                seed.group_id_95 is not None
                if self.sequence_identity_percent == 95
                else seed.group_id_100 is not None
            )
        ]
        filtered_seeds = sorted(filtered_seeds, key=lambda s: (s.year, s.entry_id))
        if max_entries is not None:
            filtered_seeds = filtered_seeds[: max(0, max_entries)]

        group_ids = {
            (
                seed.group_id_95
                if self.sequence_identity_percent == 95
                else seed.group_id_100
            )
            for seed in filtered_seeds
        }
        group_ids = {gid for gid in group_ids if gid}
        LOGGER.info(
            "SOLUTION NMR X-ray RMSD %d%%: entries to process=%d, unique groups=%d",
            self.sequence_identity_percent,
            len(filtered_seeds),
            len(group_ids),
        )

        best_xray_by_group = self._resolve_best_xray_analog_by_group(group_ids)

        records: list[SolutionNMRMonomerXrayRmsdRecord] = []
        with ThreadPoolExecutor(max_workers=self.rmsd_workers) as executor:
            future_map = {
                executor.submit(
                    self._compute_record,
                    seed=seed,
                    best_xray_by_group=best_xray_by_group,
                ): idx
                for idx, seed in enumerate(filtered_seeds, start=1)
            }
            total = len(future_map)
            for future in as_completed(future_map):
                record = future.result()
                if record is not None:
                    records.append(record)
                    if on_record is not None:
                        on_record(record)
                idx = future_map[future]
                if total > 0 and (idx % 50 == 0 or idx == total):
                    LOGGER.info(
                        "SOLUTION NMR X-ray RMSD %d%%: processed %d/%d entries",
                        self.sequence_identity_percent,
                        idx,
                        total,
                    )

        return sorted(records, key=lambda r: (r.year, r.entry_id))


def write_method_counts_csv(
    records: list[YearlyCountRecord], output_path: Path
) -> None:
    write_csv_rows(
        output_path=output_path,
        header=["year", "method", "count"],
        rows=((r.year, r.method, r.count) for r in records),
    )


def write_membrane_counts_csv(
    records: list[MembraneYearlyCountRecord], output_path: Path
) -> None:
    write_csv_rows(
        output_path=output_path,
        header=["year", "count"],
        rows=((r.year, r.count) for r in records),
    )


def write_solution_nmr_weights_csv(
    records: list[SolutionNMRWeightRecord], output_path: Path
) -> None:
    write_csv_rows(
        output_path=output_path,
        header=[
            "entry_id",
            "year",
            "molecular_weight_kda",
            "rcsb_entry_molecular_weight_kda",
            "polymer_molecular_weight_maximum",
            "modeled_molecular_weight_kda",
        ],
        rows=(
            (
                r.entry_id,
                r.year,
                f"{r.molecular_weight_kda:.3f}",
                (
                    f"{r.rcsb_entry_molecular_weight_kda:.3f}"
                    if r.rcsb_entry_molecular_weight_kda is not None
                    else ""
                ),
                (
                    f"{r.polymer_molecular_weight_maximum_kda:.3f}"
                    if r.polymer_molecular_weight_maximum_kda is not None
                    else ""
                ),
                (
                    f"{r.modeled_molecular_weight_kda:.3f}"
                    if r.modeled_molecular_weight_kda is not None
                    else ""
                ),
            )
            for r in records
        ),
    )


def write_solution_nmr_monomer_secondary_csv(
    records: list[SolutionNMRMonomerSecondaryRecord], output_path: Path
) -> None:
    header = [
        "entry_id",
        "year",
        "sequence_length",
        "secondary_structure_percent",
        "helix_fraction",
        "sheet_fraction",
        "deposited_model_count",
        "dssp_alpha_helix_fraction",
        "dssp_isolated_beta_bridge_fraction",
        "dssp_beta_strand_fraction",
        "dssp_3_10_helix_fraction",
        "dssp_pi_helix_fraction",
        "dssp_turn_fraction",
        "dssp_bend_fraction",
        "dssp_unassigned_percent",
        "dssp_secondary_structure_percent",
        "dssp_pdb_model_count",
        "dssp_analyzed_model_count",
    ]

    def _to_row(r: SolutionNMRMonomerSecondaryRecord) -> tuple[Any, ...]:
        return (
            r.entry_id,
            r.year,
            r.sequence_length,
            f"{r.secondary_structure_percent:.3f}",
            f"{r.helix_fraction:.6f}",
            f"{r.sheet_fraction:.6f}",
            r.deposited_model_count,
            f"{r.dssp_alpha_helix_fraction:.6f}",
            f"{r.dssp_isolated_beta_bridge_fraction:.6f}",
            f"{r.dssp_beta_strand_fraction:.6f}",
            f"{r.dssp_3_10_helix_fraction:.6f}",
            f"{r.dssp_pi_helix_fraction:.6f}",
            f"{r.dssp_turn_fraction:.6f}",
            f"{r.dssp_bend_fraction:.6f}",
            f"{r.dssp_unassigned_percent:.6f}",
            f"{r.dssp_secondary_structure_percent:.3f}",
            r.dssp_pdb_model_count,
            r.dssp_analyzed_model_count,
        )

    write_csv_rows(
        output_path=output_path,
        header=header,
        rows=(_to_row(r) for r in records),
    )


def write_solution_nmr_monomer_stride_csv(
    records: list[SolutionNMRMonomerStrideRecord], output_path: Path
) -> None:
    write_csv_rows(
        output_path=output_path,
        header=[
            "entry_id",
            "year",
            "sequence_length",
            "secondary_structure_percent",
            "helix_fraction",
            "sheet_fraction",
            "deposited_model_count",
            "stride_alpha_helix_fraction",
            "stride_3_10_helix_fraction",
            "stride_pi_helix_fraction",
            "stride_beta_strand_fraction",
            "stride_isolated_beta_bridge_fraction",
            "stride_turn_fraction",
            "stride_coil_fraction",
            "stride_secondary_structure_percent",
            "stride_pdb_model_count",
            "stride_analyzed_model_count",
        ],
        rows=(
            (
                r.entry_id,
                r.year,
                r.sequence_length,
                f"{r.secondary_structure_percent:.3f}",
                f"{r.helix_fraction:.6f}",
                f"{r.sheet_fraction:.6f}",
                r.deposited_model_count,
                f"{r.stride_alpha_helix_fraction:.6f}",
                f"{r.stride_3_10_helix_fraction:.6f}",
                f"{r.stride_pi_helix_fraction:.6f}",
                f"{r.stride_beta_strand_fraction:.6f}",
                f"{r.stride_isolated_beta_bridge_fraction:.6f}",
                f"{r.stride_turn_fraction:.6f}",
                f"{r.stride_coil_fraction:.6f}",
                f"{r.stride_secondary_structure_percent:.3f}",
                r.stride_pdb_model_count,
                r.stride_analyzed_model_count,
            )
            for r in records
        ),
    )


def stream_solution_nmr_monomer_stride_csv(
    records: Iterator[SolutionNMRMonomerStrideRecord],
    output_path: Path,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "entry_id",
                "year",
                "sequence_length",
                "secondary_structure_percent",
                "helix_fraction",
                "sheet_fraction",
                "deposited_model_count",
                "stride_alpha_helix_fraction",
                "stride_3_10_helix_fraction",
                "stride_pi_helix_fraction",
                "stride_beta_strand_fraction",
                "stride_isolated_beta_bridge_fraction",
                "stride_turn_fraction",
                "stride_coil_fraction",
                "stride_secondary_structure_percent",
                "stride_pdb_model_count",
                "stride_analyzed_model_count",
            ]
        )
        csvfile.flush()
        for record in records:
            writer.writerow(
                (
                    record.entry_id,
                    record.year,
                    record.sequence_length,
                    f"{record.secondary_structure_percent:.3f}",
                    f"{record.helix_fraction:.6f}",
                    f"{record.sheet_fraction:.6f}",
                    record.deposited_model_count,
                    f"{record.stride_alpha_helix_fraction:.6f}",
                    f"{record.stride_3_10_helix_fraction:.6f}",
                    f"{record.stride_pi_helix_fraction:.6f}",
                    f"{record.stride_beta_strand_fraction:.6f}",
                    f"{record.stride_isolated_beta_bridge_fraction:.6f}",
                    f"{record.stride_turn_fraction:.6f}",
                    f"{record.stride_coil_fraction:.6f}",
                    f"{record.stride_secondary_structure_percent:.3f}",
                    record.stride_pdb_model_count,
                    record.stride_analyzed_model_count,
                )
            )
            csvfile.flush()
            count += 1
    return count


def stream_solution_nmr_monomer_secondary_csv(
    records: Iterator[SolutionNMRMonomerSecondaryRecord],
    output_path: Path,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "entry_id",
                "year",
                "sequence_length",
                "secondary_structure_percent",
                "helix_fraction",
                "sheet_fraction",
                "deposited_model_count",
                "dssp_alpha_helix_fraction",
                "dssp_isolated_beta_bridge_fraction",
                "dssp_beta_strand_fraction",
                "dssp_3_10_helix_fraction",
                "dssp_pi_helix_fraction",
                "dssp_turn_fraction",
                "dssp_bend_fraction",
                "dssp_unassigned_percent",
                "dssp_secondary_structure_percent",
                "dssp_pdb_model_count",
                "dssp_analyzed_model_count",
            ]
        )
        csvfile.flush()
        for record in records:
            writer.writerow(
                (
                    record.entry_id,
                    record.year,
                    record.sequence_length,
                    f"{record.secondary_structure_percent:.3f}",
                    f"{record.helix_fraction:.6f}",
                    f"{record.sheet_fraction:.6f}",
                    record.deposited_model_count,
                    f"{record.dssp_alpha_helix_fraction:.6f}",
                    f"{record.dssp_isolated_beta_bridge_fraction:.6f}",
                    f"{record.dssp_beta_strand_fraction:.6f}",
                    f"{record.dssp_3_10_helix_fraction:.6f}",
                    f"{record.dssp_pi_helix_fraction:.6f}",
                    f"{record.dssp_turn_fraction:.6f}",
                    f"{record.dssp_bend_fraction:.6f}",
                    f"{record.dssp_unassigned_percent:.6f}",
                    f"{record.dssp_secondary_structure_percent:.3f}",
                    record.dssp_pdb_model_count,
                    record.dssp_analyzed_model_count,
                )
            )
            csvfile.flush()
            count += 1
    return count


def stream_solution_nmr_monomer_secondary_modeled_first_model_csv(
    records: Iterator[SolutionNMRMonomerSecondaryModeledFirstModelRecord],
    output_path: Path,
) -> int:
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
                "dssp_alpha_helix_fraction",
                "dssp_isolated_beta_bridge_fraction",
                "dssp_beta_strand_fraction",
                "dssp_3_10_helix_fraction",
                "dssp_pi_helix_fraction",
                "dssp_turn_fraction",
                "dssp_bend_fraction",
                "dssp_unassigned_percent",
                "dssp_secondary_structure_percent",
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
                    f"{record.dssp_alpha_helix_fraction:.6f}",
                    f"{record.dssp_isolated_beta_bridge_fraction:.6f}",
                    f"{record.dssp_beta_strand_fraction:.6f}",
                    f"{record.dssp_3_10_helix_fraction:.6f}",
                    f"{record.dssp_pi_helix_fraction:.6f}",
                    f"{record.dssp_turn_fraction:.6f}",
                    f"{record.dssp_bend_fraction:.6f}",
                    f"{record.dssp_unassigned_percent:.6f}",
                    f"{record.dssp_secondary_structure_percent:.3f}",
                )
            )
            csvfile.flush()
            count += 1
    return count


def stream_solution_nmr_monomer_stride_modeled_first_model_csv(
    records: Iterator[SolutionNMRMonomerStrideModeledFirstModelRecord],
    output_path: Path,
) -> int:
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


def write_solution_nmr_monomer_secondary_by_model_csv(
    records: list[SolutionNMRMonomerSecondaryByModelRecord], output_path: Path
) -> None:
    write_csv_rows(
        output_path=output_path,
        header=[
            "entry_id",
            "year",
            "chain_id",
            "sequence_length",
            "deposited_model_count",
            "pdb_model_count",
            "model_index",
            "assigned_residue_count",
            "secondary_structure_percent",
            "helix_fraction",
            "sheet_fraction",
            "dssp_alpha_helix_fraction",
            "dssp_isolated_beta_bridge_fraction",
            "dssp_beta_strand_fraction",
            "dssp_3_10_helix_fraction",
            "dssp_pi_helix_fraction",
            "dssp_turn_fraction",
            "dssp_bend_fraction",
            "dssp_unassigned_percent",
            "dssp_secondary_structure_percent",
        ],
        rows=(
            (
                r.entry_id,
                r.year,
                r.chain_id,
                r.sequence_length,
                r.deposited_model_count,
                r.pdb_model_count,
                r.model_index,
                r.assigned_residue_count,
                f"{r.secondary_structure_percent:.3f}",
                f"{r.helix_fraction:.6f}",
                f"{r.sheet_fraction:.6f}",
                f"{r.dssp_alpha_helix_fraction:.6f}",
                f"{r.dssp_isolated_beta_bridge_fraction:.6f}",
                f"{r.dssp_beta_strand_fraction:.6f}",
                f"{r.dssp_3_10_helix_fraction:.6f}",
                f"{r.dssp_pi_helix_fraction:.6f}",
                f"{r.dssp_turn_fraction:.6f}",
                f"{r.dssp_bend_fraction:.6f}",
                f"{r.dssp_unassigned_percent:.6f}",
                f"{r.dssp_secondary_structure_percent:.3f}",
            )
            for r in records
        ),
    )


def stream_solution_nmr_monomer_secondary_outputs_csv(
    batches: Iterator[
        tuple[
            list[SolutionNMRMonomerSecondaryRecord],
            list[SolutionNMRMonomerSecondaryByModelRecord],
        ]
    ],
    secondary_output_path: Path,
    secondary_by_model_output_path: Path,
    append: bool = False,
) -> tuple[int, int]:
    secondary_output_path.parent.mkdir(parents=True, exist_ok=True)
    secondary_by_model_output_path.parent.mkdir(parents=True, exist_ok=True)

    secondary_count = 0
    secondary_by_model_count = 0
    sec_mode = "a" if append else "w"
    sec_model_mode = "a" if append else "w"
    write_secondary_header = (
        (not append)
        or (not secondary_output_path.exists())
        or secondary_output_path.stat().st_size == 0
    )
    write_secondary_by_model_header = (
        (not append)
        or (not secondary_by_model_output_path.exists())
        or secondary_by_model_output_path.stat().st_size == 0
    )
    with secondary_output_path.open(sec_mode, newline="", encoding="utf-8") as sec_file:
        with secondary_by_model_output_path.open(
            sec_model_mode, newline="", encoding="utf-8"
        ) as sec_model_file:
            sec_writer = csv.writer(sec_file)
            sec_model_writer = csv.writer(sec_model_file)
            if write_secondary_header:
                sec_writer.writerow(
                    [
                        "entry_id",
                        "year",
                        "sequence_length",
                        "secondary_structure_percent",
                        "helix_fraction",
                        "sheet_fraction",
                        "deposited_model_count",
                        "dssp_alpha_helix_fraction",
                        "dssp_isolated_beta_bridge_fraction",
                        "dssp_beta_strand_fraction",
                        "dssp_3_10_helix_fraction",
                        "dssp_pi_helix_fraction",
                        "dssp_turn_fraction",
                        "dssp_bend_fraction",
                        "dssp_unassigned_percent",
                        "dssp_secondary_structure_percent",
                        "dssp_pdb_model_count",
                        "dssp_analyzed_model_count",
                    ]
                )
            if write_secondary_by_model_header:
                sec_model_writer.writerow(
                    [
                        "entry_id",
                        "year",
                        "chain_id",
                        "sequence_length",
                        "deposited_model_count",
                        "pdb_model_count",
                        "model_index",
                        "assigned_residue_count",
                        "secondary_structure_percent",
                        "helix_fraction",
                        "sheet_fraction",
                        "dssp_alpha_helix_fraction",
                        "dssp_isolated_beta_bridge_fraction",
                        "dssp_beta_strand_fraction",
                        "dssp_3_10_helix_fraction",
                        "dssp_pi_helix_fraction",
                        "dssp_turn_fraction",
                        "dssp_bend_fraction",
                        "dssp_unassigned_percent",
                        "dssp_secondary_structure_percent",
                    ]
                )
            sec_file.flush()
            sec_model_file.flush()

            for batch_records, batch_model_records in batches:
                for record in batch_records:
                    sec_writer.writerow(
                        (
                            record.entry_id,
                            record.year,
                            record.sequence_length,
                            f"{record.secondary_structure_percent:.3f}",
                            f"{record.helix_fraction:.6f}",
                            f"{record.sheet_fraction:.6f}",
                            record.deposited_model_count,
                            f"{record.dssp_alpha_helix_fraction:.6f}",
                            f"{record.dssp_isolated_beta_bridge_fraction:.6f}",
                            f"{record.dssp_beta_strand_fraction:.6f}",
                            f"{record.dssp_3_10_helix_fraction:.6f}",
                            f"{record.dssp_pi_helix_fraction:.6f}",
                            f"{record.dssp_turn_fraction:.6f}",
                            f"{record.dssp_bend_fraction:.6f}",
                            f"{record.dssp_unassigned_percent:.6f}",
                            f"{record.dssp_secondary_structure_percent:.3f}",
                            record.dssp_pdb_model_count,
                            record.dssp_analyzed_model_count,
                        )
                    )
                    secondary_count += 1
                sec_file.flush()

                for record in batch_model_records:
                    sec_model_writer.writerow(
                        (
                            record.entry_id,
                            record.year,
                            record.chain_id,
                            record.sequence_length,
                            record.deposited_model_count,
                            record.pdb_model_count,
                            record.model_index,
                            record.assigned_residue_count,
                            f"{record.secondary_structure_percent:.3f}",
                            f"{record.helix_fraction:.6f}",
                            f"{record.sheet_fraction:.6f}",
                            f"{record.dssp_alpha_helix_fraction:.6f}",
                            f"{record.dssp_isolated_beta_bridge_fraction:.6f}",
                            f"{record.dssp_beta_strand_fraction:.6f}",
                            f"{record.dssp_3_10_helix_fraction:.6f}",
                            f"{record.dssp_pi_helix_fraction:.6f}",
                            f"{record.dssp_turn_fraction:.6f}",
                            f"{record.dssp_bend_fraction:.6f}",
                            f"{record.dssp_unassigned_percent:.6f}",
                            f"{record.dssp_secondary_structure_percent:.3f}",
                        )
                    )
                    secondary_by_model_count += 1
                sec_model_file.flush()

    return secondary_count, secondary_by_model_count


def load_completed_solution_nmr_monomer_secondary_entry_ids(
    secondary_output_path: Path,
    secondary_by_model_output_path: Path,
) -> tuple[set[str], int, int, set[str]]:
    if not secondary_output_path.exists() or not secondary_by_model_output_path.exists():
        return set(), 0, 0, set()

    expected_model_rows_by_entry: dict[str, int] = {}
    secondary_row_count = 0
    with secondary_output_path.open("r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if not row:
                continue
            entry_id = str(row.get("entry_id") or "").strip().upper()
            if not entry_id:
                continue
            secondary_row_count += 1
            raw_model_count = str(row.get("dssp_pdb_model_count") or "").strip()
            try:
                expected_model_rows_by_entry[entry_id] = int(raw_model_count)
            except ValueError:
                expected_model_rows_by_entry[entry_id] = 0

    model_row_count_by_entry: Counter[str] = Counter()
    secondary_by_model_row_count = 0
    with secondary_by_model_output_path.open(
        "r", newline="", encoding="utf-8"
    ) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if not row:
                continue
            entry_id = str(row.get("entry_id") or "").strip().upper()
            if not entry_id:
                continue
            secondary_by_model_row_count += 1
            model_row_count_by_entry[entry_id] += 1

    completed_entry_ids = {
        entry_id
        for entry_id, expected_model_rows in expected_model_rows_by_entry.items()
        if model_row_count_by_entry.get(entry_id, 0) >= max(0, expected_model_rows)
    }
    incomplete_entry_ids = {
        entry_id
        for entry_id, expected_model_rows in expected_model_rows_by_entry.items()
        if model_row_count_by_entry.get(entry_id, 0) < max(0, expected_model_rows)
    }
    incomplete_entry_ids.update(
        set(model_row_count_by_entry.keys()) - set(expected_model_rows_by_entry.keys())
    )
    return (
        completed_entry_ids,
        secondary_row_count,
        secondary_by_model_row_count,
        incomplete_entry_ids,
    )


def prune_solution_nmr_monomer_secondary_csv_to_entry_ids(
    secondary_output_path: Path,
    secondary_by_model_output_path: Path,
    keep_entry_ids: set[str],
) -> tuple[int, int]:
    normalized_keep_ids = {entry_id.strip().upper() for entry_id in keep_entry_ids}

    def _rewrite_with_filter(path: Path) -> int:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        kept_count = 0
        with path.open("r", newline="", encoding="utf-8") as input_file:
            reader = csv.DictReader(input_file)
            fieldnames = list(reader.fieldnames or [])
            with tmp_path.open("w", newline="", encoding="utf-8") as output_file:
                writer = csv.DictWriter(output_file, fieldnames=fieldnames)
                if fieldnames:
                    writer.writeheader()
                for row in reader:
                    if not row:
                        continue
                    entry_id = str(row.get("entry_id") or "").strip().upper()
                    if entry_id and entry_id in normalized_keep_ids:
                        writer.writerow(row)
                        kept_count += 1
        tmp_path.replace(path)
        return kept_count

    kept_secondary_count = _rewrite_with_filter(secondary_output_path)
    kept_secondary_by_model_count = _rewrite_with_filter(secondary_by_model_output_path)
    return kept_secondary_count, kept_secondary_by_model_count


def read_solution_nmr_monomer_precision_csv(
    input_path: Path,
) -> list[SolutionNMRMonomerPrecisionRecord]:
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


def write_solution_nmr_monomer_precision_csv(
    records: list[SolutionNMRMonomerPrecisionRecord], output_path: Path
) -> None:
    write_csv_rows(
        output_path=output_path,
        header=list(SOLUTION_NMR_MONOMER_PRECISION_HEADER),
        rows=(_solution_nmr_monomer_precision_csv_row(r) for r in records),
    )


def write_solution_nmr_monomer_quality_csv(
    records: list[SolutionNMRMonomerQualityRecord], output_path: Path
) -> None:
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


def write_solution_nmr_monomer_xray_homolog_csv(
    records: list[SolutionNMRMonomerXrayHomologRecord], output_path: Path
) -> None:
    write_csv_rows(
        output_path=output_path,
        header=[
            "entry_id",
            "year",
            "sequence_identity_percent",
            "group_id",
            "has_xray_homolog",
        ],
        rows=(
            (
                r.entry_id,
                r.year,
                r.sequence_identity_percent,
                r.group_id or "",
                int(r.has_xray_homolog),
            )
            for r in records
        ),
    )


def read_solution_nmr_monomer_xray_rmsd_csv(
    input_path: Path,
) -> list[SolutionNMRMonomerXrayRmsdRecord]:
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
    return (
        record.entry_id,
        record.year,
        record.sequence_identity_percent,
        record.nmr_chain_id,
        record.nmr_core_start_seq_id if record.nmr_core_start_seq_id is not None else "",
        record.nmr_core_end_seq_id if record.nmr_core_end_seq_id is not None else "",
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
    write_csv_rows(
        output_path=output_path,
        header=list(SOLUTION_NMR_MONOMER_XRAY_RMSD_HEADER),
        rows=(_solution_nmr_monomer_xray_rmsd_csv_row(r) for r in records),
    )


def parse_dataset_kinds(raw_value: str) -> list[DatasetKind]:
    if raw_value.strip().lower() == "all":
        return [
            DatasetKind.METHOD_COUNTS,
            DatasetKind.MEMBRANE_PROTEIN_COUNTS,
            DatasetKind.SOLUTION_NMR_WEIGHTS,
            DatasetKind.SOLUTION_NMR_MONOMER_SECONDARY,
            DatasetKind.SOLUTION_NMR_MONOMER_SECONDARY_MODELED_FIRST_MODEL,
            DatasetKind.SOLUTION_NMR_MONOMER_STRIDE,
            DatasetKind.SOLUTION_NMR_MONOMER_STRIDE_MODELED_FIRST_MODEL,
            DatasetKind.SOLUTION_NMR_MONOMER_PRECISION,
            DatasetKind.SOLUTION_NMR_MONOMER_QUALITY,
            DatasetKind.SOLUTION_NMR_MONOMER_XRAY_HOMOLOGS,
            DatasetKind.SOLUTION_NMR_MONOMER_XRAY_RMSD,
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
    parser = argparse.ArgumentParser(
        description="Collect extensible PDB datasets from RCSB APIs."
    )
    parser.add_argument(
        "--datasets",
        type=parse_dataset_kinds,
        default=[
            DatasetKind.METHOD_COUNTS,
            DatasetKind.SOLUTION_NMR_WEIGHTS,
            DatasetKind.SOLUTION_NMR_MONOMER_SECONDARY,
        ],
        help="Comma-separated dataset kinds or 'all'. "
        "Available: method_counts, membrane_protein_counts, solution_nmr_weights, solution_nmr_monomer_secondary, solution_nmr_monomer_secondary_modeled_first_model, solution_nmr_monomer_stride, solution_nmr_monomer_stride_modeled_first_model, solution_nmr_monomer_precision, solution_nmr_monomer_quality, solution_nmr_monomer_xray_homologs, solution_nmr_monomer_xray_rmsd (default: the first three).",
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
        "--solution-nmr-output",
        type=Path,
        default=Path("data/solution_nmr_structure_weights.csv"),
        help="Output CSV path for solution_nmr_weights dataset.",
    )
    parser.add_argument(
        "--solution-nmr-monomer-secondary-output",
        type=Path,
        default=Path("data/solution_nmr_monomer_secondary_structure.csv"),
        help="Output CSV path for solution_nmr_monomer_secondary dataset.",
    )
    parser.add_argument(
        "--solution-nmr-monomer-secondary-by-model-output",
        type=Path,
        default=Path("data/solution_nmr_monomer_secondary_by_model.csv"),
        help="Output CSV path for per-model DSSP data in solution_nmr_monomer_secondary dataset.",
    )
    parser.add_argument(
        "--solution-nmr-monomer-secondary-modeled-first-model-output",
        type=Path,
        default=Path("data/solution_nmr_monomer_secondary_modeled_first_model.csv"),
        help=(
            "Output CSV path for solution_nmr_monomer_secondary_modeled_first_model "
            "dataset (DSSP for modeled residues of the first model only)."
        ),
    )
    parser.add_argument(
        "--solution-nmr-monomer-secondary-cache-dir",
        type=Path,
        default=Path("data/pdb_cache"),
        help="Directory to cache downloaded PDB files for DSSP in monomer-secondary dataset.",
    )
    parser.add_argument(
        "--solution-nmr-monomer-secondary-dssp-executable",
        type=str,
        default="",
        help=(
            "Path to DSSP executable (mkdssp/dssp) for monomer-secondary dataset. "
            "If omitted, script tries mkdssp, dssp, and dssp/build/mkdssp."
        ),
    )
    parser.add_argument(
        "--solution-nmr-monomer-secondary-overwrite",
        action="store_true",
        help=(
            "Recompute monomer-secondary CSVs from scratch (ignore existing rows and "
            "disable resume skipping)."
        ),
    )
    parser.add_argument(
        "--solution-nmr-monomer-stride-output",
        type=Path,
        default=Path("data/solution_nmr_monomer_stride_structure.csv"),
        help="Output CSV path for solution_nmr_monomer_stride dataset.",
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
        "--solution-nmr-monomer-precision-output",
        type=Path,
        default=Path("data/solution_nmr_monomer_precision.csv"),
        help="Output CSV path for solution_nmr_monomer_precision dataset.",
    )
    parser.add_argument(
        "--solution-nmr-monomer-quality-output",
        type=Path,
        default=Path("data/solution_nmr_monomer_quality_metrics.csv"),
        help="Output CSV path for solution_nmr_monomer_quality dataset.",
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
        "--solution-nmr-monomer-xray-rmsd-output",
        type=Path,
        default=Path("data/solution_nmr_monomer_xray_rmsd.csv"),
        help="Output CSV path for solution_nmr_monomer_xray_rmsd dataset.",
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
        help="Sequence identity cutoff for selecting X-ray homolog groups (95 or 100).",
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
        LOGGER.info(
            "Saved %d records to %s",
            len(membrane_records),
            args.membrane_counts_output,
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

    if DatasetKind.SOLUTION_NMR_MONOMER_SECONDARY in args.datasets:
        secondary_dssp_executable = resolve_dssp_executable(
            args.solution_nmr_monomer_secondary_dssp_executable
        )
        if secondary_dssp_executable is None:
            raise RuntimeError(
                "DSSP executable not found for solution_nmr_monomer_secondary. "
                "Provide --solution-nmr-monomer-secondary-dssp-executable or install mkdssp/dssp."
            )
        LOGGER.info(
            "SOLUTION NMR monomer-secondary: using DSSP executable %s",
            secondary_dssp_executable,
        )
        sec_collector = SolutionNMRMonomerSecondaryCollector(
            client=client,
            config=config,
            dssp_executable=secondary_dssp_executable,
            cache_dir=Path(args.solution_nmr_monomer_secondary_cache_dir),
        )
        resume_skip_ids: set[str] = set()
        existing_sec_count = 0
        existing_sec_model_count = 0
        append_mode = False
        secondary_output_path = Path(args.solution_nmr_monomer_secondary_output)
        secondary_by_model_output_path = Path(
            args.solution_nmr_monomer_secondary_by_model_output
        )
        if not args.solution_nmr_monomer_secondary_overwrite:
            (
                resume_skip_ids,
                existing_sec_count,
                existing_sec_model_count,
                incomplete_sec_entry_ids,
            ) = load_completed_solution_nmr_monomer_secondary_entry_ids(
                secondary_output_path=secondary_output_path,
                secondary_by_model_output_path=secondary_by_model_output_path,
            )
            if incomplete_sec_entry_ids:
                LOGGER.warning(
                    (
                        "SOLUTION NMR monomer-secondary: found %d incomplete entries "
                        "from previous run, pruning partial rows before resume"
                    ),
                    len(incomplete_sec_entry_ids),
                )
                (
                    existing_sec_count,
                    existing_sec_model_count,
                ) = prune_solution_nmr_monomer_secondary_csv_to_entry_ids(
                    secondary_output_path=secondary_output_path,
                    secondary_by_model_output_path=secondary_by_model_output_path,
                    keep_entry_ids=resume_skip_ids,
                )
                LOGGER.info(
                    (
                        "SOLUTION NMR monomer-secondary: retained %d completed rows and "
                        "%d completed model rows after prune"
                    ),
                    existing_sec_count,
                    existing_sec_model_count,
                )
            append_mode = bool(
                secondary_output_path.exists() and secondary_by_model_output_path.exists()
            )
            if append_mode:
                LOGGER.info(
                    (
                        "SOLUTION NMR monomer-secondary: resume enabled, "
                        "loaded %d completed entries (%d secondary rows, %d model rows)"
                    ),
                    len(resume_skip_ids),
                    existing_sec_count,
                    existing_sec_model_count,
                )

        new_sec_count, new_sec_model_count = (
            stream_solution_nmr_monomer_secondary_outputs_csv(
                batches=sec_collector.iter_batches(skip_entry_ids=resume_skip_ids),
                secondary_output_path=secondary_output_path,
                secondary_by_model_output_path=secondary_by_model_output_path,
                append=append_mode,
            )
        )
        sec_count = existing_sec_count + new_sec_count if append_mode else new_sec_count
        sec_model_count = (
            existing_sec_model_count + new_sec_model_count
            if append_mode
            else new_sec_model_count
        )
        LOGGER.info(
            "Saved %d records to %s (new: %d)",
            sec_count,
            args.solution_nmr_monomer_secondary_output,
            new_sec_count,
        )
        LOGGER.info(
            "Saved %d model+chain records to %s (new: %d)",
            sec_model_count,
            args.solution_nmr_monomer_secondary_by_model_output,
            new_sec_model_count,
        )

    if DatasetKind.SOLUTION_NMR_MONOMER_SECONDARY_MODELED_FIRST_MODEL in args.datasets:
        modeled_first_dssp_executable = resolve_dssp_executable(
            args.solution_nmr_monomer_secondary_dssp_executable
        )
        if modeled_first_dssp_executable is None:
            raise RuntimeError(
                "DSSP executable not found for solution_nmr_monomer_secondary_modeled_first_model. "
                "Provide --solution-nmr-monomer-secondary-dssp-executable or install mkdssp/dssp."
            )
        LOGGER.info(
            (
                "SOLUTION NMR monomer-secondary-modeled-first-model: "
                "using DSSP executable %s"
            ),
            modeled_first_dssp_executable,
        )
        modeled_secondary_collector = (
            SolutionNMRMonomerSecondaryModeledFirstModelCollector(
                client=client,
                config=config,
                dssp_executable=modeled_first_dssp_executable,
                cache_dir=Path(args.solution_nmr_monomer_secondary_cache_dir),
            )
        )
        modeled_secondary_count = (
            stream_solution_nmr_monomer_secondary_modeled_first_model_csv(
                records=modeled_secondary_collector.iter_records(),
                output_path=Path(
                    args.solution_nmr_monomer_secondary_modeled_first_model_output
                ),
            )
        )
        LOGGER.info(
            "Saved %d records to %s",
            modeled_secondary_count,
            args.solution_nmr_monomer_secondary_modeled_first_model_output,
        )

    if DatasetKind.SOLUTION_NMR_MONOMER_STRIDE in args.datasets:
        stride_executable = resolve_stride_executable(
            args.solution_nmr_monomer_stride_executable
        )
        if stride_executable is None:
            raise RuntimeError(
                "STRIDE executable not found for solution_nmr_monomer_stride. "
                "Provide --solution-nmr-monomer-stride-executable or install stride."
            )
        LOGGER.info(
            "SOLUTION NMR monomer-stride: using STRIDE executable %s",
            stride_executable,
        )
        stride_collector = SolutionNMRMonomerStrideCollector(
            client=client,
            config=config,
            stride_executable=stride_executable,
            cache_dir=Path(args.solution_nmr_monomer_stride_cache_dir),
        )
        stride_count = stream_solution_nmr_monomer_stride_csv(
            records=stride_collector.iter_records(),
            output_path=Path(args.solution_nmr_monomer_stride_output),
        )
        LOGGER.info(
            "Saved %d records to %s",
            stride_count,
            args.solution_nmr_monomer_stride_output,
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

    if DatasetKind.SOLUTION_NMR_MONOMER_PRECISION in args.datasets:
        existing_records: list[SolutionNMRMonomerPrecisionRecord] = []
        skip_entry_ids: set[str] = set()
        precision_output_path = Path(args.solution_nmr_monomer_precision_output)
        if (
            not args.precision_overwrite
            and precision_output_path.exists()
        ):
            existing_records = read_solution_nmr_monomer_precision_csv(precision_output_path)
            skip_entry_ids = {record.entry_id for record in existing_records}
            LOGGER.info(
                "SOLUTION NMR precision: loaded %d existing records for resume",
                len(existing_records),
            )

        precision_collector = SolutionNMRMonomerPrecisionCollector(
            client=client,
            config=config,
            cache_dir=Path(args.precision_cache_dir),
            precision_workers=args.precision_workers,
        )
        existing_records = sorted(existing_records, key=lambda r: (r.year, r.entry_id))
        precision_output_path.parent.mkdir(parents=True, exist_ok=True)
        with precision_output_path.open("w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(SOLUTION_NMR_MONOMER_PRECISION_HEADER)
            for record in existing_records:
                writer.writerow(_solution_nmr_monomer_precision_csv_row(record))
            csvfile.flush()

            def _on_precision_record(
                record: SolutionNMRMonomerPrecisionRecord,
            ) -> None:
                writer.writerow(_solution_nmr_monomer_precision_csv_row(record))
                csvfile.flush()

            new_records = precision_collector.collect(
                max_entries=args.precision_max_entries,
                skip_entry_ids=skip_entry_ids,
                on_record=_on_precision_record,
            )

        LOGGER.info(
            "Saved %d records to %s (new: %d)",
            len(existing_records) + len(new_records),
            args.solution_nmr_monomer_precision_output,
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

    if DatasetKind.SOLUTION_NMR_MONOMER_XRAY_HOMOLOGS in args.datasets:
        homolog_collector = SolutionNMRMonomerXrayHomologCollector(
            client=client, config=config
        )
        records_95, records_100 = homolog_collector.collect()
        write_solution_nmr_monomer_xray_homolog_csv(
            records=records_95,
            output_path=args.solution_nmr_monomer_xray_homolog_95_output,
        )
        write_solution_nmr_monomer_xray_homolog_csv(
            records=records_100,
            output_path=args.solution_nmr_monomer_xray_homolog_100_output,
        )
        LOGGER.info(
            "Saved %d records to %s",
            len(records_95),
            args.solution_nmr_monomer_xray_homolog_95_output,
        )
        LOGGER.info(
            "Saved %d records to %s",
            len(records_100),
            args.solution_nmr_monomer_xray_homolog_100_output,
        )

    if DatasetKind.SOLUTION_NMR_MONOMER_XRAY_RMSD in args.datasets:
        existing_records: list[SolutionNMRMonomerXrayRmsdRecord] = []
        valid_existing_records: list[SolutionNMRMonomerXrayRmsdRecord] = []
        skip_entry_ids: set[str] = set()
        xray_rmsd_output_path = Path(args.solution_nmr_monomer_xray_rmsd_output)
        if (
            not args.xray_rmsd_overwrite
            and xray_rmsd_output_path.exists()
        ):
            existing_records = read_solution_nmr_monomer_xray_rmsd_csv(
                xray_rmsd_output_path
            )
            existing_records = [
                record
                for record in existing_records
                if record.sequence_identity_percent == args.xray_rmsd_sequence_identity
            ]
            valid_existing_records = [
                record
                for record in existing_records
                if record.nmr_core_start_seq_id is not None
                and record.nmr_core_end_seq_id is not None
                and record.xray_core_start_seq_id is not None
                and record.xray_core_end_seq_id is not None
            ]
            dropped_existing = len(existing_records) - len(valid_existing_records)
            skip_entry_ids = {record.entry_id for record in valid_existing_records}
            LOGGER.info(
                "SOLUTION NMR X-ray RMSD %d%%: loaded %d existing records for resume (outdated=%d)",
                args.xray_rmsd_sequence_identity,
                len(valid_existing_records),
                dropped_existing,
            )

        rmsd_collector = SolutionNMRMonomerXrayRmsdCollector(
            client=client,
            config=config,
            cache_dir=Path(args.xray_rmsd_cache_dir),
            rmsd_workers=args.xray_rmsd_workers,
            sequence_identity_percent=args.xray_rmsd_sequence_identity,
        )
        valid_existing_records = sorted(
            valid_existing_records, key=lambda r: (r.year, r.entry_id)
        )
        xray_rmsd_output_path.parent.mkdir(parents=True, exist_ok=True)
        with xray_rmsd_output_path.open("w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(SOLUTION_NMR_MONOMER_XRAY_RMSD_HEADER)
            for record in valid_existing_records:
                writer.writerow(_solution_nmr_monomer_xray_rmsd_csv_row(record))
            csvfile.flush()

            def _on_xray_rmsd_record(record: SolutionNMRMonomerXrayRmsdRecord) -> None:
                writer.writerow(_solution_nmr_monomer_xray_rmsd_csv_row(record))
                csvfile.flush()

            new_records = rmsd_collector.collect(
                max_entries=args.xray_rmsd_max_entries,
                skip_entry_ids=skip_entry_ids,
                on_record=_on_xray_rmsd_record,
            )

        LOGGER.info(
            "Saved %d records to %s (new: %d, identity=%d%%)",
            len(valid_existing_records) + len(new_records),
            args.solution_nmr_monomer_xray_rmsd_output,
            len(new_records),
            args.xray_rmsd_sequence_identity,
        )


if __name__ == "__main__":
    main()
