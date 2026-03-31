from __future__ import annotations

import argparse
import csv
import logging
import time
import requests
import numpy as np
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
    max_workers: int = 8
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
    n_ca_core: int
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


def parse_models_ca_coords(
    pdb_path: Path,
    chain_id: str,
    start_seq_id: int | None = None,
    end_seq_id: int | None = None,
) -> list[dict[int, np.ndarray]]:

    models: list[dict[int, np.ndarray]] = []
    current_model: dict[int, np.ndarray] = {}
    has_model_records = False
    in_model = False

    with pdb_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            record = line[:6]
            if record.startswith("MODEL"):
                if in_model:
                    models.append(current_model)
                    current_model = {}
                has_model_records = True
                in_model = True
                continue
            if record.startswith("ENDMDL"):
                if in_model:
                    models.append(current_model)
                    current_model = {}
                    in_model = False
                continue
            if not record.startswith("ATOM"):
                continue

            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue
            alt_loc = line[16].strip()
            if alt_loc not in {"", "A", "1"}:
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
            if resid in current_model:
                continue
            try:
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
            except ValueError:
                continue
            current_model[resid] = np.array([x, y, z], dtype=float)

    if has_model_records:
        if in_model or current_model:
            models.append(current_model)
    elif current_model:
        models.append(current_model)
    return models


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
    def _extract_secondary_core_range(
        polymer_entity: dict[str, Any],
    ) -> tuple[int, int] | None:
        instances = polymer_entity.get("polymer_entity_instances") or []
        if len(instances) != 1:
            return None

        features = instances[0].get("rcsb_polymer_instance_feature") or []
        sec_ranges: list[tuple[int, int]] = []
        for feature in features:
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

        if not sec_ranges:
            return None

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
        self, method_label: str, query_value: str
    ) -> list[str]:
        query = {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "exptl.method",
                "operator": "exact_match",
                "value": query_value,
            },
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
                    modeled_molecular_weight_kda=modeled_weight_total_kda,
                )
            )
        return records

    def fetch_solution_nmr_monomer_secondary_records_for_ids(
        self, entry_ids: list[str]
    ) -> list[SolutionNMRMonomerSecondaryRecord]:
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
        records: list[SolutionNMRMonomerSecondaryRecord] = []

        for entry in entries:
            if not entry:
                continue
            context = self._extract_solution_nmr_monomer_context(entry)
            if context is None:
                continue
            entry_id, year, model_count, polymer_entity, chain_id = context
            entity_poly = polymer_entity.get("entity_poly") or {}

            sequence_length = entity_poly.get("rcsb_sample_sequence_length")
            if sequence_length is None or int(sequence_length) <= 0:
                continue
            sequence_length = int(sequence_length)

            instances = polymer_entity.get("polymer_entity_instances") or []
            if len(instances) != 1:
                continue
            feature_summary = (
                instances[0].get("rcsb_polymer_instance_feature_summary") or []
            )
            coverage_by_type: dict[str, float] = {}
            for item in feature_summary:
                if not item:
                    continue
                feature_type = item.get("type")
                coverage = item.get("coverage")
                if feature_type and coverage is not None:
                    coverage_by_type[str(feature_type)] = float(coverage)

            required_feature_types = {
                "UNASSIGNED_SEC_STRUCT",
            }
            if not required_feature_types.issubset(coverage_by_type):
                continue

            helix_fraction = coverage_by_type.get("HELIX_P", 0.0)
            sheet_fraction = coverage_by_type.get("SHEET", 0.0)
            unassigned_fraction = coverage_by_type.get("UNASSIGNED_SEC_STRUCT", 0.0)
            secondary_fraction = min(1.0, max(0.0, 1.0 - unassigned_fraction))

            records.append(
                SolutionNMRMonomerSecondaryRecord(
                    entry_id=entry_id,
                    year=year,
                    sequence_length=sequence_length,
                    secondary_structure_percent=secondary_fraction * 100.0,
                    helix_fraction=helix_fraction,
                    sheet_fraction=sheet_fraction,
                    deposited_model_count=model_count,
                )
            )
        return records

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
    def __init__(self, client: RCSBClient, config: CollectorConfig) -> None:
        self.client = client
        self.config = config

    def collect(self) -> list[SolutionNMRMonomerSecondaryRecord]:
        entry_ids = fetch_solution_nmr_entry_ids(
            client=self.client,
            log_label="SOLUTION NMR monomer-secondary",
        )
        batches = list(chunked(entry_ids, self.config.graphql_batch_size))
        records: list[SolutionNMRMonomerSecondaryRecord] = []

        for batch_records in collect_batch_results(
            batches=batches,
            max_workers=self.config.max_workers,
            fetch_fn=self.client.fetch_solution_nmr_monomer_secondary_records_for_ids,
            progress_label="SOLUTION NMR monomer-secondary",
        ):
            records.extend(batch_records)
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
    ) -> tuple[int, int, float] | None:
        model_maps = parse_models_ca_coords(
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
        reference_coords = coords[0]
        per_model_rmsd = [
            _superposed_rmsd(model_coord, reference_coords) for model_coord in coords
        ]

        return len(model_maps), len(sorted_resids), float(np.mean(per_model_rmsd))

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
            n_models, n_ca_core, mean_rmsd = result
            return SolutionNMRMonomerPrecisionRecord(
                entry_id=core.entry_id,
                year=core.year,
                chain_id=core.chain_id,
                core_start_seq_id=core.core_start_seq_id,
                core_end_seq_id=core.core_end_seq_id,
                n_models=n_models,
                n_ca_core=n_ca_core,
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
            )
            for r in records
        ),
    )


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
            records.append(
                SolutionNMRMonomerPrecisionRecord(
                    entry_id=str(row["entry_id"]),
                    year=int(row["year"]),
                    chain_id=str(row["chain_id"]),
                    core_start_seq_id=int(row["core_start_seq_id"]),
                    core_end_seq_id=int(row["core_end_seq_id"]),
                    n_models=int(row["n_models"]),
                    n_ca_core=int(row["n_ca_core"]),
                    mean_rmsd_angstrom=float(row["mean_rmsd_angstrom"]),
                )
            )
    return records


def write_solution_nmr_monomer_precision_csv(
    records: list[SolutionNMRMonomerPrecisionRecord], output_path: Path
) -> None:
    write_csv_rows(
        output_path=output_path,
        header=[
            "entry_id",
            "year",
            "chain_id",
            "core_start_seq_id",
            "core_end_seq_id",
            "n_models",
            "n_ca_core",
            "mean_rmsd_angstrom",
        ],
        rows=(
            (
                r.entry_id,
                r.year,
                r.chain_id,
                r.core_start_seq_id,
                r.core_end_seq_id,
                r.n_models,
                r.n_ca_core,
                f"{r.mean_rmsd_angstrom:.4f}",
            )
            for r in records
        ),
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


def write_solution_nmr_monomer_xray_rmsd_csv(
    records: list[SolutionNMRMonomerXrayRmsdRecord], output_path: Path
) -> None:
    write_csv_rows(
        output_path=output_path,
        header=[
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
        ],
        rows=(
            (
                r.entry_id,
                r.year,
                r.sequence_identity_percent,
                r.nmr_chain_id,
                r.nmr_core_start_seq_id if r.nmr_core_start_seq_id is not None else "",
                r.nmr_core_end_seq_id if r.nmr_core_end_seq_id is not None else "",
                r.xray_entry_id,
                r.xray_chain_id,
                (
                    r.xray_core_start_seq_id
                    if r.xray_core_start_seq_id is not None
                    else ""
                ),
                r.xray_core_end_seq_id if r.xray_core_end_seq_id is not None else "",
                f"{r.xray_resolution_angstrom:.4f}",
                r.n_common_ca,
                f"{r.rmsd_ca_angstrom:.4f}",
            )
            for r in records
        ),
    )


def parse_dataset_kinds(raw_value: str) -> list[DatasetKind]:
    if raw_value.strip().lower() == "all":
        return [
            DatasetKind.METHOD_COUNTS,
            DatasetKind.MEMBRANE_PROTEIN_COUNTS,
            DatasetKind.SOLUTION_NMR_WEIGHTS,
            DatasetKind.SOLUTION_NMR_MONOMER_SECONDARY,
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
        "Available: method_counts, membrane_protein_counts, solution_nmr_weights, solution_nmr_monomer_secondary, solution_nmr_monomer_precision, solution_nmr_monomer_quality, solution_nmr_monomer_xray_homologs, solution_nmr_monomer_xray_rmsd (default: the first three).",
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
        default=4,
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
        default=4,
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
        "--workers", type=int, default=8, help="Parallel workers for GraphQL calls."
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
        sec_collector = SolutionNMRMonomerSecondaryCollector(
            client=client, config=config
        )
        sec_records = sec_collector.collect()
        write_solution_nmr_monomer_secondary_csv(
            records=sec_records, output_path=args.solution_nmr_monomer_secondary_output
        )
        LOGGER.info(
            "Saved %d records to %s",
            len(sec_records),
            args.solution_nmr_monomer_secondary_output,
        )

    if DatasetKind.SOLUTION_NMR_MONOMER_PRECISION in args.datasets:
        existing_records: list[SolutionNMRMonomerPrecisionRecord] = []
        skip_entry_ids: set[str] = set()
        if (
            not args.precision_overwrite
            and Path(args.solution_nmr_monomer_precision_output).exists()
        ):
            existing_records = read_solution_nmr_monomer_precision_csv(
                Path(args.solution_nmr_monomer_precision_output)
            )
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
        new_records = precision_collector.collect(
            max_entries=args.precision_max_entries,
            skip_entry_ids=skip_entry_ids,
        )
        combined_records = sorted(
            existing_records + new_records,
            key=lambda record: (record.year, record.entry_id),
        )
        write_solution_nmr_monomer_precision_csv(
            records=combined_records,
            output_path=args.solution_nmr_monomer_precision_output,
        )
        LOGGER.info(
            "Saved %d records to %s (new: %d)",
            len(combined_records),
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
        skip_entry_ids: set[str] = set()
        if (
            not args.xray_rmsd_overwrite
            and Path(args.solution_nmr_monomer_xray_rmsd_output).exists()
        ):
            existing_records = read_solution_nmr_monomer_xray_rmsd_csv(
                Path(args.solution_nmr_monomer_xray_rmsd_output)
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
        new_records = rmsd_collector.collect(
            max_entries=args.xray_rmsd_max_entries,
            skip_entry_ids=skip_entry_ids,
        )
        combined_by_entry: dict[str, SolutionNMRMonomerXrayRmsdRecord] = {
            record.entry_id: record for record in existing_records
        }
        for record in new_records:
            combined_by_entry[record.entry_id] = record
        combined_records = sorted(
            combined_by_entry.values(),
            key=lambda record: (record.year, record.entry_id),
        )
        write_solution_nmr_monomer_xray_rmsd_csv(
            records=combined_records,
            output_path=args.solution_nmr_monomer_xray_rmsd_output,
        )
        LOGGER.info(
            "Saved %d records to %s (new: %d, identity=%d%%)",
            len(combined_records),
            args.solution_nmr_monomer_xray_rmsd_output,
            len(new_records),
            args.xray_rmsd_sequence_identity,
        )


if __name__ == "__main__":
    main()
