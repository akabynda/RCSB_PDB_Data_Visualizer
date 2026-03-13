from __future__ import annotations

import argparse
import csv
import logging
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Iterator

import requests


LOGGER = logging.getLogger(__name__)


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
    SOLUTION_NMR_WEIGHTS = "solution_nmr_weights"
    SOLUTION_NMR_MONOMER_SECONDARY = "solution_nmr_monomer_secondary"
    SOLUTION_NMR_MONOMER_PRECISION = "solution_nmr_monomer_precision"
    SOLUTION_NMR_MONOMER_QUALITY = "solution_nmr_monomer_quality"
    SOLUTION_NMR_MONOMER_XRAY_HOMOLOGS = "solution_nmr_monomer_xray_homologs"


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
class SolutionNMRWeightRecord:
    entry_id: str
    year: int
    molecular_weight_kda: float


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


class RCSBClient:
    def __init__(self, config: CollectorConfig) -> None:
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "pdb-extensible-collector/1.0"})

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
        all_ids: list[str] = []
        start = 0
        total_count: int | None = None

        while total_count is None or start < total_count:
            payload = {
                "query": {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "exptl.method",
                        "operator": "exact_match",
                        "value": query_value,
                    },
                },
                "return_type": "entry",
                "request_options": {
                    "paginate": {"start": start, "rows": self.config.page_size}
                },
            }
            data = self._post_json(self.config.search_url, payload)
            total_count = data.get("total_count", 0)
            result_set = data.get("result_set", [])
            ids = [item["identifier"] for item in result_set if "identifier" in item]
            all_ids.extend(ids)
            start += len(ids)
            if not ids:
                break
            LOGGER.info(
                "%s (%s): fetched %d/%d entry IDs",
                method_label,
                query_value,
                len(all_ids),
                total_count,
            )
        return all_ids

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

    def fetch_xray_polymer_entity_ids_for_group_ids(
        self, group_ids: list[str]
    ) -> list[str]:
        if not group_ids:
            return []
        all_ids: list[str] = []
        start = 0
        total_count: int | None = None
        while total_count is None or start < total_count:
            payload = {
                "query": {
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
                },
                "return_type": "polymer_entity",
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
        return all_ids

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
            for membership in memberships:
                if not membership:
                    continue
                if membership.get("aggregation_method") != "sequence_identity":
                    continue
                raw_cutoff = membership.get("similarity_cutoff")
                if raw_cutoff is None or int(round(float(raw_cutoff))) != similarity_cutoff:
                    continue
                group_id = membership.get("group_id")
                if group_id:
                    matching_group_ids.add(str(group_id))
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
            polymer_entities {
              entity_poly {
                rcsb_entity_polymer_type
              }
              rcsb_polymer_entity {
                formula_weight
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

            total_weight_kda = 0.0
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
                    total_weight_kda += float(entity_weight)

            records.append(
                SolutionNMRWeightRecord(
                    entry_id=entry_id,
                    year=year,
                    molecular_weight_kda=total_weight_kda,
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
            entry_id = entry.get("rcsb_id")
            if not entry_id:
                continue

            model_count = entry.get("rcsb_entry_info", {}).get("deposited_model_count")
            if model_count is None or int(model_count) <= 1:
                continue

            year = extract_year(
                entry.get("rcsb_accession_info", {}).get("deposit_date")
            )
            if year is None:
                continue

            polymer_entities = entry.get("polymer_entities") or []
            if len(polymer_entities) != 1:
                continue
            polymer_entity = polymer_entities[0] or {}

            entity_poly = polymer_entity.get("entity_poly") or {}
            if entity_poly.get("type") not in {"polypeptide(L)", "polypeptide(D)"}:
                continue

            if entity_poly.get("rcsb_entity_polymer_type") != "Protein":
                continue

            strand_id = str(entity_poly.get("pdbx_strand_id") or "").strip()
            if not strand_id or "," in strand_id:
                continue

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

            helix_fraction = coverage_by_type.get("HELIX_P", 0.0)
            sheet_fraction = coverage_by_type.get("SHEET", 0.0)
            secondary_fraction = min(1.0, max(0.0, helix_fraction + sheet_fraction))

            records.append(
                SolutionNMRMonomerSecondaryRecord(
                    entry_id=entry_id,
                    year=year,
                    sequence_length=sequence_length,
                    secondary_structure_percent=secondary_fraction * 100.0,
                    helix_fraction=helix_fraction,
                    sheet_fraction=sheet_fraction,
                    deposited_model_count=int(model_count),
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
            entry_id = entry.get("rcsb_id")
            if not entry_id:
                continue

            model_count = entry.get("rcsb_entry_info", {}).get("deposited_model_count")
            if model_count is None or int(model_count) <= 1:
                continue

            year = extract_year(
                entry.get("rcsb_accession_info", {}).get("deposit_date")
            )
            if year is None:
                continue

            polymer_entities = entry.get("polymer_entities") or []
            if len(polymer_entities) != 1:
                continue
            polymer_entity = polymer_entities[0] or {}
            entity_poly = polymer_entity.get("entity_poly") or {}

            if entity_poly.get("type") not in {"polypeptide(L)", "polypeptide(D)"}:
                continue
            if entity_poly.get("rcsb_entity_polymer_type") != "Protein":
                continue

            chain_id = str(entity_poly.get("pdbx_strand_id") or "").strip()
            if not chain_id or "," in chain_id:
                continue

            instances = polymer_entity.get("polymer_entity_instances") or []
            if len(instances) != 1:
                continue

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
                    beg_i = int(beg)
                    end_i = int(end) if end is not None else beg_i
                    sec_ranges.append((beg_i, end_i))

            if not sec_ranges:
                continue

            core_start = min(beg for beg, _ in sec_ranges)
            core_end = max(end for _, end in sec_ranges)
            if core_end < core_start:
                continue

            records.append(
                SolutionNMRMonomerCoreRegionRecord(
                    entry_id=entry_id,
                    year=year,
                    chain_id=chain_id,
                    core_start_seq_id=core_start,
                    core_end_seq_id=core_end,
                    deposited_model_count=int(model_count),
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
            entry_id = entry.get("rcsb_id")
            if not entry_id:
                continue

            model_count = entry.get("rcsb_entry_info", {}).get("deposited_model_count")
            if model_count is None or int(model_count) <= 1:
                continue

            year = extract_year(
                entry.get("rcsb_accession_info", {}).get("deposit_date")
            )
            if year is None:
                continue

            polymer_entities = entry.get("polymer_entities") or []
            if len(polymer_entities) != 1:
                continue
            entity_poly = (polymer_entities[0] or {}).get("entity_poly") or {}
            if entity_poly.get("type") not in {"polypeptide(L)", "polypeptide(D)"}:
                continue
            if entity_poly.get("rcsb_entity_polymer_type") != "Protein":
                continue
            strand_id = str(entity_poly.get("pdbx_strand_id") or "").strip()
            if not strand_id or "," in strand_id:
                continue

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
            entry_id = entry.get("rcsb_id")
            if not entry_id:
                continue

            model_count = entry.get("rcsb_entry_info", {}).get("deposited_model_count")
            if model_count is None or int(model_count) <= 1:
                continue

            year = extract_year(entry.get("rcsb_accession_info", {}).get("deposit_date"))
            if year is None:
                continue

            polymer_entities = entry.get("polymer_entities") or []
            if len(polymer_entities) != 1:
                continue
            entity_poly = (polymer_entities[0] or {}).get("entity_poly") or {}
            if entity_poly.get("type") not in {"polypeptide(L)", "polypeptide(D)"}:
                continue
            if entity_poly.get("rcsb_entity_polymer_type") != "Protein":
                continue
            strand_id = str(entity_poly.get("pdbx_strand_id") or "").strip()
            if not strand_id or "," in strand_id:
                continue

            memberships = (
                (polymer_entities[0] or {}).get("rcsb_polymer_entity_group_membership")
                or []
            )
            group_95: str | None = None
            group_100: str | None = None
            for membership in memberships:
                if not membership:
                    continue
                if membership.get("aggregation_method") != "sequence_identity":
                    continue
                group_id = membership.get("group_id")
                cutoff = membership.get("similarity_cutoff")
                if not group_id or cutoff is None:
                    continue
                cutoff_value = int(round(float(cutoff)))
                if cutoff_value == 95:
                    group_95 = str(group_id)
                elif cutoff_value == 100:
                    group_100 = str(group_id)

            records.append((str(entry_id), year, group_95, group_100))
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
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_map = {
                executor.submit(self.client.fetch_deposit_dates_for_ids, batch): idx
                for idx, batch in enumerate(batches, start=1)
            }
            for future in as_completed(future_map):
                batch_dates = future.result()
                years = filter(
                    None, (extract_year(date_value) for date_value in batch_dates)
                )
                year_counter.update(years)
                batch_idx = future_map[future]
                LOGGER.info(
                    "%s: processed batch %d/%d", method.label, batch_idx, len(batches)
                )

        return [
            YearlyCountRecord(year=year, method=method.label, count=count)
            for year, count in sorted(year_counter.items())
        ]

    def collect(self, methods: Iterable[ExperimentalMethod]) -> list[YearlyCountRecord]:
        records: list[YearlyCountRecord] = []
        for method in methods:
            records.extend(self._fetch_method_records(method))
        return sorted(records, key=lambda record: (record.year, record.method))


class SolutionNMRWeightCollector:
    def __init__(self, client: RCSBClient, config: CollectorConfig) -> None:
        self.client = client
        self.config = config

    def collect(self) -> list[SolutionNMRWeightRecord]:
        entry_ids = sorted(
            set(
                self.client.fetch_entry_ids_for_method(
                    method_label="SOLUTION NMR", query_value="SOLUTION NMR"
                )
            )
        )
        LOGGER.info("SOLUTION NMR: total unique IDs collected: %d", len(entry_ids))
        batches = list(chunked(entry_ids, self.config.graphql_batch_size))

        records: list[SolutionNMRWeightRecord] = []
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_map = {
                executor.submit(
                    self.client.fetch_solution_nmr_weight_records_for_ids, batch
                ): idx
                for idx, batch in enumerate(batches, start=1)
            }
            for future in as_completed(future_map):
                batch_records = future.result()
                records.extend(batch_records)
                batch_idx = future_map[future]
                LOGGER.info(
                    "SOLUTION NMR weights: processed batch %d/%d",
                    batch_idx,
                    len(batches),
                )
        return sorted(records, key=lambda record: (record.year, record.entry_id))


class SolutionNMRMonomerSecondaryCollector:
    def __init__(self, client: RCSBClient, config: CollectorConfig) -> None:
        self.client = client
        self.config = config

    def collect(self) -> list[SolutionNMRMonomerSecondaryRecord]:
        entry_ids = sorted(
            set(
                self.client.fetch_entry_ids_for_method(
                    method_label="SOLUTION NMR", query_value="SOLUTION NMR"
                )
            )
        )
        LOGGER.info(
            "SOLUTION NMR monomer-secondary: total unique IDs collected: %d",
            len(entry_ids),
        )
        batches = list(chunked(entry_ids, self.config.graphql_batch_size))
        records: list[SolutionNMRMonomerSecondaryRecord] = []

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_map = {
                executor.submit(
                    self.client.fetch_solution_nmr_monomer_secondary_records_for_ids,
                    batch,
                ): idx
                for idx, batch in enumerate(batches, start=1)
            }
            for future in as_completed(future_map):
                batch_records = future.result()
                records.extend(batch_records)
                batch_idx = future_map[future]
                LOGGER.info(
                    "SOLUTION NMR monomer-secondary: processed batch %d/%d",
                    batch_idx,
                    len(batches),
                )
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
        path = self.cache_dir / f"{entry_id}.pdb"
        if path.exists() and path.stat().st_size > 0:
            return path

        url = f"https://files.rcsb.org/download/{entry_id}.pdb"
        last_error: Exception | None = None
        for attempt in range(1, self.config.retries + 1):
            try:
                response = self.client.session.get(
                    url, timeout=self.config.timeout_seconds
                )
                response.raise_for_status()
                path.write_text(response.text, encoding="utf-8")
                return path
            except (requests.RequestException, OSError) as exc:
                last_error = exc
                wait_seconds = self.config.backoff_seconds * attempt
                if attempt < self.config.retries:
                    time.sleep(wait_seconds)
        raise RuntimeError(f"Failed to download {entry_id}: {last_error}")

    @staticmethod
    def _compute_mean_rmsd_to_average(
        pdb_path: Path, chain_id: str, start_seq_id: int, end_seq_id: int
    ) -> tuple[int, int, float] | None:
        import numpy as np

        def parse_models_ca_coords() -> list[dict[int, np.ndarray]]:
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
                    if resid < start_seq_id or resid > end_seq_id:
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

        def kabsch_rmsd(reference: np.ndarray, mobile: np.ndarray) -> float:
            ref_centered = reference - reference.mean(axis=0)
            mob_centered = mobile - mobile.mean(axis=0)
            covariance = mob_centered.T @ ref_centered
            v_mat, _, w_t = np.linalg.svd(covariance)
            det_sign = np.sign(np.linalg.det(v_mat @ w_t))
            correction = np.diag([1.0, 1.0, det_sign])
            rotation = v_mat @ correction @ w_t
            aligned = mob_centered @ rotation
            diff = aligned - ref_centered
            return float(np.sqrt((diff * diff).sum() / reference.shape[0]))

        model_maps = parse_models_ca_coords()
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
        mean_coords = coords.mean(axis=0)
        per_model_rmsd = [
            kabsch_rmsd(mean_coords, model_coord) for model_coord in coords
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
        entry_ids = sorted(
            set(
                self.client.fetch_entry_ids_for_method(
                    method_label="SOLUTION NMR", query_value="SOLUTION NMR"
                )
            )
        )
        LOGGER.info(
            "SOLUTION NMR precision: total unique IDs collected: %d", len(entry_ids)
        )

        batches = list(chunked(entry_ids, self.config.graphql_batch_size))
        core_records: list[SolutionNMRMonomerCoreRegionRecord] = []
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_map = {
                executor.submit(
                    self.client.fetch_solution_nmr_monomer_core_region_records_for_ids,
                    batch,
                ): idx
                for idx, batch in enumerate(batches, start=1)
            }
            for future in as_completed(future_map):
                batch_core = future.result()
                core_records.extend(batch_core)
                batch_idx = future_map[future]
                LOGGER.info(
                    "SOLUTION NMR precision core ranges: processed batch %d/%d",
                    batch_idx,
                    len(batches),
                )

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
        entry_ids = sorted(
            set(
                self.client.fetch_entry_ids_for_method(
                    method_label="SOLUTION NMR", query_value="SOLUTION NMR"
                )
            )
        )
        LOGGER.info(
            "SOLUTION NMR quality: total unique IDs collected: %d", len(entry_ids)
        )
        batches = list(chunked(entry_ids, self.config.graphql_batch_size))
        records: list[SolutionNMRMonomerQualityRecord] = []

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_map = {
                executor.submit(
                    self.client.fetch_solution_nmr_monomer_quality_records_for_ids,
                    batch,
                ): idx
                for idx, batch in enumerate(batches, start=1)
            }
            for future in as_completed(future_map):
                records.extend(future.result())
                batch_idx = future_map[future]
                LOGGER.info(
                    "SOLUTION NMR quality: processed batch %d/%d",
                    batch_idx,
                    len(batches),
                )
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
                    matched = (
                        self.client.fetch_sequence_identity_group_ids_for_polymer_entity_ids(
                            entity_batch, similarity_cutoff=similarity_cutoff
                        )
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
        entry_ids = sorted(
            set(
                self.client.fetch_entry_ids_for_method(
                    method_label="SOLUTION NMR", query_value="SOLUTION NMR"
                )
            )
        )
        LOGGER.info(
            "SOLUTION NMR monomer X-ray homologs: total unique IDs collected: %d",
            len(entry_ids),
        )
        batches = list(chunked(entry_ids, self.config.graphql_batch_size))
        base_rows: list[tuple[str, int, str | None, str | None]] = []

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_map = {
                executor.submit(
                    self.client.fetch_solution_nmr_monomer_xray_group_records_for_ids,
                    batch,
                ): idx
                for idx, batch in enumerate(batches, start=1)
            }
            for future in as_completed(future_map):
                base_rows.extend(future.result())
                batch_idx = future_map[future]
                LOGGER.info(
                    "SOLUTION NMR monomer X-ray homolog base: processed batch %d/%d",
                    batch_idx,
                    len(batches),
                )

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


def write_method_counts_csv(
    records: list[YearlyCountRecord], output_path: Path
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["year", "method", "count"])
        writer.writerows((r.year, r.method, r.count) for r in records)


def write_solution_nmr_weights_csv(
    records: list[SolutionNMRWeightRecord], output_path: Path
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["entry_id", "year", "molecular_weight_kda"])
        writer.writerows(
            (r.entry_id, r.year, f"{r.molecular_weight_kda:.3f}") for r in records
        )


def write_solution_nmr_monomer_secondary_csv(
    records: list[SolutionNMRMonomerSecondaryRecord], output_path: Path
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
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
            ]
        )
        writer.writerows(
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "entry_id",
                "year",
                "chain_id",
                "core_start_seq_id",
                "core_end_seq_id",
                "n_models",
                "n_ca_core",
                "mean_rmsd_angstrom",
            ]
        )
        writer.writerows(
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
        )


def write_solution_nmr_monomer_quality_csv(
    records: list[SolutionNMRMonomerQualityRecord], output_path: Path
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "entry_id",
                "year",
                "clashscore",
                "ramachandran_outliers_percent",
                "sidechain_outliers_percent",
            ]
        )
        writer.writerows(
            (
                r.entry_id,
                r.year,
                f"{r.clashscore:.4f}",
                f"{r.ramachandran_outliers_percent:.4f}",
                f"{r.sidechain_outliers_percent:.4f}",
            )
            for r in records
        )


def write_solution_nmr_monomer_xray_homolog_csv(
    records: list[SolutionNMRMonomerXrayHomologRecord], output_path: Path
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "entry_id",
                "year",
                "sequence_identity_percent",
                "group_id",
                "has_xray_homolog",
            ]
        )
        writer.writerows(
            (
                r.entry_id,
                r.year,
                r.sequence_identity_percent,
                r.group_id or "",
                int(r.has_xray_homolog),
            )
            for r in records
        )


def parse_dataset_kinds(raw_value: str) -> list[DatasetKind]:
    if raw_value.strip().lower() == "all":
        return [
            DatasetKind.METHOD_COUNTS,
            DatasetKind.SOLUTION_NMR_WEIGHTS,
            DatasetKind.SOLUTION_NMR_MONOMER_SECONDARY,
            DatasetKind.SOLUTION_NMR_MONOMER_PRECISION,
            DatasetKind.SOLUTION_NMR_MONOMER_QUALITY,
            DatasetKind.SOLUTION_NMR_MONOMER_XRAY_HOMOLOGS,
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
        "Available: method_counts, solution_nmr_weights, solution_nmr_monomer_secondary, solution_nmr_monomer_precision, solution_nmr_monomer_quality, solution_nmr_monomer_xray_homologs (default: the first three).",
    )
    parser.add_argument(
        "--counts-output",
        type=Path,
        default=Path("data/pdb_method_counts_by_year.csv"),
        help="Output CSV path for method_counts dataset.",
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


if __name__ == "__main__":
    main()
