from __future__ import annotations

import argparse
import csv
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pdb_data_collector import (
    DEFAULT_PDB_CACHE_DIR,
    CollectorConfig,
    RCSBClient,
    SOLUTION_NMR_METHOD,
    parse_models_ca_coords_with_stats,
)

LOGGER = logging.getLogger(__name__)

AUTH_MAPPING_AUDIT_HEADER: tuple[str, ...] = (
    "entry_id",
    "year",
    "chain_id",
    "sequence_length",
    "auth_mapping_length",
    "auth_mapping_valid_count",
    "auth_mapping_invalid_count",
    "auth_mapping_duplicate_count",
    "auth_mapping_monotonic_non_decreasing",
    "auth_mapping_monotonic_non_increasing",
    "auth_mapping_direction",
    "modeled_label_count",
    "modeled_auth_count",
    "first_model_ca_count",
    "overlap_count",
    "mapped_auth_recall",
    "first_model_ca_precision",
    "exact_match",
    "mapped_auth_min",
    "mapped_auth_max",
    "first_model_ca_min",
    "first_model_ca_max",
    "missing_in_first_model_examples",
    "extra_in_first_model_examples",
)


@dataclass(frozen=True)
class AuthMappingAuditRecord:
    entry_id: str
    year: int
    chain_id: str
    sequence_length: int
    auth_mapping_length: int
    auth_mapping_valid_count: int
    auth_mapping_invalid_count: int
    auth_mapping_duplicate_count: int
    auth_mapping_monotonic_non_decreasing: bool
    auth_mapping_monotonic_non_increasing: bool
    auth_mapping_direction: str
    modeled_label_count: int
    modeled_auth_count: int
    first_model_ca_count: int
    overlap_count: int
    mapped_auth_recall: float
    first_model_ca_precision: float
    exact_match: bool
    mapped_auth_min: int | None
    mapped_auth_max: int | None
    first_model_ca_min: int | None
    first_model_ca_max: int | None
    missing_in_first_model_examples: str
    extra_in_first_model_examples: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit how well RCSB auth_to_entity_poly_seq_mapping agrees with "
            "real auth residue numbering in locally cached PDB files."
        )
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_PDB_CACHE_DIR,
        help="Directory with locally cached PDB files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/auth_mapping_audit.csv"),
        help="Output CSV path for per-entry audit records.",
    )
    parser.add_argument(
        "--max-audited-entries",
        type=int,
        default=200,
        help=(
            "Stop after this many qualifying SOLUTION NMR monomer entries have "
            "been audited. Use 0 or a negative value to audit all qualifying "
            "entries in the cache."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=300,
        help="GraphQL batch size for entry metadata requests.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=60,
        help="HTTP timeout for GraphQL requests.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG/INFO/WARNING/ERROR).",
    )
    return parser.parse_args()


def _cached_entry_ids(cache_dir: Path) -> list[str]:
    if not cache_dir.exists():
        return []
    return sorted(
        {
            pdb_path.stem.strip().upper()
            for pdb_path in cache_dir.glob("*.pdb")
            if pdb_path.is_file() and pdb_path.stat().st_size > 0
        }
    )


def _chunked(values: list[str], chunk_size: int) -> list[list[str]]:
    return [
        values[idx : idx + chunk_size] for idx in range(0, len(values), max(1, chunk_size))
    ]


def _is_exact_solution_nmr_entry(entry: dict[str, object] | None) -> bool:
    if not entry:
        return False
    methods = {
        str(exptl.get("method")).strip()
        for exptl in (entry.get("exptl") or [])
        if isinstance(exptl, dict) and exptl.get("method")
    }
    return methods == {SOLUTION_NMR_METHOD}


def _valid_mapping_values(auth_mapping_raw: object) -> list[int]:
    if not isinstance(auth_mapping_raw, list):
        return []
    values: list[int] = []
    for raw_value in auth_mapping_raw:
        try:
            values.append(int(str(raw_value).strip()))
        except (TypeError, ValueError):
            continue
    return values


def _example_values(values: set[int], limit: int = 10) -> str:
    if not values:
        return ""
    subset = sorted(values)[: max(1, limit)]
    return ",".join(str(value) for value in subset)


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _mapping_direction(values: list[int]) -> tuple[bool, bool, str]:
    non_decreasing = all(
        left <= right for left, right in zip(values, values[1:], strict=False)
    )
    non_increasing = all(
        left >= right for left, right in zip(values, values[1:], strict=False)
    )
    if non_decreasing and non_increasing:
        return non_decreasing, non_increasing, "constant"
    if non_decreasing:
        return non_decreasing, non_increasing, "non_decreasing"
    if non_increasing:
        return non_decreasing, non_increasing, "non_increasing"
    return non_decreasing, non_increasing, "mixed"


def _audit_record_from_entry(
    client: RCSBClient,
    cache_dir: Path,
    entry: dict[str, object] | None,
) -> AuthMappingAuditRecord | None:
    if not _is_exact_solution_nmr_entry(entry):
        return None

    context = client._extract_solution_nmr_monomer_context(entry)
    if context is None:
        return None

    entry_id, year, _model_count, polymer_entity, chain_id = context
    sequence_length_raw = (polymer_entity.get("entity_poly") or {}).get(
        "rcsb_sample_sequence_length"
    )
    try:
        sequence_length = int(sequence_length_raw)
    except (TypeError, ValueError):
        return None
    if sequence_length <= 0:
        return None

    instances = polymer_entity.get("polymer_entity_instances") or []
    if len(instances) != 1:
        return None
    instance = instances[0] or {}

    modeled_label_seq_ids, modeled_auth_seq_ids = (
        client._extract_modeled_residue_sets_for_instance(
            sequence_length=sequence_length,
            instance=instance,
        )
    )
    if not modeled_label_seq_ids:
        return None

    pdb_path = cache_dir / f"{entry_id}.pdb"
    if not pdb_path.exists():
        return None

    model_maps, _raw_counts = parse_models_ca_coords_with_stats(
        pdb_path=pdb_path,
        chain_id=chain_id,
    )
    if not model_maps:
        return None
    first_model_ca_seq_ids = set(model_maps[0].keys())

    identifiers = instance.get("rcsb_polymer_entity_instance_container_identifiers") or {}
    auth_mapping_raw = identifiers.get("auth_to_entity_poly_seq_mapping") or []
    valid_mapping_values = _valid_mapping_values(auth_mapping_raw)
    auth_mapping_length = len(auth_mapping_raw) if isinstance(auth_mapping_raw, list) else 0
    auth_mapping_valid_count = len(valid_mapping_values)
    auth_mapping_invalid_count = max(0, auth_mapping_length - auth_mapping_valid_count)
    auth_mapping_duplicate_count = max(
        0, auth_mapping_valid_count - len(set(valid_mapping_values))
    )
    (
        auth_mapping_monotonic_non_decreasing,
        auth_mapping_monotonic_non_increasing,
        auth_mapping_direction,
    ) = _mapping_direction(valid_mapping_values)

    overlap = modeled_auth_seq_ids & first_model_ca_seq_ids
    missing_in_first_model = modeled_auth_seq_ids - first_model_ca_seq_ids
    extra_in_first_model = first_model_ca_seq_ids - modeled_auth_seq_ids

    return AuthMappingAuditRecord(
        entry_id=entry_id,
        year=year,
        chain_id=chain_id,
        sequence_length=sequence_length,
        auth_mapping_length=auth_mapping_length,
        auth_mapping_valid_count=auth_mapping_valid_count,
        auth_mapping_invalid_count=auth_mapping_invalid_count,
        auth_mapping_duplicate_count=auth_mapping_duplicate_count,
        auth_mapping_monotonic_non_decreasing=auth_mapping_monotonic_non_decreasing,
        auth_mapping_monotonic_non_increasing=auth_mapping_monotonic_non_increasing,
        auth_mapping_direction=auth_mapping_direction,
        modeled_label_count=len(modeled_label_seq_ids),
        modeled_auth_count=len(modeled_auth_seq_ids),
        first_model_ca_count=len(first_model_ca_seq_ids),
        overlap_count=len(overlap),
        mapped_auth_recall=_safe_ratio(len(overlap), len(modeled_auth_seq_ids)),
        first_model_ca_precision=_safe_ratio(len(overlap), len(first_model_ca_seq_ids)),
        exact_match=modeled_auth_seq_ids == first_model_ca_seq_ids,
        mapped_auth_min=min(modeled_auth_seq_ids) if modeled_auth_seq_ids else None,
        mapped_auth_max=max(modeled_auth_seq_ids) if modeled_auth_seq_ids else None,
        first_model_ca_min=min(first_model_ca_seq_ids) if first_model_ca_seq_ids else None,
        first_model_ca_max=max(first_model_ca_seq_ids) if first_model_ca_seq_ids else None,
        missing_in_first_model_examples=_example_values(missing_in_first_model),
        extra_in_first_model_examples=_example_values(extra_in_first_model),
    )


def _write_records(output_path: Path, records: list[AuthMappingAuditRecord]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(AUTH_MAPPING_AUDIT_HEADER)
        for record in records:
            writer.writerow(
                [
                    record.entry_id,
                    record.year,
                    record.chain_id,
                    record.sequence_length,
                    record.auth_mapping_length,
                    record.auth_mapping_valid_count,
                    record.auth_mapping_invalid_count,
                    record.auth_mapping_duplicate_count,
                    int(record.auth_mapping_monotonic_non_decreasing),
                    int(record.auth_mapping_monotonic_non_increasing),
                    record.auth_mapping_direction,
                    record.modeled_label_count,
                    record.modeled_auth_count,
                    record.first_model_ca_count,
                    record.overlap_count,
                    f"{record.mapped_auth_recall:.6f}",
                    f"{record.first_model_ca_precision:.6f}",
                    int(record.exact_match),
                    record.mapped_auth_min if record.mapped_auth_min is not None else "",
                    record.mapped_auth_max if record.mapped_auth_max is not None else "",
                    (
                        record.first_model_ca_min
                        if record.first_model_ca_min is not None
                        else ""
                    ),
                    (
                        record.first_model_ca_max
                        if record.first_model_ca_max is not None
                        else ""
                    ),
                    record.missing_in_first_model_examples,
                    record.extra_in_first_model_examples,
                ]
            )


def _fetch_entries_batch(client: RCSBClient, entry_ids: list[str]) -> list[dict[str, object] | None]:
    query = """
    query($ids:[String!]!) {
      entries(entry_ids:$ids) {
        rcsb_id
        exptl {
          method
        }
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
    data = client._post_json(client.config.graphql_url, payload)
    return data.get("data", {}).get("entries", [])


def _print_summary(records: list[AuthMappingAuditRecord]) -> None:
    if not records:
        print("No qualifying entries were audited.")
        return

    exact_match_count = sum(1 for record in records if record.exact_match)
    monotonic_count = sum(
        1
        for record in records
        if record.auth_mapping_direction in {"non_decreasing", "non_increasing", "constant"}
    )
    non_decreasing_count = sum(
        1 for record in records if record.auth_mapping_direction == "non_decreasing"
    )
    non_increasing_count = sum(
        1 for record in records if record.auth_mapping_direction == "non_increasing"
    )
    mixed_count = sum(1 for record in records if record.auth_mapping_direction == "mixed")
    recall_values = [record.mapped_auth_recall for record in records]
    precision_values = [record.first_model_ca_precision for record in records]
    worst_records = sorted(
        records,
        key=lambda record: (
            record.mapped_auth_recall,
            record.first_model_ca_precision,
            record.entry_id,
        ),
    )[:10]

    print(f"Audited entries: {len(records)}")
    print(
        "Exact first-model matches: "
        f"{exact_match_count}/{len(records)} ({exact_match_count / len(records):.1%})"
    )
    print(
        "Monotonic mappings (either direction): "
        f"{monotonic_count}/{len(records)} ({monotonic_count / len(records):.1%})"
    )
    print(
        "Direction split: "
        f"non_decreasing={non_decreasing_count}, "
        f"non_increasing={non_increasing_count}, "
        f"mixed={mixed_count}"
    )
    print(f"Mean mapped-auth recall: {mean(recall_values):.4f}")
    print(f"Mean first-model precision: {mean(precision_values):.4f}")
    print("Worst entries by recall/precision:")
    for record in worst_records:
        print(
            f"  {record.entry_id} chain {record.chain_id}: "
            f"recall={record.mapped_auth_recall:.4f}, "
            f"precision={record.first_model_ca_precision:.4f}, "
            f"missing={record.missing_in_first_model_examples or '-'}, "
            f"extra={record.extra_in_first_model_examples or '-'}"
        )


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    entry_ids = _cached_entry_ids(args.cache_dir)
    if not entry_ids:
        raise RuntimeError(f"No cached PDB files found in {args.cache_dir}")

    max_audited_entries = max(0, int(args.max_audited_entries))
    config = CollectorConfig(
        graphql_batch_size=max(1, args.batch_size),
        timeout_seconds=max(1, args.timeout_seconds),
    )
    client = RCSBClient(config=config)

    records: list[AuthMappingAuditRecord] = []
    batches = _chunked(entry_ids, config.graphql_batch_size)
    total_batches = len(batches)
    for batch_index, batch_entry_ids in enumerate(batches, start=1):
        entries = _fetch_entries_batch(client, batch_entry_ids)
        for entry in entries:
            record = _audit_record_from_entry(
                client=client,
                cache_dir=args.cache_dir,
                entry=entry,
            )
            if record is None:
                continue
            records.append(record)
            if max_audited_entries > 0 and len(records) >= max_audited_entries:
                break
        LOGGER.info(
            "Processed GraphQL batch %d/%d, audited %d qualifying entries so far",
            batch_index,
            total_batches,
            len(records),
        )
        if max_audited_entries > 0 and len(records) >= max_audited_entries:
            break

    _write_records(args.output, records)
    _print_summary(records)
    print(f"Saved audit CSV to {args.output}")


if __name__ == "__main__":
    main()
