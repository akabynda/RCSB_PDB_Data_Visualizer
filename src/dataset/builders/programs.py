"""Collect refinement-program counts and monomer cluster assignments."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING

from src.dataset.config import (
    LOGGER,
)
from src.dataset.downloads import (
    download_pdb_if_needed,
)
from src.dataset.programs import (
    PROGRAM_CLUSTER_DEFINITIONS,
    extract_raw_refinement_program_text_from_pdb,
    extract_refinement_programs_from_pdb,
    extract_solution_nmr_program_clusters,
)
from src.dataset.records import (
    SolutionNMRMonomerProgramClusterAssignmentRecord,
    SolutionNMRMonomerProgramClusterSummaryRecord,
    SolutionNMRProgramYearlyCountRecord,
)
from src.dataset.reporting import (
    _record_filtered_structure,
)
from src.dataset.utils import (
    chunked,
    collect_batch_results,
    fetch_solution_nmr_entry_ids,
)

if TYPE_CHECKING:
    from src.dataset.client import (
        RCSBClient,
    )
    from src.dataset.config import (
        DatasetBuildConfig,
    )
    from src.dataset.records import (
        SolutionNMRMonomerQualityRecord,
    )


class SolutionNMRProgramYearlyBuilder:
    """Build annual counts of refinement programs used by solution NMR."""

    def __init__(
        self,
        client: RCSBClient,
        config: DatasetBuildConfig,
        cache_dir: Path,
    ) -> None:
        """Initialize the SOLUTION NMR refinement-program trend builder."""
        self.client = client
        self.config = config
        self.cache_dir = cache_dir
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
        """Load refinement program labels from a cached or downloaded PDB file."""
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

    def build(self) -> list[SolutionNMRProgramYearlyCountRecord]:
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
        yearly_program_counter: Counter[tuple[int, str]] = Counter()

        entry_year_pairs = [
            (entry_id, entry_year_by_id.get(entry_id)) for entry_id in entry_ids
        ]
        for entry_id, year in entry_year_pairs:
            if year is None:
                _record_filtered_structure(
                    entry_id, "deposit year is missing or invalid"
                )

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
                    missing_program_count += 1
                    _record_filtered_structure(
                        entry_id, "PDB file could not be loaded", year=year
                    )
                elif not programs:
                    missing_program_count += 1
                    _record_filtered_structure(
                        entry_id, "refinement program is missing", year=year
                    )
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
                "without_program=%d, unique_programs=%d"
            ),
            len(entry_ids),
            missing_year_count,
            missing_program_count,
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


class SolutionNMRMonomerProgramClusterBuilder:
    """Build weighted software-cluster assignments for NMR monomers."""

    def __init__(
        self,
        quality_records: list[SolutionNMRMonomerQualityRecord],
        cache_dir: Path,
        max_workers: int,
        client: RCSBClient | None = None,
        config: DatasetBuildConfig | None = None,
    ) -> None:
        """Initialize the monomer refinement-program cluster builder."""
        self.quality_records = quality_records
        self.cache_dir = cache_dir
        self.max_workers = max(1, max_workers)
        self.client = client
        self.config = config

    def _load_program_text(self, entry_id: str) -> str:
        """Load raw refinement program text for one entry."""
        if self.client is not None and self.config is not None:
            try:
                pdb_path = download_pdb_if_needed(
                    session=self.client.session,
                    config=self.config,
                    cache_dir=self.cache_dir,
                    entry_id=entry_id,
                )
            except Exception as exc:
                LOGGER.warning("Failed to get PDB for %s: %s", entry_id, exc)
                return ""
        else:
            pdb_path = self.cache_dir / f"{entry_id}.pdb"
        if not pdb_path.exists() or pdb_path.stat().st_size <= 0:
            return ""
        return extract_raw_refinement_program_text_from_pdb(pdb_path)

    def build(
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
                cluster_score = 1.0 / len(clusters)
                for cluster_id, cluster_name in clusters:
                    assignments.append(
                        SolutionNMRMonomerProgramClusterAssignmentRecord(
                            entry_id=quality_record.entry_id,
                            year=quality_record.year,
                            cluster_id=cluster_id,
                            cluster_name=cluster_name,
                            cluster_score=cluster_score,
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
                    total_row["count"] += cluster_score
                    total_row["rama_sum"] += (
                        quality_record.ramachandran_outliers_percent * cluster_score
                    )
                    total_row["side_sum"] += (
                        quality_record.sidechain_outliers_percent * cluster_score
                    )
                    total_row["clash_sum"] += quality_record.clashscore * cluster_score

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
                count = float(totals["count"])
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
