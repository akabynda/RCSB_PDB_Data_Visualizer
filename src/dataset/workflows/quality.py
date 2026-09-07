"""Dataset collection workflows for quality."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.dataset.builders.nmr import SolutionNMRMonomerQualityBuilder
from src.dataset.builders.programs import SolutionNMRMonomerProgramClusterBuilder
from src.dataset.client import RCSBClient
from src.dataset.config import (
    LOGGER,
    DatasetBuildConfig,
)
from src.dataset.io.nmr import (
    read_solution_nmr_monomer_quality_csv,
    write_solution_nmr_monomer_quality_csv,
)
from src.dataset.io.programs import (
    write_solution_nmr_monomer_program_cluster_assignments_csv,
    write_solution_nmr_monomer_program_cluster_summary_csv,
    write_solution_nmr_monomer_program_cluster_total_csv,
    write_solution_nmr_monomer_program_cluster_yearly_summary_csv,
)
from src.dataset.program_statistics import (
    summarize_solution_nmr_monomer_program_cluster_quality_by_year,
    summarize_solution_nmr_monomer_program_cluster_quality_total,
)
from src.dataset.reporting import _import_filtered_structures


def run_quality(
    args: argparse.Namespace, client: RCSBClient, config: DatasetBuildConfig
) -> None:
    """Build monomer quality metrics."""
    quality_builder = SolutionNMRMonomerQualityBuilder(client=client, config=config)
    quality_records = quality_builder.build()
    write_solution_nmr_monomer_quality_csv(
        records=quality_records,
        output_path=args.solution_nmr_monomer_quality_output,
    )
    LOGGER.info(
        "Saved %d records to %s",
        len(quality_records),
        args.solution_nmr_monomer_quality_output,
    )


def run_program_clusters(
    args: argparse.Namespace, client: RCSBClient, config: DatasetBuildConfig
) -> None:
    """Build program cluster assignments and quality summaries."""
    quality_input_path = Path(args.solution_nmr_monomer_program_cluster_input)
    _import_filtered_structures(quality_input_path)
    quality_records = read_solution_nmr_monomer_quality_csv(quality_input_path)
    if not quality_records:
        raise RuntimeError(
            "solution_nmr_monomer_program_clusters requires a non-empty quality "
            f"CSV at {quality_input_path}"
        )
    cluster_builder = SolutionNMRMonomerProgramClusterBuilder(
        quality_records=quality_records,
        cache_dir=Path(args.solution_nmr_monomer_program_cluster_cache_dir),
        max_workers=config.max_workers,
        client=client,
        config=config,
    )
    assignment_records, summary_records = cluster_builder.build()
    write_solution_nmr_monomer_program_cluster_assignments_csv(
        records=assignment_records,
        output_path=Path(args.solution_nmr_monomer_program_cluster_assignment_output),
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
