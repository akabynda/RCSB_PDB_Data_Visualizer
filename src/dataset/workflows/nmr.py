"""Dataset collection workflows for nmr."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.dataset.builders.nmr import (
    SolutionNMRMonomerExperimentsBuilder,
    SolutionNMRWeightBuilder,
)
from src.dataset.builders.programs import SolutionNMRProgramYearlyBuilder
from src.dataset.client import RCSBClient
from src.dataset.config import (
    LOGGER,
    DatasetBuildConfig,
)
from src.dataset.io.nmr import (
    write_solution_nmr_monomer_experiments_csv,
    write_solution_nmr_weights_csv,
)
from src.dataset.io.programs import write_solution_nmr_program_counts_csv
from src.dataset.utils import contains_noesy_experiment


def run_program_counts(
    args: argparse.Namespace, client: RCSBClient, config: DatasetBuildConfig
) -> None:
    """Build the yearly solution NMR program counts."""
    nmr_program_builder = SolutionNMRProgramYearlyBuilder(
        client=client,
        config=config,
        cache_dir=Path(args.solution_nmr_program_cache_dir),
    )
    nmr_program_records = nmr_program_builder.build()
    write_solution_nmr_program_counts_csv(
        records=nmr_program_records,
        output_path=args.solution_nmr_program_counts_output,
    )
    LOGGER.info(
        "Saved %d records to %s",
        len(nmr_program_records),
        args.solution_nmr_program_counts_output,
    )


def run_weights(
    args: argparse.Namespace, client: RCSBClient, config: DatasetBuildConfig
) -> None:
    """Build solution NMR structure weight records."""
    nmr_weight_builder = SolutionNMRWeightBuilder(client=client, config=config)
    nmr_weight_records = nmr_weight_builder.build()
    write_solution_nmr_weights_csv(
        records=nmr_weight_records, output_path=args.solution_nmr_output
    )
    LOGGER.info(
        "Saved %d records to %s",
        len(nmr_weight_records),
        args.solution_nmr_output,
    )


def run_experiments(
    args: argparse.Namespace, client: RCSBClient, config: DatasetBuildConfig
) -> None:
    """Build monomer experiment records and report NOESY coverage."""
    experiments_builder = SolutionNMRMonomerExperimentsBuilder(
        client=client,
        config=config,
    )
    experiment_records = experiments_builder.build()
    write_solution_nmr_monomer_experiments_csv(
        records=experiment_records,
        output_path=args.solution_nmr_monomer_experiments_output,
    )
    noesy_count = sum(
        contains_noesy_experiment(record.nmr_experiments_conducted)
        for record in experiment_records
    )
    LOGGER.info(
        "Saved %d records to %s; NOESY entries: %d",
        len(experiment_records),
        args.solution_nmr_monomer_experiments_output,
        noesy_count,
    )
