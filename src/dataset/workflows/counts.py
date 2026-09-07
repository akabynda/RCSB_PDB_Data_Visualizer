"""Dataset collection workflows for counts."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.dataset.builders.counts import (
    MembraneProteinYearlyBuilder,
    PDBMethodYearlyBuilder,
)
from src.dataset.client import RCSBClient
from src.dataset.config import (
    LOGGER,
    DatasetBuildConfig,
    ExperimentalMethod,
)
from src.dataset.io.nmr import (
    write_membrane_counts_csv,
    write_method_counts_csv,
)
from src.dataset.reporting import _set_active_dataset_filtered_csvs


def run_method_counts(
    args: argparse.Namespace, client: RCSBClient, config: DatasetBuildConfig
) -> None:
    """Build the yearly experimental-method counts."""
    method_builder = PDBMethodYearlyBuilder(client=client, config=config)
    method_records = method_builder.build(
        [
            ExperimentalMethod.X_RAY,
            ExperimentalMethod.CRYO_EM,
            ExperimentalMethod.NMR,
        ]
    )
    write_method_counts_csv(records=method_records, output_path=args.counts_output)
    LOGGER.info("Saved %d records to %s", len(method_records), args.counts_output)


def run_membrane_counts(
    args: argparse.Namespace, client: RCSBClient, config: DatasetBuildConfig
) -> None:
    """Build membrane totals and counts by experimental method."""
    membrane_builder = MembraneProteinYearlyBuilder(client=client, config=config)
    _set_active_dataset_filtered_csvs((Path(args.membrane_counts_output),))
    membrane_records = membrane_builder.build()
    write_membrane_counts_csv(
        records=membrane_records, output_path=args.membrane_counts_output
    )
    _set_active_dataset_filtered_csvs((Path(args.membrane_method_counts_output),))
    membrane_method_records = membrane_builder.build_by_method(
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
