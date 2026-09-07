"""Dataset collection workflows for historical homologs."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.dataset.client import RCSBClient
from src.dataset.config import (
    LOGGER,
    DatasetBuildConfig,
)
from src.dataset.history import filter_xray_homolog_records_by_deposit_date
from src.dataset.io.homologs import (
    read_solution_nmr_monomer_xray_homolog_csv,
    write_solution_nmr_monomer_xray_homolog_csv,
)
from src.dataset.reporting import (
    _import_filtered_structures,
    _set_active_dataset_filtered_csvs,
)


def run_historical_homologs(
    args: argparse.Namespace, client: RCSBClient, config: DatasetBuildConfig
) -> None:
    """Filter X-ray homologs by their NMR entry deposit dates."""
    homolog_95_input_path = Path(args.solution_nmr_monomer_xray_homolog_95_output)
    homolog_100_input_path = Path(args.solution_nmr_monomer_xray_homolog_100_output)
    records_95 = read_solution_nmr_monomer_xray_homolog_csv(homolog_95_input_path)
    records_100 = read_solution_nmr_monomer_xray_homolog_csv(homolog_100_input_path)
    if not records_95 or not records_100:
        raise SystemExit(
            "No X-ray homolog records found for historical filtering. Run "
            "solution_nmr_monomer_xray_homologs first or provide the expected "
            f"homolog CSVs at {homolog_95_input_path} and {homolog_100_input_path}."
        )
    homolog_95_historical_output_path = Path(
        args.solution_nmr_monomer_xray_homolog_95_historical_output
    )
    homolog_100_historical_output_path = Path(
        args.solution_nmr_monomer_xray_homolog_100_historical_output
    )
    _set_active_dataset_filtered_csvs((homolog_95_historical_output_path,))
    _import_filtered_structures(homolog_95_input_path)
    historical_records_95 = filter_xray_homolog_records_by_deposit_date(
        records=records_95,
        client=client,
        config=config,
    )
    _set_active_dataset_filtered_csvs((homolog_100_historical_output_path,))
    _import_filtered_structures(homolog_100_input_path)
    historical_records_100 = filter_xray_homolog_records_by_deposit_date(
        records=records_100,
        client=client,
        config=config,
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
