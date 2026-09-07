"""Backward-compatible imports and command-line entry point for dataset building.

Implementation lives in :mod:`src.dataset`. New code should import from the
module responsible for its task; this facade preserves existing helper scripts.
"""

# Re-exports intentionally retain the original module API.
# ruff: noqa: F401, E402
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import gzip
import hashlib
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import tempfile
import textwrap
import time
import uuid
import warnings
import requests
import numpy as np
from Bio import BiopythonWarning
from Bio.PDB import MMCIFParser, PDBIO, PDBParser, Select
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.SeqUtils import seq1
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from threading import Lock, RLock, local
from typing import Any, TypeVar
from MDAnalysis.analysis.align import rotation_matrix as mda_rotation_matrix
from MDAnalysis.analysis.rms import rmsd as mda_rmsd

from src.dataset.arguments import (
    parse_args,
    parse_dataset_kinds,
)
from src.dataset.builders.counts import (
    MembraneProteinYearlyBuilder,
    PDBMethodYearlyBuilder,
)
from src.dataset.builders.homologs import (
    SolutionNMRMonomerXrayHomologBuilder,
)
from src.dataset.builders.nmr import (
    SolutionNMRMonomerExperimentsBuilder,
    SolutionNMRMonomerQualityBuilder,
    SolutionNMRWeightBuilder,
)
from src.dataset.builders.precision import (
    SolutionNMRMonomerPrecisionBuilder,
    SolutionNMRMonomerPrecisionStrideModeledFirstModelBuilder,
)
from src.dataset.builders.programs import (
    SolutionNMRMonomerProgramClusterBuilder,
    SolutionNMRProgramYearlyBuilder,
)
from src.dataset.builders.rmsd_outputs import (
    build_solution_nmr_monomer_xray_rmsd_extremes_to_csv,
    build_solution_nmr_monomer_xray_rmsd_outputs_to_csv,
    build_solution_nmr_monomer_xray_rmsd_to_csv,
)
from src.dataset.builders.stride import (
    SolutionNMRMonomerStrideModeledFirstModelBuilder,
)
from src.dataset.builders.xray_rmsd import (
    SolutionNMRMonomerXrayRmsdBuilder,
    _build_xray_rmsd_extremes_record_from_candidates,
    _select_ordinary_xray_rmsd_record,
    _xray_rmsd_records_for_homolog_view,
)
from src.dataset.ca_cache import (
    _coordinate_source_sha256,
    _first_model_ca_cache_lock,
    _first_model_ca_cache_path,
    _parse_first_model_ca_data_by_chain,
    _read_first_model_ca_cache,
    _write_first_model_ca_cache,
    load_cached_first_model_ca_data,
)
from src.dataset.cache import (
    _atomic_install_pdb_response,
    _atomic_write_json,
    _cache_metadata_from_response,
    _cache_revision,
    _cached_pdb_matches_metadata,
    _close_http_response,
    _load_pdb_cache_metadata,
    _mmcif_download_sources,
    _pdb_cache_entry_lock,
    _pdb_cache_metadata_path,
    _pdb_cache_validation_is_fresh,
    _pdb_download_sources,
    _prioritize_cached_source,
    _response_chunks,
    _sha256_file,
    _utc_now_iso,
)
from src.dataset.cli import (
    _selected_dataset_output_paths,
    main,
)
from src.dataset.client import (
    RCSBClient,
    ThreadLocalRequestsSession,
)
from src.dataset.config import (
    DEFAULT_MAX_WORKERS,
    DEFAULT_PDB_CACHE_DIR,
    DEFAULT_PDB_CACHE_VALIDATION_HOURS,
    DEFAULT_STRIDE_CACHE_DIR,
    DEFAULT_STRIDE_INSTALL_DIR,
    DatasetBuildConfig,
    DatasetKind,
    ExperimentalMethod,
    LOCAL_STRIDE_CANDIDATE,
    LOGGER,
    MEMBRANE_ANNOTATION_TYPES,
    PDB_CACHE_METADATA_SCHEMA_VERSION,
    PDB_CHAIN_ID_POOL,
    PROTEIN_MONOMER_ENTITY_TYPES,
    PROTEIN_POLYMER_TYPE,
    SEQUENCE_IDENTITY_AGGREGATION_METHOD,
    SOLUTION_NMR_METHOD,
    STRIDE_CORE_STATE_CODES,
    STRIDE_REPOSITORY_URL,
    STRIDE_SETUP_TIMEOUT_SECONDS,
    STRIDE_SOURCE_REVISION,
    STRIDE_STATE_CODES,
    XRAY_CA_CACHE_SCHEMA_VERSION,
    XRAY_CA_PARSER_REVISION,
    XRAY_HOMOLOG_HETATM_REJECTION_REASON,
    XRAY_HOMOLOG_METHOD_REJECTION_REASON,
    XRAY_HOMOLOG_SEARCH_MAX_ATTEMPTS,
)
from src.dataset.coordinates import (
    _alt_loc_tiebreak_key,
    _insertion_code_tiebreak_key,
    _is_better_ca_candidate,
    _parse_first_model_ca_line_fields,
    _parse_pdb_modres_identity_map,
    _parse_pdb_occupancy,
    extract_model_pdb_texts,
    parse_first_model_ca_residue_sequence,
    parse_first_model_ca_residues,
    parse_first_model_modeled_ca_auth_seq_ids,
    parse_models_ca_coords,
    parse_models_ca_coords_with_stats,
)
from src.dataset.downloads import (
    _download_mmcif_if_needed,
    _download_mmcif_if_needed_locked,
    _download_pdb_chain_subset_if_needed_locked,
    _download_pdb_if_needed_locked,
    _load_valid_cached_chain_subset,
    download_pdb_chain_subset_if_needed,
    download_pdb_if_needed,
)
from src.dataset.errors import (
    NMRCoreContainsHetatmError,
    NMRHomologyQueryIneligibleError,
    XrayHomologEvaluationError,
    _is_http_server_error,
)
from src.dataset.geometry import (
    _aligned_coordinates_to_reference,
    _ca_rmsd_to_mean_structure,
    _coordinates_aligned_to_first_model,
    _superposed_rmsd,
)
from src.dataset.history import (
    filter_xray_homolog_records_by_deposit_date,
)
from src.dataset.io.common import (
    _atomic_write_csv_rows,
    write_csv_rows,
)
from src.dataset.io.homologs import (
    REJECTED_XRAY_HOMOLOG_HEADER,
    SOLUTION_NMR_MONOMER_XRAY_HOMOLOG_HEADER,
    _read_rejected_xray_homolog_csv_with_status,
    _read_xray_homolog_resume_checkpoint,
    _rejected_xray_homolog_csv_row,
    _solution_nmr_monomer_xray_homolog_csv_row,
    _write_xray_homolog_resume_checkpoint_statuses,
    _xray_homolog_resume_checkpoint_path,
    read_rejected_xray_homolog_csv,
    read_solution_nmr_monomer_xray_homolog_csv,
    rejected_xray_homologs_csv_path,
    write_rejected_xray_homolog_csv,
    write_solution_nmr_monomer_xray_homolog_csv,
)
from src.dataset.io.nmr import (
    SOLUTION_NMR_MONOMER_PRECISION_HEADER,
    _solution_nmr_monomer_precision_csv_row,
    read_solution_nmr_monomer_precision_csv,
    read_solution_nmr_monomer_quality_csv,
    stream_solution_nmr_monomer_stride_modeled_first_model_csv,
    write_membrane_counts_csv,
    write_method_counts_csv,
    write_solution_nmr_monomer_experiments_csv,
    write_solution_nmr_monomer_quality_csv,
    write_solution_nmr_weights_csv,
)
from src.dataset.io.programs import (
    read_solution_nmr_monomer_program_cluster_assignments_csv,
    write_solution_nmr_monomer_program_cluster_assignments_csv,
    write_solution_nmr_monomer_program_cluster_summary_csv,
    write_solution_nmr_monomer_program_cluster_total_csv,
    write_solution_nmr_monomer_program_cluster_yearly_summary_csv,
    write_solution_nmr_program_counts_csv,
)
from src.dataset.io.rmsd import (
    SOLUTION_NMR_MONOMER_XRAY_RMSD_EXTREMES_HEADER,
    SOLUTION_NMR_MONOMER_XRAY_RMSD_HEADER,
    _solution_nmr_monomer_xray_rmsd_csv_row,
    _solution_nmr_monomer_xray_rmsd_extremes_csv_row,
    read_solution_nmr_monomer_xray_rmsd_csv,
    read_solution_nmr_monomer_xray_rmsd_extremes_csv,
    write_solution_nmr_monomer_xray_rmsd_csv,
    write_solution_nmr_monomer_xray_rmsd_extremes_csv,
)
from src.dataset.matching import (
    _ca_residue_has_hetatm,
    _find_gapped_modeled_ca_core_identity_match,
    _split_xray_ca_residues_at_hetatm,
    find_modeled_ca_core_identity_matches,
)
from src.dataset.program_statistics import (
    summarize_solution_nmr_monomer_program_cluster_quality_by_year,
    summarize_solution_nmr_monomer_program_cluster_quality_total,
)
from src.dataset.programs import (
    NMR_REMARK_FIELD_PATTERN,
    NMR_REMARK_PATTERN,
    NMR_SOFTWARE_REMARK_PATTERN,
    PROGRAM_CLUSTER_DEFINITIONS,
    PROGRAM_CLUSTER_MATCH_PATTERNS,
    PROGRAM_CLUSTER_NAME_BY_ID,
    PROGRAM_EMPTY_VALUES,
    PROGRAM_HAS_LETTER_PATTERN,
    PROGRAM_PARENTHESIS_PATTERN,
    PROGRAM_REMARK_PATTERN,
    PROGRAM_SPLIT_PATTERN,
    PROGRAM_TRAILING_VERSION_PATTERN,
    XPLOR_NIH_PATTERNS,
    XPLOR_PATTERN,
    _extract_program_cluster_matches,
    _normalize_refinement_program_name,
    _prepend_mmcif_software_remarks,
    extract_raw_refinement_program_text_from_mmcif,
    extract_raw_refinement_program_text_from_pdb,
    extract_refinement_programs_from_pdb,
    extract_solution_nmr_program_clusters,
)
from src.dataset.records import (
    CAResidueRecord,
    MembraneYearlyCountRecord,
    PreparedNMRCoreData,
    PreparedXrayCAData,
    RejectedXrayHomologRecord,
    SolutionNMRMonomerExperimentsRecord,
    SolutionNMRMonomerModeledFirstModelSeedRecord,
    SolutionNMRMonomerPrecisionRecord,
    SolutionNMRMonomerProgramClusterAssignmentRecord,
    SolutionNMRMonomerProgramClusterSummaryRecord,
    SolutionNMRMonomerProgramClusterTotalRecord,
    SolutionNMRMonomerProgramClusterYearlySummaryRecord,
    SolutionNMRMonomerQualityRecord,
    SolutionNMRMonomerStrideModeledFirstModelRecord,
    SolutionNMRMonomerXrayHomologRecord,
    SolutionNMRMonomerXrayHomologSeedRecord,
    SolutionNMRMonomerXrayRmsdExtremesRecord,
    SolutionNMRMonomerXrayRmsdRecord,
    SolutionNMRProgramYearlyCountRecord,
    SolutionNMRWeightRecord,
    XrayEntityGroupMappingRecord,
    XrayPolymerEntityCandidateRecord,
    YearlyCountRecord,
)
from src.dataset.reporting import (
    ActiveDatasetWarningLogFilter,
    _configure_dataset_filtered_csvs,
    _configure_dataset_warning_logs,
    _import_filtered_structures,
    _normalize_filtered_structure_year,
    _record_filtered_structure,
    _set_active_dataset_filtered_csvs,
    _set_active_dataset_warning_logs,
    filtered_structures_csv_path,
)
from src.dataset.stride import (
    _extract_stride_core_range_for_modeled_auth_seq_ids,
    _load_cached_stride_state_by_chain,
    _parse_stride_state_by_chain,
    _run_stride_for_model_text,
    _select_stride_chain_states,
    _stride_state_cache_path,
    _write_cached_stride_state_by_chain,
    compute_stride_core_range_for_modeled_auth_seq_ids_in_first_model,
    compute_stride_state_coverages_for_chain_modeled_first_model,
    load_first_model_stride_state_by_chain,
)
from src.dataset.stride_install import (
    _build_stride_checkout,
    _is_usable_stride_executable,
    _managed_stride_checkout_dir,
    _managed_stride_executable_path,
    _run_stride_setup_command,
    _stride_install_lock,
    _verify_existing_stride_checkout,
    download_and_build_stride,
    ensure_stride_executable,
    resolve_stride_executable,
)
from src.dataset.structures import (
    ChainSubsetSelect,
    _apply_chain_id_map_without_transient_conflicts,
    _chain_subset_cache_stem,
    _coerce_selected_structure_chain_ids_for_pdbio,
    _coerce_structure_chain_ids_for_pdbio,
    load_cached_chain_id_map,
    load_chain_id_map,
    parse_mmcif_structure,
    parse_pdb_structure,
)
from src.dataset.utils import (
    _record_entries_missing_from_response,
    chunked,
    collect_batch_results,
    contains_noesy_experiment,
    extract_year,
    fetch_solution_nmr_entry_ids,
    parse_rcsb_datetime,
)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        LOGGER.warning("Dataset builder interrupted by the user")
        raise
    except SystemExit as exc:
        if exc.code not in {None, 0}:
            LOGGER.error("Dataset builder terminated: %s", exc)
        raise
    except Exception:
        LOGGER.exception("Dataset builder terminated with an unhandled error")
        raise
