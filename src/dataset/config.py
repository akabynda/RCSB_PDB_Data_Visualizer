"""Dataset kinds, shared configuration, and pipeline defaults."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

LOGGER = logging.getLogger("src.pdb_dataset_builder")


PDB_CHAIN_ID_POOL = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


XRAY_HOMOLOG_SEARCH_MAX_ATTEMPTS = 3


XRAY_HOMOLOG_HETATM_REJECTION_REASON = (
    "no eligible HETATM-free modeled core match at requested sequence identity"
)


XRAY_HOMOLOG_METHOD_REJECTION_REASON = (
    "experimental method set is not exactly [X-RAY DIFFRACTION]"
)


SOLUTION_NMR_METHOD = "SOLUTION NMR"


PROTEIN_MONOMER_ENTITY_TYPES: frozenset[str] = frozenset(
    {"polypeptide(L)", "polypeptide(D)"}
)


PROTEIN_POLYMER_TYPE = "Protein"


SEQUENCE_IDENTITY_AGGREGATION_METHOD = "sequence_identity"


DEFAULT_PDB_CACHE_DIR = Path("data/pdb_cache")


DEFAULT_STRIDE_CACHE_DIR = Path("data/stride_cache")


DEFAULT_STRIDE_INSTALL_DIR = Path("data/stride")


DEFAULT_PDB_CACHE_VALIDATION_HOURS = 24.0


PDB_CACHE_METADATA_SCHEMA_VERSION = 1


XRAY_CA_CACHE_SCHEMA_VERSION = 1


XRAY_CA_PARSER_REVISION = 2


STRIDE_REPOSITORY_URL = "https://github.com/MDAnalysis/stride.git"


STRIDE_SOURCE_REVISION = "867a5eb0f2479cb16615512a53ee472c54649505"


STRIDE_SETUP_TIMEOUT_SECONDS = 300.0


LOCAL_STRIDE_CANDIDATE = Path("/tmp/stride_src/src/stride")


STRIDE_STATE_CODES: tuple[str, ...] = ("H", "G", "I", "E", "B", "T", "C")


STRIDE_CORE_STATE_CODES: frozenset[str] = frozenset({"H", "G", "I", "E", "B"})


DEFAULT_MAX_WORKERS = max(1, os.cpu_count() or 1)


class ExperimentalMethod(Enum):
    """Map supported experimental methods to labels and RCSB query values."""

    X_RAY = ("X-ray", ("X-RAY DIFFRACTION",))
    CRYO_EM = ("cryo-EM", ("ELECTRON MICROSCOPY",))
    NMR = ("NMR", ("SOLUTION NMR", "SOLID-STATE NMR"))

    @property
    def label(self) -> str:
        """Return the display label for the experimental method."""
        return self.value[0]

    @property
    def query_values(self) -> tuple[str, ...]:
        """Return the RCSB method values used in search queries."""
        return self.value[1]

    @property
    def exact_method_sets(self) -> tuple[tuple[str, ...], ...]:
        """Return exact experimental-method sets assigned to this category."""
        single_method_sets = tuple((value,) for value in self.query_values)
        if self is ExperimentalMethod.NMR:
            return single_method_sets + (("SOLID-STATE NMR", "SOLUTION NMR"),)
        return single_method_sets


class DatasetKind(str, Enum):
    """Identify each CSV dataset that the command-line builder can produce."""

    METHOD_COUNTS = "method_counts"
    MEMBRANE_PROTEIN_COUNTS = "membrane_protein_counts"
    SOLUTION_NMR_PROGRAM_COUNTS = "solution_nmr_program_counts"
    SOLUTION_NMR_MONOMER_PROGRAM_CLUSTERS = "solution_nmr_monomer_program_clusters"
    SOLUTION_NMR_WEIGHTS = "solution_nmr_weights"
    SOLUTION_NMR_MONOMER_STRIDE_MODELED_FIRST_MODEL = (
        "solution_nmr_monomer_stride_modeled_first_model"
    )
    SOLUTION_NMR_MONOMER_PRECISION_STRIDE_MODELED_FIRST_MODEL = (
        "solution_nmr_monomer_precision_stride_modeled_first_model"
    )
    SOLUTION_NMR_MONOMER_QUALITY = "solution_nmr_monomer_quality"
    SOLUTION_NMR_MONOMER_EXPERIMENTS = "solution_nmr_monomer_experiments"
    SOLUTION_NMR_MONOMER_XRAY_HOMOLOGS = "solution_nmr_monomer_xray_homologs"
    SOLUTION_NMR_MONOMER_XRAY_HOMOLOGS_HISTORICAL = (
        "solution_nmr_monomer_xray_homologs_historical"
    )
    SOLUTION_NMR_MONOMER_XRAY_RMSD = "solution_nmr_monomer_xray_rmsd"
    SOLUTION_NMR_MONOMER_XRAY_RMSD_HISTORICAL = (
        "solution_nmr_monomer_xray_rmsd_historical"
    )
    SOLUTION_NMR_MONOMER_XRAY_RMSD_EXTREMES = "solution_nmr_monomer_xray_rmsd_extremes"
    SOLUTION_NMR_MONOMER_XRAY_RMSD_EXTREMES_HISTORICAL = (
        "solution_nmr_monomer_xray_rmsd_extremes_historical"
    )


@dataclass(frozen=True)
class DatasetBuildConfig:
    """Configure RCSB endpoints, batching, retries, caches, and concurrency."""

    search_url: str = "https://search.rcsb.org/rcsbsearch/v2/query"
    graphql_url: str = "https://data.rcsb.org/graphql"
    page_size: int = 10000
    graphql_batch_size: int = 300
    max_workers: int = DEFAULT_MAX_WORKERS
    timeout_seconds: int = 60
    retries: int = 4
    backoff_seconds: float = 1.3
    pdb_cache_validation_hours: float = DEFAULT_PDB_CACHE_VALIDATION_HOURS


MEMBRANE_ANNOTATION_TYPES: tuple[str, ...] = ("OPM", "PDBTM", "MemProtMD", "mpstruc")
