"""Typed records shared by dataset collection, calculations, and CSV serialization."""

from __future__ import annotations

from dataclasses import dataclass
from functools import total_ordering
from numbers import Integral
import re

import numpy as np


@total_ordering
@dataclass(frozen=True, eq=False)
class ResidueId:
    """Author residue number and insertion code, independent of atom altLoc."""

    seq_id: int
    insertion_code: str = ""

    def __post_init__(self) -> None:
        if len(self.insertion_code) > 1 or self.insertion_code.isspace():
            raise ValueError("Insertion code must be empty or one nonblank character")

    def __hash__(self) -> int:
        # Preserve compatibility with integer keys for ordinary numbered residues.
        return (
            hash((self.seq_id, self.insertion_code))
            if self.insertion_code
            else hash(self.seq_id)
        )

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Integral):
            return not self.insertion_code and self.seq_id == other
        if isinstance(other, ResidueId):
            return (self.seq_id, self.insertion_code) == (
                other.seq_id,
                other.insertion_code,
            )
        return NotImplemented

    def __lt__(self, other: object) -> bool:
        if isinstance(other, Integral):
            other = ResidueId(int(other))
        if isinstance(other, ResidueId):
            return (self.seq_id, self.insertion_code) < (
                other.seq_id,
                other.insertion_code,
            )
        return NotImplemented

    def __int__(self) -> int:
        return self.seq_id

    def __str__(self) -> str:
        # Numeric insertion codes must not become an extra digit of resSeq.
        suffix = self.insertion_code
        if suffix and not suffix.isalpha():
            suffix = "^" + suffix
        return f"{self.seq_id}{suffix}"


def parse_residue_id(value: str | int | ResidueId) -> ResidueId:
    """Read a CSV/STRIDE residue label without discarding its insertion code."""
    if isinstance(value, ResidueId):
        return value
    if isinstance(value, Integral):
        return ResidueId(int(value))
    match = re.fullmatch(r"([+-]?\d+)(?:\^([^\s])|([A-Za-z]))?", str(value).strip())
    if match is None:
        raise ValueError(f"Invalid author residue identifier: {value!r}")
    return ResidueId(int(match.group(1)), match.group(2) or match.group(3) or "")


@dataclass(frozen=True)
class CAResidueRecord:
    """Describe one modeled alpha-carbon residue parsed from a PDB file."""

    resid: int
    identity: str
    is_standard_atom: bool
    has_hetatm_ca: bool = False
    insertion_code: str = ""

    @property
    def key(self) -> ResidueId:
        """Return the full identity used for residue and coordinate lookups."""
        return ResidueId(self.resid, self.insertion_code)


PreparedNMRCoreData = tuple[
    tuple[CAResidueRecord, ...],
    dict[ResidueId, np.ndarray],
]


PreparedXrayCAData = tuple[
    tuple[CAResidueRecord, ...],
    dict[ResidueId, np.ndarray],
]


@dataclass(frozen=True)
class YearlyCountRecord:
    """Store the annual structure count for an experimental method."""

    year: int
    method: str
    count: int


@dataclass(frozen=True)
class MembraneYearlyCountRecord:
    """Store the number of membrane-protein structures in one year."""

    year: int
    count: int


@dataclass(frozen=True)
class SolutionNMRProgramYearlyCountRecord:
    """Store a yearly count for one solution-NMR refinement program."""

    year: int
    program: str
    count: int


@dataclass(frozen=True)
class SolutionNMRMonomerProgramClusterAssignmentRecord:
    """Store one monomer's weighted assignment to a software cluster."""

    entry_id: str
    year: int
    cluster_id: str
    cluster_name: str
    cluster_score: float
    has_program_text: bool
    program_text: str


@dataclass(frozen=True)
class SolutionNMRMonomerProgramClusterSummaryRecord:
    """Store yearly count and quality metrics for a software cluster."""

    year: int
    cluster_id: str
    cluster_name: str
    structure_count: float
    avg_ramachandran_outliers_percent: float | None
    avg_sidechain_outliers_percent: float | None
    avg_clashscore: float | None


@dataclass(frozen=True)
class SolutionNMRMonomerProgramClusterYearlySummaryRecord:
    """Store aggregate monomer count and quality metrics for one year."""

    year: int
    structure_count: int
    avg_ramachandran_outliers_percent: float | None
    avg_sidechain_outliers_percent: float | None
    avg_clashscore: float | None


@dataclass(frozen=True)
class SolutionNMRMonomerProgramClusterTotalRecord:
    """Store all-years count and quality metrics for a software cluster."""

    cluster_name: str
    structure_count: float
    avg_ramachandran_outliers_percent: float | None
    avg_sidechain_outliers_percent: float | None
    avg_clashscore: float | None


@dataclass(frozen=True)
class SolutionNMRWeightRecord:
    """Store a solution-NMR entry's deposition year and molecular weight."""

    entry_id: str
    year: int
    molecular_weight_kda: float


@dataclass(frozen=True)
class SolutionNMRMonomerExperimentsRecord:
    """Store experiments reported for one solution-NMR monomer entry."""

    entry_id: str
    year: int
    nmr_experiments_conducted: tuple[str, ...]


@dataclass(frozen=True)
class SolutionNMRMonomerStrideModeledFirstModelRecord:
    """Store STRIDE composition for an entry's first modeled chain."""

    entry_id: str
    year: int
    chain_id: str
    modeled_start_seq_id: ResidueId | int
    modeled_end_seq_id: ResidueId | int
    modeled_sequence_length: int
    stride_alpha_helix_fraction: float
    stride_3_10_helix_fraction: float
    stride_pi_helix_fraction: float
    stride_beta_strand_fraction: float
    stride_isolated_beta_bridge_fraction: float
    stride_turn_fraction: float
    stride_coil_fraction: float
    stride_secondary_structure_percent: float


@dataclass(frozen=True)
class SolutionNMRMonomerModeledFirstModelSeedRecord:
    """Identify an eligible solution-NMR monomer and its modeled chain."""

    entry_id: str
    year: int
    chain_id: str


@dataclass(frozen=True)
class SolutionNMRMonomerPrecisionRecord:
    """Store ensemble precision measured over a modeled structural core."""

    entry_id: str
    year: int
    chain_id: str
    core_start_seq_id: ResidueId | int
    core_end_seq_id: ResidueId | int
    n_models: int
    n_ca_core_used: int
    n_ca_core_raw: int
    mean_rmsd_angstrom: float


@dataclass(frozen=True)
class SolutionNMRMonomerQualityRecord:
    """Store validation quality metrics for a solution-NMR monomer."""

    entry_id: str
    year: int
    clashscore: float
    ramachandran_outliers_percent: float
    sidechain_outliers_percent: float


@dataclass(frozen=True)
class RejectedXrayHomologRecord:
    """Describe one sequence-search hit rejected by an eligibility check."""

    nmr_entry_id: str
    nmr_year: int
    nmr_chain_id: str
    sequence_identity_percent: int
    nmr_core_start_seq_id: ResidueId | int | None
    nmr_core_end_seq_id: ResidueId | int | None
    nmr_query_sequence_length: int
    xray_entry_id: str
    xray_entity_id: str
    xray_chain_ids: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class SolutionNMRMonomerXrayHomologRecord:
    """Store X-ray homolog search results for a solution-NMR monomer."""

    entry_id: str
    year: int
    sequence_identity_percent: int
    nmr_core_start_seq_id: ResidueId | int | None
    nmr_core_end_seq_id: ResidueId | int | None
    nmr_query_sequence_length: int
    xray_homolog_entry_ids: tuple[str, ...]
    xray_homolog_entity_ids: tuple[str, ...]
    has_xray_homolog: bool
    rejected_xray_homologs: tuple[RejectedXrayHomologRecord, ...] = ()


@dataclass(frozen=True)
class SolutionNMRMonomerXrayHomologSeedRecord:
    """Identify a solution-NMR monomer chain for an X-ray homolog search."""

    entry_id: str
    year: int
    chain_id: str


@dataclass(frozen=True)
class XrayEntityGroupMappingRecord:
    """Map an X-ray polymer entity and chains to a sequence-identity group."""

    polymer_entity_id: str
    entry_id: str
    chain_ids: tuple[str, ...]
    group_id: str


@dataclass(frozen=True)
class SolutionNMRMonomerXrayRmsdRecord:
    """Store an NMR-to-X-ray alpha-carbon RMSD measurement."""

    entry_id: str
    year: int
    sequence_identity_percent: int
    nmr_chain_id: str
    nmr_core_start_seq_id: ResidueId | int | None
    nmr_core_end_seq_id: ResidueId | int | None
    nmr_query_sequence_length: int
    xray_homolog_entity_id: str
    xray_homolog_count: int
    xray_entry_id: str
    xray_chain_id: str
    xray_core_start_seq_id: ResidueId | int | None
    xray_core_end_seq_id: ResidueId | int | None
    xray_resolution_angstrom: float
    n_common_ca: int
    rmsd_ca_angstrom: float


@dataclass(frozen=True)
class SolutionNMRMonomerXrayRmsdExtremesRecord:
    """Store the minimum and maximum X-ray RMSD matches for an NMR entry."""

    entry_id: str
    year: int
    sequence_identity_percent: int
    nmr_chain_id: str
    nmr_core_start_seq_id: ResidueId | int | None
    nmr_core_end_seq_id: ResidueId | int | None
    nmr_query_sequence_length: int
    xray_homolog_count: int
    successful_xray_homolog_count: int
    best_xray_homolog_entity_id: str
    best_xray_entry_id: str
    best_xray_chain_id: str
    best_xray_resolution_angstrom: float
    best_xray_core_start_seq_id: ResidueId | int | None
    best_xray_core_end_seq_id: ResidueId | int | None
    best_n_common_ca: int
    best_rmsd_ca_angstrom: float
    worst_xray_homolog_entity_id: str
    worst_xray_entry_id: str
    worst_xray_chain_id: str
    worst_xray_resolution_angstrom: float
    worst_xray_core_start_seq_id: ResidueId | int | None
    worst_xray_core_end_seq_id: ResidueId | int | None
    worst_n_common_ca: int
    worst_rmsd_ca_angstrom: float
    rmsd_delta_angstrom: float


@dataclass(frozen=True)
class XrayPolymerEntityCandidateRecord:
    """Describe an X-ray search candidate and its eligibility metadata."""

    polymer_entity_id: str
    entry_id: str
    chain_ids: tuple[str, ...]
    resolution_angstrom: float
    experimental_methods: tuple[str, ...] = ()
