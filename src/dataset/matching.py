"""Match modeled NMR cores against eligible X-ray alpha-carbon sequences."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.dataset.records import (
        CAResidueRecord,
    )


def find_modeled_ca_core_identity_matches(
    nmr_residues: list[CAResidueRecord],
    xray_residues: list[CAResidueRecord],
    sequence_identity_percent: int,
) -> list[list[tuple[CAResidueRecord, CAResidueRecord]]]:
    """Find HETATM-free X-ray CA ranges matching a modeled NMR core."""
    if not nmr_residues or not xray_residues:
        return []
    if not 0 <= sequence_identity_percent <= 100:
        raise ValueError("sequence_identity_percent must be between 0 and 100")
    nmr_identities = [record.identity for record in nmr_residues]
    query_length = len(nmr_identities)
    xray_regions = _split_xray_ca_residues_at_hetatm(xray_residues)
    if sequence_identity_percent < 100:
        min_count = (query_length * sequence_identity_percent + 99) // 100
        matches: list[list[tuple[CAResidueRecord, CAResidueRecord]]] = []
        for xray_region in xray_regions:
            if len(xray_region) < min_count:
                continue
            match = _find_gapped_modeled_ca_core_identity_match(
                nmr_residues=nmr_residues,
                xray_residues=xray_region,
                sequence_identity_percent=sequence_identity_percent,
            )
            if match:
                matches.append(match)
        return matches

    matches: list[list[tuple[CAResidueRecord, CAResidueRecord]]] = []
    query_sequence = "".join(nmr_identities)
    for xray_region in xray_regions:
        if query_length > len(xray_region):
            continue
        xray_sequence = "".join(record.identity for record in xray_region)
        start_idx = xray_sequence.find(query_sequence)
        while start_idx >= 0:
            end_idx = start_idx + query_length
            matches.append(list(zip(nmr_residues, xray_region[start_idx:end_idx])))
            start_idx = xray_sequence.find(query_sequence, start_idx + 1)
    return matches


def _split_xray_ca_residues_at_hetatm(
    xray_residues: Sequence[CAResidueRecord],
) -> list[list[CAResidueRecord]]:
    """Split an X-ray chain into regions containing no HETATM CA positions."""
    regions: list[list[CAResidueRecord]] = []
    current_region: list[CAResidueRecord] = []
    for record in xray_residues:
        if _ca_residue_has_hetatm(record):
            if current_region:
                regions.append(current_region)
                current_region = []
            continue
        current_region.append(record)
    if current_region:
        regions.append(current_region)
    return regions


def _ca_residue_has_hetatm(record: CAResidueRecord) -> bool:
    """Return whether a modeled CA position includes a HETATM record."""
    return record.has_hetatm_ca or not record.is_standard_atom


def _find_gapped_modeled_ca_core_identity_match(
    nmr_residues: list[CAResidueRecord],
    xray_residues: list[CAResidueRecord],
    sequence_identity_percent: int,
) -> list[tuple[CAResidueRecord, CAResidueRecord]] | None:
    """Find a modeled core match allowing residue gaps at CA positions."""
    nmr_len = len(nmr_residues)
    min_count = (nmr_len * sequence_identity_percent + 99) // 100
    if min_count <= 0:
        return None

    nmr_identities = [record.identity for record in nmr_residues]
    xray_identities = [record.identity for record in xray_residues]
    rows = nmr_len + 1
    cols = len(xray_residues) + 1
    scores = np.zeros((rows, cols), dtype=float)
    pointers = np.zeros((rows, cols), dtype=np.int8)
    best_score = 0.0
    best_pos: tuple[int, int] | None = None

    for i in range(1, rows):
        nmr_identity = nmr_identities[i - 1]
        for j in range(1, cols):
            xray_identity = xray_identities[j - 1]
            diag_score = scores[i - 1, j - 1] + (
                2.0 if nmr_identity == xray_identity else -1.0
            )
            up_score = scores[i - 1, j] - 1.0
            left_score = scores[i, j - 1] - 1.0
            cell_score = max(0.0, diag_score, up_score, left_score)
            scores[i, j] = cell_score
            if cell_score == 0.0:
                continue
            if cell_score == diag_score:
                pointers[i, j] = 1
            elif cell_score == up_score:
                pointers[i, j] = 2
            else:
                pointers[i, j] = 3
            if cell_score > best_score:
                best_score = cell_score
                best_pos = (i, j)

    if best_pos is None:
        return None

    i, j = best_pos
    pairs: list[tuple[CAResidueRecord, CAResidueRecord]] = []
    identity_count = 0
    while i > 0 and j > 0 and scores[i, j] > 0.0:
        pointer = pointers[i, j]
        if pointer == 1:
            nmr_record = nmr_residues[i - 1]
            xray_record = xray_residues[j - 1]
            pairs.append((nmr_record, xray_record))
            if nmr_record.identity == xray_record.identity:
                identity_count += 1
            i -= 1
            j -= 1
        elif pointer == 2:
            i -= 1
        elif pointer == 3:
            j -= 1
        else:
            break

    pairs.reverse()
    if len(pairs) < min_count:
        return None
    if identity_count < min_count:
        return None
    return pairs
