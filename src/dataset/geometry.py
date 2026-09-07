"""Rigid alignment and alpha-carbon ensemble RMSD calculations."""

from __future__ import annotations

import numpy as np
from MDAnalysis.analysis.align import rotation_matrix as mda_rotation_matrix
from MDAnalysis.analysis.rms import rmsd as mda_rmsd


def _superposed_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    """Compute RMSD after optimal rigid-body superposition."""
    return float(mda_rmsd(a, b, center=True, superposition=True))


def _aligned_coordinates_to_reference(
    mobile: np.ndarray, reference: np.ndarray
) -> np.ndarray:
    """Align coordinates onto a reference coordinate set."""
    mobile_center = np.mean(mobile, axis=0)
    reference_center = np.mean(reference, axis=0)
    mobile_centered = mobile - mobile_center
    reference_centered = reference - reference_center

    rotation, _ = mda_rotation_matrix(mobile_centered, reference_centered)
    return mobile_centered @ rotation.T + reference_center


def _coordinates_aligned_to_first_model(coords: np.ndarray) -> np.ndarray:
    """Align every model coordinate set onto the first model."""
    if coords.ndim != 3 or coords.shape[0] == 0:
        raise ValueError("coords must have shape (n_models, n_atoms, 3)")
    reference = np.asarray(coords[0], dtype=float)
    return np.asarray(
        [_aligned_coordinates_to_reference(model, reference) for model in coords],
        dtype=float,
    )


def _ca_rmsd_to_mean_structure(coords: np.ndarray) -> float:
    """Compute sqrt(1 / (N*n) * sum_i sum_j |r_ij - r_mean,j|^2)."""
    if coords.ndim != 3 or coords.shape[0] == 0:
        raise ValueError("coords must have shape (n_models, n_atoms, 3)")
    mean_coords = np.mean(coords, axis=0)
    squared_distances = np.sum(np.square(coords - mean_coords), axis=2)
    return float(np.sqrt(np.mean(squared_distances)))
