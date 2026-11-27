"""Single-sample (N = 1) intra-sample invariant internal control calibration.

Implements the self-contained Toray 3D-Gene Human miRNA Oligo Chip
(GPL21263) invariant internal control normalization protocol (Matsuzaki et al.,
2023) using three stable endogenous reference microRNAs:
``hsa-miR-149-3p``, ``hsa-miR-2861``, and ``hsa-miR-4463``.

Because calibration operates strictly row-wise on each patient specimen's own
internal control trio without batch peers or cohort statistics, it supports
prospective single-patient (N = 1) clinical inference without ComBat or
pooled Z-score harmonization.
"""

from __future__ import annotations

from typing import Sequence
import numpy as np

from highway_bilstm.constants import INVARIANT_CONTROLS

DETECTION_FLOOR: float = 0.1
CALIBRATION_OFFSET: float = 10.9


def normalize_invariant_controls(
    X: np.ndarray,
    mirna_names: Sequence[str],
    ctrl_names: Sequence[str] = INVARIANT_CONTROLS,
    floor: float = DETECTION_FLOOR,
    offset: float = CALIBRATION_OFFSET,
) -> np.ndarray:
    """Calibrate raw miRNA intensities using intra-sample invariant internal controls.

    For each individual sample vector :math:`x`, raw hybridization signals are
    clipped at ``floor`` (default ``0.1``), transformed to :math:`\\log_2`,
    centered by subtracting the sample-specific mean of the invariant control
    trio, and shifted by ``offset`` (default ``10.9``).

    Args:
        X: Raw intensity matrix of shape ``(n_samples, n_features)`` or a single
            patient profile of shape ``(n_features,)``.
        mirna_names: Ordered sequence of miRNA names matching the feature axis.
        ctrl_names: Reference control miRNA names. Defaults to
            ``("hsa-miR-149-3p", "hsa-miR-2861", "hsa-miR-4463")``.
        floor: Minimum detection floor applied before ``log2`` transformation.
        offset: Global dynamic-range offset added after control centering.

    Returns:
        Calibrated ``np.float32`` array matching the input dimensionality
        (``(n_features,)`` for 1D input or ``(n_samples, n_features)`` for 2D).

    Raises:
        ValueError: If none of ``ctrl_names`` are present in ``mirna_names``.
    """
    X_arr = np.asarray(X, dtype=np.float32)
    is_1d = X_arr.ndim == 1
    if is_1d:
        X_arr = X_arr[np.newaxis, :]

    name_to_idx = {name: i for i, name in enumerate(mirna_names)}
    ctrl_indices = [name_to_idx[c] for c in ctrl_names if c in name_to_idx]
    if not ctrl_indices:
        raise ValueError(f"None of invariant controls {ctrl_names} found in mirna_names.")

    log_X = np.log2(np.maximum(X_arr, floor))
    sample_ctrl_mean = np.mean(log_X[:, ctrl_indices], axis=1, keepdims=True)
    norm_X = (log_X - sample_ctrl_mean + offset).astype(np.float32)
    return norm_X[0] if is_1d else norm_X
