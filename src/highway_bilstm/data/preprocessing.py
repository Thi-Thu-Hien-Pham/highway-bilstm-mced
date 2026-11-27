"""Preprocessing utilities for single-sample calibration and tokenization.

This module provides helpers for preparing microRNA expression data before feeding
it into the highway BiLSTM model. Key operations include invariant-control
normalization, rule-sequence computation, ANOVA-based feature selection, and
reshaping flattened feature vectors into token grids.
"""

from __future__ import annotations

from typing import Sequence
import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif

from highway_bilstm.constants import RULE_PAIRS
from highway_bilstm.data.normalization import normalize_invariant_controls


def compute_rule_sequence(
    df: pd.DataFrame,
    rule_pairs: Sequence[tuple[str, str]] = RULE_PAIRS,
) -> np.ndarray:
    """Compute a rule-based feature sequence from paired gene expression columns.

    For each pair (g1, g2), three features are derived:
        1. Binary indicator whether g1 > g2.
        2. Log2 difference between the two values.
        3. Hyperbolic tangent of that difference.

    All features are concatenated along a new axis to form a token-ready tensor.

    Args:
        df: DataFrame containing expression data.
        rule_pairs: List of tuples, each containing two gene names.

    Returns:
        NumPy array with shape (n_samples, n_rules * 3) containing the
            stacked rule features.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     "g1": [1.0, 2.0],
        ...     "g2": [0.5, 1.5]
        ... })
        >>> rules = [("g1", "g2")]
        >>> seq = compute_rule_sequence(df, rules)
        >>> seq.shape
        (2, 3)
    """
    seq_features = []
    for g1, g2 in rule_pairs:
        x1 = df[g1].values.astype(np.float32)
        x2 = df[g2].values.astype(np.float32)
        b_k = (x1 > x2).astype(np.float32)
        delta_k = np.log2(np.maximum(x1, 0.0) + 1.0) - np.log2(np.maximum(x2, 0.0) + 1.0)
        s_k = np.tanh(delta_k)
        seq_features.append(np.stack([b_k, delta_k, s_k], axis=1))
    return np.stack(seq_features, axis=1).astype(np.float32)


def select_top_anova_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    top_k: int = 150,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select the top-k features using ANOVA F-test for binary classification.

    Applies f_classif from sklearn to rank features by their F-statistic, then
    returns the top-k indices along with their corresponding F and p values.
    NaN and infinite F-values are coerced to 0, and p-values to 1.0 for safety.

    Args:
        X_train: Feature matrix of shape (n_samples, n_features).
        y_train: Target labels of shape (n_samples,).
        top_k: Number of features to keep.

    Returns:
        Tuple of:
            - sorted_idx: Indices of the top-k features.
            - f_vals: F-statistics for the selected features.
            - p_vals: p-values for the selected features.

    Example:
        >>> import numpy as np
        >>> X = np.random.rand(100, 50).astype(np.float32)
        >>> y = (np.random.rand(100) > 0.5).astype(int)
        >>> idx, f, p = select_top_anova_features(X, y, top_k=10)
        >>> idx.shape
        (10,)
    """
    f_vals, p_vals = f_classif(X_train, y_train)
    f_vals = np.nan_to_num(f_vals, nan=0.0, posinf=0.0, neginf=0.0)
    p_vals = np.nan_to_num(p_vals, nan=1.0)
    sorted_idx = np.argsort(f_vals)[::-1][:top_k]
    return sorted_idx, f_vals[sorted_idx], p_vals[sorted_idx]


def reshape_to_tokens(
    X_scaled: np.ndarray,
    seq_len: int = 30,
    step_dim: int = 5,
) -> np.ndarray:
    """Reshape a flat feature vector into a 3D token grid.

    This function converts a 2D array of shape (n_samples, seq_len * step_dim)
    into a 3D tensor of shape (n_samples, seq_len, step_dim), suitable for
    sequence-based models.

    Args:
        X_scaled: Input array of shape (n_samples, seq_len * step_dim).
        seq_len: Number of tokens in the sequence dimension.
        step_dim: Number of features per token.

    Returns:
        Reshaped array with shape (n_samples, seq_len, step_dim).

    Raises:
        ValueError: If the input is not 2D or the second dimension does not
            match the expected size (seq_len * step_dim).

    Example:
        >>> import numpy as np
        >>> X = np.random.rand(5, 150).astype(np.float32)
        >>> tokens = reshape_to_tokens(X, seq_len=30, step_dim=5)
        >>> tokens.shape
        (5, 30, 5)
    """
    X_arr = np.asarray(X_scaled, dtype=np.float32)
    expected = seq_len * step_dim
    if X_arr.ndim != 2 or X_arr.shape[1] != expected:
        raise ValueError(
            f"Expected 2D array with {expected} features ({seq_len}x{step_dim}), got {X_arr.shape}."
        )
    return X_arr.reshape(-1, seq_len, step_dim)
