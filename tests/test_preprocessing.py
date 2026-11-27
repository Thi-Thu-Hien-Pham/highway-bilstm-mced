"""Unit tests for preprocessing utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from highway_bilstm.constants import INVARIANT_CONTROLS, RULE_PAIRS
from highway_bilstm.data.normalization import (
    CALIBRATION_OFFSET,
    DETECTION_FLOOR,
    normalize_invariant_controls as normalize_direct,
)
from highway_bilstm.data.preprocessing import (
    compute_rule_sequence,
    normalize_invariant_controls,
    reshape_to_tokens,
    select_top_anova_features,
)


def test_normalize_invariant_controls_single_sample_invariance() -> None:
    assert normalize_direct is normalize_invariant_controls
    assert DETECTION_FLOOR == 0.1
    assert CALIBRATION_OFFSET == 10.9
    mirna_names = ["hsa-miR-21-5p", *INVARIANT_CONTROLS]
    x1 = np.array([10.0, 2.0, 4.0, 8.0], dtype=np.float32)
    x2 = np.array([100.0, 1.0, 1.0, 1.0], dtype=np.float32)

    out_single = normalize_invariant_controls(x1, mirna_names)
    out_batch = normalize_invariant_controls(np.stack([x1, x2], axis=0), mirna_names)

    assert out_single.ndim == 1
    assert out_batch.shape == (2, 4)
    np.testing.assert_allclose(out_batch[0], out_single, rtol=1e-5)


def test_normalize_invariant_controls_floor_and_missing_error() -> None:
    mirna_names = ["m1", "hsa-miR-149-3p"]
    x = np.array([-5.0, 0.0], dtype=np.float32)
    out = normalize_invariant_controls(x, mirna_names, floor=0.1, offset=10.9)
    np.testing.assert_allclose(out, np.array([10.9, 10.9], dtype=np.float32), rtol=1e-5)

    with pytest.raises(ValueError, match="None of invariant controls"):
        normalize_invariant_controls(x, ["m1", "m2"])


def test_compute_rule_sequence_shape_and_scale_invariance() -> None:
    df = pd.DataFrame(
        {
            "g1": [4.0, 1.0, 5.0],
            "g2": [1.0, 2.0, 5.0],
        }
    )
    seq = compute_rule_sequence(df, [("g1", "g2")])
    assert seq.shape == (3, 1, 3)
    assert seq.dtype == np.float32

    # Binary indicator b_k
    np.testing.assert_allclose(seq[:, 0, 0], [1.0, 0.0, 0.0])
    # Log2 ratio delta_k and tanh(delta_k)
    expected_delta = np.log2(5.0) - np.log2(2.0)
    np.testing.assert_allclose(seq[0, 0, 1], expected_delta, rtol=1e-5)
    np.testing.assert_allclose(seq[0, 0, 2], np.tanh(expected_delta), rtol=1e-5)

    # Monotonic rescaling preserves b_k ordering
    df_scaled = df * 100.0
    seq_scaled = compute_rule_sequence(df_scaled, [("g1", "g2")])
    np.testing.assert_array_equal(seq[:, 0, 0], seq_scaled[:, 0, 0])

    # Default 24 rules produce (N, 24, 3)
    full_df = pd.DataFrame(
        {m: np.linspace(1.0, 5.0, 4) for pair in RULE_PAIRS for m in pair}
    )
    full_seq = compute_rule_sequence(full_df)
    assert full_seq.shape == (4, len(RULE_PAIRS), 3)


def test_select_top_anova_features_descending() -> None:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(60, 20)).astype(np.float32)
    y = np.array([0] * 30 + [1] * 30)
    X[y == 1, 3] += 5.0  # Strong signal at feature 3

    idx, f_vals, p_vals = select_top_anova_features(X, y, top_k=10)
    assert idx.shape == (10,)
    assert idx[0] == 3
    assert np.all(np.diff(f_vals) <= 0)
    assert not np.any(np.isnan(f_vals))
    assert not np.any(np.isnan(p_vals))


def test_reshape_to_tokens_and_errors() -> None:
    X = np.zeros((4, 150), dtype=np.float64)
    tokens = reshape_to_tokens(X, seq_len=30, step_dim=5)
    assert tokens.shape == (4, 30, 5)
    assert tokens.dtype == np.float32

    with pytest.raises(ValueError, match="Expected 2D array"):
        reshape_to_tokens(np.zeros((150,), dtype=np.float32))
    with pytest.raises(ValueError, match="Expected 2D array with 150 features"):
        reshape_to_tokens(np.zeros((4, 100), dtype=np.float32))
