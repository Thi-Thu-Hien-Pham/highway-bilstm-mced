"""Leakage-free stratified train/validation/test partitioning for Tasks 1 and 2."""

from __future__ import annotations

from typing import Any
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from highway_bilstm.constants import CANCER_CLASSES, SEED
from highway_bilstm.data.preprocessing import select_top_anova_features


def split_task1_binary(
    organs: np.ndarray,
    binary_classes: np.ndarray,
    test_size: float = 0.20,
    seed: int = SEED,
) -> dict[str, Any]:
    """Create the 80/20 stratified split (12,952 train / 3,238 test) for Task 1.

    Stratifies by `organs` so every solid cancer organ and benign/healthy
    control group maintains exact 80/20 representation across splits.
    Healthy controls correspond to `(binary == 'non_cancer') & (organs == 'VOL')`,
    while benign controls correspond to `(binary == 'non_cancer') & (organs != 'VOL')`.
    """
    indices = np.arange(len(organs))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=organs,
    )
    y_binary = (binary_classes == "cancer").astype(np.int64)
    is_healthy = (binary_classes == "non_cancer") & (organs == "VOL")
    is_benign = (binary_classes == "non_cancer") & (organs != "VOL")
    return {
        "train_idx": train_idx,
        "test_idx": test_idx,
        "y_binary": y_binary,
        "is_healthy": is_healthy,
        "is_benign": is_benign,
    }


def split_task2_too(
    norm_X: np.ndarray,
    organs: np.ndarray,
    binary_classes: np.ndarray,
    sample_ids: np.ndarray | None = None,
    test_size: float = 0.20,
    val_size: float = 0.15,
    top_k: int = 150,
    seed: int = SEED,
) -> dict[str, Any]:
    """Create leakage-free Task 2 splits (7,936 train / 1,985 test) and 85/15 tuning split.

    ANOVA feature selection (`f_classif`) and `StandardScaler` are fitted strictly
    on the training split (`X_tr_sub_raw` for tuning and `X_tr_raw` for full-train
    evaluation), matching the leakage-free protocol in Revision 3 Section 2.3.
    """
    c2i = {c: i for i, c in enumerate(CANCER_CLASSES)}
    cancer_mask = (binary_classes == "cancer") & np.isin(organs, CANCER_CLASSES)
    X_cancer = norm_X[cancer_mask]
    y_cancer = np.array([c2i[c] for c in organs[cancer_mask]], dtype=np.int64)
    cancer_sample_ids = sample_ids[cancer_mask] if sample_ids is not None else None

    indices = np.arange(len(X_cancer))
    tr_full_rel_idx, test_rel_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=y_cancer,
    )
    X_tr_raw, X_test_raw = X_cancer[tr_full_rel_idx], X_cancer[test_rel_idx]
    y_train_full, y_test = y_cancer[tr_full_rel_idx], y_cancer[test_rel_idx]

    tr_sub_rel_idx, val_rel_idx = train_test_split(
        np.arange(len(X_tr_raw)),
        test_size=val_size,
        random_state=seed,
        stratify=y_train_full,
    )
    X_tr_sub_raw, X_val_raw = X_tr_raw[tr_sub_rel_idx], X_tr_raw[val_rel_idx]
    y_train_sub, y_val = y_train_full[tr_sub_rel_idx], y_train_full[val_rel_idx]

    # Fit ANOVA and scalers strictly on training partitions
    top_idx_sub, f_vals_sub, p_vals_sub = select_top_anova_features(X_tr_sub_raw, y_train_sub, top_k=top_k)
    scaler_sub = StandardScaler()
    X_tr_sub_s = scaler_sub.fit_transform(X_tr_sub_raw[:, top_idx_sub])
    X_val_s = scaler_sub.transform(X_val_raw[:, top_idx_sub])
    X_test_sub_s = scaler_sub.transform(X_test_raw[:, top_idx_sub])

    # Baseline Task 2 uses ANOVA computed on the full 7,936 training split
    top_idx_full, f_vals_full, p_vals_full = select_top_anova_features(X_tr_raw, y_train_full, top_k=top_k)
    scaler_full_baseline = StandardScaler()
    X_tr_full_base_s = scaler_full_baseline.fit_transform(X_tr_raw[:, top_idx_full])
    X_test_full_base_s = scaler_full_baseline.transform(X_test_raw[:, top_idx_full])

    # Optuna retraining fits scaler on full 7,936 train using the tuning top_idx_sub
    scaler_full_opt = StandardScaler()
    X_tr_full_opt_s = scaler_full_opt.fit_transform(X_tr_raw[:, top_idx_sub])
    X_test_full_opt_s = scaler_full_opt.transform(X_test_raw[:, top_idx_sub])

    return {
        "X_tr_sub_s": X_tr_sub_s,
        "y_train_sub": y_train_sub,
        "X_val_s": X_val_s,
        "y_val": y_val,
        "X_test_sub_s": X_test_sub_s,
        "X_tr_full_base_s": X_tr_full_base_s,
        "X_test_full_base_s": X_test_full_base_s,
        "X_tr_full_opt_s": X_tr_full_opt_s,
        "X_test_full_opt_s": X_test_full_opt_s,
        "y_train_full": y_train_full,
        "y_test": y_test,
        "top_idx_sub": top_idx_sub,
        "f_vals_sub": f_vals_sub,
        "p_vals_sub": p_vals_sub,
        "top_idx_full": top_idx_full,
        "f_vals_full": f_vals_full,
        "p_vals_full": p_vals_full,
        "scaler_full_baseline": scaler_full_baseline,
        "scaler_full_opt": scaler_full_opt,
        "test_sample_ids": cancer_sample_ids[test_rel_idx] if cancer_sample_ids is not None else None,
        "train_sample_ids": cancer_sample_ids[tr_full_rel_idx] if cancer_sample_ids is not None else None,
    }
