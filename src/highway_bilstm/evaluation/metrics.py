"""Statistical evaluation metrics, OvR per-class tables, stage stratification, and PPV/NPV.

This module provides evaluation utilities for the highway BiLSTM multi-class
cancer detection model.  It covers binary (cancer vs. control) summary
statistics, one-vs-rest per-class confusion-matrix tables, stage-stratified
sensitivity and accuracy breakdowns, control-subgroup specificity, and
Bayesian positive/negative predictive value calculations.
"""

from __future__ import annotations

from typing import Any, Sequence
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from highway_bilstm.constants import CANCER_CLASSES, CANCER_FULL_NAMES


def compute_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute summary metrics for a binary cancer vs. control split.

    Args:
        y_true: Ground-truth labels (0 = control, 1 = cancer).
        y_pred: Predicted labels (0 or 1).
        y_prob: Optional predicted positive-class probabilities. When supplied
            and both classes are present, a ROC AUC score is added to the
            result.

    Returns:
        A dictionary with keys ``n_samples``, ``n_cancer``, ``n_control``,
        ``accuracy``, ``sensitivity``, ``specificity``, ``precision``, and
        ``f1``. The key ``roc_auc`` is present only when ``y_prob`` is
        provided and both classes are represented in ``y_true``.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    metrics: dict[str, float] = {
        "n_samples": float(len(y_true)),
        "n_cancer": float(np.sum(y_true == 1)),
        "n_control": float(np.sum(y_true == 0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(recall_score(1 - y_true, 1 - y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_prob is not None and len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    return metrics


def compute_too_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    class_names: Sequence[str] = CANCER_CLASSES,
    full_names: dict[str, str] = CANCER_FULL_NAMES,
) -> dict[str, Any]:
    """Compute overall and per-class (one-vs-rest) multi-class metrics.

    Args:
        y_true: Integer-encoded ground-truth class indices.
        y_pred: Integer-encoded predicted class indices.
        y_prob: Optional array of shape ``(n_samples, n_classes)`` with
            predicted class probabilities. Used to compute a macro OvR AUC.
        class_names: Ordered class-code identifiers. Defaults to
            :data:`highway_bilstm.constants.CANCER_CLASSES`.
        full_names: Mapping from class code to human-readable cancer name.
            Defaults to :data:`highway_bilstm.constants.CANCER_FULL_NAMES`.

    Returns:
        A dictionary with three keys:

        * ``summary`` – aggregate metrics including sample count, accuracy,
          weighted and macro F1, and macro OvR AUC.
        * ``per_class_df`` – a :class:`pandas.DataFrame` with per-class
          sensitivity, OvR specificity, precision, F1, and support.
        * ``confusion_matrix`` – the raw :class:`numpy.ndarray` confusion
          matrix.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    num_classes = len(class_names)
    labels = list(range(num_classes))

    acc = float(accuracy_score(y_true, y_pred))
    weighted_f1 = float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))

    macro_auc = float("nan")
    if y_prob is not None:
        y_onehot = np.eye(num_classes)[y_true]
        present = y_onehot.sum(axis=0) > 0
        if np.all(present):
            macro_auc = float(roc_auc_score(y_onehot, y_prob, average="macro", multi_class="ovr"))
        else:
            aucs = [
                roc_auc_score(y_onehot[:, c], y_prob[:, c])
                for c in range(num_classes)
                if 0 < y_onehot[:, c].sum() < len(y_true)
            ]
            macro_auc = float(np.mean(aucs)) if aucs else float("nan")

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    per_class_rows = []
    for idx, code in enumerate(class_names):
        tp = int(cm[idx, idx])
        fn = int(cm[idx, :].sum() - tp)
        fp = int(cm[:, idx].sum() - tp)
        tn = int(len(y_true) - (tp + fn + fp))
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1_c = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0
        per_class_rows.append(
            {
                "Code": code,
                "Cancer Type": full_names.get(code, code),
                "Support": int(cm[idx, :].sum()),
                "Sensitivity": sens,
                "OvR_Specificity": spec,
                "Precision": prec,
                "F1_Score": f1_c,
            }
        )

    return {
        "summary": {
            "n_samples": len(y_true),
            "accuracy": acc,
            "weighted_f1": weighted_f1,
            "macro_f1": macro_f1,
            "macro_ovr_auc": macro_auc,
        },
        "per_class_df": pd.DataFrame(per_class_rows),
        "confusion_matrix": cm,
    }


def compute_stage_stratified_binary_sensitivity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    stages: Sequence[str],
) -> dict[str, dict[str, float]]:
    """Compute binary cancer-detection sensitivity stratified by stage.

    Args:
        y_true: Ground-truth binary labels (0 = control, 1 = cancer).
        y_pred: Predicted binary labels.
        stages: Stage string for each sample (e.g. ``"Stage 0"``,
            ``"Stage I"``, ``"Stage II"``, ``"Stage III"``, ``"Stage IV"``,
            or ``"Unstaged"``).

    Returns:
        A mapping from stage-group name to a dict with keys ``n`` (count of
        cancer samples in the group), ``detected`` (true positives), and
        ``sensitivity`` (ratio; NaN when no cancer samples are present).
        Groups include individual stages, combined 0-II and III-IV bands,
        and the unstaged remainder.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    stage_arr = np.asarray(stages, dtype=str)
    cancer_mask = y_true == 1

    groups = {
        "Stage 0": stage_arr == "Stage 0",
        "Stage I": stage_arr == "Stage I",
        "Stage II": stage_arr == "Stage II",
        "Combined Stages 0-II": np.isin(stage_arr, ["Stage 0", "Stage I", "Stage II"]),
        "Stage III": stage_arr == "Stage III",
        "Stage IV": stage_arr == "Stage IV",
        "Combined Stages III-IV": np.isin(stage_arr, ["Stage III", "Stage IV"]),
        "Unstaged": stage_arr == "Unstaged",
    }
    out: dict[str, dict[str, float]] = {}
    for label, mask in groups.items():
        m = cancer_mask & mask
        n = int(np.sum(m))
        tp = int(np.sum(y_pred[m] == 1)) if n > 0 else 0
        out[label] = {
            "n": n,
            "detected": tp,
            "sensitivity": (tp / n) if n > 0 else float("nan"),
        }
    return out


def compute_stage_stratified_too_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    stages: Sequence[str],
) -> dict[str, dict[str, float]]:
    """Compute multi-class accuracy and weighted F1 stratified by stage.

    Args:
        y_true: Integer-encoded ground-truth multi-class labels.
        y_pred: Integer-encoded predicted multi-class labels.
        stages: Stage string for each sample.

    Returns:
        A mapping from stage-group name to a dict with keys ``n``,
        ``correct``, ``accuracy``, and ``weighted_f1``. Groups and combined
        bands match those in
        :func:`compute_stage_stratified_binary_sensitivity`.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    stage_arr = np.asarray(stages, dtype=str)

    groups = {
        "Stage 0": stage_arr == "Stage 0",
        "Stage I": stage_arr == "Stage I",
        "Stage II": stage_arr == "Stage II",
        "Combined Stages 0-II": np.isin(stage_arr, ["Stage 0", "Stage I", "Stage II"]),
        "Stage III": stage_arr == "Stage III",
        "Stage IV": stage_arr == "Stage IV",
        "Combined Stages III-IV": np.isin(stage_arr, ["Stage III", "Stage IV"]),
        "Unstaged": stage_arr == "Unstaged",
    }
    out: dict[str, dict[str, float]] = {}
    for label, mask in groups.items():
        n = int(np.sum(mask))
        if n == 0:
            out[label] = {"n": 0, "correct": 0, "accuracy": float("nan"), "weighted_f1": float("nan")}
            continue
        correct = int(np.sum(y_pred[mask] == y_true[mask]))
        out[label] = {
            "n": n,
            "correct": correct,
            "accuracy": correct / n,
            "weighted_f1": float(f1_score(y_true[mask], y_pred[mask], average="weighted", zero_division=0)),
        }
    return out


def compute_control_stratified_specificity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    healthy_mask: np.ndarray,
    benign_mask: np.ndarray,
    organs: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute specificity for control subgroups.

    Args:
        y_true: Integer-encoded ground-truth labels where 0 denotes a control
            sample (healthy or benign).
        y_pred: Integer-encoded predicted labels.
        healthy_mask: Boolean array marking healthy controls.
        benign_mask: Boolean array marking benign controls.
        organs: Optional array of organ identifiers; when supplied, benign
            specificity is further broken down by organ.

    Returns:
        A dictionary with keys ``combined``, ``healthy``, and ``benign``,
        each containing ``n``, ``true_negative``, and ``specificity``. If
        ``organs`` is given, an additional ``benign_by_subgroup`` mapping is
        included.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    ctrl_mask = y_true == 0

    def _spec(mask: np.ndarray) -> dict[str, float]:
        n = int(np.sum(mask))
        tn = int(np.sum(y_pred[mask] == 0)) if n > 0 else 0
        return {
            "n": n,
            "true_negative": tn,
            "specificity": (tn / n) if n > 0 else float("nan"),
        }

    res: dict[str, Any] = {
        "combined": _spec(ctrl_mask),
        "healthy": _spec(ctrl_mask & healthy_mask),
        "benign": _spec(ctrl_mask & benign_mask),
    }
    if organs is not None:
        benign_by_organ: dict[str, dict[str, float]] = {}
        b_mask = ctrl_mask & benign_mask
        for org in sorted(np.unique(organs[b_mask])):
            benign_by_organ[str(org)] = _spec(b_mask & (organs == org))
        res["benign_by_subgroup"] = benign_by_organ
    return res


def compute_ppv_npv(
    sensitivity: float,
    specificity: float,
    prevalences: Sequence[float] = (0.005, 0.01, 0.05),
) -> list[dict[str, float]]:
    """Compute Bayesian PPV and NPV across a set of disease prevalences.

    Args:
        sensitivity: True-positive rate of the detector.
        specificity: True-negative rate of the detector.
        prevalences: Iterable of disease-prevalence values to evaluate.
            Defaults to ``(0.005, 0.01, 0.05)``.

    Returns:
        A list of dictionaries, one per prevalence, each containing
        ``prevalence``, ``ppv``, and ``npv``.
    """
    results = []
    for p in prevalences:
        tp_rate = sensitivity * p
        fp_rate = (1.0 - specificity) * (1.0 - p)
        tn_rate = specificity * (1.0 - p)
        fn_rate = (1.0 - sensitivity) * p
        ppv = tp_rate / (tp_rate + fp_rate) if (tp_rate + fp_rate) > 0 else 0.0
        npv = tn_rate / (tn_rate + fn_rate) if (tn_rate + fn_rate) > 0 else 0.0
        results.append(
            {
                "prevalence": float(p),
                "ppv": float(ppv),
                "npv": float(npv),
            }
        )
    return results
