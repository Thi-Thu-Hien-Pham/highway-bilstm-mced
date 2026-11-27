"""Unit tests for evaluation metrics, OvR specificity, stratification, and PPV/NPV."""

from __future__ import annotations

import math
import numpy as np

from highway_bilstm.evaluation.metrics import (
    compute_binary_metrics,
    compute_control_stratified_specificity,
    compute_ppv_npv,
    compute_stage_stratified_binary_sensitivity,
    compute_stage_stratified_too_metrics,
    compute_too_metrics,
)


def test_compute_binary_metrics() -> None:
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1, 1, 0, 0])
    y_prob = np.array([0.1, 0.2, 0.3, 0.6, 0.9, 0.8, 0.4, 0.3])

    out = compute_binary_metrics(y_true, y_pred, y_prob)
    assert math.isclose(out["accuracy"], 5 / 8, rel_tol=1e-6)
    assert math.isclose(out["sensitivity"], 0.5, rel_tol=1e-6)
    assert math.isclose(out["specificity"], 0.75, rel_tol=1e-6)
    assert math.isclose(out["precision"], 2 / 3, rel_tol=1e-6)
    assert "roc_auc" in out and 0.0 <= out["roc_auc"] <= 1.0


def test_compute_too_metrics_ovr_specificity() -> None:
    classes = ("A", "B", "C")
    full_names = {"A": "Cancer A", "B": "Cancer B", "C": "Cancer C"}
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 0, 1, 0, 1, 1, 2, 2])
    y_prob = np.eye(3)[y_pred]

    out = compute_too_metrics(y_true, y_pred, y_prob, class_names=classes, full_names=full_names)
    df = out["per_class_df"]

    assert out["summary"]["n_samples"] == 10
    assert math.isclose(out["summary"]["accuracy"], 0.8, rel_tol=1e-6)
    # Class A: TP=3, FN=1, FP=1, TN=5 -> OvR spec = 5/6
    spec_a = df.loc[df["Code"] == "A", "OvR_Specificity"].iloc[0]
    assert math.isclose(spec_a, 5 / 6, rel_tol=1e-6)
    # Class C: TP=2, FN=0, FP=0, TN=8 -> OvR spec = 1.0
    spec_c = df.loc[df["Code"] == "C", "OvR_Specificity"].iloc[0]
    assert math.isclose(spec_c, 1.0, rel_tol=1e-6)


def test_stage_stratified_metrics() -> None:
    y_true = np.array([0, 1, 1, 1, 1])
    y_pred = np.array([0, 1, 0, 1, 1])
    stages = ["Unstaged", "Stage 0", "Stage I", "Stage III", "Stage IV"]

    bin_s = compute_stage_stratified_binary_sensitivity(y_true, y_pred, stages)
    assert bin_s["Combined Stages 0-II"]["n"] == 2
    assert math.isclose(bin_s["Combined Stages 0-II"]["sensitivity"], 0.5, rel_tol=1e-6)
    assert math.isclose(bin_s["Combined Stages III-IV"]["sensitivity"], 1.0, rel_tol=1e-6)

    too_s = compute_stage_stratified_too_metrics(
        np.array([0, 1, 2, 0]),
        np.array([0, 1, 0, 0]),
        ["Stage 0", "Stage I", "Stage III", "Unstaged"],
    )
    assert math.isclose(too_s["Combined Stages 0-II"]["accuracy"], 1.0, rel_tol=1e-6)
    assert math.isclose(too_s["Stage III"]["accuracy"], 0.0, rel_tol=1e-6)


def test_control_stratified_specificity_and_ppv_npv() -> None:
    y_true = np.array([0, 0, 0, 0, 1])
    y_pred = np.array([0, 0, 0, 1, 1])
    healthy = np.array([True, True, False, False, False])
    benign = np.array([False, False, True, True, False])
    organs = np.array(["VOL", "VOL", "PR", "SA", "BC"])

    spec = compute_control_stratified_specificity(y_true, y_pred, healthy, benign, organs)
    assert math.isclose(spec["healthy"]["specificity"], 1.0, rel_tol=1e-6)
    assert math.isclose(spec["benign"]["specificity"], 0.5, rel_tol=1e-6)
    assert math.isclose(spec["benign_by_subgroup"]["PR"]["specificity"], 1.0, rel_tol=1e-6)
    assert math.isclose(spec["benign_by_subgroup"]["SA"]["specificity"], 0.0, rel_tol=1e-6)

    proj = compute_ppv_npv(0.9815, 0.9435, prevalences=(0.01,))
    assert math.isclose(proj[0]["ppv"], 0.14927, abs_tol=1e-3)
    assert math.isclose(proj[0]["npv"], 0.99980, abs_tol=1e-4)
