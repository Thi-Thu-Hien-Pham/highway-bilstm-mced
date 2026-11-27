"""Zero-shot binary pan-cancer evaluation across the four external GPL21263 GEO cohorts (N=7,189)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from highway_bilstm.constants import (
    EXTERNAL_COHORT_SPECS,
    EXTERNAL_DESC_TO_CODE,
    FIXED_EXTERNAL_RULE_THRESHOLD,
    INVARIANT_CONTROLS,
    RULE_PAIRS,
)
from highway_bilstm.data.external import (
    classify_external_control_subgroups,
    load_mimat_mapping,
    parse_external_series_matrix,
)
from highway_bilstm.data.preprocessing import normalize_invariant_controls
from highway_bilstm.evaluation.metrics import (
    compute_binary_metrics,
    compute_control_stratified_specificity,
)
from highway_bilstm.models.baselines import MiRBiTBaseline, RuleScoreClassifier


def main() -> None:
    parser = argparse.ArgumentParser(description="Zero-shot binary screening validation on external GPL21263 GEO cohorts.")
    parser.add_argument("--external-dir", default="data_cache/external_cohorts", help="Directory containing GSE*_series_matrix.txt.gz.")
    parser.add_argument("--meta-dir", default="pancancermirna data/test", help="Directory containing GSE*_metadatah.csv.")
    parser.add_argument("--mimat-map", default="configs/gpl21263_mimat_to_mirna.json", help="Path to MIMAT->miRNA JSON mapping.")
    parser.add_argument("--raw-dir", default=None, help="Optional fallback GSE211692_RAW directory if --mimat-map is absent.")
    parser.add_argument("--threshold", type=float, default=FIXED_EXTERNAL_RULE_THRESHOLD, help="Prespecified 24-rule score threshold (default: 9/24 = 0.375).")
    parser.add_argument("--output-dir", default="outputs/external_binary", help="Output directory.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mimat_to_mirna = load_mimat_mapping(json_path=args.mimat_map, raw_dir=args.raw_dir)

    needed_mirnas = sorted({m for pair in RULE_PAIRS for m in pair} | set(INVARIANT_CONTROLS))
    rule_clf = RuleScoreClassifier(pairs=RULE_PAIRS, threshold=args.threshold)
    mirbit_clf = MiRBiTBaseline()

    summary_rows = []
    detailed_report = {}

    for acc, spec in EXTERNAL_COHORT_SPECS.items():
        gz_path = Path(args.external_dir) / spec["matrix_file"]
        meta_path = Path(args.meta_dir) / spec["meta_file"]
        df_expr, y_true, descriptions, _ = parse_external_series_matrix(
            gz_path, meta_path, mimat_to_mirna, target_mirnas=needed_mirnas
        )
        norm_arr = normalize_invariant_controls(df_expr.values, list(df_expr.columns))
        df_norm = pd.DataFrame(norm_arr, index=df_expr.index, columns=df_expr.columns)

        scores = rule_clf.score(df_norm)
        preds = (scores >= args.threshold).astype(np.int64)
        mirbit_preds = mirbit_clf.predict(df_norm)

        full_m = compute_binary_metrics(y_true, preds, scores)
        mb_m = compute_binary_metrics(y_true, mirbit_preds)
        healthy_mask, benign_mask = classify_external_control_subgroups(acc, y_true, descriptions)
        ctrl_spec = compute_control_stratified_specificity(y_true, preds, healthy_mask, benign_mask)

        # Target-organ subset (target organ cancers vs controls)
        target_organ_label = spec["target_organ"]
        target_code = spec["target_code"]
        mapped_codes = np.array([EXTERNAL_DESC_TO_CODE.get(d, "") for d in descriptions])
        target_subset_mask = (mapped_codes == target_code) | (y_true == 0)
        target_m = compute_binary_metrics(y_true[target_subset_mask], preds[target_subset_mask], scores[target_subset_mask])

        healthy_only_mask = (mapped_codes == target_code) | healthy_mask
        target_vs_healthy = compute_binary_metrics(
            y_true[healthy_only_mask], preds[healthy_only_mask], scores[healthy_only_mask]
        )

        summary_rows.append(
            {
                "Accession": acc,
                "N_Samples": int(full_m["n_samples"]),
                "Cancer": int(full_m["n_cancer"]),
                "Control": int(full_m["n_control"]),
                "Healthy_Ctrl": ctrl_spec["healthy"]["n"],
                "Benign_Ctrl": ctrl_spec["benign"]["n"],
                "AUC": round(full_m["roc_auc"], 4),
                "Accuracy": round(full_m["accuracy"], 4),
                "Sensitivity": round(full_m["sensitivity"], 4),
                "Specificity": round(full_m["specificity"], 4),
                "Healthy_Specificity": round(ctrl_spec["healthy"]["specificity"], 4),
                "Benign_Specificity": (
                    round(ctrl_spec["benign"]["specificity"], 4)
                    if ctrl_spec["benign"]["n"] > 0
                    else float("nan")
                ),
                "Precision": round(full_m["precision"], 4),
                "F1": round(full_m["f1"], 4),
                "miRBiT_Rerun_Acc": round(mb_m["accuracy"], 4),
                "miRBiT_Rerun_Spec": round(mb_m["specificity"], 4),
                "Target_Organ": target_organ_label,
                "Target_Only_AUC": round(target_m["roc_auc"], 4),
                "Target_Only_Sens": round(target_m["sensitivity"], 4),
                "Target_vs_Healthy_AUC": round(target_vs_healthy["roc_auc"], 4),
            }
        )
        detailed_report[acc] = {
            "full_cohort": full_m,
            "control_stratified": ctrl_spec,
            "target_organ_vs_all_controls": target_m,
            "target_organ_vs_healthy_controls": target_vs_healthy,
            "mirbit_4_rule_rerun": mb_m,
        }

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(out_dir / "external_binary_summary.csv", index=False)
    with open(out_dir / "external_binary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(detailed_report, f, indent=2)

    print(df_summary[["Accession", "N_Samples", "AUC", "Accuracy", "Sensitivity", "Specificity", "Healthy_Specificity", "Benign_Specificity", "F1"]].to_string(index=False))
    print(f"\nSaved external binary evaluation to {out_dir}")


if __name__ == "__main__":
    main()
