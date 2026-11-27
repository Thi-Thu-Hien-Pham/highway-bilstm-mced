"""Zero-shot external 13-class Tissue-of-Origin evaluation of the trained Highway-BiLSTM model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score

from highway_bilstm.constants import (
    CANCER_CLASSES,
    EXTERNAL_COHORT_SPECS,
    EXTERNAL_DESC_TO_CODE,
    INVARIANT_CONTROLS,
    SEED,
    set_seed,
)
from highway_bilstm.data.external import load_mimat_mapping, parse_external_series_matrix
from highway_bilstm.data.gse211692 import load_full_mirnome_npz
from highway_bilstm.data.preprocessing import normalize_invariant_controls, reshape_to_tokens
from highway_bilstm.data.splits import split_task2_too
from highway_bilstm.models.highway_bilstm import HighwayBiLSTM


def main() -> None:
    parser = argparse.ArgumentParser(description="Zero-shot 13-class TOO evaluation on external GPL21263 cohorts.")
    parser.add_argument("--checkpoint", required=True, help="Path to trained Optuna Highway-BiLSTM .pt state_dict.")
    parser.add_argument("--config", default="configs/best_hyperparameters.json", help="Path to model hyperparameter JSON.")
    parser.add_argument("--discovery-npz", default="data_cache/full_mirnome_matrix.npz", help="Discovery NPZ to reconstruct training top-150 features and StandardScaler.")
    parser.add_argument("--external-dir", default="data_cache/external_cohorts", help="Directory containing GSE*_series_matrix.txt.gz.")
    parser.add_argument("--meta-dir", default="pancancermirna data/test", help="Directory containing GSE*_metadatah.csv.")
    parser.add_argument("--mimat-map", default="configs/gpl21263_mimat_to_mirna.json", help="Path to MIMAT->miRNA JSON mapping.")
    parser.add_argument("--output-dir", default="outputs/external_too", help="Output directory.")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config, "r", encoding="utf-8") as f:
        best_p = json.load(f)

    model = HighwayBiLSTM.from_config(best_p, num_classes=len(CANCER_CLASSES)).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()

    X_raw, sample_ids, organs, binary_classes, mirna_names = load_full_mirnome_npz(args.discovery_npz)
    norm_X = normalize_invariant_controls(X_raw, mirna_names)
    splits = split_task2_too(norm_X, organs, binary_classes, sample_ids=sample_ids, seed=args.seed)
    top_150_names = [mirna_names[i] for i in splits["top_idx_sub"]]
    scaler = splits["scaler_full_opt"]

    mimat_to_mirna = load_mimat_mapping(json_path=args.mimat_map)
    target_mirnas = sorted(set(top_150_names) | set(INVARIANT_CONTROLS))
    c2i = {c: i for i, c in enumerate(CANCER_CLASSES)}

    summary_rows = []
    per_organ_details = {}

    for acc, spec in EXTERNAL_COHORT_SPECS.items():
        gz_path = Path(args.external_dir) / spec["matrix_file"]
        meta_path = Path(args.meta_dir) / spec["meta_file"]
        df_expr, _, descriptions, _ = parse_external_series_matrix(
            gz_path, meta_path, mimat_to_mirna, target_mirnas=target_mirnas
        )

        # Keep only confirmed cancer specimens matching one of the 13 solid tumor codes
        keep_mask = np.array([d in EXTERNAL_DESC_TO_CODE for d in descriptions])
        df_cancer = df_expr.iloc[keep_mask]
        y_true_codes = [EXTERNAL_DESC_TO_CODE[d] for d in descriptions[keep_mask]]
        y_true = np.array([c2i[c] for c in y_true_codes], dtype=np.int64)

        norm_arr = normalize_invariant_controls(df_cancer.values, list(df_cancer.columns))
        df_norm = pd.DataFrame(norm_arr, index=df_cancer.index, columns=df_cancer.columns)

        aligned = np.zeros((len(df_norm), 150), dtype=np.float32)
        for pos, mname in enumerate(top_150_names):
            if mname in df_norm.columns:
                aligned[:, pos] = df_norm[mname].values

        X_s = scaler.transform(aligned)
        X_seq = torch.tensor(reshape_to_tokens(X_s, 30, 5), dtype=torch.float32, device=device)
        X_flat = torch.tensor(X_s, dtype=torch.float32, device=device)

        with torch.no_grad():
            logits, _ = model(X_seq, X_flat)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

        overall_acc = accuracy_score(y_true, preds)
        target_code = spec["target_code"]
        target_mask = np.array(y_true_codes) == target_code
        target_recall = (
            accuracy_score(y_true[target_mask], preds[target_mask])
            if np.sum(target_mask) > 0
            else 0.0
        )

        summary_rows.append(
            {
                "Cohort": acc,
                "Target_Organ": spec["target_organ"],
                "N_Target": int(np.sum(target_mask)),
                "Target_Recall_pct": round(target_recall * 100.0, 2),
                "Target_Correct": int(np.sum(preds[target_mask] == y_true[target_mask])),
                "N_All_Cancer": len(y_true),
                "Overall_TOO_Accuracy_pct": round(overall_acc * 100.0, 2),
                "Overall_TOO_Correct": int(np.sum(preds == y_true)),
            }
        )

        organ_breakdown = {}
        for code in sorted(set(y_true_codes)):
            cmask = np.array(y_true_codes) == code
            organ_breakdown[code] = {
                "n": int(np.sum(cmask)),
                "correct": int(np.sum(preds[cmask] == y_true[cmask])),
                "recall_pct": round(float(accuracy_score(y_true[cmask], preds[cmask]) * 100.0), 2),
            }
        per_organ_details[acc] = organ_breakdown

    df_out = pd.DataFrame(summary_rows)
    df_out.to_csv(out_dir / "external_cohorts_too_evaluation.csv", index=False)
    with open(out_dir / "external_cohorts_too_details.json", "w", encoding="utf-8") as f:
        json.dump(per_organ_details, f, indent=2)

    print(df_out.to_string(index=False))
    print(f"\nSaved external TOO evaluation to {out_dir}")


if __name__ == "__main__":
    main()
