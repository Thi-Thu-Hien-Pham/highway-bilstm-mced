"""Train and evaluate the baseline 13-class Highway-BiLSTM (Task 2 Tissue-of-Origin)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from highway_bilstm.constants import CANCER_CLASSES, SEED, set_seed
from highway_bilstm.data.gse211692 import load_full_mirnome_npz, parse_clinical_stage_map
from highway_bilstm.data.preprocessing import normalize_invariant_controls, reshape_to_tokens
from highway_bilstm.data.splits import split_task2_too
from highway_bilstm.evaluation.metrics import (
    compute_stage_stratified_too_metrics,
    compute_too_metrics,
)
from highway_bilstm.models.highway_bilstm import HighwayBiLSTM


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate baseline 13-class Highway-BiLSTM (Task 2 TOO).")
    parser.add_argument("--npz", default="data_cache/full_mirnome_matrix.npz", help="Path to full_mirnome_matrix.npz.")
    parser.add_argument("--clinical-csv", default="data_cache/GSE211692_clinical_metadata.csv", help="Optional clinical metadata CSV/gz.")
    parser.add_argument("--output-dir", default="outputs/task2_too_baseline", help="Output directory.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_raw, sample_ids, organs, binary_classes, mirna_names = load_full_mirnome_npz(args.npz)
    norm_X = normalize_invariant_controls(X_raw, mirna_names)
    splits = split_task2_too(norm_X, organs, binary_classes, sample_ids=sample_ids, seed=args.seed)

    X_tr_s = splits["X_tr_full_base_s"]
    X_te_s = splits["X_test_full_base_s"]
    y_train = splits["y_train_full"]
    y_test = splits["y_test"]

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(reshape_to_tokens(X_tr_s, 30, 5), dtype=torch.float32),
            torch.tensor(X_tr_s, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(
            torch.tensor(reshape_to_tokens(X_te_s, 30, 5), dtype=torch.float32),
            torch.tensor(X_te_s, dtype=torch.float32),
        ),
        batch_size=128,
        shuffle=False,
    )

    model = HighwayBiLSTM.from_baseline_config(num_classes=len(CANCER_CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for xs_b, xf_b, y_b in train_loader:
            xs_b, xf_b, y_b = xs_b.to(device), xf_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits, _ = model(xs_b, xf_b)
            loss = criterion(logits, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * len(y_b)
        scheduler.step()
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:02d}/{args.epochs} | Loss: {running_loss / len(X_tr_s):.4f}")

    model.eval()
    all_probs = []
    with torch.no_grad():
        for xs_b, xf_b in test_loader:
            logits, _ = model(xs_b.to(device), xf_b.to(device))
            all_probs.extend(torch.softmax(logits, dim=1).cpu().numpy())

    probs_arr = np.asarray(all_probs)
    preds = np.argmax(probs_arr, axis=1)
    too_res = compute_too_metrics(y_test, preds, probs_arr)

    stage_metrics = None
    if Path(args.clinical_csv).exists() and splits["test_sample_ids"] is not None:
        stage_map = parse_clinical_stage_map(args.clinical_csv)
        test_stages = [stage_map.get(str(sid), "Unstaged") for sid in splits["test_sample_ids"]]
        stage_metrics = compute_stage_stratified_too_metrics(y_test, preds, test_stages)

    # Save checkpoint, per-class metrics, confusion matrix, and token mapping
    torch.save(model.state_dict(), out_dir / "baseline_highway_bilstm.pt")
    too_res["per_class_df"].to_csv(out_dir / "per_class_metrics.csv", index=False)
    pd.DataFrame(too_res["confusion_matrix"], index=CANCER_CLASSES, columns=CANCER_CLASSES).to_csv(
        out_dir / "confusion_matrix.csv"
    )
    token_rows = [
        {
            "Global_Rank": rank + 1,
            "Token_ID": f"Token {rank // 5 + 1}",
            "Position_in_Token": (rank % 5) + 1,
            "microRNA_Name": mirna_names[f_idx],
            "ANOVA_F_Statistic": float(splits["f_vals_full"][rank]),
            "p_value": float(splits["p_vals_full"][rank]),
        }
        for rank, f_idx in enumerate(splits["top_idx_full"])
    ]
    pd.DataFrame(token_rows).to_csv(out_dir / "anova_tokenization_mapping.csv", index=False)

    summary = {"too_summary": too_res["summary"], "stage_stratified": stage_metrics}
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    s = too_res["summary"]
    print(f"\n[Task 2 - Baseline Highway-BiLSTM] Acc: {s['accuracy']*100:.2f}% | "
          f"Weighted F1: {s['weighted_f1']:.4f} | Macro F1: {s['macro_f1']:.4f} | "
          f"Macro AUC: {s['macro_ovr_auc']:.4f}")
    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
