"""Train and evaluate the 150-feature dual-stream Highway-BiLSTM for Task 1 binary screening."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from highway_bilstm.constants import SEED, set_seed
from highway_bilstm.data.gse211692 import load_full_mirnome_npz, parse_clinical_stage_map
from highway_bilstm.data.preprocessing import (
    normalize_invariant_controls,
    reshape_to_tokens,
    select_top_anova_features,
)
from highway_bilstm.data.splits import split_task1_binary
from highway_bilstm.evaluation.metrics import (
    compute_binary_metrics,
    compute_control_stratified_specificity,
    compute_ppv_npv,
    compute_stage_stratified_binary_sensitivity,
)
from highway_bilstm.models.highway_bilstm import HighwayBiLSTM


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate 150-feature Highway-BiLSTM for Task 1 binary screening.")
    parser.add_argument("--npz", default="data_cache/full_mirnome_matrix.npz", help="Path to full_mirnome_matrix.npz.")
    parser.add_argument("--config", default="configs/best_hyperparameters.json", help="Model config JSON.")
    parser.add_argument("--clinical-csv", default="data_cache/GSE211692_clinical_metadata.csv", help="Optional clinical metadata CSV/gz.")
    parser.add_argument("--output-dir", default="outputs/task1_highway_bilstm", help="Output directory.")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    X_raw, sample_ids, organs, binary_classes, mirna_names = load_full_mirnome_npz(args.npz)
    norm_X = normalize_invariant_controls(X_raw, mirna_names)

    split = split_task1_binary(organs, binary_classes, test_size=0.20, seed=args.seed)
    train_idx, test_idx = split["train_idx"], split["test_idx"]
    y_binary = split["y_binary"]

    # Leakage-free ANOVA feature selection & scaling strictly on Task 1 training partition
    top_150_idx, _, _ = select_top_anova_features(norm_X[train_idx], y_binary[train_idx], top_k=150)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(norm_X[train_idx][:, top_150_idx])
    X_te_s = scaler.transform(norm_X[test_idx][:, top_150_idx])

    X_tr_seq = torch.tensor(reshape_to_tokens(X_tr_s, 30, 5), dtype=torch.float32)
    X_te_seq = torch.tensor(reshape_to_tokens(X_te_s, 30, 5), dtype=torch.float32)
    X_tr_flat = torch.tensor(X_tr_s, dtype=torch.float32)
    X_te_flat = torch.tensor(X_te_s, dtype=torch.float32)
    y_tr_t = torch.tensor(y_binary[train_idx], dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_tr_seq, X_tr_flat, y_tr_t), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_te_seq, X_te_flat), batch_size=128, shuffle=False)

    model = HighwayBiLSTM.from_config(cfg, num_classes=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for xs_b, xf_b, y_b in train_loader:
            xs_b, xf_b, y_b = xs_b.to(device), xf_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits, _ = model(xs_b, xf_b)
            loss = criterion(logits.squeeze(-1), y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * len(y_b)
        scheduler.step()
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:02d}/{args.epochs} | Loss: {running_loss / len(X_tr_seq):.4f}")

    model.eval()
    probs = []
    with torch.no_grad():
        for xs_b, xf_b in test_loader:
            logits, _ = model(xs_b.to(device), xf_b.to(device))
            probs.extend(torch.sigmoid(logits.squeeze(-1)).cpu().numpy())

    probs_arr = np.asarray(probs)
    preds = (probs_arr >= 0.5).astype(np.int64)
    y_test = y_binary[test_idx]

    binary_metrics = compute_binary_metrics(y_test, preds, probs_arr)
    ctrl_spec = compute_control_stratified_specificity(
        y_test,
        preds,
        healthy_mask=split["is_healthy"][test_idx],
        benign_mask=split["is_benign"][test_idx],
        organs=organs[test_idx],
    )
    ppv_npv = compute_ppv_npv(binary_metrics["sensitivity"], binary_metrics["specificity"])

    stage_metrics = None
    if Path(args.clinical_csv).exists():
        stage_map = parse_clinical_stage_map(args.clinical_csv)
        test_stages = [stage_map.get(str(sid), "Unstaged") for sid in sample_ids[test_idx]]
        stage_metrics = compute_stage_stratified_binary_sensitivity(y_test, preds, test_stages)

    torch.save(model.state_dict(), out_dir / "highway_bilstm_binary.pt")
    summary = {
        "binary_metrics": binary_metrics,
        "control_stratified_specificity": ctrl_spec,
        "ppv_npv_projections": ppv_npv,
        "stage_stratified_sensitivity": stage_metrics,
        "top_150_mirnas": [mirna_names[i] for i in top_150_idx],
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[Task 1 - 150-Feature Highway-BiLSTM] Acc: {binary_metrics['accuracy']*100:.2f}% | "
          f"Sens: {binary_metrics['sensitivity']*100:.2f}% | Spec: {binary_metrics['specificity']*100:.2f}% | "
          f"AUC: {binary_metrics['roc_auc']:.4f}")
    print(f"[Healthy vs Benign Spec] Healthy: {ctrl_spec['healthy']['specificity']*100:.2f}% | "
          f"Benign: {ctrl_spec['benign']['specificity']*100:.2f}%")
    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
