"""Train and evaluate the 24-rule RuleChainingBiLSTM (Task 1 binary + naive TOO ablation) and 4-rule miRBiT baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from highway_bilstm.constants import CANCER_CLASSES, RULE_PAIRS, SEED, set_seed
from highway_bilstm.data.gse211692 import load_32_mirna_parquet, parse_clinical_stage_map
from highway_bilstm.data.preprocessing import compute_rule_sequence
from highway_bilstm.data.splits import split_task1_binary
from highway_bilstm.evaluation.metrics import (
    compute_binary_metrics,
    compute_control_stratified_specificity,
    compute_ppv_npv,
    compute_stage_stratified_binary_sensitivity,
    compute_too_metrics,
)
from highway_bilstm.models.baselines import MiRBiTBaseline
from highway_bilstm.models.rule_bilstm import RuleChainingBiLSTM


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate 24-rule RuleChainingBiLSTM on GSE211692.")
    parser.add_argument("--parquet", default="data_cache/mirna_32_dataset.parquet", help="Path to mirna_32_dataset.parquet.")
    parser.add_argument("--clinical-csv", default="data_cache/GSE211692_clinical_metadata.csv", help="Optional clinical metadata CSV/gz for stage stratification.")
    parser.add_argument("--output-dir", default="outputs/task1_rule_bilstm", help="Output directory for checkpoints and metrics.")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_32_mirna_parquet(args.parquet)
    X_seq = compute_rule_sequence(df, RULE_PAIRS)
    split = split_task1_binary(df["organ"].values, df["binary_class"].values, test_size=0.20, seed=args.seed)
    train_idx, test_idx = split["train_idx"], split["test_idx"]
    y_binary = split["y_binary"]

    c2i = {c: i for i, c in enumerate(CANCER_CLASSES)}
    y_too = np.array([c2i.get(o, -1) for o in df["organ"].values], dtype=np.int64)
    is_cancer = y_binary == 1

    train_ds = TensorDataset(
        torch.tensor(X_seq[train_idx], dtype=torch.float32),
        torch.tensor(y_binary[train_idx], dtype=torch.float32),
        torch.tensor(y_too[train_idx], dtype=torch.long),
        torch.tensor(is_cancer[train_idx], dtype=torch.bool),
    )
    test_ds = TensorDataset(
        torch.tensor(X_seq[test_idx], dtype=torch.float32),
        torch.tensor(y_binary[test_idx], dtype=torch.float32),
        torch.tensor(y_too[test_idx], dtype=torch.long),
        torch.tensor(is_cancer[test_idx], dtype=torch.bool),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    model = RuleChainingBiLSTM(num_rules=len(RULE_PAIRS)).to(device)
    bce_loss = nn.BCEWithLogitsLoss()
    ce_loss = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for x_b, y_bin_b, y_too_b, c_mask_b in train_loader:
            x_b = x_b.to(device)
            y_bin_b = y_bin_b.to(device)
            y_too_b = y_too_b.to(device)
            c_mask_b = c_mask_b.to(device)
            optimizer.zero_grad()
            bin_logits, too_logits, _ = model(x_b)
            loss_bin = bce_loss(bin_logits, y_bin_b)
            loss_too = ce_loss(too_logits[c_mask_b], y_too_b[c_mask_b]) if c_mask_b.sum() > 0 else 0.0
            loss = loss_bin + 0.8 * loss_too
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(x_b)
        scheduler.step()
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:02d}/{args.epochs} | Loss: {running_loss / len(train_ds):.4f}")

    model.eval()
    bin_probs, too_probs, attn_list = [], [], []
    with torch.no_grad():
        for x_b, _, _, _ in test_loader:
            b_log, t_log, attn_w = model(x_b.to(device))
            bin_probs.extend(torch.sigmoid(b_log).cpu().numpy())
            too_probs.extend(torch.softmax(t_log, dim=1).cpu().numpy())
            attn_list.append(attn_w.cpu().numpy())

    bin_probs_arr = np.asarray(bin_probs)
    bin_preds = (bin_probs_arr >= 0.5).astype(np.int64)
    too_probs_arr = np.asarray(too_probs)
    too_preds = np.argmax(too_probs_arr, axis=1)
    mean_attn = np.vstack(attn_list).mean(axis=0)

    y_test_bin = y_binary[test_idx]
    binary_metrics = compute_binary_metrics(y_test_bin, bin_preds, bin_probs_arr)
    ctrl_spec = compute_control_stratified_specificity(
        y_test_bin,
        bin_preds,
        healthy_mask=split["is_healthy"][test_idx],
        benign_mask=split["is_benign"][test_idx],
        organs=df["organ"].values[test_idx],
    )
    ppv_npv = compute_ppv_npv(binary_metrics["sensitivity"], binary_metrics["specificity"])

    # Naive rule-chaining TOO ablation on cancer samples in binary test split
    test_cancer_mask = is_cancer[test_idx] & (y_too[test_idx] >= 0)
    too_ablation = compute_too_metrics(
        y_too[test_idx][test_cancer_mask],
        too_preds[test_cancer_mask],
        too_probs_arr[test_cancer_mask],
    )["summary"]

    # 4-rule miRBiT baseline on same test split
    mirbit_preds = MiRBiTBaseline().predict(df.iloc[test_idx].reset_index(drop=True))
    mirbit_metrics = compute_binary_metrics(y_test_bin, mirbit_preds)

    stage_metrics = None
    if Path(args.clinical_csv).exists():
        stage_map = parse_clinical_stage_map(args.clinical_csv)
        test_stages = [stage_map.get(str(sid), "Unstaged") for sid in df["sample_id"].values[test_idx]]
        stage_metrics = compute_stage_stratified_binary_sensitivity(y_test_bin, bin_preds, test_stages)

    # Save artifacts
    torch.save(model.state_dict(), out_dir / "rule_bilstm.pt")
    attn_df = pd.DataFrame(
        [
            {
                "rule_index": i + 1,
                "mirna_a": pair[0],
                "mirna_b": pair[1],
                "mean_attention_weight": float(mean_attn[i]),
            }
            for i, pair in enumerate(RULE_PAIRS)
        ]
    ).sort_values("mean_attention_weight", ascending=False)
    attn_df.to_csv(out_dir / "rule_attention_weights.csv", index=False)

    summary = {
        "rule_bilstm_binary": binary_metrics,
        "control_stratified_specificity": ctrl_spec,
        "ppv_npv_projections": ppv_npv,
        "stage_stratified_sensitivity": stage_metrics,
        "naive_rule_too_ablation": too_ablation,
        "mirbit_4_rule_baseline": mirbit_metrics,
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[Task 1 - 24-Rule BiLSTM] Acc: {binary_metrics['accuracy']*100:.2f}% | "
          f"Sens: {binary_metrics['sensitivity']*100:.2f}% | Spec: {binary_metrics['specificity']*100:.2f}% | "
          f"AUC: {binary_metrics['roc_auc']:.4f}")
    print(f"[Task 1 - Healthy vs Benign Spec] Healthy: {ctrl_spec['healthy']['specificity']*100:.2f}% | "
          f"Benign: {ctrl_spec['benign']['specificity']*100:.2f}%")
    print(f"[Naive Rule-Chaining TOO Ablation] Acc: {too_ablation['accuracy']*100:.2f}% | "
          f"Weighted F1: {too_ablation['weighted_f1']:.4f}")
    print(f"[4-Rule miRBiT Baseline] Acc: {mirbit_metrics['accuracy']*100:.2f}% | "
          f"Sens: {mirbit_metrics['sensitivity']*100:.2f}% | Spec: {mirbit_metrics['specificity']*100:.2f}%")
    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
