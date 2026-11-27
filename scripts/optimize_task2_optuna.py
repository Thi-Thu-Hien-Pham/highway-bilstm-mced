"""Optuna Bayesian hyperparameter optimization and final retraining for 13-class TOO Highway-BiLSTM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
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


def _make_loaders(X_flat: np.ndarray, y: np.ndarray, batch_size: int = 64, shuffle: bool = True) -> DataLoader:
    X_seq = torch.tensor(reshape_to_tokens(X_flat, 30, 5), dtype=torch.float32)
    X_f = torch.tensor(X_flat, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    return DataLoader(TensorDataset(X_seq, X_f, y_t), batch_size=batch_size, shuffle=shuffle)


def _inverse_class_weights(y: np.ndarray, num_classes: int, device: torch.device) -> torch.Tensor:
    _, counts = np.unique(y, return_counts=True)
    weights = len(y) / (float(num_classes) * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna hyperparameter search and retraining for Task 2 TOO Highway-BiLSTM.")
    parser.add_argument("--npz", default="data_cache/full_mirnome_matrix.npz", help="Path to full_mirnome_matrix.npz.")
    parser.add_argument("--config", default="configs/best_hyperparameters.json", help="Config JSON used when --skip-search is set.")
    parser.add_argument("--clinical-csv", default="data_cache/GSE211692_clinical_metadata.csv", help="Optional clinical metadata CSV/gz.")
    parser.add_argument("--output-dir", default="outputs/task2_optuna", help="Output directory.")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--tune-epochs", type=int, default=25)
    parser.add_argument("--retrain-epochs", type=int, default=35)
    parser.add_argument("--skip-search", action="store_true", help="Skip Optuna search and retrain directly from --config.")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_raw, sample_ids, organs, binary_classes, mirna_names = load_full_mirnome_npz(args.npz)
    norm_X = normalize_invariant_controls(X_raw, mirna_names)
    splits = split_task2_too(norm_X, organs, binary_classes, sample_ids=sample_ids, seed=args.seed)

    if args.skip_search:
        with open(args.config, "r", encoding="utf-8") as f:
            best_p = json.load(f)
    else:
        train_loader = _make_loaders(splits["X_tr_sub_s"], splits["y_train_sub"], batch_size=64, shuffle=True)
        val_loader = _make_loaders(splits["X_val_s"], splits["y_val"], batch_size=128, shuffle=False)
        loss_weights = _inverse_class_weights(splits["y_train_sub"], len(CANCER_CLASSES), device)

        def objective(trial: optuna.Trial) -> float:
            set_seed(args.seed + trial.number)
            hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
            num_layers = trial.suggest_int("num_layers", 1, 2)
            attn_type = trial.suggest_categorical("attn_type", ["additive", "multihead"])
            num_heads = trial.suggest_categorical("num_heads", [2, 4, 8]) if attn_type == "multihead" else 1
            highway_dim = trial.suggest_categorical("highway_dim", [64, 128, 256])
            highway_layers = trial.suggest_int("highway_layers", 1, 2)
            fused_dim = trial.suggest_categorical("fused_dim", [64, 128, 256])
            dropout = trial.suggest_float("dropout", 0.15, 0.40, step=0.05)
            lr = trial.suggest_categorical("lr", [5e-4, 1e-3, 2e-3])
            weight_decay = trial.suggest_categorical("weight_decay", [1e-5, 1e-4, 1e-3])

            model = HighwayBiLSTM(
                step_dim=5,
                flat_dim=150,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                attn_type=attn_type,
                num_heads=num_heads,
                highway_dim=highway_dim,
                highway_layers=highway_layers,
                fused_dim=fused_dim,
                dropout=dropout,
                num_classes=len(CANCER_CLASSES),
            ).to(device)
            criterion = nn.CrossEntropyLoss(weight=loss_weights)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, args.tune_epochs), eta_min=1e-5
            )

            best_val_f1 = 0.0
            for epoch in range(args.tune_epochs):
                model.train()
                for xs_b, xf_b, y_b in train_loader:
                    xs_b, xf_b, y_b = xs_b.to(device), xf_b.to(device), y_b.to(device)
                    optimizer.zero_grad()
                    logits, _ = model(xs_b, xf_b)
                    loss = criterion(logits, y_b)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.5)
                    optimizer.step()
                scheduler.step()

                model.eval()
                val_preds, val_targets = [], []
                with torch.no_grad():
                    for xs_b, xf_b, y_b in val_loader:
                        logits, _ = model(xs_b.to(device), xf_b.to(device))
                        val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                        val_targets.extend(y_b.numpy())

                val_f1 = f1_score(val_targets, val_preds, average="weighted")
                best_val_f1 = max(best_val_f1, val_f1)
                trial.report(val_f1, epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
            return best_val_f1

        sampler = optuna.samplers.TPESampler(seed=args.seed)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
        study.optimize(objective, n_trials=args.n_trials)
        best_p = study.best_params
        study.trials_dataframe().to_csv(out_dir / "optuna_trials_summary.csv", index=False)

    with open(out_dir / "best_hyperparameters.json", "w", encoding="utf-8") as f:
        json.dump(best_p, f, indent=4)

    # Retrain optimal configuration on full 80% training split (7,936 samples)
    set_seed(args.seed)
    best_model = HighwayBiLSTM.from_config(best_p, num_classes=len(CANCER_CLASSES)).to(device)
    full_train_loader = _make_loaders(splits["X_tr_full_opt_s"], splits["y_train_full"], batch_size=64, shuffle=True)
    full_test_loader = _make_loaders(splits["X_test_full_opt_s"], splits["y_test"], batch_size=128, shuffle=False)

    full_weights = _inverse_class_weights(splits["y_train_full"], len(CANCER_CLASSES), device)
    criterion = nn.CrossEntropyLoss(weight=full_weights)
    optimizer = torch.optim.AdamW(
        best_model.parameters(), lr=float(best_p["lr"]), weight_decay=float(best_p["weight_decay"])
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=float(best_p["lr"]) * 2.0,
        steps_per_epoch=len(full_train_loader),
        epochs=max(1, args.retrain_epochs),
    )

    for epoch in range(1, args.retrain_epochs + 1):
        best_model.train()
        for xs_b, xf_b, y_b in full_train_loader:
            xs_b, xf_b, y_b = xs_b.to(device), xf_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits, _ = best_model(xs_b, xf_b)
            loss = criterion(logits, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(best_model.parameters(), 1.5)
            optimizer.step()
            scheduler.step()

    best_model.eval()
    all_preds, all_probs = [], []
    with torch.no_grad():
        for xs_b, xf_b, _ in full_test_loader:
            logits, _ = best_model(xs_b.to(device), xf_b.to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(np.argmax(probs, axis=1))

    preds_arr = np.asarray(all_preds)
    probs_arr = np.asarray(all_probs)
    y_test = splits["y_test"]
    too_res = compute_too_metrics(y_test, preds_arr, probs_arr)

    stage_metrics = None
    if Path(args.clinical_csv).exists() and splits["test_sample_ids"] is not None:
        stage_map = parse_clinical_stage_map(args.clinical_csv)
        test_stages = [stage_map.get(str(sid), "Unstaged") for sid in splits["test_sample_ids"]]
        stage_metrics = compute_stage_stratified_too_metrics(y_test, preds_arr, test_stages)

    torch.save(best_model.state_dict(), out_dir / "best_optimized_model.pt")
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
            "ANOVA_F_Statistic": round(float(splits["f_vals_sub"][rank]), 2),
            "p_value": "< 1e-300" if splits["p_vals_sub"][rank] == 0.0 else f"{splits['p_vals_sub'][rank]:.2e}",
        }
        for rank, f_idx in enumerate(splits["top_idx_sub"])
    ]
    pd.DataFrame(token_rows).to_csv(out_dir / "anova_tokenization_mapping.csv", index=False)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"too_summary": too_res["summary"], "stage_stratified": stage_metrics}, f, indent=2)

    s = too_res["summary"]
    print(f"\n[Task 2 - Optuna Highway-BiLSTM] Acc: {s['accuracy']*100:.2f}% | "
          f"Weighted F1: {s['weighted_f1']:.4f} | Macro F1: {s['macro_f1']:.4f} | "
          f"Macro AUC: {s['macro_ovr_auc']:.4f}")
    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
