"""Unit tests for neural and rule-based models."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from highway_bilstm.constants import MIRBIT_PAIRS, RULE_PAIRS
from highway_bilstm.models.baselines import MiRBiTBaseline, RuleScoreClassifier
from highway_bilstm.models.highway_bilstm import AttentionModule, HighwayBiLSTM
from highway_bilstm.models.rule_bilstm import RuleChainingBiLSTM


@pytest.mark.parametrize("attn_type,num_heads", [("additive", 1), ("multihead", 8)])
@pytest.mark.parametrize("num_classes", [1, 13])
def test_highway_bilstm_forward_and_backward(attn_type: str, num_heads: int, num_classes: int) -> None:
    model = HighwayBiLSTM(
        step_dim=5,
        flat_dim=150,
        hidden_dim=64,
        num_layers=2,
        attn_type=attn_type,
        num_heads=num_heads,
        highway_dim=64,
        highway_layers=2,
        fused_dim=64,
        dropout=0.15,
        num_classes=num_classes,
    )
    x_seq = torch.randn(4, 30, 5)
    x_flat = torch.randn(4, 150)
    logits, attn_w = model(x_seq, x_flat)

    assert logits.shape == (4, num_classes)
    if attn_type == "multihead":
        assert attn_w.shape == (4, 30, 30)
    else:
        assert attn_w.shape == (4, 30)

    loss = logits.sum()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_highway_bilstm_config_factories() -> None:
    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "best_hyperparameters.json"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    opt_model = HighwayBiLSTM.from_config(cfg, num_classes=13)
    base_model = HighwayBiLSTM.from_baseline_config(num_classes=13)

    x_seq = torch.randn(2, 30, 5)
    x_flat = torch.randn(2, 150)
    for m in (opt_model, base_model):
        logits, _ = m(x_seq, x_flat)
        assert logits.shape == (2, 13)


def test_attention_module_invalid_type() -> None:
    with pytest.raises(ValueError, match="Unsupported attn_type"):
        AttentionModule(embed_dim=64, attn_type="invalid")


def test_rule_chaining_bilstm_forward_and_backward() -> None:
    model = RuleChainingBiLSTM(num_rules=24, input_dim=3, embed_dim=32, hidden_dim=64, num_cancer_types=13)
    x = torch.randn(6, 24, 3)
    bin_logits, too_logits, attn_w = model(x)

    assert bin_logits.shape == (6,)
    assert too_logits.shape == (6, 13)
    assert attn_w.shape == (6, 24)
    np.testing.assert_allclose(attn_w.sum(dim=1).detach().numpy(), np.ones(6), rtol=1e-5)

    (bin_logits.sum() + too_logits.sum()).backward()
    assert model.rule_id_emb.grad is not None


def test_rule_baselines_mirbit_and_score_classifier() -> None:
    cols = {m: np.zeros(2, dtype=np.float32) for pair in MIRBIT_PAIRS for m in pair}
    for g1, g2 in MIRBIT_PAIRS:
        cols[g1][0] = 5.0
        cols[g2][0] = 1.0
        cols[g1][1] = 1.0
        cols[g2][1] = 5.0

    df_mb = pd.DataFrame(cols)
    mb_preds = MiRBiTBaseline(MIRBIT_PAIRS).predict(df_mb)
    np.testing.assert_array_equal(mb_preds, [1, 0])

    # Disjoint synthetic 24-rule panel to test exact fraction scoring
    syn_pairs = [(f"a{i}", f"b{i}") for i in range(24)]
    syn_cols = {}
    for i, (g1, g2) in enumerate(syn_pairs):
        syn_cols[g1] = np.array([5.0 if i < 10 else 1.0, 1.0], dtype=np.float32)
        syn_cols[g2] = np.array([1.0 if i < 10 else 5.0, 5.0], dtype=np.float32)

    df_rc = pd.DataFrame(syn_cols)
    rc = RuleScoreClassifier(syn_pairs, threshold=9.0 / 24.0)
    scores = rc.score(df_rc)
    np.testing.assert_allclose(scores, [10.0 / 24.0, 0.0], rtol=1e-5)
    np.testing.assert_array_equal(rc.predict(df_rc), [1, 0])
