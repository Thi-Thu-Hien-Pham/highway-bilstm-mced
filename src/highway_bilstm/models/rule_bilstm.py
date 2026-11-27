"""Rule-Chaining BiLSTM model operating on 24 pairwise microRNA diagnostic rules."""

from __future__ import annotations

import torch
import torch.nn as nn


class RuleChainingBiLSTM(nn.Module):
    """Two-layer BiLSTM with additive attention over 24 pairwise miRNA rule tokens."""

    def __init__(
        self,
        num_rules: int = 24,
        input_dim: int = 3,
        embed_dim: int = 32,
        hidden_dim: int = 64,
        num_cancer_types: int = 13,
    ) -> None:
        super().__init__()
        self.num_rules = num_rules
        self.rule_proj = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        self.rule_id_emb = nn.Parameter(torch.randn(1, num_rules, embed_dim) * 0.02)
        self.bilstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.15,
        )
        self.attn_proj = nn.Linear(hidden_dim * 2, 32)
        self.attn_v = nn.Linear(32, 1, bias=False)
        self.binary_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )
        self.too_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(64, num_cancer_types),
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        e = self.rule_proj(x) + self.rule_id_emb
        lstm_out, _ = self.bilstm(e)
        u = torch.tanh(self.attn_proj(lstm_out))
        attn_weights = torch.softmax(self.attn_v(u), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        binary_logits = self.binary_head(context).squeeze(-1)
        too_logits = self.too_head(context)
        return binary_logits, too_logits, attn_weights.squeeze(-1)
