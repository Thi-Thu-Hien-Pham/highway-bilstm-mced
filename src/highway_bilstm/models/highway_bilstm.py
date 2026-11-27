"""Dual-stream Highway-BiLSTM architecture with Multi-Head or Additive Attention.

This module defines the ``HighwayBiLSTM`` model, which combines a sequential
BiLSTM+Attention stream with a tabular Highway MLP stream. The two streams
are fused before a classification head.

Classes:
    AttentionModule: Sequence attention pooling (multi-head or additive).
    HighwayBiLSTM: Dual-stream architecture with BiLSTM, attention,
        highway MLP, and fusion head.

Typical usage:
    model = HighwayBiLSTM.from_baseline_config(num_classes=13)
    logits, attn_weights = model(x_seq, x_flat)
"""

from __future__ import annotations

from typing import Any
import torch
import torch.nn as nn


class AttentionModule(nn.Module):
    """Sequence attention pooling supporting multi-head self-attention or additive attention.

    Two attention strategies are available:

    * ``"multihead"`` — applies ``nn.MultiheadAttention`` over the sequence
      and pools the output by taking the mean over time.  The returned
      attention weights are the raw head-level weights produced by the
      multi-head module.
    * ``"additive"`` — computes a scalar score per timestep with a small
      two-layer MLP, applies softmax over time, and returns a weighted sum
      of the sequence values.

    Args:
        embed_dim: Feature dimension of the input sequence.
        attn_type: Either ``"multihead"`` or ``"additive"``.
        num_heads: Number of attention heads (used only for ``"multihead"``).

    Raises:
        ValueError: If ``attn_type`` is not one of the supported strings.
    """

    def __init__(
        self,
        embed_dim: int,
        attn_type: str = "multihead",
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        self.attn_type = attn_type
        if attn_type == "multihead":
            self.mha = nn.MultiheadAttention(
                embed_dim=embed_dim, num_heads=num_heads, batch_first=True
            )
        elif attn_type == "additive":
            self.additive = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.Tanh(),
                nn.Linear(embed_dim // 2, 1),
            )
        else:
            raise ValueError(f"Unsupported attn_type '{attn_type}'. Expected 'multihead' or 'additive'.")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute a context vector and attention weights for the input sequence.

        Args:
            x: Input tensor of shape ``(batch, seq_len, embed_dim)``.

        Returns:
            A tuple of ``(context, weights)`` where:

            * ``context`` has shape ``(batch, embed_dim)`` — the pooled
              representation of the sequence.
            * ``weights`` has shape ``(batch, num_heads, seq_len, seq_len)``
              for multi-head attention, or ``(batch, seq_len)`` for additive
              attention.
        """
        if self.attn_type == "multihead":
            attn_out, weights = self.mha(x, x, x)
            context = torch.mean(attn_out, dim=1)
            return context, weights
        weights = torch.softmax(self.additive(x), dim=1)
        context = torch.sum(weights * x, dim=1)
        return context, weights.squeeze(-1)


class HighwayBiLSTM(nn.Module):
    """Dual-stream Highway-BiLSTM combining pseudo-sequential BiLSTM+Attention with tabular Highway MLP.

    The model has two parallel streams that are fused before the
    classification head:

    1. **Sequential stream** — input timesteps are projected, passed
       through a bidirectional LSTM, and pooled with ``AttentionModule``.
    2. **Tabular stream** — a flat feature vector is processed by a
       stacked Highway-style MLP (linear → batch-norm → GELU → dropout
       blocks).

    The two stream outputs are concatenated and fed through a final
    head that produces class logits.

    Args:
        step_dim: Number of features per timestep in the sequential input.
        flat_dim: Number of features in the tabular (flat) input.
        hidden_dim: Hidden size of the LSTM (per direction).
        num_layers: Number of stacked LSTM layers.
        attn_type: Attention type for the sequential stream.
        num_heads: Number of attention heads (used only for ``"multihead"``).
        highway_dim: Width of each Highway MLP block.
        highway_layers: Number of Highway MLP blocks.
        fused_dim: Width of the fusion head's intermediate layer.
        dropout: Dropout probability used throughout the model.
        num_classes: Number of output classes.
    """

    def __init__(
        self,
        step_dim: int = 5,
        flat_dim: int = 150,
        hidden_dim: int = 256,
        num_layers: int = 2,
        attn_type: str = "multihead",
        num_heads: int = 8,
        highway_dim: int = 256,
        highway_layers: int = 2,
        fused_dim: int = 256,
        dropout: float = 0.15,
        num_classes: int = 13,
    ) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(step_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout,
        )
        lstm_out_dim = hidden_dim * 2
        self.attn = AttentionModule(
            embed_dim=lstm_out_dim,
            attn_type=attn_type,
            num_heads=num_heads,
        )

        hw_modules: list[nn.Module] = []
        in_f = flat_dim
        for _ in range(highway_layers):
            hw_modules.extend(
                [
                    nn.Linear(in_f, highway_dim),
                    nn.BatchNorm1d(highway_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            in_f = highway_dim
        self.highway = nn.Sequential(*hw_modules)

        fused_in = lstm_out_dim + highway_dim
        self.head = nn.Sequential(
            nn.Linear(fused_in, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, num_classes),
        )

    @classmethod
    def from_config(cls, config: dict[str, Any], num_classes: int = 13) -> "HighwayBiLSTM":
        """Build a ``HighwayBiLSTM`` from a configuration dictionary.

        Reads hyper-parameters from *config* and returns a fully
        initialised model.  Missing keys fall back to sensible defaults.

        Args:
            config: Dictionary with at least the keys ``hidden_dim``,
                ``num_layers``, ``highway_dim``, ``highway_layers``,
                ``fused_dim``, and ``dropout``.  Optional keys
                ``attn_type`` and ``num_heads`` override the defaults.
            num_classes: Number of output classes.

        Returns:
            A ``HighwayBiLSTM`` instance configured from *config*.
        """
        attn_type = config.get("attn_type", "multihead")
        num_heads = config.get("num_heads", 1 if attn_type == "additive" else 8)
        return cls(
            step_dim=5,
            flat_dim=150,
            hidden_dim=int(config["hidden_dim"]),
            num_layers=int(config["num_layers"]),
            attn_type=attn_type,
            num_heads=int(num_heads),
            highway_dim=int(config["highway_dim"]),
            highway_layers=int(config["highway_layers"]),
            fused_dim=int(config["fused_dim"]),
            dropout=float(config["dropout"]),
            num_classes=num_classes,
        )

    @classmethod
    def from_baseline_config(cls, num_classes: int = 13) -> "HighwayBiLSTM":
        """Instantiate the baseline Highway-BiLSTM configuration (Revision 3 Section 2.5).

        Args:
            num_classes: Number of output classes.

        Returns:
            A ``HighwayBiLSTM`` with the baseline hyper-parameters.
        """
        return cls(
            step_dim=5,
            flat_dim=150,
            hidden_dim=128,
            num_layers=2,
            attn_type="additive",
            num_heads=1,
            highway_dim=128,
            highway_layers=2,
            fused_dim=128,
            dropout=0.25,
            num_classes=num_classes,
        )

    def forward(
        self, x_seq: torch.Tensor, x_flat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run forward pass through both streams and the fusion head.

        Args:
            x_seq: Sequential input of shape ``(batch, seq_len, step_dim)``.
            x_flat: Tabular (flat) input of shape ``(batch, flat_dim)``.

        Returns:
            A tuple of ``(logits, attn_weights)`` where:

            * ``logits`` has shape ``(batch, num_classes)``.
            * ``attn_weights`` is the attention weight tensor returned by
              the ``AttentionModule``.
        """
        h_seq = self.proj(x_seq)
        out, _ = self.lstm(h_seq)
        c_lstm, attn_w = self.attn(out)
        c_hw = self.highway(x_flat)
        fused = torch.cat([c_lstm, c_hw], dim=1)
        logits = self.head(fused)
        return logits, attn_w
