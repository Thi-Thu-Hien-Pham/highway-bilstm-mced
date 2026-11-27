"""Scale-invariant pairwise rule baselines (4-rule miRBiT and 24-rule ensemble)."""

from __future__ import annotations

from typing import Sequence
import numpy as np
import pandas as pd

from highway_bilstm.constants import (
    FIXED_EXTERNAL_RULE_THRESHOLD,
    MIRBIT_PAIRS,
    RULE_PAIRS,
)


class MiRBiTBaseline:
    """4-rule k-TSP majority voting baseline (Krishnamoorthy et al., 2026)."""

    def __init__(self, pairs: Sequence[tuple[str, str]] = MIRBIT_PAIRS) -> None:
        self.pairs = list(pairs)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        votes = np.zeros(len(df), dtype=np.int64)
        for g1, g2 in self.pairs:
            if g1 in df.columns and g2 in df.columns:
                votes += (df[g1].values > df[g2].values).astype(np.int64)
        return (votes > 2).astype(np.int64)


class RuleScoreClassifier:
    """Unweighted 24-rule satisfaction score evaluated at a fixed prespecified threshold."""

    def __init__(
        self,
        pairs: Sequence[tuple[str, str]] = RULE_PAIRS,
        threshold: float = FIXED_EXTERNAL_RULE_THRESHOLD,
    ) -> None:
        self.pairs = list(pairs)
        self.threshold = threshold

    def score(self, df: pd.DataFrame) -> np.ndarray:
        votes = np.zeros(len(df), dtype=np.float32)
        for g1, g2 in self.pairs:
            v1 = df[g1].values if g1 in df.columns else np.zeros(len(df), dtype=np.float32)
            v2 = df[g2].values if g2 in df.columns else np.zeros(len(df), dtype=np.float32)
            votes += (v1 > v2).astype(np.float32)
        return votes / float(len(self.pairs))

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return (self.score(df) >= self.threshold).astype(np.int64)
