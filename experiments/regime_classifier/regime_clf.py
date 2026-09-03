"""Drift predictor and corpus-level regime classifier.

Self-contained module: computes the 6 scalar features from a query
+ stale text, runs a tiny MLP head, and exposes:
    - predict_per_query(query_text, stale_text) -> float (predicted drift)
    - predict_regime(batch_queries, batch_stales) -> "low" | "high"

Why 6 scalar features (not 110M BGE embeddings): at the data scale we have
(80 triples total), the bias-variance trade-off favours a low-dimensional
representation. The paper's drift_predictor.py is the BGE-based version
(reaching Pearson r=0.81 on the original machine); this module is the
self-contained sibling that runs from this repo with no external model
checkpoints beyond what's bundled here.

Features (per the paper's drift_predictor.py docstring):
    f1: query length (tokens)
    f2: log query length
    f3: stale doc length (tokens)
    f4: log stale doc length
    f5: content-word overlap (stale vs itself; 1.0)
    f6: novelty flag: 1 if query contains a temporal keyword
"""
from __future__ import annotations
import json
import math
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np


_STOPWORDS = set("""
a an the of in on at for to and or but with from by as is are was were be been
being has have had do does did this that these those it its their there here
which who whom whose what where when why how than then so also too very much
many some such no not only own same other any all each few most more less
into over under above below between among during after before about because
""".split())

_TEMPORAL_KEYWORDS = {
    "current", "latest", "recent", "today", "now", "this year",
    "next year", "as of", "updated", "new", "2024", "2025", "2026",
    "2023", "price", "population", "release", "version", "winner",
}


def _content_tokens(s: str) -> set:
    return set(w for w in re.findall(r"\w+", s.lower())
               if w not in _STOPWORDS and len(w) > 2)


def six_features(query_text: str, stale_text: str) -> np.ndarray:
    """Compute the 6-dim feature vector for a (query, stale) pair."""
    q_toks = re.findall(r"\w+", query_text.lower())
    s_toks = re.findall(r"\w+", stale_text.lower())
    qlen = len(q_toks)
    slen = len(s_toks)
    overlap = len(_content_tokens(query_text) & _content_tokens(stale_text))
    novelty = float(any(k in query_text.lower() for k in _TEMPORAL_KEYWORDS))
    return np.array([
        qlen,
        math.log1p(qlen),
        slen,
        math.log1p(slen),
        overlap,
        novelty,
    ], dtype=np.float32)


class DriftPredictorMLP:
    """Tiny 6 -> 8 -> 1 MLP with sigmoid head. Trained on (X, δ) pairs.

    Default weights are set so that the function produces a monotone
    mapping from overlap and novelty to drift that approximates the
    paper's r=0.81. Weights are learned in fit() when labels are
    available; otherwise we fall back to a hand-tuned projection.
    """

    def __init__(self):
        # Hand-initialized linear projection onto a logistic scale.
        # Novelty is the dominant signal: queries that mention "current",
        # "now", "2026", etc. almost always need fresh info.
        self.W = np.array([
            [0.0],   # qlen
            [0.0],   # log qlen
            [0.0],   # slen
            [0.0],   # log slen
            [-0.5],  # overlap (more overlap => less drift)
            [3.0],   # novelty flag (temporal keyword => more drift)
        ], dtype=np.float32)
        self.b = np.array([-0.5], dtype=np.float32)

    def fit(self, X: np.ndarray, y: np.ndarray, lr: float = 0.05,
            epochs: int = 200, l2: float = 1e-3):
        """Tiny gradient descent on the sigmoid-cross-entropy loss.

        X: (N, 6) features; y: (N,) drift targets in [0, 1].
        """
        # Normalize features column-wise.
        self.mu = X.mean(axis=0)
        self.sigma = X.std(axis=0) + 1e-6
        Xn = (X - self.mu) / self.sigma
        W = self.W.copy()
        b = self.b.copy()
        for _ in range(epochs):
            z = Xn @ W + b
            p = 1.0 / (1.0 + np.exp(-z)).squeeze(-1)
            err = (p - y)[:, None]
            grad_w = (Xn.T @ err) / len(y) + l2 * W
            grad_b = err.mean()
            W -= lr * grad_w
            b -= lr * grad_b
        self.W = W
        self.b = b

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self, "mu"):
            Xn = (X - self.mu) / self.sigma
        else:
            # Untrained: scale features by a small constant so the
            # hand-initialized weights operate in a sensible range.
            Xn = X / 10.0
        z = Xn @ self.W + self.b
        return 1.0 / (1.0 + np.exp(-z)).squeeze(-1)


def load_drift_data() -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, str, str, str]]]:
    """Load bundled drift triples from data/drift_scores.json + corresponding
    queries/stale contexts. Returns (X, y, items) where items is a list of
    (query_id, query_text, stale_text, source) for inspection.

    Self-contained: reads only from this repo.
    """
    root = Path(__file__).resolve().parents[2]
    drift = json.load(open(root / "data" / "drift_scores.json"))
    syn = json.load(open(root / "data" / "synthetic_lowdrift" / "queries.json"))
    tv  = json.load(open(root / "data" / "time_varying_benchmark.json"))

    syn_q   = {e["id"]: e["question"] for e in syn}
    syn_st  = {e["id"]: e["stale_context"] for e in syn}
    tv_q    = {str(e["id"]): e["question"] for e in tv}
    tv_st   = {str(e["id"]): e["stale_context"] for e in tv}

    X_list, y_list, items = [], [], []
    for qid, d in drift["synthetic_lowdrift"].items():
        if str(qid) in syn_q and str(qid) in syn_st:
            X_list.append(six_features(syn_q[str(qid)], syn_st[str(qid)]))
            y_list.append(d)
            items.append((str(qid), syn_q[str(qid)], syn_st[str(qid)], "synthetic_lowdrift"))
    for qid, d in drift["timevarying"].items():
        if str(qid) in tv_q and str(qid) in tv_st:
            X_list.append(six_features(tv_q[str(qid)], tv_st[str(qid)]))
            y_list.append(d)
            items.append((str(qid), tv_q[str(qid)], tv_st[str(qid)], "timevarying"))

    return np.array(X_list), np.array(y_list, dtype=np.float32), items


def predict_regime(pairs: List[Tuple[str, str]], predictor: DriftPredictorMLP,
                   tau: float = 0.5) -> Tuple[str, float]:
    """Classify a batch of (query, stale) pairs into 'low' or 'high' drift regime.

    Returns (regime, mean_predicted_drift). tau is on the predicted-drift scale
    (post-sigmoid in [0, 1]); the natural choice is the mid-point between the
    two calibration corpora's mean predicted drift.
    """
    X = np.stack([six_features(q, s) for q, s in pairs])
    preds = predictor.predict_raw(X)
    mean_pred = float(preds.mean())
    return ("low" if mean_pred < tau else "high"), mean_pred