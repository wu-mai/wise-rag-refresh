#!/usr/bin/env python3
"""Evaluate the existing FreshQA-trained router on a stratified source-domain split.

The Round 4 review noted that the existing heldout split (rows 350-500) contains
zero fast-changing queries, so the router's source-domain performance is
under-validated. Here we evaluate the *existing* models/router.pt on three
splits of the same FreshQA test file and report the regression quality
(MSE, Spearman, AUROC for binary fresh/not-fresh classification):

  - train_rows (0-350): in-sample
  - heldout_rows (350-500): original (no fast-changing)
  - stratified_random (n=150, ~25% of file, balanced fact_type): re-split

This does NOT retrain the model. It only re-evaluates on a fairer split.
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, mean_squared_error

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fresh_rag.trained_router import TrainedRouter
from transformers import AutoTokenizer

CKPT = PROJECT_ROOT / "models" / "router.pt"
OUT_PATH = PROJECT_ROOT / "results" / "blockE_router_source_eval.json"


def main():
    df = pd.read_parquet(PROJECT_ROOT / "data" / "freshqa_test.parquet")
    # Three evaluation splits
    splits = {
        "train_rows_0_350": df.iloc[0:350],
        "heldout_rows_350_500": df.iloc[350:500],
        # Stratified: take 25% from each fact_type (3 categories)
        "stratified_25pct": (
            df.groupby("fact_type", group_keys=False)
            .apply(lambda g: g.sample(frac=0.25, random_state=42))
        ),
    }
    # Fact-type → p_fresh label mapping (mirror trained_router)
    label_map = {"never-changing": 0.0, "slow-changing": 0.5, "fast-changing": 1.0}

    router = TrainedRouter(str(CKPT), device="cuda" if torch.cuda.is_available() else "cpu")
    print(f"Router loaded: _loaded={router._loaded}, model_name={CKPT}", flush=True)

    results = {}
    for split_name, sub in splits.items():
        texts = sub["question"].tolist()
        labels = sub["fact_type"].map(label_map).values.astype(float)
        preds = np.array([router.route_str(t) if hasattr(router, "route_str") else
                           router.route(_StubQuery(t)) for t in texts])
        mse = mean_squared_error(labels, preds)
        sp, _ = spearmanr(labels, preds)
        # Binary "needs refresh" = fast-changing (1.0); threshold at 0.5
        binary_labels = (labels >= 0.5).astype(int)
        binary_preds = (preds >= 0.5).astype(int)
        try:
            auroc = roc_auc_score(binary_labels, preds)
        except Exception:
            auroc = float("nan")
        tp = int(((binary_labels == 1) & (binary_preds == 1)).sum())
        fp = int(((binary_labels == 0) & (binary_preds == 1)).sum())
        fn = int(((binary_labels == 1) & (binary_preds == 0)).sum())
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        results[split_name] = {
            "n": int(len(texts)),
            "mse": float(mse),
            "spearman": float(sp),
            "auroc_fast_vs_not": float(auroc) if not np.isnan(auroc) else None,
            "precision_at_0.5": float(prec),
            "recall_at_0.5": float(rec),
            "fact_type_counts": sub["fact_type"].value_counts().to_dict(),
        }
        print(f"\n[{split_name}] n={len(texts)}")
        print(f"  fact_type: {dict(sub['fact_type'].value_counts())}")
        print(f"  MSE={mse:.4f}  Spearman={sp:.3f}  AUROC={auroc:.3f}")
        print(f"  precision@0.5={prec:.3f}  recall@0.5={rec:.3f}")

    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {OUT_PATH}")


class _StubQuery:
    def __init__(self, t):
        self.text = t


if __name__ == "__main__":
    main()
