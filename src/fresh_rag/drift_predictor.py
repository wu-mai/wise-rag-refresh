"""Drift Predictor: δ̂(query, stale_text, meta) → predicted drift.

GOAL: Predict the (retrospective) per-query drift score from stale-side signals
alone. If this works, the Drift-Aware Meta-Router can be deployed online
(because δ̂ doesn't need the fresh text — only the stale-side features).

Architecture:
    BGE-base encoder (frozen) → mean-pool over stale tokens → concat with
    [query_emb, query_len, log_query_len, stale_len, source_id] → MLP → scalar

The source_id is a learned embedding for {freshqa, timevarying, nq, ...} so the
model can learn corpus-level drift priors.

Trained with MSE on observed (query, stale_doc, δ) triples. Reported metrics:
    - Pearson r(δ, δ̂) per-fold
    - MAE(δ, δ̂)
    - 5-fold CV mean

NOTE on data scale: only ~136 triples. We use:
    - Frozen BGE (no fine-tuning) to avoid overfitting
    - Strong dropout (0.3)
    - Small MLP head (2 layers)
    - 5-fold CV for honest evaluation
"""
import os
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

PROJECT_ROOT = Path("/home/tylg-jsjxy-3/guochaoyi/服务器3/wise222/wise-rag-project")
SOURCE_NAMES = ["freshqa", "timevarying"]  # extendable
SOURCE_TO_IDX = {s: i for i, s in enumerate(SOURCE_NAMES)}


class DriftPredictor(nn.Module):
    def __init__(self, encoder_name: str, num_sources: int = 2, dropout: float = 0.3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        H = self.encoder.config.hidden_size
        self.source_emb = nn.Embedding(num_sources, 16)
        # Input: text_emb (H) + query_emb (H) + source_emb (16) + side features (4)
        side_dim = 4  # query_len, log_query_len, stale_len, log_stale_len
        self.head = nn.Sequential(
            nn.LayerNorm(2 * H + 16 + side_dim),
            nn.Linear(2 * H + 16 + side_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def encode(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).float()
        summed = (out.last_hidden_state * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return summed / denom  # mean pool (B, H)

    def forward(self, query_ids, query_mask, stale_ids, stale_mask, source_idx, query_len, stale_len):
        q_emb = self.encode(query_ids, query_mask)        # (B, H)
        s_emb = self.encode(stale_ids, stale_mask)        # (B, H)
        src = self.source_emb(source_idx)                # (B, 16)
        # side features
        ql = torch.stack([query_len.float(), torch.log1p(query_len.float()),
                         stale_len.float(), torch.log1p(stale_len.float())], dim=-1)
        joint = torch.cat([q_emb, s_emb, src, ql], dim=-1)
        return self.head(joint).squeeze(-1)


class DriftDataset(Dataset):
    """One item = (query_text, stale_text, source_name, true_drift)."""

    def __init__(self, items: List[Tuple[str, str, str, float]], tokenizer,
                 max_query_len: int = 32, max_stale_len: int = 128):
        self.items = items
        self.tok = tokenizer
        self.max_q = max_query_len
        self.max_s = max_stale_len

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        q, s, src, drift = self.items[i]
        q_enc = self.tok([q], padding="max_length", truncation=True,
                          max_length=self.max_q, return_tensors="pt")
        s_enc = self.tok([s], padding="max_length", truncation=True,
                          max_length=self.max_s, return_tensors="pt")
        return {
            "query_ids": q_enc["input_ids"][0],
            "query_mask": q_enc["attention_mask"][0],
            "stale_ids": s_enc["input_ids"][0],
            "stale_mask": s_enc["attention_mask"][0],
            "source_idx": torch.tensor(SOURCE_TO_IDX.get(src, 0), dtype=torch.long),
            "query_len": torch.tensor(len(q.split()), dtype=torch.long),
            "stale_len": torch.tensor(len(s.split()), dtype=torch.long),
            "drift": torch.tensor(float(drift), dtype=torch.float32),
        }


def load_drift_items():
    """Load (query, stale_text, source, drift) triples from existing results."""
    drift = json.load(open(PROJECT_ROOT / "results" / "drift_scores.json"))

    # FreshQA queries + their stale texts (from corpus)
    import pandas as pd
    fq_df = pd.read_parquet(PROJECT_ROOT / "data" / "freshqa_dev.parquet")
    fq_q = {str(row["id"]): row["question"] for _, row in fq_df.iterrows()}
    fq_stale = {}
    fq_dir = PROJECT_ROOT / "data" / "freshqa_corpus"
    for qid in fq_q:
        path = fq_dir / f"{qid}_stale.txt"
        if path.exists():
            fq_stale[qid] = path.read_text()[:2000]  # cap length

    # TimeVarying queries + stale texts (from benchmark)
    tv_data = json.load(open(PROJECT_ROOT / "data" / "time_varying_benchmark.json"))
    tv_q = {str(item["id"]): item["question"] for item in tv_data}
    tv_stale = {str(item["id"]): item.get("stale_context", "")[:2000] for item in tv_data}

    items = []
    for qid, drift_val in drift["freshqa"].items():
        if qid in fq_q and qid in fq_stale and fq_stale[qid].strip():
            items.append((fq_q[qid], fq_stale[qid], "freshqa", drift_val))
    for qid, drift_val in drift["timevarying"].items():
        if qid in tv_q and qid in tv_stale and tv_stale[qid].strip():
            items.append((tv_q[qid], tv_stale[qid], "timevarying", drift_val))

    return items


def train_predictor(items, tokenizer, encoder_name, n_folds=5, epochs=10,
                    batch_size=8, lr=2e-4, device="cpu"):
    """5-fold CV: train predictor, return per-fold Pearson r + MAE."""
    from sklearn.model_selection import KFold
    from scipy.stats import pearsonr

    items = np.array(items)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_metrics = []
    all_preds = []
    all_truths = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(items)):
        train_items = items[train_idx].tolist()
        test_items = items[test_idx].tolist()
        train_ds = DriftDataset(train_items, tokenizer)
        test_ds = DriftDataset(test_items, tokenizer)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        model = DriftPredictor(encoder_name, num_sources=len(SOURCE_NAMES)).to(device)
        opt = torch.optim.AdamW(model.head.parameters(), lr=lr, weight_decay=0.01)
        # Freeze encoder
        for p in model.encoder.parameters():
            p.requires_grad = False

        # Train
        for epoch in range(epochs):
            model.train()
            for batch in train_loader:
                opt.zero_grad()
                pred = model(
                    batch["query_ids"].to(device),
                    batch["query_mask"].to(device),
                    batch["stale_ids"].to(device),
                    batch["stale_mask"].to(device),
                    batch["source_idx"].to(device),
                    batch["query_len"].to(device),
                    batch["stale_len"].to(device),
                )
                loss = F.mse_loss(pred, batch["drift"].to(device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.head.parameters(), 1.0)
                opt.step()

        # Evaluate
        model.eval()
        preds, truths = [], []
        with torch.no_grad():
            for batch in test_loader:
                pred = model(
                    batch["query_ids"].to(device),
                    batch["query_mask"].to(device),
                    batch["stale_ids"].to(device),
                    batch["stale_mask"].to(device),
                    batch["source_idx"].to(device),
                    batch["query_len"].to(device),
                    batch["stale_len"].to(device),
                )
                preds.extend(pred.cpu().tolist())
                truths.extend(batch["drift"].tolist())
        r, _ = pearsonr(preds, truths) if len(set(truths)) > 1 else (0.0, 1.0)
        mae = float(np.mean(np.abs(np.array(preds) - np.array(truths))))
        fold_metrics.append({"fold": fold, "pearson_r": float(r), "mae": mae})
        print(f"  Fold {fold+1}: Pearson r={r:.3f}, MAE={mae:.3f}")
        all_preds.extend(preds)
        all_truths.extend(truths)

    overall_r, _ = pearsonr(all_preds, all_truths) if len(set(all_truths)) > 1 else (0.0, 1.0)
    overall_mae = float(np.mean(np.abs(np.array(all_preds) - np.array(all_truths))))
    print(f"\n  Overall: Pearson r={overall_r:.3f}, MAE={overall_mae:.3f}")
    return {
        "fold_metrics": fold_metrics,
        "overall_pearson_r": float(overall_r),
        "overall_mae": overall_mae,
        "n_train_total": len(items),
    }


def main():
    import glob
    encoder_name = sorted(glob.glob(str(PROJECT_ROOT / "models_cache" / "models--BAAI--bge-base-en-v1.5" / "snapshots" / "*")))[0]

    print("=" * 70)
    print("Drift Predictor: predict δ from (query, stale_text, source) alone")
    print("=" * 70)
    items = load_drift_items()
    print(f"Loaded {len(items)} (query, stale, source, drift) triples")
    print(f"  Drift range: {min(d[3] for d in items):.3f} - {max(d[3] for d in items):.3f}, "
          f"mean={np.mean([d[3] for d in items]):.3f}")

    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    results = train_predictor(items, tokenizer, encoder_name,
                              n_folds=5, epochs=10, batch_size=8, lr=2e-4)

    out_path = PROJECT_ROOT / "results" / "drift_predictor_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved: {out_path}")

    r = results["overall_pearson_r"]
    if r > 0.5:
        verdict = "STRONG POSITIVE — drift predictor has real signal; can replace retrospective δ"
    elif r > 0.3:
        verdict = "POSITIVE — moderate signal; useful as auxiliary feature"
    elif r > 0.1:
        verdict = "WEAK — small signal; predictor mostly noise"
    else:
        verdict = "NEGATIVE — no signal; cannot predict drift from stale-side alone"
    print(f"\n>>> VERDICT: {verdict}")


if __name__ == "__main__":
    main()