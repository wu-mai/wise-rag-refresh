"""Drift-Aware Meta-Router (B->A architecture).

A small classifier that takes a query + the per-query drift score
(1 - cos(BGE(stale), BGE(fresh))) and predicts which downstream
refresh policy will work best on this query:
  - 0: Never Refresh
  - 1: Always Refresh
  - 2: Keyword Router
  - 3: Trained Router (or LLM Router, for FreshQA)

Training: the per-query "best system" is computed offline (oracle
selection using each system's per-query F1 on a held-out set). At
inference, the classifier picks the system to use.

This is a B->A style architecture contribution: standard DeBERTa
text encoder (B) + drift-score side feature (A = the new context
that web-RAG refresh gives us). The novelty is the *input* and
*output structure*, not the encoder.
"""
import os
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


# 4-class output space
SYSTEM_NAMES = ["never", "always", "keyword", "trained"]
NUM_SYSTEMS = len(SYSTEM_NAMES)


class DriftAwareRouter(nn.Module):
    """DeBERTa(text) + concat[drift, query_len] -> 2-layer MLP -> 4-class softmax.

    Architecture:
        text_emb     = mean-pool(last_hidden_state, attention_mask)   # (B, H)
        feat_vec     = [drift, query_len_norm, log_query_len_norm]    # (B, 3)
        joint        = LayerNorm(text_emb || feat_vec)                # (B, H+3)
        hidden       = GELU(Linear(H+3, H))                          # (B, H)
        dropout
        logits       = Linear(H, NUM_SYSTEMS)                         # (B, 4)
    """

    def __init__(self, model_name: str, num_systems: int = NUM_SYSTEMS, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        H = self.encoder.config.hidden_size
        # Input feature dim: [drift, query_len_norm, log_query_len_norm] = 3
        self.feat_proj = nn.Sequential(
            nn.LayerNorm(H + 3),
            nn.Linear(H + 3, H),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(H, num_systems)
        self.num_systems = num_systems

    def forward(self, input_ids, attention_mask, drift, query_len_norm):
        """drift: (B,) raw drift score in [0, 1]; query_len_norm: (B, 2) = [len, log1p(len)]."""
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Mean pool over non-padding tokens
        mask = attention_mask.unsqueeze(-1).float()
        summed = (out.last_hidden_state * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        text_emb = summed / denom
        # Side features
        if drift.dim() == 1:
            drift = drift.unsqueeze(-1)
        if query_len_norm.dim() == 1:
            query_len_norm = query_len_norm.unsqueeze(-1)
        side = torch.cat([drift.float(), query_len_norm.float()], dim=-1)
        joint = torch.cat([text_emb, side], dim=-1)
        h = self.feat_proj(joint)
        return self.classifier(h)

    def predict(self, logits):
        return logits.argmax(dim=-1)


def featurize_query(question: str):
    """Returns (log_query_len_norm, query_len_norm) for the side features."""
    L = max(1, len(question.split()))
    return (
        np.log1p(L) / 5.0,   # rough normalization to ~[0, 1.5]
        min(L / 30.0, 1.0),  # raw length, clipped
    )


def build_oracle_labels(per_query_f1s: dict, drift_threshold: float = 0.05):
    """For each query, pick the best system given the drift bucket.

    per_query_f1s: {system_name: {qid: f1}}
    Returns: {qid: system_index}
    """
    qids = sorted(set.intersection(*[set(d.keys()) for d in per_query_f1s.values()]))
    labels = {}
    for qid in qids:
        scores = {sys: per_query_f1s[sys].get(qid, 0.0) for sys in per_query_f1s}
        best_sys = max(scores, key=lambda s: scores[s])
        labels[qid] = SYSTEM_NAMES.index(best_sys) if best_sys in SYSTEM_NAMES else 0
    return labels


def train_router(
    router: DriftAwareRouter,
    tokenizer,
    texts: List[str],
    drifts: List[float],
    labels: List[int],
    epochs: int = 5,
    batch_size: int = 8,
    lr: float = 2e-5,
    device: str = "cpu",
    verbose: bool = True,
) -> List[float]:
    """Train the router on (text, drift) -> system label data."""
    router.to(device)
    router.train()
    opt = torch.optim.AdamW(router.parameters(), lr=lr)
    losses = []
    n = len(texts)
    for epoch in range(epochs):
        # Shuffle
        idx = np.random.permutation(n)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, n, batch_size):
            batch_idx = idx[i : i + batch_size]
            batch_texts = [texts[j] for j in batch_idx]
            enc = tokenizer(batch_texts, padding=True, truncation=True,
                            max_length=64, return_tensors="pt").to(device)
            batch_drift = torch.tensor([drifts[j] for j in batch_idx], dtype=torch.float32).to(device)
            qln = torch.tensor([featurize_query(texts[j])[0] for j in batch_idx],
                               dtype=torch.float32).to(device)
            ql  = torch.tensor([featurize_query(texts[j])[1] for j in batch_idx],
                               dtype=torch.float32).to(device)
            qln_full = torch.stack([qln, ql], dim=-1)
            batch_labels = torch.tensor([labels[j] for j in batch_idx],
                                        dtype=torch.long).to(device)
            opt.zero_grad()
            logits = router(enc["input_ids"], enc["attention_mask"], batch_drift, qln_full)
            loss = F.cross_entropy(logits, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(router.parameters(), 1.0)
            opt.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        avg = epoch_loss / max(1, n_batches)
        losses.append(avg)
        if verbose:
            print(f"  epoch {epoch+1}/{epochs}: loss={avg:.4f}")
    router.eval()
    return losses


def predict_router(router, tokenizer, texts, drifts, device="cpu", batch_size=16):
    """Return predicted system index for each input."""
    router.to(device).eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            enc = tokenizer(batch_texts, padding=True, truncation=True,
                            max_length=64, return_tensors="pt").to(device)
            batch_drift = torch.tensor(drifts[i : i + batch_size], dtype=torch.float32).to(device)
            qln_full = torch.tensor(
                [featurize_query(t) for t in batch_texts], dtype=torch.float32
            ).to(device)
            logits = router(enc["input_ids"], enc["attention_mask"], batch_drift, qln_full)
            preds.extend(logits.argmax(dim=-1).cpu().tolist())
    return preds
