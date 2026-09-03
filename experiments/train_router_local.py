#!/usr/bin/env python3
"""Local-friendly training of the freshness router on FreshQA dev.

Mirrors experiments/train_router.py but:
  - Reads FreshQA dev parquet (100 rows; split into 70 train / 30 heldout)
  - Uses the SPM-tokenizer workaround (no AutoTokenizer tiktoken bug)
  - Forces fp32 (the trained_router.py fix from earlier in this session)

Saves models/router.pt in the same format as train_router.py so that
experiments/decomp_freshqa/run_decomp.py --mode freshqa and
experiments/label_efficiency/run_label_efficiency.py can both load it.
"""
import json
import os
import sys
from pathlib import Path

# Don't force HF offline — we need to load DeBERTa weights from local cache.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn

from src.fresh_rag.backends.freshqa import FreshQABackend
from src.fresh_rag.spm_tokenizer import get_tokenizer
from src.fresh_rag.trained_router import AutoModelForRegression


DEFAULT_MODEL = "microsoft/deberta-v3-small"
EPOCHS = 6
BATCH = 16
LR = 2e-5
MAX_LEN = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = PROJECT_ROOT / "models"
CKPT = MODEL_DIR / "router.pt"


def main():
    backend = FreshQABackend(str(PROJECT_ROOT / "data" / "freshqa_dev.parquet"), role="qa_eval")
    all_queries = backend.queries()
    print(f"Loaded {len(all_queries)} queries from FreshQA dev")

    # 70 / 30 split, deterministic
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(all_queries))
    n_train = int(0.7 * len(all_queries))
    train_qs = [all_queries[i] for i in idx[:n_train]]
    held_qs  = [all_queries[i] for i in idx[n_train:]]
    print(f"Split: {len(train_qs)} train / {len(held_qs)} heldout")

    tok = get_tokenizer()

    def encode(queries):
        texts = [q.text for q in queries]
        labels = [backend.get_pfresh_label(q.id) for q in queries]
        enc = tok(texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
        return {k: v.to(DEVICE) for k, v in enc.items()}, torch.tensor(labels, dtype=torch.float32, device=DEVICE)

    enc_tr, y_tr = encode(train_qs)
    enc_ho, y_ho = encode(held_qs)

    model = AutoModelForRegression(DEFAULT_MODEL).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    best_mse = float("inf")
    n = enc_tr["input_ids"].size(0)
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        ep_loss = 0.0; n_b = 0
        for i in range(0, n, BATCH):
            ix = perm[i:i+BATCH]
            pred = model(input_ids=enc_tr["input_ids"][ix], attention_mask=enc_tr["attention_mask"][ix])
            loss = nn.functional.mse_loss(pred, y_tr[ix])
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += float(loss.item()); n_b += 1
        print(f"epoch {epoch+1}/{EPOCHS}  train loss={ep_loss/max(1,n_b):.4f}", flush=True)

        # Eval
        model.eval()
        with torch.no_grad():
            pred_ho = model(input_ids=enc_ho["input_ids"], attention_mask=enc_ho["attention_mask"])
            mse = float(((pred_ho - y_ho) ** 2).mean())
        print(f"  held-out MSE = {mse:.4f}", flush=True)
        if mse < best_mse:
            best_mse = mse
            MODEL_DIR.mkdir(exist_ok=True, parents=True)
            torch.save({"model_name": DEFAULT_MODEL, "state_dict": model.state_dict()}, CKPT)
            print(f"  ✓ saved {CKPT}", flush=True)

    print(f"\nBest held-out MSE: {best_mse:.4f}")
    metrics_path = MODEL_DIR / "router_metrics.json"
    metrics_path.write_text(json.dumps({"best_heldout_mse": best_mse, "epochs": EPOCHS,
                                        "n_train": len(train_qs), "n_heldout": len(held_qs)}, indent=2))


if __name__ == "__main__":
    main()