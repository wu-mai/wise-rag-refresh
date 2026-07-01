#!/usr/bin/env python3
"""Train the freshness router on FreshQA.

Uses the 'router_train' split (350 rows from FreshQA test). Saves a
DeBERTa-v3-small + linear-head regression model to models/router.pt.
"""
import json
import os
import sys
from pathlib import Path

# Force offline mode if we have local model cache
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.fresh_rag.backends.freshqa import FreshQABackend
from src.fresh_rag.trained_router import AutoModelForRegression

# Local DeBERTa path; will be overridden by CLI arg
DEFAULT_MODEL = "microsoft/deberta-v3-small"
EPOCHS = 6
BATCH = 16
LR = 2e-5
MAX_LEN = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)
CKPT = MODEL_DIR / "router.pt"


class FreshQADataset(Dataset):
    def __init__(self, backend, tokenizer):
        self.queries = backend.queries()
        self.targets = [backend.get_pfresh_label(q.id) for q in self.queries]
        self.tok = tokenizer

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, i):
        q = self.queries[i]
        enc = self.tok(q.text, truncation=True, max_length=MAX_LEN, padding="max_length", return_tensors="pt")
        return {
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            "label": torch.tensor(self.targets[i], dtype=torch.float32),
        }


def main(model_name=None):
    if model_name is None:
        model_name = DEFAULT_MODEL
    backend = FreshQABackend(str(PROJECT_ROOT / "data" / "freshqa_test.parquet"), role="router_train")
    heldout = FreshQABackend(str(PROJECT_ROOT / "data" / "freshqa_test.parquet"), role="router_heldout")
    print(f"Train: {len(backend.queries())} | Held-out: {len(heldout.queries())}")

    tok = AutoTokenizer.from_pretrained(model_name)
    train_ds = FreshQADataset(backend, tok)
    heldout_ds = FreshQADataset(heldout, tok)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    heldout_loader = DataLoader(heldout_ds, batch_size=BATCH)

    model = AutoModelForRegression(model_name).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    sched = get_linear_schedule_with_warmup(opt, 0, EPOCHS * len(train_loader))

    best_mse = float("inf")
    for epoch in range(EPOCHS):
        model.train()
        for batch in train_loader:
            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            y = batch["label"].to(DEVICE)
            pred = model(ids, mask)
            loss = ((pred - y) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        # Eval
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for batch in heldout_loader:
                ids = batch["input_ids"].to(DEVICE)
                mask = batch["attention_mask"].to(DEVICE)
                y = batch["label"].to(DEVICE)
                p = model(ids, mask)
                preds.extend(p.cpu().tolist())
                labels.extend(y.cpu().tolist())
        mse = float(np.mean((np.array(preds) - np.array(labels)) ** 2))
        print(f"epoch {epoch+1}/{EPOCHS}  held-out MSE={mse:.4f}", flush=True)
        if mse < best_mse:
            best_mse = mse
            # save with weights_only=False compatible format
            torch.save({"model_name": model_name, "state_dict": model.state_dict()}, CKPT)
            print(f"  ✓ saved {CKPT} (best so far)", flush=True)

    metrics_path = MODEL_DIR / "router_metrics.json"
    metrics_path.write_text(json.dumps({"best_heldout_mse": best_mse, "epochs": EPOCHS}, indent=2))
    print(f"\n✓ Final held-out MSE: {best_mse:.4f}", flush=True)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default=None)
    args = p.parse_args()
    main(args.model_name)
