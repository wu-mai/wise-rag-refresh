#!/usr/bin/env python3
"""Label-efficiency experiment for the Fact-Type Router adaptation story.

Setup (fixed hold-out, varying training subset):
  - Hold-out test set: 11 entertainment queries (fixed for all k).
  - Training pool: 49 remaining queries (politics 11, sports 11, tech 10,
    demographics 11, other 6).
  - For each k in {4, 8, 16, 32, 49}: sample k from training pool,
    fine-tune router.pt for 5 epochs (lr=1e-5, AdamW, MSE on
    freshness_sensitive targets = 1.0), evaluate F1 on hold-out.
  - 3 seeds per k (42, 1, 0), report mean +/- std.

Output:
  - results/label_efficiency.json: raw per-(k, seed) F1 + refresh_rate
  - figures/fig_label_efficiency.pdf: line plot with error bars
"""
import json
import os
import sys
import time
import random
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fresh_rag.backends.timevarying import TimeVaryingBackend
from src.fresh_rag.dense_retrieval import DenseRetriever
from src.fresh_rag.metrics import f1_score
from src.fresh_rag.pipeline import FreshnessAwareRAG
from src.fresh_rag.trained_router import (
    AutoModelForRegression, _load_model,
)


SRC_CKPT = PROJECT_ROOT / "models" / "router.pt"
RESULTS_PATH = PROJECT_ROOT / "results" / "label_efficiency.json"
ADAPTED_DIR = PROJECT_ROOT / "models" / "label_efficiency"
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
ADAPTED_DIR.mkdir(parents=True, exist_ok=True)

# Experiment config (matches the round-table discussion with the user)
HOLDOUT_CATEGORY = "entertainment"  # fixed hold-out across all k
K_VALUES = [4, 8, 16, 32, 49]
SEEDS = [42, 1, 0]
N_EPOCHS = 5
LR = 1e-5
BATCH_SIZE = 8
MAX_LEN = 64
THRESHOLD = 0.5
M = 3
LAMBDA_FRESH = 1.0
LAMBDA_CHANGED = 1.0


def _local_bge_path() -> str:
    import glob
    snaps = PROJECT_ROOT / "models_cache" / "models--BAAI--bge-base-en-v1.5" / "snapshots"
    candidates = sorted(glob.glob(str(snaps / "*")))
    return candidates[0] if candidates else "BAAI/bge-base-en-v1.5"


def get_categories(backend):
    """Return {qid: category} dict from the backend's private entries."""
    return {str(e["id"]): e.get("category", "unknown") for e in backend._entries}


def finetune_router(src_ckpt_path, train_queries, target_pfresh, seed,
                    out_ckpt_path, tokenizer, model_name):
    """Load router.pt, fine-tune N_EPOCHS on train_queries, save adapted ckpt."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = AutoModelForRegression(model_name)
    ckpt = torch.load(src_ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.train()

    optim = torch.optim.AdamW(model.parameters(), lr=LR)

    texts = [q.text for q in train_queries]
    enc = tokenizer(texts, padding=True, truncation=True, max_length=MAX_LEN,
                    return_tensors="pt")
    if hasattr(enc, "to"):
        enc = enc.to(device)
    else:
        enc = {k: v.to(device) for k, v in enc.items()}
    target = torch.full((len(texts),), float(target_pfresh), dtype=torch.float32, device=device)

    n = len(texts)
    for epoch in range(N_EPOCHS):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            ids = enc["input_ids"][idx]
            mask = enc["attention_mask"][idx]
            y = target[idx]
            optim.zero_grad()
            pred = model(input_ids=ids, attention_mask=mask)
            loss = nn.functional.mse_loss(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        print(f"     epoch {epoch+1}/{N_EPOCHS}: avg loss = {epoch_loss/max(1,n_batches):.4f}",
              flush=True)

    model.eval()
    torch.save({"model_name": model_name, "state_dict": model.state_dict()},
               out_ckpt_path)


class AdaptedRouter:
    """Light wrapper exposing .route(query) -> float using an adapted ckpt."""
    def __init__(self, ckpt_path):
        self.tok, self.model = _load_model(ckpt_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self._cache = {}

    def route(self, query) -> float:
        if query.text in self._cache:
            return self._cache[query.text]
        enc = self.tok(query.text, return_tensors="pt", truncation=True,
                        max_length=MAX_LEN)
        if hasattr(enc, "to"):
            enc = enc.to(self.device)
        else:
            enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            logit = self.model(**enc).item()
        score = float(max(0.0, min(1.0, logit)))
        self._cache[query.text] = score
        return score


def eval_finetuned_router(adapted_ckpt_path, holdout_queries, backend,
                          dense_retr, llm) -> dict:
    """Wrap adapted router in TrainedRouter, run through RAG pipeline, eval F1."""
    router = AdaptedRouter(adapted_ckpt_path)
    rag = FreshnessAwareRAG(backend=backend, llm_client=llm, router=router,
                            threshold=THRESHOLD, m=M, retriever=dense_retr,
                            lambda_fresh=LAMBDA_FRESH, lambda_changed=LAMBDA_CHANGED)

    f1s = []
    refreshes = 0
    for i, q in enumerate(holdout_queries):
        try:
            ans, scored = rag.answer(q)
            f1s.append(f1_score(ans, backend.get_correct_answer(q.id)))
            refreshes += sum(1 for sd in scored if sd.doc.is_live)
        except Exception as e:
            print(f"     [holdout eval] err on {q.id}: {e}", flush=True)
            f1s.append(0.0)
        if (i + 1) % 5 == 0:
            print(f"     [holdout eval] {i+1}/{len(holdout_queries)}", flush=True)
    return {
        "f1": float(np.mean(f1s)),
        "refresh_rate": refreshes / max(1, len(holdout_queries) * M),
        "n": len(holdout_queries),
    }


def main():
    print("=" * 80)
    print("Label-efficiency experiment for Fact-Type Router adaptation")
    print("=" * 80)

    backend = TimeVaryingBackend()
    queries = backend.queries()
    print(f"Loaded {len(queries)} queries")

    qid_to_cat = get_categories(backend)
    q_by_id = {q.id: q for q in queries}

    holdout_ids = [qid for qid, c in qid_to_cat.items() if c == HOLDOUT_CATEGORY]
    pool_ids = [qid for qid, c in qid_to_cat.items() if c != HOLDOUT_CATEGORY]
    holdout_queries = [q_by_id[qid] for qid in holdout_ids]
    pool_queries = [q_by_id[qid] for qid in pool_ids]
    print(f"Hold-out ({HOLDOUT_CATEGORY}): {len(holdout_queries)} queries: {holdout_ids}")
    print(f"Training pool: {len(pool_queries)} queries "
          f"({sorted(set(qid_to_cat[qid] for qid in pool_ids))})")

    all_docs = [backend.get_documents(q.id, "stale")[0] for q in queries
                if backend.get_documents(q.id, "stale")[0].text]
    dense_retr = DenseRetriever(docs=all_docs, model_name=_local_bge_path())
    print(f"Dense retriever: {len(all_docs)} docs  loaded={dense_retr._loaded}")

    from src.fresh_rag.backends.codex_llm import CodexLLMClient
    llm = CodexLLMClient()
    print(f"LLM: {llm.model}")

    src_tok, _ = _load_model(str(SRC_CKPT))
    model_name = torch.load(str(SRC_CKPT), map_location="cpu", weights_only=False)["model_name"]
    print(f"Source model: {model_name}")

    target_pfresh = 1.0  # all TimeVarying queries are freshness_sensitive

    results = {
        "holdout_category": HOLDOUT_CATEGORY,
        "holdout_size": len(holdout_queries),
        "pool_size": len(pool_queries),
        "k_values": K_VALUES,
        "seeds": SEEDS,
        "n_epochs": N_EPOCHS,
        "lr": LR,
        "runs": [],
    }

    print("\n--- Zero-shot baseline (source router.pt as-is, no fine-tuning) ---")
    zs = eval_finetuned_router(str(SRC_CKPT), holdout_queries, backend,
                              dense_retr, llm)
    print(f"  Zero-shot hold-out F1 = {zs['f1']:.3f}  refresh = {zs['refresh_rate']:.1%}")
    results["zero_shot"] = zs

    for k in K_VALUES:
        for seed in SEEDS:
            tag = f"k{k}_s{seed}"
            out_ckpt = ADAPTED_DIR / f"adapted_{tag}.pt"
            print(f"\n--- k={k}, seed={seed} ---", flush=True)

            rng = random.Random(seed)
            sampled = rng.sample(pool_queries, k)
            sampled_ids = [q.id for q in sampled]
            print(f"  Sampled training ids: {sampled_ids}")

            t0 = time.time()
            finetune_router(
                src_ckpt_path=str(SRC_CKPT),
                train_queries=sampled,
                target_pfresh=target_pfresh,
                seed=seed,
                out_ckpt_path=str(out_ckpt),
                tokenizer=src_tok,
                model_name=model_name,
            )
            ft_time = time.time() - t0
            print(f"  Fine-tune done in {ft_time:.1f}s")

            r = eval_finetuned_router(str(out_ckpt), holdout_queries, backend,
                                      dense_retr, llm)
            print(f"  Hold-out F1 = {r['f1']:.3f}  refresh = {r['refresh_rate']:.1%}")
            results["runs"].append({
                "k": k, "seed": seed, "sampled_train_ids": sampled_ids,
                "f1": r["f1"], "refresh_rate": r["refresh_rate"],
                "finetune_seconds": ft_time,
            })

            # Save incrementally so we don't lose progress
            with open(RESULTS_PATH, "w") as f:
                json.dump(results, f, indent=2)

    # Aggregate
    by_k = {k: [] for k in K_VALUES}
    for run in results["runs"]:
        by_k[run["k"]].append(run["f1"])
    results["summary"] = {
        str(k): {
            "mean_f1": float(np.mean(v)),
            "std_f1": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
            "n_seeds": len(v),
            "raw_f1s": v,
        } for k, v in by_k.items()
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n=== Saved: {RESULTS_PATH} ===")

    print("\n--- Summary (mean ± std over 3 seeds) ---")
    print(f"{'k':>4} {'mean F1':>9} {'std':>7} {'raw F1s':>30}")
    for k in K_VALUES:
        s = results["summary"][str(k)]
        print(f"{k:>4} {s['mean_f1']:>9.3f} {s['std_f1']:>7.3f} "
              f"{[f'{x:.3f}' for x in s['raw_f1s']]}")


if __name__ == "__main__":
    main()