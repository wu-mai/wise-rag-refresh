#!/usr/bin/env python3
"""Target-only from-scratch baseline for the Fact-Type Router adaptation story.

Goal: separate *transfer of policy structure* (proxy-trained init) from
*small-sample calibration to corpus-specific priors* by training an
identical architecture from the generic `microsoft/deberta-v3-small`
init on the same target queries, with the same hold-out and evaluation
protocol as `experiments/label_efficiency/run_label_efficiency.py`.

Protocol (identical to the adapted arm):
  - Hold-out test set: 11 entertainment queries (fixed for all k).
  - Training pool: 49 remaining queries.
  - For each k in {4, 8, 16, 32, 49}: sample k from training pool
    (sampling must reproduce the exact ids used by
    run_label_efficiency.py -- this script asserts equality).
  - For each seed in {42, 1, 0}: 5 epochs, AdamW, MSE on
    freshness_sensitive targets = 1.0; evaluate F1 + refresh rate
    on the hold-out via FreshnessAwareRAG with BGE retriever and
    CodexLLMClient (gpt-5.4).
  - Two training lrs are reported so the from-scratch arm gets a fair
    shot:
      * lr=1e-5 (the lr of the adapted arm — literal apples-to-apples)
      * lr=2e-5 (the lr of train_router.py — gives a stronger from-scratch
        update, in case 1e-5 is too gentle to escape the generic init)
  - One zero-shot row: un-trained generic init evaluated once.

Output:
  - results/target_only_baseline.json — schema mirrors
    results/label_efficiency.json plus a `variant` field.
  - models/target_only/ — per-(k, seed, variant) checkpoints.
"""
import json
import os
import sys
import time
import random
import argparse
from pathlib import Path

# Note: we deliberately do NOT force HF_HUB_OFFLINE here -- the script
# needs network access on first run to fetch BGE-base-en-v1.5 and the
# DeBERTa-v3-small weights (both are auto-downloaded by transformers).
# Subsequent runs will use the local HF cache at /root/gcy/hf_cache.
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)

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


LABEL_EFFICIENCY_PATH = PROJECT_ROOT / "results" / "label_efficiency.json"
RESULTS_PATH = PROJECT_ROOT / "results" / "target_only_baseline.json"
CKPT_DIR = PROJECT_ROOT / "models" / "target_only"
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# Experiment config — identical to run_label_efficiency.py
HOLDOUT_CATEGORY = "entertainment"
K_VALUES = [4, 8, 16, 32, 49]
SEEDS = [42, 1, 0]
N_EPOCHS = 5
BATCH_SIZE = 8
MAX_LEN = 64
THRESHOLD = 0.5
M = 3
LAMBDA_FRESH = 1.0
LAMBDA_CHANGED = 1.0
# Generic init -- the only thing we vary across runs is the seed.
MODEL_NAME = "microsoft/deberta-v3-small"
TARGET_PFRESH = 1.0  # all TimeVarying queries are freshness_sensitive
# Two lr variants -- both are within the same order of magnitude as
# the adapted arm (1e-5); 2e-5 is the lr train_router.py itself uses
# when training from the same generic init on FreshQA, so it gives the
# from-scratch arm a fair shot in case 1e-5 is too gentle to escape
# random init on a tiny target set.
LR_VARIANTS = {"scratch_lr1e-5": 1e-5, "scratch_lr2e-5": 2e-5}


def _local_bge_path() -> str:
    import glob
    snaps = PROJECT_ROOT / "models_cache" / "models--BAAI--bge-base-en-v1.5" / "snapshots"
    candidates = sorted(glob.glob(str(snaps / "*")))
    return candidates[0] if candidates else "BAAI/bge-base-en-v1.5"


def get_categories(backend):
    return {str(e["id"]): e.get("category", "unknown") for e in backend._entries}


def finetune_from_scratch(train_queries, target_pfresh, seed,
                          out_ckpt_path, tokenizer, model_name, lr):
    """Init DeBERTa-v3-small from generic weights, fine-tune N_EPOCHS, save ckpt."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = AutoModelForRegression(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.train()

    optim = torch.optim.AdamW(model.parameters(), lr=lr)

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


class FromScratchRouter:
    """Light wrapper exposing .route(query) -> float for a from-scratch ckpt.

    Bypasses the hard-coded `router.pt` path inside `_load_model` by
    loading the model directly with the local model_name and the
    sentencepiece workaround tokenizer.
    """
    def __init__(self, ckpt_path):
        # ckpt was saved with model_name = "microsoft/deberta-v3-small"
        from src.fresh_rag.spm_tokenizer import get_tokenizer
        self.tok = get_tokenizer()
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.model = AutoModelForRegression(ckpt["model_name"])
        self.model.load_state_dict(ckpt["state_dict"])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
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


def eval_router(router, holdout_queries, backend, dense_retr, llm) -> dict:
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


def assert_sampled_ids_match(rng_seed, k, sampled_ids, le_runs):
    """Sanity check: our sampling reproduces the label-efficiency script's ids."""
    expected = None
    for r in le_runs:
        if r["k"] == k and r["seed"] == rng_seed:
            expected = r["sampled_train_ids"]
            break
    if expected is None:
        return
    assert list(sampled_ids) == list(expected), (
        f"sampling mismatch for (k={k}, seed={rng_seed}): "
        f"expected {expected}, got {list(sampled_ids)}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=list(LR_VARIANTS.keys()),
                    choices=list(LR_VARIANTS.keys()),
                    help="which lr variants to run (default: both)")
    ap.add_argument("--skip-zero-shot", action="store_true",
                    help="skip the generic-init zero-shot baseline row")
    args = ap.parse_args()

    print("=" * 80)
    print("Target-only from-scratch baseline (DeBERTa-v3-small, generic init)")
    print("=" * 80)

    # Load prior label-efficiency runs so we can assert sampling matches.
    if LABEL_EFFICIENCY_PATH.exists():
        with open(LABEL_EFFICIENCY_PATH) as f:
            le = json.load(f)
        le_runs = le.get("runs", [])
        print(f"Loaded {len(le_runs)} runs from {LABEL_EFFICIENCY_PATH} for sampling check.")
    else:
        le_runs = []
        print(f"WARNING: {LABEL_EFFICIENCY_PATH} not found; sampling check disabled.")

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
    print(f"Training pool: {len(pool_queries)} queries")

    all_docs = [backend.get_documents(q.id, "stale")[0] for q in queries
                if backend.get_documents(q.id, "stale")[0].text]
    dense_retr = DenseRetriever(docs=all_docs, model_name=_local_bge_path())
    print(f"Dense retriever: {len(all_docs)} docs  loaded={dense_retr._loaded}")

    from src.fresh_rag.backends.codex_llm import CodexLLMClient
    llm = CodexLLMClient()
    print(f"LLM: {llm.model}")

    from src.fresh_rag.spm_tokenizer import get_tokenizer
    tok = get_tokenizer()

    results = {
        "holdout_category": HOLDOUT_CATEGORY,
        "holdout_size": len(holdout_queries),
        "pool_size": len(pool_queries),
        "k_values": K_VALUES,
        "seeds": SEEDS,
        "n_epochs": N_EPOCHS,
        "lr_variants": args.variants,
        "model_name": MODEL_NAME,
        "sampling_match_with_label_efficiency": True,
        "runs": [],
    }

    if not args.skip_zero_shot:
        # Generic-init zero-shot row: untrained model has random head.
        # Wrap the from-scratch router around a never-trained AutoModelForRegression.
        from src.fresh_rag.spm_tokenizer import get_tokenizer as _get_tok
        zs_tok = _get_tok()
        zs_model = AutoModelForRegression(MODEL_NAME)
        zs_device = "cuda" if torch.cuda.is_available() else "cpu"
        zs_model.to(zs_device); zs_model.eval()

        class _GenericZeroShotRouter:
            def __init__(self, tok, model):
                self.tok, self.model, self.device = tok, model, zs_device
                self._cache = {}
            def route(self, query):
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

        zs_router = _GenericZeroShotRouter(zs_tok, zs_model)
        zs_eval = eval_router(zs_router, holdout_queries, backend, dense_retr, llm)
        print(f"\n--- Zero-shot (generic DeBERTa, no training) ---")
        print(f"  F1 = {zs_eval['f1']:.3f}  refresh = {zs_eval['refresh_rate']:.1%}")
        results["zero_shot"] = zs_eval
        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2)

    for variant in args.variants:
        lr = LR_VARIANTS[variant]
        variant_runs = []
        for k in K_VALUES:
            for seed in SEEDS:
                tag = f"{variant}_k{k}_s{seed}"
                out_ckpt = CKPT_DIR / f"{tag}.pt"
                print(f"\n--- variant={variant} (lr={lr}) | k={k}, seed={seed} ---",
                      flush=True)

                rng = random.Random(seed)
                sampled = rng.sample(pool_queries, k)
                sampled_ids = [q.id for q in sampled]
                if le_runs:
                    assert_sampled_ids_match(seed, k, sampled_ids, le_runs)
                print(f"  Sampled training ids: {sampled_ids}")

                t0 = time.time()
                finetune_from_scratch(
                    train_queries=sampled,
                    target_pfresh=TARGET_PFRESH,
                    seed=seed,
                    out_ckpt_path=str(out_ckpt),
                    tokenizer=tok,
                    model_name=MODEL_NAME,
                    lr=lr,
                )
                ft_time = time.time() - t0
                print(f"  Fine-tune done in {ft_time:.1f}s")

                router = FromScratchRouter(str(out_ckpt))
                r = eval_router(router, holdout_queries, backend, dense_retr, llm)
                print(f"  Hold-out F1 = {r['f1']:.3f}  refresh = {r['refresh_rate']:.1%}")

                run_record = {
                    "variant": variant,
                    "lr": lr,
                    "k": k, "seed": seed,
                    "sampled_train_ids": sampled_ids,
                    "f1": r["f1"],
                    "refresh_rate": r["refresh_rate"],
                    "finetune_seconds": ft_time,
                }
                variant_runs.append(run_record)
                results["runs"].append(run_record)

                # Save incrementally so we don't lose progress.
                with open(RESULTS_PATH, "w") as f:
                    json.dump(results, f, indent=2)

        # Per-variant summary
        by_k = {k: [] for k in K_VALUES}
        for run in variant_runs:
            by_k[run["k"]].append(run["f1"])
        results.setdefault("summary_by_variant", {})[variant] = {
            str(k): {
                "mean_f1": float(np.mean(v)),
                "std_f1": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                "n_seeds": len(v),
                "raw_f1s": v,
            } for k, v in by_k.items()
        }
        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2)

    # Overall aggregate by (variant, k)
    overall = {}
    for variant in args.variants:
        for k in K_VALUES:
            key = f"{variant}_k{k}"
            overall[key] = results["summary_by_variant"][variant][str(k)]
    results["overall_summary"] = overall
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== Saved: {RESULTS_PATH} ===")
    print("\n--- Per-variant summary (mean +/- std over 3 seeds) ---")
    print(f"{'variant':<18} {'k':>4} {'mean F1':>9} {'std':>7}")
    for variant in args.variants:
        for k in K_VALUES:
            s = results["summary_by_variant"][variant][str(k)]
            print(f"{variant:<18} {k:>4} {s['mean_f1']:>9.3f} {s['std_f1']:>7.3f}")


if __name__ == "__main__":
    main()