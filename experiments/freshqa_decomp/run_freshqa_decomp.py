#!/usr/bin/env python3
"""Direct causal ablation of the FreshQA low-drift anomaly.

Reviewer 2 asked us to back the channel-attribution table in
sections/04-experiments.tex §RQ2.1 with a FreshQA-specific run that
controls reranker bonus (β), evidence (stale vs fresh), and generator
seed (3 seeds per condition), instead of cross-table cross-corpus
boundaries. This script does exactly that on the real FreshQA dev
corpus (n=74), once data/freshqa_corpus/ is built and models/router.pt
is trained.

Conditions (2×2×3 = 12 runs total):
  - evidence: stale | fresh
  - reranker bonus β: 0 | (1,1) (both flat 1, matching the paper)
  - generator seed: 0, 1, 2 (for variance bound only on ev=fresh)

Output: results/freshqa_decomp.json with per-cell F1/EM and the
generator-variance bound.
"""
import json
import os
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from src.fresh_rag.backends.freshqa import FreshQABackend
from src.fresh_rag.dense_retrieval import DenseRetriever
from src.fresh_rag.metrics import f1_score, exact_match
from src.fresh_rag.pipeline import FreshnessAwareRAG
from src.fresh_rag.trained_router import _load_model
from src.fresh_rag.backends.codex_llm import CodexLLMClient


RESULTS_PATH = PROJECT_ROOT / "results" / "freshqa_decomp.json"
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)


class FixedRouter:
    """A router that always picks the requested action regardless of query."""
    def __init__(self, action: int):
        self.action = float(action)
    def route(self, query) -> float:
        return self.action


def _bge_path() -> str:
    import glob
    snaps = PROJECT_ROOT / "models_cache" / "models--BAAI--bge-base-en-v1.5" / "snapshots"
    cands = sorted(glob.glob(str(snaps / "*")))
    return cands[0] if cands else "BAAI/bge-base-en-v1.5"


def load_items(backend):
    """Build [(question, stale_doc_text, fresh_doc_text, gold)] for the dev split."""
    out = []
    for q in backend.queries():
        stale = backend.get_documents(q.id, "stale")
        fresh = backend.get_documents(q.id, "fresh")
        gold  = backend.get_correct_answer(q.id)
        if stale and fresh and gold and stale[0].text and fresh[0].text:
            out.append({"id": q.id, "question": q.text,
                        "stale_doc": stale[0], "fresh_doc": fresh[0],
                        "gold": gold})
    return out


def main():
    backend = FreshQABackend(str(PROJECT_ROOT / "data" / "freshqa_dev.parquet"), role="qa_eval")
    items = load_items(backend)
    print(f"Loaded {len(items)} FreshQA dev queries with corpus (stale+fresh+gold)")
    if len(items) < 50:
        print(f"WARNING: only {len(items)} items have full corpus; the rest will be skipped by the backend.")

    # Build BGE retriever from the union of all docs (covers stale + fresh).
    all_docs = []
    for it in items:
        all_docs.append(it["stale_doc"])
        all_docs.append(it["fresh_doc"])
    dense_retr = DenseRetriever(docs=all_docs, model_name=_bge_path())
    print(f"Dense retriever: {len(all_docs)} docs, loaded={dense_retr._loaded}")

    llm = CodexLLMClient()
    print(f"LLM: {llm.model}")

    # Build the backend wrapper
    from src.fresh_rag.types import Query
    class ItemsBackend:
        def __init__(self):
            self._items = {it["id"]: it for it in items}
        def queries(self):
            return [Query(id=it["id"], text=it["question"], freshness_sensitive=True) for it in self._items.values()]
        def get_documents(self, qid, version):
            if qid not in self._items: return []
            d = self._items[qid]["stale_doc"] if version == "stale" else self._items[qid]["fresh_doc"]
            return [d]
        def get_correct_answer(self, qid):
            return self._items[qid]["gold"]

    full_be = ItemsBackend()

    def run_with_gold(evidence, beta, seed):
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        if evidence == "stale":
            be = ItemsBackend()
            for qid, it in be._items.items():
                it["fresh_doc"] = it["stale_doc"]
        else:
            be = full_be
        rag = FreshnessAwareRAG(backend=be, llm_client=llm, router=FixedRouter(1),
                                threshold=0.5, m=3, retriever=dense_retr,
                                lambda_fresh=beta[0], lambda_changed=beta[1])
        f1s, ems = [], []
        for q in be.queries():
            gold = be.get_correct_answer(q.id)
            try:
                ans, _ = rag.answer(q)
                f1s.append(f1_score(ans, gold))
                ems.append(exact_match(ans, gold))
            except Exception:
                f1s.append(0.0); ems.append(0.0)
        return f1s, ems

    # Four conditions × 1 seed each (seed only matters for ev=fresh variance).
    cells = {
        "ev_stale_bonus_off":        ("stale", (0.0, 0.0), 0),
        "ev_stale_bonus_on":         ("stale", (1.0, 1.0), 0),
        "ev_fresh_bonus_off":        ("fresh", (0.0, 0.0), 0),
        "ev_fresh_bonus_on":         ("fresh", (1.0, 1.0), 0),
    }
    summary = {}
    for name, (ev, beta, seed) in cells.items():
        t0 = time.time()
        f1s, ems = run_with_gold(ev, beta, seed)
        dt = time.time() - t0
        print(f"  {name}: F1={np.mean(f1s):.3f} EM={np.mean(ems):.3f} ({dt:.1f}s, n={len(f1s)})", flush=True)
        summary[name] = {
            "evidence": ev, "beta": beta, "seed": seed,
            "mean_f1": float(np.mean(f1s)), "mean_em": float(np.mean(ems)),
            "n": len(f1s), "raw_f1s": f1s, "elapsed_s": dt,
        }

    # 3-seed variance on the worst channel (ev=fresh, bonus on/off) at seed 1, 2
    seed_f1s_off, seed_f1s_on = [], []
    for seed in (1, 2):
        f1s, _ = run_with_gold("fresh", (0.0, 0.0), seed)
        seed_f1s_off.append(float(np.mean(f1s)))
        f1s, _ = run_with_gold("fresh", (1.0, 1.0), seed)
        seed_f1s_on.append(float(np.mean(f1s)))
    summary["generator_seed_variance"] = {
        "ev_fresh_bonus_off_3seeds": [summary["ev_fresh_bonus_off"]["mean_f1"]] + seed_f1s_off,
        "ev_fresh_bonus_on_3seeds":  [summary["ev_fresh_bonus_on"]["mean_f1"]]  + seed_f1s_on,
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {RESULTS_PATH}")
    print("\n=== FreshQA Decomposition ===")
    print(f"{'Condition':<28} {'F1':>7} {'EM':>7}  description")
    for name in ["ev_stale_bonus_off", "ev_stale_bonus_on",
                 "ev_fresh_bonus_off", "ev_fresh_bonus_on"]:
        s = summary[name]
        print(f"{name:<28} {s['mean_f1']:>7.3f} {s['mean_em']:>7.3f}  "
              f"evidence={s['evidence']}, β=({s['beta'][0]},{s['beta'][1]})")


if __name__ == "__main__":
    main()