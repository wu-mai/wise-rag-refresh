#!/usr/bin/env python3
"""No-reranker-bonus ablation: report the headline systems with lambda=0.

This script addresses CRITICAL issue 2 of Round 4 review: the main-text
results are confounded with the reranker is_live/is_changed bonuses, and the
appendix claim "qualitative ordering preserved" was qualitative only. Here
we re-run all 5 systems (Never, Always, Keyword, LLM, Fact-Type) on
TimeVarying with lambda_fresh=lambda_changed=0, and report F1 + refresh
rate. The with-bonus numbers (from blockC_tv_results.json) and the
without-bonus numbers go side-by-side in Table A.2 (and in §4 in the next
revision).
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fresh_rag.backends.timevarying import TimeVaryingBackend
from src.fresh_rag.dense_retrieval import DenseRetriever
from src.fresh_rag.keyword_router import KeywordRouter
from src.fresh_rag.metrics import f1_score
from src.fresh_rag.pipeline import FreshnessAwareRAG
from src.fresh_rag.router import Router
from src.fresh_rag.trained_router import TrainedRouter
from src.fresh_rag.types import ScoredDoc
from src.fresh_rag.backends.codex_llm import CodexLLMClient


SRC_CKPT = PROJECT_ROOT / "models" / "router.pt"
RESULTS_PATH = PROJECT_ROOT / "results" / "blockE_nobonus_results.json"
THRESHOLD = 0.5
M = 3


def _local_bge_path() -> str:
    import glob
    snaps = PROJECT_ROOT / "models_cache" / "models--BAAI--bge-base-en-v1.5" / "snapshots"
    candidates = sorted(glob.glob(str(snaps / "*")))
    return candidates[0] if candidates else "BAAI/bge-base-en-v1.5"


def static_rag(backend, version, llm):
    """Bypass the router/retriever and feed top-3 docs of a given version."""
    from src.fresh_rag.generator import Generator
    gen = Generator(llm_client=llm)
    def answer(q):
        docs = backend.get_documents(q.id, version)
        scored = [ScoredDoc(d, retrieval_score=1.0 - i * 0.1, final_score=0.0)
                  for i, d in enumerate(docs[:3])]
        return gen.generate(q, scored)
    return answer


def eval_static(name, answer_fn, queries, backend):
    rs = []
    for i, q in enumerate(queries):
        try:
            ans = answer_fn(q)
            rs.append({"id": q.id, "f1": f1_score(ans, backend.get_correct_answer(q.id))})
        except Exception:
            rs.append({"id": q.id, "f1": 0.0})
        if (i + 1) % 10 == 0:
            print(f"     [{name}] {i+1}/{len(queries)}", flush=True)
    f1 = sum(r["f1"] for r in rs) / max(1, len(rs))
    return {"f1": f1, "n": len(rs)}


def eval_rag(name, rag, queries, backend, k_used=3):
    rs = []
    refreshes = 0
    for i, q in enumerate(queries):
        try:
            ans, scored = rag.answer(q)
            refreshes += sum(1 for sd in scored if sd.doc.is_live)
            rs.append({"id": q.id, "f1": f1_score(ans, backend.get_correct_answer(q.id))})
        except Exception:
            rs.append({"id": q.id, "f1": 0.0})
        if (i + 1) % 10 == 0:
            print(f"     [{name}] {i+1}/{len(queries)}", flush=True)
    f1 = sum(r["f1"] for r in rs) / max(1, len(rs))
    return {"f1": f1, "refresh_rate": refreshes / max(1, len(rs) * k_used), "n": len(rs)}


def main():
    print("=" * 80)
    print("No-bonus reranker ablation on TimeVarying (lambda_fresh=lambda_changed=0)")
    print("=" * 80)

    backend = TimeVaryingBackend()
    queries = backend.queries()
    print(f"Loaded {len(queries)} queries")

    all_docs = [backend.get_documents(q.id, "stale")[0] for q in queries
                if backend.get_documents(q.id, "stale")[0].text]
    dense_retr = DenseRetriever(docs=all_docs, model_name=_local_bge_path())
    print(f"Dense retriever: {len(all_docs)} docs  _loaded={dense_retr._loaded}")

    llm = CodexLLMClient()
    print(f"LLM: {llm.model}")

    out = {"lambda_fresh": 0.0, "lambda_changed": 0.0, "results": {}}

    # Static baselines (Never / Always)
    for name, ver in [("Never Refresh", "stale"), ("Always Refresh", "fresh")]:
        print(f"\n-- {name} --", flush=True)
        r = eval_static(name, static_rag(backend, ver, llm), queries, backend)
        r["refresh_rate"] = 0.0 if name == "Never Refresh" else 1.0
        print(f"   F1={r['f1']:.3f}  refresh={r['refresh_rate']:.0%}", flush=True)
        out["results"][name] = r

    # Routed systems
    router_specs = [
        ("Keyword Router", KeywordRouter()),
        ("LLM Router", Router(llm_client=llm)),
        ("Fact-Type Router", TrainedRouter(str(SRC_CKPT))),
    ]
    for name, router in router_specs:
        print(f"\n-- {name} --", flush=True)
        rag = FreshnessAwareRAG(backend=backend, llm_client=llm, router=router,
                                threshold=THRESHOLD, m=M, retriever=dense_retr,
                                lambda_fresh=0.0, lambda_changed=0.0)
        r = eval_rag(name, rag, queries, backend)
        print(f"   F1={r['f1']:.3f}  refresh={r['refresh_rate']:.1%}", flush=True)
        out["results"][name] = r

    with open(RESULTS_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {RESULTS_PATH}")

    # Pretty comparison
    print("\n" + "=" * 80)
    print("TimeVarying: with-bonus (blockC_tv) vs without-bonus (this run)")
    print("=" * 80)
    with_bonus = {
        "Never Refresh": 0.191, "Always Refresh": 0.705,
        "Keyword Router": 0.612, "LLM Router": 0.614, "Fact-Type Router": 0.411,
    }
    print(f"{'System':<22} {'with-bonus F1':>14} {'no-bonus F1':>14} {'Δ':>8}")
    for sys_name in ["Never Refresh", "Always Refresh", "Keyword Router",
                     "LLM Router", "Fact-Type Router"]:
        wb = with_bonus[sys_name]
        nb = out["results"][sys_name]["f1"]
        print(f"{sys_name:<22} {wb:>14.3f} {nb:>14.3f} {nb-wb:>+8.3f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
