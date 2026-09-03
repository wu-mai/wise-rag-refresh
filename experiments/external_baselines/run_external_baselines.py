#!/usr/bin/env python3
"""External baseline comparison for the refresh-policy study.

Adds three published or canonical refresh policies that are not in
the five-policy table of §RQ2 (Never / Always / Keyword / LLM Router /
Fact-Type Router), and runs them on the same TimeVarying benchmark
with the same pipeline so the F1 numbers are directly comparable:

  - Random-p%:    refreshes the top-m documents with fixed probability p,
                  independently of the query. Anchors the cost-accuracy
                  frontier on both ends (p=0 -> never, p=1 -> always).
  - TTL:          refreshes every document that is older than T query
                  intervals (classical web-cache heuristic). Since our
                  cache is "logically one interval old" (stale version
                  cached at corpus build time, fresh version fetched on
                  refresh), TTL with T=1 degenerates to Always Refresh;
                  we evaluate T = ceil(60 / p) for p in {0.2, 0.4, 0.6}.
  - Self-RAG-style [RETRIEVE] router: a zero-shot LLM is asked to emit
                  [RETRIEVE] or [NO RETRIEVE] after the query; refresh
                  iff [RETRIEVE]. Adapted from Asai et al. (ICLR'24),
                  but with [RETRIEVE] mapped to refresh rather than
                  external retrieval (our cache has both stale and fresh
                  versions of the same document).

Outputs results/external_baselines.json with the same schema as
results/label_efficiency.json, plus a per-policy mean F1/EM/refresh.
"""
import json
import os
import re
import sys
import time
import random
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.fresh_rag.backends.timevarying import TimeVaryingBackend
from src.fresh_rag.dense_retrieval import DenseRetriever
from src.fresh_rag.metrics import f1_score, exact_match
from src.fresh_rag.pipeline import FreshnessAwareRAG
from src.fresh_rag.backends.codex_llm import CodexLLMClient
from src.fresh_rag.types import Query

RESULTS_PATH = PROJECT_ROOT / "results" / "external_baselines.json"
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

M = 3
THRESHOLD = 0.5
LAMBDA_FRESH = 1.0
LAMBDA_CHANGED = 1.0
N_QUERIES = 60


class FixedRouter:
    """Router that ignores the query and returns a fixed score."""
    def __init__(self, score: float):
        self.score = float(score)
    def route(self, query) -> float:
        return self.score


class RandomRouter:
    """Router that samples refresh probability from a Bernoulli(p) per query."""
    def __init__(self, p: float, seed: int = 0):
        self.p = float(p)
        self.rng = random.Random(seed)
    def route(self, query) -> float:
        return 1.0 if self.rng.random() < self.p else 0.0


class TTLRouter:
    """Refresh every document older than T queries since the last refresh.

    With a stateless cache (all docs are equally stale at corpus-build
    time), TTL with T=1 effectively refreshes every document on the
    next query after each refresh — which is Always Refresh. To make
    TTL a *meaningful* baseline on TimeVarying (n=60), we parameterize
    T as the integer interval at which the system refreshes its entire
    index: refresh on query index i iff i % T == 0. With T = ceil(60 / p)
    we get refresh-rate p.
    """
    def __init__(self, refresh_rate: float, n_total: int = N_QUERIES):
        self.p = float(refresh_rate)
        self.T = max(1, int(round(n_total / max(0.01, refresh_rate))))
        self.counter = -1
    def route(self, query) -> float:
        self.counter += 1
        return 1.0 if (self.counter % self.T == 0) else 0.0


class SelfRAGStyleRouter:
    """Zero-shot LLM router with [RETRIEVE] / [NO RETRIEVE] tokens.

    Adapted from Asai et al. 2024 (Self-RAG, ICLR'24): the router is an
    LLM that, given a query, decides whether the answer needs up-to-date
    external evidence. We map [RETRIEVE] -> refresh (re-fetch the live
    page from cache), [NO RETRIEVE] -> reuse.
    """
    PROMPT = (
        "You are deciding whether a question needs up-to-date information "
        "from the web to be answered correctly.\n"
        "Consider: are there time-sensitive keywords (current, latest, "
        "this year, now, etc.)? Does the answer depend on facts that "
        "change frequently (people in office, prices, sports scores, "
        "recent events)?\n\n"
        "Question: {q}\n\n"
        "Reply with exactly one token: either [RETRIEVE] or [NO RETRIEVE]. "
        "No explanation."
    )

    def __init__(self, llm):
        self.llm = llm
        self._cache = {}

    def route(self, query) -> float:
        if query.text in self._cache:
            return self._cache[query.text]
        try:
            resp = self.llm.chat(self.PROMPT.format(q=query.text))
        except Exception:
            self._cache[query.text] = 0.0
            return 0.0
        text = resp.strip().upper()
        score = 1.0 if "[RETRIEVE]" in text else 0.0
        self._cache[query.text] = score
        return score


def _bge_path() -> str:
    snaps = PROJECT_ROOT / "models_cache" / "models--BAAI--bge-base-en-v1.5" / "snapshots"
    cands = sorted(glob=str(snaps / "*")) if snaps.exists() else []
    return cands[0] if cands else "BAAI/bge-base-en-v1.5"


def _build_rag(backend, router, dense_retr, llm):
    return FreshnessAwareRAG(backend=backend, llm_client=llm, router=router,
                            threshold=THRESHOLD, m=M, retriever=dense_retr,
                            lambda_fresh=LAMBDA_FRESH, lambda_changed=LAMBDA_CHANGED)


def _eval(backend, dense_retr, llm, name, router_factory, queries, is_repeatable):
    """Run one baseline over queries. is_repeatable: True if router is deterministic."""
    f1s, ems, refreshes = [], [], 0
    rag = None
    for i, q in enumerate(queries):
        try:
            if rag is None:
                rag = _build_rag(backend, router_factory(i), dense_retr, llm)
            else:
                rag.router = router_factory(i)
            ans, scored = rag.answer(q)
            f1s.append(f1_score(ans, backend.get_correct_answer(q.id)))
            ems.append(exact_match(ans, backend.get_correct_answer(q.id)))
            refreshes += sum(1 for sd in scored if sd.doc.is_live)
        except Exception as e:
            print(f"     [{name}] err on {q.id}: {type(e).__name__}: {e}", flush=True)
            f1s.append(0.0); ems.append(0.0)
        if (i + 1) % 10 == 0:
            print(f"     [{name}] {i+1}/{len(queries)}", flush=True)
    return {
        "name": name,
        "mean_f1": float(np.mean(f1s)),
        "mean_em": float(np.mean(ems)),
        "refresh_rate": refreshes / max(1, len(f1s) * M),
        "n": len(f1s),
        "raw_f1s": f1s,
    }


def main():
    backend = TimeVaryingBackend()
    queries = backend.queries()
    print(f"Loaded {len(queries)} queries from TimeVarying")

    all_docs = [backend.get_documents(q.id, "stale")[0] for q in queries
                if backend.get_documents(q.id, "stale")[0].text]
    dense_retr = DenseRetriever(docs=all_docs, model_name=_bge_path())
    print(f"Dense retriever: {len(all_docs)} docs, loaded={dense_retr._loaded}")

    llm = CodexLLMClient()
    print(f"LLM: {llm.model}")

    # Seed: for random routers, use deterministic per-query seed = query.id integer
    def make_random(p, base_seed=0):
        # Single RNG shared across queries; each TR instance delegates to it.
        rng = random.Random(base_seed + int(p * 100))
        class RR:
            def route(self, q):
                return 1.0 if rng.random() < p else 0.0
        def factory(i):
            return RR()
        return factory

    def make_ttl(p):
        # "Every T-th query triggers a full index refresh" -- T chosen so the
        # refresh rate is ~p over n=60 queries. p=0.2 -> T=5, p=0.4 -> T=3
        # (ceil gives 2.5->3), p=0.6 -> T=2.
        T = max(1, int(round(1.0 / max(0.01, p))))
        counter = [0]
        class TR:
            def route(self, q):
                counter[0] += 1
                return 1.0 if (counter[0] % T == 0) else 0.0
        def factory(i):
            return TR()
        return factory

    def make_selfrag():
        # Self-RAG-style router is stateful (cached across queries), so the
        # factory returns the same router object every call.
        r = SelfRAGStyleRouter(llm)
        # Pre-warm by evaluating on the first 3 queries to catch LLM errors early.
        for q in queries[:3]:
            try:
                r.route(q)
            except Exception:
                pass
        def factory(i):
            return r
        return factory

    def make_fixed(score):
        def factory(i):
            return FixedRouter(score)
        return factory

    baselines = []
    # Random-p%
    for p in [0.20, 0.40, 0.60]:
        baselines.append((f"Random-{int(p*100)}%",
                          make_random(p), False))
    # TTL: refresh every T-th query; label by T so the semantics are clear
    for p in [0.20, 0.50]:
        T = max(1, int(round(1.0 / p)))
        baselines.append((f"TTL-T={T}",
                          make_ttl(p), False))
    # Self-RAG-style
    selfrag_factory = make_selfrag()
    baselines.append(("Self-RAG-style", selfrag_factory, False))
    # Anchors
    baselines.append(("Always (anchor)",  make_fixed(1.0), True))
    baselines.append(("Never (anchor)",    make_fixed(0.0), True))

    results = {"n_queries": len(queries), "n_epochs": 1,
               "model_name": "gpt-5.4", "runs": []}
    for name, factory, det in baselines:
        t0 = time.time()
        print(f"\n=== {name} ===", flush=True)
        r = _eval(backend, dense_retr, llm, name, factory, queries, det)
        dt = time.time() - t0
        r["elapsed_s"] = dt
        results["runs"].append(r)
        print(f"  {name}: F1={r['mean_f1']:.3f} EM={r['mean_em']:.3f} refresh={r['refresh_rate']:.1%} ({dt:.1f}s)")
        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2)

    # Summary table
    print("\n=== External baseline summary ===")
    print(f"{'Baseline':<22} {'F1':>7} {'EM':>7} {'Refresh':>8}")
    for r in results["runs"]:
        print(f"{r['name']:<22} {r['mean_f1']:>7.3f} {r['mean_em']:>7.3f} {r['refresh_rate']:>7.1%}")

    print(f"\nSaved: {RESULTS_PATH}")


if __name__ == "__main__":
    main()