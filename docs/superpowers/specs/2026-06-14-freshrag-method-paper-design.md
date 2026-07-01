# FreshRAG Method Paper — Implementation Design

**Date**: 2026-06-14
**Status**: Draft (pending user review)
**Parent**: `PAPER_PLAN.md` (upgraded from diagnostic → method paper)
**Venue**: WISE 2026
**Page budget**: 8 pages main body

## Goal

Upgrade the Freshness-Aware RAG system from an MVP with synthetic data and a
prompt-based router into a **method paper** with three genuine method
contributions, validated on a real temporal QA dataset (FreshQA) with real
content drift (Wayback Machine snapshots).

The central method claim to validate:

> A **query-conditioned refresh policy** driven by a **trained freshness
> router** achieves better cost-accuracy tradeoffs than never-refresh,
> always-refresh, and keyword-heuristic baselines on a real temporal QA
> benchmark.

## Why this upgrade (context)

The current MVP has two weaknesses that prevent it from being a credible
method paper:

1. **Synthetic staleness.** `NQBackend` fabricates stale/fresh docs by wrapping
   the *same answer* in different hedges ("may be outdated" vs "verified
   recently"). Any measured effect is a prompt-engineering artifact, not real
   content drift. Reviewers will reject this.

2. **Prompt-only router.** The LLM router is a fixed GPT-5.4 prompt. Its only
   method contribution is "we asked an LLM to rate freshness." This is too
   thin for a method paper.

This design fixes both: real content via Wayback, and a trained router with
FreshQA's `fact_type` labels as supervision.

## Out of Scope (deferred)

- End-to-end joint training of router + reranker (the "heavy" option we
  declined — kept as future work).
- Online/active refresh policies (future work).
- Multi-modal or non-English corpora.
- Re-running the old simulated/NQ synthetic experiments beyond what's already
  done (kept as appendix pilot only).

## Data

### Primary dataset: FreshQA

Source: `natyou/freshqa_10_06` on HuggingFace (Google FreshQA, October 2023
snapshot). Downloaded via parquet API (no auth needed).

- **dev**: 100 queries
- **test**: 500 queries
- **Total**: 600 queries, 99.7% with a Wikipedia `source` URL

Key fields used:

| Field | Use |
|-------|-----|
| `question` | Query text |
| `fact_type` ∈ {never, slow, fast-changing} | **Ground-truth freshness label** — router supervision + eval |
| `source` (URL) | Retrieval corpus source; Wayback target |
| `answer_0..9` | Gold answers (the dataset's `effective_year` answer) |
| `effective_year` | Which year the answer is valid for |
| `false_premise` | Tag; used for error analysis, not core training |

Label distribution (full 600): never-changing 225, slow-changing 222,
fast-changing 153. Slightly imbalanced but all three classes well-represented.

### Real content versions: Wayback Machine + LLM augmentation (hybrid)

For each FreshQA query's `source` URL, build two document versions:

1. **Fresh version**: fetch the *current* live page via the proxy (or fall
   back to the most recent Wayback snapshot if the live page is gone).
2. **Stale version**: fetch a Wayback snapshot dated to the query's
   `effective_year` (±6 months), so the stale doc genuinely reflects the
   era the gold answer was written for.

**Hybrid fallback**: if Wayback has no snapshot near the target year for a
given URL (~10-15% expected), generate a stale doc with GPT-5.4 by prompting
it to rewrite the current page as it would have appeared in `effective_year`,
conditioned on the gold answer. Tag these as `synthetic=true` so we can
report results on the real-only subset separately.

Wayback access verified working through the Clash proxy (tested
2026-06-14: `web.archive.org` returns 200; availability API returns a 2020
snapshot for the test URL with full HTML body).

### Freshness label mapping

For the trained router (regression target):

```
never-changing  →  p_fresh = 0.0
slow-changing   →  p_fresh = 0.5
fast-changing   →  p_fresh = 1.0
```

The router learns to predict this continuous score. The 0.5 for slow is a
modeling choice (documented); τ=0.5 in the pipeline then means "refresh only
fast-changing queries by default," which is interpretable.

## Architecture

```
[Query] → [Trained Freshness Router] → p_fresh ∈ [0,1]
                ↓
[Dense Retriever (BGE)] → top-k candidate docs (from stale corpus)
                ↓
[Selective Refresh]  if p_fresh > τ: swap top-m docs to fresh version
                ↓
[Learned Reranker]  s = α·s_retrieval + β·is_fresh + γ·is_changed
                ↓
[Generator (GPT-5.4)]  top-3 docs → answer
```

### Component responsibilities (one purpose each)

| Component | Input | Output | Decides |
|-----------|-------|--------|---------|
| **FreshnessRouter** | query text | `p_fresh ∈ [0,1]` | How much does this query need fresh info? |
| **DenseRetriever** | query, stale corpus, k | `list[Document]` | Which stale docs are relevant? |
| **Refresher** | docs, p_fresh, τ, m | `list[Document]` | Which docs to swap to live? |
| **Reranker** | scored docs, α/β/γ | sorted `list[ScoredDoc]` | Final ordering for generation |
| **Generator** | query, top-r docs | answer text | What to say given evidence |
| **Pipeline** | all of the above | (answer, trace) | Wire it together |

The router and retriever are **pluggable**: the pipeline accepts any object
implementing `.route(query) -> float` and `.retrieve(query, k) -> list[Doc]`.
This lets us ablate (keyword vs dense retriever; prompt vs trained router)
without touching the pipeline.

## Component Specifications

### 1. FreshQA Backend (`src/fresh_rag/backends/freshqa.py`) — NEW

**Purpose**: Load FreshQA, build per-query stale/fresh document pairs, expose
the same `Backend` interface as `SimulatedBackend`/`NQBackend`.

**Responsibilities**:
- Load `data/freshqa_dev.parquet` + `data/freshqa_test.parquet`.
- For each query: fetch (or load cached) fresh + stale document text.
- Cache fetched pages to `data/freshqa_corpus/<id>_{fresh,stale}.txt` to
  avoid re-fetching across experiment runs.
- Implement `queries()`, `get_documents(query_id, version)`,
  `get_correct_answer(query_id)`, `get_fact_type(query_id)`.

**Snapshot fetcher** (`src/fresh_rag/backends/wayback.py`) — NEW:
- `fetch_fresh(url)`: GET current page, strip HTML → text. On failure, use
  most recent Wayback snapshot.
- `fetch_stale(url, target_year)`: Wayback availability API → closest
  snapshot within ±6 months of Jan 1 `target_year` → fetch → strip HTML.
- `synthesize_stale(url, answer, year)`: LLM fallback (GPT-5.4) when no
  snapshot. Sets `synthetic=True` flag.

### 2. Trained Freshness Router (`src/fresh_rag/trained_router.py`) — NEW

**Purpose**: A fine-tuned DeBERTa-v3-small regressor predicting `p_fresh`.

**Training** (`experiments/train_router.py`):
- Model: `microsoft/deberta-v3-small` (140M params).
- Input: FreshQA `question` text.
- Target: continuous `p_fresh` (0.0/0.5/1.0 from `fact_type`).
- Loss: MSE.

**Data split** (FreshQA's own `split` field is dev=100 / test=500; we define
our own roles on top to avoid name collisions):

| Our role | FreshQA source | Size | Used for |
|----------|---------------|------|----------|
| Router train | FreshQA test, rows 0-350 | 350 | Train DeBERTa router |
| Router held-out | FreshQA test, rows 350-500 | 150 | Block B router eval (MSE/AUROC/macro-F1) |
| QA eval | FreshQA dev | 100 | Blocks A/C/D downstream answer-quality eval |

Router training never sees the QA eval set; router eval never sees training
data; QA eval uses a router checkpoint frozen on router-train. No leakage.

- ~5-10 epochs, batch 16, lr 2e-5, fp16, early-stop on router held-out MSE.
  ~30-60 min on one 4090.

**Inference** (`trained_router.route(query) -> float`):
- Tokenize, forward, take sigmoid of logit → `p_fresh`.
- Cached per query text.

The existing `KeywordRouter` and LLM prompt router remain as baselines.

### 3. Dense Retriever (`src/fresh_rag/dense_retrieval.py`) — NEW

**Purpose**: Replace Jaccard keyword matching with BGE semantic retrieval.

- Model: `BAAI/bge-base-en-v1.5` (110M params).
- Precompute embeddings for all stale corpus docs (one-time, cache to disk).
- At query time: embed query, cosine-sim against cached doc embeddings, top-k.
- Falls back to the existing keyword `Retriever` if BGE unavailable.

The keyword retriever remains as an ablation baseline.

### 4. Selective Refresh (`src/fresh_rag/refresh.py`) — EXISTING, minor change

No logic change. Already implements: if `p_fresh > τ`, swap top-m cached
docs for fresh versions. The FreshQA backend's `get_documents(version=...)`
supplies the real fresh/stale text.

### 5. Reranker (`src/fresh_rag/reranker.py`) — EXISTING, minor tune

Currently: `s_final = s_retrieval + λ₁·fresh + λ₂·changed` (fixed weights).

We do a small weight sweep `(α, β, γ)` on the router-train split, picking
the tuple that maximizes downstream F1 at τ=0.5. This is **weight tuning**,
explicitly *not* claimed as a learned/neural reranker — the paper describes
it as a fixed-formula reranker with tuned coefficients. The method's learned
component is the router, not the reranker.

### 6. Generator (`src/fresh_rag/generator.py`) — EXISTING, unchanged

GPT-5.4 via Codex API, same prompt as MVP.

### 7. Pipeline (`src/fresh_rag/pipeline.py`) — EXISTING, extend

Already accepts pluggable `router`. Extend to accept pluggable `retriever`
(keyword vs dense). Same `answer(query) -> (answer, trace)` interface.

## Experiment Design

Four blocks. Each block maps to a paper section / claim.

### Block A: Real staleness degradation (§4.1, C1)

**Claim**: On FreshQA with real Wayback content, stale retrieval degrades
answer quality, and degradation is concentrated in fast/slow-changing queries.

- Systems: Stale-RAG (always stale docs) vs Fresh-RAG (always fresh docs).
- Query groups: never / slow / fast-changing.
- Metrics: F1, EM.
- Output: 2×3 degradation table + per-type breakdown.

This replaces the synthetic NQ Block 1 as the paper's headline diagnostic.

### Block B: Router evaluation (§4.2, C2)

**Claim**: The trained freshness router predicts query freshness need better
than keyword or prompt baselines.

- Evaluate on held-out router test set.
- Metrics: MSE vs `fact_type`-mapped target; AUROC for binary
  "needs refresh" (fast vs never); macro-F1 on the 3 classes.
- Baselines: KeywordRouter (keyword hit), LLM prompt router (GPT-5.4).
- Output: router comparison table.

### Block C: Main method vs baselines (§4.3, C3 — the core)

**Claim**: Query-conditioned refresh with the trained router dominates
never/always-refresh and keyword baselines on the cost-accuracy frontier.

- Systems:
  1. Never Refresh (stale only, lower bound)
  2. Always Refresh (fresh only, upper bound, expensive)
  3. Keyword Router + selective refresh
  4. LLM Prompt Router + selective refresh
  5. **Trained Router (ours) + selective refresh**
- Dense retriever used for all systems (controlled).
- Metrics: F1, EM, refresh rate (cost), latency.
- Output: main results table + cost-accuracy frontier scatter.

### Block D: Threshold ablation (§4.4, C4)

**Claim**: τ and m let operators tune the cost-accuracy frontier; a single
default (τ=0.5) is near-optimal.

- Sweep τ ∈ {0.1, 0.3, 0.5, 0.7, 0.9}, m ∈ {1, 3, 5}.
- Fixed baselines: random-20%, top-20% refresh.
- Output: frontier curves.

### Compute & GPU plan

| Task | GPU | Est. time |
|------|-----|-----------|
| FreshQA + Wayback corpus build | CPU | 2-4 h (network-bound) |
| BGE embedding precompute | 1× 4090, <4GB | ~20 min |
| Router training (DeBERTa-small) | 1× 4090, <8GB | 30-60 min |
| Blocks A–D (inference + API) | 1× 4090 for BGE inference + API calls | ~3-4 h |

GPU is **shared with zgz** (both 4090s currently occupied). Our jobs are
small (<8GB, short). Strategy: run GPU jobs with `CUDA_VISIBLE_DEVICES` on
whichever card has headroom; the BGE/router jobs are short enough to fit in
gaps. No long-running training that would conflict.

## Testing Strategy

Unit tests (pytest) for new components, following the existing TDD pattern:

- `test_freshqa_backend.py`: loads parquet, returns queries/docs, label mapping.
- `test_wayback.py`: fetch + parse (mock the network), fallback to synthetic.
- `test_trained_router.py`: route() returns float in [0,1]; mock the model.
- `test_dense_retrieval.py`: returns top-k; falls back when BGE missing.

Integration: each experiment script (`run_blockA.py`, etc.) runs end-to-end
on a 20-query subset with a mock LLM before the full run.

## File Layout (additions)

```
src/fresh_rag/
├── backends/
│   ├── freshqa.py          # NEW — FreshQA + doc pair builder
│   └── wayback.py          # NEW — snapshot fetcher + LLM fallback
├── trained_router.py       # NEW — DeBERTa freshness regressor
├── dense_retrieval.py      # NEW — BGE retriever
└── (existing files extended)

experiments/
├── build_freshqa_corpus.py # NEW — fetch/cache all stale/fresh docs
├── train_router.py         # NEW — train DeBERTa router
├── blockA/run_blockA.py    # NEW — staleness degradation
├── blockB/run_blockB.py    # NEW — router evaluation
├── blockC/run_blockC.py    # NEW — main method vs baselines
└── blockD/run_blockD.py    # NEW — threshold ablation

tests/
├── test_freshqa_backend.py
├── test_wayback.py
├── test_trained_router.py
└── test_dense_retrieval.py

data/freshqa_corpus/        # cached stale/fresh doc text per query
models/                     # trained router checkpoint
```

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Wayback rate-limiting / blocks | Medium | High | Cache aggressively; add delays; LLM fallback for misses |
| FreshQA gold answers themselves stale | Medium | Medium | Use `effective_year`; report per-type; acknowledge in paper |
| GPU contention with zgz | Medium | Medium | Jobs are short (<8GB); run in gaps; precompute embeddings once |
| Router overfits 600 examples | Medium | Medium | Strong regularization; early stopping; report held-out metrics |
| Dense retriever doesn't improve over keyword | Low | Medium | Report both; the method claim holds on controlled comparison |
| BGE/DeBERTa download blocked | Low | High | HF verified reachable via proxy; pre-download checkpoints |

## Definition of Done

- [ ] FreshQA corpus built (real Wayback snapshots for ≥85% of queries)
- [ ] Trained router checkpoint saved, held-out MSE reported
- [ ] BGE embeddings cached, dense retriever integrated
- [ ] All 4 experiment blocks run on FreshQA, artifacts saved
- [ ] Block C shows trained router on/near the cost-accuracy frontier
- [ ] Unit tests pass for all new components
- [ ] Results reconciled into updated PAPER_PLAN.md (method paper framing)

## Open Questions

None blocking. The 0.5 mapping for slow-changing is a documented choice;
if Block B shows it hurts router calibration, we can revisit, but the
regression target is robust to small label shifts.
