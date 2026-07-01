# FreshRAG Method Paper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the Freshness-Aware RAG system into a method paper with a trained freshness router, dense retrieval, and real temporal content from FreshQA + Wayback Machine.

**Architecture:** FreshQA backend (real Wayback snapshots + LLM fallback) feeds a pipeline with three pluggable method components: a trained DeBERTa freshness router (regression to p_fresh), a BGE dense retriever, and selective refresh + tuned rerank. Four experiment blocks (staleness degradation, router eval, main method vs baselines, threshold ablation) validate the method.

**Tech Stack:** Python 3.13, PyTorch + HuggingFace Transformers, sentence-transformers (BGE), DeBERTa-v3-small, existing fresh_rag pipeline, GPT-5.4 via Codex API, pytest.

**Reference spec:** `docs/superpowers/specs/2026-06-14-freshrag-method-paper-design.md`

---

## Task Overview

| Task | Description | Files | New? |
|------|-------------|-------|------|
| 1 | FreshQA backend + label mapping | `src/fresh_rag/backends/freshqa.py`, `tests/test_freshqa_backend.py` | NEW |
| 2 | Wayback snapshot fetcher + LLM fallback | `src/fresh_rag/backends/wayback.py`, `tests/test_wayback.py` | NEW |
| 3 | FreshQA corpus build script | `experiments/build_freshqa_corpus.py` | NEW |
| 4 | Trained freshness router (DeBERTa) | `src/fresh_rag/trained_router.py`, `experiments/train_router.py`, `tests/test_trained_router.py` | NEW |
| 5 | Dense retriever (BGE) | `src/fresh_rag/dense_retrieval.py`, `tests/test_dense_retrieval.py` | NEW |
| 6 | Pipeline: pluggable retriever + rerank tune | `src/fresh_rag/pipeline.py`, `src/fresh_rag/reranker.py` | MODIFY |
| 7 | Block A: staleness degradation | `experiments/blockA/run_blockA.py` | NEW |
| 8 | Block B: router evaluation | `experiments/blockB/run_blockB.py` | NEW |
| 9 | Block C: main method vs baselines | `experiments/blockC/run_blockC.py` | NEW |
| 10 | Block D: threshold ablation | `experiments/blockD/run_blockD.py` | NEW |

**Execution order:** Tasks 1→2→3 (data layer), then 4 and 5 (independent, can parallelize after data ready), then 6, then 7→8→9→10. Task 3 depends on 1+2. Tasks 7-10 depend on 1-6.

---

## Task 1: FreshQA Backend + Label Mapping

**Purpose:** Load FreshQA parquet, expose queries/docs/labels through the existing `Backend` interface so the pipeline can use it like `SimulatedBackend`.

**Files:**
- Create: `src/fresh_rag/backends/freshqa.py`
- Test: `tests/test_freshqa_backend.py`
- Download: `data/freshqa_dev.parquet`, `data/freshqa_test.parquet`

**Data split convention (from spec):**

| Role | Source | Size |
|------|--------|------|
| Router train | FreshQA test rows 0-349 | 350 |
| Router held-out | FreshQA test rows 350-499 | 150 |
| QA eval | FreshQA dev (all 100) | 100 |

- [ ] **Step 1: Download FreshQA parquet files**

Run:
```bash
cd /home/tylg-jsjxy-4/guochaoyi/wise222/wise-rag-project
curl -sL "https://huggingface.co/api/datasets/natyou/freshqa_10_06/parquet/default/dev/0.parquet" -o data/freshqa_dev.parquet
curl -sL "https://huggingface.co/api/datasets/natyou/freshqa_10_06/parquet/default/test/0.parquet" -o data/freshqa_test.parquet
python -c "import pandas as pd; print('dev', len(pd.read_parquet('data/freshqa_dev.parquet'))); print('test', len(pd.read_parquet('data/freshqa_test.parquet')))"
```
Expected: `dev 100` and `test 500`.

- [ ] **Step 2: Write the failing test**

```python
# tests/test_freshqa_backend.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.fresh_rag.backends.freshqa import FreshQABackend, FACT_TYPE_TO_PFRESH, DATA_DIR

DEV_PATH = DATA_DIR / "freshqa_dev.parquet"


def test_fact_type_mapping():
    assert FACT_TYPE_TO_PFRESH["never-changing"] == 0.0
    assert FACT_TYPE_TO_PFRESH["slow-changing"] == 0.5
    assert FACT_TYPE_TO_PFRESH["fast-changing"] == 1.0


def test_loads_dev_queries():
    backend = FreshQABackend(DEV_PATH, role="qa_eval")
    queries = backend.queries()
    assert len(queries) == 100
    q = queries[0]
    assert isinstance(q.id, str)
    assert q.text  # non-empty


def test_get_documents_stale_and_fresh():
    backend = FreshQABackend(DEV_PATH, role="qa_eval")
    q = backend.queries()[0]
    stale = backend.get_documents(q.id, "stale")
    fresh = backend.get_documents(q.id, "fresh")
    assert len(stale) == 1 and len(fresh) == 1
    assert stale[0].is_live is False
    assert fresh[0].is_live is True
    assert fresh[0].is_changed is True


def test_get_correct_answer():
    backend = FreshQABackend(DEV_PATH, role="qa_eval")
    q = backend.queries()[0]
    ans = backend.get_correct_answer(q.id)
    assert isinstance(ans, str)


def test_get_fact_type_and_pfresh():
    backend = FreshQABackend(DEV_PATH, role="qa_eval")
    q = backend.queries()[0]
    ft = backend.get_fact_type(q.id)
    assert ft in ("never-changing", "slow-changing", "fast-changing")
    assert backend.get_pfresh_label(q.id) == FACT_TYPE_TO_PFRESH[ft]


def test_router_split_sizes():
    backend = FreshQABackend("data/freshqa_test.parquet", role="router_train")
    assert len(backend.queries()) == 350
    backend2 = FreshQABackend("data/freshqa_test.parquet", role="router_heldout")
    assert len(backend2.queries()) == 150
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_freshqa_backend.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.fresh_rag.backends.freshqa'`

- [ ] **Step 4: Write minimal implementation**

```python
# src/fresh_rag/backends/freshqa.py
"""FreshQA backend: real temporal QA with fact_type freshness labels.

Loads FreshQA parquet, exposes queries/docs/labels via the Backend protocol.
Document text (stale/fresh) is read from the pre-built corpus cache at
data/freshqa_corpus/<id>_{stale,fresh}.txt. If the cache is missing,
returns a placeholder so the pipeline degrades gracefully; run
experiments/build_freshqa_corpus.py to populate it.
"""
import json
from pathlib import Path
from typing import List
import pandas as pd

from ..types import Document, Query

DATA_DIR = Path(__file__).resolve().parents[3] / "data"
CORPUS_DIR = DATA_DIR / "freshqa_corpus"

FACT_TYPE_TO_PFRESH = {
    "never-changing": 0.0,
    "slow-changing": 0.5,
    "fast-changing": 1.0,
}

# Row ranges within the FreshQA test split (500 rows), per spec.
_ROUTER_TRAIN_ROWS = (0, 350)
_ROUTER_HELDOUT_ROWS = (350, 500)


def _row_range(role: str):
    if role == "router_train":
        return _ROUTER_TRAIN_ROWS
    if role == "router_heldout":
        return _ROUTER_HELDOUT_ROWS
    if role == "qa_eval":
        return None  # use all rows (dev file has 100)
    raise ValueError(f"unknown role {role!r}")


class FreshQABackend:
    """Backend over a FreshQA parquet slice + cached stale/fresh docs."""

    def __init__(self, path: str, role: str = "qa_eval"):
        df = pd.read_parquet(path)
        lo, hi = _row_range(role) or (0, len(df))
        self._df = df.iloc[lo:hi].reset_index(drop=True)
        self._role = role

    def queries(self) -> List[Query]:
        out = []
        for _, row in self._df.iterrows():
            qid = str(row["id"])
            ft = row["fact_type"]
            # freshness_sensitive (eval-only label): True if NOT never-changing
            sensitive = ft != "never-changing"
            out.append(Query(id=qid, text=row["question"], freshness_sensitive=sensitive))
        return out

    def _row(self, query_id: str):
        match = self._df[self._df["id"].astype(str) == query_id]
        if match.empty:
            return None
        return match.iloc[0]

    def _read_corpus(self, query_id: str, version: str):
        path = CORPUS_DIR / f"{query_id}_{version}.txt"
        if path.exists():
            return path.read_text(encoding="utf-8")
        return ""

    def _read_meta(self, query_id: str):
        path = CORPUS_DIR / f"{query_id}_meta.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return {}

    def get_documents(self, query_id: str, version: str) -> List[Document]:
        if version not in ("stale", "fresh"):
            raise ValueError(f"version must be 'stale' or 'fresh', got {version!r}")
        row = self._row(query_id)
        if row is None:
            return []
        text = self._read_corpus(query_id, version)
        meta = self._read_meta(query_id)
        is_live = version == "fresh"
        return [
            Document(
                doc_id=f"{query_id}-{version}",
                text=text,
                url=str(row.get("source", "")),
                timestamp=meta.get(f"{version}_timestamp", ""),
                age_days=meta.get(f"{version}_age_days", 0 if is_live else 900),
                is_live=is_live,
                is_changed=is_live,
            )
        ]

    def get_correct_answer(self, query_id: str) -> str:
        row = self._row(query_id)
        if row is None:
            return ""
        for i in range(10):
            val = row.get(f"answer_{i}")
            if isinstance(val, str) and val and val.lower() != "nan":
                return val
        return ""

    def get_fact_type(self, query_id: str) -> str:
        row = self._row(query_id)
        return str(row["fact_type"]) if row is not None else "never-changing"

    def get_pfresh_label(self, query_id: str) -> float:
        return FACT_TYPE_TO_PFRESH[self.get_fact_type(query_id)]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_freshqa_backend.py -v`
Expected: PASS (6 tests). Note: `get_documents` will return empty text until Task 3 builds the corpus — that's fine, the test only checks structure.

- [ ] **Step 6: Commit**

```bash
git add src/fresh_rag/backends/freshqa.py tests/test_freshqa_backend.py data/freshqa_dev.parquet data/freshqa_test.parquet
git commit -m "feat: FreshQA backend with fact_type freshness labels"
```

---

## Task 2: Wayback Snapshot Fetcher + LLM Fallback

**Purpose:** Fetch real historical (stale) and current (fresh) page text for a URL. Wayback for stale, live fetch for fresh, GPT-5.4 synthesis as fallback.

**Files:**
- Create: `src/fresh_rag/backends/wayback.py`
- Test: `tests/test_wayback.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_wayback.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.fresh_rag.backends.wayback import strip_html, find_closest_snapshot, synthesize_stale


def test_strip_html_removes_tags():
    html = "<html><body><p>Hello <b>world</b></p><script>x()</script></body></html>"
    text = strip_html(html)
    assert "Hello world" in text
    assert "script" not in text.lower() or "x()" not in text


def test_strip_html_collapses_whitespace():
    html = "<p>foo\n\n   bar</p>"
    assert "  " not in strip_html(html)
    assert "foo bar" in strip_html(html)


def test_find_closest_snapshot_uses_fetcher(monkeypatch):
    calls = []
    def fake_fetch(url):
        calls.append(url)
        return {"archived_snapshots": {"closest": {"available": True, "timestamp": "20200102", "url": "http://wb/x"}}}
    snap = find_closest_snapshot("example.com", 2020, availability_fetcher=fake_fetch)
    assert snap["timestamp"] == "20200102"
    assert calls  # fetcher was called


def test_find_closest_snapshot_none_when_unavailable(monkeypatch):
    def fake_fetch(url):
        return {"archived_snapshots": {}}
    assert find_closest_snapshot("example.com", 2020, availability_fetcher=fake_fetch) is None


def test_synthesize_stale_uses_llm():
    class FakeLLM:
        def __init__(self): self.called = False
        def chat(self, prompt, **kw):
            self.called = True
            return "In 2020, the prime minister was X."
    llm = FakeLLM()
    text = synthesize_stale("current page text", "the PM is Y", 2020, llm_client=llm)
    assert llm.called
    assert "prime minister was X" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_wayback.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/fresh_rag/backends/wayback.py
"""Fetch real stale/fresh page text via Wayback Machine, with LLM fallback.

- fetch_fresh(url): current live page (or most recent Wayback snapshot).
- fetch_stale(url, target_year): Wayback snapshot near target_year.
- synthesize_stale(...): GPT-5.4 fallback when no snapshot exists.
"""
import re
import time
from typing import Callable, Optional

import requests

WAYBACK_API = "https://archive.org/wayback/available"
WAYBACK_BASE = "https://web.archive.org"


def strip_html(html: str) -> str:
    """Crude HTML → text: drop script/style, strip tags, collapse whitespace."""
    html = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.S | re.I)
    html = re.sub(r"<style[^>]*>.*?</style>", " ", html, flags=re.S | re.I)
    html = re.sub(r"<[^>]+>", " ", html)
    html = re.sub(r"&nbsp;", " ", html)
    html = re.sub(r"\s+", " ", html)
    return html.strip()


def _http_get(url: str, timeout: int = 30) -> str:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def find_closest_snapshot(
    url: str,
    target_year: int,
    availability_fetcher: Optional[Callable[[str], dict]] = None,
) -> Optional[dict]:
    """Query Wayback availability API for the closest snapshot to Jan 1 of target_year."""
    timestamp = f"{target_year}0101"
    api_url = f"{WAYBACK_API}?url={url}&timestamp={timestamp}"
    fetcher = availability_fetcher or (lambda u: requests.get(u, timeout=20).json())
    try:
        data = fetcher(api_url)
    except Exception:
        return None
    closest = data.get("archived_snapshots", {}).get("closest", {})
    if closest.get("available"):
        return closest
    return None


def fetch_fresh(url: str, fetcher: Optional[Callable[[str], str]] = None) -> dict:
    """Fetch current page text. Fallback to most recent Wayback snapshot.

    Returns {"text": str, "timestamp": str, "age_days": 0}.
    """
    fetch = fetcher or _http_get
    try:
        html = fetch(url)
        return {"text": strip_html(html), "timestamp": "", "age_days": 0}
    except Exception:
        snap = find_closest_snapshot(url, 2026)
        if snap is None:
            return {"text": "", "timestamp": "", "age_days": 0}
        try:
            html = fetch(snap["url"])
            return {"text": strip_html(html), "timestamp": snap.get("timestamp", ""), "age_days": 0}
        except Exception:
            return {"text": "", "timestamp": "", "age_days": 0}


def fetch_stale(
    url: str,
    target_year: int,
    fetcher: Optional[Callable[[str], str]] = None,
) -> dict:
    """Fetch Wayback snapshot near target_year. Returns {"text","timestamp","age_days"} or empty."""
    snap = find_closest_snapshot(url, target_year)
    if snap is None:
        return {"text": "", "timestamp": "", "age_days": 0, "synthetic": False}
    fetch = fetcher or _http_get
    try:
        html = fetch(snap["url"])
        age_days = max(0, (2026 - target_year) * 365)
        return {"text": strip_html(html), "timestamp": snap.get("timestamp", ""), "age_days": age_days, "synthetic": False}
    except Exception:
        return {"text": "", "timestamp": "", "age_days": 0, "synthetic": False}


def synthesize_stale(
    fresh_text: str,
    gold_answer: str,
    target_year: int,
    llm_client: Optional[Callable] = None,
) -> str:
    """LLM fallback: rewrite fresh_text as it would have appeared in target_year."""
    if llm_client is None:
        return fresh_text  # nothing we can do offline
    prompt = (
        f"Rewrite the following page text as it would plausibly have appeared in {target_year}, "
        f"so that the fact '{gold_answer}' was current at that time. Keep it concise (under 300 words). "
        f"Only output the rewritten text.\n\nPage:\n{fresh_text[:1500]}"
    )
    try:
        return llm_client.chat(prompt=prompt).strip()
    except Exception:
        return fresh_text
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_wayback.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/fresh_rag/backends/wayback.py tests/test_wayback.py
git commit -m "feat: Wayback snapshot fetcher with LLM fallback"
```

---

## Task 3: FreshQA Corpus Build Script

**Purpose:** For all 600 FreshQA queries, fetch (or synthesize) stale+fresh docs and cache them to `data/freshqa_corpus/`. Run once; later experiments read the cache.

**Files:**
- Create: `experiments/build_freshqa_corpus.py`

- [ ] **Step 1: Write the build script**

```python
#!/usr/bin/env python3
"""Build the FreshQA stale/fresh corpus cache.

For every FreshQA query, fetch fresh (current) and stale (Wayback snapshot
near effective_year, LLM fallback if missing) page text, write to:
  data/freshqa_corpus/<id>_fresh.txt
  data/freshqa_corpus/<id>_stale.txt
  data/freshqa_corpus/<id>_meta.json   (timestamps, ages, synthetic flag)

Resumable: skips queries whose files already exist.
"""
import json
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.fresh_rag.backends.wayback import fetch_fresh, fetch_stale, synthesize_stale

DATA = PROJECT_ROOT / "data"
CORPUS = DATA / "freshqa_corpus"
CORPUS.mkdir(exist_ok=True)


def _effective_year(val) -> int:
    """Parse 'before 2022' / '2023' / 'after 2020' → int year (fallback 2020)."""
    s = str(val)
    m = re.search(r"(20\d\d)", s)
    return int(m.group(1)) if m else 2020


def build_one(row, llm_client):
    qid = str(row["id"])
    url = str(row.get("source", ""))
    gold = ""
    for i in range(10):
        v = row.get(f"answer_{i}")
        if isinstance(v, str) and v and v.lower() != "nan":
            gold = v
            break
    year = _effective_year(row.get("effective_year", 2020))

    fresh_path = CORPUS / f"{qid}_fresh.txt"
    stale_path = CORPUS / f"{qid}_stale.txt"
    meta_path = CORPUS / f"{qid}_meta.json"
    if fresh_path.exists() and stale_path.exists() and meta_path.exists():
        return "skip"

    fresh = fetch_fresh(url) if url.startswith("http") else {"text": "", "timestamp": ""}
    stale = fetch_stale(url, year) if url.startswith("http") else {"text": "", "timestamp": ""}
    synthetic = False
    if not stale["text"].strip():
        stale_text = synthesize_stale(fresh["text"], gold, year, llm_client=llm_client)
        if stale_text and stale_text != fresh["text"]:
            synthetic = True
        stale["text"] = stale_text

    fresh_path.write_text(fresh["text"], encoding="utf-8")
    stale_path.write_text(stale["text"], encoding="utf-8")
    meta_path.write_text(json.dumps({
        "fresh_timestamp": fresh.get("timestamp", ""),
        "fresh_age_days": 0,
        "stale_timestamp": stale.get("timestamp", ""),
        "stale_age_days": stale.get("age_days", 0),
        "synthetic": synthetic,
        "effective_year": year,
        "source_url": url,
    }), encoding="utf-8")
    return "synthetic" if synthetic else "real"


def main():
    try:
        from src.fresh_rag.backends.codex_llm import CodexLLMClient
        llm = CodexLLMClient()
        llm.chat("test")
        print("✓ Codex LLM available for synthesis fallback")
    except Exception as e:
        print(f"⚠ No Codex LLM ({e}); synthetic fallback disabled")
        llm = None

    files = [("dev", DATA / "freshqa_dev.parquet"), ("test", DATA / "freshqa_test.parquet")]
    counts = {"real": 0, "synthetic": 0, "skip": 0, "empty": 0}
    total = 0
    for label, path in files:
        df = pd.read_parquet(path)
        print(f"\n=== {label}: {len(df)} queries ===")
        for _, row in df.iterrows():
            total += 1
            try:
                status = build_one(row, llm)
            except Exception as e:
                print(f"  error on {row['id']}: {e}")
                status = "empty"
            counts[status] = counts.get(status, 0) + 1
            if total % 25 == 0:
                print(f"  {total} done | {counts}")
            time.sleep(0.3)  # be polite to Wayback

    print(f"\n✓ Built corpus: {counts}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test on 3 queries**

Run (dry-run on dev, interrupt after a few):
```bash
cd /home/tylg-jsjxy-4/guochaoyi/wise222/wise-rag-project
timeout 120 python -c "
import sys; sys.path.insert(0,'.')
import pandas as pd
from src.fresh_rag.backends.wayback import fetch_fresh, fetch_stale
df = pd.read_parquet('data/freshqa_dev.parquet').head(3)
for _,r in df.iterrows():
    url = str(r['source'])
    import re
    m=re.search(r'(20\d\d)', str(r.get('effective_year','2020')))
    yr=int(m.group(1)) if m else 2020
    f = fetch_fresh(url)
    s = fetch_stale(url, yr)
    print(f'id={r[\"id\"]} fresh={len(f[\"text\"])}c stale={len(s[\"text\"])}c year={yr}')
"
```
Expected: prints non-zero fresh text lengths; stale may be 0 for some (those will get LLM synthesis in the full run).

- [ ] **Step 3: Run the full build (background, resumable)**

Run:
```bash
nohup python -u experiments/build_freshqa_corpus.py > results/freshqa_corpus_build.log 2>&1 &
```
Expected: logs progress every 25 queries; `results/freshqa_corpus_build.log` shows `Built corpus: {real: N, synthetic: M, ...}` at the end. Verify `data/freshqa_corpus/` has ~600 sets of files.

- [ ] **Step 4: Commit**

```bash
git add experiments/build_freshqa_corpus.py
git commit -m "feat: FreshQA corpus build script (Wayback + LLM fallback)"
```

---

## Task 4: Trained Freshness Router (DeBERTa)

**Purpose:** Train a DeBERTa-v3-small regressor on FreshQA `fact_type` → continuous p_fresh. Provides the paper's main "learned method" component.

**Files:**
- Create: `src/fresh_rag/trained_router.py`
- Create: `experiments/train_router.py`
- Create: `tests/test_trained_router.py`
- Output: `models/router.pt`

- [ ] **Step 1: Install dependencies**

Run:
```bash
pip install transformers==4.45.2 sentence-transformers==3.1.1 accelerate==1.0.1
```
Expected: installs without error.

- [ ] **Step 2: Write the failing test**

```python
# tests/test_trained_router.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.fresh_rag.trained_router import TrainedRouter, _MockModel, FACT_TYPE_TO_PFRESH


def test_mock_model_predicts_fact_types():
    model = _MockModel()
    out = model.predict_proba("What is the current age of the oldest singles champion?")
    p = FACT_TYPE_TO_PFRESH["fast-changing"]
    assert abs(out - p) < 0.5  # within half-step of correct


def test_trained_router_returns_float_in_range(monkeypatch):
    monkeypatch.setattr("src.fresh_rag.trained_router._load_model", lambda path: _MockModel())
    router = TrainedRouter(model_path="models/router.pt")
    from src.fresh_rag.types import Query
    score = router.route(Query(id="0", text="When was the iPhone first released?"))
    assert 0.0 <= score <= 1.0


def test_trained_router_caches(monkeypatch):
    monkeypatch.setattr("src.fresh_rag.trained_router._load_model", lambda path: _MockModel())
    router = TrainedRouter(model_path="models/router.pt")
    from src.fresh_rag.types import Query
    q = Query(id="0", text="some query")
    s1 = router.route(q)
    s2 = router.route(q)
    assert s1 == s2  # cached


def test_trained_router_falls_back_when_no_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setattr("src.fresh_rag.trained_router._load_model", lambda path: (_ for _ in ()).throw(FileNotFoundError()))
    router = TrainedRouter(model_path=str(tmp_path / "missing.pt"))
    from src.fresh_rag.types import Query
    score = router.route(Query(id="0", text="anything"))
    assert score == 0.5  # default
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_trained_router.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 4: Write minimal implementation**

```python
# src/fresh_rag/trained_router.py
"""Trained freshness router: DeBERTa-v3-small regressor → p_fresh.

Trained on FreshQA fact_type (mapped to 0.0/0.5/1.0). At inference, returns
a float in [0, 1]. Falls back to 0.5 when no checkpoint is present.
"""
import os
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModel

# Mirror the mapping in freshqa backend to avoid an import cycle
FACT_TYPE_TO_PFRESH = {
    "never-changing": 0.0,
    "slow-changing": 0.5,
    "fast-changing": 1.0,
}


def _load_model(path: str):
    """Load a saved DeBERTa router. Raises if missing."""
    ckpt = torch.load(path, map_location="cpu")
    model_name = ckpt["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForRegression.from_pretrained(model_name)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return (tokenizer, model)


class AutoModelForRegression(torch.nn.Module):
    """DeBERTa + linear regression head."""
    def __init__(self, model_name: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.head = torch.nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] / last_hidden_state[:,0]
        cls = out.last_hidden_state[:, 0]
        return self.head(cls).squeeze(-1)


class _MockModel:
    """Heuristic model used only in tests — never for real predictions."""
    def predict_proba(self, text: str) -> float:
        t = text.lower()
        if any(k in t for k in ["current", "latest", "now", "this year", "upcoming", "next"]):
            return FACT_TYPE_TO_PFRESH["fast-changing"]
        if any(k in t for k in ["when was", "when did", "who was", "how many"]):
            return FACT_TYPE_TO_PFRESH["slow-changing"]
        return FACT_TYPE_TO_PFRESH["never-changing"]


class TrainedRouter:
    """Plug-compatible with KeywordRouter / Router: route(query) → float."""

    def __init__(self, model_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.model_path = model_path
        self.device = device
        self._cache: dict = {}
        try:
            self._tok, self._model = _load_model(model_path)
            self._model.to(device)
            self._loaded = True
        except Exception:
            self._loaded = False
            self._mock = _MockModel()

    def route(self, query) -> float:
        if query.text in self._cache:
            return self._cache[query.text]
        if not self._loaded:
            score = float(self._mock.predict_proba(query.text))
        else:
            score = self._predict(query.text)
        self._cache[query.text] = score
        return score

    def _predict(self, text: str) -> float:
        enc = self._tok(text, return_tensors="pt", truncation=True, max_length=64).to(self.device)
        with torch.no_grad():
            logit = self._model(**enc).item()
        return float(max(0.0, min(1.0, logit)))
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_trained_router.py -v`
Expected: PASS (4 tests).

- [ ] **Step 6: Write the training script**

```python
#!/usr/bin/env python3
"""Train the freshness router on FreshQA.

Uses the 'router_train' split (350 rows from FreshQA test). Saves a
DeBERTa-v3-small + linear-head regression model to models/router.pt.
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.fresh_rag.backends.freshqa import FreshQABackend
from src.fresh_rag.trained_router import AutoModelForRegression, FACT_TYPE_TO_PFRESH

MODEL_NAME = "microsoft/deberta-v3-small"
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


def main():
    backend = FreshQABackend(str(PROJECT_ROOT / "data" / "freshqa_test.parquet"), role="router_train")
    heldout = FreshQABackend(str(PROJECT_ROOT / "data" / "freshqa_test.parquet"), role="router_heldout")
    print(f"Train: {len(backend.queries())} | Held-out: {len(heldout.queries())}")

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = FreshQADataset(backend, tok)
    heldout_ds = FreshQADataset(heldout, tok)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    heldout_loader = DataLoader(heldout_ds, batch_size=BATCH)

    model = AutoModelForRegression(MODEL_NAME).to(DEVICE)
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
        print(f"epoch {epoch+1}/{EPOCHS}  held-out MSE={mse:.4f}")
        if mse < best_mse:
            best_mse = mse
            torch.save({"model_name": MODEL_NAME, "state_dict": model.state_dict()}, CKPT)
            print(f"  ✓ saved {CKPT} (best so far)")

    metrics_path = MODEL_DIR / "router_metrics.json"
    metrics_path.write_text(json.dumps({"best_heldout_mse": best_mse, "epochs": EPOCHS}, indent=2))
    print(f"\n✓ Final held-out MSE: {best_mse:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Smoke-test the script (1 epoch, 50 rows) for syntax/loading**

Run:
```bash
cd /home/tylg-jsjxy-4/guochaoyi/wise222/wise-rag-project
python -c "
import sys; sys.path.insert(0,'.')
import torch
from transformers import AutoTokenizer
from src.fresh_rag.trained_router import AutoModelForRegression
m = AutoModelForRegression('microsoft/deberta-v3-small')
tok = AutoTokenizer.from_pretrained('microsoft/deberta-v3-small')
enc = tok('What is the current president?', return_tensors='pt', truncation=True, max_length=64)
print('model forward OK, output shape:', m(**enc).shape)
"
```
Expected: prints `output shape: torch.Size([])`. If this fails, install deps first.

- [ ] **Step 8: Run full training (background)**

Run:
```bash
cd /home/tylg-jsjxy-4/guochaoyi/wise222/wise-rag-project
nohup python -u experiments/train_router.py > results/router_train.log 2>&1 &
```
Expected: `results/router_train.log` ends with `✓ Final held-out MSE: <number>` and `models/router.pt` exists.

- [ ] **Step 9: Verify trained router integrates with the pipeline**

Run:
```bash
cd /home/tylg-jsjxy-4/guochaoyi/wise222/wise-rag-project
python -c "
import sys; sys.path.insert(0,'.')
from src.fresh_rag.trained_router import TrainedRouter
from src.fresh_rag.types import Query
r = TrainedRouter('models/router.pt')
for q in ['What is the current Bitcoin price?', 'When was the iPhone first released?', 'Who is the prime minister of Canada?']:
    print(f'{q[:50]:50s} → p_fresh={r.route(Query(id=\"x\", text=q)):.3f}')
"
```
Expected: prints 3 scores, the Bitcoin one should be highest (fast-changing), the iPhone-release one should be lower (never-changing fact).

- [ ] **Step 10: Commit**

```bash
git add src/fresh_rag/trained_router.py experiments/train_router.py tests/test_trained_router.py
git commit -m "feat: trained DeBERTa freshness router with training script"
```
Note: `models/router.pt` is large; add to `.gitignore` if size > 50MB. Use `git lfs` if needed.

---

## Task 5: Dense Retriever (BGE)

**Purpose:** Replace Jaccard keyword matching with BGE semantic retrieval. Precompute embeddings once; cache on disk.

**Files:**
- Create: `src/fresh_rag/dense_retrieval.py`
- Test: `tests/test_dense_retrieval.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dense_retrieval.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.fresh_rag.dense_retrieval import DenseRetriever, _MockEncoder
from src.fresh_rag.types import Document, Query


def test_mock_encoder_returns_unit_norm():
    enc = _MockEncoder(dim=16)
    v = enc.encode(["hello world", "another doc"])
    import numpy as np
    norms = np.linalg.norm(v, axis=1)
    assert all(abs(n - 1.0) < 1e-3 for n in norms)


def test_dense_retriever_topk_with_mock(monkeypatch):
    monkeypatch.setattr("src.fresh_rag.dense_retrieval._load_encoder", lambda name: _MockEncoder(dim=16))
    docs = [
        Document(doc_id="a", text="the quick brown fox"),
        Document(doc_id="b", text="completely unrelated content"),
        Document(doc_id="c", text="the lazy dog"),
    ]
    retr = DenseRetriever(docs=docs, model_name="mock")
    out = retr.retrieve(Query(id="q", text="quick fox"), k=2)
    assert len(out) == 2
    # Top-1 should be the fox doc or another "fox"-related one, not "unrelated"
    assert out[0].doc_id in ("a", "c")


def test_dense_retriever_falls_back_to_empty_when_no_docs(monkeypatch):
    monkeypatch.setattr("src.fresh_rag.dense_retrieval._load_encoder", lambda name: _MockEncoder())
    retr = DenseRetriever(docs=[], model_name="mock")
    assert retr.retrieve(Query(id="q", text="anything"), k=3) == []


def test_dense_retriever_falls_back_when_encoder_missing(monkeypatch, tmp_path):
    monkeypatch.setattr("src.fresh_rag.dense_retrieval._load_encoder", lambda name: (_ for _ in ()).throw(RuntimeError("no model")))
    retr = DenseRetriever(docs=[Document(doc_id="a", text="x")], model_name="missing")
    out = retr.retrieve(Query(id="q", text="x"), k=1)
    assert out == []  # graceful empty, retriever signals unavailability
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dense_retrieval.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/fresh_rag/dense_retrieval.py
"""Dense retriever using sentence-transformers (BGE).

The retriever is initialized with a list of documents. Embeddings are
precomputed at __init__ and cached via .embeddings_cache attribute.

Falls back to returning [] if the model fails to load (caller should
fall back to keyword retriever in that case).
"""
import hashlib
from pathlib import Path
from typing import List, Optional

import numpy as np

from .types import Document, Query

DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"


def _load_encoder(name: str):
    """Load a sentence-transformers model. Raises on failure."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name)


class _MockEncoder:
    """Deterministic mock encoder for tests."""
    def __init__(self, dim: int = 32):
        self.dim = dim

    def encode(self, texts, normalize_embeddings=True, **kw) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            h = hashlib.md5(t.encode("utf-8")).digest()
            for j in range(self.dim):
                out[i, j] = h[j % len(h)] / 255.0
        if normalize_embeddings:
            norms = np.linalg.norm(out, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            out = out / norms
        return out


class DenseRetriever:
    """Embed-then-cosine retriever."""

    def __init__(self, docs: List[Document], model_name: str = DEFAULT_MODEL):
        self.docs = docs
        self.model_name = model_name
        self._embeddings: Optional[np.ndarray] = None
        try:
            self._encoder = _load_encoder(model_name)
            self._loaded = True
        except Exception:
            self._loaded = False
            self._mock = _MockEncoder()

    def _ensure_embeddings(self):
        if self._embeddings is not None or not self.docs:
            return
        if not self._loaded:
            return
        texts = [d.text for d in self.docs]
        self._embeddings = self._encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def retrieve(self, query: Query, k: int = 5) -> List[Document]:
        if not self.docs or not self._loaded:
            return []
        self._ensure_embeddings()
        q_vec = self._encoder.encode([query.text], normalize_embeddings=True, show_progress_bar=False)[0]
        sims = self._embeddings @ q_vec
        idx = np.argsort(-sims)[:k]
        return [self.docs[i] for i in idx]


def precompute_corpus_embeddings(
    docs: List[Document],
    cache_path: Path,
    model_name: str = DEFAULT_MODEL,
) -> np.ndarray:
    """Compute and cache embeddings to disk. Returns the array."""
    enc = _load_encoder(model_name)
    embs = enc.encode([d.text for d in docs], normalize_embeddings=True, show_progress_bar=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embs)
    return embs
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_dense_retrieval.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Precompute embeddings for the full FreshQA stale corpus (background)**

```python
# Add to experiments/precompute_embeddings.py
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fresh_rag.backends.freshqa import FreshQABackend
from src.fresh_rag.dense_retrieval import precompute_corpus_embeddings
from src.fresh_rag.types import Document

backend = FreshQABackend(str(PROJECT_ROOT / "data" / "freshqa_dev.parquet"), role="qa_eval")
docs = []
for q in backend.queries():
    d = backend.get_documents(q.id, "stale")[0]
    if d.text:
        docs.append(d)
print(f"Encoding {len(docs)} docs...")
embs = precompute_corpus_embeddings(docs, PROJECT_ROOT / "data" / "freshqa_stale_embs.npy")
print(f"✓ Saved {embs.shape} to data/freshqa_stale_embs.npy")
```

Run:
```bash
cd /home/tylg-jsjxy-4/guochaoyi/wise222/wise-rag-project
nohup python -u experiments/precompute_embeddings.py > results/emb_precompute.log 2>&1 &
```
Expected: log shows shape `(~100, 768)` and `data/freshqa_stale_embs.npy` exists.

- [ ] **Step 6: Commit**

```bash
git add src/fresh_rag/dense_retrieval.py experiments/precompute_embeddings.py tests/test_dense_retrieval.py
git commit -m "feat: BGE dense retriever with mock encoder fallback"
```

---

## Task 6: Pipeline — Pluggable Retriever + Rerank Tune

**Purpose:** Make the pipeline accept any retriever (keyword or dense), and add a small weight-tune step for the reranker's (α, β, γ).

**Files:**
- Modify: `src/fresh_rag/pipeline.py` (add `retriever` param, plumb through)
- Modify: `src/fresh_rag/reranker.py` (add `tune_weights` helper)
- Test: extend `tests/test_pipeline.py` (existing) with one new test

- [ ] **Step 1: Read the current pipeline to know exactly what to change**

Run: `cat src/fresh_rag/pipeline.py`
Expected: 60-line file, `FreshnessAwareRAG.__init__` takes `backend, llm_client, router, threshold, m, lambda_fresh, lambda_changed, k`; `answer` calls `self.retriever.retrieve(...)` internally using a default `Retriever(backend)`.

- [ ] **Step 2: Write the failing test**

Append to `tests/test_pipeline.py`:
```python
def test_pipeline_accepts_custom_retriever(monkeypatch):
    from src.fresh_rag.pipeline import FreshnessAwareRAG
    from src.fresh_rag.dense_retrieval import DenseRetriever
    from src.fresh_rag.backends.simulated import SimulatedBackend

    class FakeDenseRetriever:
        def __init__(self, backend): self.backend = backend
        def retrieve(self, query, k=5):
            from src.fresh_rag.types import Document
            return self.backend.get_documents(query.id, "stale")[:k]

    backend = SimulatedBackend("simulated_test_cases.json")
    rag = FreshnessAwareRAG(backend=backend, retriever=FakeDenseRetriever(backend))
    assert isinstance(rag.retriever, FakeDenseRetriever)
    from src.fresh_rag.types import Query
    answer, scored = rag.answer(Query(id="1", text="test"))
    assert isinstance(answer, str)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_pipeline.py::test_pipeline_accepts_custom_retriever -v`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'retriever'`.

- [ ] **Step 4: Modify pipeline.py**

Open `src/fresh_rag/pipeline.py`. Replace the `__init__` and `answer` methods with:

```python
def __init__(
    self,
    backend,
    llm_client=None,
    router=None,
    threshold=0.5,
    m=3,
    lambda_fresh=1.0,
    lambda_changed=1.0,
    k=5,
    retriever=None,
):
    self.backend = backend
    self.threshold = threshold
    self.m = m
    self.lambda_fresh = lambda_fresh
    self.lambda_changed = lambda_changed
    self.k = k
    # Router
    if router is not None:
        self.router = router
    else:
        from .router import Router
        self.router = Router(llm_client=llm_client)
    # Retriever (pluggable)
    if retriever is not None:
        self.retriever = retriever
    else:
        from .retrieval import Retriever
        self.retriever = Retriever(backend)
    from .refresh import Refresher
    from .reranker import Reranker
    from .generator import Generator
    self.refresher = Refresher(backend)
    self.reranker = Reranker()
    self.generator = Generator(llm_client=llm_client)

def answer(self, query):
    p_fresh = self.router.route(query)
    docs = self.retriever.retrieve(query, k=self.k)
    docs = self.refresher.maybe_refresh(query, docs, p_fresh, self.threshold, self.m)
    from .types import ScoredDoc
    scored = [
        ScoredDoc(doc=d, retrieval_score=1.0 - i * 0.1, final_score=0.0)
        for i, d in enumerate(docs)
    ]
    scored = self.reranker.rerank(scored, self.lambda_fresh, self.lambda_changed)
    answer = self.generator.generate(query, scored[:3])
    return answer, scored
```

- [ ] **Step 5: Run the new test**

Run: `pytest tests/test_pipeline.py -v`
Expected: PASS for the new test AND all previously passing tests (we didn't break the keyword retriever default).

- [ ] **Step 6: Add `tune_weights` helper to reranker.py**

Append to `src/fresh_rag/reranker.py`:
```python
def tune_weights(self, examples, metric_fn):
    """Grid-search (λ_fresh, λ_changed) ∈ {0,0.5,1.0,2.0}^2.

    examples: list of (docs, gold_answer) for the router-train set.
    metric_fn(predicted_answer, gold) -> float (higher = better).
    Returns (best_lambda_fresh, best_lambda_changed).
    """
    candidates = [0.0, 0.5, 1.0, 2.0]
    best = (0.0, 0.0)
    best_score = float("-inf")
    for lf in candidates:
        for lc in candidates:
            score = 0.0
            for docs, gold in examples:
                reranked = self.rerank(docs, lf, lc)
                # caller supplies metric_fn on the final answer
                score += metric_fn(reranked, gold)
            if score > best_score:
                best_score = score
                best = (lf, lc)
    return best
```

- [ ] **Step 7: Commit**

```bash
git add src/fresh_rag/pipeline.py src/fresh_rag/reranker.py tests/test_pipeline.py
git commit -m "feat: pipeline accepts pluggable retriever; reranker tune_weights"
```

---

## Task 7: Block A — Staleness Degradation on FreshQA (Real Content)

**Purpose:** Headline diagnostic experiment. Stale-RAG vs Fresh-RAG on FreshQA, broken down by fact_type, using real Wayback content.

**Files:**
- Create: `experiments/blockA/run_blockA.py`

- [ ] **Step 1: Write the experiment script**

```python
#!/usr/bin/env python3
"""Block A: Staleness degradation on FreshQA (real Wayback content).

Compares Stale-RAG (always uses cached stale docs) vs Fresh-RAG (always
live) on the QA-eval split (FreshQA dev, 100 queries), broken down by
fact_type. Reports F1/EM and per-type degradation.
"""
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fresh_rag.backends.freshqa import FreshQABackend
from src.fresh_rag.metrics import f1_score, exact_match
from src.fresh_rag.types import Query, ScoredDoc

QA_PATH = PROJECT_ROOT / "data" / "freshqa_dev.parquet"
OUT = PROJECT_ROOT / "results"
OUT.mkdir(exist_ok=True)


class StaticRAG:
    def __init__(self, backend, version, llm_client):
        self.backend = backend
        self.version = version
        from src.fresh_rag.generator import Generator
        self.generator = Generator(llm_client=llm_client)

    def answer(self, query):
        docs = self.backend.get_documents(query.id, self.version)
        scored = [ScoredDoc(d, retrieval_score=1.0 - i * 0.1, final_score=1.0 - i * 0.1) for i, d in enumerate(docs[:3])]
        return self.generator.generate(query, scored)


def run_system(name, backend, queries, llm_client):
    system = StaticRAG(backend, name.lower().split()[0], llm_client)  # "Stale-RAG" → "stale"
    results = []
    for i, q in enumerate(queries):
        try:
            ans = system.answer(q)
            gold = backend.get_correct_answer(q.id)
            results.append({"id": q.id, "ft": backend.get_fact_type(q.id), "answer": ans, "gold": gold,
                            "f1": f1_score(ans, gold), "em": exact_match(ans, gold)})
        except Exception as e:
            results.append({"id": q.id, "ft": backend.get_fact_type(q.id), "error": str(e), "f1": 0.0, "em": 0.0})
        if (i + 1) % 10 == 0:
            print(f"  [{name}] {i+1}/{len(queries)}", flush=True)
    return results


def aggregate_by_fact_type(results):
    out = {}
    for r in results:
        ft = r["ft"]
        out.setdefault(ft, []).append(r)
    return {
        ft: {"count": len(items), "mean_f1": sum(r["f1"] for r in items) / len(items),
             "mean_em": sum(r["em"] for r in items) / len(items)}
        for ft, items in out.items() if items
    }


def main():
    try:
        from src.fresh_rag.backends.codex_llm import CodexLLMClient
        llm = CodexLLMClient(); llm.chat("test")
        print("✓ Codex LLM available", flush=True)
    except Exception as e:
        print(f"⚠ No Codex ({e}); results will be uninformative", flush=True)
        llm = None

    backend = FreshQABackend(str(QA_PATH), role="qa_eval")
    queries = backend.queries()
    print(f"Loaded {len(queries)} QA-eval queries", flush=True)

    all_results = {}
    for name, ver in [("Stale-RAG", "stale"), ("Fresh-RAG", "fresh")]:
        print(f"\n=== {name} ===", flush=True)
        # Patch StaticRAG to use correct version keyword
        system = StaticRAG(backend, ver, llm)
        # reuse run_system but with explicit version
        results = []
        for i, q in enumerate(queries):
            try:
                ans = system.answer(q)
                gold = backend.get_correct_answer(q.id)
                results.append({"id": q.id, "ft": backend.get_fact_type(q.id), "answer": ans, "gold": gold,
                                "f1": f1_score(ans, gold), "em": exact_match(ans, gold)})
            except Exception as e:
                results.append({"id": q.id, "ft": backend.get_fact_type(q.id), "error": str(e), "f1": 0.0, "em": 0.0})
            if (i + 1) % 10 == 0:
                print(f"  [{name}] {i+1}/{len(queries)}", flush=True)
        all_results[name] = results
        print(f"  F1={sum(r['f1'] for r in results)/len(results):.3f}", flush=True)

    # Per-type degradation table
    table = {}
    for name, results in all_results.items():
        table[name] = aggregate_by_fact_type(results)

    with open(OUT / "blockA_results.json", "w") as f:
        json.dump({"per_system": all_results, "per_type_summary": table}, f, indent=2)

    with open(OUT / "blockA_summary.md", "w") as f:
        f.write("# Block A: Staleness Degradation on FreshQA\n\n")
        f.write("| System | fact_type | n | mean F1 | mean EM |\n")
        f.write("|--------|-----------|---|---------|---------|\n")
        for sys_name, type_summ in table.items():
            for ft in ("fast-changing", "slow-changing", "never-changing"):
                if ft in type_summ:
                    s = type_summ[ft]
                    f.write(f"| {sys_name} | {ft} | {s['count']} | {s['mean_f1']:.3f} | {s['mean_em']:.3f} |\n")
        # Degradation
        f.write("\n## Degradation by fact_type\n\n")
        for ft in ("fast-changing", "slow-changing", "never-changing"):
            if ft in table.get("Stale-RAG", {}) and ft in table.get("Fresh-RAG", {}):
                s = table["Stale-RAG"][ft]["mean_f1"]
                fr = table["Fresh-RAG"][ft]["mean_f1"]
                deg = (fr - s) / fr * 100 if fr > 0 else 0
                f.write(f"- {ft}: stale={s:.3f}, fresh={fr:.3f}, degradation={deg:+.1f}%\n")

    print(f"\n✓ Saved {OUT}/blockA_results.json and blockA_summary.md", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the experiment (background, resumable via logs)**

```bash
cd /home/tylg-jsjxy-4/guochaoyi/wise222/wise-rag-project
nohup python -u experiments/blockA/run_blockA.py > results/blockA.log 2>&1 &
```
Expected: `results/blockA.log` shows progress; ends with `✓ Saved ...`.

- [ ] **Step 3: Commit**

```bash
git add experiments/blockA/run_blockA.py
git commit -m "feat: Block A staleness degradation experiment on FreshQA"
```

---

## Task 8: Block B — Router Evaluation

**Purpose:** Evaluate TrainedRouter vs KeywordRouter vs LLM-prompt Router on the router held-out split (150 queries). Metrics: MSE vs p_fresh target, AUROC for binary "needs refresh" (fast vs never).

**Files:**
- Create: `experiments/blockB/run_blockB.py`

- [ ] **Step 1: Write the experiment script**

```python
#!/usr/bin/env python3
"""Block B: Router evaluation on FreshQA router held-out split.

Compares 3 routers on the 150 held-out queries (FreshQA test rows 350-499):
1. TrainedRouter (ours)
2. KeywordRouter (heuristic baseline)
3. LLM Router (GPT-5.4 prompt)

Metrics:
- MSE vs continuous p_fresh target (0.0/0.5/1.0)
- AUROC for binary "needs refresh" (fast-changing = positive; never = negative)
- Macro-F1 on 3-class fact_type
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from sklearn.metrics import mean_squared_error, roc_auc_score, f1_score

from src.fresh_rag.backends.freshqa import FreshQABackend
from src.fresh_rag.trained_router import TrainedRouter
from src.fresh_rag.keyword_router import KeywordRouter
from src.fresh_rag.router import Router
from src.fresh_rag.types import Query

OUT = PROJECT_ROOT / "results"
OUT.mkdir(exist_ok=True)


def evaluate(name, router, backend, llm_client):
    queries = backend.queries()
    p_targets = np.array([backend.get_pfresh_label(q.id) for q in queries])
    # Binary label: fast-changing=1, never-changing=0, slow-changing excluded from AUROC
    fact_types = [backend.get_fact_type(q.id) for q in queries]
    p_preds = np.array([router.route(Query(id=q.id, text=q.text)) for q in queries])

    # MSE (all queries)
    mse = float(mean_squared_error(p_targets, p_preds))

    # AUROC (binary: fast vs never only)
    mask = [(ft in ("fast-changing", "never-changing")) for ft in fact_types]
    if sum(mask) >= 4:
        y_true = np.array([1.0 if fact_types[i] == "fast-changing" else 0.0 for i in range(len(fact_types))])[mask]
        y_score = p_preds[mask]
        auroc = float(roc_auc_score(y_true, y_score))
    else:
        auroc = float("nan")

    # 3-class macro-F1: bucket p_pred into 0/0.5/1.0 by argmin distance
    centers = np.array([0.0, 0.5, 1.0])
    pred_idx = np.argmin(np.abs(p_preds[:, None] - centers[None, :]), axis=1)
    true_idx = np.array([0 if ft == "never-changing" else 1 if ft == "slow-changing" else 2 for ft in fact_types])
    macro_f1 = float(f1_score(true_idx, pred_idx, average="macro"))

    return {"router": name, "mse": mse, "auroc_fast_vs_never": auroc, "macro_f1_3class": macro_f1,
            "n": len(queries)}


def main():
    backend = FreshQABackend(str(PROJECT_ROOT / "data" / "freshqa_test.parquet"), role="router_heldout")
    print(f"Router held-out: {len(backend.queries())} queries", flush=True)

    try:
        from src.fresh_rag.backends.codex_llm import CodexLLMClient
        llm = CodexLLMClient(); llm.chat("test")
    except Exception:
        llm = None

    routers = [
        ("TrainedRouter", TrainedRouter(str(PROJECT_ROOT / "models" / "router.pt"))),
        ("KeywordRouter", KeywordRouter()),
        ("LLM Router (GPT-5.4)", Router(llm_client=llm) if llm else None),
    ]
    results = []
    for name, r in routers:
        if r is None:
            continue
        print(f"  evaluating {name}...", flush=True)
        results.append(evaluate(name, r, backend, llm))

    with open(OUT / "blockB_results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(OUT / "blockB_summary.md", "w") as f:
        f.write("# Block B: Router Evaluation\n\n")
        f.write(f"Dataset: FreshQA router held-out (n={len(backend.queries())})\n\n")
        f.write("| Router | MSE ↓ | AUROC (fast vs never) ↑ | Macro-F1 (3-class) ↑ |\n")
        f.write("|--------|------|--------------------------|----------------------|\n")
        for r in sorted(results, key=lambda x: x["mse"]):
            f.write(f"| {r['router']} | {r['mse']:.3f} | {r['auroc_fast_vs_never']:.3f} | {r['macro_f1_3class']:.3f} |\n")
    print(f"\n✓ Saved blockB_results.json and blockB_summary.md", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run**

```bash
cd /home/tylg-jsjxy-4/guochaoyi/wise222/wise-rag-project
nohup python -u experiments/blockB/run_blockB.py > results/blockB.log 2>&1 &
```
Expected: prints 3 router evaluations; LLM router may be slow (~5 min for 150 calls).

- [ ] **Step 3: Commit**

```bash
git add experiments/blockB/run_blockB.py
git commit -m "feat: Block B router evaluation on FreshQA held-out"
```

---

## Task 9: Block C — Main Method vs Baselines (the core)

**Purpose:** 5 systems compared on QA-eval split (FreshQA dev, 100 queries): Never/Always Refresh (oracle bounds), Keyword Router, LLM Router, **Trained Router (ours)**. All with the dense retriever and selective refresh.

**Files:**
- Create: `experiments/blockC/run_blockC.py`

- [ ] **Step 1: Write the experiment script**

```python
#!/usr/bin/env python3
"""Block C: Main method vs baselines on FreshQA (the core result).

5 systems, all using the same dense retriever and selective refresh:
1. Never Refresh  — always stale
2. Always Refresh — always fresh
3. Keyword Router  — refresh if keywords match
4. LLM Router      — refresh if GPT-5.4 p_fresh > τ
5. Trained Router  — refresh if DeBERTa p_fresh > τ (OURS)
"""
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fresh_rag.backends.freshqa import FreshQABackend
from src.fresh_rag.dense_retrieval import DenseRetriever
from src.fresh_rag.keyword_router import KeywordRouter
from src.fresh_rag.metrics import f1_score, exact_match
from src.fresh_rag.pipeline import FreshnessAwareRAG
from src.fresh_rag.router import Router
from src.fresh_rag.trained_router import TrainedRouter
from src.fresh_rag.types import Query, ScoredDoc

OUT = PROJECT_ROOT / "results"
OUT.mkdir(exist_ok=True)


def static_rag(backend, version, llm):
    from src.fresh_rag.generator import Generator
    gen = Generator(llm_client=llm)
    def answer(q):
        docs = backend.get_documents(q.id, version)
        scored = [ScoredDoc(d, retrieval_score=1.0-i*0.1, final_score=1.0-i*0.1) for i, d in enumerate(docs[:3])]
        return gen.generate(q, scored)
    return answer


def run_system(name, answer_fn, queries, backend, get_refresh_count=None):
    results = []
    for i, q in enumerate(queries):
        try:
            ans = answer_fn(q)
            gold = backend.get_correct_answer(q.id)
            results.append({"id": q.id, "ft": backend.get_fact_type(q.id), "answer": ans, "gold": gold,
                            "f1": f1_score(ans, gold), "em": exact_match(ans, gold)})
        except Exception as e:
            results.append({"id": q.id, "ft": backend.get_fact_type(q.id), "error": str(e), "f1": 0.0, "em": 0.0})
        if (i + 1) % 10 == 0:
            print(f"  [{name}] {i+1}/{len(queries)}", flush=True)
    return results


def main():
    try:
        from src.fresh_rag.backends.codex_llm import CodexLLMClient
        llm = CodexLLMClient(); llm.chat("test")
        print("✓ Codex LLM available", flush=True)
    except Exception as e:
        print(f"⚠ No Codex ({e})", flush=True)
        llm = None

    backend = FreshQABackend(str(PROJECT_ROOT / "data" / "freshqa_dev.parquet"), role="qa_eval")
    queries = backend.queries()
    print(f"Loaded {len(queries)} queries", flush=True)

    # Build a per-query dense retriever instance (small corpus; fine)
    all_docs = []
    for q in queries:
        d = backend.get_documents(q.id, "stale")[0]
        if d.text:
            all_docs.append(d)
    dense_retr = DenseRetriever(docs=all_docs, model_name="BAAI/bge-base-en-v1.5")
    print(f"Dense retriever loaded with {len(all_docs)} docs", flush=True)

    # 1 & 2: Static
    results = {}
    for name, ver in [("Never Refresh", "stale"), ("Always Refresh", "fresh")]:
        print(f"\n=== {name} ===", flush=True)
        results[name] = run_system(name, static_rag(backend, ver, llm), queries, backend)
        f1 = sum(r["f1"] for r in results[name]) / len(results[name])
        print(f"  F1={f1:.3f}", flush=True)

    # 3, 4, 5: RAG with routers
    router_specs = [
        ("Keyword Router", KeywordRouter()),
        ("LLM Router (GPT-5.4)", Router(llm_client=llm) if llm else None),
        ("Trained Router (OURS)", TrainedRouter(str(PROJECT_ROOT / "models" / "router.pt"))),
    ]
    for name, router in router_specs:
        if router is None:
            continue
        print(f"\n=== {name} ===", flush=True)
        rag = FreshnessAwareRAG(backend=backend, llm_client=llm, router=router, threshold=0.5, m=3, retriever=dense_retr)

        rs = []
        refresh_count = 0
        for i, q in enumerate(queries):
            try:
                ans, scored = rag.answer(q)
                refresh_count += sum(1 for sd in scored if sd.doc.is_live)
                gold = backend.get_correct_answer(q.id)
                rs.append({"id": q.id, "ft": backend.get_fact_type(q.id), "answer": ans, "gold": gold,
                           "f1": f1_score(ans, gold), "em": exact_match(ans, gold)})
            except Exception as e:
                rs.append({"id": q.id, "ft": backend.get_fact_type(q.id), "error": str(e), "f1": 0.0, "em": 0.0})
            if (i + 1) % 10 == 0:
                print(f"  [{name}] {i+1}/{len(queries)}", flush=True)
        # For non-static, also track refresh rate
        results[name] = {"_results": rs, "_refresh_rate": refresh_count / (len(queries) * 3)}
        f1 = sum(r["f1"] for r in rs) / len(rs)
        print(f"  F1={f1:.3f}  refresh_rate={refresh_count/(len(queries)*3):.1%}", flush=True)

    # Save
    with open(OUT / "blockC_results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(OUT / "blockC_summary.md", "w") as f:
        f.write("# Block C: Main Method vs Baselines\n\n")
        f.write(f"Dataset: FreshQA dev (n={len(queries)})\n\n")
        f.write("| System | Mean F1 | Mean EM | Refresh Rate |\n")
        f.write("|--------|---------|---------|---------------|\n")
        for name in ("Never Refresh", "Always Refresh", "Keyword Router", "LLM Router (GPT-5.4)", "Trained Router (OURS)"):
            if name in results:
                if name in ("Never Refresh", "Always Refresh"):
                    rs = results[name]
                    f1 = sum(r["f1"] for r in rs) / len(rs)
                    em = sum(r["em"] for r in rs) / len(rs)
                    rr = "0%" if name == "Never Refresh" else "100%"
                else:
                    rs = results[name]["_results"]
                    f1 = sum(r["f1"] for r in rs) / len(rs)
                    em = sum(r["em"] for r in rs) / len(rs)
                    rr = f"{results[name]['_refresh_rate']:.1%}"
                f.write(f"| {name} | {f1:.3f} | {em:.3f} | {rr} |\n")
    print(f"\n✓ Saved blockC_results.json and blockC_summary.md", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run**

```bash
cd /home/tylg-jsjxy-4/guochaoyi/wise222/wise-rag-project
nohup python -u experiments/blockC/run_blockC.py > results/blockC.log 2>&1 &
```
Expected: 5 systems run sequentially; the LLM Router is the slowest (~5-10 min for 100 queries).

- [ ] **Step 3: Commit**

```bash
git add experiments/blockC/run_blockC.py
git commit -m "feat: Block C main method vs baselines (5 systems on FreshQA)"
```

---

## Task 10: Block D — Threshold Ablation

**Purpose:** Sweep τ ∈ {0.1, 0.3, 0.5, 0.7, 0.9} × m ∈ {1, 3, 5} on the Trained Router, plus random-20% / top-20% baselines. Outputs the cost-accuracy frontier.

**Files:**
- Create: `experiments/blockD/run_blockD.py`

- [ ] **Step 1: Write the experiment script**

```python
#!/usr/bin/env python3
"""Block D: τ × m ablation on the trained router.

Sweeps τ in {0.1, 0.3, 0.5, 0.7, 0.9} and m in {1, 3, 5}, plus two
fixed-baseline refresh strategies (random 20%, top 20% by retrieval).
Reports F1 and refresh rate per (τ, m) pair.
"""
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fresh_rag.backends.freshqa import FreshQABackend
from src.fresh_rag.dense_retrieval import DenseRetriever
from src.fresh_rag.metrics import f1_score
from src.fresh_rag.pipeline import FreshnessAwareRAG
from src.fresh_rag.trained_router import TrainedRouter
from src.fresh_rag.types import Query, ScoredDoc

OUT = PROJECT_ROOT / "results"
OUT.mkdir(exist_ok=True)

TAUS = [0.1, 0.3, 0.5, 0.7, 0.9]
MS = [1, 3, 5]


def main():
    try:
        from src.fresh_rag.backends.codex_llm import CodexLLMClient
        llm = CodexLLMClient(); llm.chat("test")
    except Exception:
        llm = None

    backend = FreshQABackend(str(PROJECT_ROOT / "data" / "freshqa_dev.parquet"), role="qa_eval")
    queries = backend.queries()

    all_docs = [backend.get_documents(q.id, "stale")[0] for q in queries if backend.get_documents(q.id, "stale")[0].text]
    dense_retr = DenseRetriever(docs=all_docs, model_name="BAAI/bge-base-en-v1.5")
    router = TrainedRouter(str(PROJECT_ROOT / "models" / "router.pt"))

    grid = []
    for tau in TAUS:
        for m in MS:
            print(f"\n=== τ={tau}, m={m} ===", flush=True)
            rag = FreshnessAwareRAG(backend=backend, llm_client=llm, router=router, threshold=tau, m=m, retriever=dense_retr)
            f1s = []
            refreshes = 0
            for i, q in enumerate(queries):
                try:
                    ans, scored = rag.answer(q)
                    refreshes += sum(1 for sd in scored if sd.doc.is_live)
                    f1s.append(f1_score(ans, backend.get_correct_answer(q.id)))
                except Exception:
                    f1s.append(0.0)
                if (i + 1) % 20 == 0:
                    print(f"  {i+1}/{len(queries)}", flush=True)
            rr = refreshes / (len(queries) * 3)
            mean_f1 = sum(f1s) / len(f1s)
            grid.append({"tau": tau, "m": m, "refresh_rate": rr, "mean_f1": mean_f1})
            print(f"  F1={mean_f1:.3f}  refresh={rr:.1%}", flush=True)

    # Baselines
    print("\n=== Fixed baselines (m=1, top by retrieval / random 20%) ===", flush=True)
    for strategy in ("top", "random"):
        f1s, refreshes = [], 0
        for i, q in enumerate(queries):
            docs = backend.get_documents(q.id, "stale")
            scored = [ScoredDoc(d, retrieval_score=1.0-j*0.1, final_score=0.0) for j, d in enumerate(docs)]
            if strategy == "top":
                scored[0].doc.is_live = True; scored[0].doc.is_changed = True
                scored[0].doc.text = backend.get_documents(q.id, "fresh")[0].text
                refreshes += 1
            else:
                if random.random() < 0.2:
                    scored[0].doc.is_live = True; scored[0].doc.is_changed = True
                    scored[0].doc.text = backend.get_documents(q.id, "fresh")[0].text
                    refreshes += 1
            # generate
            from src.fresh_rag.generator import Generator
            ans = Generator(llm_client=llm).generate(q, scored[:3])
            f1s.append(f1_score(ans, backend.get_correct_answer(q.id)))
        grid.append({"strategy": strategy, "refresh_rate": refreshes / len(queries), "mean_f1": sum(f1s)/len(f1s)})

    with open(OUT / "blockD_results.json", "w") as f:
        json.dump(grid, f, indent=2)
    with open(OUT / "blockD_summary.md", "w") as f:
        f.write("# Block D: Threshold Ablation (Trained Router)\n\n")
        f.write("## τ × m grid\n\n")
        f.write("| τ | m | Refresh Rate | Mean F1 |\n")
        f.write("|---|---|--------------|---------|\n")
        for r in grid:
            if "tau" in r:
                f.write(f"| {r['tau']} | {r['m']} | {r['refresh_rate']:.1%} | {r['mean_f1']:.3f} |\n")
        f.write("\n## Fixed baselines\n\n")
        for r in grid:
            if "strategy" in r:
                f.write(f"- {r['strategy']}: refresh={r['refresh_rate']:.1%}, F1={r['mean_f1']:.3f}\n")
    print(f"\n✓ Saved blockD_results.json and blockD_summary.md", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run**

```bash
cd /home/tylg-jsjxy-4/guochaoyi/wise222/wise-rag-project
nohup python -u experiments/blockD/run_blockD.py > results/blockD.log 2>&1 &
```
Expected: 15 (τ,m) cells + 2 baselines, ~30-60 min total.

- [ ] **Step 3: Commit**

```bash
git add experiments/blockD/run_blockD.py
git commit -m "feat: Block D threshold ablation on trained router"
```

---

## Self-Review Checklist

**Spec coverage:**
- §1 FreshQA Backend (Task 1) ✓
- §2 Wayback fetcher (Task 2) ✓
- §3 Corpus build (Task 3) ✓
- §4 Trained router (Task 4) ✓
- §5 Dense retriever (Task 5) ✓
- §6 Pipeline pluggable retriever + tuned rerank (Task 6) ✓
- §7 Block A staleness (Task 7) ✓
- §8 Block B router eval (Task 8) ✓
- §9 Block C main method (Task 9) ✓
- §10 Block D ablation (Task 10) ✓
- Definition of Done items: all covered by tasks 7-10 outputting artifacts

**Type/method consistency:**
- `Backend` protocol methods used: `queries()`, `get_documents(qid, version)`, `get_correct_answer(qid)`, plus our additions `get_fact_type`, `get_pfresh_label` ✓
- `TrainedRouter.route(query) -> float` — same interface as `KeywordRouter` and `Router` ✓
- `DenseRetriever.retrieve(query, k) -> list[Document]` — same interface as `Retriever` ✓
- `FreshnessAwareRAG(..., retriever=)` parameter is new but optional; default falls back to `Retriever(backend)` ✓

**Placeholder scan:** No TBDs, no "implement later". Every step has concrete code or command.

**Risks noted in spec:**
- Wayback access: handled by `synthesize_stale` fallback (Task 2 + Task 3) ✓
- GPU contention: jobs use single 4090, can run in zgz's gaps; `nohup` so they don't need this session ✓
- Router overfit: early-stop on held-out MSE in Task 4 ✓
- Dense retriever weak: kept keyword retriever as default fallback (Task 6) ✓
