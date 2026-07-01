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
