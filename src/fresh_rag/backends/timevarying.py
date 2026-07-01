"""Time-varying QA backend: real time-sensitive queries with curated stale/fresh context.

Used for Block C/D extended evaluation on a benchmark where stale and fresh texts
*substantively* differ (not just a warning prompt). The 60 cases come from
data/time_varying_benchmark.json (50 LLM-generated + 10 hand-crafted).
"""
import json
from pathlib import Path
from typing import List

from ..types import Document, Query

DATA_DIR = Path(__file__).resolve().parents[3] / "data"


class TimeVaryingBackend:
    def __init__(self, path: str = None):
        p = path or (DATA_DIR / "time_varying_benchmark.json")
        with open(p, "r", encoding="utf-8") as f:
            self._entries = json.load(f)
        # all entries are freshness_sensitive
        for e in self._entries:
            e["freshness_sensitive"] = True

    def queries(self) -> List[Query]:
        return [
            Query(
                id=str(e["id"]),
                text=e["question"],
                freshness_sensitive=True,
            )
            for e in self._entries
        ]

    def get_documents(self, query_id: str, version: str) -> List[Document]:
        if version not in ("stale", "fresh"):
            raise ValueError(f"version must be 'stale' or 'fresh', got {version!r}")
        entry = next((e for e in self._entries if str(e["id"]) == str(query_id)), None)
        if entry is None:
            return []
        text_field = "fresh_context" if version == "fresh" else "stale_context"
        text = entry.get(text_field, "")
        is_live = version == "fresh"
        return [
            Document(
                doc_id=f"{query_id}-{version}",
                text=text,
                url=entry.get("url", f"timevarying://{query_id}"),
                timestamp="2026-06" if is_live else "2019-01",
                age_days=0 if is_live else 2500,
                is_live=is_live,
                is_changed=is_live,
            )
        ]

    def get_correct_answer(self, query_id: str) -> str:
        entry = next((e for e in self._entries if str(e["id"]) == str(query_id)), None)
        if entry is None:
            return ""
        return entry.get("correct_answer", "")

    def get_fact_type(self, query_id: str) -> str:
        # all time-varying; default to "fast-changing"
        return "fast-changing"

    def get_pfresh_label(self, query_id: str) -> float:
        return 1.0
