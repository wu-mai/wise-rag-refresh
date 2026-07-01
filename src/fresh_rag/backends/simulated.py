"""Simulated backend using hand-crafted test cases."""
import json
from typing import List
from ..types import Document, Query


class SimulatedBackend:
    """Backend for the 10 hand-crafted test cases in simulated_test_cases.json.

    Each entry has both stale_context and fresh_context. The fresh context is
    treated as the live version (is_live=True, is_changed=True).
    """

    def __init__(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._entries = {str(e["id"]): e for e in data}

    def queries(self) -> List[Query]:
        return [
            Query(
                id=str(e["id"]),
                text=e["question"],
                freshness_sensitive=bool(e.get("freshness_sensitive", False)),
            )
            for e in self._entries.values()
        ]

    def get_documents(self, query_id: str, version: str) -> List[Document]:
        if version not in ("stale", "fresh"):
            raise ValueError(f"version must be 'stale' or 'fresh', got {version!r}")
        entry = self._entries.get(query_id)
        if entry is None:
            return []
        text_field = "fresh_context" if version == "fresh" else "stale_context"
        text = entry.get(text_field, "")
        is_live = version == "fresh"
        is_changed = version == "fresh"
        return [
            Document(
                doc_id=f"{query_id}-{version}",
                text=text,
                url=f"simulated://{query_id}",
                timestamp="2026-06-12" if is_live else "2024-01-01",
                age_days=0 if is_live else 900,
                is_live=is_live,
                is_changed=is_changed,
            )
        ]

    def get_correct_answer(self, query_id: str) -> str:
        """Helper for evaluation. Returns the gold answer."""
        entry = self._entries.get(query_id)
        if entry is None:
            return ""
        return entry.get("correct_answer", "")
