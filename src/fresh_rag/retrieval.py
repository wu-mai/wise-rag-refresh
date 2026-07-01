"""Keyword-overlap retriever.

A simple, deterministic stand-in for a dense retriever. Computes Jaccard
similarity between query tokens and document tokens. This is intentional
for Block 1 — we want to isolate the freshness signal.
"""
import re
from typing import List
from .backends.base import Backend
from .types import Document, Query


def _tokenize(text: str) -> set:
    return set(re.findall(r"\w+", text.lower()))


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class Retriever:
    def __init__(self, backend: Backend):
        self.backend = backend

    def retrieve(self, query: Query, k: int = 5) -> List[Document]:
        """Return top-k documents for the query, ranked by Jaccard overlap."""
        # Always start from the cached (stale) version
        candidates = self.backend.get_documents(query.id, version="stale")
        if not candidates:
            return []
        query_tokens = _tokenize(query.text)
        scored = [
            (doc, _jaccard(query_tokens, _tokenize(doc.text)))
            for doc in candidates
        ]
        # Sort by score desc, stable on ties
        scored.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored[:k]]
