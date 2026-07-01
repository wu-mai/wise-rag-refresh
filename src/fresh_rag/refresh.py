"""Selective refresh module: swap top-m cached docs for live versions.

Decides whether to refresh based on the router's p_fresh score and a
threshold. If refresh is triggered, fetches live content from the backend
and substitutes the top-m documents.
"""
from typing import List
from .backends.base import Backend
from .types import Document, Query


class Refresher:
    def __init__(self, backend: Backend):
        self.backend = backend

    def maybe_refresh(
        self,
        query: Query,
        docs: List[Document],
        p_fresh: float,
        threshold: float,
        m: int,
    ) -> List[Document]:
        """If p_fresh > threshold, swap top-m docs for live versions.

        On backend failure, returns the cached docs unchanged (graceful
        degradation per the proposal's failure modes table).
        """
        if not docs or p_fresh <= threshold:
            return docs

        # Determine how many to actually swap
        n_swap = min(m, len(docs))

        try:
            live_docs = self.backend.get_documents(query.id, version="fresh")
        except Exception:
            # Graceful fallback: keep cached docs as-is
            return list(docs)

        if not live_docs:
            return list(docs)

        # Build a lookup of live docs by doc_id (stripped of the version suffix)
        # The SimulatedBackend uses "{query_id}-fresh" as doc_id, so we match by prefix
        live_by_base_id = {}
        for ld in live_docs:
            base_id = ld.doc_id.replace("-fresh", "").replace("-stale", "")
            live_by_base_id[base_id] = ld

        result = []
        swapped = 0
        for i, doc in enumerate(docs):
            if swapped < n_swap:
                base_id = doc.doc_id.replace("-fresh", "").replace("-stale", "")
                if base_id in live_by_base_id:
                    result.append(live_by_base_id[base_id])
                    swapped += 1
                    continue
            result.append(doc)
        return result
