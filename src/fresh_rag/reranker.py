"""Deterministic reranking using freshness and change signals."""
from typing import Callable, List, Tuple
from .types import ScoredDoc


class Reranker:
    """Reranks documents using s_final = s_retrieval + λ₁·fresh + λ₂·changed."""

    def rerank(
        self,
        docs: List[ScoredDoc],
        lambda_fresh: float,
        lambda_changed: float,
    ) -> List[ScoredDoc]:
        """Returns docs sorted by final_score descending. Stable on ties."""
        for sd in docs:
            fresh = 1.0 if sd.doc.is_live else 0.0
            changed = 1.0 if sd.doc.is_changed else 0.0
            sd.final_score = (
                sd.retrieval_score + lambda_fresh * fresh + lambda_changed * changed
            )
        # sorted is stable in Python: ties preserve input order
        return sorted(docs, key=lambda sd: sd.final_score, reverse=True)

    def tune_weights(
        self,
        examples: List[Tuple[List[ScoredDoc], str]],
        metric_fn: Callable[[List[ScoredDoc], str], float],
        candidates: List[float] = None,
    ) -> Tuple[float, float]:
        """Grid-search (λ_fresh, λ_changed) to maximize metric on examples.

        examples: list of (docs, gold) pairs. The caller supplies metric_fn
        (e.g., lambda docs, gold: f1_score(docs[0].doc.text, gold) if a live
        doc is on top, else 0).
        candidates: weight grid (default [0.0, 0.5, 1.0, 2.0]).

        Returns (best_lambda_fresh, best_lambda_changed).
        """
        if candidates is None:
            candidates = [0.0, 0.5, 1.0, 2.0]
        best = (1.0, 1.0)
        best_score = float("-inf")
        for lf in candidates:
            for lc in candidates:
                score = 0.0
                for docs, gold in examples:
                    reranked = self.rerank([ScoredDoc(sd.doc,
                                                       sd.retrieval_score,
                                                       0.0)
                                            for sd in docs], lf, lc)
                    score += metric_fn(reranked, gold)
                if score > best_score:
                    best_score = score
                    best = (lf, lc)
        return best

