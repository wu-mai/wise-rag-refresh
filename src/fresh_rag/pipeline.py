"""End-to-end Freshness-Aware RAG pipeline.

Wires together Router -> Retriever -> Refresher -> Reranker -> Generator.
Both router and retriever are pluggable.
"""
from typing import Callable, List, Optional, Tuple

from .backends.base import Backend
from .generator import Generator
from .refresh import Refresher
from .reranker import Reranker
from .retrieval import Retriever
from .router import Router
from .types import Query, ScoredDoc


class FreshnessAwareRAG:
    def __init__(
        self,
        backend: Backend,
        llm_client: Optional[Callable] = None,
        router=None,
        threshold: float = 0.5,
        m: int = 3,
        lambda_fresh: float = 1.0,
        lambda_changed: float = 1.0,
        k: int = 5,
        retriever=None,
    ):
        self.backend = backend
        self.threshold = threshold
        self.m = m
        self.lambda_fresh = lambda_fresh
        self.lambda_changed = lambda_changed
        self.k = k
        # Router (pluggable: TrainedRouter, KeywordRouter, LLM Router, ...)
        if router is not None:
            self.router = router
        else:
            self.router = Router(llm_client=llm_client)
        # Retriever (pluggable: keyword, dense, mock)
        if retriever is not None:
            self.retriever = retriever
        else:
            self.retriever = Retriever(backend)
        self.refresher = Refresher(backend)
        self.reranker = Reranker()
        self.generator = Generator(llm_client=llm_client)

    def answer(self, query: Query) -> Tuple[str, List[ScoredDoc]]:
        """Run the full pipeline. Returns (answer_text, ranked_docs)."""
        p_fresh = self.router.route(query)
        docs = self.retriever.retrieve(query, k=self.k)
        docs = self.refresher.maybe_refresh(
            query, docs, p_fresh, self.threshold, self.m
        )
        # Position-based retrieval scores for keyword retriever
        scored = [
            ScoredDoc(
                doc=d,
                retrieval_score=1.0 - i * 0.1,
                final_score=0.0,
            )
            for i, d in enumerate(docs)
        ]
        scored = self.reranker.rerank(
            scored, self.lambda_fresh, self.lambda_changed
        )
        answer = self.generator.generate(query, scored[:3])
        return answer, scored
