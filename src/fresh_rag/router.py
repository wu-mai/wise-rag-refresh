"""LLM freshness router.

Estimates p_fresh(query) — the probability that the query requires
up-to-date information to answer correctly. Uses a fixed prompt and
a single LLM call per unique query.
"""
import re
from typing import Callable, Optional
from .types import Query


_ROUTER_PROMPT = """On a scale of 0 to 1, how much does this query require up-to-date information to answer correctly? Consider time-sensitive keywords, topics that change frequently, and implicit recency requirements.

Query: {query}

Return only a single number between 0 and 1. No explanation."""


class Router:
    def __init__(self, llm_client: Optional[Callable] = None):
        # llm_client must have a .chat(prompt: str) -> str method.
        # If None, uses a dummy that always returns 0.5 (for tests/offline).
        self.llm_client = llm_client or _DummyClient()
        self._cache: dict = {}

    def route(self, query: Query) -> float:
        if query.text in self._cache:
            return self._cache[query.text]
        try:
            prompt = _ROUTER_PROMPT.format(query=query.text)
            response = self.llm_client.chat(prompt=prompt)
            score = self._parse_score(response)
        except Exception:
            score = 0.5
        self._cache[query.text] = score
        return score

    @staticmethod
    def _parse_score(response: str) -> float:
        """Extract first float in [0,1] from response, fallback to 0.5."""
        match = re.search(r"-?\d+\.?\d*", response)
        if not match:
            return 0.5
        try:
            value = float(match.group())
        except ValueError:
            return 0.5
        # Clamp
        return max(0.0, min(1.0, value))


class _DummyClient:
    """Used when no real LLM client is provided. Returns 0.5 for all queries."""
    def chat(self, prompt: str) -> str:
        return "0.5"
