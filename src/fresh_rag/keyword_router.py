"""Keyword-based freshness router (baseline)."""
from typing import Set

_KEYWORDS = {
    "latest", "current", "2025", "2026", "recent", "price", "news",
    "today", "now", "new", "updated", "release", "version",
}


class KeywordRouter:
    """Routes based on keyword overlap — cheaper than LLM, less accurate."""

    def __init__(self, keywords: Set[str] | None = None):
        self.keywords = keywords or _KEYWORDS

    def route(self, query) -> float:
        """Returns 1.0 if any keyword found, else 0.0."""
        query_lower = query.text.lower()
        for keyword in self.keywords:
            if keyword in query_lower:
                return 1.0
        return 0.0
