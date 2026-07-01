"""Core data types for the Freshness-Aware RAG pipeline."""
from dataclasses import dataclass, field
from typing import List


@dataclass
class Query:
    """A user query.

    `freshness_sensitive` is ground truth from manual annotation,
    used ONLY for evaluation. Never feed it to the router.
    """
    id: str
    text: str
    freshness_sensitive: bool = False


@dataclass
class Document:
    """A retrieved or fetched document."""
    doc_id: str
    text: str
    url: str = ""
    timestamp: str = ""
    age_days: int = 0
    is_live: bool = False
    is_changed: bool = False


@dataclass
class ScoredDoc:
    """A document paired with retrieval and final scores."""
    doc: Document
    retrieval_score: float
    final_score: float = 0.0
