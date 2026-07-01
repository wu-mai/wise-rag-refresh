"""Protocol for pluggable document backends."""
from typing import List, Protocol
from ..types import Document, Query


class Backend(Protocol):
    """A document corpus that can answer queries.

    Implementations must be able to:
    - Enumerate all queries (with freshness_sensitive labels for evaluation)
    - Return documents for a query, in either "stale" or "fresh" version
    """

    def queries(self) -> List[Query]:
        """Return all queries in the corpus (with ground-truth labels)."""
        ...

    def get_documents(self, query_id: str, version: str) -> List[Document]:
        """Return documents for the query.

        version: "stale" returns cached old content, "fresh" returns live content.
        The fresh version should have is_live=True, is_changed=True.
        """
        ...
