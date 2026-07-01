"""Natural Questions backend with synthetic stale/fresh split.

The NQ dataset itself does not contain versioned documents. We synthesize
a "stale" version (older, with a hint about being old) and a "fresh"
version (current answer) for each query. The synthetic split is good
enough for Block 1's MVP because the goal is to demonstrate the
freshness signal, not to benchmark against real temporal drift.
"""
import json
from datetime import datetime, timedelta
from typing import List
from ..types import Document, Query


class NQBackend:
    def __init__(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._entries = {str(e["id"]): e for e in data}

    def queries(self) -> List[Query]:
        # NQ has no freshness_sensitive label. We conservatively mark
        # all as True for the staleness study; ground-truth is the
        # SimulatedBackend (where the label is human-verified).
        return [
            Query(
                id=str(e["id"]),
                text=e["question"],
                freshness_sensitive=True,
            )
            for e in self._entries.values()
        ]

    def get_documents(self, query_id: str, version: str) -> List[Document]:
        if version not in ("stale", "fresh"):
            raise ValueError(f"version must be 'stale' or 'fresh', got {version!r}")
        entry = self._entries.get(query_id)
        if entry is None:
            return []
        answer = self._first_real_answer(entry)
        url = entry.get("document_url", "")
        title = entry.get("document_title", "")

        if version == "fresh":
            text = (
                f"{title}. According to the most current sources, the answer is: "
                f"{answer}. This information was verified recently."
            )
            return [
                Document(
                    doc_id=f"{query_id}-fresh",
                    text=text,
                    url=url,
                    timestamp=datetime.now().isoformat(),
                    age_days=0,
                    is_live=True,
                    is_changed=True,
                )
            ]
        else:
            # Stale: same answer, but wrapped in a sentence that hints at age
            old_date = (datetime.now() - timedelta(days=900)).isoformat()
            text = (
                f"{title}. As of an older snapshot, the answer is: {answer}. "
                f"This information may be outdated."
            )
            return [
                Document(
                    doc_id=f"{query_id}-stale",
                    text=text,
                    url=url,
                    timestamp=old_date,
                    age_days=900,
                    is_live=False,
                    is_changed=False,
                )
            ]

    def get_correct_answer(self, query_id: str) -> str:
        entry = self._entries.get(query_id)
        if entry is None:
            return ""
        return self._first_real_answer(entry)

    def _first_real_answer(self, entry) -> str:
        """Return the first non-empty short answer.

        NQ data stores short_answers as a list of strings, some of which
        are the literal '[]' (meaning no answer from that annotator).
        Skip those and return the first real answer, or '' if none.
        """
        for ans in entry.get("short_answers", []):
            if isinstance(ans, str) and ans.strip() and ans.strip() != "[]":
                # Some answers are stored as Python list-string repr like
                # "['Wilhelm Röntgen']" — strip the wrapping.
                a = ans.strip()
                if a.startswith("[") and a.endswith("]"):
                    try:
                        parsed = eval(a)  # safe: data is local/trusted
                        if isinstance(parsed, list) and parsed:
                            a = str(parsed[0]).strip()
                    except Exception:
                        pass
                if a:
                    return a
        return ""
