"""LLM answer generator."""
from typing import Callable, List, Optional
from .types import Query, ScoredDoc


_GENERATOR_PROMPT = """Based on the following context, answer the question briefly and directly. If the context doesn't contain the answer, say so.

Question: {query}

Context:
{context}

Answer:"""


class Generator:
    def __init__(self, llm_client: Optional[Callable] = None, max_doc_chars: int = 800):
        self.llm_client = llm_client or _DummyClient()
        self.max_doc_chars = max_doc_chars

    def generate(self, query: Query, docs: List[ScoredDoc]) -> str:
        try:
            context = self._format_context(docs)
            prompt = _GENERATOR_PROMPT.format(query=query.text, context=context)
            return self.llm_client.chat(prompt=prompt).strip()
        except Exception:
            return ""

    def _format_context(self, docs: List[ScoredDoc]) -> str:
        if not docs:
            return "(no context)"
        lines = []
        for i, sd in enumerate(docs, 1):
            text = sd.doc.text
            if len(text) > self.max_doc_chars:
                text = text[: self.max_doc_chars] + "..."
            lines.append(f"[{i}] {text}")
        return "\n".join(lines)


class _DummyClient:
    def chat(self, prompt: str) -> str:
        return "dummy answer"
