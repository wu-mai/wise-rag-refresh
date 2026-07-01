import pytest
from unittest.mock import MagicMock
from src.fresh_rag.generator import Generator
from src.fresh_rag.types import Document, Query, ScoredDoc


def make_doc(doc_id: str, text: str) -> Document:
    return Document(doc_id=doc_id, text=text, url=f"u://{doc_id}")


def make_scored(text: str, score: float) -> ScoredDoc:
    return ScoredDoc(
        doc=make_doc("d", text),
        retrieval_score=score,
        final_score=score,
    )


class TestGenerator:
    def test_returns_llm_response(self):
        client = MagicMock()
        client.chat.return_value = "Donald Trump"
        gen = Generator(llm_client=client)
        q = Query(id="1", text="Who is the president?")
        docs = [make_scored("The current president is Donald Trump.", 0.9)]
        answer = gen.generate(q, docs)
        assert answer == "Donald Trump"

    def test_prompt_contains_query_and_docs(self):
        client = MagicMock()
        client.chat.return_value = "answer"
        gen = Generator(llm_client=client)
        q = Query(id="1", text="UNIQUE_QUERY")
        docs = [
            make_scored("DOC_ONE_TEXT", 0.9),
            make_scored("DOC_TWO_TEXT", 0.5),
        ]
        gen.generate(q, docs)
        prompt = client.chat.call_args.kwargs.get("prompt", "")
        assert "UNIQUE_QUERY" in prompt
        assert "DOC_ONE_TEXT" in prompt
        assert "DOC_TWO_TEXT" in prompt

    def test_truncates_long_docs(self):
        client = MagicMock()
        client.chat.return_value = "answer"
        gen = Generator(llm_client=client, max_doc_chars=50)
        q = Query(id="1", text="test")
        long_text = "X" * 1000
        docs = [make_scored(long_text, 0.9)]
        gen.generate(q, docs)
        prompt = client.chat.call_args.kwargs.get("prompt", "")
        # The full long text should NOT be in the prompt
        assert long_text not in prompt
        # But a truncated version should be
        assert "X" * 50 in prompt

    def test_handles_empty_docs(self):
        client = MagicMock()
        client.chat.return_value = "I don't know"
        gen = Generator(llm_client=client)
        q = Query(id="1", text="test")
        answer = gen.generate(q, [])
        assert answer == "I don't know"

    def test_api_error_returns_empty_string(self):
        client = MagicMock()
        client.chat.side_effect = RuntimeError("API down")
        gen = Generator(llm_client=client)
        q = Query(id="1", text="test")
        answer = gen.generate(q, [make_scored("doc", 0.5)])
        assert answer == ""
