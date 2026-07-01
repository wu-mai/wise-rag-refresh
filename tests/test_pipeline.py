import pytest
from unittest.mock import MagicMock
from src.fresh_rag.pipeline import FreshnessAwareRAG
from src.fresh_rag.types import Document, Query


def make_mock_backend(include_live: bool = True):
    backend = MagicMock()
    stale_doc = Document(doc_id="a-stale", text="old content", is_live=False, is_changed=False)
    live_doc = Document(doc_id="a-fresh", text="new content", is_live=True, is_changed=True)

    def get_docs(qid, version):
        if version == "fresh" and include_live:
            return [live_doc]
        return [stale_doc]

    backend.get_documents.side_effect = get_docs
    return backend


def make_mock_llm(router_response: str = "0.9", gen_response: str = "new answer"):
    client = MagicMock()
    responses = [router_response, gen_response]

    def chat(prompt):
        return responses.pop(0) if responses else gen_response

    client.chat.side_effect = chat
    return client


class TestFreshnessAwareRAG:
    def test_low_p_fresh_skips_refresh(self):
        # Router says query is not fresh; should not fetch live
        backend = make_mock_backend()
        llm = make_mock_llm(router_response="0.2", gen_response="old answer")
        rag = FreshnessAwareRAG(
            backend=backend, llm_client=llm,
            threshold=0.5, m=3, lambda_fresh=1.0, lambda_changed=1.0,
        )
        q = Query(id="1", text="test")
        answer, scored = rag.answer(q)
        # Should NOT have called get_documents with "fresh"
        backend.get_documents.assert_called_with("1", version="stale")
        assert answer == "old answer"

    def test_high_p_fresh_triggers_refresh(self):
        backend = make_mock_backend()
        llm = make_mock_llm(router_response="0.9", gen_response="new answer")
        rag = FreshnessAwareRAG(
            backend=backend, llm_client=llm,
            threshold=0.5, m=3, lambda_fresh=1.0, lambda_changed=1.0,
        )
        q = Query(id="1", text="test")
        answer, scored = rag.answer(q)
        # Should have called with both "stale" (retrieval) and "fresh" (refresh)
        calls = [c.kwargs.get("version") or (c.args[1] if len(c.args) > 1 else None) for c in backend.get_documents.call_args_list]
        assert "stale" in calls
        assert "fresh" in calls
        assert answer == "new answer"

    def test_reranker_orders_live_first(self):
        backend = make_mock_backend()
        llm = make_mock_llm(router_response="0.9", gen_response="answer")
        rag = FreshnessAwareRAG(
            backend=backend, llm_client=llm,
            threshold=0.5, m=3, lambda_fresh=1.0, lambda_changed=1.0,
        )
        q = Query(id="1", text="test")
        _, scored = rag.answer(q)
        # Live doc should be ranked first after rerank
        assert scored[0].doc.is_live is True

    def test_threshold_is_strict_greater_than(self):
        # p_fresh == threshold: no refresh
        backend = make_mock_backend()
        llm = make_mock_llm(router_response="0.5", gen_response="old answer")
        rag = FreshnessAwareRAG(
            backend=backend, llm_client=llm,
            threshold=0.5, m=3, lambda_fresh=1.0, lambda_changed=1.0,
        )
        q = Query(id="1", text="test")
        rag.answer(q)
        backend.get_documents.assert_called_with("1", version="stale")

    def test_pipeline_accepts_custom_retriever(self):
        """Pipeline should accept a pluggable retriever via the new parameter."""
        backend = make_mock_backend()
        llm = make_mock_llm(router_response="0.0", gen_response="answer")

        # Custom retriever: just return the stale doc directly
        class FakeRetriever:
            def __init__(self, backend):
                self.backend = backend
                self.called = False

            def retrieve(self, query, k=5):
                self.called = True
                return self.backend.get_documents(query.id, "stale")[:k]

        retr = FakeRetriever(backend)
        rag = FreshnessAwareRAG(
            backend=backend, llm_client=llm, retriever=retr,
            threshold=0.5, m=3,
        )
        assert isinstance(rag.retriever, FakeRetriever)
        q = Query(id="1", text="test")
        answer, scored = rag.answer(q)
        assert retr.called
        assert answer == "answer"
