import pytest
from unittest.mock import MagicMock
from src.fresh_rag.retrieval import Retriever
from src.fresh_rag.types import Document, Query


def make_mock_backend():
    backend = MagicMock()
    # Query 1 about presidents has matching docs
    doc_a = Document(doc_id="a", text="The president lives in the White House.", url="u1")
    doc_b = Document(doc_id="b", text="Paris is the capital of France.", url="u2")
    doc_c = Document(doc_id="c", text="The current president is Donald Trump.", url="u3")
    backend.get_documents.return_value = [doc_a, doc_b, doc_c]
    return backend, [doc_a, doc_b, doc_c]


class TestRetriever:
    def test_returns_top_k_documents(self):
        backend, all_docs = make_mock_backend()
        retriever = Retriever(backend)
        query = Query(id="1", text="Who is the president?")
        result = retriever.retrieve(query, k=2)
        assert len(result) == 2
        assert all(isinstance(d, Document) for d in result)

    def test_relevant_doc_ranked_higher(self):
        # Doc c ("current president is Donald Trump") is most relevant
        # Doc a ("The president lives in the White House") is somewhat relevant
        # Doc b (Paris) is irrelevant
        backend, _ = make_mock_backend()
        retriever = Retriever(backend)
        query = Query(id="1", text="Who is the current president?")
        result = retriever.retrieve(query, k=3)
        doc_ids = [d.doc_id for d in result]
        assert doc_ids[0] == "c"  # most relevant first
        assert doc_ids[-1] == "b"  # irrelevant last

    def test_k_larger_than_corpus_returns_all(self):
        backend, all_docs = make_mock_backend()
        retriever = Retriever(backend)
        query = Query(id="1", text="test")
        result = retriever.retrieve(query, k=100)
        assert len(result) == 3

    def test_empty_corpus_returns_empty_list(self):
        backend = MagicMock()
        backend.get_documents.return_value = []
        retriever = Retriever(backend)
        query = Query(id="1", text="test")
        result = retriever.retrieve(query, k=5)
        assert result == []

    def test_uses_cached_version_by_default(self):
        backend, _ = make_mock_backend()
        retriever = Retriever(backend)
        query = Query(id="1", text="test")
        retriever.retrieve(query, k=5)
        # First call should be for "stale" version
        backend.get_documents.assert_called_with("1", version="stale")
