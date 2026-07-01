import pytest
from unittest.mock import MagicMock
from src.fresh_rag.refresh import Refresher
from src.fresh_rag.types import Document, Query


def make_backend_with_live():
    backend = MagicMock()
    cached = Document(doc_id="a", text="old content", is_live=False, is_changed=False)
    live = Document(doc_id="a", text="new content", is_live=True, is_changed=True)
    backend.get_documents.return_value = [live]
    return backend, cached, live


class TestRefresher:
    def test_below_threshold_returns_unchanged(self):
        backend, cached, live = make_backend_with_live()
        refresher = Refresher(backend)
        query = Query(id="1", text="test")
        docs = [cached]
        result = refresher.maybe_refresh(query, docs, p_fresh=0.3, threshold=0.5, m=3)
        # p_fresh < threshold: no fetch, no live
        assert result[0].text == "old content"
        assert result[0].is_live is False
        assert result[0].is_changed is False
        backend.get_documents.assert_not_called()

    def test_above_threshold_swaps_top_m(self):
        backend, cached, live = make_backend_with_live()
        refresher = Refresher(backend)
        query = Query(id="1", text="test")
        docs = [cached, cached, cached, cached, cached]  # 5 docs
        result = refresher.maybe_refresh(query, docs, p_fresh=0.8, threshold=0.5, m=2)
        # First 2 should be live (swapped), rest still cached
        assert result[0].text == "new content"
        assert result[0].is_live is True
        assert result[0].is_changed is True
        assert result[1].text == "new content"
        assert result[2].text == "old content"
        assert result[2].is_live is False

    def test_above_threshold_with_m_larger_than_docs(self):
        backend, cached, live = make_backend_with_live()
        refresher = Refresher(backend)
        query = Query(id="1", text="test")
        docs = [cached, cached]
        result = refresher.maybe_refresh(query, docs, p_fresh=0.9, threshold=0.5, m=5)
        # All docs should be swapped
        assert all(d.text == "new content" for d in result)
        assert all(d.is_live for d in result)

    def test_threshold_boundary_inclusive(self):
        # p_fresh == threshold should NOT trigger refresh (must be strictly greater)
        backend, cached, live = make_backend_with_live()
        refresher = Refresher(backend)
        query = Query(id="1", text="test")
        docs = [cached]
        result = refresher.maybe_refresh(query, docs, p_fresh=0.5, threshold=0.5, m=3)
        assert result[0].is_live is False
        backend.get_documents.assert_not_called()

    def test_empty_docs_returns_empty(self):
        backend = MagicMock()
        refresher = Refresher(backend)
        query = Query(id="1", text="test")
        result = refresher.maybe_refresh(query, [], p_fresh=0.9, threshold=0.5, m=3)
        assert result == []
        backend.get_documents.assert_not_called()

    def test_backend_failure_falls_back_to_cached(self):
        backend = MagicMock()
        backend.get_documents.side_effect = RuntimeError("network down")
        refresher = Refresher(backend)
        query = Query(id="1", text="test")
        cached = Document(doc_id="a", text="old", is_live=False, is_changed=False)
        result = refresher.maybe_refresh(query, [cached], p_fresh=0.9, threshold=0.5, m=1)
        # Should keep cached doc, mark as is_live=False but with a warning flag
        assert result[0].text == "old"
        assert result[0].is_live is False
