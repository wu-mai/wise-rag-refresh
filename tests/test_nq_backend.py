import json
import pytest
from pathlib import Path
from src.fresh_rag.backends.nq import NQBackend


@pytest.fixture
def temp_nq_corpus(tmp_path: Path):
    data = [
        {
            "id": 0,
            "question": "when was the first iphone released",
            "document_url": "https://en.wikipedia.org/wiki/IPhone",
            "document_title": "iPhone",
            "short_answers": ["June 29, 2007"],
        },
        {
            "id": 1,
            "question": "what is the capital of france",
            "document_url": "https://en.wikipedia.org/wiki/Paris",
            "document_title": "Paris",
            "short_answers": ["Paris"],
        },
    ]
    f = tmp_path / "nq.json"
    f.write_text(json.dumps(data))
    return str(f)


class TestNQBackend:
    def test_loads_queries(self, temp_nq_corpus):
        backend = NQBackend(temp_nq_corpus)
        queries = backend.queries()
        assert len(queries) == 2
        assert queries[0].text == "when was the first iphone released"
        # NQ questions are factoid; we conservatively mark all as sensitive
        # for the staleness study (no ground truth label in NQ)
        assert queries[0].freshness_sensitive is True

    def test_get_stale_documents_returns_synthetic_stale(self, temp_nq_corpus):
        backend = NQBackend(temp_nq_corpus)
        docs = backend.get_documents("0", version="stale")
        assert len(docs) == 1
        assert docs[0].is_live is False
        assert docs[0].is_changed is False
        # Should contain some hint of being old
        assert docs[0].age_days > 0

    def test_get_fresh_documents_returns_synthetic_fresh(self, temp_nq_corpus):
        backend = NQBackend(temp_nq_corpus)
        docs = backend.get_documents("0", version="fresh")
        assert len(docs) == 1
        assert docs[0].is_live is True
        assert docs[0].is_changed is True
        assert docs[0].age_days == 0

    def test_get_correct_answer(self, temp_nq_corpus):
        backend = NQBackend(temp_nq_corpus)
        assert backend.get_correct_answer("0") == "June 29, 2007"
        assert backend.get_correct_answer("1") == "Paris"
