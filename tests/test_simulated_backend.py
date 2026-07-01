import pytest
import json
import tempfile
from pathlib import Path
from src.fresh_rag.backends.simulated import SimulatedBackend


@pytest.fixture
def temp_corpus(tmp_path: Path, sample_simulated_data):
    f = tmp_path / "corpus.json"
    f.write_text(json.dumps(sample_simulated_data))
    return str(f)


class TestSimulatedBackend:
    def test_loads_queries(self, temp_corpus):
        backend = SimulatedBackend(temp_corpus)
        queries = backend.queries()
        assert len(queries) == 2
        assert queries[0].id == "1"
        assert queries[0].text == "Who is the current president?"
        assert queries[0].freshness_sensitive is True
        assert queries[1].freshness_sensitive is False

    def test_get_stale_documents(self, temp_corpus):
        backend = SimulatedBackend(temp_corpus)
        docs = backend.get_documents("1", version="stale")
        assert len(docs) == 1
        assert "Joe Biden" in docs[0].text
        assert docs[0].is_live is False
        assert docs[0].is_changed is False

    def test_get_fresh_documents(self, temp_corpus):
        backend = SimulatedBackend(temp_corpus)
        docs = backend.get_documents("1", version="fresh")
        assert len(docs) == 1
        assert "Donald Trump" in docs[0].text
        assert docs[0].is_live is True
        assert docs[0].is_changed is True

    def test_unknown_query_id_returns_empty(self, temp_corpus):
        backend = SimulatedBackend(temp_corpus)
        docs = backend.get_documents("nonexistent", version="stale")
        assert docs == []

    def test_invalid_version_raises(self, temp_corpus):
        backend = SimulatedBackend(temp_corpus)
        with pytest.raises(ValueError, match="version must be"):
            backend.get_documents("1", version="bogus")
