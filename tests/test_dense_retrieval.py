import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from src.fresh_rag.dense_retrieval import DenseRetriever, _MockEncoder
from src.fresh_rag.types import Document, Query


def test_mock_encoder_returns_unit_norm():
    enc = _MockEncoder(dim=16)
    v = enc.encode(["hello world", "another doc"])
    norms = np.linalg.norm(v, axis=1)
    assert all(abs(n - 1.0) < 1e-3 for n in norms)


def test_dense_retriever_topk_with_mock(monkeypatch):
    monkeypatch.setattr("src.fresh_rag.dense_retrieval._load_encoder", lambda name: _MockEncoder(dim=16))
    docs = [
        Document(doc_id="a", text="the quick brown fox"),
        Document(doc_id="b", text="completely unrelated content"),
        Document(doc_id="c", text="the lazy dog"),
    ]
    retr = DenseRetriever(docs=docs, model_name="mock")
    out = retr.retrieve(Query(id="q", text="quick fox"), k=2)
    assert len(out) == 2


def test_dense_retriever_falls_back_to_empty_when_no_docs(monkeypatch):
    monkeypatch.setattr("src.fresh_rag.dense_retrieval._load_encoder", lambda name: _MockEncoder())
    retr = DenseRetriever(docs=[], model_name="mock")
    assert retr.retrieve(Query(id="q", text="anything"), k=3) == []


def test_dense_retriever_falls_back_when_encoder_missing(monkeypatch):
    def fail(name): raise RuntimeError("no model")
    monkeypatch.setattr("src.fresh_rag.dense_retrieval._load_encoder", fail)
    retr = DenseRetriever(docs=[Document(doc_id="a", text="x")], model_name="missing")
    out = retr.retrieve(Query(id="q", text="x"), k=1)
    assert out == []
