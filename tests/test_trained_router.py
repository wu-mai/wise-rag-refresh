import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from src.fresh_rag.trained_router import _MockModel, FACT_TYPE_TO_PFRESH, TrainedRouter
from src.fresh_rag.types import Query


def test_mock_model_predicts_fact_types():
    model = _MockModel()
    out = model.predict_proba("What is the current age of the oldest singles champion?")
    p = FACT_TYPE_TO_PFRESH["fast-changing"]
    assert abs(out - p) < 0.5


def test_trained_router_returns_float_in_range(monkeypatch):
    monkeypatch.setattr("src.fresh_rag.trained_router._load_model", lambda path: _MockModel())
    router = TrainedRouter(model_path="models/router.pt")
    score = router.route(Query(id="0", text="When was the iPhone first released?"))
    assert 0.0 <= score <= 1.0


def test_trained_router_caches(monkeypatch):
    monkeypatch.setattr("src.fresh_rag.trained_router._load_model", lambda path: _MockModel())
    router = TrainedRouter(model_path="models/router.pt")
    q = Query(id="0", text="some query")
    s1 = router.route(q)
    s2 = router.route(q)
    assert s1 == s2


def test_trained_router_falls_back_when_no_checkpoint(monkeypatch, tmp_path):
    def fail(path): raise FileNotFoundError(path)
    monkeypatch.setattr("src.fresh_rag.trained_router._load_model", fail)
    router = TrainedRouter(model_path=str(tmp_path / "missing.pt"))
    score = router.route(Query(id="0", text="anything"))
    # Mock model returns 0.0 for unrecognized queries (never-changing). Just
    # check it's a valid probability in [0, 1] — the exact value isn't what
    # we're testing; we're testing the fallback path doesn't crash.
    assert 0.0 <= score <= 1.0
