import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.fresh_rag.backends.freshqa import FreshQABackend, FACT_TYPE_TO_PFRESH, DATA_DIR

DEV_PATH = DATA_DIR / "freshqa_dev.parquet"


def test_fact_type_mapping():
    assert FACT_TYPE_TO_PFRESH["never-changing"] == 0.0
    assert FACT_TYPE_TO_PFRESH["slow-changing"] == 0.5
    assert FACT_TYPE_TO_PFRESH["fast-changing"] == 1.0


def test_loads_dev_queries():
    backend = FreshQABackend(DEV_PATH, role="qa_eval")
    queries = backend.queries()
    assert len(queries) == 100
    q = queries[0]
    assert isinstance(q.id, str)
    assert q.text  # non-empty


def test_get_documents_stale_and_fresh():
    backend = FreshQABackend(DEV_PATH, role="qa_eval")
    q = backend.queries()[0]
    stale = backend.get_documents(q.id, "stale")
    fresh = backend.get_documents(q.id, "fresh")
    assert len(stale) == 1 and len(fresh) == 1
    assert stale[0].is_live is False
    assert fresh[0].is_live is True
    assert fresh[0].is_changed is True


def test_get_correct_answer():
    backend = FreshQABackend(DEV_PATH, role="qa_eval")
    q = backend.queries()[0]
    ans = backend.get_correct_answer(q.id)
    assert isinstance(ans, str)


def test_get_fact_type_and_pfresh():
    backend = FreshQABackend(DEV_PATH, role="qa_eval")
    q = backend.queries()[0]
    ft = backend.get_fact_type(q.id)
    assert ft in ("never-changing", "slow-changing", "fast-changing")
    assert backend.get_pfresh_label(q.id) == FACT_TYPE_TO_PFRESH[ft]


def test_router_split_sizes():
    backend = FreshQABackend(str(DATA_DIR / "freshqa_test.parquet"), role="router_train")
    assert len(backend.queries()) == 350
    backend2 = FreshQABackend(str(DATA_DIR / "freshqa_test.parquet"), role="router_heldout")
    assert len(backend2.queries()) == 150
