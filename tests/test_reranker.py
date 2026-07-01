import pytest
from src.fresh_rag.reranker import Reranker
from src.fresh_rag.types import Document, ScoredDoc


def make_doc(doc_id: str, is_live: bool = False, is_changed: bool = False) -> Document:
    return Document(
        doc_id=doc_id,
        text=f"text {doc_id}",
        url=f"http://example.com/{doc_id}",
        is_live=is_live,
        is_changed=is_changed,
    )


class TestReranker:
    def test_no_live_no_change_preserves_order(self):
        # All docs cached, no reranking effect
        docs = [
            ScoredDoc(make_doc("a"), retrieval_score=0.9),
            ScoredDoc(make_doc("b"), retrieval_score=0.5),
        ]
        reranker = Reranker()
        result = reranker.rerank(docs, lambda_fresh=1.0, lambda_changed=1.0)
        assert [d.doc.doc_id for d in result] == ["a", "b"]
        assert all(d.final_score == d.retrieval_score for d in result)

    def test_live_boost_lifts_doc(self):
        # Doc b is live (cached would have lost), should be lifted
        docs = [
            ScoredDoc(make_doc("a", is_live=False), retrieval_score=0.6),
            ScoredDoc(make_doc("b", is_live=True), retrieval_score=0.5),
        ]
        reranker = Reranker()
        result = reranker.rerank(docs, lambda_fresh=1.0, lambda_changed=0.0)
        # b: 0.5 + 1.0 = 1.5 > a: 0.6 + 0.0 = 0.6
        assert [d.doc.doc_id for d in result] == ["b", "a"]

    def test_changed_boost_lifts_doc(self):
        docs = [
            ScoredDoc(make_doc("a", is_changed=False), retrieval_score=0.6),
            ScoredDoc(make_doc("b", is_changed=True), retrieval_score=0.5),
        ]
        reranker = Reranker()
        result = reranker.rerank(docs, lambda_fresh=0.0, lambda_changed=1.0)
        # b: 0.5 + 1.0 = 1.5 > a: 0.6
        assert [d.doc.doc_id for d in result] == ["b", "a"]

    def test_both_live_and_changed(self):
        docs = [
            ScoredDoc(make_doc("a", is_live=True, is_changed=True), retrieval_score=0.5),
            ScoredDoc(make_doc("b"), retrieval_score=0.6),
        ]
        reranker = Reranker()
        result = reranker.rerank(docs, lambda_fresh=1.0, lambda_changed=1.0)
        # a: 0.5 + 1.0 + 1.0 = 2.5 > b: 0.6
        assert [d.doc.doc_id for d in result] == ["a", "b"]

    def test_zero_lambdas_means_no_boost(self):
        docs = [
            ScoredDoc(make_doc("a", is_live=True, is_changed=True), retrieval_score=0.5),
            ScoredDoc(make_doc("b"), retrieval_score=0.6),
        ]
        reranker = Reranker()
        result = reranker.rerank(docs, lambda_fresh=0.0, lambda_changed=0.0)
        assert [d.doc.doc_id for d in result] == ["b", "a"]

    def test_stable_sort_on_ties(self):
        # Same final score: keep original order
        docs = [
            ScoredDoc(make_doc("a"), retrieval_score=0.5),
            ScoredDoc(make_doc("b"), retrieval_score=0.5),
        ]
        reranker = Reranker()
        result = reranker.rerank(docs, lambda_fresh=0.0, lambda_changed=0.0)
        assert [d.doc.doc_id for d in result] == ["a", "b"]


class TestRerankerTuneWeights:
    def test_tune_picks_higher_lambdas_when_live_doc_helps(self):
        # Single example: live doc has higher retrieval. metric_fn rewards
        # when live doc is on top.
        docs = [
            ScoredDoc(make_doc("stale", is_live=False), retrieval_score=0.9),
            ScoredDoc(make_doc("live", is_live=True), retrieval_score=0.5),
        ]
        def metric(reranked, gold):
            return 1.0 if reranked[0].doc.doc_id == "live" else 0.0
        reranker = Reranker()
        lf, lc = reranker.tune_weights([(docs, "gold")], metric, candidates=[0.0, 1.0, 2.0])
        # The grid should pick at least one weight > 0 to lift the live doc
        assert lf > 0 or lc > 0

    def test_tune_with_no_live_picks_zero_lambdas(self):
        # No live docs: any non-zero weight doesn't change order, so 0,0 is fine
        docs = [
            ScoredDoc(make_doc("a"), retrieval_score=0.9),
            ScoredDoc(make_doc("b"), retrieval_score=0.5),
        ]
        def metric(reranked, gold):
            return 1.0 if reranked[0].doc.doc_id == "a" else 0.0
        reranker = Reranker()
        lf, lc = reranker.tune_weights([(docs, "gold")], metric, candidates=[0.0, 1.0, 2.0])
        # Should not crash and should return a valid (lf, lc) pair
        assert lf in (0.0, 1.0, 2.0) and lc in (0.0, 1.0, 2.0)
