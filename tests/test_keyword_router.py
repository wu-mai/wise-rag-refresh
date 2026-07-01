"""Tests for KeywordRouter baseline."""
import pytest
from src.fresh_rag.keyword_router import KeywordRouter
from src.fresh_rag.types import Query


class TestKeywordRouter:
    def test_keyword_found_returns_1(self):
        router = KeywordRouter()
        q = Query(id="1", text="What is the latest price of Bitcoin?")
        assert router.route(q) == 1.0

    def test_no_keyword_returns_0(self):
        router = KeywordRouter()
        q = Query(id="1", text="What is the capital of France?")
        assert router.route(q) == 0.0

    def test_case_insensitive(self):
        router = KeywordRouter()
        q = Query(id="1", text="What happened Today in the news?")
        assert router.route(q) == 1.0

    def test_custom_keywords(self):
        router = KeywordRouter(keywords={"bitcoin", "crypto"})
        q = Query(id="1", text="bitcoin price today")
        assert router.route(q) == 1.0

    def test_empty_query(self):
        router = KeywordRouter()
        q = Query(id="1", text="")
        assert router.route(q) == 0.0
