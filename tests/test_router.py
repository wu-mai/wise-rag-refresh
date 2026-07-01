import pytest
from unittest.mock import MagicMock
from src.fresh_rag.router import Router
from src.fresh_rag.types import Query


def make_mock_client(response: str = "0.7"):
    client = MagicMock()
    client.chat.return_value = response
    return client


class TestRouter:
    def test_returns_float_in_unit_interval(self):
        client = make_mock_client("0.7")
        router = Router(llm_client=client)
        score = router.route(Query(id="1", text="Who is the president?"))
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_parses_decimal_response(self):
        client = make_mock_client("0.85")
        router = Router(llm_client=client)
        assert router.route(Query(id="1", text="test")) == 0.85

    def test_parses_response_with_explanation(self):
        # LLM may return "0.8 because..." — should still extract 0.8
        client = make_mock_client("0.8 — this query asks about current events.")
        router = Router(llm_client=client)
        assert router.route(Query(id="1", text="test")) == 0.8

    def test_parses_integer_response(self):
        client = make_mock_client("1")
        router = Router(llm_client=client)
        assert router.route(Query(id="1", text="test")) == 1.0

    def test_clamps_above_one(self):
        client = make_mock_client("1.5")
        router = Router(llm_client=client)
        assert router.route(Query(id="1", text="test")) == 1.0

    def test_clamps_below_zero(self):
        client = make_mock_client("-0.3")
        router = Router(llm_client=client)
        assert router.route(Query(id="1", text="test")) == 0.0

    def test_parse_failure_returns_0_5(self):
        client = make_mock_client("I don't know")
        router = Router(llm_client=client)
        assert router.route(Query(id="1", text="test")) == 0.5

    def test_api_error_returns_0_5(self):
        client = MagicMock()
        client.chat.side_effect = RuntimeError("API down")
        router = Router(llm_client=client)
        assert router.route(Query(id="1", text="test")) == 0.5

    def test_caches_results(self):
        client = make_mock_client("0.7")
        router = Router(llm_client=client)
        q = Query(id="1", text="same question")
        router.route(q)
        router.route(q)
        router.route(q)
        # LLM should only be called once
        assert client.chat.call_count == 1

    def test_prompt_contains_query(self):
        client = make_mock_client("0.5")
        router = Router(llm_client=client)
        router.route(Query(id="1", text="UNIQUE_QUERY_42"))
        # Check that the prompt sent to the LLM contains the query
        call_args = client.chat.call_args
        # The call is chat(prompt=prompt), so it's in kwargs
        prompt = call_args.kwargs.get("prompt", "")
        assert "UNIQUE_QUERY_42" in prompt
