import pytest
from unittest.mock import MagicMock


@pytest.fixture
def mock_llm_client():
    """Returns a mock Codex MCP client that returns canned responses."""
    client = MagicMock()
    client.chat.return_value = "0.7"
    return client


@pytest.fixture
def sample_simulated_data():
    """Minimal test data mirroring simulated_test_cases.json structure."""
    return [
        {
            "id": "1",
            "question": "Who is the current president?",
            "fresh_context": "Donald Trump is the 47th president as of January 2025.",
            "stale_context": "Joe Biden is the 46th president as of January 2021.",
            "correct_answer": "Donald Trump",
            "freshness_sensitive": True,
        },
        {
            "id": "2",
            "question": "What is the capital of France?",
            "fresh_context": "Paris is the capital of France.",
            "stale_context": "Paris is the capital of France.",
            "correct_answer": "Paris",
            "freshness_sensitive": False,
        },
    ]
