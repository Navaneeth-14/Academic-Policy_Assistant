"""Unit tests for agents/rewriter_agent.py"""
import pytest
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import rewriter_agent


def test_rewriter_returns_required_keys():
    """Rewriter must return original_query, rewritten_query, was_rewritten."""
    with patch("agents.rewriter_agent.client") as mock_client:
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="What is the minimum attendance requirement?"))]
        )
        result = rewriter_agent.run("attendance?")
    for key in ["original_query", "rewritten_query", "was_rewritten"]:
        assert key in result


def test_rewriter_preserves_original_query():
    """original_query must always match the input."""
    with patch("agents.rewriter_agent.client") as mock_client:
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="What is the attendance policy?"))]
        )
        result = rewriter_agent.run("attendance?")
    assert result["original_query"] == "attendance?"


def test_rewriter_was_rewritten_true_when_changed():
    """was_rewritten must be True when query is changed."""
    with patch("agents.rewriter_agent.client") as mock_client:
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="What is the minimum attendance requirement?"))]
        )
        result = rewriter_agent.run("attendance?")
    assert result["was_rewritten"] is True


def test_rewriter_was_rewritten_false_when_same():
    """was_rewritten must be False when rewritten query matches original."""
    query = "What is the minimum attendance required?"
    with patch("agents.rewriter_agent.client") as mock_client:
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=query))]
        )
        result = rewriter_agent.run(query)
    assert result["was_rewritten"] is False


def test_rewriter_fallback_on_api_error():
    """Rewriter must return original query unchanged if API call fails."""
    with patch("agents.rewriter_agent.client") as mock_client:
        mock_client.chat.completions.create.side_effect = Exception("API error")
        result = rewriter_agent.run("what about attendance")
    assert result["rewritten_query"] == "what about attendance"
    assert result["was_rewritten"] is False


def test_rewriter_fallback_on_empty_response():
    """Rewriter must fall back to original if LLM returns empty string."""
    with patch("agents.rewriter_agent.client") as mock_client:
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="   "))]
        )
        result = rewriter_agent.run("what about attendance")
    assert result["rewritten_query"] == "what about attendance"
    assert result["was_rewritten"] is False
