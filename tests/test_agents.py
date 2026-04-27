"""
Unit tests for the multi-agent system.
- test_retrieval_agent: tests Agent 1 output structure and fallback behaviour
- test_validation_agent: tests Agent 2 verdict logic with mocked LLM
- test_agent_runner: tests sequential handoff between agents
"""
import pytest
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag import load_chunks, build_index
from agents import retrieval_agent, validation_agent, agent_runner


@pytest.fixture(scope="module")
def rag_setup():
    chunks = load_chunks("data.txt")
    index, _ = build_index(chunks)
    return chunks, index


# ── Retrieval Agent tests ─────────────────────────────────────────────────────

def test_retrieval_agent_returns_required_keys(rag_setup):
    """Retrieval agent output must contain all required keys."""
    chunks, index = rag_setup
    result = retrieval_agent.run("What is the attendance requirement?", chunks, index)
    for key in ["action", "tool_used", "description", "response", "context", "top_score", "results"]:
        assert key in result, f"Missing key: {key}"


def test_retrieval_agent_fallback_on_unrelated_query(rag_setup):
    """Unrelated query must trigger fallback action."""
    chunks, index = rag_setup
    result = retrieval_agent.run("What is the canteen menu today?", chunks, index)
    assert result["action"] == "fallback"
    assert result["tool_used"] == "none"


def test_retrieval_agent_context_is_string(rag_setup):
    """Context returned by retrieval agent must be a non-empty string."""
    chunks, index = rag_setup
    result = retrieval_agent.run("What is the grading policy?", chunks, index)
    assert isinstance(result["context"], str)
    assert len(result["context"]) > 0


def test_retrieval_agent_top_score_is_float(rag_setup):
    """top_score must be a float between 0 and 1."""
    chunks, index = rag_setup
    result = retrieval_agent.run("What is the attendance requirement?", chunks, index)
    assert isinstance(result["top_score"], float)
    assert 0.0 <= result["top_score"] <= 1.0


def test_retrieval_agent_results_list(rag_setup):
    """results must be a list of dicts with section, text, score."""
    chunks, index = rag_setup
    result = retrieval_agent.run("What is the plagiarism policy?", chunks, index)
    assert isinstance(result["results"], list)
    assert len(result["results"]) > 0
    for r in result["results"]:
        assert "section" in r and "text" in r and "score" in r


# ── Validation Agent tests ────────────────────────────────────────────────────

def test_validation_agent_skips_fallback_response():
    """Validation agent must return UNVERIFIABLE for fallback responses."""
    result = validation_agent.run(
        response="I couldn't find a closely matching policy.",
        context="some context",
        query="some query"
    )
    assert result["verdict"] == "UNVERIFIABLE"
    assert result["agent"] == "validation_agent"


def test_validation_agent_returns_required_keys():
    """Validation agent output must contain verdict, reason, agent keys."""
    result = validation_agent.run(
        response="I couldn't find a closely matching policy.",
        context="context",
        query="query"
    )
    for key in ["verdict", "reason", "agent"]:
        assert key in result


def test_validation_agent_valid_verdict_values():
    """Verdict must be one of the three allowed values."""
    result = validation_agent.run(
        response="I couldn't find a closely matching policy.",
        context="context",
        query="query"
    )
    assert result["verdict"] in ("GROUNDED", "PARTIALLY_GROUNDED", "UNVERIFIABLE")


@patch("agents.validation_agent.client")
def test_validation_agent_grounded_verdict(mock_client):
    """Validation agent must return GROUNDED when LLM says so."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "GROUNDED\nReason: The response matches the context exactly."
    mock_client.chat.completions.create.return_value = mock_response

    result = validation_agent.run(
        response="Students must maintain 75% attendance.",
        context="[ATTENDANCE] Students must maintain a minimum of 75% attendance per semester.",
        query="What is the attendance requirement?"
    )
    assert result["verdict"] == "GROUNDED"
    assert "reason" in result


@patch("agents.validation_agent.client")
def test_validation_agent_unverifiable_on_api_error(mock_client):
    """Validation agent must return UNVERIFIABLE if the API call fails."""
    mock_client.chat.completions.create.side_effect = Exception("API error")

    result = validation_agent.run(
        response="Some response",
        context="Some context",
        query="Some query"
    )
    assert result["verdict"] == "UNVERIFIABLE"


# ── Agent Runner tests ────────────────────────────────────────────────────────

def test_agent_runner_returns_required_keys(rag_setup):
    """Agent runner output must contain all keys from both agents."""
    chunks, index = rag_setup
    result = agent_runner.run_agents("What is the attendance requirement?", chunks, index)
    for key in ["action", "tool_used", "description", "response",
                "context", "top_score", "results", "verdict", "reason"]:
        assert key in result, f"Missing key: {key}"


def test_agent_runner_fallback_skips_validation(rag_setup):
    """Agent runner must set verdict to N/A when retrieval triggers fallback."""
    chunks, index = rag_setup
    result = agent_runner.run_agents("What is the canteen menu today?", chunks, index)
    assert result["action"] == "fallback"
    assert result["verdict"] == "N/A"
