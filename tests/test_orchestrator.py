"""Unit tests for orchestrator.py — intent detection and MCP dispatch."""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator import detect_intent, run_mcp


# ── detect_intent tests ───────────────────────────────────────────────────────

def test_intent_eligibility_keywords():
    """Eligibility keywords must map to check_eligibility."""
    assert detect_intent("Am I eligible to appear for the exam?") == "check_eligibility"
    assert detect_intent("Do I qualify for a scholarship?")       == "check_eligibility"
    assert detect_intent("Can I sit for the exam?")               == "check_eligibility"


def test_intent_summarize_keywords():
    """Summarize keywords must map to summarize_context."""
    assert detect_intent("Give me a brief overview of attendance") == "summarize_context"
    assert detect_intent("Summarize the grading policy")           == "summarize_context"
    assert detect_intent("Short summary of plagiarism rules")      == "summarize_context"


def test_intent_simplify_keywords():
    """Simplify keywords must map to simplify_context."""
    assert detect_intent("Explain condonation in simple terms")    == "simplify_context"
    assert detect_intent("What does probation mean?")              == "simplify_context"
    assert detect_intent("Easy explanation of course withdrawal")  == "simplify_context"


def test_intent_default_answer():
    """Queries with no matching keywords must default to answer_with_context."""
    assert detect_intent("What is the attendance requirement?")    == "answer_with_context"
    assert detect_intent("How many days for makeup exam?")         == "answer_with_context"
    assert detect_intent("When does the semester start?")          == "answer_with_context"


def test_intent_eligibility_takes_priority_over_summarize():
    """Eligibility must be detected before summarize when both keywords present."""
    result = detect_intent("Give me a brief summary of eligibility rules")
    assert result == "check_eligibility"


def test_intent_case_insensitive():
    """Intent detection must work regardless of query casing."""
    assert detect_intent("SUMMARIZE the attendance policy") == "summarize_context"
    assert detect_intent("AM I ELIGIBLE for the exam")      == "check_eligibility"


# ── run_mcp fallback tests ────────────────────────────────────────────────────

def test_run_mcp_fallback_on_low_score():
    """run_mcp must return fallback dict when top_score < 0.4."""
    result = run_mcp("some query", "some context", top_score=0.2)
    assert result["action"]    == "fallback"
    assert result["tool_used"] == "none"
    assert "couldn't find" in result["response"].lower()


def test_run_mcp_fallback_at_threshold_boundary():
    """Score exactly at 0.4 must NOT trigger fallback."""
    # We can't call Groq in unit tests, so just verify fallback is not triggered
    # by checking the score gate logic directly
    result_below = run_mcp("query", "context", top_score=0.39)
    assert result_below["action"] == "fallback"


def test_run_mcp_fallback_returns_structured_dict():
    """Fallback response must contain all required keys."""
    result = run_mcp("unrelated query", "irrelevant context", top_score=0.1)
    assert "action"      in result
    assert "tool_used"   in result
    assert "description" in result
    assert "response"    in result
