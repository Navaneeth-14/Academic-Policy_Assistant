"""Unit tests for rag.py — retrieval pipeline."""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag import load_chunks, build_index, retrieve


@pytest.fixture(scope="module")
def rag_setup():
    """Load chunks and build index once for all tests."""
    chunks = load_chunks("data.txt")
    index, _ = build_index(chunks)
    return chunks, index


def test_load_chunks_count(rag_setup):
    """data.txt should produce exactly 10 chunks."""
    chunks, _ = rag_setup
    assert len(chunks) == 10


def test_load_chunks_have_section_and_text(rag_setup):
    """Every chunk must have a non-empty section and text."""
    chunks, _ = rag_setup
    for chunk in chunks:
        assert "section" in chunk
        assert "text" in chunk
        assert len(chunk["section"]) > 0
        assert len(chunk["text"]) > 0


def test_known_sections_present(rag_setup):
    """Key policy sections must be present in the loaded chunks."""
    chunks, _ = rag_setup
    sections = [c["section"] for c in chunks]
    for expected in ["ATTENDANCE", "EXAM ELIGIBILITY", "GRADING POLICY", "PLAGIARISM"]:
        assert expected in sections, f"Section '{expected}' not found"


def test_retrieve_returns_top_k(rag_setup):
    """retrieve() must return exactly top_k results."""
    chunks, index = rag_setup
    results = retrieve("attendance policy", chunks, index, top_k=3)
    assert len(results) == 3


def test_retrieve_result_structure(rag_setup):
    """Each result must have section, text, and score keys."""
    chunks, index = rag_setup
    results = retrieve("grading system", chunks, index, top_k=2)
    for r in results:
        assert "section" in r
        assert "text" in r
        assert "score" in r


def test_retrieve_scores_between_0_and_1(rag_setup):
    """Similarity scores must be in [0, 1] range."""
    chunks, index = rag_setup
    results = retrieve("exam eligibility", chunks, index, top_k=3)
    for r in results:
        assert 0.0 <= r["score"] <= 1.0


def test_retrieve_scores_descending(rag_setup):
    """Results must be ordered by score descending."""
    chunks, index = rag_setup
    results = retrieve("attendance requirement", chunks, index, top_k=3)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_relevant_chunk_retrieved_for_attendance(rag_setup):
    """Attendance query must retrieve the ATTENDANCE chunk as top result."""
    chunks, index = rag_setup
    results = retrieve("What is the minimum attendance required?", chunks, index, top_k=3)
    assert results[0]["section"] == "ATTENDANCE"


def test_relevant_chunk_retrieved_for_plagiarism(rag_setup):
    """Plagiarism query must retrieve the PLAGIARISM chunk as top result."""
    chunks, index = rag_setup
    results = retrieve("What is the penalty for plagiarism?", chunks, index, top_k=3)
    assert results[0]["section"] == "PLAGIARISM"


def test_unrelated_query_low_score(rag_setup):
    """Completely unrelated query must return a low top score (below 0.4)."""
    chunks, index = rag_setup
    results = retrieve("What is the canteen menu today?", chunks, index, top_k=3)
    assert results[0]["score"] < 0.4
