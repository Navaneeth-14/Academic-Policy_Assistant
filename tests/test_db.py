"""Unit tests for db.py — SQLite logging."""
import pytest
import sqlite3
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import db

TEST_DB = "test_logs.db"


@pytest.fixture(autouse=True)
def use_test_db(monkeypatch, tmp_path):
    """Redirect DB_PATH to a temp file for each test."""
    test_db_path = str(tmp_path / "test_logs.db")
    monkeypatch.setattr(db, "DB_PATH", test_db_path)
    yield test_db_path
    if os.path.exists(test_db_path):
        os.remove(test_db_path)


def test_init_db_creates_table(use_test_db):
    """init_db must create the query_log table."""
    db.init_db()
    conn = sqlite3.connect(use_test_db)
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='query_log'"
    ).fetchall()
    conn.close()
    assert len(tables) == 1


def test_init_db_idempotent(use_test_db):
    """Calling init_db twice must not raise an error."""
    db.init_db()
    db.init_db()


def test_log_query_inserts_row(use_test_db):
    """log_query must insert a row into query_log."""
    db.init_db()
    db.log_query("test query", "answer", "test response")
    conn = sqlite3.connect(use_test_db)
    rows = conn.execute("SELECT * FROM query_log").fetchall()
    conn.close()
    assert len(rows) == 1


def test_log_query_stores_correct_values(use_test_db):
    """Logged row must contain the exact query, action, and response."""
    db.init_db()
    db.log_query("attendance query", "summarize", "summary response")
    conn = sqlite3.connect(use_test_db)
    row = conn.execute("SELECT query, action, response FROM query_log").fetchone()
    conn.close()
    assert row[0] == "attendance query"
    assert row[1] == "summarize"
    assert row[2] == "summary response"


def test_log_query_stores_timestamp(use_test_db):
    """Logged row must have a non-empty timestamp."""
    db.init_db()
    db.log_query("q", "a", "r")
    conn = sqlite3.connect(use_test_db)
    row = conn.execute("SELECT timestamp FROM query_log").fetchone()
    conn.close()
    assert row[0] is not None
    assert len(row[0]) > 0


def test_multiple_queries_logged(use_test_db):
    """Multiple log_query calls must each insert a separate row."""
    db.init_db()
    db.log_query("q1", "answer",    "r1")
    db.log_query("q2", "summarize", "r2")
    db.log_query("q3", "fallback",  "r3")
    conn = sqlite3.connect(use_test_db)
    count = conn.execute("SELECT COUNT(*) FROM query_log").fetchone()[0]
    conn.close()
    assert count == 3
