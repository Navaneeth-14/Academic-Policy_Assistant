import sqlite3
from datetime import datetime, timezone

DB_PATH = "logs.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS query_log (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            query     TEXT,
            action    TEXT,
            response  TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

def log_query(query: str, action: str, response: str):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO query_log (query, action, response, timestamp) VALUES (?, ?, ?, ?)",
        (query, action, response, datetime.now(timezone.utc).isoformat())
    )
    conn.commit()
    conn.close()
