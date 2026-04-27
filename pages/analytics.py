import sqlite3
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Analytics", page_icon="📊", layout="wide")

st.title("📊 Query Analytics")
st.caption("Live stats from the query log.")
st.divider()

conn = sqlite3.connect("logs.db")

# ── Total queries metric ──────────────────────────────────────────────────────
total = conn.execute("SELECT COUNT(*) FROM query_log").fetchone()[0]

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Queries", total)

with col2:
    most_used = conn.execute(
        "SELECT action FROM query_log GROUP BY action ORDER BY COUNT(*) DESC LIMIT 1"
    ).fetchone()
    st.metric("Most Used Action", most_used[0].upper() if most_used else "—")

with col3:
    last_query = conn.execute(
        "SELECT timestamp FROM query_log ORDER BY id DESC LIMIT 1"
    ).fetchone()
    st.metric("Last Query At", last_query[0][:16] if last_query else "—")

st.divider()

# ── Tool usage bar chart ──────────────────────────────────────────────────────
st.subheader("Tool Usage Breakdown")
tool_df = pd.read_sql_query(
    "SELECT action, COUNT(*) as count FROM query_log GROUP BY action", conn
)
if not tool_df.empty:
    st.bar_chart(tool_df.set_index("action"))
else:
    st.caption("No data yet.")

st.divider()

# ── Recent query log table ────────────────────────────────────────────────────
st.subheader("Recent Query Log")
log_df = pd.read_sql_query(
    "SELECT query, action, timestamp FROM query_log ORDER BY id DESC LIMIT 20", conn
)
if not log_df.empty:
    st.dataframe(log_df, use_container_width=True)
else:
    st.caption("No queries logged yet.")

conn.close()
