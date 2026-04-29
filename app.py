import os
import sqlite3
import tempfile
import streamlit as st
from rag import (load_chunks, load_chunks_from_pdf, build_index, retrieve,
                 build_chroma_index, retrieve_from_chroma)
from agents.agent_runner import run_agents
from db import init_db, log_query
from observability import QueryTimer

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Academic Policy Assistant",
    page_icon="🎓",
    layout="wide"
)

# ── Init ──────────────────────────────────────────────────────────────────────
init_db()

# ── Sidebar — recent queries ──────────────────────────────────────────────────
def get_recent_queries(limit=10):
    conn = sqlite3.connect("logs.db")
    rows = conn.execute(
        "SELECT query, action, timestamp FROM query_log ORDER BY id DESC LIMIT ?",
        (limit,)
    ).fetchall()
    conn.close()
    return rows

ACTION_LABELS = {
    "answer":            "Direct Answer 💬",
    "summarize":         "Summary 📝",
    "simplify":          "Simplified 🧩",
    "check_eligibility": "Eligibility Check ✅",
    "fallback":          "No Match 🚫"
}

with st.sidebar:
    st.header("🕘 Recent Queries")
    history = get_recent_queries()
    if history:
        for q, action, ts in history:
            label = ACTION_LABELS.get(action, action.upper())
            st.markdown(f"**{label}**")
            st.caption(q)
            st.caption(ts[:16])
            st.divider()
    else:
        st.caption("No queries yet.")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🎓 Academic Policy Assistant")
st.caption("Ask anything about academic policies — powered by RAG + MCP.")
st.divider()

# ── Data source ───────────────────────────────────────────────────────────────
with st.expander("📄 Upload a custom policy PDF"):
    uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")

@st.cache_resource
def load_default_rag():
    chunks = load_chunks("data.txt")
    index, _ = build_index(chunks)
    return chunks, index, "faiss"

if uploaded_pdf:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_pdf.read())
        tmp_path = tmp.name
    chunks = load_chunks_from_pdf(tmp_path)
    os.unlink(tmp_path)

    # Fix 4 — warn if PDF has no extractable text (scanned/image PDF)
    if not chunks:
        st.error("⚠️ No text could be extracted from this PDF. It may be a scanned or image-based document. Please upload a text-based PDF.")
        chunks, index, store_type = load_default_rag()
        collection = None
    else:
        # Use Chroma for PDF — persistent, heading-aware chunks
        collection = build_chroma_index(chunks)
        index      = None
        store_type = "chroma"
        st.success(f"Loaded {len(chunks)} chunks from PDF (using Chroma index).")
else:
    chunks, index, store_type = load_default_rag()
    collection = None

# ── Query input ───────────────────────────────────────────────────────────────
query = st.text_input(
    "Your question",
    placeholder="e.g. What is the attendance requirement? / Am I eligible for the exam?",
    label_visibility="visible"
)

col1, col2 = st.columns([1, 5])
with col1:
    submitted = st.button("Submit", use_container_width=True, type="primary")

if submitted and query.strip():
    action_ref = ["unknown"]
    with QueryTimer(query, action_ref):
        with st.spinner("Retrieving and thinking..."):
            # Run full multi-agent pipeline (Rewriter → Retrieval → Validation)
            result = run_agents(query, chunks, index,
                                store_type=store_type,
                                collection=collection if store_type == "chroma" else None)
            action_ref[0] = result["action"]

            # Unpack for display
            results   = result["results"]
            top_score = result["top_score"]

            # Log to SQLite
            log_query(query, result["action"], result["response"])

    st.divider()

    # ── Show rewritten query if it was changed ────────────────────────────────
    if result.get("was_rewritten"):
        st.caption(f"🔄 Query rewritten to: _{result['rewritten_query']}_")

    # ── Action badge + confidence ─────────────────────────────────────────────
    col_a, col_b = st.columns([2, 3])
    with col_a:
        action_colors = {
            "answer":            "🟢",
            "summarize":         "🔵",
            "simplify":          "🟡",
            "check_eligibility": "🟠",
            "fallback":          "🔴"
        }
        icon = action_colors.get(result["action"], "⚪")
        st.markdown(f"### {icon} Action: `{result['action']}`")
        st.caption(f"Tool: `{result['tool_used']}` — {result['description']}")

    with col_b:
        if result["action"] != "fallback":
            st.caption("Retrieval confidence")
            st.progress(min(top_score, 1.0))
            st.caption(f"Top chunk score: {top_score:.3f}")

    st.divider()

    # ── Response ──────────────────────────────────────────────────────────────
    # Fix 2 — treat "no information" LLM responses as soft fallback in UI
    no_info_response = (
        result["action"] != "fallback" and
        "don't have that information" in result["response"].lower()
    )

    if result["action"] == "fallback":
        st.warning(result["response"])
    elif no_info_response:
        st.warning("🔍 The system couldn't find relevant information for this query in the loaded document. Try rephrasing or uploading a more specific policy document.")
    else:
        st.markdown("#### Response")
        st.write(result["response"])

    # ── Source chunks ─────────────────────────────────────────────────────────
    if result["action"] != "fallback":
        with st.expander("📚 Source chunks used"):
            for r in results:
                st.markdown(f"**[{r['section']}]** — score: `{r['score']:.3f}`")
                st.write(r["text"])
                st.divider()

    # ── Validation Agent verdict ───────────────────────────────────────────────
    if result["action"] != "fallback":
        verdict = result.get("verdict", "UNVERIFIABLE")
        reason  = result.get("reason", "")
        verdict_icons = {
            "GROUNDED":            "✅ GROUNDED",
            "PARTIALLY_GROUNDED":  "⚠️ PARTIALLY GROUNDED",
            "UNVERIFIABLE":        "❌ UNVERIFIABLE"
        }
        if verdict == "PARTIALLY_GROUNDED":
            st.warning(f"⚠️ Response may contain minor additions beyond the source policy. Verify with the source chunks below.")
        with st.expander("🔍 Validation Agent — Grounding Check"):
            st.markdown(f"**Verdict: {verdict_icons.get(verdict, verdict)}**")
            st.caption(f"Reason: {reason}")
