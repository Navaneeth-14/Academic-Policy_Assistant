"""
Agent Runner — Sequential Multi-Agent Orchestrator
Runs Agent 0 (Rewriter) → Agent 1 (Retrieval) → Agent 2 (Validation) in sequence.

Agent 0 rewrites vague queries for better retrieval.
Agent 1 generates the response using RAG + MCP.
Agent 2 validates the response against the retrieved context.
The runner combines all outputs into a single result dict.

app.py calls run_agents() as the single entry point.
"""
from agents import retrieval_agent, validation_agent, rewriter_agent
from observability import log_response_generated
import time


def run_agents(query: str, chunks: list, index,
               store_type: str = "faiss", collection=None) -> dict:
    """
    Run the full multi-agent pipeline:
      Agent 0 (Rewriter) → Agent 1 (Retrieval) → Agent 2 (Validation)

    Args:
        query:      The user's question
        chunks:     Loaded policy chunks
        index:      FAISS index (None if using Chroma)
        store_type: "faiss" or "chroma"
        collection: Chroma collection (None if using FAISS)

    Returns:
        Combined dict with all agent outputs
    """
    start = time.perf_counter()

    # Agent 0 — Query Rewriter Agent
    rewrite_result  = rewriter_agent.run(query)
    effective_query = rewrite_result["rewritten_query"]

    # Agent 1 — Retrieval Agent
    retrieval_result = retrieval_agent.run(
        effective_query, chunks, index,
        store_type=store_type, collection=collection
    )

    # Agent 2 — Validation Agent (skip if fallback)
    if retrieval_result["action"] == "fallback":
        validation_result = {
            "verdict": "N/A",
            "reason":  "Fallback triggered — no response to validate.",
            "agent":   "validation_agent"
        }
    else:
        validation_result = validation_agent.run(
            response=retrieval_result["response"],
            context=retrieval_result["context"],
            query=effective_query
        )

    duration_ms = (time.perf_counter() - start) * 1000
    log_response_generated(query, retrieval_result["action"], duration_ms)

    return {
        # From rewriter agent
        "original_query":  rewrite_result["original_query"],
        "rewritten_query": rewrite_result["rewritten_query"],
        "was_rewritten":   rewrite_result["was_rewritten"],
        # From retrieval agent
        "action":          retrieval_result["action"],
        "tool_used":       retrieval_result["tool_used"],
        "description":     retrieval_result["description"],
        "response":        retrieval_result["response"],
        "context":         retrieval_result["context"],
        "top_score":       retrieval_result["top_score"],
        "results":         retrieval_result["results"],
        # From validation agent
        "verdict":         validation_result["verdict"],
        "reason":          validation_result["reason"],
    }
