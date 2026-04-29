"""
Agent 1 — Retrieval Agent
Responsible for:
  - Receiving the user query
  - Running RAG retrieval (FAISS for data.txt, Chroma for PDF uploads)
  - Detecting intent and dispatching to the correct tool
  - Returning a structured response dict

This agent wraps the existing RAG + orchestrator pipeline.
It is the first agent in the multi-agent sequence.
"""
from rag import retrieve, retrieve_from_chroma
from orchestrator import run_mcp
from observability import log_query_received, log_retrieval, log_intent_detected, log_fallback


def run(query: str, chunks: list, index, store_type: str = "faiss", collection=None) -> dict:
    """
    Run the retrieval agent.

    Args:
        query:      The user's question
        chunks:     Loaded policy chunks (used for FAISS)
        index:      FAISS index (None if using Chroma)
        store_type: "faiss" or "chroma"
        collection: Chroma collection (None if using FAISS)

    Returns:
        A dict with keys: action, tool_used, description, response,
        context, top_score, results
    """
    log_query_received(query)

    # Step 1: Retrieve relevant chunks from the appropriate store
    if store_type == "chroma" and collection is not None:
        results = retrieve_from_chroma(query, collection, top_k=5)
    else:
        results = retrieve(query, chunks, index, top_k=5)

    context   = "\n\n".join([f"[{r['section']}] {r['text']}" for r in results])
    top_score = results[0]["score"]
    log_retrieval(query, top_score, results[0]["section"], len(results))

    # Step 2: MCP — intent detection + tool dispatch + fallback gate
    result = run_mcp(query, context, top_score=top_score)

    if result["action"] == "fallback":
        log_fallback(query, top_score)
    else:
        log_intent_detected(query, result["tool_used"], top_score)

    # Attach retrieval metadata for the validation agent
    result["context"]   = context
    result["top_score"] = top_score
    result["results"]   = results

    return result
