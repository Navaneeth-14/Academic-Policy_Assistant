"""
Agent 0 — Query Rewriter Agent
Responsible for:
  - Receiving the raw user query
  - Rewriting vague or ambiguous queries into clearer, more specific ones
  - Improving RAG retrieval quality before chunks are fetched

This agent runs FIRST in the pipeline, before the Retrieval Agent.
If rewriting fails or the query is already clear, it returns the original query unchanged.
"""
import os
import logging
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

MODEL  = "llama-3.3-70b-versatile"
logger = logging.getLogger("policy_assistant")


def run(query: str) -> dict:
    """
    Rewrite the query to be more specific and retrieval-friendly.

    Args:
        query: The raw user query

    Returns:
        dict with keys: original_query, rewritten_query, was_rewritten
    """
    prompt = (
        "You are a query rewriter for an academic policy assistant. "
        "Your job is to fix grammar and make vague or incomplete queries more specific — "
        "but ONLY using words and topics already present in the original query.\n\n"
        "Rules:\n"
        "- Do NOT add any new topics, subjects, or keywords not in the original query\n"
        "- Do NOT assume or infer what the user meant beyond what they wrote\n"
        "- If the query is already clear, return it exactly as-is\n"
        "- Only fix grammar, sentence structure, or vagueness\n"
        "- Keep the rewritten query under 20 words\n"
        "- Return ONLY the rewritten query, nothing else\n\n"
        f"Query: {query}\n"
        "Rewritten query:"
    )

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=60
        )
        rewritten = response.choices[0].message.content.strip().strip('"').strip("'")

        # Sanity check — if rewritten is empty or too long, fall back to original
        if not rewritten or len(rewritten) > 200:
            rewritten = query
            was_rewritten = False
        else:
            was_rewritten = rewritten.lower() != query.lower()

        logger.info(
            f"[QUERY_REWRITER] original=\"{query}\" "
            f"rewritten=\"{rewritten}\" "
            f"was_rewritten={was_rewritten}"
        )

        return {
            "original_query": query,
            "rewritten_query": rewritten,
            "was_rewritten": was_rewritten
        }

    except Exception as e:
        logger.error(f"[REWRITER_ERROR] {e} — using original query")
        return {
            "original_query":  query,
            "rewritten_query": query,
            "was_rewritten":   False
        }
