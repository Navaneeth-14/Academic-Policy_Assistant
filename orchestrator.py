from tools import summarize_context, simplify_context, answer_with_context, check_eligibility

# ── Tool registry ─────────────────────────────────────────────────────────────
TOOLS = {
    "summarize_context": {
        "name":        "summarize_context",
        "description": "Summarizes retrieved policy chunks in 2-3 sentences.",
        "fn":          summarize_context
    },
    "simplify_context": {
        "name":        "simplify_context",
        "description": "Explains policy in plain language for a student with no prior knowledge.",
        "fn":          simplify_context
    },
    "answer_with_context": {
        "name":        "answer_with_context",
        "description": "Answers the query directly using retrieved context.",
        "fn":          answer_with_context
    },
    "check_eligibility": {
        "name":        "check_eligibility",
        "description": "Determines student eligibility based on policy rules.",
        "fn":          check_eligibility
    }
}

# ── Intent detection (eligibility checked first) ──────────────────────────────
def detect_intent(query: str) -> str:
    q = query.lower()

    # Priority 1: eligibility check
    if any(kw in q for kw in ["eligible", "eligibility", "can i appear", "am i allowed",
                               "do i qualify", "will i pass", "can i sit", "qualify"]):
        return "check_eligibility"

    # Priority 2: summarize
    if any(kw in q for kw in ["summarize", "brief", "overview", "summary", "short"]):
        return "summarize_context"

    # Priority 3: simplify
    if any(kw in q for kw in ["explain", "simple", "what does", "mean", "layman", "easy"]):
        return "simplify_context"

    return "answer_with_context"

# ── Dispatcher ────────────────────────────────────────────────────────────────
def dispatch(tool_name: str, query: str, context: str) -> dict:
    tool = TOOLS[tool_name]
    response = tool["fn"](context, query)
    action_label = tool_name.replace("_context", "").replace("_with", "")
    return {
        "action":      action_label,
        "tool_used":   tool["name"],
        "description": tool["description"],
        "response":    response
    }

# ── Single entry point for app.py ─────────────────────────────────────────────
def run_mcp(query: str, context: str, top_score: float = 1.0) -> dict:
    # Confidence-based fallback — short-circuits before any Groq call
    if top_score < 0.4:
        return {
            "action":      "fallback",
            "tool_used":   "none",
            "description": "Retrieval confidence too low",
            "response":    "I couldn't find a closely matching policy for this query. "
                           "Try rephrasing or being more specific."
        }
    tool_name = detect_intent(query)
    return dispatch(tool_name, query, context)
