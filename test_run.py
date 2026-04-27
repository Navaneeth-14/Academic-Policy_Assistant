"""Smoke test covering all features including new ones."""
import time
from rag import load_chunks, build_index, retrieve
from orchestrator import run_mcp

# ── Load RAG ──────────────────────────────────────────────────────────────────
print("Loading chunks and building FAISS index...")
chunks = load_chunks("data.txt")
index, _ = build_index(chunks)
print(f"  {len(chunks)} chunks loaded.\n")

# ── Test queries ──────────────────────────────────────────────────────────────
queries = [
    # (query,                                               expected_action)
    ("What is the minimum attendance required?",            "answer"),
    ("Give me a brief overview of the grading policy.",     "summarize"),
    ("What does academic probation mean in simple terms?",  "simplify"),
    ("Am I eligible to appear for the exam?",               "check_eligibility"),
    ("What is the policy on quantum physics experiments?",  "fallback"),  # low score expected
]

for i, (q, expected) in enumerate(queries):
    if i > 0:
        print("  Waiting 12s between queries...")
        time.sleep(12)

    results   = retrieve(q, chunks, index, top_k=3)
    context   = "\n\n".join([f"[{r['section']}] {r['text']}" for r in results])
    top_score = results[0]["score"]

    result = run_mcp(q, context, top_score=top_score)

    status = "✓" if result["action"] == expected else "✗"
    print(f"{status} Query   : {q}")
    print(f"  Expected: {expected} | Got: {result['action']} | Tool: {result['tool_used']} | Score: {top_score:.3f}")
    print(f"  Response: {result['response'][:180]}...")
    print("-" * 70)
