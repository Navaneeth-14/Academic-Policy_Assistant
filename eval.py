"""
Evaluation script — measures:
  1. RAG retrieval quality   (avg similarity score, score distribution)
  2. MCP intent accuracy     (correct tool selected / total)
  3. Fallback trigger rate   (how often low-confidence fallback fires)

No LLM calls are made — this only tests retrieval and intent detection.
"""
from rag import load_chunks, build_index, retrieve
from orchestrator import detect_intent, run_mcp

# ── Ground truth dataset ──────────────────────────────────────────────────────
# Format: (query, expected_intent, expected_top_section)
EVAL_SET = [
    # answer
    ("What is the minimum attendance required?",              "answer_with_context",  "ATTENDANCE"),
    ("What happens if I miss an internal assessment?",        "answer_with_context",  "INTERNAL ASSESSMENT"),
    ("How many days do I have to apply for a makeup exam?",   "answer_with_context",  "MAKEUP EXAM"),
    ("What is the penalty for plagiarism?",                   "answer_with_context",  "PLAGIARISM"),
    ("When can I withdraw from a course?",                    "answer_with_context",  "COURSE WITHDRAWAL"),

    # summarize
    ("Give me a brief overview of the grading policy.",       "summarize_context",    "GRADING POLICY"),
    ("Summarize the attendance rules.",                       "summarize_context",    "ATTENDANCE"),
    ("Give a short summary of plagiarism policy.",            "summarize_context",    "PLAGIARISM"),

    # simplify
    ("What does academic probation mean in simple terms?",    "simplify_context",     "ACADEMIC PROBATION"),
    ("Explain condonation in easy language.",                 "simplify_context",     "CONDONATION"),
    ("What does course withdrawal mean?",                     "simplify_context",     "COURSE WITHDRAWAL"),

    # eligibility
    ("Am I eligible to appear for the exam?",                 "check_eligibility",    "EXAM ELIGIBILITY"),
    ("Do I qualify for a merit scholarship?",                 "check_eligibility",    "SCHOLARSHIP ELIGIBILITY"),
    ("Can I sit for the exam with 60% attendance?",           "check_eligibility",    "EXAM ELIGIBILITY"),

    # fallback (unrelated — should score < 0.4)
    ("What is the policy on quantum physics experiments?",    "fallback",             None),
    ("Tell me about the college canteen menu.",               "fallback",             None),
]

# ── Load RAG ──────────────────────────────────────────────────────────────────
print("Loading chunks and building index...")
chunks = load_chunks("data.txt")
index, _ = build_index(chunks)
print(f"  {len(chunks)} chunks loaded.\n")

# ── Run evaluation ────────────────────────────────────────────────────────────
intent_correct      = 0
retrieval_correct   = 0
fallback_triggered  = 0
scores              = []

print(f"{'#':<3} {'QUERY':<50} {'EXP INTENT':<22} {'GOT INTENT':<22} {'SCORE':<7} {'TOP SECTION':<25} {'I✓'} {'R✓'}")
print("-" * 145)

for i, (query, exp_intent, exp_section) in enumerate(EVAL_SET, 1):
    results   = retrieve(query, chunks, index, top_k=3)
    top_score = results[0]["score"]
    top_section = results[0]["section"]
    scores.append(top_score)

    # Intent check (use run_mcp to also test fallback path)
    context = "\n\n".join([f"[{r['section']}] {r['text']}" for r in results])
    result  = run_mcp(query, context, top_score=top_score)
    got_intent = result["tool_used"] if result["action"] != "fallback" else "fallback"

    # Score intent
    i_correct = (got_intent == exp_intent) or \
                (exp_intent == "fallback" and result["action"] == "fallback")
    if i_correct:
        intent_correct += 1

    # Score retrieval (top chunk section matches expected)
    r_correct = (exp_section is None) or (top_section == exp_section)
    if r_correct:
        retrieval_correct += 1

    if result["action"] == "fallback":
        fallback_triggered += 1

    i_mark = "✓" if i_correct else "✗"
    r_mark = "✓" if r_correct else "✗"
    print(f"{i:<3} {query[:49]:<50} {exp_intent:<22} {got_intent:<22} {top_score:<7.3f} {top_section:<25} {i_mark}  {r_mark}")

# ── Summary ───────────────────────────────────────────────────────────────────
total = len(EVAL_SET)
non_fallback = [s for s in scores if s >= 0.4]

print("\n" + "=" * 60)
print("EVALUATION SUMMARY")
print("=" * 60)
print(f"Total test cases          : {total}")
print(f"Intent detection accuracy : {intent_correct}/{total} ({intent_correct/total*100:.1f}%)")
print(f"Retrieval accuracy        : {retrieval_correct}/{total} ({retrieval_correct/total*100:.1f}%)")
print(f"Avg retrieval score       : {sum(scores)/len(scores):.3f}")
print(f"Avg score (non-fallback)  : {sum(non_fallback)/len(non_fallback):.3f}" if non_fallback else "")
print(f"Fallback triggered        : {fallback_triggered}/{total} ({fallback_triggered/total*100:.1f}%)")
print(f"Min score                 : {min(scores):.3f}")
print(f"Max score                 : {max(scores):.3f}")
print("=" * 60)
