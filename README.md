# Academic Policy Assistant

A minimal RAG + MCP system that answers questions about academic policies.
The user asks a question, the system retrieves relevant policy rules, detects what kind of response is needed, and returns a grounded answer using an LLM — all without any agent framework.
---
## How the Project Works (Step by Step)

**Step 1 — Data is loaded and indexed**

When the app starts, it reads `data.txt` (or an uploaded PDF) and splits the content into chunks. Each chunk is one policy rule with a section label like `[ATTENDANCE]` or `[GRADING POLICY]`. Every chunk is converted into a vector embedding using a local sentence-transformer model and stored in a FAISS index in memory.

**Step 2 — User submits a query**

The user types a question in the Streamlit UI and clicks Submit. The query is also converted into an embedding using the same model.

**Step 3 — Agent 0: Query Rewriter Agent**

Before retrieval, the Query Rewriter Agent (`agents/rewriter_agent.py`) takes the raw user query and rewrites it into a clearer, more specific question if needed. For example:
- "attendance?" → "What is the minimum attendance requirement per semester?"
- "can i miss exam?" → "What are the conditions for exam eligibility?"

If the query is already clear, it is returned unchanged. If the rewriter fails for any reason, the original query is used as-is. The rewritten query is what gets passed to the Retrieval Agent.

**Step 4 — Agent 1: Retrieval Agent**

The Retrieval Agent (`agents/retrieval_agent.py`) takes over. It:
- Runs RAG retrieval — compares the query embedding against all chunk embeddings and returns the top 3 most relevant chunks
- Checks the confidence fallback gate — if the top score is below 0.4, returns a fallback response immediately without calling the LLM
- Detects intent from the query using keyword matching (eligibility → summarize → simplify → answer)
- Dispatches to the correct tool and calls the Groq LLM with the retrieved context

This agent acts as a perceive → reason → act loop: it perceives the query, reasons about intent, and acts by calling the right tool.

**Step 5 — Agent 2: Validation Agent**

The Validation Agent (`agents/validation_agent.py`) receives the response from Agent 1 and checks whether it is grounded in the retrieved context. It makes one LLM call with a strict fact-checking prompt and returns a verdict:
- `GROUNDED` — response is fully supported by the context
- `PARTIALLY_GROUNDED` — mostly correct but contains minor additions
- `UNVERIFIABLE` — contains claims not found in the context

This agent acts as the guardrail layer — it catches hallucinations or inaccurate paraphrasing before the response reaches the user.

**Step 6 — Agent Runner combines all outputs**

`agents/agent_runner.py` runs all three agents in sequence (Rewriter → Retrieval → Validation) and combines their outputs into a single result dict passed to the UI. If the query was rewritten, the UI shows the rewritten version alongside the response.

**Step 7 — Response is displayed and logged**

The UI shows the action, tool used, confidence bar, response, source chunks, and the validation verdict. The query is logged to SQLite. Structured observability events are printed to the console.

---

## File-by-File Breakdown

### `app.py` — Streamlit UI and main flow

This is the entry point. It:
1. Initialises the SQLite database on startup
2. Shows a query history sidebar with the last 10 queries, labeled with action badges (Direct Answer 💬, Summary 📝, Simplified 🧩, Eligibility Check ✅, No Match 🚫)
3. Offers a PDF uploader — if a PDF is uploaded it builds a fresh index from it, otherwise loads `data.txt` (cached so it only runs once)
4. Takes the user's query, retrieves chunks, passes the top score to `run_mcp()` for the fallback gate, then displays the result

```python
# The entire response flow in app.py is just these two calls:
results = retrieve(query, chunks, index, top_k=3)
result  = run_mcp(query, context, top_score=results[0]["score"])
```

If `action == "fallback"`, the UI shows `st.warning()` instead of a normal response. Otherwise it shows the action badge, confidence bar, response, and source chunks.

---

### `rag.py` — Embeddings, FAISS index, and retrieval

Contains four functions:

**`load_chunks(filepath)`**
Reads `data.txt` line by line. When it sees a `[SECTION]` label it starts a new chunk. Each chunk stores the section name and the paragraph text below it.

**`load_chunks_from_pdf(filepath)`**
Opens a PDF using `pdfplumber`, extracts text page by page, and splits each page into paragraphs on double newlines. Paragraphs shorter than 40 characters are skipped. Each chunk is labelled with its page number.

**`build_index(chunks)`**
Encodes all chunk texts into 384-dimensional vectors using `all-MiniLM-L6-v2`. Normalises them to unit length, then adds them to a FAISS `IndexFlatIP` (inner product = cosine similarity on normalised vectors).

**`retrieve(query, chunks, index, top_k=3)`**
Encodes the query the same way, runs a FAISS search, and returns the top 3 chunks with their section label, text, and similarity score.

---

### `agents/` — Multi-Agent System

The agents folder contains the two-agent pipeline that replaces the direct `run_mcp()` call in `app.py`.

**`agents/rewriter_agent.py` — Agent 0 (Query Rewriter)**
Runs first in the pipeline. Takes the raw user query and rewrites it into a clearer, more specific question using a single LLM call. Falls back to the original query silently if the API fails or returns an empty response. Returns `original_query`, `rewritten_query`, and `was_rewritten`.

**`agents/retrieval_agent.py` — Agent 1**
Wraps the RAG + orchestrator pipeline into a single agent. Implements a perceive → reason → act loop:
- Perceive: receives the (rewritten) query, runs retrieval
- Reason: checks confidence, detects intent
- Act: dispatches to the correct tool, calls LLM

Returns a result dict with the response plus retrieval metadata (context, top_score, results) needed by Agent 2.

**`agents/validation_agent.py` — Agent 2 (Guardrail Agent)**
Receives the response and context from Agent 1. Makes one LLM call with a strict fact-checking prompt to verify the response is grounded in the retrieved context. Returns a verdict (`GROUNDED`, `PARTIALLY_GROUNDED`, `UNVERIFIABLE`) and a one-sentence reason. Skips validation for fallback responses.

**`agents/agent_runner.py` — Sequential Orchestrator**
Runs all three agents in sequence: Rewriter → Retrieval → Validation. Combines all outputs into a single dict. This is the only function called by `app.py`.

```python
# app.py calls just this:
result = run_agents(query, chunks, index)
```

---

### `orchestrator.py` — Intent detection, fallback, and tool dispatch

This is the MCP layer. It has four parts:

**`TOOLS` dict (tool registry)**
A dictionary mapping tool names to their description and the actual Python function to call. All 4 tools are registered here.

```python
TOOLS = {
    "summarize_context":   { "name": ..., "description": ..., "fn": summarize_context   },
    "simplify_context":    { "name": ..., "description": ..., "fn": simplify_context    },
    "answer_with_context": { "name": ..., "description": ..., "fn": answer_with_context },
    "check_eligibility":   { "name": ..., "description": ..., "fn": check_eligibility   }
}
```

**`detect_intent(query)`**
Checks the query for keywords in priority order (eligibility → summarize → simplify → answer) and returns the tool name.

**`dispatch(tool_name, query, context)`**
Looks up the tool in the registry, calls its function, and returns a structured dict with `action`, `tool_used`, `description`, and `response`.

**`run_mcp(query, context, top_score)`**
The single function called by `app.py`. Checks the confidence score first — if below `0.4`, returns a fallback dict without calling any tool or LLM. Otherwise calls `detect_intent()` then `dispatch()`.

---

### `tools.py` — The four LLM tools

Sets up the Groq client using the OpenAI-compatible API. Uses `llama-3.3-70b-versatile` at temperature `0.1` to keep responses close to the source text.

All tools share one internal function `_call_llm(instruction, context, query)`.

The system prompt used in every call:
> "Answer ONLY using the exact information in the context below. Do NOT rephrase, invert, or reinterpret the rules — quote or closely follow the original wording. If the answer is not in the context, say 'I don't have that information.'"

**`summarize_context(context, query)`**
Returns a 2-3 sentence summary of the retrieved chunks.

**`simplify_context(context, query)`**
Explains the context in plain language for a student with no prior knowledge. Avoids jargon.

**`answer_with_context(context, query)`**
Answers the query directly and clearly from the context.

**`check_eligibility(context, query)`**
Returns a structured eligibility response in this format:
```
- Eligibility: Yes / No / Conditional
- Reason: (one sentence from the policy)
- Condition: (if any, what the student needs to do)
```

---

### `orchestrator.py` — Intent detection and tool dispatch

### `db.py` — SQLite logging

Two functions:

**`init_db()`**
Creates `logs.db` and the `query_log` table if they don't exist. Called once on app startup.

**`log_query(query, action, response)`**
Inserts a row with the query text, action chosen, LLM response, and UTC timestamp.

Table schema:
```
id | query | action | response | timestamp
```

---

### `pages/analytics.py` — Analytics dashboard

A second Streamlit page (auto-detected by Streamlit from the `pages/` folder). Shows:
- 3 metrics: total queries, most used action, last query time
- Bar chart of tool usage breakdown
- Full recent query log table (last 20 entries)

No extra routing needed — Streamlit picks it up automatically.

---

### `observability.py` — Structured Event Logging

Provides structured console logging for every stage of the query lifecycle. Each event has a type tag so logs can be filtered or piped to a monitoring tool.

Events logged: `[QUERY_RECEIVED]`, `[RETRIEVAL]`, `[QUERY_REWRITER]`, `[INTENT_DETECTED]`, `[FALLBACK_TRIGGERED]`, `[VALIDATION]`, `[RESPONSE_GENERATED]`, `[ERROR]`

Also provides `QueryTimer` — a context manager that measures and logs total response duration in milliseconds automatically.

---

### `eval.py` — Accuracy evaluation script

Runs 16 predefined test cases covering all 4 tools and the fallback, without making any LLM calls. Measures:
- **Intent detection accuracy** — correct tool selected vs expected
- **Retrieval accuracy** — correct top section returned vs expected
- **Average retrieval score** — mean cosine similarity across all queries
- **Fallback trigger rate** — how often low-confidence queries are caught

Latest results:
```
Intent detection accuracy : 15/16 (93.8%)
Retrieval accuracy        : 16/16 (100.0%)
Avg retrieval score       : 0.570
Avg score (non-fallback)  : 0.653
Fallback triggered        : 3/16  (18.8%)
```

---

### `test_run.py` — Backend smoke test

Runs 5 representative queries end-to-end (including one LLM call per query) without Streamlit. Adds a 12-second delay between queries to stay within Groq's free tier rate limits. Useful for quick verification after any code change.

---

### `data.txt` — Default policy data

Contains 10 academic policy rules in this format:

```
[SECTION NAME]
Policy text goes here as one or more sentences.
```

Sections: Attendance, Condonation, Exam Eligibility, Grading Policy, Internal Assessment, Makeup Exam, Academic Probation, Plagiarism, Course Withdrawal, Scholarship Eligibility.

---

### `.env` — API key

```
GROQ_API_KEY=your-key-here
```

Loaded by `tools.py` via `python-dotenv`. Listed in `.gitignore` — never committed.

---

### `.streamlit/config.toml` — Streamlit config

Disables the file watcher to suppress noisy warnings from `transformers` submodules that require `torchvision`.

```toml
[server]
fileWatcherType = "none"
```

---

## Project Structure

```
academic-policy-assistant/
├── app.py                  → Streamlit UI + main flow
├── rag.py                  → Embeddings, FAISS index, PDF loader
├── orchestrator.py         → Fallback gate, intent detection, tool dispatch
├── tools.py                → 4 LLM tools (answer, summarize, simplify, eligibility)
├── db.py                   → SQLite query logging
├── observability.py        → Structured event logging + query timer
├── eval.py                 → Accuracy evaluation (no LLM calls)
├── test_run.py             → End-to-end smoke test
├── data.txt                → Default policy data
├── requirements.txt        → Dependencies
├── .env                    → API key (not committed)
├── .gitignore              → Ignores env/, .env, logs.db
├── README.md               → Full project documentation
├── .streamlit/
│   └── config.toml         → Disables file watcher
├── agents/
│   ├── __init__.py
│   ├── rewriter_agent.py   → Agent 0 — rewrites vague queries
│   ├── retrieval_agent.py  → Agent 1 — RAG + MCP tool dispatch
│   ├── validation_agent.py → Agent 2 — validates response grounding
│   └── agent_runner.py     → Runs all 3 agents sequentially
├── pages/
│   └── analytics.py        → Analytics dashboard page
└── tests/
    ├── test_rag.py             → 10 unit tests
    ├── test_orchestrator.py    → 9 unit tests
    ├── test_db.py              → 6 unit tests
    ├── test_agents.py          → 12 unit tests
    └── test_rewriter_agent.py  → 6 unit tests  (43 total, all passing)
```

---

## Setup

```bash
# 1. Create and activate virtual environment
python -m venv env
env\Scripts\Activate.ps1        # Windows PowerShell

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add Groq API key to .env
# Get a free key at https://console.groq.com
GROQ_API_KEY=your-key-here

# 4. Run the app
streamlit run app.py
```

---

## Tech Stack

| Component | Library / Service |
|---|---|
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` |
| Vector store | `faiss-cpu` (in-memory) |
| LLM | Groq API — `llama-3.3-70b-versatile` |
| UI | `streamlit` |
| PDF parsing | `pdfplumber` |
| Query logging | `sqlite3` (Python built-in) |
| Analytics | `pandas` |
| Env management | `python-dotenv` |

---

## What is NOT used

- No LangChain, LlamaIndex, or any agent framework
- No OpenAI API (Groq's free tier via OpenAI-compatible client)
- No vector database service (FAISS runs fully in memory)
- No external database (SQLite file stored locally)
