import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

MODEL = "llama-3.3-70b-versatile"

SYSTEM_CONSTRAINT = (
    "Answer ONLY using the exact information in the context below. "
    "Do NOT rephrase, invert, or reinterpret the rules — quote or closely follow the original wording. "
    "If the answer is not in the context, say 'I don't have that information.'"
)

def _call_llm(instruction: str, context: str, query: str) -> str:
    prompt = f"Context:\n{context}\n\nQuestion: {query}"
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": f"{SYSTEM_CONSTRAINT}\n{instruction}"},
            {"role": "user",   "content": prompt}
        ],
        temperature=0.1,
        max_tokens=300
    )
    return response.choices[0].message.content.strip()


def summarize_context(context: str, query: str) -> str:
    """Return a 2-3 sentence summary of the retrieved context."""
    return _call_llm("Summarize the context in 2-3 sentences. Be concise.", context, query)


def simplify_context(context: str, query: str) -> str:
    """Explain the context in plain language for a student with no prior knowledge."""
    return _call_llm(
        "Explain the context in simple, plain language as if speaking to a student with no prior knowledge. Avoid jargon.",
        context, query
    )


def answer_with_context(context: str, query: str) -> str:
    """Answer the query directly using the retrieved context."""
    return _call_llm("Answer the question directly and clearly.", context, query)


def check_eligibility(context: str, query: str) -> str:
    """Determine student eligibility based on policy rules in the context."""
    instruction = (
        "Based only on the context, answer whether the student is eligible or not. "
        "Structure your response as:\n"
        "- Eligibility: Yes / No / Conditional\n"
        "- Reason: (one sentence from the policy)\n"
        "- Condition: (if any, what they need to do)\n\n"
        "If unclear, say 'I don't have enough information to determine eligibility.'"
    )
    return _call_llm(instruction, context, query)
