"""
Observability layer — structured logging for every query lifecycle event.
Logs to console with timestamps. Each event has a type, so logs can be
filtered or piped to a monitoring tool later.
"""
import logging
import time
from datetime import datetime, timezone

# ── Logger setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("policy_assistant")


def log_query_received(query: str):
    """Log when a query is received."""
    logger.info(f"[QUERY_RECEIVED] query=\"{query}\"")


def log_retrieval(query: str, top_score: float, top_section: str, num_chunks: int):
    """Log RAG retrieval result."""
    logger.info(
        f"[RETRIEVAL] query=\"{query}\" "
        f"top_score={top_score:.3f} "
        f"top_section=\"{top_section}\" "
        f"chunks_retrieved={num_chunks}"
    )


def log_intent_detected(query: str, intent: str, top_score: float):
    """Log which tool was selected by the orchestrator."""
    logger.info(
        f"[INTENT_DETECTED] query=\"{query}\" "
        f"intent=\"{intent}\" "
        f"confidence={top_score:.3f}"
    )


def log_fallback(query: str, top_score: float):
    """Log when the confidence fallback is triggered."""
    logger.warning(
        f"[FALLBACK_TRIGGERED] query=\"{query}\" "
        f"top_score={top_score:.3f} reason=\"below_threshold\""
    )


def log_response_generated(query: str, action: str, duration_ms: float):
    """Log when a response is successfully generated."""
    logger.info(
        f"[RESPONSE_GENERATED] query=\"{query}\" "
        f"action=\"{action}\" "
        f"duration_ms={duration_ms:.1f}"
    )


def log_error(query: str, error: str):
    """Log any error during query processing."""
    logger.error(f"[ERROR] query=\"{query}\" error=\"{error}\"")


class QueryTimer:
    """Context manager to measure and log query duration."""
    def __init__(self, query: str, action_ref: list):
        self.query      = query
        self.action_ref = action_ref   # mutable list so action can be set after init
        self.start      = None

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.perf_counter() - self.start) * 1000
        action = self.action_ref[0] if self.action_ref else "unknown"
        if exc_type:
            log_error(self.query, str(exc_val))
        else:
            log_response_generated(self.query, action, duration_ms)
