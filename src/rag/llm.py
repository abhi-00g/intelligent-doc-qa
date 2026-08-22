"""
LLM module — calls Gemini with telemetry logging and structured error handling.

Telemetry flows to two destinations:
1. Local JSON file (data/telemetry.json) — always, via record_llm_call
2. AI Cost Dashboard backend — when SDK is configured, via cost_tracker.log()

The dashboard integration is additive — if the SDK isn't configured or
the backend is unreachable, the RAG app works exactly as before.
"""

import os
import logging
import google.generativeai as genai
from .telemetry import track_latency, record_llm_call
from .config import cost_tracker

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful assistant. Answer concisely using only
the supplied context. If the answer is not in the context, say you don't know.
Cite the source document when possible."""

MODEL_NAME = "gemini-2.5-flash"


def _send_to_dashboard(
    input_tokens: int,
    output_tokens: int,
    latency_ms: float,
    status: str = "success",
    error_message: str | None = None,
    feature: str = "qa_pipeline",
) -> None:
    """
    Send telemetry to the AI Cost Dashboard via the SDK.

    This is fire-and-forget — the SDK's background thread handles
    batching, retries, and graceful failure. Never blocks or crashes.
    """
    if cost_tracker is None:
        return

    try:
        cost_tracker.log(
            model=MODEL_NAME,
            provider="google",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=int(latency_ms),
            feature=feature,
            status=status,
            error_message=error_message,
        )
    except Exception as e:
        # Absolute safety net — SDK should never raise, but just in case
        logger.debug("Dashboard SDK error (non-fatal): %s", e)


def answer_with_gemini(question: str, context_chunks: list) -> str:
    """Generate an answer using Gemini, with full telemetry and error handling."""

    # --- Guard: missing API key ---
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        logger.error("GEMINI_API_KEY not set")
        record_llm_call(MODEL_NAME, 0, 0, 0, question, status="error", error="API key missing")
        _send_to_dashboard(0, 0, 0, status="error", error_message="API key missing")
        return "Error: GEMINI_API_KEY not set in your .env file."

    # --- Guard: empty context ---
    if not context_chunks:
        logger.warning("No context chunks provided for question: %s", question[:80])
        record_llm_call(MODEL_NAME, 0, 0, 0, question, status="error", error="Empty context")
        _send_to_dashboard(0, 0, 0, status="error", error_message="Empty context")
        return "No relevant context found in the document. Try rephrasing your question."

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(MODEL_NAME)

    context = "\n\n".join([f"- {c[:1200]}" for c in context_chunks])
    prompt = f"{SYSTEM_PROMPT}\n\nQuestion: {question}\n\nContext:\n{context}\n\nAnswer:"

    # --- Call Gemini with latency tracking ---
    try:
        with track_latency() as timing:
            response = model.generate_content(prompt)

        # Extract token counts from response metadata
        usage = getattr(response, "usage_metadata", None)
        input_tokens = getattr(usage, "prompt_token_count", 0) if usage else 0
        output_tokens = getattr(usage, "candidates_token_count", 0) if usage else 0

        answer = response.text.strip()

        # Log to local JSON file
        entry = record_llm_call(
            model=MODEL_NAME,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=timing["elapsed_ms"],
            question=question,
            status="success",
        )
        logger.info(
            "Gemini call: %d input tokens, %d output tokens, %.0fms, $%.6f",
            input_tokens, output_tokens, timing["elapsed_ms"], entry["cost_usd"]
        )

        # Send to AI Cost Dashboard
        _send_to_dashboard(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=timing["elapsed_ms"],
            status="success",
        )

        return answer

    except genai.types.BlockedPromptException:
        logger.warning("Prompt blocked by Gemini safety filters")
        record_llm_call(MODEL_NAME, 0, 0, 0, question, status="error", error="Prompt blocked by safety filters")
        _send_to_dashboard(0, 0, 0, status="error", error_message="Prompt blocked by safety filters")
        return "Your question was blocked by the model's safety filters. Please rephrase."

    except Exception as e:
        error_msg = str(e)
        logger.error("Gemini API error: %s", error_msg)
        record_llm_call(MODEL_NAME, 0, 0, 0, question, status="error", error=error_msg)
        _send_to_dashboard(0, 0, 0, status="error", error_message=error_msg)

        if "429" in error_msg or "quota" in error_msg.lower():
            return "Rate limit reached. Please wait a moment and try again."
        if "503" in error_msg or "unavailable" in error_msg.lower():
            return "The Gemini API is temporarily unavailable. Please try again shortly."

        return f"An error occurred while generating the answer. Please try again."
