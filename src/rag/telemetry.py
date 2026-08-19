"""
Telemetry module — logs every LLM call with tokens, cost, and latency.
Stores logs in data/telemetry.json and exposes helpers for the UI.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from contextlib import contextmanager
from .config import BASE_DIR

TELEMETRY_FILE = BASE_DIR / "data" / "telemetry.json"

# Gemini 2.5 Flash pricing (per 1M tokens, as of mid-2026)
# Free tier has generous limits, but we track cost anyway
# so we can show what it *would* cost at scale.
PRICING = {
    "gemini-2.5-flash": {"input": 0.15, "output": 0.60},   # per 1M tokens
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
}


@contextmanager
def track_latency():
    """Context manager that yields a dict you can read `elapsed_ms` from after exit."""
    result = {"elapsed_ms": 0}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["elapsed_ms"] = round((time.perf_counter() - start) * 1000, 2)


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost in USD based on token counts."""
    rates = PRICING.get(model, PRICING["gemini-2.5-flash"])
    cost = (input_tokens * rates["input"] / 1_000_000) + \
           (output_tokens * rates["output"] / 1_000_000)
    return round(cost, 6)


def log_call(entry: dict):
    """Append a telemetry entry to the JSON log file."""
    TELEMETRY_FILE.parent.mkdir(parents=True, exist_ok=True)

    logs = []
    if TELEMETRY_FILE.exists():
        try:
            logs = json.loads(TELEMETRY_FILE.read_text())
        except (json.JSONDecodeError, Exception):
            logs = []

    logs.append(entry)
    TELEMETRY_FILE.write_text(json.dumps(logs, indent=2))


def record_llm_call(
    model: str,
    input_tokens: int,
    output_tokens: int,
    latency_ms: float,
    question: str,
    status: str = "success",
    error: str = None,
):
    """Record a single LLM API call with all telemetry data."""
    cost = estimate_cost(model, input_tokens, output_tokens)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "cost_usd": cost,
        "latency_ms": latency_ms,
        "question_preview": question[:100],
        "status": status,
    }
    if error:
        entry["error"] = error

    log_call(entry)
    return entry


def get_session_stats() -> dict:
    """Return aggregate stats from the telemetry log."""
    if not TELEMETRY_FILE.exists():
        return {"total_calls": 0, "total_tokens": 0, "total_cost": 0.0}

    try:
        logs = json.loads(TELEMETRY_FILE.read_text())
    except (json.JSONDecodeError, Exception):
        return {"total_calls": 0, "total_tokens": 0, "total_cost": 0.0}

    successful = [l for l in logs if l.get("status") == "success"]
    return {
        "total_calls": len(logs),
        "successful_calls": len(successful),
        "failed_calls": len(logs) - len(successful),
        "total_tokens": sum(l.get("total_tokens", 0) for l in logs),
        "total_input_tokens": sum(l.get("input_tokens", 0) for l in logs),
        "total_output_tokens": sum(l.get("output_tokens", 0) for l in logs),
        "total_cost_usd": round(sum(l.get("cost_usd", 0) for l in logs), 6),
        "avg_latency_ms": round(
            sum(l.get("latency_ms", 0) for l in successful) / max(len(successful), 1), 1
        ),
    }