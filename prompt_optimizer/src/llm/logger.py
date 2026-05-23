"""JSONL logging for every LLM API call.

Each call is appended as a single JSON line to ``{run_dir}/llm_calls.jsonl``
with full input/output, token counts, latency, and estimated cost.
"""

import datetime
import json
from pathlib import Path
from typing import Any


# Pricing per 1M tokens (input, output) — approximate free-tier equivalents.
_PRICE_TABLE: dict[str, tuple[float, float]] = {
    "gemini-1.5-flash":        (0.075, 0.30),
    "gemini-2.0-flash":        (0.10,  0.40),
    "llama-3.1-8b-instant":    (0.05,  0.08),
    "llama-3.1-70b-versatile": (0.59,  0.79),
}


class LLMCallLogger:
    """Appends structured JSONL records for every LLM call."""

    def __init__(self, log_path: Path) -> None:
        """Open (or create) the JSONL log file.

        Args:
            log_path: Filesystem path for the ``.jsonl`` log file.
        """
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._path = log_path
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_cost = 0.0
        self._total_calls = 0

    def log(
        self,
        *,
        call_type: str,
        provider: str,
        model: str,
        input_prompt: str,
        output: str,
        input_tokens: int,
        output_tokens: int,
        latency_seconds: float,
        success: bool,
        error: str | None = None,
        has_pdf: bool = False,
    ) -> None:
        """Record a single LLM call to the JSONL log.

        Args:
            call_type: ``'extract'``, ``'generate'``, or ``'judge'``.
            provider: ``'gemini'`` or ``'groq'``.
            model: Model identifier string.
            input_prompt: The prompt sent (may be truncated).
            output: The raw model response (may be truncated).
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
            latency_seconds: Wall-clock time for the call.
            success: Whether the call succeeded.
            error: Error message if the call failed.
            has_pdf: Whether a PDF was attached.
        """
        cost = _estimate_cost(model, input_tokens, output_tokens)

        self._total_input_tokens += input_tokens
        self._total_output_tokens += output_tokens
        self._total_cost += cost
        self._total_calls += 1

        record = {
            "timestamp": _now_iso(),
            "call_type": call_type,
            "provider": provider,
            "model": model,
            "input_prompt": input_prompt,
            "input_has_pdf": has_pdf,
            "output": output,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "estimated_cost_usd": round(cost, 6),
            "latency_seconds": round(latency_seconds, 3),
            "success": success,
            "error": error,
        }

        with open(self._path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

    # -- Accessors for budget tracking ------------------------------------

    @property
    def total_tokens(self) -> int:
        """Cumulative token usage across all logged calls."""
        return self._total_input_tokens + self._total_output_tokens

    @property
    def total_cost(self) -> float:
        """Cumulative estimated cost (USD) across all logged calls."""
        return self._total_cost

    @property
    def total_calls(self) -> int:
        """Total number of logged calls."""
        return self._total_calls

    def get_usage_summary(self) -> dict[str, Any]:
        """Return a summary dict of cumulative usage."""
        return {
            "total_calls": self._total_calls,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self._total_cost, 4),
        }


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate the USD cost of a call using the price table."""
    prices = _PRICE_TABLE.get(model, (0.10, 0.40))
    input_cost = (input_tokens / 1_000_000) * prices[0]
    output_cost = (output_tokens / 1_000_000) * prices[1]
    return input_cost + output_cost


def _now_iso() -> str:
    """Return the current UTC time in ISO 8601 format."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat()
