"""LLM-based prompt mutator for generating improved extraction prompts.

The mutator constructs a *meta-prompt* that includes the current prompt,
its validation scores, the worst-performing fields with failure examples,
and recent mutation history — then asks a cheap LLM (Groq / Llama 3.1 8B)
to propose a targeted improvement.
"""

import hashlib
import json
from typing import Any

from src.llm.client import LLMClient
from src.scoring.scorer import ScoringResult


class PromptMutator:
    """Generates candidate prompts by asking an LLM to improve the current one."""

    def __init__(self, mutation_client: LLMClient) -> None:
        """Create a mutator backed by the given LLM client.

        Args:
            mutation_client: An LLM client used for proposing mutations.
        """
        self._client = mutation_client

    def mutate(
        self,
        current_prompt: str,
        scoring_result: ScoringResult,
        schema_text: str,
        history: list[dict[str, Any]] | None = None,
        diversify: bool = False,
    ) -> str:
        """Propose an improved extraction prompt.

        Args:
            current_prompt: The prompt to improve.
            scoring_result: Most recent scoring result on the validation set.
            schema_text: The JSON schema (for context).
            history: List of recent mutation outcomes.
            diversify: If ``True``, instruct the LLM to make larger changes.

        Returns:
            The proposed new prompt string.
        """
        system = self._build_system_prompt(diversify)
        user = self._build_user_prompt(
            current_prompt, scoring_result, schema_text, history, diversify,
        )

        raw = self._client.generate(system, user)
        return self._extract_prompt(raw)

    # -- Prompt construction -----------------------------------------------

    @staticmethod
    def _build_system_prompt(diversify: bool) -> str:
        base = (
            "You are an expert prompt engineer specializing in "
            "structured JSON extraction from PDF documents.\n\n"
            "YOUR PRIMARY GOAL:\n"
            "Write prompts that generalize across ALL documents "
            "of this type. Every rule must apply broadly to any "
            "document in this category -- not just the one whose "
            "scores you are seeing.\n\n"
            "MUTATION STRATEGIES:\n"
            "1. Add field-specific rules for failing fields\n"
            "2. Add explicit null/missing value handling\n"
            "3. Add edge case rules for formatting variations\n"
            "4. Clarify ambiguous field definitions from schema\n"
            "5. Add output format enforcement\n"
            "6. Add instructions for tables, lists, headers\n"
            "7. Specify how to handle conflicting information\n\n"
            "STRICT RULES:\n"
            "- Return ONLY improved prompt inside <PROMPT> tags\n"
            "- Prompt MUST contain {schema} placeholder\n"
            "- Only change what is failing, keep what works\n"
            "- Every instruction must work on unseen documents\n"
            "- Never reference specific document content\n"
            "- Keep prompt under 500 words\n"
        )
        if diversify:
            base += (
                "\nDIVERSIFICATION MODE:\n"
                "Small mutations failed repeatedly. Try ONE of:\n"
                "- Add a concrete extraction example\n"
                "- Switch from bullets to numbered steps\n"
                "- Add chain-of-thought: think before extracting\n"
                "- Restructure the entire prompt from scratch\n"
            )
        return base

    @staticmethod
    def _build_user_prompt(
        current_prompt: str,
        scoring: ScoringResult,
        schema_text: str,
        history: list[dict[str, Any]] | None,
        diversify: bool,
    ) -> str:
        """User prompt with full context for the mutation LLM."""
        worst_fields = _get_worst_fields(scoring, n=5)
        history_text = _format_history(history or [])

        parts = [
            "## Current Prompt",
            f"```\n{current_prompt}\n```\n",
            "## Validation Scores",
            f"- Aggregate score: {scoring.aggregate_score:.3f}",
            f"- F1: {scoring.aggregate_f1:.3f}",
            f"- Matched: {scoring.matched_fields}/{scoring.total_fields} fields",
            f"- Missing: {scoring.missing_fields} fields\n",
        ]

        if worst_fields:
            parts.append("## Worst-Performing Fields")
            for path, fs in worst_fields:
                parts.append(
                    f"- **{path}** ({fs.metric_type}): score={fs.score:.2f} "
                    f"| gold={_trunc(fs.gold_value)} "
                    f"| pred={_trunc(fs.predicted_value)} "
                    f"| reason: {fs.reason}"
                )
            parts.append("")

        if scoring.subtree_scores:
            parts.append("## Per-Section Scores")
            for key, val in sorted(scoring.subtree_scores.items()):
                parts.append(f"- {key}: {val:.3f}")
            parts.append("")

        if history_text:
            parts.append("## Recent Mutation History")
            parts.append(history_text)
            parts.append("")

        if scoring.aggregate_score < 0.7:
            parts.append("## Suggested Improvement Direction")
            parts.append(
                "Score is below 0.7 — the prompt needs significant "
                "improvement. Consider adding a concrete example like:\n"
                "```\n"
                "EXAMPLE:\n"
                "Document text: 'John Smith, PhD. Harvard University. 2023.'\n"
                "Correct extraction: {\"author\": \"John Smith\", "
                "\"degree\": \"PhD\", \"institution\": \"Harvard University\","
                " \"year\": 2023}\n"
                "```\n"
                "Adapt the example to match the actual schema fields."
            )
            parts.append("")

        parts.append(
            "## Instructions\n"
            "Propose an improved prompt that addresses the weaknesses above. "
            "Wrap the new prompt in <PROMPT> ... </PROMPT> tags."
        )

        return "\n".join(parts)

    @staticmethod
    def _extract_prompt(raw: str) -> str:
        """Extract the prompt from between <PROMPT> tags."""
        import re

        match = re.search(r"<PROMPT>(.*?)</PROMPT>", raw, re.DOTALL)
        if match:
            return match.group(1).strip()

        if "{schema}" in raw:
            return raw.strip()

        return raw.strip()


# ---------------------------------------------------------------------------
#  Utilities
# ---------------------------------------------------------------------------

def _get_worst_fields(
    scoring: ScoringResult, n: int = 5,
) -> list[tuple[str, Any]]:
    """Return the *n* fields with the lowest scores."""
    items = [
        (path, fs) for path, fs in scoring.field_scores.items()
        if not path.endswith("._array_avg")
    ]
    items.sort(key=lambda x: x[1].score)
    return items[:n]


def _format_history(history: list[dict[str, Any]]) -> str:
    """Format recent mutation history as a compact string."""
    if not history:
        return ""
    lines: list[str] = []
    for h in history[-5:]:
        status = "✓ accepted" if h.get("accepted") else "✗ rejected"
        score = h.get("score", 0)
        reason = h.get("reason", "")
        lines.append(f"- Iter {h.get('iteration', '?')}: {status} "
                      f"(score={score:.3f}) {reason}")
    return "\n".join(lines)


def _trunc(value: Any, max_len: int = 80) -> str:
    """Truncate a value's string repr for display."""
    s = str(value)
    if len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s


def prompt_hash(prompt: str) -> str:
    """Return a short SHA-256 hash of a prompt string."""
    return hashlib.sha256(prompt.encode()).hexdigest()[:16]
