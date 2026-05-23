"""Prompt diff utility — unified diff between iterations.

Usage::

    from src.observability.diff import prompt_diff
    print(prompt_diff(state, iteration_a=0, iteration_b=3))
"""

import difflib
from typing import Any

from src.optimizer.state import RunState


def prompt_diff(
    state: RunState,
    iteration_a: int = 0,
    iteration_b: int | None = None,
) -> str:
    """Generate a unified diff between two iterations' prompts.

    Args:
        state: The run state containing prompt history.
        iteration_a: The earlier iteration number (default: 0 = seed).
        iteration_b: The later iteration number (default: last accepted).

    Returns:
        A unified-diff string.
    """
    prompt_a = _get_prompt(state, iteration_a)
    if iteration_b is None:
        prompt_b = state.current_best_prompt
        iteration_b = state.current_iteration
    else:
        prompt_b = _get_prompt(state, iteration_b)

    if prompt_a is None:
        return f"[ERROR] No prompt found for iteration {iteration_a}"
    if prompt_b is None:
        return f"[ERROR] No prompt found for iteration {iteration_b}"

    lines_a = prompt_a.splitlines(keepends=True)
    lines_b = prompt_b.splitlines(keepends=True)

    diff = difflib.unified_diff(
        lines_a,
        lines_b,
        fromfile=f"iteration_{iteration_a}",
        tofile=f"iteration_{iteration_b}",
        lineterm="",
    )

    return "\n".join(diff)


def prompt_diff_summary(state: RunState) -> str:
    """One-line summary of how the prompt changed from seed to best.

    Args:
        state: The run state.

    Returns:
        A short summary string.
    """
    seed = state.seed_prompt
    best = state.current_best_prompt

    if seed == best:
        return "No changes — seed prompt is still the best."

    seed_lines = seed.strip().splitlines()
    best_lines = best.strip().splitlines()

    added = sum(1 for l in best_lines if l not in seed_lines)
    removed = sum(1 for l in seed_lines if l not in best_lines)

    return (
        f"Prompt changed: +{added} lines, -{removed} lines "
        f"(seed: {len(seed_lines)} → best: {len(best_lines)} lines)"
    )


def _get_prompt(state: RunState, iteration: int) -> str | None:
    """Retrieve the prompt text for a given iteration number."""
    for record in state.prompt_history:
        if record.iteration == iteration:
            return record.prompt
    if iteration == 0:
        return state.seed_prompt
    return None
