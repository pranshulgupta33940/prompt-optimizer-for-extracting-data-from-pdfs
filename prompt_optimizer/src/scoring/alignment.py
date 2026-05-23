"""Array alignment using the Hungarian algorithm (optimal bipartite matching).

Policy: Hungarian (Munkres) algorithm via ``scipy.optimize.linear_sum_assignment``
provides the *globally optimal* one-to-one assignment between predicted and gold
array items, minimising total cost (i.e. maximising total similarity).

For arrays of objects, pairwise similarity is the average field-level score
across all evaluation-config-bearing fields.  For arrays of primitives,
similarity is computed via the field's own metric.
"""

from typing import Any, Callable

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.data.schema import FieldEvalConfig
from src.scoring.metrics import evaluate_field, string_fuzzy


def align_arrays(
    predicted: list[Any],
    gold: list[Any],
    field_configs: dict[str, FieldEvalConfig] | None = None,
    llm_judge_fn: Callable | None = None,
) -> list[tuple[int, int, float]]:
    """Align predicted items to gold items via Hungarian algorithm.

    Args:
        predicted: Predicted array items.
        gold: Gold array items.
        field_configs: Eval configs for object-item fields (dotted paths
            relative to the array item).  ``None`` for primitive arrays.
        llm_judge_fn: Optional LLM judge callable.

    Returns:
        List of ``(pred_idx, gold_idx, similarity)`` triples.
        Unmatched predicted items get ``gold_idx = -1``.
        Unmatched gold items get ``pred_idx = -1``.
    """
    n_pred = len(predicted)
    n_gold = len(gold)

    if n_pred == 0 and n_gold == 0:
        return []
    if n_pred == 0:
        return [(-1, g, 0.0) for g in range(n_gold)]
    if n_gold == 0:
        return [(p, -1, 0.0) for p in range(n_pred)]

    cost_matrix = _build_cost_matrix(
        predicted, gold, field_configs, llm_judge_fn,
    )

    row_idx, col_idx = linear_sum_assignment(cost_matrix)

    pairs: list[tuple[int, int, float]] = []
    matched_pred: set[int] = set()
    matched_gold: set[int] = set()

    for r, c in zip(row_idx, col_idx):
        if r < n_pred and c < n_gold:
            sim = 1.0 - cost_matrix[r, c]
            pairs.append((r, c, sim))
            matched_pred.add(r)
            matched_gold.add(c)

    for p in range(n_pred):
        if p not in matched_pred:
            pairs.append((p, -1, 0.0))

    for g in range(n_gold):
        if g not in matched_gold:
            pairs.append((-1, g, 0.0))

    return pairs


def compute_item_similarity(
    predicted: Any,
    gold: Any,
    field_configs: dict[str, FieldEvalConfig] | None = None,
    llm_judge_fn: Callable | None = None,
) -> float:
    """Compute similarity between two array items.

    For dicts: average score across all configured fields.
    For primitives: use fuzzy string matching as default.

    Args:
        predicted: A single predicted item.
        gold: A single gold item.
        field_configs: Eval configs keyed by field name (for dicts).
        llm_judge_fn: Optional LLM judge callable.

    Returns:
        Similarity score in ``[0, 1]``.
    """
    if isinstance(predicted, dict) and isinstance(gold, dict):
        return _dict_similarity(predicted, gold, field_configs, llm_judge_fn)

    score, _ = string_fuzzy(predicted, gold)
    return score


# ---------------------------------------------------------------------------
#  Internal helpers
# ---------------------------------------------------------------------------

def _build_cost_matrix(
    predicted: list[Any],
    gold: list[Any],
    field_configs: dict[str, FieldEvalConfig] | None,
    llm_judge_fn: Callable | None,
) -> np.ndarray:
    """Build a cost matrix for the Hungarian algorithm.

    Uses padding so the matrix is square when arrays differ in length.
    Padding cells get cost 1.0 (zero similarity).
    """
    n_pred = len(predicted)
    n_gold = len(gold)
    size = max(n_pred, n_gold)

    cost = np.ones((size, size), dtype=float)

    for p_idx in range(n_pred):
        for g_idx in range(n_gold):
            sim = compute_item_similarity(
                predicted[p_idx], gold[g_idx], field_configs, llm_judge_fn,
            )
            cost[p_idx, g_idx] = 1.0 - sim

    return cost


def _dict_similarity(
    predicted: dict,
    gold: dict,
    field_configs: dict[str, FieldEvalConfig] | None,
    llm_judge_fn: Callable | None,
) -> float:
    """Average field score between two dicts."""
    if not field_configs:
        return _fallback_dict_similarity(predicted, gold)

    scores: list[float] = []
    for field_path, config in field_configs.items():
        parts = field_path.split(".")
        pred_val = _nested_get(predicted, parts)
        gold_val = _nested_get(gold, parts)
        field_score, _ = evaluate_field(
            pred_val, gold_val, config, llm_judge_fn,
        )
        scores.append(field_score)

    if not scores:
        return _fallback_dict_similarity(predicted, gold)

    return sum(scores) / len(scores)


def _fallback_dict_similarity(predicted: dict, gold: dict) -> float:
    """Fallback similarity using shared-key string comparison."""
    all_keys = set(predicted.keys()) | set(gold.keys())
    if not all_keys:
        return 1.0

    matches = 0
    for key in all_keys:
        if key in predicted and key in gold:
            score, _ = string_fuzzy(predicted[key], gold[key])
            if score >= 0.8:
                matches += 1

    return matches / len(all_keys)


def _nested_get(data: dict, parts: list[str]) -> Any:
    """Traverse a nested dict by a list of keys."""
    current: Any = data
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current
