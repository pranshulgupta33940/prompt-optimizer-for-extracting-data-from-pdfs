"""Individual metric implementations for all evaluation_config types.

Every public metric function follows the same signature:
    (predicted, gold, **params) -> tuple[float, str]
where the float is a score in [0, 1] and the str explains the outcome.
"""

from difflib import SequenceMatcher
from typing import Any, Callable

from src.data.schema import FieldEvalConfig


# ---------------------------------------------------------------------------
#  String metrics
# ---------------------------------------------------------------------------

def string_exact(predicted: Any, gold: Any, **_: Any) -> tuple[float, str]:
    """Case-sensitive exact string match."""
    pred_str = _to_str(predicted)
    gold_str = _to_str(gold)

    if pred_str is None and gold_str is None:
        return 1.0, "Both null"
    if pred_str is None or gold_str is None:
        return 0.0, f"Null mismatch: pred={pred_str!r}, gold={gold_str!r}"
    if pred_str == gold_str:
        return 1.0, "Exact match"
    return 0.0, f"Mismatch: '{pred_str}' != '{gold_str}'"


def string_semantic(
    predicted: Any,
    gold: Any,
    llm_judge_fn: Callable | None = None,
    **_: Any,
) -> tuple[float, str]:
    """LLM-based semantic similarity; falls back to fuzzy matching."""
    pred_str = _to_str(predicted)
    gold_str = _to_str(gold)

    if pred_str is None and gold_str is None:
        return 1.0, "Both null"
    if pred_str is None or gold_str is None:
        return 0.0, f"Null mismatch: pred={pred_str!r}, gold={gold_str!r}"
    if pred_str == gold_str:
        return 1.0, "Exact match (semantic shortcut)"

    if llm_judge_fn is not None:
        return llm_judge_fn(pred_str, gold_str)

    return string_fuzzy(predicted, gold)


def string_fuzzy(predicted: Any, gold: Any, **_: Any) -> tuple[float, str]:
    """Normalised Levenshtein similarity via ``SequenceMatcher``."""
    pred_str = _to_str(predicted)
    gold_str = _to_str(gold)

    if pred_str is None and gold_str is None:
        return 1.0, "Both null"
    if pred_str is None or gold_str is None:
        return 0.0, f"Null mismatch: pred={pred_str!r}, gold={gold_str!r}"

    ratio = SequenceMatcher(None, pred_str, gold_str).ratio()
    return ratio, f"Fuzzy ratio={ratio:.3f}"


def string_case_insensitive(
    predicted: Any, gold: Any, **_: Any,
) -> tuple[float, str]:
    """Case-insensitive exact string match."""
    pred_str = _to_str(predicted)
    gold_str = _to_str(gold)

    if pred_str is None and gold_str is None:
        return 1.0, "Both null"
    if pred_str is None or gold_str is None:
        return 0.0, f"Null mismatch: pred={pred_str!r}, gold={gold_str!r}"

    if pred_str.lower() == gold_str.lower():
        return 1.0, "Case-insensitive match"
    return 0.0, f"Mismatch (ci): '{pred_str}' != '{gold_str}'"


# ---------------------------------------------------------------------------
#  Numeric metrics
# ---------------------------------------------------------------------------

def integer_exact(predicted: Any, gold: Any, **_: Any) -> tuple[float, str]:
    """Exact integer match after coercion."""
    pred_int = _to_int(predicted)
    gold_int = _to_int(gold)

    if pred_int is None and gold_int is None:
        return 1.0, "Both null"
    if pred_int is None or gold_int is None:
        return 0.0, f"Null/type mismatch: pred={predicted!r}, gold={gold!r}"
    if pred_int == gold_int:
        return 1.0, f"Exact integer match: {pred_int}"
    return 0.0, f"Integer mismatch: {pred_int} != {gold_int}"


def number_tolerance(
    predicted: Any,
    gold: Any,
    tolerance: float = 0.001,
    **_: Any,
) -> tuple[float, str]:
    """Match within a relative tolerance.

    Passes when ``|pred - gold| / max(|gold|, 1e-9) <= tolerance``.
    """
    pred_num = _to_float(predicted)
    gold_num = _to_float(gold)

    if pred_num is None and gold_num is None:
        return 1.0, "Both null"
    if pred_num is None or gold_num is None:
        return 0.0, f"Null/type mismatch: pred={predicted!r}, gold={gold!r}"

    denom = max(abs(gold_num), 1e-9)
    rel_err = abs(pred_num - gold_num) / denom

    if rel_err <= tolerance:
        return 1.0, f"Within tolerance: err={rel_err:.6f} <= {tolerance}"
    return 0.0, f"Outside tolerance: err={rel_err:.6f} > {tolerance}"


def number_exact(predicted: Any, gold: Any, **_: Any) -> tuple[float, str]:
    """Exact numeric match after float coercion."""
    pred_num = _to_float(predicted)
    gold_num = _to_float(gold)

    if pred_num is None and gold_num is None:
        return 1.0, "Both null"
    if pred_num is None or gold_num is None:
        return 0.0, f"Null/type mismatch: pred={predicted!r}, gold={gold!r}"
    if pred_num == gold_num:
        return 1.0, f"Exact number match: {pred_num}"
    return 0.0, f"Number mismatch: {pred_num} != {gold_num}"


# ---------------------------------------------------------------------------
#  Boolean metric
# ---------------------------------------------------------------------------

def boolean_exact(predicted: Any, gold: Any, **_: Any) -> tuple[float, str]:
    """Exact boolean match after coercion."""
    pred_bool = _to_bool(predicted)
    gold_bool = _to_bool(gold)

    if pred_bool is None and gold_bool is None:
        return 1.0, "Both null"
    if pred_bool is None or gold_bool is None:
        return 0.0, f"Null/type mismatch: pred={predicted!r}, gold={gold!r}"
    if pred_bool == gold_bool:
        return 1.0, f"Boolean match: {pred_bool}"
    return 0.0, f"Boolean mismatch: {pred_bool} != {gold_bool}"


# ---------------------------------------------------------------------------
#  Array metric
# ---------------------------------------------------------------------------

def array_llm(
    predicted: Any,
    gold: Any,
    llm_judge_fn: Callable | None = None,
    **_: Any,
) -> tuple[float, str]:
    """LLM-as-judge for array-level comparison.

    Falls back to set-overlap heuristic when no judge function is provided.
    """
    pred_list = predicted if isinstance(predicted, list) else []
    gold_list = gold if isinstance(gold, list) else []

    if not gold_list and not pred_list:
        return 1.0, "Both empty arrays"
    if not gold_list:
        return 0.0, f"Gold is empty but predicted has {len(pred_list)} items"
    if not pred_list:
        return 0.0, f"Predicted is empty but gold has {len(gold_list)} items"

    if llm_judge_fn is not None:
        return llm_judge_fn(pred_list, gold_list)

    return _array_set_overlap(pred_list, gold_list)


# ---------------------------------------------------------------------------
#  Dispatcher
# ---------------------------------------------------------------------------

METRIC_REGISTRY: dict[str, Callable[..., tuple[float, str]]] = {
    "string_exact": string_exact,
    "string_semantic": string_semantic,
    "string_fuzzy": string_fuzzy,
    "string_case_insensitive": string_case_insensitive,
    "integer_exact": integer_exact,
    "number_tolerance": number_tolerance,
    "number_exact": number_exact,
    "boolean_exact": boolean_exact,
    "array_llm": array_llm,
}


def evaluate_field(
    predicted: Any,
    gold: Any,
    eval_config: FieldEvalConfig,
    llm_judge_fn: Callable | None = None,
) -> tuple[float, str]:
    """Dispatch to the appropriate metric for a single field.

    Args:
        predicted: The predicted value.
        gold: The gold (expected) value.
        eval_config: Specifies which metric to use and its parameters.
        llm_judge_fn: Optional callable for LLM-based metrics.

    Returns:
        ``(score, reason)`` tuple.
    """
    metric_fn = METRIC_REGISTRY.get(eval_config.metric_id)
    if metric_fn is None:
        return 0.0, f"Unknown metric: {eval_config.metric_id}"

    params = dict(eval_config.params) if eval_config.params else {}
    params["llm_judge_fn"] = llm_judge_fn

    return metric_fn(predicted, gold, **params)


# ---------------------------------------------------------------------------
#  Type-coercion helpers
# ---------------------------------------------------------------------------

def _to_str(value: Any) -> str | None:
    """Coerce a value to a string, treating None as None."""
    if value is None:
        return None
    return str(value).strip()


def _to_int(value: Any) -> int | None:
    """Coerce a value to an integer, returning None on failure."""
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> float | None:
    """Coerce a value to a float, returning None on failure."""
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _to_bool(value: Any) -> bool | None:
    """Coerce a value to a boolean, returning None on failure."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        low = value.strip().lower()
        if low in ("true", "1", "yes"):
            return True
        if low in ("false", "0", "no"):
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return None


def _array_set_overlap(
    predicted: list[Any], gold: list[Any],
) -> tuple[float, str]:
    """Simple set-overlap fallback for array comparison."""
    pred_strs = [str(item).strip().lower() for item in predicted]
    gold_strs = [str(item).strip().lower() for item in gold]

    matched = 0
    used_pred: set[int] = set()

    for g_str in gold_strs:
        for p_idx, p_str in enumerate(pred_strs):
            if p_idx not in used_pred and p_str == g_str:
                matched += 1
                used_pred.add(p_idx)
                break

    total = max(len(gold_strs), 1)
    score_val = matched / total
    reason = f"Set overlap: {matched}/{len(gold_strs)} gold matched"
    return score_val, reason
