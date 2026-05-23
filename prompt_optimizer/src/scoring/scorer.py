"""Main scoring orchestrator — the public ``score()`` entry point.

Fully independent of the optimisation loop.  Import and call as::

    from src.scoring.scorer import score, ScoringResult
    result = score(predicted, gold, schema)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from src.data.schema import (
    FieldEvalConfig,
    extract_eval_configs,
    resolve_refs,
)
from src.scoring.alignment import align_arrays, compute_item_similarity
from src.scoring.cache import MetricCache
from src.scoring.metrics import evaluate_field


# ---------------------------------------------------------------------------
#  Data classes
# ---------------------------------------------------------------------------

@dataclass
class FieldScore:
    """Evaluation result for a single leaf field."""

    field_path: str
    metric_type: str
    score: float
    passed: bool
    gold_value: Any = field(repr=False)
    predicted_value: Any = field(repr=False)
    reason: str = ""


@dataclass
class ScoringResult:
    """Aggregate result of scoring an entire predicted dict against gold."""

    aggregate_score: float
    aggregate_precision: float
    aggregate_recall: float
    aggregate_f1: float
    field_scores: dict[str, FieldScore]
    subtree_scores: dict[str, float]
    total_fields: int
    matched_fields: int
    missing_fields: int
    extra_fields: int


# ---------------------------------------------------------------------------
#  Public entry point
# ---------------------------------------------------------------------------

def score(
    predicted: dict,
    gold: dict,
    schema: dict,
    cache_dir: Path | None = None,
    llm_judge_fn: Callable | None = None,
) -> ScoringResult:
    """Score a predicted extraction against a gold annotation.

    This function is **fully independent** of the optimisation loop and
    can be imported and called standalone.

    Args:
        predicted: The model-predicted JSON dict.
        gold: The gold (expected) JSON dict.
        schema: The raw JSON schema dict (may contain ``$ref``).
        cache_dir: Optional directory for the metric cache DB.
        llm_judge_fn: Optional LLM judge callable for semantic metrics.

    Returns:
        A ``ScoringResult`` with per-field and aggregate scores.
    """
    resolved = resolve_refs(schema)

    working_schema = resolved
    if "schema_definition" in resolved:
        working_schema = resolved["schema_definition"]

    field_configs = extract_eval_configs(working_schema)

    metric_cache: MetricCache | None = None
    if cache_dir is not None:
        metric_cache = MetricCache(cache_dir / "metric_cache.db")

    cached_judge = _wrap_with_cache(llm_judge_fn, metric_cache)

    field_scores = _walk_and_score(
        predicted, gold, working_schema, field_configs, cached_judge, "",
    )

    if metric_cache is not None:
        metric_cache.close()

    return _aggregate(field_scores, predicted, gold)


# ---------------------------------------------------------------------------
#  Tree walker
# ---------------------------------------------------------------------------

def _walk_and_score(
    predicted: Any,
    gold: Any,
    schema: dict,
    field_configs: dict[str, FieldEvalConfig],
    llm_judge_fn: Callable | None,
    prefix: str,
) -> dict[str, FieldScore]:
    """Recursively walk the schema tree and score every leaf field."""
    results: dict[str, FieldScore] = {}

    props = schema.get("properties", {})
    if props:
        results.update(
            _score_object_fields(
                predicted, gold, props, field_configs, llm_judge_fn, prefix,
            )
        )
        return results

    if schema.get("type") == "array" or "items" in schema:
        results.update(
            _score_array(
                predicted, gold, schema, field_configs, llm_judge_fn, prefix,
            )
        )
        return results

    for union_key in ("anyOf", "oneOf"):
        if union_key in schema:
            for variant in schema[union_key]:
                if isinstance(variant, dict) and variant.get("type") != "null":
                    if "properties" in variant or "items" in variant:
                        results.update(
                            _walk_and_score(
                                predicted, gold, variant,
                                field_configs, llm_judge_fn, prefix,
                            )
                        )
                        return results

    path = prefix
    if path and path in field_configs:
        config = field_configs[path]
        sc, reason = evaluate_field(predicted, gold, config, llm_judge_fn)
        results[path] = FieldScore(
            field_path=path,
            metric_type=config.metric_id,
            score=sc,
            passed=sc >= _pass_threshold(config.metric_id),
            gold_value=gold,
            predicted_value=predicted,
            reason=reason,
        )

    return results


def _score_object_fields(
    predicted: Any,
    gold: Any,
    properties: dict,
    field_configs: dict[str, FieldEvalConfig],
    llm_judge_fn: Callable | None,
    prefix: str,
) -> dict[str, FieldScore]:
    """Score each property of a JSON object."""
    results: dict[str, FieldScore] = {}
    pred_dict = predicted if isinstance(predicted, dict) else {}
    gold_dict = gold if isinstance(gold, dict) else {}

    for prop_name, prop_schema in properties.items():
        child_prefix = f"{prefix}.{prop_name}" if prefix else prop_name
        pred_val = pred_dict.get(prop_name)
        gold_val = gold_dict.get(prop_name)

        child_results = _walk_and_score(
            pred_val, gold_val, prop_schema,
            field_configs, llm_judge_fn, child_prefix,
        )

        if child_results:
            results.update(child_results)
        elif child_prefix in field_configs:
            config = field_configs[child_prefix]
            sc, reason = evaluate_field(
                pred_val, gold_val, config, llm_judge_fn,
            )
            results[child_prefix] = FieldScore(
                field_path=child_prefix,
                metric_type=config.metric_id,
                score=sc,
                passed=sc >= _pass_threshold(config.metric_id),
                gold_value=gold_val,
                predicted_value=pred_val,
                reason=reason,
            )

    return results


def _score_array(
    predicted: Any,
    gold: Any,
    schema: dict,
    field_configs: dict[str, FieldEvalConfig],
    llm_judge_fn: Callable | None,
    prefix: str,
) -> dict[str, FieldScore]:
    """Score an array field — either via LLM judge or Hungarian alignment."""
    results: dict[str, FieldScore] = {}
    pred_list = predicted if isinstance(predicted, list) else []
    gold_list = gold if isinstance(gold, list) else []

    if prefix in field_configs:
        config = field_configs[prefix]
        if config.metric_id == "array_llm":
            sc, reason = evaluate_field(
                pred_list, gold_list, config, llm_judge_fn,
            )
            results[prefix] = FieldScore(
                field_path=prefix,
                metric_type="array_llm",
                score=sc,
                passed=sc >= 0.5,
                gold_value=gold_list,
                predicted_value=pred_list,
                reason=reason,
            )
            return results

    items_schema = schema.get("items", {})
    item_configs = _get_item_field_configs(field_configs, prefix)

    if not gold_list and not pred_list:
        return results

    alignment = align_arrays(pred_list, gold_list, item_configs, llm_judge_fn)

    pair_scores: list[float] = []
    for pred_idx, gold_idx, sim in alignment:
        if pred_idx >= 0 and gold_idx >= 0:
            item_prefix = f"{prefix}[{gold_idx}]"

            if isinstance(gold_list[gold_idx], dict) and "properties" in items_schema:
                item_results = _walk_and_score(
                    pred_list[pred_idx], gold_list[gold_idx],
                    items_schema, field_configs, llm_judge_fn,
                    f"{prefix}[]",
                )
                for fp, fs in item_results.items():
                    keyed = fp.replace("[]", f"[{gold_idx}]")
                    results[keyed] = fs
                    pair_scores.append(fs.score)
            else:
                pair_scores.append(sim)
        elif gold_idx >= 0:
            pair_scores.append(0.0)

    if pair_scores and prefix:
        avg = sum(pair_scores) / len(pair_scores)
        results[f"{prefix}._array_avg"] = FieldScore(
            field_path=f"{prefix}._array_avg",
            metric_type="array_alignment",
            score=avg,
            passed=avg >= 0.5,
            gold_value=f"{len(gold_list)} items",
            predicted_value=f"{len(pred_list)} items",
            reason=f"Hungarian alignment avg={avg:.3f}",
        )

    return results


# ---------------------------------------------------------------------------
#  Aggregation
# ---------------------------------------------------------------------------

def _aggregate(
    field_scores: dict[str, FieldScore],
    predicted: Any,
    gold: Any,
) -> ScoringResult:
    """Compute aggregate metrics from per-field scores."""
    if not field_scores:
        return ScoringResult(
            aggregate_score=0.0,
            aggregate_precision=0.0,
            aggregate_recall=0.0,
            aggregate_f1=0.0,
            field_scores={},
            subtree_scores={},
            total_fields=0,
            matched_fields=0,
            missing_fields=0,
            extra_fields=0,
        )

    scores = [fs.score for fs in field_scores.values()]
    agg_score = sum(scores) / len(scores)

    passed = sum(1 for fs in field_scores.values() if fs.passed)
    total = len(field_scores)

    gold_fields = _count_leaf_fields(gold) if isinstance(gold, dict) else total
    pred_fields = _count_leaf_fields(predicted) if isinstance(predicted, dict) else total

    precision = passed / max(pred_fields, 1)
    recall = passed / max(gold_fields, 1)
    f1 = _harmonic_mean(precision, recall)

    subtree_scores = _compute_subtree_scores(field_scores)

    missing = sum(
        1 for fs in field_scores.values()
        if fs.predicted_value is None and fs.gold_value is not None
    )
    extra = max(0, pred_fields - gold_fields)

    return ScoringResult(
        aggregate_score=agg_score,
        aggregate_precision=precision,
        aggregate_recall=recall,
        aggregate_f1=f1,
        field_scores=field_scores,
        subtree_scores=subtree_scores,
        total_fields=total,
        matched_fields=passed,
        missing_fields=missing,
        extra_fields=extra,
    )


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _pass_threshold(metric_id: str) -> float:
    """Return the pass/fail threshold for a given metric type."""
    fuzzy_metrics = {"string_fuzzy", "string_semantic"}
    if metric_id in fuzzy_metrics:
        return 0.8
    return 1.0


def _compute_subtree_scores(
    field_scores: dict[str, FieldScore],
) -> dict[str, float]:
    """Group field scores by their top-level key and average each group."""
    groups: dict[str, list[float]] = {}
    for path, fs in field_scores.items():
        top_key = path.split(".")[0].split("[")[0]
        groups.setdefault(top_key, []).append(fs.score)

    return {
        key: sum(vals) / len(vals) for key, vals in groups.items() if vals
    }


def _harmonic_mean(a: float, b: float) -> float:
    """Compute harmonic mean of two values, returning 0 if either is 0."""
    if a + b == 0:
        return 0.0
    return 2 * a * b / (a + b)


def _count_leaf_fields(data: Any, depth: int = 0) -> int:
    """Count leaf (non-dict, non-list) values in a nested structure."""
    if depth > 20:
        return 1
    if isinstance(data, dict):
        total = 0
        for val in data.values():
            total += _count_leaf_fields(val, depth + 1)
        return max(total, 1)
    if isinstance(data, list):
        total = 0
        for item in data:
            total += _count_leaf_fields(item, depth + 1)
        return max(total, 1)
    return 1


def _get_item_field_configs(
    field_configs: dict[str, FieldEvalConfig],
    array_prefix: str,
) -> dict[str, FieldEvalConfig] | None:
    """Extract eval configs for array-item fields.

    Given configs like ``authors[].name`` and ``array_prefix='authors'``,
    returns ``{'name': FieldEvalConfig(...)}``.
    """
    item_prefix = f"{array_prefix}[]."
    item_configs: dict[str, FieldEvalConfig] = {}

    for path, config in field_configs.items():
        if path.startswith(item_prefix):
            relative = path[len(item_prefix):]
            item_configs[relative] = config

    return item_configs if item_configs else None


def _wrap_with_cache(
    llm_judge_fn: Callable | None,
    cache: MetricCache | None,
) -> Callable | None:
    """Wrap an LLM judge function with caching if both are provided."""
    if llm_judge_fn is None or cache is None:
        return llm_judge_fn

    def cached_judge(predicted: Any, gold: Any) -> tuple[float, str]:
        metric_id = "llm_judge"
        cached = cache.get(metric_id, predicted, gold)
        if cached is not None:
            return cached
        result = llm_judge_fn(predicted, gold)
        cache.put(metric_id, predicted, gold, result[0], result[1])
        return result

    return cached_judge
