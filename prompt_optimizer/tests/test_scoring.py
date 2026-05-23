"""Unit tests for the scoring module — all 9 metric types + aggregation."""

import pytest
from pathlib import Path

from src.data.schema import FieldEvalConfig
from src.scoring.metrics import (
    array_llm,
    boolean_exact,
    evaluate_field,
    integer_exact,
    number_exact,
    number_tolerance,
    string_case_insensitive,
    string_exact,
    string_fuzzy,
    string_semantic,
)
from src.scoring.scorer import score


# ============================================================================
#  string_exact
# ============================================================================

class TestStringExact:
    def test_exact_match(self):
        s, _ = string_exact("hello", "hello")
        assert s == 1.0

    def test_case_mismatch(self):
        s, _ = string_exact("Hello", "hello")
        assert s == 0.0

    def test_both_none(self):
        s, _ = string_exact(None, None)
        assert s == 1.0

    def test_pred_none(self):
        s, _ = string_exact(None, "hello")
        assert s == 0.0

    def test_gold_none(self):
        s, _ = string_exact("hello", None)
        assert s == 0.0

    def test_numeric_as_string(self):
        s, _ = string_exact(42, "42")
        assert s == 1.0

    def test_empty_strings(self):
        s, _ = string_exact("", "")
        assert s == 1.0

    def test_whitespace_handling(self):
        s, _ = string_exact("  hello  ", "hello")
        assert s == 1.0


# ============================================================================
#  string_fuzzy
# ============================================================================

class TestStringFuzzy:
    def test_exact(self):
        s, _ = string_fuzzy("hello world", "hello world")
        assert s == 1.0

    def test_minor_diff(self):
        s, _ = string_fuzzy("hello world", "hello worl")
        assert s > 0.8

    def test_totally_different(self):
        s, _ = string_fuzzy("abc", "xyz")
        assert s < 0.5

    def test_both_none(self):
        s, _ = string_fuzzy(None, None)
        assert s == 1.0


# ============================================================================
#  string_case_insensitive
# ============================================================================

class TestStringCaseInsensitive:
    def test_same_case(self):
        s, _ = string_case_insensitive("hello", "hello")
        assert s == 1.0

    def test_different_case(self):
        s, _ = string_case_insensitive("HELLO", "hello")
        assert s == 1.0

    def test_mismatch(self):
        s, _ = string_case_insensitive("hello", "world")
        assert s == 0.0


# ============================================================================
#  string_semantic
# ============================================================================

class TestStringSemantic:
    def test_exact_shortcut(self):
        s, r = string_semantic("hello", "hello")
        assert s == 1.0
        assert "shortcut" in r.lower() or "exact" in r.lower()

    def test_fallback_to_fuzzy(self):
        """Without LLM judge, falls back to fuzzy matching."""
        s, _ = string_semantic("hello world", "hello worl")
        assert s > 0.8

    def test_with_judge(self):
        judge = lambda p, g: (0.9, "semantically similar")
        s, r = string_semantic("cat", "feline", llm_judge_fn=judge)
        assert s == 0.9


# ============================================================================
#  integer_exact
# ============================================================================

class TestIntegerExact:
    def test_equal(self):
        s, _ = integer_exact(42, 42)
        assert s == 1.0

    def test_not_equal(self):
        s, _ = integer_exact(42, 43)
        assert s == 0.0

    def test_string_coercion(self):
        s, _ = integer_exact("42", 42)
        assert s == 1.0

    def test_float_coercion(self):
        s, _ = integer_exact(42.0, 42)
        assert s == 1.0

    def test_both_none(self):
        s, _ = integer_exact(None, None)
        assert s == 1.0

    def test_string_float_coercion(self):
        s, _ = integer_exact("42.7", 42)
        assert s == 1.0  # int(42.7) == 42


# ============================================================================
#  number_tolerance
# ============================================================================

class TestNumberTolerance:
    def test_exact(self):
        s, _ = number_tolerance(100.0, 100.0)
        assert s == 1.0

    def test_within_tolerance(self):
        s, _ = number_tolerance(100.05, 100.0, tolerance=0.001)
        assert s == 1.0  # 0.0005 <= 0.001

    def test_outside_tolerance(self):
        s, _ = number_tolerance(110.0, 100.0, tolerance=0.001)
        assert s == 0.0  # 0.1 > 0.001

    def test_zero_gold(self):
        s, _ = number_tolerance(0.0, 0.0)
        assert s == 1.0

    def test_string_coercion(self):
        s, _ = number_tolerance("100.0", 100.0)
        assert s == 1.0


# ============================================================================
#  number_exact
# ============================================================================

class TestNumberExact:
    def test_equal(self):
        s, _ = number_exact(3.14, 3.14)
        assert s == 1.0

    def test_not_equal(self):
        s, _ = number_exact(3.14, 3.15)
        assert s == 0.0


# ============================================================================
#  boolean_exact
# ============================================================================

class TestBooleanExact:
    def test_true_match(self):
        s, _ = boolean_exact(True, True)
        assert s == 1.0

    def test_false_match(self):
        s, _ = boolean_exact(False, False)
        assert s == 1.0

    def test_mismatch(self):
        s, _ = boolean_exact(True, False)
        assert s == 0.0

    def test_string_coercion(self):
        s, _ = boolean_exact("true", True)
        assert s == 1.0

    def test_int_coercion(self):
        s, _ = boolean_exact(1, True)
        assert s == 1.0


# ============================================================================
#  array_llm
# ============================================================================

class TestArrayLlm:
    def test_both_empty(self):
        s, _ = array_llm([], [])
        assert s == 1.0

    def test_pred_empty(self):
        s, _ = array_llm([], ["a", "b"])
        assert s == 0.0

    def test_gold_empty(self):
        s, _ = array_llm(["a"], [])
        assert s == 0.0

    def test_fallback_perfect_match(self):
        s, _ = array_llm(["a", "b"], ["a", "b"])
        assert s == 1.0

    def test_fallback_partial_match(self):
        s, _ = array_llm(["a", "c"], ["a", "b"])
        assert s == 0.5

    def test_with_judge(self):
        judge = lambda p, g: (0.75, "3/4 matched")
        s, r = array_llm(["a", "b"], ["a", "b", "c", "d"], llm_judge_fn=judge)
        assert s == 0.75


# ============================================================================
#  evaluate_field dispatcher
# ============================================================================

class TestEvaluateField:
    def test_string_exact_dispatch(self):
        cfg = FieldEvalConfig(metric_id="string_exact")
        s, _ = evaluate_field("hello", "hello", cfg)
        assert s == 1.0

    def test_number_tolerance_with_params(self):
        cfg = FieldEvalConfig(
            metric_id="number_tolerance",
            params={"tolerance": 0.01},
        )
        s, _ = evaluate_field(100.5, 100.0, cfg)
        assert s == 1.0  # 0.005 <= 0.01

    def test_unknown_metric(self):
        cfg = FieldEvalConfig(metric_id="unknown_metric")
        s, r = evaluate_field("a", "b", cfg)
        assert s == 0.0
        assert "unknown" in r.lower()


# ============================================================================
#  Full score() function
# ============================================================================

class TestScore:
    def test_simple_flat_schema(self):
        schema = {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "evaluation_config": "string_exact",
                },
                "age": {
                    "type": "integer",
                    "evaluation_config": "integer_exact",
                },
            },
        }
        predicted = {"name": "Alice", "age": 30}
        gold = {"name": "Alice", "age": 30}

        result = score(predicted, gold, schema)
        assert result.aggregate_score == 1.0
        assert result.matched_fields == result.total_fields

    def test_missing_field(self):
        schema = {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "evaluation_config": "string_exact",
                },
                "age": {
                    "type": "integer",
                    "evaluation_config": "integer_exact",
                },
            },
        }
        predicted = {"name": "Alice"}
        gold = {"name": "Alice", "age": 30}

        result = score(predicted, gold, schema)
        assert result.aggregate_score < 1.0

    def test_nested_object(self):
        schema = {
            "type": "object",
            "properties": {
                "address": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "evaluation_config": "string_exact",
                        },
                    },
                },
            },
        }
        predicted = {"address": {"city": "NYC"}}
        gold = {"address": {"city": "NYC"}}

        result = score(predicted, gold, schema)
        assert result.aggregate_score == 1.0

    def test_empty_prediction(self):
        schema = {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "evaluation_config": "string_exact",
                },
            },
        }
        result = score({}, {"title": "Hello"}, schema)
        assert result.aggregate_score == 0.0
