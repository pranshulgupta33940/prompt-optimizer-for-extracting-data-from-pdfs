"""Unit tests for Hungarian array alignment."""

import pytest

from src.data.schema import FieldEvalConfig
from src.scoring.alignment import align_arrays, compute_item_similarity


class TestAlignArrays:
    """Tests for the Hungarian alignment function."""

    def test_both_empty(self):
        result = align_arrays([], [])
        assert result == []

    def test_pred_empty(self):
        result = align_arrays([], ["a", "b"])
        assert len(result) == 2
        assert all(p == -1 for p, g, s in result)

    def test_gold_empty(self):
        result = align_arrays(["a", "b"], [])
        assert len(result) == 2
        assert all(g == -1 for p, g, s in result)

    def test_perfect_match(self):
        result = align_arrays(["a", "b", "c"], ["a", "b", "c"])
        matched = [(p, g, s) for p, g, s in result if p >= 0 and g >= 0]
        assert len(matched) == 3
        assert all(s == 1.0 for _, _, s in matched)

    def test_partial_overlap(self):
        result = align_arrays(["a", "x"], ["a", "b"])
        matched = [(p, g, s) for p, g, s in result if p >= 0 and g >= 0]
        assert len(matched) == 2
        scores = {s for _, _, s in matched}
        assert 1.0 in scores  # "a" matches perfectly

    def test_unequal_lengths_more_pred(self):
        result = align_arrays(["a", "b", "c", "d"], ["a", "b"])
        matched = [(p, g) for p, g, _ in result if p >= 0 and g >= 0]
        unmatched_pred = [(p, g) for p, g, _ in result if g == -1]
        assert len(matched) == 2
        assert len(unmatched_pred) == 2

    def test_unequal_lengths_more_gold(self):
        result = align_arrays(["a"], ["a", "b", "c"])
        matched = [(p, g) for p, g, _ in result if p >= 0 and g >= 0]
        unmatched_gold = [(p, g) for p, g, _ in result if p == -1]
        assert len(matched) == 1
        assert len(unmatched_gold) == 2


class TestAlignDictArrays:
    """Tests for alignment of arrays of objects."""

    def test_dict_perfect_match(self):
        pred = [{"name": "Alice", "age": "30"}, {"name": "Bob", "age": "25"}]
        gold = [{"name": "Alice", "age": "30"}, {"name": "Bob", "age": "25"}]

        configs = {
            "name": FieldEvalConfig(metric_id="string_exact"),
            "age": FieldEvalConfig(metric_id="string_exact"),
        }

        result = align_arrays(pred, gold, field_configs=configs)
        matched = [(p, g, s) for p, g, s in result if p >= 0 and g >= 0]
        assert len(matched) == 2
        assert all(s == 1.0 for _, _, s in matched)

    def test_dict_swapped_order(self):
        """Hungarian should find optimal assignment even if order differs."""
        pred = [{"name": "Bob"}, {"name": "Alice"}]
        gold = [{"name": "Alice"}, {"name": "Bob"}]

        configs = {"name": FieldEvalConfig(metric_id="string_exact")}

        result = align_arrays(pred, gold, field_configs=configs)
        matched = [(p, g, s) for p, g, s in result if p >= 0 and g >= 0]
        assert len(matched) == 2
        # Both should match perfectly (Hungarian finds optimal pairing)
        assert all(s == 1.0 for _, _, s in matched)


class TestComputeItemSimilarity:
    """Tests for the item similarity function."""

    def test_primitive_match(self):
        sim = compute_item_similarity("hello", "hello")
        assert sim == 1.0

    def test_primitive_mismatch(self):
        sim = compute_item_similarity("hello", "world")
        assert sim < 1.0

    def test_dict_with_configs(self):
        configs = {"name": FieldEvalConfig(metric_id="string_exact")}
        sim = compute_item_similarity(
            {"name": "Alice"}, {"name": "Alice"}, field_configs=configs,
        )
        assert sim == 1.0

    def test_dict_without_configs(self):
        """Falls back to key-overlap fuzzy matching."""
        sim = compute_item_similarity(
            {"name": "Alice"}, {"name": "Alice"},
        )
        assert sim > 0.0
