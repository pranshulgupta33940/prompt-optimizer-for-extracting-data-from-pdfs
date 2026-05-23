"""Scoring module for evaluating structured extractions against gold annotations."""

from src.scoring.scorer import FieldScore, ScoringResult, score

__all__ = [
    "FieldScore",
    "ScoringResult",
    "score",
]
