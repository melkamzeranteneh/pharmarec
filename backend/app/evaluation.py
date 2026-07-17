"""
PharmaRec - Evaluation Metrics

Centralized, reusable implementations of the recommender-system metrics used
across the project. Both the standalone evaluation and the HybridRecommender
comparison table rely on these functions so there is a single source of truth.

Metrics:
    - precision_at_k
    - recall_at_k
    - rmse
    - mae
    - coverage
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence


def precision_at_k(
    recommended: Iterable[str],
    relevant: Iterable[str],
    k: int = 10,
) -> float:
    """Precision@K.

    Fraction of the top-K recommended items that are relevant.

    Args:
        recommended: Ordered iterable of recommended item ids.
        relevant: Iterable of ground-truth relevant item ids.
        k: Cut-off rank.

    Returns:
        Precision@K in [0, 1].
    """
    if k <= 0:
        return 0.0
    top_k = list(recommended)[:k]
    if not top_k:
        return 0.0
    relevant_set = set(relevant)
    hits = sum(1 for item in top_k if item in relevant_set)
    return hits / k


def recall_at_k(
    recommended: Iterable[str],
    relevant: Iterable[str],
    k: int = 10,
) -> float:
    """Recall@K.

    Fraction of relevant items that appear in the top-K recommendations.

    Args:
        recommended: Ordered iterable of recommended item ids.
        relevant: Iterable of ground-truth relevant item ids.
        k: Cut-off rank.

    Returns:
        Recall@K in [0, 1]. Returns 0.0 when there are no relevant items.
    """
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    top_k = set(list(recommended)[:k])
    hits = len(top_k & relevant_set)
    return hits / len(relevant_set)


def rmse(predicted: Sequence[float], actual: Sequence[float]) -> float:
    """Root Mean Squared Error between predicted and actual ratings.

    Args:
        predicted: Predicted rating values.
        actual: Ground-truth rating values (same length as ``predicted``).

    Returns:
        RMSE value. Returns 0.0 for empty input.

    Raises:
        ValueError: If the two sequences have different lengths.
    """
    if len(predicted) != len(actual):
        raise ValueError("predicted and actual must have the same length")
    if not predicted:
        return 0.0
    total = sum((p - a) ** 2 for p, a in zip(predicted, actual))
    return math.sqrt(total / len(predicted))


def mae(predicted: Sequence[float], actual: Sequence[float]) -> float:
    """Mean Absolute Error between predicted and actual ratings.

    Args:
        predicted: Predicted rating values.
        actual: Ground-truth rating values (same length as ``predicted``).

    Returns:
        MAE value. Returns 0.0 for empty input.

    Raises:
        ValueError: If the two sequences have different lengths.
    """
    if len(predicted) != len(actual):
        raise ValueError("predicted and actual must have the same length")
    if not predicted:
        return 0.0
    total = sum(abs(p - a) for p, a in zip(predicted, actual))
    return total / len(predicted)


def coverage(recommended_items: Iterable[str], catalog: Iterable[str]) -> float:
    """Catalog coverage.

    Proportion of the total item catalog that appears at least once across
    all recommendations.

    Args:
        recommended_items: All item ids recommended across the evaluation.
        catalog: All item ids available in the dataset.

    Returns:
        Coverage in [0, 1]. Returns 0.0 for an empty catalog.
    """
    catalog_set = set(catalog)
    if not catalog_set:
        return 0.0
    recommended_set = set(recommended_items) & catalog_set
    return len(recommended_set) / len(catalog_set)
