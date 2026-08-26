"""Unit tests for the metric functions in src/evaluation/evaluate.py."""
from __future__ import annotations

import pandas as pd
import pytest

from src.evaluation import evaluate as ev

pytestmark = pytest.mark.unit


def test_spearman_rank_correlation_perfect_agreement() -> None:
    a = pd.Series([1.0, 2.0, 3.0, 4.0])
    b = pd.Series([10.0, 20.0, 30.0, 40.0])
    assert ev.spearman_rank_correlation(a, b) == pytest.approx(1.0)


def test_spearman_rank_correlation_zero_when_one_series_is_constant() -> None:
    a = pd.Series([1.0, 1.0, 1.0])
    b = pd.Series([1.0, 2.0, 3.0])
    assert ev.spearman_rank_correlation(a, b) == 0.0


def test_top_decile_lift_rewards_scores_that_concentrate_events_at_the_top() -> None:
    scores = pd.Series(range(20))
    events = pd.Series([1.0 if v >= 15 else 0.0 for v in range(20)])
    lift = ev.top_decile_lift(scores, events)
    assert lift == pytest.approx(4.0)


def test_top_decile_lift_zero_when_event_rate_is_zero() -> None:
    scores = pd.Series(range(10))
    events = pd.Series([0.0] * 10)
    assert ev.top_decile_lift(scores, events) == 0.0


def test_metrics_block_defines_events_as_top_quartile_of_target_proxy() -> None:
    scores = pd.Series([5.0, 1.0, 4.0, 2.0, 3.0])
    target_proxy = pd.Series([50.0, 10.0, 40.0, 20.0, 30.0])

    block = ev.metrics_block(scores, target_proxy)

    assert set(block.keys()) == {"spearman_risk_vs_target_proxy", "top_decile_lift"}
    # scores and target_proxy are perfectly rank-aligned by construction.
    assert block["spearman_risk_vs_target_proxy"] == pytest.approx(1.0)
