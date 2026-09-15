import numpy as np

from range_bars_ml.model import choose_thresholds, signal_metrics
from range_bars_ml.splits import walk_forward_splits


def test_thresholds_meet_required_coverage():
    probability = np.array([0.99, 0.95, 0.60, 0.40, 0.05, 0.01])
    target = np.array([1, 1, 0, 1, 0, 0])
    long_threshold, short_threshold, metrics = choose_thresholds(probability, target, minimum_coverage=0.30, steps=5)
    assert metrics["coverage"] >= 0.30
    assert long_threshold >= 0.5 and short_threshold <= 0.5
    assert signal_metrics(probability, target, long_threshold, short_threshold)["signals"] >= 2


def test_scoring_rewards_abstention_over_bad_predictions():
    probability = np.array([0.90, 0.10, 0.80, 0.20])
    # Every directional call above would be wrong.
    target = np.array([0, 1, 0, 1])
    long_threshold, short_threshold, metrics = choose_thresholds(probability, target, steps=5)
    assert long_threshold >= 1.0
    assert short_threshold <= 0.0
    assert metrics["signals"] == 0
    assert metrics["abstentions"] == 4
    assert metrics["score"] == -0.004


def test_signal_score_applies_correct_wrong_and_abstention_values():
    probability = np.array([0.9, 0.8, 0.5, 0.1])
    target = np.array([1, 0, 1, 1])
    metrics = signal_metrics(probability, target, long_threshold=0.85, short_threshold=0.05)
    assert metrics["correct"] == 1
    assert metrics["wrong"] == 0
    assert metrics["abstentions"] == 3
    assert metrics["score"] == 0.997


def test_walk_forward_is_chronological_and_purged():
    plan = walk_forward_splits(1_000, n_folds=3, purge_rows=1)
    assert plan.test[0] > plan.development[-1]
    for train, validation in plan.folds:
        assert train[-1] + 1 < validation[0]
