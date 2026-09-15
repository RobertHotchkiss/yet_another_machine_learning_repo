"""Chronological, purged split helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SplitPlan:
    development: np.ndarray
    test: np.ndarray
    folds: tuple[tuple[np.ndarray, np.ndarray], ...]


def walk_forward_splits(
    n_rows: int,
    *,
    test_fraction: float = 0.20,
    n_folds: int = 3,
    initial_train_fraction: float = 0.40,
    purge_rows: int = 1,
) -> SplitPlan:
    """Create expanding train / chronological validation folds plus final test rows."""
    if not 0 < test_fraction < 1 or n_folds < 1 or not 0 < initial_train_fraction < 1:
        raise ValueError("Invalid chronological split configuration")
    development_end = int(n_rows * (1 - test_fraction))
    initial_end = int(development_end * initial_train_fraction)
    if initial_end <= purge_rows or development_end - initial_end < n_folds:
        raise ValueError("Not enough rows for requested walk-forward split")
    validation_edges = np.linspace(initial_end, development_end, n_folds + 1, dtype=int)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for fold in range(n_folds):
        validation_start, validation_end = validation_edges[fold], validation_edges[fold + 1]
        train_end = validation_start - purge_rows
        folds.append((np.arange(train_end), np.arange(validation_start, validation_end)))
    return SplitPlan(np.arange(development_end), np.arange(development_end, n_rows), tuple(folds))
