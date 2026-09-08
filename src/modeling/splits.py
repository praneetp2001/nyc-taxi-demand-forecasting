"""Chronological expanding-window walk-forward CV splits.

Folds are defined purely by timestamp cutoffs -- never a random shuffle -- so
a model is always validated on hours strictly after everything it was
trained on. Combined with the causal (shift-based) lag/rolling features in
`src.features.engineer`, this avoids the leakage that a plain random
train/test split would introduce for time series data.
"""
from dataclasses import dataclass

import pandas as pd


@dataclass
class Fold:
    name: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp  # exclusive
    val_end: pd.Timestamp  # exclusive


def get_walk_forward_folds(
    dates: pd.Series, min_train_weeks: int = 2, fold_weeks: int = 1
) -> list[Fold]:
    start = dates.min().normalize()
    end = dates.max()

    boundaries = pd.date_range(start, end, freq=f"{fold_weeks}W")
    boundaries = [b for b in boundaries if b <= end]
    if boundaries[-1] < end:
        boundaries.append(end + pd.Timedelta(hours=1))

    folds = []
    for i in range(min_train_weeks, len(boundaries) - 1):
        train_end = boundaries[i]
        val_end = boundaries[i + 1]
        folds.append(Fold(name=f"fold_{i - min_train_weeks + 1}", train_start=start, train_end=train_end, val_end=val_end))
    return folds


def fold_masks(dates: pd.Series, fold: Fold) -> tuple[pd.Series, pd.Series]:
    train_mask = (dates >= fold.train_start) & (dates < fold.train_end)
    val_mask = (dates >= fold.train_end) & (dates < fold.val_end)
    return train_mask, val_mask
