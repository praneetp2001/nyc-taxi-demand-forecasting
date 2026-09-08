"""Seasonal-naive baseline: predict demand as the value 7 days (168h) prior.

This reuses the `lag_168h` feature column directly -- exactly the causal
value a naive forecaster would use at prediction time. The same value is
used for all three quantiles since the naive method carries no notion of
uncertainty; this still yields a valid (if crude) pinball-loss comparison.
"""
import pandas as pd


def baseline_predict(df: pd.DataFrame) -> pd.Series:
    return df["lag_168h"].fillna(df["demand"].mean())
