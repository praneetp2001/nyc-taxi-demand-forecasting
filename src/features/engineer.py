"""Feature engineering for the zone-hour demand panel.

All lag/rolling features are computed with `.shift()` before any window
aggregation, so a row at time t only ever sees information from t-1 and
earlier -- this is what makes the walk-forward CV leakage-free.
"""
import numpy as np
import pandas as pd
import holidays as holidays_lib
import h3

LAGS = [1, 24, 168]
ROLLING_WINDOWS = [3, 24, 168]


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df["hour_ts"].dt.hour
    df["dow"] = df["hour_ts"].dt.dayofweek
    df["month"] = df["hour_ts"].dt.month
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)
    return df


def add_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    us_holidays = holidays_lib.US(years=[2024, 2025])
    holiday_dates = sorted(us_holidays.keys())
    dates = df["hour_ts"].dt.date

    df["is_holiday"] = dates.isin(set(holiday_dates)).astype(int)

    holiday_arr = np.array(holiday_dates)
    date_arr = dates.to_numpy()
    days_to = np.array(
        [np.min(np.abs((holiday_arr - d).astype("timedelta64[D]").astype(int))) for d in date_arr]
    )
    df["days_to_holiday"] = days_to
    return df


def add_lag_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-zone causal lag and rolling features. df must be sorted by zone, hour_ts."""
    df = df.copy()
    grouped = df.groupby("LocationID")["demand"]

    for lag in LAGS:
        df[f"lag_{lag}h"] = grouped.shift(lag)

    shifted = grouped.shift(1)
    for window in ROLLING_WINDOWS:
        df[f"rolling_mean_{window}h"] = shifted.groupby(df["LocationID"]).transform(
            lambda s: s.rolling(window, min_periods=1).mean()
        )
        df[f"rolling_std_{window}h"] = shifted.groupby(df["LocationID"]).transform(
            lambda s: s.rolling(window, min_periods=1).std()
        )
    return df


def add_h3_neighbor_feature(df: pd.DataFrame, zones: pd.DataFrame, max_k: int = 6) -> pd.DataFrame:
    """Mean lag_1h demand among the nearest zones by H3 grid distance.

    Zone sizes vary widely (dense Manhattan zones vs. sprawling Staten Island /
    airport zones), so a fixed k=1 ring leaves many zones with zero neighbors.
    Instead we expand k until at least one neighboring zone is found.
    """
    df = df.copy()
    zone_to_cell = dict(zip(zones["LocationID"], zones["h3_cell"]))
    cell_to_zones: dict[str, list[int]] = {}
    for loc_id, cell in zone_to_cell.items():
        cell_to_zones.setdefault(cell, []).append(loc_id)

    neighbor_zones: dict[int, list[int]] = {}
    for loc_id, cell in zone_to_cell.items():
        neighbors: list[int] = []
        for k in range(1, max_k + 1):
            ring = h3.grid_disk(cell, k)
            neighbors = []
            for c in ring:
                if c == cell:
                    continue
                neighbors.extend(cell_to_zones.get(c, []))
            neighbors = [n for n in neighbors if n != loc_id]
            if neighbors:
                break
        neighbor_zones[loc_id] = neighbors

    # hour x zone matrix of lag_1h values
    pivot = df.pivot(index="hour_ts", columns="LocationID", values="lag_1h")

    neighbor_mean = pd.DataFrame(index=pivot.index, columns=pivot.columns, dtype=float)
    for loc_id in pivot.columns:
        neighbors = neighbor_zones.get(loc_id, [])
        valid = [z for z in neighbors if z in pivot.columns]
        if valid:
            neighbor_mean[loc_id] = pivot[valid].mean(axis=1)
        else:
            neighbor_mean[loc_id] = np.nan

    stacked = neighbor_mean.stack(future_stack=True).rename("h3_neighbor_lag_1h").reset_index()
    df = df.merge(stacked, on=["hour_ts", "LocationID"], how="left")
    return df


def build_features(panel: pd.DataFrame, zones: pd.DataFrame) -> pd.DataFrame:
    panel = panel.sort_values(["LocationID", "hour_ts"]).reset_index(drop=True)
    panel = add_time_features(panel)
    panel = add_holiday_features(panel)
    panel = add_lag_rolling_features(panel)
    panel = add_h3_neighbor_feature(panel, zones)
    return panel
