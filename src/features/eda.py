"""Exploratory analysis of the zone-hour demand panel. Writes PNGs to reports/eda/."""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import DATA_PROCESSED, EDA_DIR


def main() -> None:
    df = pd.read_parquet(DATA_PROCESSED / "zone_hour_panel.parquet")

    # 1. Demand distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    df["demand"].clip(upper=df["demand"].quantile(0.99)).hist(bins=60, ax=ax)
    ax.set_title("Zone-hour demand distribution (clipped at p99)")
    ax.set_xlabel("trips per zone-hour")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(EDA_DIR / "demand_distribution.png", dpi=120)
    plt.close(fig)

    # 2. Hour-of-day x day-of-week heatmap
    pivot = df.pivot_table(index="dow", columns="hour", values="demand", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(9, 4))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    ax.set_yticks(range(7))
    ax.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels(range(0, 24, 2))
    ax.set_xlabel("hour of day")
    ax.set_title("Mean demand per zone by hour x day-of-week")
    fig.colorbar(im, ax=ax, label="mean trips/zone-hour")
    fig.tight_layout()
    fig.savefig(EDA_DIR / "hour_dow_heatmap.png", dpi=120)
    plt.close(fig)

    # 3. Holiday effect
    holiday_means = df.groupby("is_holiday")["demand"].mean()
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.bar(["Non-holiday", "Holiday"], holiday_means.values, color=["#4c72b0", "#dd8452"])
    ax.set_ylabel("mean trips/zone-hour")
    ax.set_title("Holiday effect on demand")
    fig.tight_layout()
    fig.savefig(EDA_DIR / "holiday_effect.png", dpi=120)
    plt.close(fig)

    # 4. Weather correlation
    weather_cols = ["temperature_2m", "precipitation", "snowfall", "windspeed_10m"]
    corr = df[weather_cols + ["demand"]].corr()["demand"].drop("demand")
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.barh(corr.index, corr.values, color="#55a868")
    ax.set_xlabel("correlation with demand")
    ax.set_title("Weather correlation with zone-hour demand")
    fig.tight_layout()
    fig.savefig(EDA_DIR / "weather_correlation.png", dpi=120)
    plt.close(fig)

    # 5. Missingness summary
    missing = df.isna().mean().sort_values(ascending=False)
    missing = missing[missing > 0]
    print("[missingness] fraction of rows NaN per column:")
    print(missing)

    print(f"[saved] plots -> {EDA_DIR}")
    print(df[["demand"] + weather_cols].describe())


if __name__ == "__main__":
    sys.exit(main())
