"""Build the zone x hour modeling panel: full grid, weather/zone joins, engineered features."""
import sys

import pandas as pd

from src.config import DATA_PROCESSED, PERIOD_START, PERIOD_END
from src.features.engineer import build_features


def main() -> None:
    trips = pd.read_parquet(DATA_PROCESSED / "trips_raw.parquet")
    zones = pd.read_parquet(DATA_PROCESSED / "zones.parquet")
    weather = pd.read_parquet(DATA_PROCESSED / "weather_hourly.parquet")

    trips["hour_ts"] = trips["tpep_pickup_datetime"].dt.floor("h")
    demand = (
        trips.groupby(["PULocationID", "hour_ts"])
        .size()
        .rename("demand")
        .reset_index()
        .rename(columns={"PULocationID": "LocationID"})
    )

    all_hours = pd.date_range(PERIOD_START, PERIOD_END, freq="h", inclusive="left")
    zone_ids = zones["LocationID"].unique()
    grid = pd.MultiIndex.from_product([zone_ids, all_hours], names=["LocationID", "hour_ts"]).to_frame(
        index=False
    )

    panel = grid.merge(demand, on=["LocationID", "hour_ts"], how="left")
    panel["demand"] = panel["demand"].fillna(0).astype(int)

    panel = panel.merge(
        zones[["LocationID", "Borough", "h3_cell", "centroid_lat", "centroid_lon"]],
        on="LocationID",
        how="left",
    )
    panel = panel.merge(weather, on="hour_ts", how="left")

    print(f"[grid] {len(zone_ids)} zones x {len(all_hours)} hours = {len(grid):,} rows")
    print(f"[demand] mean={panel['demand'].mean():.2f} zero_pct={(panel['demand'] == 0).mean():.1%}")

    panel = build_features(panel, zones)

    n_rows = len(panel)
    n_nan_warmup = panel["lag_168h"].isna().sum()
    print(f"[panel] shape={panel.shape}, rows missing lag_168h (warmup, expected)={n_nan_warmup:,}/{n_rows:,}")

    non_warmup = panel.dropna(subset=["lag_168h"])
    remaining_nans = non_warmup.isna().sum()
    remaining_nans = remaining_nans[remaining_nans > 0]
    if len(remaining_nans):
        print(f"[warn] unexpected NaNs outside warmup period:\n{remaining_nans}")
    else:
        print("[ok] no unexpected NaNs outside the lag warmup period")

    out_path = DATA_PROCESSED / "zone_hour_panel.parquet"
    panel.to_parquet(out_path, index=False)
    print(f"[saved] {out_path} shape={panel.shape}")


if __name__ == "__main__":
    sys.exit(main())
