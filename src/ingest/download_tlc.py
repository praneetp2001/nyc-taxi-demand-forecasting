"""Download NYC TLC yellow taxi trip parquet files and reduce to pickup zone/time."""
import sys

import pandas as pd
import requests

from src.config import DATA_RAW, DATA_PROCESSED, TLC_BASE_URL, TLC_MONTHS, PERIOD_START, PERIOD_END

KEEP_COLUMNS = ["tpep_pickup_datetime", "PULocationID"]


def download_month(month: str) -> None:
    dest = DATA_RAW / f"yellow_tripdata_{month}.parquet"
    if dest.exists():
        print(f"[skip] {dest.name} already downloaded")
        return
    url = TLC_BASE_URL.format(month=month)
    print(f"[download] {url}")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    size_mb = dest.stat().st_size / 1e6
    print(f"[done] {dest.name} ({size_mb:.1f} MB)")


def load_trips() -> pd.DataFrame:
    frames = []
    for month in TLC_MONTHS:
        path = DATA_RAW / f"yellow_tripdata_{month}.parquet"
        df = pd.read_parquet(path, columns=KEEP_COLUMNS)
        frames.append(df)
    trips = pd.concat(frames, ignore_index=True)

    before = len(trips)
    trips = trips.dropna(subset=["tpep_pickup_datetime", "PULocationID"])
    trips["PULocationID"] = trips["PULocationID"].astype(int)
    # Unknown / N/A zone codes in the TLC zone lookup
    trips = trips[~trips["PULocationID"].isin([264, 265])]
    trips = trips[
        (trips["tpep_pickup_datetime"] >= PERIOD_START)
        & (trips["tpep_pickup_datetime"] < PERIOD_END)
    ]
    after = len(trips)
    print(f"[filter] kept {after:,}/{before:,} rows ({after / before:.1%})")
    print(f"[range] {trips['tpep_pickup_datetime'].min()} .. {trips['tpep_pickup_datetime'].max()}")
    return trips


def main() -> None:
    for month in TLC_MONTHS:
        download_month(month)
    trips = load_trips()
    out_path = DATA_PROCESSED / "trips_raw.parquet"
    trips.to_parquet(out_path, index=False)
    print(f"[saved] {out_path} shape={trips.shape}")


if __name__ == "__main__":
    sys.exit(main())
