"""Download hourly NYC weather from the Open-Meteo historical archive API.

Substituted for NOAA per project decision: NOAA's CDO API requires a registered
token; Open-Meteo needs no signup and provides equivalent hourly variables.
"""
import json
import sys

import pandas as pd
import requests

from src.config import DATA_RAW, DATA_PROCESSED, WEATHER_LAT, WEATHER_LON, WEATHER_START, WEATHER_END

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HOURLY_VARS = ["temperature_2m", "precipitation", "snowfall", "windspeed_10m"]


def main() -> None:
    raw_path = DATA_RAW / "weather.json"
    if raw_path.exists():
        print(f"[skip] {raw_path.name} already downloaded")
        payload = json.loads(raw_path.read_text())
    else:
        params = {
            "latitude": WEATHER_LAT,
            "longitude": WEATHER_LON,
            "start_date": WEATHER_START,
            "end_date": WEATHER_END,
            "hourly": ",".join(HOURLY_VARS),
            "timezone": "America/New_York",
        }
        print(f"[download] {ARCHIVE_URL} {params}")
        r = requests.get(ARCHIVE_URL, params=params, timeout=60)
        r.raise_for_status()
        payload = r.json()
        raw_path.write_text(json.dumps(payload))
        print(f"[done] saved {raw_path}")

    hourly = payload["hourly"]
    df = pd.DataFrame({"time": hourly["time"], **{v: hourly[v] for v in HOURLY_VARS}})
    df["time"] = pd.to_datetime(df["time"])
    df = df.rename(columns={"time": "hour_ts"})

    out_path = DATA_PROCESSED / "weather_hourly.parquet"
    df.to_parquet(out_path, index=False)
    print(f"[saved] {out_path} shape={df.shape}")
    print(f"[range] {df['hour_ts'].min()} .. {df['hour_ts'].max()}")


if __name__ == "__main__":
    sys.exit(main())
