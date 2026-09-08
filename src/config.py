from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
EDA_DIR = REPORTS_DIR / "eda"

for d in [DATA_RAW, DATA_PROCESSED, MODELS_DIR, REPORTS_DIR, EDA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

TLC_MONTHS = ["2025-01", "2025-02"]
TLC_BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{month}.parquet"
ZONE_LOOKUP_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
ZONE_SHAPEFILE_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"

# NYC reference point for weather (Central Park)
WEATHER_LAT = 40.7829
WEATHER_LON = -73.9654
WEATHER_START = "2024-12-31"
WEATHER_END = "2025-03-01"

# Modeling period bounds (drop stray rows outside this range)
PERIOD_START = "2025-01-01"
PERIOD_END = "2025-03-01"  # exclusive

H3_RESOLUTION = 8
QUANTILES = [0.1, 0.5, 0.9]
