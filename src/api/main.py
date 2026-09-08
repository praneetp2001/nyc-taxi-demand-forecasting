"""FastAPI serving endpoint for taxi demand quantile forecasts.

Serves over the held-out evaluation window (the last slice of Jan-Feb 2025,
carved out in src.modeling.train.train_final_models) where lag/weather/holiday
features are already computed in the processed panel -- this is a batch,
single-machine project, not a live feature pipeline, so "live" here means
replaying held-out history through the trained models on demand.
"""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.config import DATA_PROCESSED, MODELS_DIR
from src.explain.shap_utils import explain_row, make_explainer
from src.api.schemas import PredictRequest, PredictResponse, ZoneInfo, ZonePrediction

app = FastAPI(title="Taxi Demand Forecast API")

_state: dict = {}


@app.on_event("startup")
def load_artifacts() -> None:
    with open(MODELS_DIR / "feature_columns.json") as f:
        cols = json.load(f)
    feature_columns = cols["numeric"] + cols["categorical"]
    categorical = cols["categorical"]

    boosters = {
        0.1: lgb.Booster(model_file=str(MODELS_DIR / "lgbm_q10.txt")),
        0.5: lgb.Booster(model_file=str(MODELS_DIR / "lgbm_q50.txt")),
        0.9: lgb.Booster(model_file=str(MODELS_DIR / "lgbm_q90.txt")),
    }

    holdout = pd.read_parquet(DATA_PROCESSED / "holdout_val.parquet")
    for col in categorical:
        holdout[col] = holdout[col].astype("category")

    zones = pd.read_parquet(DATA_PROCESSED / "zones.parquet")

    _state.update(
        feature_columns=feature_columns,
        categorical=categorical,
        boosters=boosters,
        holdout=holdout,
        zones=zones,
        explainer=make_explainer(boosters[0.5]),
    )
    print(f"[startup] loaded models, {len(holdout):,} holdout rows, {len(zones)} zones")


@app.get("/meta")
def meta():
    holdout = _state["holdout"]
    return {
        "min_timestamp": holdout["hour_ts"].min().isoformat(),
        "max_timestamp": holdout["hour_ts"].max().isoformat(),
        "n_zones": int(holdout["LocationID"].nunique()),
    }


@app.get("/zones", response_model=list[ZoneInfo])
def zones():
    z = _state["zones"]
    return [
        ZoneInfo(
            zone_id=int(row.LocationID),
            zone=row.Zone if pd.notna(row.Zone) else None,
            borough=row.Borough if pd.notna(row.Borough) else None,
            centroid_lat=float(row.centroid_lat),
            centroid_lon=float(row.centroid_lon),
        )
        for row in z.itertuples()
    ]


def _lookup_row(zone_id: int, timestamp: pd.Timestamp) -> pd.DataFrame:
    holdout = _state["holdout"]
    match = holdout[(holdout["LocationID"] == zone_id) & (holdout["hour_ts"] == timestamp)]
    if match.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No data for zone_id={zone_id} at timestamp={timestamp}. "
            f"Valid range: see GET /meta.",
        )
    return match


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    timestamp = pd.Timestamp(req.timestamp)
    row = _lookup_row(req.zone_id, timestamp)
    feature_columns = _state["feature_columns"]
    X = row[feature_columns]

    raw_preds = [max(0.0, float(_state["boosters"][q].predict(X)[0])) for q in (0.1, 0.5, 0.9)]
    p10, p50, p90 = np.sort(raw_preds)
    top_features = explain_row(_state["explainer"], X)

    return PredictResponse(
        zone_id=req.zone_id,
        timestamp=req.timestamp,
        p10=float(p10),
        p50=float(p50),
        p90=float(p90),
        top_features=top_features,
    )


@app.get("/predict/all", response_model=list[ZonePrediction])
def predict_all(timestamp: str):
    ts = pd.Timestamp(timestamp)
    holdout = _state["holdout"]
    rows = holdout[holdout["hour_ts"] == ts]
    if rows.empty:
        raise HTTPException(status_code=404, detail=f"No data at timestamp={ts}. Valid range: see GET /meta.")

    feature_columns = _state["feature_columns"]
    X = rows[feature_columns]
    raw_preds = np.column_stack(
        [_state["boosters"][q].predict(X).clip(min=0) for q in (0.1, 0.5, 0.9)]
    )
    ordered_preds = np.sort(raw_preds, axis=1)

    return [
        ZonePrediction(
            zone_id=int(loc_id),
            p10=float(ordered_preds[i, 0]),
            p50=float(ordered_preds[i, 1]),
            p90=float(ordered_preds[i, 2]),
        )
        for i, loc_id in enumerate(rows["LocationID"].to_numpy())
    ]
