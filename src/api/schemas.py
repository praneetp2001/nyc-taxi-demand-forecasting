from datetime import datetime

from pydantic import BaseModel


class PredictRequest(BaseModel):
    zone_id: int
    timestamp: datetime


class FeatureContribution(BaseModel):
    feature: str
    value: float | str | None
    shap_value: float


class PredictResponse(BaseModel):
    zone_id: int
    timestamp: datetime
    p10: float
    p50: float
    p90: float
    top_features: list[FeatureContribution]


class ZoneInfo(BaseModel):
    zone_id: int
    zone: str | None
    borough: str | None
    centroid_lat: float
    centroid_lon: float


class ZonePrediction(BaseModel):
    zone_id: int
    p10: float
    p50: float
    p90: float
