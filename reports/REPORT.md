# Urban Mobility Demand Forecasting — Results

## Problem

Predict NYC yellow taxi demand per zone per hour, with uncertainty estimates
(P10/P50/P90), evaluated so that the reported error is an honest estimate of
performance on unseen future hours — not an artifact of leakage.

## Data

| Source | Detail |
|---|---|
| Trips | NYC TLC Yellow Taxi, Jan–Feb 2025, 7,034,591 trips after filtering (99.7% retained) |
| Weather | Open-Meteo historical archive API, hourly, Central Park reference point |
| Holidays | US federal holidays via the `holidays` Python package |
| Zones | 263 TLC zones, centroid + H3 resolution-8 cell per zone |

**Weather source substitution:** the original spec called for NOAA. NOAA's CDO
API requires a registered API token before any data can be pulled; Open-Meteo's
historical archive exposes equivalent hourly variables (temperature,
precipitation, snowfall, wind speed) for any past date range with no signup.
Given the project's time constraints, Open-Meteo was used instead — flagged
here for transparency rather than silently swapped.

The modeling table is a full (zone × hour) grid — 263 zones × 1,416 hours =
372,408 rows — with true zero-demand hours kept explicit (47.1% of rows have
zero trips) rather than dropped, so the model learns a real zero baseline
instead of only ever seeing positive counts.

## Features

- **Time**: hour/day-of-week/month, cyclical sin/cos encodings, weekend flag
- **Holiday**: federal holiday flag, days-to-nearest-holiday
- **Weather**: temperature, precipitation, snowfall, wind speed (broadcast across zones per hour)
- **Lag/rolling** (all causal — computed with `.shift()` before any window aggregation): lag at 1h/24h/168h, rolling mean/std over 3h/24h/168h
- **Spatial**: zone centroid lat/lon, H3-neighbor lag feature — mean `lag_1h` demand among the nearest zones by H3 grid distance (expanding the ring radius per zone until at least one neighboring zone is found, since zone sizes vary too much for a fixed ring to work everywhere — e.g. Newark Airport (zone 1) has no neighbor at all, being across the Hudson from every other TLC zone)

## Leakage control: walk-forward CV

Folds are chronological and expanding — never a random shuffle:

| Fold | Train | Validate |
|---|---|---|
| 1 | Jan 8 – Jan 26 | Jan 26 – Feb 2 |
| 2 | Jan 8 – Feb 2 | Feb 2 – Feb 9 |
| 3 | Jan 8 – Feb 9 | Feb 9 – Feb 16 |
| 4 | Jan 8 – Feb 16 | Feb 16 – Feb 23 |
| 5 | Jan 8 – Feb 23 | Feb 23 – Mar 1 |

(The first 7 days, through Jan 8, are dropped as lag-feature warmup — `lag_168h` needs a full prior week.)

## Model

LightGBM, `objective="quantile"`, three independently trained boosters
(α = 0.1, 0.5, 0.9), `num_leaves=63`, `learning_rate=0.05`, early stopping on
a validation split, `LocationID`/`Borough`/`hour`/`dow`/`month` as native
categorical features. Final deployed models are refit on all usable history
(Jan 8 – Feb 28) after CV honestly estimated generalization error.

## Results: model vs. seasonal-naive baseline

Baseline = demand in the same zone, same hour, exactly 7 days earlier
(`lag_168h`), used as the point estimate for all three quantiles.

Mean across the 5 walk-forward folds:

| Metric | Model | Baseline | Improvement |
|---|---|---|---|
| RMSE | 11.10 | 14.55 | −23.7% |
| MAE | 3.59 | 4.61 | −22.2% |
| MAPE (nonzero-demand rows) | 50.3% | 65.0% | −22.6% |
| Pinball loss, P10 | 0.795 | 2.246 | −64.6% |
| Pinball loss, P50 | 1.793 | 2.304 | −22.2% |
| Pinball loss, P90 | 0.909 | 2.362 | −61.5% |

The model beats the baseline on every fold and every quantile. The P10/P90
pinball-loss gap is especially large because the naive baseline has no notion
of uncertainty: it uses the same point value for all three quantiles. The
quantile models are trained independently; the API sorts returned values if a
rare crossing occurs so that its displayed interval remains ordered.

Full per-fold numbers: [`cv_metrics.json`](cv_metrics.json).

## Explainability (SHAP)

Global feature importance (mean |SHAP value| on the P50 model, held-out set),
see [`eda/shap_global.png`](eda/shap_global.png):

1. `lag_1h` — demand one hour ago
2. `lag_168h` — demand at the same hour last week
3. `lag_24h` — demand at the same hour yesterday
4. `rolling_mean_3h` — short-term trend
5. `rolling_mean_168h` — weekly-seasonal baseline level
6. `hour`, `h3_neighbor_lag_1h`, `LocationID` — smaller but non-trivial contributions

Recent demand dominates, as expected for a short-horizon operational
forecast, but the spatial (H3 neighbor) and calendar features do meaningfully
shift individual predictions — see the per-prediction waterfall served by
`POST /predict` and rendered live in the dashboard.

## Serving

- **FastAPI** (`src/api/main.py`): `/predict` (single zone-hour + SHAP
  explanation), `/predict/all` (all 263 zones at one hour, for the map),
  `/zones`, `/meta`. Runs over the held-out evaluation window (Feb 26–28,
  2025) where features are already computed — this is a batch,
  single-machine project, so "live" means replaying held-out history through
  the trained models on demand, not a real-time feature pipeline.
- **Streamlit dashboard** (`src/dashboard/app.py`): hour slider, centroid
  bubble map of predicted P50 demand across zones, and a per-zone SHAP chart.

## Experiment tracking

All CV metrics, hyperparameters, and model artifacts are logged to MLflow
under the `taxi-demand-quantile-lgbm` experiment (`mlruns/`).

## How to reproduce

See [`README.md`](../README.md) for the full run order (ingest → features → train → serve → dashboard).
