"""Train LightGBM quantile regression models (P10/P50/P90) with walk-forward CV,
compare against the seasonal-naive baseline, and log everything to MLflow.
"""
import json
import sys

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd

from src.config import DATA_PROCESSED, MODELS_DIR, REPORTS_DIR, QUANTILES
from src.modeling.baseline import baseline_predict
from src.modeling.evaluate import pinball_loss, point_metrics
from src.modeling.splits import get_walk_forward_folds, fold_masks

CATEGORICAL_FEATURES = ["LocationID", "Borough", "hour", "dow", "month"]
NUMERIC_FEATURES = [
    "centroid_lat", "centroid_lon",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "is_weekend", "is_holiday", "days_to_holiday",
    "temperature_2m", "precipitation", "snowfall", "windspeed_10m",
    "lag_1h", "lag_24h", "lag_168h",
    "rolling_mean_3h", "rolling_std_3h",
    "rolling_mean_24h", "rolling_std_24h",
    "rolling_mean_168h", "rolling_std_168h",
    "h3_neighbor_lag_1h",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "demand"

LGB_PARAMS = {
    "objective": "quantile",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbose": -1,
}
NUM_BOOST_ROUND = 500
EARLY_STOPPING_ROUNDS = 30


def prep_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype("category")
    return df


def train_quantile_model(train_df, val_df, alpha: float):
    params = {**LGB_PARAMS, "alpha": alpha}
    dtrain = lgb.Dataset(train_df[FEATURE_COLUMNS], label=train_df[TARGET], categorical_feature=CATEGORICAL_FEATURES)
    dval = lgb.Dataset(val_df[FEATURE_COLUMNS], label=val_df[TARGET], reference=dtrain, categorical_feature=CATEGORICAL_FEATURES)
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(0)],
    )
    return booster


def run_cv(df: pd.DataFrame) -> list[dict]:
    folds = get_walk_forward_folds(df["hour_ts"])
    fold_results = []
    for fold in folds:
        train_mask, val_mask = fold_masks(df["hour_ts"], fold)
        train_df, val_df = df[train_mask], df[val_mask]

        result = {"fold": fold.name, "train_n": int(train_mask.sum()), "val_n": int(val_mask.sum())}
        baseline_pred = baseline_predict(val_df)
        result["baseline_point"] = point_metrics(val_df[TARGET], baseline_pred)
        result["baseline_pinball"] = {
            q: pinball_loss(val_df[TARGET], baseline_pred, q) for q in QUANTILES
        }

        model_preds = {}
        for q in QUANTILES:
            booster = train_quantile_model(train_df, val_df, q)
            model_preds[q] = pd.Series(booster.predict(val_df[FEATURE_COLUMNS]), index=val_df.index)

        result["model_pinball"] = {q: pinball_loss(val_df[TARGET], model_preds[q], q) for q in QUANTILES}
        result["model_point"] = point_metrics(val_df[TARGET], model_preds[0.5])

        print(
            f"[{fold.name}] train={result['train_n']:,} val={result['val_n']:,} "
            f"model_pinball(P50)={result['model_pinball'][0.5]:.3f} "
            f"baseline_pinball(P50)={result['baseline_pinball'][0.5]:.3f} "
            f"model_RMSE={result['model_point']['rmse']:.2f} baseline_RMSE={result['baseline_point']['rmse']:.2f}"
        )
        fold_results.append(result)
    return fold_results


def summarize(fold_results: list[dict]) -> dict:
    summary = {"model_pinball_mean": {}, "baseline_pinball_mean": {}}
    for q in QUANTILES:
        summary["model_pinball_mean"][q] = float(np.mean([f["model_pinball"][q] for f in fold_results]))
        summary["baseline_pinball_mean"][q] = float(np.mean([f["baseline_pinball"][q] for f in fold_results]))
    for metric in ["rmse", "mae", "mape_nonzero"]:
        summary[f"model_{metric}_mean"] = float(np.mean([f["model_point"][metric] for f in fold_results]))
        summary[f"baseline_{metric}_mean"] = float(np.mean([f["baseline_point"][metric] for f in fold_results]))
    return summary


def train_final_models(df: pd.DataFrame) -> dict:
    """Refit on all usable history for deployment, after CV has honestly estimated generalization error."""
    n = len(df)
    split_idx = int(n * 0.95)
    df_sorted = df.sort_values("hour_ts")
    train_df = df_sorted.iloc[:split_idx]
    val_df = df_sorted.iloc[split_idx:]

    boosters = {}
    for q in QUANTILES:
        boosters[q] = train_quantile_model(train_df, val_df, q)
        path = MODELS_DIR / f"lgbm_q{int(q * 100)}.txt"
        boosters[q].save_model(str(path))
        print(f"[saved] {path}")

    with open(MODELS_DIR / "feature_columns.json", "w") as f:
        json.dump({"numeric": NUMERIC_FEATURES, "categorical": CATEGORICAL_FEATURES, "target": TARGET}, f, indent=2)

    val_df.to_parquet(DATA_PROCESSED / "holdout_val.parquet", index=False)
    return boosters


def main() -> None:
    df = pd.read_parquet(DATA_PROCESSED / "zone_hour_panel.parquet")
    df = df.dropna(subset=["lag_168h"]).reset_index(drop=True)
    df = prep_frame(df)

    mlflow.set_experiment("taxi-demand-quantile-lgbm")
    with mlflow.start_run(run_name="walk_forward_cv"):
        mlflow.log_params(LGB_PARAMS)
        mlflow.log_param("num_boost_round", NUM_BOOST_ROUND)
        mlflow.log_param("quantiles", QUANTILES)

        fold_results = run_cv(df)
        summary = summarize(fold_results)

        for q in QUANTILES:
            mlflow.log_metric(f"model_pinball_q{int(q*100)}", summary["model_pinball_mean"][q])
            mlflow.log_metric(f"baseline_pinball_q{int(q*100)}", summary["baseline_pinball_mean"][q])
        for metric in ["rmse", "mae", "mape_nonzero"]:
            mlflow.log_metric(f"model_{metric}", summary[f"model_{metric}_mean"])
            mlflow.log_metric(f"baseline_{metric}", summary[f"baseline_{metric}_mean"])

        metrics_path = REPORTS_DIR / "cv_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump({"folds": fold_results, "summary": summary}, f, indent=2, default=str)
        mlflow.log_artifact(str(metrics_path))

        print("\n=== CV SUMMARY (mean across folds) ===")
        print(json.dumps(summary, indent=2))

        print("\n=== Training final models on full history ===")
        train_final_models(df)
        for q in QUANTILES:
            mlflow.log_artifact(str(MODELS_DIR / f"lgbm_q{int(q*100)}.txt"))
        mlflow.log_artifact(str(MODELS_DIR / "feature_columns.json"))


if __name__ == "__main__":
    sys.exit(main())
