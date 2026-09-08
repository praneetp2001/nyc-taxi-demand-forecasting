"""Evaluation metrics: pinball (quantile) loss and point-forecast metrics."""
import numpy as np
import pandas as pd


def pinball_loss(y_true: pd.Series, y_pred: pd.Series, quantile: float) -> float:
    diff = y_true.to_numpy() - y_pred.to_numpy()
    return float(np.mean(np.maximum(quantile * diff, (quantile - 1) * diff)))


def point_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    y_true_arr = y_true.to_numpy()
    y_pred_arr = y_pred.to_numpy()
    rmse = float(np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2)))
    mae = float(np.mean(np.abs(y_true_arr - y_pred_arr)))
    # MAPE is undefined at y_true=0 (common here -- ~47% of zone-hours have zero
    # demand), so report it only over rows with nonzero actual demand.
    nonzero = y_true_arr > 0
    mape = float(np.mean(np.abs((y_true_arr[nonzero] - y_pred_arr[nonzero]) / y_true_arr[nonzero]))) * 100
    return {"rmse": rmse, "mae": mae, "mape_nonzero": mape}
