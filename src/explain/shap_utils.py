"""SHAP explainability for the P50 LightGBM quantile model: global + per-prediction."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def make_explainer(booster) -> shap.TreeExplainer:
    return shap.TreeExplainer(booster)


def global_summary(explainer: shap.TreeExplainer, X: pd.DataFrame, out_path) -> pd.Series:
    shap_values = explainer.shap_values(X)
    mean_abs = pd.Series(np.abs(shap_values).mean(axis=0), index=X.columns).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(7, 6))
    top = mean_abs.head(15).iloc[::-1]
    ax.barh(top.index, top.values, color="#4c72b0")
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title("Global feature importance (P50 model)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return mean_abs


def explain_row(explainer: shap.TreeExplainer, row: pd.DataFrame, top_n: int = 8) -> list[dict]:
    shap_values = explainer.shap_values(row)[0]
    contributions = pd.Series(shap_values, index=row.columns)
    contributions = contributions.reindex(contributions.abs().sort_values(ascending=False).index)
    top = contributions.head(top_n)
    return [
        {"feature": name, "value": _serialize(row.iloc[0][name]), "shap_value": float(val)}
        for name, val in top.items()
    ]


def _serialize(value):
    if isinstance(value, (np.generic,)):
        return value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if pd.api.types.is_scalar(value) and pd.isna(value):
        return None
    return value
