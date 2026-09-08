"""Generate the global SHAP summary for the deployed P50 model over the holdout set."""
import json
import sys

import lightgbm as lgb
import pandas as pd

from src.config import DATA_PROCESSED, MODELS_DIR, EDA_DIR
from src.explain.shap_utils import explain_row, global_summary, make_explainer


def main() -> None:
    with open(MODELS_DIR / "feature_columns.json") as f:
        cols = json.load(f)
    feature_columns = cols["numeric"] + cols["categorical"]

    booster = lgb.Booster(model_file=str(MODELS_DIR / "lgbm_q50.txt"))
    holdout = pd.read_parquet(DATA_PROCESSED / "holdout_val.parquet")
    for col in cols["categorical"]:
        holdout[col] = holdout[col].astype("category")

    X = holdout[feature_columns]
    explainer = make_explainer(booster)

    out_path = EDA_DIR / "shap_global.png"
    mean_abs = global_summary(explainer, X, out_path)
    print(f"[saved] {out_path}")
    print("\nTop 10 features by mean |SHAP|:")
    print(mean_abs.head(10))

    sample_row = X.iloc[[len(X) // 2]]
    example = explain_row(explainer, sample_row)
    print("\nExample per-prediction explanation:")
    print(json.dumps(example, indent=2))


if __name__ == "__main__":
    sys.exit(main())
