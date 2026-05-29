#!/usr/bin/env python
from __future__ import annotations

import argparse

import config as cfg
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from utils import ensure_dir, get_feature_columns, load_table, save_csv, save_json, time_based_split, top_decile_lift


def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def build_model(numeric_cols, categorical_cols):
    pre = ColumnTransformer([("num", "passthrough", numeric_cols), ("cat", make_ohe(), categorical_cols)], remainder="drop")
    try:
        from lightgbm import LGBMRegressor
        model = LGBMRegressor(objective="tweedie", tweedie_variance_power=1.3, n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=1, verbosity=-1, force_col_wise=True)
        name = "lightgbm_tweedie_regressor"
    except Exception:
        model = HistGradientBoostingRegressor(loss="poisson", learning_rate=0.05, max_iter=300, random_state=42)
        name = "sklearn_hist_gradient_boosting_poisson"
    return Pipeline([("preprocess", pre), ("model", model)]), name


def main():
    parser = argparse.ArgumentParser(description="Train future risk-burden regression model.")
    parser.add_argument("--input", default=cfg.RISK_STATE_DATA_PATH)
    parser.add_argument("--output-dir", default=cfg.STEP_06_DIR)
    parser.add_argument("--horizon", type=int, default=cfg.DEFAULT_HORIZON)
    parser.add_argument("--test-frac", type=float, default=cfg.TEST_FRAC)
    args = parser.parse_args()
    output_dir = ensure_dir(args.output_dir)
    model_dir = ensure_dir(output_dir / "models")
    df = load_table(args.input)
    df["as_of_date"] = pd.to_datetime(df["as_of_date"], errors="coerce")
    target = f"future_risk_burden_{args.horizon}d"
    if target not in df.columns:
        raise ValueError(f"Missing target column: {target}")
    exclude = ["as_of_date", "risk_theme_id"] + [c for c in df.columns if c.startswith("future_")]
    numeric_cols, categorical_cols = get_feature_columns(df, target_cols=[target], extra_exclude=exclude)
    categorical_cols = [c for c in categorical_cols if c in ["site", "department", "risk_theme_name"]]
    train_idx, test_idx, cutoff = time_based_split(df, "as_of_date", args.test_frac)
    train, test = df.loc[train_idx].copy(), df.loc[test_idx].copy()
    X_train, y_train = train[numeric_cols + categorical_cols], pd.to_numeric(train[target], errors="coerce").fillna(0)
    X_test, y_test = test[numeric_cols + categorical_cols], pd.to_numeric(test[target], errors="coerce").fillna(0)
    model, model_name = build_model(numeric_cols, categorical_cols)
    model.fit(X_train, y_train)
    pred_test = np.clip(model.predict(X_test), 0, None) if len(test) else np.array([])
    pred_train = np.clip(model.predict(X_train), 0, None) if len(train) else np.array([])
    metrics = {
        "model_name": model_name,
        "target": target,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "time_cutoff": str(cutoff),
        "mae": float(mean_absolute_error(y_test, pred_test)) if len(test) else np.nan,
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred_test))) if len(test) else np.nan,
        "r2": float(r2_score(y_test, pred_test)) if len(test) and y_test.nunique() > 1 else np.nan,
        "top10_lift_by_predicted_risk_burden": float(top_decile_lift(y_test.to_numpy(), pred_test)) if len(test) else np.nan,
    }
    keep = ["as_of_date", "site", "department", "risk_theme_id", "risk_theme_name", target]
    train_pred = train[keep].copy(); train_pred[f"predicted_risk_burden_{args.horizon}d"] = pred_train; train_pred["split"] = "train"
    test_pred = test[keep].copy(); test_pred[f"predicted_risk_burden_{args.horizon}d"] = pred_test; test_pred["split"] = "test"
    save_csv(pd.concat([train_pred, test_pred], ignore_index=True), output_dir / f"risk_burden_predictions_h{args.horizon}.csv.gz")
    save_csv(pd.DataFrame([metrics]), output_dir / f"model_evaluation_risk_burden_h{args.horizon}.csv")
    dump({"pipeline": model, "numeric_cols": numeric_cols, "categorical_cols": categorical_cols, "target": target, "metrics": metrics}, model_dir / f"risk_burden_model_h{args.horizon}.joblib")
    save_json(metrics, output_dir / f"06_risk_burden_model_summary_h{args.horizon}.json")
    print(metrics)


if __name__ == "__main__":
    main()
