#!/usr/bin/env python
"""Step 7: train elevated risk probability classifier.

Model:
  LightGBM binary classifier + probability calibration when a valid calibration
  split is available.

Input:
  risk_state_training_data.csv.gz from Step 5.

Output:
  models/elevated_risk_classifier_h{horizon}.joblib
  elevated_risk_predictions_h{horizon}.csv.gz
  model_evaluation_elevated_risk_h{horizon}.csv
"""
from __future__ import annotations

import argparse

import config as cfg
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from utils import ensure_dir, get_feature_columns, load_table, precision_at_top_frac, save_csv, save_json, time_based_split


def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def build_base_pipeline(numeric_cols, categorical_cols):
    pre = ColumnTransformer([("num", "passthrough", numeric_cols), ("cat", make_ohe(), categorical_cols)], remainder="drop")
    try:
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(
            objective="binary",
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            class_weight="balanced",
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=1,
            verbosity=-1,
            force_col_wise=True,
        )
        name = "lightgbm_calibrated_classifier"
    except Exception:
        model = GradientBoostingClassifier(n_estimators=300, learning_rate=0.03, random_state=42)
        name = "sklearn_gradient_boosting_calibrated_classifier"
    return Pipeline([("preprocess", pre), ("model", model)]), name


def fit_probability_calibrator(base_pipeline, X_cal, y_cal, method: str):
    raw = base_pipeline.predict_proba(X_cal)[:, 1]
    y = np.asarray(y_cal, dtype=int)
    if method == "isotonic" and len(y) >= 30:
        cal = IsotonicRegression(out_of_bounds="clip")
        cal.fit(raw, y)
        return cal, "isotonic"
    # Sigmoid/Platt fallback. Works with smaller calibration samples.
    cal = LogisticRegression(solver="lbfgs")
    cal.fit(raw.reshape(-1, 1), y)
    return cal, "sigmoid"


def apply_probability_calibration(base_pipeline, calibrator, calibration_method, X):
    raw = base_pipeline.predict_proba(X)[:, 1]
    if calibrator is None:
        return raw
    if calibration_method == "isotonic":
        return calibrator.predict(raw)
    return calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]


def safe_metrics(y_true, prob, threshold=0.5):
    y_true = np.asarray(y_true, dtype=int)
    prob = np.asarray(prob, dtype=float)
    pred = (prob >= threshold).astype(int)
    out = {
        "positive_rate": float(np.mean(y_true)) if len(y_true) else np.nan,
        "mean_predicted_probability": float(np.mean(prob)) if len(prob) else np.nan,
        "precision_at_top10pct": float(precision_at_top_frac(y_true, prob, top_frac=0.10)) if len(y_true) else np.nan,
        "precision_at_0_50": float(precision_score(y_true, pred, zero_division=0)) if len(y_true) else np.nan,
        "recall_at_0_50": float(recall_score(y_true, pred, zero_division=0)) if len(y_true) else np.nan,
    }
    if len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, prob))
        out["average_precision"] = float(average_precision_score(y_true, prob))
        out["brier_score"] = float(brier_score_loss(y_true, prob))
    else:
        out["roc_auc"] = np.nan
        out["average_precision"] = np.nan
        out["brier_score"] = np.nan
    return out


def main():
    parser = argparse.ArgumentParser(description="Train calibrated elevated-risk classifier.")
    parser.add_argument("--input", default=cfg.RISK_STATE_DATA_PATH)
    parser.add_argument("--output-dir", default=cfg.STEP_07_DIR)
    parser.add_argument("--horizon", type=int, default=cfg.DEFAULT_HORIZON)
    parser.add_argument("--positive-quantile", type=float, default=cfg.POSITIVE_QUANTILE)
    parser.add_argument("--fixed-threshold", type=float, default=cfg.FIXED_RISK_THRESHOLD)
    parser.add_argument("--test-frac", type=float, default=cfg.TEST_FRAC)
    parser.add_argument("--calibration-frac", type=float, default=cfg.CALIBRATION_FRAC)
    parser.add_argument("--calibration-method", choices=["isotonic", "sigmoid"], default=cfg.CALIBRATION_METHOD)
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    model_dir = ensure_dir(output_dir / "models")
    df = load_table(args.input)
    df["as_of_date"] = pd.to_datetime(df["as_of_date"], errors="coerce")
    burden_target = f"future_risk_burden_{args.horizon}d"
    if burden_target not in df.columns:
        raise ValueError(f"Missing target column: {burden_target}")
    df[burden_target] = pd.to_numeric(df[burden_target], errors="coerce").fillna(0).clip(lower=0)

    train_full_idx, test_idx, test_cutoff = time_based_split(df, "as_of_date", args.test_frac)
    train_full = df.loc[train_full_idx].copy()
    test = df.loc[test_idx].copy()

    if args.fixed_threshold is not None:
        risk_threshold = float(args.fixed_threshold)
    else:
        train_values = train_full[burden_target].to_numpy(float)
        risk_threshold = float(np.quantile(train_values, args.positive_quantile))
        if risk_threshold <= 0 and np.any(train_values > 0):
            risk_threshold = float(np.quantile(train_values[train_values > 0], args.positive_quantile))
    df["high_risk_target"] = (df[burden_target] >= risk_threshold).astype(int)
    train_full["high_risk_target"] = (train_full[burden_target] >= risk_threshold).astype(int)
    test["high_risk_target"] = (test[burden_target] >= risk_threshold).astype(int)

    if train_full["high_risk_target"].nunique() < 2:
        raise ValueError("Training split has one class only. Lower --positive-quantile or use --fixed-threshold.")

    # Time split inside training data for calibration.
    train_sub_idx, cal_idx, cal_cutoff = time_based_split(train_full, "as_of_date", args.calibration_frac)
    train_sub = train_full.loc[train_sub_idx].copy()
    cal = train_full.loc[cal_idx].copy()
    if train_sub["high_risk_target"].nunique() < 2:
        train_sub = train_full.copy()
        cal = pd.DataFrame(columns=train_full.columns)

    exclude = ["as_of_date", "risk_theme_id", burden_target, "high_risk_target"] + [c for c in df.columns if c.startswith("future_")]
    numeric_cols, categorical_cols = get_feature_columns(df, target_cols=["high_risk_target"], extra_exclude=exclude)
    categorical_cols = [c for c in categorical_cols if c in ["site", "department", "risk_theme_name"]]
    feature_cols = numeric_cols + categorical_cols

    base_pipeline, model_name = build_base_pipeline(numeric_cols, categorical_cols)
    base_pipeline.fit(train_sub[feature_cols], train_sub["high_risk_target"].astype(int))

    calibrated = False
    calibrator = None
    calibration_method_used = None
    if len(cal) > 0 and cal["high_risk_target"].nunique() > 1:
        calibrator, calibration_method_used = fit_probability_calibrator(
            base_pipeline,
            cal[feature_cols],
            cal["high_risk_target"].astype(int),
            args.calibration_method,
        )
        calibrated = True
    else:
        print("WARNING: Calibration skipped because calibration split was empty or single-class.")

    metrics_rows = []
    prediction_frames = []
    for split_name, split_df in [("train_fit", train_sub), ("calibration", cal), ("test", test)]:
        if len(split_df) == 0:
            continue
        prob = apply_probability_calibration(base_pipeline, calibrator, calibration_method_used, split_df[feature_cols])
        m = safe_metrics(split_df["high_risk_target"].to_numpy(int), prob)
        m.update({"split": split_name, "row_count": int(len(split_df)), "risk_threshold": risk_threshold})
        metrics_rows.append(m)
        keep = ["as_of_date", "site", "department", "risk_theme_id", "risk_theme_name", burden_target, "high_risk_target"]
        pred = split_df[keep].copy()
        pred["p_high_risk"] = prob
        pred["risk_level"] = pd.cut(pred["p_high_risk"], [-0.01, 0.25, 0.50, 0.75, 1.01], labels=["Low", "Medium", "High", "Critical"]).astype(str)
        pred["split"] = split_name
        prediction_frames.append(pred)

    metrics = pd.DataFrame(metrics_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    model_path = model_dir / f"elevated_risk_classifier_h{args.horizon}.joblib"
    dump(
        {
            "pipeline": base_pipeline,
            "base_pipeline": base_pipeline,
            "calibrator": calibrator,
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "feature_cols": feature_cols,
            "burden_target": burden_target,
            "target": "high_risk_target",
            "risk_threshold": risk_threshold,
            "model_name": model_name,
            "calibrated": calibrated,
            "calibration_method": calibration_method_used if calibrated else None,
            "metrics": metrics.to_dict(orient="records"),
        },
        model_path,
    )
    save_csv(predictions, output_dir / f"elevated_risk_predictions_h{args.horizon}.csv.gz")
    save_csv(metrics, output_dir / f"model_evaluation_elevated_risk_h{args.horizon}.csv")
    save_json(
        {
            "model_path": str(model_path),
            "burden_target": burden_target,
            "risk_threshold": risk_threshold,
            "calibrated": calibrated,
            "test_cutoff": str(test_cutoff),
            "calibration_cutoff": str(cal_cutoff) if 'cal_cutoff' in locals() else None,
            "metrics": metrics.to_dict(orient="records"),
        },
        output_dir / f"07_elevated_risk_classifier_summary_h{args.horizon}.json",
    )
    print(metrics.to_string(index=False))
    print({"model_path": str(model_path), "risk_threshold": risk_threshold, "calibrated": calibrated})


if __name__ == "__main__":
    main()
