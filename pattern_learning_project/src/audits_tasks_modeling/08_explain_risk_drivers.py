#!/usr/bin/env python
"""Step 8: explain model risk drivers with SHAP.

Default explanation target is the Step 6 risk-burden regressor. The script also
handles the Step 7 elevated-risk classifier bundle.

Outputs:
  - risk_driver_explanations_h{horizon}.csv.gz
  - global_feature_importance_h{horizon}.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import config as cfg

import joblib
import numpy as np
import pandas as pd

from utils import ensure_dir, load_table, save_csv, save_json


def feature_names_from_preprocessor(preprocessor) -> list[str]:
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        names = []
        for name, transformer, cols in preprocessor.transformers_:
            if name == "remainder" or transformer == "drop":
                continue
            if hasattr(transformer, "get_feature_names_out"):
                try:
                    names.extend(list(transformer.get_feature_names_out(cols)))
                except Exception:
                    names.extend([f"{name}__{c}" for c in cols])
            else:
                names.extend([f"{name}__{c}" for c in cols])
        return names


def to_dense(x):
    return x.toarray() if hasattr(x, "toarray") else x


def unwrap_for_shap(bundle):
    pipe = bundle.get("base_pipeline") or bundle.get("pipeline")
    # If classifier was calibrated, pipeline may be CalibratedClassifierCV, which
    # is not directly SHAP-friendly. Use base_pipeline when present.
    if not hasattr(pipe, "named_steps"):
        raise ValueError("Model bundle does not contain a SHAP-compatible sklearn Pipeline.")
    preprocessor = pipe.named_steps["preprocess"]
    model = pipe.named_steps["model"]
    return pipe, preprocessor, model


def predict_bundle(bundle, df_features: pd.DataFrame) -> np.ndarray:
    pipe = bundle.get("pipeline")
    if pipe is None:
        raise ValueError("Model bundle has no pipeline.")
    if hasattr(pipe, "predict_proba"):
        try:
            return pipe.predict_proba(df_features)[:, 1]
        except Exception:
            pass
    pred = pipe.predict(df_features)
    return np.clip(np.asarray(pred, dtype=float), 0, None)


def shap_values(model, x_transformed):
    import shap
    explainer = shap.TreeExplainer(model)
    vals = explainer.shap_values(x_transformed)
    if isinstance(vals, list):
        vals = vals[1] if len(vals) > 1 else vals[0]
    return np.asarray(vals)


def summarize(values: np.ndarray, feature_names: list[str], top_n: int) -> tuple[str, str]:
    pos_order = np.argsort(values)[::-1]
    neg_order = np.argsort(values)
    pos, neg = [], []
    for i in pos_order:
        if values[i] <= 0 or len(pos) >= top_n:
            continue
        pos.append(f"{feature_names[i]} ({values[i]:+.4f})")
    for i in neg_order:
        if values[i] >= 0 or len(neg) >= top_n:
            continue
        neg.append(f"{feature_names[i]} ({values[i]:+.4f})")
    return "; ".join(pos), "; ".join(neg)


def fallback_importance(model, feature_names: list[str]) -> pd.DataFrame:
    if hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        imp = np.abs(np.asarray(model.coef_).ravel())
    else:
        imp = np.zeros(len(feature_names))
    return pd.DataFrame({"feature": feature_names[: len(imp)], "importance": imp}).sort_values("importance", ascending=False)


def main():
    parser = argparse.ArgumentParser(description="Explain risk model drivers using SHAP.")
    parser.add_argument("--input", default=cfg.RISK_STATE_DATA_PATH)
    parser.add_argument("--output-dir", default=cfg.STEP_08_DIR)
    parser.add_argument("--model", default=None, help="Model bundle path. Defaults to Step 6 horizon model.")
    parser.add_argument("--horizon", type=int, default=cfg.DEFAULT_HORIZON)
    parser.add_argument("--sample-size", type=int, default=cfg.EXPLANATION_SAMPLE_SIZE)
    parser.add_argument("--top-n", type=int, default=cfg.EXPLANATION_TOP_N)
    parser.add_argument("--top-predictions", action="store_true", default=cfg.EXPLAIN_TOP_PREDICTIONS, help="Explain highest predicted risk rows instead of random sample.")
    parser.add_argument("--random-state", type=int, default=cfg.RANDOM_STATE)
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    model_path = Path(args.model) if args.model else cfg.STEP_06_DIR / "models" / f"risk_burden_model_h{args.horizon}.joblib"
    bundle = joblib.load(model_path)
    df = load_table(args.input)
    df["as_of_date"] = pd.to_datetime(df["as_of_date"], errors="coerce")
    numeric_cols = bundle.get("numeric_cols", [])
    categorical_cols = bundle.get("categorical_cols", [])
    feature_cols = bundle.get("feature_cols", numeric_cols + categorical_cols)
    feature_cols = [c for c in feature_cols if c in df.columns]

    pred = predict_bundle(bundle, df[feature_cols])
    if args.top_predictions:
        sample_idx = np.argsort(-pred)[: min(args.sample_size, len(df))]
    else:
        rng = np.random.default_rng(args.random_state)
        sample_idx = rng.choice(np.arange(len(df)), size=min(args.sample_size, len(df)), replace=False)

    pipe_for_shap, preprocessor, model_for_shap = unwrap_for_shap(bundle)
    x_sample_trans = preprocessor.transform(df.iloc[sample_idx][feature_cols])
    feature_names = feature_names_from_preprocessor(preprocessor)

    try:
        vals = shap_values(model_for_shap, to_dense(x_sample_trans))
        if vals.ndim == 1:
            vals = vals.reshape(1, -1)
        global_imp = pd.DataFrame({"feature": feature_names[: vals.shape[1]], "mean_abs_shap": np.mean(np.abs(vals), axis=0)}).sort_values("mean_abs_shap", ascending=False)
        rows = []
        sample_df = df.iloc[sample_idx].reset_index(drop=True)
        for i in range(len(sample_df)):
            pos, neg = summarize(vals[i], feature_names, args.top_n)
            rows.append({
                "as_of_date": sample_df.loc[i].get("as_of_date"),
                "site": sample_df.loc[i].get("site"),
                "department": sample_df.loc[i].get("department"),
                "risk_theme_id": sample_df.loc[i].get("risk_theme_id"),
                "risk_theme_name": sample_df.loc[i].get("risk_theme_name"),
                "prediction": float(pred[sample_idx[i]]),
                "top_positive_drivers": pos,
                "top_negative_drivers": neg,
            })
        explanations = pd.DataFrame(rows).sort_values("prediction", ascending=False)
        method = "shap_tree_explainer"
    except Exception as exc:
        print(f"WARNING: SHAP failed ({exc}). Falling back to global feature importance.")
        global_imp = fallback_importance(model_for_shap, feature_names)
        sample_df = df.iloc[sample_idx].copy()
        explanations = sample_df[[c for c in ["as_of_date", "site", "department", "risk_theme_id", "risk_theme_name"] if c in sample_df.columns]].copy()
        explanations["prediction"] = pred[sample_idx]
        explanations["top_positive_drivers"] = "; ".join(global_imp.head(args.top_n)["feature"].astype(str).tolist())
        explanations["top_negative_drivers"] = ""
        explanations = explanations.sort_values("prediction", ascending=False)
        method = "fallback_feature_importance"

    explanations_path = output_dir / f"risk_driver_explanations_h{args.horizon}.csv.gz"
    importance_path = output_dir / f"global_feature_importance_h{args.horizon}.csv"
    save_csv(explanations, explanations_path)
    save_csv(global_imp, importance_path)
    summary = {
        "model_path": str(model_path),
        "method": method,
        "row_count_explained": int(len(explanations)),
        "explanations_path": str(explanations_path),
        "importance_path": str(importance_path),
    }
    save_json(summary, output_dir / f"08_risk_driver_explanations_summary_h{args.horizon}.json")
    print(summary)
    print(global_imp.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
