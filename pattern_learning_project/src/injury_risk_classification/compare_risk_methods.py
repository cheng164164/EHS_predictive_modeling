"""Compare simple risk-ranking baselines against ML risk models.

Run from the project root after placing this file under:

    src/injury_risk_classification/compare_risk_methods.py

Command:

    python src/injury_risk_classification/compare_risk_methods.py

This script is intentionally independent from train_injury_risk_classifier.py's
main() workflow. It rebuilds the modeling dataset, creates simple rule/count
scores, trains temporary holdout ML models for comparison, and writes a single
comparison table under:

    outputs/ml/injury_risk_classification/method_comparison/<run_id>/

It does not overwrite or replace your production model artifacts.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    fbeta_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

# Allow direct execution from project root without installing the package.
if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from injury_risk_classification import config
from injury_risk_classification.feature_engineering import build_classification_dataset
from injury_risk_classification.train_injury_risk_classifier import (
    DEFAULT_CATEGORICAL_COLS,
    build_pipeline,
    fit_pipeline,
    infer_feature_columns,
    make_time_holdout,
    predict_proba_positive,
    target_col,
)
from injury_risk_classification.utils import ensure_dir, now_run_id, save_json, top_k_capture


LOCATION_CATEGORICAL_COLS = list(DEFAULT_CATEGORICAL_COLS)
DEFAULT_COMPARISON_WINDOWS = [3, 6, 12]
DEFAULT_TOP_FRACTIONS = [0.05, 0.10, 0.20]


# -----------------------------------------------------------------------------
# Generic metrics
# -----------------------------------------------------------------------------

def _safe_auc(y_true: np.ndarray, y_score: np.ndarray, auc_type: str) -> float:
    """Return ROC-AUC or PR-AUC; NaN if the metric is undefined."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    finite = np.isfinite(y_score)
    y_true = y_true[finite]
    y_score = y_score[finite]
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return np.nan
    try:
        if auc_type == "roc":
            return float(roc_auc_score(y_true, y_score))
        if auc_type == "pr":
            return float(average_precision_score(y_true, y_score))
    except Exception:
        return np.nan
    raise ValueError(auc_type)


def _score_supports_brier(y_score: np.ndarray) -> bool:
    """Brier score is meaningful only for probability-like scores in [0, 1]."""
    values = np.asarray(y_score, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return False
    return bool(values.min() >= 0.0 and values.max() <= 1.0)


def select_threshold_from_scores(
    y_true: np.ndarray,
    y_score: np.ndarray,
    strategy: str,
    top_percent: float,
    fixed_threshold: float,
) -> tuple[float, pd.DataFrame]:
    """Select a threshold for arbitrary risk scores.

    Unlike the training script's F2 threshold selector, this implementation works
    for both probabilities and raw count scores. For rule-based baselines, scores
    can be 0, 1, 2, ... rather than probabilities.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    finite = np.isfinite(y_score)
    y_true = y_true[finite]
    y_score = y_score[finite]

    if len(y_score) == 0:
        return float(fixed_threshold), pd.DataFrame({"threshold": [fixed_threshold], "strategy": ["empty_fallback"]})

    strategy = str(strategy or "top_percent").lower()
    if strategy == "fixed":
        return float(fixed_threshold), pd.DataFrame({"threshold": [fixed_threshold], "strategy": ["fixed"]})

    if strategy == "top_percent":
        top_percent = float(top_percent)
        top_percent = min(max(top_percent, 1e-6), 0.999999)
        threshold = float(np.quantile(y_score, 1.0 - top_percent))
        return threshold, pd.DataFrame({"threshold": [threshold], "strategy": ["top_percent"], "top_percent": [top_percent]})

    # F2 strategy for raw scores: evaluate unique score thresholds. If there are
    # many unique values, use quantile-based candidates to keep runtime stable.
    unique_scores = np.unique(y_score[np.isfinite(y_score)])
    if len(unique_scores) > 500:
        quantiles = np.linspace(0.01, 0.99, 199)
        candidates = np.unique(np.quantile(y_score, quantiles))
    else:
        candidates = unique_scores

    rows: list[dict[str, Any]] = []
    for threshold in candidates:
        y_pred = (y_score >= threshold).astype(int)
        rows.append({
            "threshold": float(threshold),
            "f2": float(fbeta_score(y_true, y_pred, beta=2.0, zero_division=0)),
            "predicted_positive_rate": float(y_pred.mean()) if len(y_pred) else np.nan,
        })
    table = pd.DataFrame(rows)
    if table.empty:
        return float(fixed_threshold), pd.DataFrame({"threshold": [fixed_threshold], "strategy": ["fallback_fixed"]})
    best = table.sort_values(["f2", "predicted_positive_rate"], ascending=[False, True]).iloc[0]
    return float(best["threshold"]), table


def evaluate_score_method(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    top_fractions: list[float] | None = None,
) -> dict[str, Any]:
    """Evaluate a score vector as both a ranking and thresholded classifier."""
    top_fractions = top_fractions or DEFAULT_TOP_FRACTIONS
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_score = np.nan_to_num(y_score, nan=0.0, posinf=np.nanmax(y_score[np.isfinite(y_score)]) if np.isfinite(y_score).any() else 0.0, neginf=0.0)
    y_pred = (y_score >= float(threshold)).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics: dict[str, Any] = {
        "n_rows": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "positive_rate": float(y_true.mean()) if len(y_true) else np.nan,
        "threshold": float(threshold),
        "roc_auc": _safe_auc(y_true, y_score, "roc"),
        "pr_auc": _safe_auc(y_true, y_score, "pr"),
        "brier_score": float(brier_score_loss(y_true, y_score)) if _score_supports_brier(y_score) and len(np.unique(y_true)) > 1 else np.nan,
        "accuracy": float(accuracy_score(y_true, y_pred)) if len(y_true) else np.nan,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else np.nan,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }

    for frac in top_fractions:
        capture = top_k_capture(y_true, y_score, float(frac))
        pct = int(round(frac * 100))
        metrics[f"recall_at_top_{pct}pct"] = capture["recall_at_top"]
        metrics[f"precision_at_top_{pct}pct"] = capture["precision_at_top"]
        metrics[f"lift_at_top_{pct}pct"] = capture.get("lift_at_top")
        metrics[f"positives_at_top_{pct}pct"] = capture["positives_in_top"]
        metrics[f"rows_at_top_{pct}pct"] = capture["top_n"]
    return metrics


# -----------------------------------------------------------------------------
# Score construction for simple baselines
# -----------------------------------------------------------------------------

def _num(df: pd.DataFrame, col: str) -> pd.Series:
    """Return a numeric series; missing columns become zeros."""
    if col not in df.columns:
        return pd.Series(0.0, index=df.index)
    return pd.to_numeric(df[col], errors="coerce").fillna(0.0)


def _sum_columns(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    score = pd.Series(0.0, index=df.index)
    for col in cols:
        score = score + _num(df, col)
    return score.astype(float)


def _positive_increase(df: pd.DataFrame, monthly_base: str, window: int) -> pd.Series:
    """Return max(0, current rolling count - previous rolling count)."""
    last_col = f"{monthly_base}_last_{window}m"
    prev_col = f"{monthly_base}_prev_{window}m"
    return (_num(df, last_col) - _num(df, prev_col)).clip(lower=0.0)


def _near_miss_hazard_count_score(df: pd.DataFrame, window: int) -> pd.Series:
    return _sum_columns(df, [
        f"near_miss_count_m_last_{window}m",
        f"hazard_identification_count_m_last_{window}m",
    ])


def _near_miss_hazard_increase_score(df: pd.DataFrame, window: int) -> pd.Series:
    return (
        _positive_increase(df, "near_miss_count_m", window)
        + _positive_increase(df, "hazard_identification_count_m", window)
    ).astype(float)


def _near_miss_hazard_trend_score(df: pd.DataFrame, window: int) -> pd.Series:
    """Simple leading-indicator score: recent volume plus positive increase."""
    return (_near_miss_hazard_count_score(df, window) + _near_miss_hazard_increase_score(df, window)).astype(float)


def build_rule_score_specs(windows: list[int]) -> list[dict[str, Any]]:
    """Define the simple non-ML methods to compare."""
    specs: list[dict[str, Any]] = []
    for window in windows:
        specs.append({
            "method": f"injury_count_last_{window}m",
            "method_group": "count_baseline",
            "description": f"Rank by injury incident count in the previous {window} months.",
            "score_function": lambda df, w=window: _num(df, f"injury_incident_count_m_last_{w}m"),
            "uses_ml": False,
            "uses_location_categorical": False,
            "uses_pattern_features": False,
        })
    for window in [w for w in windows if w >= 6]:
        specs.append({
            "method": f"severe_count_last_{window}m",
            "method_group": "severe_count_baseline",
            "description": f"Rank by severe actual injury count in the previous {window} months.",
            "score_function": lambda df, w=window: _num(df, f"severe_actual_count_m_last_{w}m"),
            "uses_ml": False,
            "uses_location_categorical": False,
            "uses_pattern_features": False,
        })
    for window in windows:
        specs.append({
            "method": f"near_miss_hazard_count_last_{window}m",
            "method_group": "near_miss_hazard_baseline",
            "description": f"Rank by near-miss plus hazard-identification count in the previous {window} months.",
            "score_function": lambda df, w=window: _near_miss_hazard_count_score(df, w),
            "uses_ml": False,
            "uses_location_categorical": False,
            "uses_pattern_features": False,
        })
        specs.append({
            "method": f"near_miss_hazard_increase_last_{window}m",
            "method_group": "near_miss_hazard_trend_baseline",
            "description": f"Rank by positive increase in near-miss plus hazard-identification counts over the previous {window} months.",
            "score_function": lambda df, w=window: _near_miss_hazard_increase_score(df, w),
            "uses_ml": False,
            "uses_location_categorical": False,
            "uses_pattern_features": False,
        })
        specs.append({
            "method": f"near_miss_hazard_trend_score_last_{window}m",
            "method_group": "near_miss_hazard_trend_baseline",
            "description": f"Rank by recent near-miss/hazard volume plus positive increase over the previous {window} months.",
            "score_function": lambda df, w=window: _near_miss_hazard_trend_score(df, w),
            "uses_ml": False,
            "uses_location_categorical": False,
            "uses_pattern_features": False,
        })
    return specs


# -----------------------------------------------------------------------------
# ML method helpers
# -----------------------------------------------------------------------------

def _split_from_test_start(df: pd.DataFrame, test_start: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    out["anchor_month"] = pd.to_datetime(out["anchor_month"])
    train = out[out["anchor_month"] < test_start].copy()
    test = out[out["anchor_month"] >= test_start].copy()
    return train, test


def _feature_columns_for_variant(
    df: pd.DataFrame,
    target: str,
    variant: str,
) -> tuple[list[str], list[str], list[str]]:
    """Return feature columns for full/no-location/location-only ML variants."""
    numeric_cols, categorical_cols = infer_feature_columns(df, target)
    variant = str(variant)

    if variant == "full":
        return numeric_cols, categorical_cols, numeric_cols + categorical_cols

    if variant == "no_location":
        location_set = set(LOCATION_CATEGORICAL_COLS)
        categorical_cols = [c for c in categorical_cols if c not in location_set]
        numeric_cols = [c for c in numeric_cols if c not in location_set]
        return numeric_cols, categorical_cols, numeric_cols + categorical_cols

    if variant == "location_only":
        categorical_cols = [c for c in LOCATION_CATEGORICAL_COLS if c in df.columns]
        calendar_cols = [c for c in ["calendar_month", "calendar_month_sin", "calendar_month_cos"] if c in df.columns]
        numeric_cols = calendar_cols
        return numeric_cols, categorical_cols, numeric_cols + categorical_cols

    raise ValueError(f"Unknown ML variant: {variant}")


def train_and_score_ml_method(
    df: pd.DataFrame,
    target: str,
    test_start: pd.Timestamp,
    method: str,
    method_group: str,
    description: str,
    variant: str,
    uses_pattern_features: bool,
    run_dir: Path,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    """Train a temporary holdout ML model and return metrics plus predictions."""
    train_df, test_df = _split_from_test_start(df, test_start)
    if train_df.empty or test_df.empty:
        raise ValueError(f"{method}: empty train/test split")
    if train_df[target].nunique() < 2:
        raise ValueError(f"{method}: training data has only one target class")

    numeric_cols, categorical_cols, feature_cols = _feature_columns_for_variant(train_df, target, variant)
    if not feature_cols:
        raise ValueError(f"{method}: no features available for variant={variant}")

    model_dir = ensure_dir(run_dir / "temporary_ml_models" / method)
    pipeline = build_pipeline(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        model_type=config.MODEL_TYPE,
        min_category_frequency=config.MIN_CATEGORY_FREQUENCY,
    )
    pipeline = fit_pipeline(pipeline, train_df[feature_cols], train_df[target].astype(int))

    train_score = predict_proba_positive(pipeline, train_df[feature_cols])
    test_score = predict_proba_positive(pipeline, test_df[feature_cols])
    threshold, threshold_table = select_threshold_from_scores(
        train_df[target].astype(int).to_numpy(),
        train_score,
        strategy=getattr(config, "THRESHOLD_STRATEGY", "top_percent"),
        top_percent=float(getattr(config, "TOP_PERCENT_THRESHOLD", 0.10)),
        fixed_threshold=float(getattr(config, "FIXED_THRESHOLD", 0.50)),
    )
    threshold_table.to_csv(model_dir / "threshold_selection.csv", index=False)
    joblib.dump(pipeline, model_dir / "temporary_model_fit_on_train_period.joblib")

    metrics = evaluate_score_method(test_df[target].astype(int).to_numpy(), test_score, threshold)
    metrics.update({
        "method": method,
        "method_group": method_group,
        "description": description,
        "score_used": "temporary_ml_predicted_probability",
        "uses_ml": True,
        "uses_location_categorical": bool(any(c in categorical_cols for c in LOCATION_CATEGORICAL_COLS)),
        "uses_pattern_features": bool(uses_pattern_features),
        "test_start_month": str(pd.Timestamp(test_start).date()),
        "test_end_month": str(pd.to_datetime(test_df["anchor_month"]).max().date()),
        "n_train_rows": int(len(train_df)),
        "n_test_rows": int(len(test_df)),
        "n_numeric_features": int(len(numeric_cols)),
        "n_categorical_features": int(len(categorical_cols)),
        "model_type_requested": config.MODEL_TYPE,
        "model_class_used": pipeline.named_steps["model"].__class__.__name__,
        "threshold_strategy": getattr(config, "THRESHOLD_STRATEGY", "top_percent"),
    })

    pred = test_df[["anchor_month", "site_name_filled", "department_name_filled", target]].copy()
    pred["method"] = method
    pred["score"] = test_score
    pred["rank"] = pred["score"].rank(method="first", ascending=False).astype(int)
    pred["predicted_label"] = (pred["score"] >= threshold).astype(int)

    manifest = {
        "method": method,
        "variant": variant,
        "target": target,
        "feature_columns": feature_cols,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "temporary_model_file": str(model_dir / "temporary_model_fit_on_train_period.joblib"),
        "metrics": metrics,
    }
    save_json(manifest, model_dir / "method_manifest.json")
    return metrics, pred, manifest


# -----------------------------------------------------------------------------
# Comparison workflow
# -----------------------------------------------------------------------------

def _ranking_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    """Rank methods across operational metrics."""
    ranked = comparison.copy()
    rank_specs = [
        ("pr_auc", False),
        ("recall_at_top_10pct", False),
        ("precision_at_top_10pct", False),
        ("lift_at_top_10pct", False),
        ("roc_auc", False),
        ("false_negative", True),
    ]
    rank_cols = []
    for metric, ascending in rank_specs:
        if metric in ranked.columns:
            col = f"rank_{metric}"
            ranked[col] = ranked[metric].rank(ascending=ascending, method="min")
            rank_cols.append(col)
    if rank_cols:
        ranked["overall_rank_score"] = ranked[rank_cols].mean(axis=1)
        sort_cols = ["overall_rank_score"]
        ascending = [True]
        if "pr_auc" in ranked.columns:
            sort_cols.append("pr_auc")
            ascending.append(False)
        ranked = ranked.sort_values(sort_cols, ascending=ascending)
    else:
        ranked["overall_rank_score"] = np.nan
    return ranked.reset_index(drop=True)


def _build_recommendation(comparison: pd.DataFrame, ranked: pd.DataFrame) -> dict[str, Any]:
    """Create a concise machine-readable interpretation of the comparison."""
    if comparison.empty:
        return {"best_method": None, "summary": "No methods were evaluated."}

    best = ranked.iloc[0].to_dict()
    simple = comparison[comparison["uses_ml"].eq(False)].copy()
    ml = comparison[comparison["uses_ml"].eq(True)].copy()

    best_simple = None
    best_ml = None
    if not simple.empty:
        best_simple = simple.sort_values(["recall_at_top_10pct", "precision_at_top_10pct", "pr_auc"], ascending=[False, False, False]).iloc[0].to_dict()
    if not ml.empty:
        best_ml = ml.sort_values(["recall_at_top_10pct", "precision_at_top_10pct", "pr_auc"], ascending=[False, False, False]).iloc[0].to_dict()

    delta = None
    if best_simple and best_ml:
        delta = {
            "top10_recall_delta_ml_minus_simple": float(best_ml.get("recall_at_top_10pct", np.nan) - best_simple.get("recall_at_top_10pct", np.nan)),
            "top10_precision_delta_ml_minus_simple": float(best_ml.get("precision_at_top_10pct", np.nan) - best_simple.get("precision_at_top_10pct", np.nan)),
            "pr_auc_delta_ml_minus_simple": float(best_ml.get("pr_auc", np.nan) - best_simple.get("pr_auc", np.nan)),
        }

    no_location = comparison[comparison["method"].str.contains("no_location", na=False)].copy()
    full_location = comparison[comparison["method"].str.contains("full_location", na=False)].copy()
    location_message = "No no-location ML method was evaluated."
    if not no_location.empty and not full_location.empty:
        best_no_loc = no_location.sort_values(["recall_at_top_10pct", "pr_auc"], ascending=[False, False]).iloc[0]
        best_full = full_location.sort_values(["recall_at_top_10pct", "pr_auc"], ascending=[False, False]).iloc[0]
        location_message = (
            "Compare full-location vs no-location rows to estimate whether the model is relying on site/department identity. "
            f"Best full-location top-10 recall={best_full.get('recall_at_top_10pct', np.nan):.3f}; "
            f"best no-location top-10 recall={best_no_loc.get('recall_at_top_10pct', np.nan):.3f}."
        )

    return {
        "best_method_by_rank": best.get("method"),
        "best_method_group_by_rank": best.get("method_group"),
        "best_simple_method_by_top10": None if best_simple is None else best_simple.get("method"),
        "best_ml_method_by_top10": None if best_ml is None else best_ml.get("method"),
        "ml_vs_simple_delta": delta,
        "location_dependency_note": location_message,
        "selection_guidance": (
            "Use recall_at_top_10pct, precision_at_top_10pct, lift_at_top_10pct, and PR-AUC as the primary method-selection metrics. "
            "Accuracy is less important for operational safety-risk ranking."
        ),
    }


def main() -> None:
    run_id = getattr(config, "RUN_ID", None) or now_run_id()
    run_dir = ensure_dir(Path(config.OUTPUT_DIR) / "ml" / "injury_risk_classification" / "method_comparison" / run_id)

    comparison_windows = sorted(set(
        int(x) for x in list(getattr(config, "ROLLING_WINDOWS", [3, 6])) + list(getattr(config, "COMPARISON_WINDOWS", DEFAULT_COMPARISON_WINDOWS))
    ))
    target_type = getattr(config, "TARGET_TYPE", "any_injury")
    target = target_col(config.HORIZON_MONTHS, target_type)

    clustered_path = Path(config.CLUSTERED_PATTERN_RECORDS_PATH)
    clustered_records = clustered_path if clustered_path.exists() else None
    if clustered_records is None:
        warnings.warn(f"Clustered/theme file not found at {clustered_path}. Pattern ML comparison rows will be skipped.")

    print("Building comparison dataset...")
    print("Target:", target)
    print("Rolling windows:", comparison_windows)
    bundle = build_classification_dataset(
        input_dir=config.INPUT_DIR,
        output_dir=config.OUTPUT_DIR,
        clustered_records_path=clustered_records,
        horizon_months=config.HORIZON_MONTHS,
        target_type=target_type,
        rolling_windows=comparison_windows,
        top_n_clusters=getattr(config, "TOP_N_CLUSTERS", 0),
        min_history_months=config.MIN_HISTORY_MONTHS,
        reference_date=config.REFERENCE_DATE,
        write_outputs=False,
        pattern_feature_config=config.get_pattern_feature_config(),
        pattern_feature_experiments=[],
    )
    save_json(bundle.metadata, run_dir / "comparison_dataset_metadata.json")

    base_df = bundle.baseline_dataset.copy()
    if target not in base_df.columns:
        raise ValueError(f"Target column not found in baseline dataset: {target}")
    base_df["anchor_month"] = pd.to_datetime(base_df["anchor_month"])
    _, base_test, test_start = make_time_holdout(base_df, int(config.TEST_MONTHS))
    base_train, base_test = _split_from_test_start(base_df, test_start)

    identity_cols = [c for c in ["anchor_month", "site_name_filled", "department_name_filled", target] if c in base_test.columns]
    wide_scores = base_test[identity_cols].copy().reset_index(drop=True)

    rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    threshold_rows: list[pd.DataFrame] = []

    # Rule/count baseline methods.
    print("Evaluating simple count/trend baselines...")
    for spec in build_rule_score_specs(comparison_windows):
        method = spec["method"]
        train_score = spec["score_function"](base_train).astype(float).to_numpy()
        test_score = spec["score_function"](base_test).astype(float).to_numpy()
        threshold, threshold_table = select_threshold_from_scores(
            base_train[target].astype(int).to_numpy(),
            train_score,
            strategy=getattr(config, "THRESHOLD_STRATEGY", "top_percent"),
            top_percent=float(getattr(config, "TOP_PERCENT_THRESHOLD", 0.10)),
            fixed_threshold=float(getattr(config, "FIXED_THRESHOLD", 0.50)),
        )
        threshold_table = threshold_table.copy()
        threshold_table["method"] = method
        threshold_rows.append(threshold_table)

        metrics = evaluate_score_method(base_test[target].astype(int).to_numpy(), test_score, threshold)
        metrics.update({
            "method": method,
            "method_group": spec["method_group"],
            "description": spec["description"],
            "score_used": method,
            "uses_ml": False,
            "uses_location_categorical": False,
            "uses_pattern_features": False,
            "test_start_month": str(pd.Timestamp(test_start).date()),
            "test_end_month": str(pd.to_datetime(base_test["anchor_month"]).max().date()),
            "n_train_rows": int(len(base_train)),
            "n_test_rows": int(len(base_test)),
            "n_numeric_features": 1,
            "n_categorical_features": 0,
            "model_type_requested": "rule_score",
            "model_class_used": "none",
            "threshold_strategy": getattr(config, "THRESHOLD_STRATEGY", "top_percent"),
        })
        rows.append(metrics)

        pred = base_test[identity_cols].copy()
        pred["method"] = method
        pred["score"] = test_score
        pred["rank"] = pd.Series(test_score).rank(method="first", ascending=False).astype(int).to_numpy()
        pred["predicted_label"] = (test_score >= threshold).astype(int)
        prediction_frames.append(pred)
        wide_scores[f"{method}_score"] = test_score
        wide_scores[f"{method}_rank"] = pred["rank"].to_numpy()

    # Temporary ML holdout models.
    print("Training temporary ML comparison models...")
    ml_datasets: list[tuple[str, pd.DataFrame, bool]] = [
        ("baseline", base_df, False),
    ]
    if bundle.with_cluster_dataset is not None and not bundle.with_cluster_dataset.empty:
        ml_datasets.append(("with_patterns", bundle.with_cluster_dataset.copy(), True))

    for dataset_name, df, uses_patterns in ml_datasets:
        for variant, suffix, description_suffix in [
            ("full", "full_location", "with location categorical features"),
            ("no_location", "no_location", "without site/department/region/country categorical features"),
        ]:
            method = f"ml_{dataset_name}_{suffix}"
            try:
                metrics, pred, manifest = train_and_score_ml_method(
                    df=df,
                    target=target,
                    test_start=test_start,
                    method=method,
                    method_group="ml_model",
                    description=f"Temporary {config.MODEL_TYPE} model using {dataset_name} features {description_suffix}.",
                    variant=variant,
                    uses_pattern_features=uses_patterns,
                    run_dir=run_dir,
                )
                rows.append(metrics)
                prediction_frames.append(pred)
                aligned = pred.sort_values(["anchor_month", "site_name_filled", "department_name_filled"]).reset_index(drop=True)
                # Merge instead of assuming identical row order.
                wide_scores = wide_scores.merge(
                    pred[["anchor_month", "site_name_filled", "department_name_filled", "score", "rank"]].rename(columns={
                        "score": f"{method}_score",
                        "rank": f"{method}_rank",
                    }),
                    on=["anchor_month", "site_name_filled", "department_name_filled"],
                    how="left",
                )
            except Exception as exc:
                warnings.warn(f"Skipping {method}: {exc}")

    # Optional location-only benchmark to quantify memorization of location risk.
    try:
        metrics, pred, manifest = train_and_score_ml_method(
            df=base_df,
            target=target,
            test_start=test_start,
            method="ml_location_only",
            method_group="location_only_ml_model",
            description="Temporary ML model using only site/department/region/business-unit/country plus calendar features.",
            variant="location_only",
            uses_pattern_features=False,
            run_dir=run_dir,
        )
        rows.append(metrics)
        prediction_frames.append(pred)
        wide_scores = wide_scores.merge(
            pred[["anchor_month", "site_name_filled", "department_name_filled", "score", "rank"]].rename(columns={
                "score": "ml_location_only_score",
                "rank": "ml_location_only_rank",
            }),
            on=["anchor_month", "site_name_filled", "department_name_filled"],
            how="left",
        )
    except Exception as exc:
        warnings.warn(f"Skipping ml_location_only: {exc}")

    comparison = pd.DataFrame(rows)
    if comparison.empty:
        raise ValueError("No comparison methods were evaluated.")

    # Put the most useful columns first while preserving all metrics.
    preferred = [
        "method", "method_group", "description", "score_used", "uses_ml", "uses_location_categorical", "uses_pattern_features",
        "n_rows", "n_positive", "positive_rate", "threshold", "roc_auc", "pr_auc", "brier_score",
        "recall_at_top_5pct", "precision_at_top_5pct", "lift_at_top_5pct", "positives_at_top_5pct",
        "recall_at_top_10pct", "precision_at_top_10pct", "lift_at_top_10pct", "positives_at_top_10pct",
        "recall_at_top_20pct", "precision_at_top_20pct", "lift_at_top_20pct", "positives_at_top_20pct",
        "accuracy", "balanced_accuracy", "precision", "recall", "f1", "false_negative", "false_positive", "true_positive", "true_negative",
        "test_start_month", "test_end_month", "n_train_rows", "n_test_rows", "n_numeric_features", "n_categorical_features",
        "model_type_requested", "model_class_used", "threshold_strategy",
    ]
    ordered_cols = [c for c in preferred if c in comparison.columns] + [c for c in comparison.columns if c not in preferred]
    comparison = comparison[ordered_cols]

    ranked = _ranking_summary(comparison)
    recommendation = _build_recommendation(comparison, ranked)

    comparison.to_csv(run_dir / "method_comparison_holdout.csv", index=False)
    ranked.to_csv(run_dir / "method_comparison_ranked.csv", index=False)
    if prediction_frames:
        pd.concat(prediction_frames, ignore_index=True).to_csv(run_dir / "method_predictions_long_holdout.csv", index=False)
    wide_scores.to_csv(run_dir / "method_scores_wide_holdout.csv", index=False)
    if threshold_rows:
        pd.concat(threshold_rows, ignore_index=True).to_csv(run_dir / "threshold_selection_by_method.csv", index=False)
    save_json(recommendation, run_dir / "method_comparison_recommendation.json")

    summary = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "target": target,
        "target_type": target_type,
        "horizon_months": int(config.HORIZON_MONTHS),
        "test_months": int(config.TEST_MONTHS),
        "test_start_month": str(pd.Timestamp(test_start).date()),
        "rolling_windows": comparison_windows,
        "clustered_records_path": str(clustered_records) if clustered_records else None,
        "outputs": {
            "method_comparison_holdout": str(run_dir / "method_comparison_holdout.csv"),
            "method_comparison_ranked": str(run_dir / "method_comparison_ranked.csv"),
            "method_predictions_long_holdout": str(run_dir / "method_predictions_long_holdout.csv"),
            "method_scores_wide_holdout": str(run_dir / "method_scores_wide_holdout.csv"),
            "recommendation": str(run_dir / "method_comparison_recommendation.json"),
        },
        "recommendation": recommendation,
    }
    save_json(summary, run_dir / "method_comparison_run_summary.json")

    display_cols = [c for c in [
        "method", "method_group", "pr_auc", "roc_auc", "recall_at_top_10pct", "precision_at_top_10pct",
        "lift_at_top_10pct", "false_negative", "uses_ml", "uses_location_categorical", "uses_pattern_features",
    ] if c in ranked.columns]
    print("\nSaved method comparison outputs to:", run_dir)
    print("\nRanked comparison:")
    print(ranked[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
