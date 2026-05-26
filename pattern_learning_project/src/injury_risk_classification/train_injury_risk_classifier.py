"""Train the simplified future any-injury risk ranking model with optional pattern features.

Run directly from the project root after editing config.py if needed:

    python src/injury_risk_classification/train_injury_risk_classifier.py

The default MVP compares two feature sets:

1. baseline: operational, injury-history, corrective-action, audit, and location features.
2. with_clusters: baseline + simple aggregate/theme pattern features.

The target defaults to future_any_injury_3m and the output is intended to be
used as a risk ranking, not a hard yes/no guarantee.

All tunable parameters and paths are configured in:

    src/injury_risk_classification/config.py

All artifacts are saved under outputs/ml/injury_risk_classification/runs/<run_id>/.
"""

from __future__ import annotations

import json
import re
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import joblib
import numpy as np
import pandas as pd

# Allow direct execution from project root without installing the package.
if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from injury_risk_classification import config
from injury_risk_classification.feature_engineering import build_classification_dataset
from injury_risk_classification.utils import ensure_dir, now_run_id, save_json, top_k_capture

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    fbeta_score,
    precision_recall_fscore_support,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


IDENTIFIER_AND_LEAKAGE_PATTERNS = (
    "future_",
    "has_complete_future_window",
    "eligible_for_modeling",
    "eligible_for_scoring",
)

DEFAULT_CATEGORICAL_COLS = [
    "site_name_filled",
    "department_name_filled",
    "region_name_filled",
    "business_unit_name_filled",
    "country_name_filled",
]

ALWAYS_EXCLUDE_COLS = {
    "anchor_month",
    "months_since_entity_start",
}


def load_args_from_config() -> SimpleNamespace:
    """Create the argument namespace used by the training functions from config.py.

    Keeping this adapter lets the rest of the training code stay clean while
    allowing you to run the script directly without command-line inputs.
    """
    return SimpleNamespace(
        input_dir=str(config.INPUT_DIR),
        output_dir=str(config.OUTPUT_DIR),
        clustered_records=str(config.CLUSTERED_PATTERN_RECORDS_PATH),
        feature_set=config.FEATURE_SET,
        model_type=config.MODEL_TYPE,
        target_type=getattr(config, "TARGET_TYPE", "any_injury"),
        horizon_months=config.HORIZON_MONTHS,
        rolling_windows_list=list(config.ROLLING_WINDOWS),
        top_n_clusters=config.TOP_N_CLUSTERS,
        min_history_months=config.MIN_HISTORY_MONTHS,
        test_months=config.TEST_MONTHS,
        cv_splits=config.CV_SPLITS,
        threshold_strategy=config.THRESHOLD_STRATEGY,
        top_percent_threshold=config.TOP_PERCENT_THRESHOLD,
        fixed_threshold=config.FIXED_THRESHOLD,
        min_category_frequency=config.MIN_CATEGORY_FREQUENCY,
        max_train_rows=config.MAX_TRAIN_ROWS,
        reference_date=config.REFERENCE_DATE,
        save_feature_datasets=config.SAVE_FEATURE_DATASETS_DURING_TRAINING,
        run_id=config.RUN_ID,
        pattern_feature_config=config.get_pattern_feature_config(),
        pattern_feature_experiments=config.get_pattern_feature_experiments(),
        save_model_input_feature_tables=getattr(config, "SAVE_MODEL_INPUT_FEATURE_TABLES", True),
        model_input_preview_rows=getattr(config, "MODEL_INPUT_PREVIEW_ROWS", 10000),
        leakage_validation_enabled=getattr(config, "LEAKAGE_VALIDATION_ENABLED", True),
        fail_on_leakage=getattr(config, "FAIL_ON_LEAKAGE", False),
        leakage_correlation_warning_threshold=getattr(config, "LEAKAGE_CORRELATION_WARNING_THRESHOLD", 0.98),
        leakage_auc_warning_threshold=getattr(config, "LEAKAGE_AUC_WARNING_THRESHOLD", 0.995),
    )

def target_col(horizon_months: int, target_type: str | None = None) -> str:
    """Return the configured Boolean target column name."""
    target_type = str(target_type or getattr(config, "TARGET_TYPE", "any_injury")).lower()
    if target_type in {"any_injury", "any", "injury"}:
        return f"future_any_injury_{horizon_months}m"
    if target_type in {"severe_actual", "severe", "sif_proxy"}:
        return f"future_severe_actual_{horizon_months}m"
    raise ValueError(f"Unsupported TARGET_TYPE={target_type!r}. Use 'any_injury' or 'severe_actual'.")


def is_leakage_or_identifier_col(col: str, target: str) -> bool:
    if col == target:
        return True
    if col in ALWAYS_EXCLUDE_COLS:
        return True
    return any(col.startswith(pattern) for pattern in IDENTIFIER_AND_LEAKAGE_PATTERNS)


def infer_feature_columns(df: pd.DataFrame, target: str) -> tuple[list[str], list[str]]:
    """Infer numeric and categorical feature columns from a classification dataset."""
    categorical_cols = [c for c in DEFAULT_CATEGORICAL_COLS if c in df.columns]
    exclude = set(categorical_cols)
    numeric_cols: list[str] = []
    for col in df.columns:
        if col in exclude or is_leakage_or_identifier_col(col, target):
            continue
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
            numeric_cols.append(col)
    return numeric_cols, categorical_cols


def make_one_hot_encoder(min_frequency: int, dense_output: bool) -> OneHotEncoder:
    """Create a version-compatible OneHotEncoder."""
    try:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=min_frequency, sparse_output=not dense_output)
    except TypeError:  # sklearn<1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=not dense_output)


def create_estimator(model_type: str):
    """Create a classifier using hyperparameters from config.py."""
    random_state = config.RANDOM_STATE
    if model_type in {"auto", "lightgbm"}:
        try:
            from lightgbm import LGBMClassifier
            params = dict(config.LIGHTGBM_PARAMS)
            params["random_state"] = random_state
            return LGBMClassifier(**params)
        except Exception as exc:
            if model_type == "lightgbm":
                raise ImportError("LightGBM was requested but could not be imported. Install lightgbm or set MODEL_TYPE='auto' or another sklearn model in config.py.") from exc
            warnings.warn("LightGBM is not installed; falling back to sklearn SGDClassifier logistic baseline. Install lightgbm for the recommended gradient-boosting model.")
            from sklearn.linear_model import SGDClassifier
            params = dict(config.SGD_LOGISTIC_PARAMS)
            params["random_state"] = random_state
            return SGDClassifier(**params)
    if model_type == "hist_gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingClassifier
        params = dict(config.HIST_GRADIENT_BOOSTING_PARAMS)
        params["random_state"] = random_state
        try:
            params["class_weight"] = "balanced"
            return HistGradientBoostingClassifier(**params)
        except TypeError:
            params.pop("class_weight", None)
            return HistGradientBoostingClassifier(**params)
    if model_type == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(**dict(config.LOGISTIC_REGRESSION_PARAMS))
    if model_type == "sgd_logistic":
        from sklearn.linear_model import SGDClassifier
        params = dict(config.SGD_LOGISTIC_PARAMS)
        params["random_state"] = random_state
        return SGDClassifier(**params)
    raise ValueError(f"Unsupported model_type: {model_type}")

def build_pipeline(numeric_cols: list[str], categorical_cols: list[str], model_type: str, min_category_frequency: int) -> Pipeline:
    """Build preprocessing + classifier pipeline."""
    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),
    ])
    dense_output = model_type == "hist_gradient_boosting"
    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", make_one_hot_encoder(min_category_frequency, dense_output=dense_output)),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0 if dense_output else 1.0,
        verbose_feature_names_out=True,
    )
    model = create_estimator(model_type)
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def get_feature_names(pipeline: Pipeline) -> list[str]:
    """Extract transformed feature names from fitted pipeline."""
    try:
        return list(pipeline.named_steps["preprocess"].get_feature_names_out())
    except Exception:
        return []


def predict_proba_positive(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Return positive-class probabilities from a fitted classifier pipeline."""
    model = pipeline.named_steps["model"]
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(X)
        if proba.shape[1] == 1:
            return np.zeros(len(X)) if getattr(model, "classes_", [0])[0] == 0 else np.ones(len(X))
        return proba[:, 1]
    if hasattr(pipeline, "decision_function"):
        scores = pipeline.decision_function(X)
        return 1 / (1 + np.exp(-scores))
    preds = pipeline.predict(X)
    return np.asarray(preds, dtype=float)


def fit_pipeline(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> Pipeline:
    """Fit with balanced sample weights where supported."""
    sample_weight = compute_sample_weight(class_weight="balanced", y=y.astype(int))
    try:
        pipeline.fit(X, y, model__sample_weight=sample_weight)
    except TypeError:
        pipeline.fit(X, y)
    return pipeline


def safe_auc(y_true: np.ndarray, y_score: np.ndarray, auc_type: str) -> float:
    """AUC metrics are undefined when y_true has one class."""
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return np.nan
    if auc_type == "roc":
        return float(roc_auc_score(y_true, y_score))
    if auc_type == "pr":
        return float(average_precision_score(y_true, y_score))
    raise ValueError(auc_type)


def evaluate_predictions(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, Any]:
    """Calculate classification and ranking metrics."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = {
        "n_rows": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "positive_rate": float(y_true.mean()) if len(y_true) else np.nan,
        "threshold": float(threshold),
        "roc_auc": safe_auc(y_true, y_score, "roc"),
        "pr_auc": safe_auc(y_true, y_score, "pr"),
        "brier_score": float(brier_score_loss(y_true, y_score)) if len(np.unique(y_true)) > 1 else np.nan,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else np.nan,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }
    for frac in [0.05, 0.10, 0.20]:
        capture = top_k_capture(y_true, y_score, frac)
        metrics[f"recall_at_top_{int(frac*100)}pct"] = capture["recall_at_top"]
        metrics[f"precision_at_top_{int(frac*100)}pct"] = capture["precision_at_top"]
        metrics[f"lift_at_top_{int(frac*100)}pct"] = capture.get("lift_at_top")
        metrics[f"positives_at_top_{int(frac*100)}pct"] = capture["positives_in_top"]
    return metrics


def select_threshold(y_true: np.ndarray, y_score: np.ndarray, strategy: str, top_percent: float, fixed_threshold: float) -> tuple[float, pd.DataFrame]:
    """Select an operating threshold from CV predictions."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if strategy == "fixed":
        return float(fixed_threshold), pd.DataFrame({"threshold": [fixed_threshold], "strategy": ["fixed"]})
    if strategy == "top_percent":
        threshold = float(np.quantile(y_score, 1.0 - top_percent)) if len(y_score) else fixed_threshold
        return threshold, pd.DataFrame({"threshold": [threshold], "strategy": ["top_percent"], "top_percent": [top_percent]})
    # Default: maximize F2 to favor recall more than precision.
    rows = []
    for threshold in np.linspace(0.01, 0.99, 99):
        y_pred = (y_score >= threshold).astype(int)
        rows.append({
            "threshold": threshold,
            "f2": fbeta_score(y_true, y_pred, beta=2.0, zero_division=0),
            "predicted_positive_rate": y_pred.mean(),
        })
    table = pd.DataFrame(rows)
    best = table.sort_values(["f2", "predicted_positive_rate"], ascending=[False, True]).iloc[0]
    return float(best["threshold"]), table


def make_time_holdout(df: pd.DataFrame, test_months: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Split latest N months as hold-out test set."""
    months = pd.Series(pd.to_datetime(df["anchor_month"].unique())).sort_values()
    if len(months) <= test_months + 3:
        raise ValueError(f"Not enough months ({len(months)}) for a {test_months}-month holdout plus training.")
    test_start = months.iloc[-test_months]
    train = df[pd.to_datetime(df["anchor_month"]) < test_start].copy()
    test = df[pd.to_datetime(df["anchor_month"]) >= test_start].copy()
    return train, test, pd.Timestamp(test_start)


def generate_time_series_cv_splits(df: pd.DataFrame, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create expanding time-series CV splits by unique anchor months."""
    months = pd.Series(pd.to_datetime(df["anchor_month"].unique())).sort_values().reset_index(drop=True)
    n_splits = min(n_splits, max(2, len(months) - 2))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    splits = []
    month_values = months.values
    for train_month_idx, val_month_idx in tscv.split(month_values):
        train_months = set(month_values[train_month_idx])
        val_months = set(month_values[val_month_idx])
        train_idx = df.index[pd.to_datetime(df["anchor_month"]).isin(train_months)].to_numpy()
        val_idx = df.index[pd.to_datetime(df["anchor_month"]).isin(val_months)].to_numpy()
        if len(train_idx) and len(val_idx):
            splits.append((train_idx, val_idx))
    return splits



def _raw_feature_from_transformed_name(feature: str, categorical_cols: list[str]) -> str:
    """Map sklearn transformed feature names back to raw dataframe columns."""
    text = str(feature)
    if text.startswith("num__"):
        return text.replace("num__", "", 1)
    if text.startswith("cat__"):
        body = text.replace("cat__", "", 1)
        for col in sorted(categorical_cols, key=len, reverse=True):
            if body == col or body.startswith(f"{col}_"):
                return col
        return body.split("_")[0] if body else text
    return text


def _feature_family(raw_feature: str) -> str:
    """Assign a broad family label used in dashboard and feature-comparison outputs."""
    name = str(raw_feature).lower()
    if name in DEFAULT_CATEGORICAL_COLS:
        return "location_categorical"
    if "top_theme" in name or "unique_theme" in name or "theme" in name:
        return "pattern_theme"
    if "top_cluster" in name or "unique_cluster" in name or "cluster" in name:
        return "pattern_cluster"
    if "pattern" in name or "membership" in name or "outlier" in name:
        return "pattern_aggregate"
    if "task" in name or "overdue" in name or "closure" in name:
        return "corrective_action"
    if "audit" in name or "observation" in name or "inspection" in name or "risk_assessment" in name:
        return "audit_observation"
    if "injury" in name or "severe" in name:
        return "injury_history"
    if "near_miss" in name or "hazard" in name or "incident" in name:
        return "incident_history"
    if "calendar" in name:
        return "calendar"
    return "other"


def _write_dataframe_with_preview(df: pd.DataFrame, output_base: Path, preview_rows: int = 10000) -> dict[str, str]:
    """Write a dataframe as parquet/pickle plus a CSV preview."""
    output_base = Path(output_base)
    ensure_dir(output_base.parent)
    paths: dict[str, str] = {}
    try:
        full_path = output_base.with_suffix(".parquet")
        df.to_parquet(full_path, index=False)
        paths["full"] = str(full_path)
        paths["format"] = "parquet"
    except Exception:
        full_path = output_base.with_suffix(".pkl")
        df.to_pickle(full_path)
        paths["full"] = str(full_path)
        paths["format"] = "pickle"
    preview_path = output_base.with_name(output_base.name + f"_preview_{int(preview_rows)}_rows").with_suffix(".csv")
    df.head(int(preview_rows)).to_csv(preview_path, index=False)
    paths["preview_csv"] = str(preview_path)
    paths["n_rows"] = str(len(df))
    paths["n_cols"] = str(df.shape[1])
    return paths


def build_feature_catalog(df: pd.DataFrame, target: str, numeric_cols: list[str], categorical_cols: list[str]) -> pd.DataFrame:
    """Create a human-readable catalog of raw pre-encoding features used by the model."""
    selected = set(numeric_cols + categorical_cols)
    rows = []
    for col in df.columns:
        if col == target:
            role = "target"
        elif col in selected:
            role = "model_feature"
        elif is_leakage_or_identifier_col(col, target):
            role = "excluded_leakage_or_identifier"
        else:
            role = "not_selected"
        s = df[col]
        row = {
            "column": col,
            "role": role,
            "feature_family": _feature_family(col),
            "dtype": str(s.dtype),
            "is_numeric_feature": bool(col in numeric_cols),
            "is_categorical_feature": bool(col in categorical_cols),
            "missing_rate": float(s.isna().mean()) if len(s) else np.nan,
            "n_unique": int(s.nunique(dropna=True)) if len(s) else 0,
        }
        if col in numeric_cols:
            values = pd.to_numeric(s, errors="coerce")
            row.update({
                "mean": float(values.mean()) if values.notna().any() else np.nan,
                "median": float(values.median()) if values.notna().any() else np.nan,
                "p75": float(values.quantile(0.75)) if values.notna().any() else np.nan,
                "p90": float(values.quantile(0.90)) if values.notna().any() else np.nan,
                "max": float(values.max()) if values.notna().any() else np.nan,
                "nonzero_rate": float(values.fillna(0).ne(0).mean()) if len(values) else np.nan,
            })
        rows.append(row)
    catalog = pd.DataFrame(rows)
    order = {"target": 0, "model_feature": 1, "excluded_leakage_or_identifier": 2, "not_selected": 3}
    catalog["role_order"] = catalog["role"].map(order).fillna(99)
    return catalog.sort_values(["role_order", "feature_family", "column"]).drop(columns="role_order").reset_index(drop=True)


def compute_feature_reference_stats(df: pd.DataFrame, numeric_cols: list[str]) -> dict[str, dict[str, float]]:
    """Store reference statistics used later for dashboard driver explanations."""
    stats: dict[str, dict[str, float]] = {}
    for col in numeric_cols:
        values = pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(dtype=float)
        if values.notna().sum() == 0:
            stats[col] = {"median": 0.0, "mean": 0.0, "std": 0.0, "p75": 0.0, "p90": 0.0, "max": 0.0, "nonzero_rate": 0.0}
            continue
        stats[col] = {
            "median": float(values.median()),
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0) if values.notna().sum() > 1 else 0.0),
            "p75": float(values.quantile(0.75)),
            "p90": float(values.quantile(0.90)),
            "max": float(values.max()),
            "nonzero_rate": float(values.fillna(0).ne(0).mean()),
        }
    return stats


def save_raw_model_input_tables(
    feature_dir: Path,
    df_all: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    target: str,
    preview_rows: int,
) -> dict[str, dict[str, str]]:
    """Save exact pre-encoding model input tables for QA and dashboard inspection."""
    identity_cols = [c for c in [
        "anchor_month", "site_name_filled", "department_name_filled", "region_name_filled",
        "business_unit_name_filled", "country_name_filled", target,
    ] if c in df_all.columns]
    ordered_cols = []
    for col in identity_cols + feature_cols:
        if col in df_all.columns and col not in ordered_cols:
            ordered_cols.append(col)
    out_dir = ensure_dir(feature_dir / "raw_model_input_features")
    paths = {
        "all_eligible_rows": _write_dataframe_with_preview(df_all[ordered_cols].copy(), out_dir / "model_input_features_raw_all_eligible_rows", preview_rows),
        "train_period_rows": _write_dataframe_with_preview(train_df[ordered_cols].copy(), out_dir / "model_input_features_raw_train_period", preview_rows),
        "holdout_test_rows": _write_dataframe_with_preview(test_df[ordered_cols].copy(), out_dir / "model_input_features_raw_holdout_test", preview_rows),
    }
    return paths


def validate_leakage(
    df: pd.DataFrame,
    feature_cols: list[str],
    target: str,
    output_dir: Path,
    correlation_threshold: float,
    auc_threshold: float,
) -> dict[str, Any]:
    """Run conservative leakage checks on raw pre-encoding model features."""
    ensure_dir(output_dir)
    y = df[target].astype(int).to_numpy() if target in df.columns else np.asarray([])
    rows: list[dict[str, Any]] = []
    severe_issue_count = 0

    for col in feature_cols:
        issue_types: list[str] = []
        severity = "ok"
        if is_leakage_or_identifier_col(col, target):
            issue_types.append("forbidden_name_pattern_in_selected_features")
            severity = "error"
        if col == target:
            issue_types.append("target_column_selected_as_feature")
            severity = "error"

        row: dict[str, Any] = {
            "feature": col,
            "feature_family": _feature_family(col),
            "dtype": str(df[col].dtype) if col in df.columns else "missing",
            "selected_as_feature": True,
            "severity": severity,
            "issue_types": ";".join(issue_types),
            "missing_rate": float(df[col].isna().mean()) if col in df.columns and len(df) else np.nan,
            "n_unique": int(df[col].nunique(dropna=True)) if col in df.columns else 0,
            "abs_corr_with_target": np.nan,
            "single_feature_auc": np.nan,
            "equal_to_target": False,
        }

        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]) and len(y) and len(np.unique(y)) > 1:
            values = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            if np.nanstd(values) > 0:
                corr = np.corrcoef(values, y)[0, 1]
                if np.isfinite(corr):
                    row["abs_corr_with_target"] = float(abs(corr))
                    if abs(corr) >= correlation_threshold:
                        issue_types.append("very_high_correlation_with_target")
                        severity = "error" if abs(corr) >= 0.999 else "warning"
                try:
                    auc = roc_auc_score(y, values)
                    auc = max(float(auc), float(1.0 - auc))
                    row["single_feature_auc"] = auc
                    if auc >= auc_threshold:
                        issue_types.append("near_perfect_single_feature_auc")
                        severity = "error" if auc >= 0.999 else "warning"
                except Exception:
                    pass
            if set(np.unique(values)).issubset({0.0, 1.0}) and len(values) == len(y):
                equal = bool(np.array_equal(values.astype(int), y.astype(int)))
                row["equal_to_target"] = equal
                if equal:
                    issue_types.append("feature_exactly_equals_target")
                    severity = "error"

        row["severity"] = severity
        row["issue_types"] = ";".join(issue_types)
        if severity == "error":
            severe_issue_count += 1
        rows.append(row)

    report = pd.DataFrame(rows).sort_values(["severity", "abs_corr_with_target", "single_feature_auc"], ascending=[True, False, False])
    report.to_csv(output_dir / "leakage_validation_report.csv", index=False)
    summary = {
        "target": target,
        "row_count": int(len(df)),
        "feature_count": int(len(feature_cols)),
        "error_count": int((report["severity"] == "error").sum()) if not report.empty else 0,
        "warning_count": int((report["severity"] == "warning").sum()) if not report.empty else 0,
        "passed": bool(severe_issue_count == 0),
        "correlation_warning_threshold": float(correlation_threshold),
        "auc_warning_threshold": float(auc_threshold),
        "report_file": str(output_dir / "leakage_validation_report.csv"),
    }
    save_json(summary, output_dir / "leakage_validation_summary.json")
    return summary


def save_feature_importance(pipeline: Pipeline, output_path: Path, categorical_cols: list[str] | None = None) -> pd.DataFrame:
    """Save transformed and raw-column feature importance when available."""
    categorical_cols = categorical_cols or []
    names = get_feature_names(pipeline)
    model = pipeline.named_steps["model"]
    values = None
    importance_type = None
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
        importance_type = "model_feature_importance"
    elif hasattr(model, "coef_"):
        values = model.coef_[0]
        importance_type = "coefficient"
    if values is None or not len(names):
        empty = pd.DataFrame(columns=["feature", "raw_feature", "importance", "importance_type", "feature_family"])
        empty.to_csv(output_path, index=False)
        empty.to_csv(output_path.with_name("feature_importance_raw.csv"), index=False)
        return empty
    n = min(len(names), len(values))
    imp = pd.DataFrame({"feature": names[:n], "importance": values[:n], "importance_type": importance_type})
    imp["raw_feature"] = imp["feature"].map(lambda x: _raw_feature_from_transformed_name(x, categorical_cols))
    imp["feature_family"] = imp["raw_feature"].map(_feature_family)
    imp["abs_importance"] = imp["importance"].abs()
    imp = imp.sort_values("abs_importance", ascending=False)
    imp.drop(columns="abs_importance").to_csv(output_path, index=False)

    raw = (
        imp.groupby(["raw_feature", "feature_family", "importance_type"], dropna=False)["importance"]
        .agg(lambda s: float(np.sum(np.abs(s))))
        .reset_index()
        .sort_values("importance", ascending=False)
    )
    raw.to_csv(output_path.with_name("feature_importance_raw.csv"), index=False)
    family = raw.groupby("feature_family", dropna=False)["importance"].sum().reset_index().sort_values("importance", ascending=False)
    family.to_csv(output_path.with_name("feature_importance_by_family.csv"), index=False)
    return raw

def plot_diagnostics(y_true: np.ndarray, y_score: np.ndarray, threshold: float, output_dir: Path, prefix: str) -> None:
    """Save ROC, PR, and score distribution plots."""
    if plt is None:
        return
    output_dir = ensure_dir(output_dir)
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        fig = plt.figure(figsize=(7, 5))
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{prefix} ROC Curve")
        fig.tight_layout()
        fig.savefig(output_dir / f"{prefix}_roc_curve.png", dpi=150)
        plt.close(fig)

        precision, recall, _ = precision_recall_curve(y_true, y_score)
        fig = plt.figure(figsize=(7, 5))
        plt.plot(recall, precision)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"{prefix} Precision-Recall Curve")
        fig.tight_layout()
        fig.savefig(output_dir / f"{prefix}_precision_recall_curve.png", dpi=150)
        plt.close(fig)

    fig = plt.figure(figsize=(7, 5))
    plt.hist(y_score[y_true == 0], bins=30, alpha=0.7, label="No future target event")
    plt.hist(y_score[y_true == 1], bins=30, alpha=0.7, label="Future target event")
    plt.axvline(threshold, linestyle="--", label="Selected threshold")
    plt.xlabel("Predicted probability")
    plt.ylabel("Rows")
    plt.title(f"{prefix} Score Distribution")
    plt.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_score_distribution.png", dpi=150)
    plt.close(fig)


def train_one_feature_set(
    df: pd.DataFrame,
    feature_set_name: str,
    args: SimpleNamespace,
    run_dir: Path,
    pattern_feature_config: dict | None = None,
) -> dict[str, Any]:
    """Train, cross-validate, hold-out test, and save one feature-set model."""
    target = target_col(args.horizon_months, args.target_type)
    if target not in df.columns:
        raise ValueError(f"Target column not found: {target}")
    feature_dir = ensure_dir(run_dir / feature_set_name)
    df = df.copy()
    df["anchor_month"] = pd.to_datetime(df["anchor_month"])
    if args.max_train_rows:
        # Development-only sampling: keep the most recent rows so the sample is more
        # likely to include recent positive target examples. Do not use for final runs.
        df = df.sort_values("anchor_month").tail(args.max_train_rows).copy()
    train_df, test_df, test_start = make_time_holdout(df, args.test_months)
    numeric_cols, categorical_cols = infer_feature_columns(df, target)
    feature_cols = numeric_cols + categorical_cols

    # Save clean, pre-encoding feature QA artifacts. These are the exact raw
    # dataframe columns that go into sklearn before imputation, scaling, and
    # categorical one-hot encoding.
    feature_catalog = build_feature_catalog(df, target, numeric_cols, categorical_cols)
    feature_catalog.to_csv(feature_dir / "model_feature_catalog_raw_columns.csv", index=False)

    raw_feature_table_paths = {}
    if getattr(args, "save_model_input_feature_tables", True):
        raw_feature_table_paths = save_raw_model_input_tables(
            feature_dir=feature_dir,
            df_all=df,
            train_df=train_df,
            test_df=test_df,
            feature_cols=feature_cols,
            target=target,
            preview_rows=int(getattr(args, "model_input_preview_rows", 10000)),
        )

    leakage_summary = {}
    if getattr(args, "leakage_validation_enabled", True):
        leakage_summary = validate_leakage(
            df=train_df,
            feature_cols=feature_cols,
            target=target,
            output_dir=feature_dir / "leakage_validation",
            correlation_threshold=float(getattr(args, "leakage_correlation_warning_threshold", 0.98)),
            auc_threshold=float(getattr(args, "leakage_auc_warning_threshold", 0.995)),
        )
        if bool(getattr(args, "fail_on_leakage", False)) and not leakage_summary.get("passed", True):
            raise ValueError(
                f"Leakage validation failed for {feature_set_name}. "
                f"See {leakage_summary.get('report_file')}"
            )

    if not numeric_cols and not categorical_cols:
        raise ValueError("No feature columns inferred.")
    if train_df[target].nunique() < 2:
        raise ValueError(f"Training data for {feature_set_name} has only one target class. Increase history or reduce test months.")

    X_train_all = train_df[feature_cols]
    y_train_all = train_df[target].astype(int)
    X_test = test_df[feature_cols]
    y_test = test_df[target].astype(int)

    cv_predictions = []
    cv_metrics = []
    splits = generate_time_series_cv_splits(train_df.reset_index(drop=True), args.cv_splits)
    train_reset = train_df.reset_index(drop=True)
    for fold, (tr_idx, val_idx) in enumerate(splits, start=1):
        fold_train = train_reset.iloc[tr_idx]
        fold_val = train_reset.iloc[val_idx]
        if fold_train[target].nunique() < 2:
            warnings.warn(f"Skipping fold {fold}: training fold has one target class.")
            continue
        pipeline = build_pipeline(numeric_cols, categorical_cols, args.model_type, args.min_category_frequency)
        pipeline = fit_pipeline(pipeline, fold_train[feature_cols], fold_train[target].astype(int))
        val_score = predict_proba_positive(pipeline, fold_val[feature_cols])
        fold_pred = fold_val[["anchor_month", "site_name_filled", "department_name_filled", target]].copy()
        fold_pred["fold"] = fold
        fold_pred["score"] = val_score
        cv_predictions.append(fold_pred)
        fold_metrics = evaluate_predictions(fold_val[target].astype(int).to_numpy(), val_score, threshold=0.5)
        fold_metrics.update({
            "fold": fold,
            "val_start_month": str(pd.to_datetime(fold_val["anchor_month"]).min().date()),
            "val_end_month": str(pd.to_datetime(fold_val["anchor_month"]).max().date()),
        })
        cv_metrics.append(fold_metrics)

    if cv_predictions:
        cv_pred_df = pd.concat(cv_predictions, ignore_index=True)
    else:
        cv_pred_df = pd.DataFrame(columns=["anchor_month", "site_name_filled", "department_name_filled", target, "fold", "score"])
    cv_pred_df.to_csv(feature_dir / "cv_predictions.csv", index=False)
    cv_metrics_df = pd.DataFrame(cv_metrics)
    cv_metrics_df.to_csv(feature_dir / "cv_metrics_by_fold_threshold_0_5.csv", index=False)

    if len(cv_pred_df) and cv_pred_df[target].nunique() > 1:
        selected_threshold, threshold_table = select_threshold(
            cv_pred_df[target].astype(int).to_numpy(),
            cv_pred_df["score"].to_numpy(),
            strategy=args.threshold_strategy,
            top_percent=args.top_percent_threshold,
            fixed_threshold=args.fixed_threshold,
        )
    else:
        selected_threshold, threshold_table = args.fixed_threshold, pd.DataFrame({"threshold": [args.fixed_threshold], "strategy": ["fallback_fixed"]})
    threshold_table.to_csv(feature_dir / "threshold_selection.csv", index=False)

    final_test_pipeline = build_pipeline(numeric_cols, categorical_cols, args.model_type, args.min_category_frequency)
    final_test_pipeline = fit_pipeline(final_test_pipeline, X_train_all, y_train_all)
    test_score = predict_proba_positive(final_test_pipeline, X_test)
    test_metrics = evaluate_predictions(y_test.to_numpy(), test_score, selected_threshold)
    test_metrics.update({
        "feature_set": feature_set_name,
        "test_start_month": str(test_start.date()),
        "test_end_month": str(pd.to_datetime(test_df["anchor_month"]).max().date()),
        "n_train_rows": int(len(train_df)),
        "n_test_rows": int(len(test_df)),
        "n_numeric_features": int(len(numeric_cols)),
        "n_categorical_features": int(len(categorical_cols)),
        "model_type_requested": args.model_type,
        "model_class_used": final_test_pipeline.named_steps["model"].__class__.__name__,
    })
    pd.DataFrame([test_metrics]).to_csv(feature_dir / "holdout_test_metrics.csv", index=False)

    test_pred_df = test_df[["anchor_month", "site_name_filled", "department_name_filled", target]].copy()
    test_pred_df["score"] = test_score
    test_pred_df["predicted_label"] = (test_score >= selected_threshold).astype(int)
    test_pred_df["risk_rank_in_test"] = test_pred_df["score"].rank(method="first", ascending=False).astype(int)
    test_pred_df.to_csv(feature_dir / "holdout_test_predictions.csv", index=False)

    # Save final model fitted on pre-holdout data for test reproducibility.
    joblib.dump(final_test_pipeline, feature_dir / "model_fit_on_train_period.joblib")
    save_feature_importance(final_test_pipeline, feature_dir / "feature_importance.csv", categorical_cols=categorical_cols)
    plot_diagnostics(y_test.to_numpy(), test_score, selected_threshold, feature_dir / "plots", prefix="holdout_test")

    # Refit production model on all eligible data after validation/testing.
    production_pipeline = build_pipeline(numeric_cols, categorical_cols, args.model_type, args.min_category_frequency)
    production_pipeline = fit_pipeline(production_pipeline, df[feature_cols], df[target].astype(int))
    joblib.dump(production_pipeline, feature_dir / "model_final_refit_all_eligible_data.joblib")
    save_feature_importance(production_pipeline, feature_dir / "feature_importance_final_model.csv", categorical_cols=categorical_cols)

    feature_reference_stats = compute_feature_reference_stats(train_df, numeric_cols)

    manifest = {
        "feature_set": feature_set_name,
        "row_definition": "site + department + month",
        "target": target,
        "target_mean_all": float(df[target].mean()),
        "selected_threshold": float(selected_threshold),
        "threshold_strategy": args.threshold_strategy,
        "feature_columns": feature_cols,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "pattern_feature_config": pattern_feature_config,
        "raw_model_input_feature_tables": raw_feature_table_paths,
        "feature_catalog_file": str(feature_dir / "model_feature_catalog_raw_columns.csv"),
        "leakage_validation_summary": leakage_summary,
        "feature_reference_stats": feature_reference_stats,
        "test_metrics": test_metrics,
        "recommended_primary_metric": "recall_at_top_10pct, precision_at_top_10pct, PR-AUC, and lift for operational risk ranking",
        "target_type": args.target_type,
        "model_files": {
            "test_period_model": str(feature_dir / "model_fit_on_train_period.joblib"),
            "final_refit_model": str(feature_dir / "model_final_refit_all_eligible_data.joblib"),
        },
    }
    save_json(manifest, feature_dir / "model_manifest.json")
    return manifest


def _pattern_features_requested(feature_set: str) -> bool:
    return feature_set in {"with_clusters", "both", "experiments", "all"}


def _pattern_config_with_fixed_ids(pattern_config: dict | None, feature_map: dict | None) -> dict | None:
    """Attach selected top theme/cluster IDs to a config for stable scoring.

    During training, feature engineering chooses top pattern IDs by frequency.
    For production scoring, we want the same columns even if the latest data
    changes frequency ranking, so the chosen IDs are stored in the manifest.
    """
    if pattern_config is None:
        return None
    cfg = dict(pattern_config)
    top = (feature_map or {}).get("top_pattern_ids", {}) if isinstance(feature_map, dict) else {}
    fixed: dict[str, list[int]] = {}
    for level, level_map in top.items():
        if isinstance(level_map, dict):
            fixed[str(level)] = [int(x) for x in level_map.keys()]
    if fixed:
        cfg["fixed_pattern_ids"] = fixed
    return cfg



def save_feature_set_comparison_outputs(comparison: pd.DataFrame, run_dir: Path) -> dict[str, Any]:
    """Create ranked model-comparison artifacts for feature-set selection."""
    if comparison.empty:
        summary = {"best_feature_set": None, "reason": "No feature sets were trained."}
        save_json(summary, run_dir / "feature_set_recommendation.json")
        return summary

    ranked = comparison.copy()
    # Rank most operational metrics in descending order. Missing metrics are ignored.
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
        ranked = ranked.sort_values(["overall_rank_score", "pr_auc" if "pr_auc" in ranked.columns else "feature_set"], ascending=[True, False])
    else:
        ranked["overall_rank_score"] = np.nan
    ranked.to_csv(run_dir / "feature_set_comparison_ranked.csv", index=False)

    best = ranked.iloc[0].to_dict()
    summary = {
        "best_feature_set": best.get("feature_set"),
        "selection_rule": "Lowest average rank across PR-AUC, top-10 recall/precision/lift, ROC-AUC, and false negatives.",
        "recommended_operational_metric": "Prioritize recall_at_top_10pct, precision_at_top_10pct, lift_at_top_10pct, and PR-AUC over accuracy.",
        "best_metrics": {k: best.get(k) for k in [
            "pr_auc", "roc_auc", "recall_at_top_10pct", "precision_at_top_10pct",
            "lift_at_top_10pct", "false_negative", "threshold",
        ] if k in best},
        "ranked_comparison_file": str(run_dir / "feature_set_comparison_ranked.csv"),
    }
    save_json(summary, run_dir / "feature_set_recommendation.json")
    return summary

def main() -> None:
    args = load_args_from_config()
    rolling_windows = [int(x) for x in args.rolling_windows_list]
    run_id = args.run_id or now_run_id()
    run_dir = ensure_dir(Path(args.output_dir) / "ml" / "injury_risk_classification" / "runs" / run_id)

    clustered_path = Path(args.clustered_records)
    clustering_requested = _pattern_features_requested(args.feature_set)
    if clustering_requested and not clustered_path.exists() and config.REQUIRE_CLUSTERED_RECORDS_FOR_CLUSTER_FEATURES:
        raise FileNotFoundError(
            f"Pattern features were requested, but the clustered/theme pattern file does not exist: {clustered_path}\n"
            "Run the unsupervised pipeline first: "
            "python src/pattern_learning_unsupervised/train_pattern_clusters_hdbscan.py"
        )
    if clustering_requested and not clustered_path.exists():
        warnings.warn(f"Clustered/theme records file not found: {clustered_path}. Pattern feature sets will be skipped.")
        clustered_records = None
    elif clustering_requested:
        clustered_records = clustered_path
    else:
        clustered_records = None

    experiments = args.pattern_feature_experiments if args.feature_set in {"experiments", "all"} else []
    bundle = build_classification_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        clustered_records_path=clustered_records,
        horizon_months=args.horizon_months,
        target_type=args.target_type,
        rolling_windows=rolling_windows,
        top_n_clusters=args.top_n_clusters,
        min_history_months=args.min_history_months,
        reference_date=args.reference_date,
        write_outputs=args.save_feature_datasets,
        pattern_feature_config=args.pattern_feature_config,
        pattern_feature_experiments=experiments,
    )
    save_json(bundle.metadata, run_dir / "dataset_metadata.json")

    datasets: dict[str, tuple[pd.DataFrame, dict | None]] = {}
    if args.feature_set in {"baseline", "both", "experiments", "all"}:
        datasets["baseline"] = (bundle.baseline_dataset, None)
    if args.feature_set in {"with_clusters", "both", "all"}:
        if bundle.with_cluster_dataset is None:
            warnings.warn("with_clusters requested, but no valid clustered/theme records were available. Skipping with_clusters.")
        else:
            default_feature_map = bundle.metadata.get("default_pattern_feature_map", {})
            default_pattern_cfg = _pattern_config_with_fixed_ids(args.pattern_feature_config, default_feature_map)
            datasets["with_clusters"] = (bundle.with_cluster_dataset, default_pattern_cfg)
    if args.feature_set in {"experiments", "all"}:
        pattern_datasets = bundle.pattern_datasets or {}
        normalized_experiments = bundle.metadata.get("pattern_feature_experiments", [])
        experiment_cfg_by_name = {str(cfg.get("name")): cfg for cfg in normalized_experiments}
        experiment_maps = bundle.metadata.get("experiment_pattern_feature_maps", {})
        for name, df in pattern_datasets.items():
            exp_cfg = _pattern_config_with_fixed_ids(experiment_cfg_by_name.get(name), experiment_maps.get(name, {}))
            datasets[name] = (df, exp_cfg)

    if not datasets:
        raise ValueError("No datasets available for training. Check FEATURE_SET and clustered/theme records path.")

    manifests = []
    for name, (df, pattern_cfg) in datasets.items():
        print(f"\nTraining feature set: {name} | rows={len(df):,}")
        manifest = train_one_feature_set(df, name, args, run_dir, pattern_feature_config=pattern_cfg)
        manifests.append(manifest)

    comparison_rows = []
    for manifest in manifests:
        row = {"feature_set": manifest["feature_set"]}
        row.update(manifest["test_metrics"])
        comparison_rows.append(row)
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(run_dir / "model_comparison_holdout_test.csv", index=False)
    feature_set_recommendation = save_feature_set_comparison_outputs(comparison, run_dir)

    final_summary = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "feature_sets_trained": [m["feature_set"] for m in manifests],
        "comparison_file": str(run_dir / "model_comparison_holdout_test.csv"),
        "primary_comparison_guidance": "Compare feature sets using PR-AUC, recall_at_top_10pct, precision_at_top_10pct, and lift. For operational use, prioritize top-risk ranking over accuracy.",
        "target_type": args.target_type,
        "feature_set_recommendation": feature_set_recommendation,
    }
    save_json(final_summary, run_dir / "run_summary.json")
    print("\nSaved run outputs to:", run_dir)
    display_cols = [c for c in ["feature_set", "pr_auc", "roc_auc", "recall_at_top_10pct", "precision_at_top_10pct", "lift_at_top_10pct", "false_negative"] if c in comparison.columns]
    print(comparison[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
