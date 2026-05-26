"""Score current site/department/month any-injury risk using a saved model.

Run directly from the project root after training:

    python src/injury_risk_classification/score_current_site_risk.py

All paths and tunable parameters are configured in:

    src/injury_risk_classification/config.py

If config.MODEL_DIR is None, this script automatically finds the latest training
run and uses config.SCORE_FEATURE_SET. The default MVP output is a ranked list
of site/departments by future-any-injury risk score.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from injury_risk_classification import config
from injury_risk_classification.feature_engineering import build_classification_dataset
from injury_risk_classification.train_injury_risk_classifier import predict_proba_positive, target_col
from injury_risk_classification.utils import ensure_dir, read_json


def find_latest_model_dir() -> Path:
    """Find the latest training run folder for config.SCORE_FEATURE_SET."""
    run_root = Path(config.RUN_OUTPUT_DIR)
    if not run_root.exists():
        raise FileNotFoundError(f"No training run folder exists yet: {run_root}")
    candidate_runs = sorted([p for p in run_root.iterdir() if p.is_dir()], key=lambda p: p.name, reverse=True)
    for run_dir in candidate_runs:
        method_dir = run_dir / config.SCORE_FEATURE_SET
        manifest = method_dir / "model_manifest.json"
        model = method_dir / "model_final_refit_all_eligible_data.joblib"
        if manifest.exists() and model.exists():
            return method_dir
    raise FileNotFoundError(
        f"Could not find a trained model for SCORE_FEATURE_SET={config.SCORE_FEATURE_SET!r} under {run_root}. "
        "Train the model first or set MODEL_DIR in config.py."
    )


def _build_scoring_bundle(feature_set: str, pattern_feature_config: dict | None, target_type: str):
    clustered_path = Path(config.CLUSTERED_PATTERN_RECORDS_PATH)
    needs_patterns = feature_set != "baseline"
    if needs_patterns and not clustered_path.exists() and config.REQUIRE_CLUSTERED_RECORDS_FOR_CLUSTER_FEATURES:
        raise FileNotFoundError(
            f"The selected model requires pattern features, but this file does not exist: {clustered_path}\n"
            "Run the unsupervised pipeline first: "
            "python src/pattern_learning_unsupervised/train_pattern_clusters_hdbscan.py"
        )
    clustered_records = clustered_path if clustered_path.exists() and needs_patterns else None
    return build_classification_dataset(
        input_dir=config.INPUT_DIR,
        output_dir=config.OUTPUT_DIR,
        clustered_records_path=clustered_records,
        horizon_months=config.HORIZON_MONTHS,
        target_type=target_type,
        rolling_windows=list(config.ROLLING_WINDOWS),
        top_n_clusters=config.TOP_N_CLUSTERS,
        min_history_months=config.MIN_HISTORY_MONTHS,
        reference_date=config.REFERENCE_DATE,
        write_outputs=False,
        pattern_feature_config=pattern_feature_config or config.get_pattern_feature_config(),
        pattern_feature_experiments=[],
    )


def _add_missing_model_columns(df: pd.DataFrame, feature_cols: list[str], categorical_cols: list[str]) -> pd.DataFrame:
    """Guarantee scoring has the exact columns expected by the trained model.

    Pattern ID columns can disappear if a selected pattern has no current records.
    Those columns should be zero rather than causing scoring to fail.
    """
    out = df.copy()
    categorical = set(categorical_cols or [])
    for col in feature_cols:
        if col not in out.columns:
            out[col] = "Unknown" if col in categorical else 0
    return out



def _feature_family(feature: str) -> str:
    name = str(feature).lower()
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
    if name in {"site_name_filled", "department_name_filled", "region_name_filled", "business_unit_name_filled", "country_name_filled"}:
        return "location_categorical"
    return "other"


def _load_raw_feature_importance(model_dir: Path) -> pd.DataFrame:
    """Load raw-column feature importance for dashboard driver explanations."""
    candidates = [
        model_dir / "feature_importance_raw.csv",
        model_dir / "feature_importance_final_model_raw.csv",
        model_dir / "feature_importance.csv",
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            if "raw_feature" not in df.columns and "feature" in df.columns:
                df = df.rename(columns={"feature": "raw_feature"})
            if "importance" not in df.columns:
                df["importance"] = 0.0
            if "feature_family" not in df.columns:
                df["feature_family"] = df["raw_feature"].map(_feature_family)
            df["importance"] = pd.to_numeric(df["importance"], errors="coerce").fillna(0.0)
            return df.sort_values("importance", ascending=False).reset_index(drop=True)
    return pd.DataFrame(columns=["raw_feature", "importance", "feature_family"])


def _friendly_feature_name(feature: str) -> str:
    """Convert internal feature names into compact dashboard labels."""
    text = str(feature)
    text = re.sub(r"^top_(theme|cluster)_\d+_", r"\1: ", text)
    text = text.replace("_count_m_last_", " count last ")
    text = text.replace("_growth_", " growth ")
    text = text.replace("_last_", " last ")
    text = text.replace("_prev_", " previous ")
    text = text.replace("_m", " month")
    text = text.replace("_", " ")
    return re.sub(r"\s+", " ", text).strip().title()


def _format_number(value: object) -> str:
    try:
        v = float(value)
    except Exception:
        return str(value)
    if not np.isfinite(v):
        return "NA"
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:.2f}"


def _driver_score(value: float, stats: dict, importance: float) -> float:
    """Combine model importance and current row magnitude for dashboard drivers."""
    try:
        v = float(value)
    except Exception:
        return 0.0
    if not np.isfinite(v):
        return 0.0
    median = float(stats.get("median", 0.0) or 0.0)
    std = float(stats.get("std", 0.0) or 0.0)
    p90 = float(stats.get("p90", 0.0) or 0.0)
    scale = std if std > 0 else max(abs(p90 - median), 1.0)
    magnitude = max(0.0, (v - median) / scale)
    if v != 0 and magnitude == 0:
        magnitude = 0.25
    return float(max(0.0, importance) * magnitude)


def _top_drivers_for_row(row: pd.Series, importance_df: pd.DataFrame, stats: dict, feature_cols: list[str], top_n: int) -> str:
    """Return a semicolon-separated list of top raw feature drivers for one row."""
    if importance_df.empty:
        return ""
    feature_set = set(feature_cols)
    candidates = []
    for _, item in importance_df.iterrows():
        feature = str(item.get("raw_feature", ""))
        if feature not in feature_set or feature not in row.index:
            continue
        if feature in {"site_name_filled", "department_name_filled", "region_name_filled", "business_unit_name_filled", "country_name_filled"}:
            continue
        value = row.get(feature)
        if isinstance(value, str):
            continue
        score = _driver_score(value, stats.get(feature, {}), float(item.get("importance", 0.0)))
        if score <= 0:
            continue
        candidates.append((score, feature, value))
    candidates = sorted(candidates, key=lambda x: x[0], reverse=True)[: int(top_n)]
    return "; ".join(f"{_friendly_feature_name(f)}={_format_number(v)}" for _, f, v in candidates)


def _trend_label(row: pd.Series, monthly_base: str, window: int) -> str:
    """Convert rolling count and previous-window count into a trend label."""
    last_col = f"{monthly_base}_last_{window}m"
    prev_col = f"{monthly_base}_prev_{window}m"
    if last_col not in row.index or prev_col not in row.index:
        return "Not available"
    current = float(row.get(last_col, 0) or 0)
    prior = float(row.get(prev_col, 0) or 0)
    if current == 0 and prior == 0:
        return "No recent activity"
    if prior == 0 and current > 0:
        return "New / increasing"
    diff = current - prior
    if current >= prior * 1.25 and diff >= 1:
        return "Increasing"
    if current <= prior * 0.75 and -diff >= 1:
        return "Decreasing"
    return "Stable"


def _extract_top_pattern_columns(row: pd.Series, level: str, window: int, top_n: int) -> str:
    """Extract highest-count theme/cluster features for dashboard display."""
    pattern = re.compile(rf"^top_{level}_(\d+)_(.+)_count_m_last_{int(window)}m$")
    items = []
    for col in row.index:
        match = pattern.match(str(col))
        if not match:
            continue
        value = float(row.get(col, 0) or 0)
        if value <= 0:
            continue
        label = match.group(2).replace("_", " ").strip().title()
        growth_col = str(col).replace(f"_last_{int(window)}m", f"_growth_{int(window)}m")
        growth = row.get(growth_col, np.nan) if growth_col in row.index else np.nan
        growth_txt = ""
        try:
            if np.isfinite(float(growth)) and abs(float(growth)) > 0:
                growth_txt = f", growth {float(growth):+.2f}"
        except Exception:
            pass
        items.append((value, f"{label}: {_format_number(value)}{growth_txt}"))
    items = sorted(items, key=lambda x: x[0], reverse=True)[: int(top_n)]
    return "; ".join(text for _, text in items)


def _recent_pattern_increases(row: pd.Series, window: int, top_n: int) -> str:
    """Summarize pattern/theme growth features with positive increases."""
    items = []
    for col in row.index:
        name = str(col)
        if not (name.startswith("top_theme_") or name.startswith("top_cluster_")):
            continue
        if not name.endswith(f"_growth_{int(window)}m"):
            continue
        growth = float(row.get(col, 0) or 0)
        if growth <= 0:
            continue
        label = re.sub(r"^top_(theme|cluster)_\d+_", "", name)
        label = re.sub(rf"_count_m_growth_{int(window)}m$", "", label)
        label = label.replace("_", " ").strip().title()
        items.append((growth, f"{label}: {growth:+.2f}"))
    items = sorted(items, key=lambda x: x[0], reverse=True)[: int(top_n)]
    return "; ".join(text for _, text in items)


def _suggest_action(row: pd.Series) -> str:
    tier = str(row.get("risk_tier", "Monitor"))
    overdue = float(row.get("overdue_open_task_count", 0) or 0)
    pattern_trend = str(row.get("pattern_trend_direction", ""))
    injury_trend = str(row.get("injury_trend_direction", ""))
    if tier == "Critical":
        return "Review within 7 days; validate top themes, overdue actions, and recent injury/near-miss trend with site EHS and department owner."
    if tier == "High":
        return "Review within 14 days; confirm leading indicators and assign focused follow-up actions."
    if overdue > 0 or "Increasing" in pattern_trend or "Increasing" in injury_trend:
        return "Add to watchlist; check overdue actions and increasing pattern areas during the next EHS review."
    return "Monitor through normal safety review cadence."


def _review_due_days(tier: str) -> int:
    if tier == "Critical":
        return int(getattr(config, "CRITICAL_REVIEW_DUE_DAYS", 7))
    if tier == "High":
        return int(getattr(config, "HIGH_REVIEW_DUE_DAYS", 14))
    if tier == "Watchlist":
        return int(getattr(config, "WATCHLIST_REVIEW_DUE_DAYS", 30))
    return int(getattr(config, "MONITOR_REVIEW_DUE_DAYS", 60))


def _add_dashboard_columns(current: pd.DataFrame, manifest: dict, model_dir: Path, feature_cols: list[str]) -> pd.DataFrame:
    """Add Power BI / operational-review columns to the scored current-month table."""
    out = current.copy()
    window = int(getattr(config, "DASHBOARD_ROLLING_WINDOW_MONTHS", 3))
    top_n_drivers = int(getattr(config, "DASHBOARD_TOP_N_DRIVERS", 6))
    top_n_themes = int(getattr(config, "DASHBOARD_TOP_N_THEMES", 5))
    importance = _load_raw_feature_importance(model_dir)
    stats = manifest.get("feature_reference_stats", {}) or {}

    out["near_miss_trend_direction"] = out.apply(lambda r: _trend_label(r, "near_miss_count_m", window), axis=1)
    out["hazard_trend_direction"] = out.apply(lambda r: _trend_label(r, "hazard_identification_count_m", window), axis=1)
    out["injury_trend_direction"] = out.apply(lambda r: _trend_label(r, "injury_incident_count_m", window), axis=1)
    out["pattern_trend_direction"] = out.apply(lambda r: _trend_label(r, "pattern_event_count_m", window), axis=1)
    out["audit_observation_trend_direction"] = out.apply(lambda r: _trend_label(r, "observation_count_m", window), axis=1)
    out["top_themes_last_3m"] = out.apply(lambda r: _extract_top_pattern_columns(r, "theme", window, top_n_themes), axis=1)
    out["top_clusters_last_3m"] = out.apply(lambda r: _extract_top_pattern_columns(r, "cluster", window, top_n_themes), axis=1)
    out["recent_pattern_increases"] = out.apply(lambda r: _recent_pattern_increases(r, window, top_n_themes), axis=1)
    out["top_driver_features"] = out.apply(lambda r: _top_drivers_for_row(r, importance, stats, feature_cols, top_n_drivers), axis=1)

    for col in [
        "open_task_count", "overdue_open_task_count", "task_closure_rate_last_3m",
        "near_miss_count_m_last_3m", "hazard_identification_count_m_last_3m",
        "injury_incident_count_m_last_3m", "pattern_event_count_m_last_3m",
        "assigned_pattern_count_m_last_3m", "outlier_pattern_rate_last_3m",
    ]:
        if col not in out.columns:
            out[col] = 0

    out["risk_reason_summary"] = out.apply(
        lambda r: "; ".join([x for x in [
            f"Top drivers: {r.get('top_driver_features', '')}" if r.get("top_driver_features", "") else "",
            f"Themes: {r.get('top_themes_last_3m', '')}" if r.get("top_themes_last_3m", "") else "",
            f"Pattern trend: {r.get('pattern_trend_direction', '')}" if r.get("pattern_trend_direction", "") else "",
            f"Overdue actions: {_format_number(r.get('overdue_open_task_count', 0))}" if float(r.get("overdue_open_task_count", 0) or 0) > 0 else "",
        ] if x]),
        axis=1,
    )
    out["recommended_action"] = out.apply(_suggest_action, axis=1)
    out["review_due_days"] = out["risk_tier"].map(_review_due_days).astype(int)
    out["notification_audience"] = out["risk_tier"].map({
        "Critical": "Site EHS Manager; Department Manager; Business Unit EHS",
        "High": "Site EHS Manager; Department Manager",
        "Watchlist": "Site EHS Manager",
        "Monitor": "Standard review cadence",
    }).fillna("Standard review cadence")
    out["operational_status"] = "New"
    out["review_owner"] = out.get("site_name_filled", "")
    out["review_queue_rank"] = out["risk_rank"].astype(int)
    return out


def _save_dashboard_files(current: pd.DataFrame, output_file: Path) -> None:
    """Save dashboard, queue, tier summary, and site rollup artifacts."""
    if not bool(getattr(config, "SAVE_DASHBOARD_OUTPUTS", True)):
        return
    dashboard_file = Path(getattr(config, "DASHBOARD_OUTPUT_FILE", output_file.with_name("current_risk_dashboard.csv")))
    queue_file = Path(getattr(config, "OPERATIONAL_REVIEW_QUEUE_OUTPUT_FILE", output_file.with_name("operational_review_queue.csv")))
    tier_summary_file = Path(getattr(config, "DASHBOARD_TIER_SUMMARY_OUTPUT_FILE", output_file.with_name("risk_tier_summary.csv")))
    site_rollup_file = Path(getattr(config, "DASHBOARD_SITE_ROLLUP_OUTPUT_FILE", output_file.with_name("site_risk_rollup.csv")))
    for path in [dashboard_file, queue_file, tier_summary_file, site_rollup_file]:
        ensure_dir(path.parent)

    dashboard_cols = [c for c in [
        "anchor_month", "site_name_filled", "department_name_filled", "region_name_filled",
        "business_unit_name_filled", "country_name_filled", "risk_score", "risk_rank",
        "risk_percentile", "risk_tier", "risk_flag", "near_miss_count_m_last_3m",
        "hazard_identification_count_m_last_3m", "injury_incident_count_m_last_3m",
        "pattern_event_count_m_last_3m", "assigned_pattern_count_m_last_3m",
        "outlier_pattern_rate_last_3m", "open_task_count", "overdue_open_task_count",
        "task_closure_rate_last_3m", "near_miss_trend_direction", "hazard_trend_direction",
        "injury_trend_direction", "pattern_trend_direction", "audit_observation_trend_direction",
        "top_themes_last_3m", "top_clusters_last_3m", "recent_pattern_increases",
        "top_driver_features", "risk_reason_summary", "recommended_action", "notification_audience",
        "review_due_days", "review_owner", "operational_status", "model_feature_set", "model_dir",
    ] if c in current.columns]
    current[dashboard_cols].sort_values("risk_rank").to_csv(dashboard_file, index=False)

    queue_tiers = set(getattr(config, "OPERATIONAL_QUEUE_TIERS", ["Critical", "High", "Watchlist"]))
    queue = current[current["risk_tier"].isin(queue_tiers)].copy()
    if queue.empty:
        queue = current.sort_values("risk_rank").head(max(1, int(np.ceil(len(current) * 0.10)))).copy()
    queue[dashboard_cols].sort_values("risk_rank").to_csv(queue_file, index=False)

    tier_summary = current.groupby("risk_tier", dropna=False).agg(
        row_count=("risk_score", "size"),
        min_score=("risk_score", "min"),
        mean_score=("risk_score", "mean"),
        max_score=("risk_score", "max"),
        overdue_open_tasks=("overdue_open_task_count", "sum"),
    ).reset_index()
    tier_summary.to_csv(tier_summary_file, index=False)

    site_rollup = current.groupby(["site_name_filled"], dropna=False).agg(
        department_count=("department_name_filled", "nunique"),
        max_risk_score=("risk_score", "max"),
        mean_risk_score=("risk_score", "mean"),
        critical_count=("risk_tier", lambda s: int((s == "Critical").sum())),
        high_count=("risk_tier", lambda s: int((s == "High").sum())),
        watchlist_count=("risk_tier", lambda s: int((s == "Watchlist").sum())),
        overdue_open_tasks=("overdue_open_task_count", "sum"),
    ).reset_index().sort_values(["critical_count", "high_count", "max_risk_score"], ascending=[False, False, False])
    site_rollup.to_csv(site_rollup_file, index=False)

def main() -> None:
    model_dir = Path(config.MODEL_DIR) if config.MODEL_DIR else find_latest_model_dir()
    manifest = read_json(model_dir / "model_manifest.json")
    feature_set = manifest["feature_set"]
    model_file = model_dir / "model_final_refit_all_eligible_data.joblib"
    if not model_file.exists():
        raise FileNotFoundError(f"Missing model file: {model_file}")

    target_type = manifest.get("target_type", getattr(config, "TARGET_TYPE", "any_injury"))
    pattern_feature_config = manifest.get("pattern_feature_config") if feature_set != "baseline" else None
    bundle = _build_scoring_bundle(feature_set, pattern_feature_config, target_type=target_type)
    if feature_set == "baseline":
        df = bundle.baseline_scoring_dataset if bundle.baseline_scoring_dataset is not None else bundle.baseline_dataset
    else:
        df = bundle.with_cluster_scoring_dataset if bundle.with_cluster_scoring_dataset is not None else bundle.with_cluster_dataset
    if df is None or df.empty:
        raise ValueError("No scoring rows were created. Check input data, reference date, and feature set.")

    feature_cols = list(manifest["feature_columns"])
    categorical_cols = list(manifest.get("categorical_columns", []))
    df = _add_missing_model_columns(df, feature_cols, categorical_cols)

    latest_month = pd.to_datetime(df["anchor_month"]).max()
    current = df[pd.to_datetime(df["anchor_month"]).eq(latest_month)].copy()
    model = joblib.load(model_file)
    target = target_col(config.HORIZON_MONTHS, target_type)
    score_col = "future_any_injury_risk_score" if str(target_type).lower() in {"any_injury", "any", "injury"} else "future_severe_injury_risk_score"
    current[score_col] = predict_proba_positive(model, current[feature_cols])
    current["risk_score"] = current[score_col]
    threshold = float(manifest.get("selected_threshold", 0.5))
    current["risk_flag"] = current[score_col].ge(threshold).astype(int)
    current["risk_rank"] = current[score_col].rank(method="first", ascending=False).astype(int)
    current["risk_percentile"] = current[score_col].rank(method="first", pct=True, ascending=True)
    n_current = max(1, len(current))
    rank_fraction = current["risk_rank"] / float(n_current)
    current["risk_tier"] = "Monitor"
    current.loc[rank_fraction.le(float(getattr(config, "RISK_TIER_WATCHLIST_FRACTION", 0.25))), "risk_tier"] = "Watchlist"
    current.loc[rank_fraction.le(float(getattr(config, "RISK_TIER_HIGH_FRACTION", 0.10))), "risk_tier"] = "High"
    current.loc[rank_fraction.le(float(getattr(config, "RISK_TIER_CRITICAL_FRACTION", 0.05))), "risk_tier"] = "Critical"
    current["model_feature_set"] = feature_set
    current["model_dir"] = str(model_dir)
    current["scoring_month_is_complete_future_window"] = current.get("has_complete_future_window", False)
    current = _add_dashboard_columns(current, manifest, model_dir, feature_cols)

    keep_cols = [
        "anchor_month", "site_name_filled", "department_name_filled", "region_name_filled",
        "business_unit_name_filled", "country_name_filled", score_col, "risk_score",
        "risk_flag", "risk_rank", "risk_percentile", "risk_tier", "top_driver_features",
        "top_themes_last_3m", "recent_pattern_increases", "pattern_trend_direction",
        "overdue_open_task_count", "recommended_action", "notification_audience",
        "review_due_days", "model_feature_set", "scoring_month_is_complete_future_window", "model_dir",
    ]
    # Only include target if this month truly has a complete future window. For
    # current operational scoring, it usually will not.
    if target in current.columns and bool(current.get("has_complete_future_window", pd.Series(False, index=current.index)).all()):
        keep_cols.append(target)

    output_file = Path(config.CURRENT_RISK_OUTPUT_FILE)
    ensure_dir(output_file.parent)
    keep_cols = [c for c in keep_cols if c in current.columns]
    current[keep_cols].sort_values("risk_rank").to_csv(output_file, index=False)
    _save_dashboard_files(current, output_file)
    print("Using model:", model_dir)
    print("Scored feature set:", feature_set)
    print("Target:", target)
    print("Scoring month:", latest_month.date())
    print("Saved current site risk scores to:", output_file)
    if bool(getattr(config, "SAVE_DASHBOARD_OUTPUTS", True)):
        print("Saved dashboard file to:", getattr(config, "DASHBOARD_OUTPUT_FILE", output_file.with_name("current_risk_dashboard.csv")))
        print("Saved operational review queue to:", getattr(config, "OPERATIONAL_REVIEW_QUEUE_OUTPUT_FILE", output_file.with_name("operational_review_queue.csv")))


if __name__ == "__main__":
    main()
