"""EDA table and plot generation for Pattern Learning."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import value_counts_table


def _save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _finish_plot(path: Path, title: str, xlabel: str | None = None, ylabel: str | None = None) -> None:
    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_bar(table: pd.DataFrame, x_col: str, y_col: str, path: Path, title: str, horizontal: bool = False) -> None:
    plt.figure(figsize=(10, 6))
    if horizontal:
        plt.barh(table[x_col].astype(str), table[y_col])
        plt.gca().invert_yaxis()
        _finish_plot(path, title, xlabel=y_col, ylabel=x_col)
    else:
        plt.bar(table[x_col].astype(str), table[y_col])
        plt.xticks(rotation=45, ha="right")
        _finish_plot(path, title, xlabel=x_col, ylabel=y_col)


def _plot_line(df: pd.DataFrame, path: Path, title: str) -> None:
    plt.figure(figsize=(12, 6))
    for col in df.columns:
        plt.plot(df.index, df[col], marker="o", markersize=2, label=str(col))
    plt.legend()
    _finish_plot(path, title, xlabel="Month", ylabel="Record count")


def _plot_hist(series: pd.Series, path: Path, title: str, bins: int = 50) -> None:
    plt.figure(figsize=(10, 6))
    cleaned = pd.to_numeric(series, errors="coerce").dropna()
    if cleaned.empty:
        cleaned = pd.Series([0])
    upper = cleaned.quantile(0.99)
    cleaned = cleaned[cleaned <= upper]
    plt.hist(cleaned, bins=bins)
    _finish_plot(path, title, xlabel=series.name or "value", ylabel="Frequency")


def create_eda_outputs(
    raw_shapes: Mapping[str, tuple[int, int]],
    incident_enriched: pd.DataFrame,
    pattern_records: pd.DataFrame,
    injury_agg: pd.DataFrame,
    task_enriched: pd.DataFrame,
    audit_enriched: pd.DataFrame,
    location_hierarchy: pd.DataFrame,
    site_month_features: pd.DataFrame,
    output_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Create EDA tables, plots, and a markdown summary."""
    eda_dir = output_dir / "eda"
    table_dir = eda_dir / "tables"
    plot_dir = eda_dir / "plots"
    table_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    tables: dict[str, pd.DataFrame] = {}

    tables["raw_row_counts"] = pd.DataFrame([
        {"dataset": name, "rows": shape[0], "columns": shape[1]} for name, shape in raw_shapes.items()
    ]).sort_values("dataset")

    tables["incident_category_counts"] = value_counts_table(incident_enriched["incident_category_name"], "incident_category_name")
    tables["incident_status_counts"] = value_counts_table(incident_enriched["incident_status_name"], "incident_status_name")
    tables["pattern_category_counts"] = value_counts_table(pattern_records["incident_category_name"], "incident_category_name")
    tables["top_pattern_sites"] = value_counts_table(pattern_records.get("site_name_filled", pd.Series(dtype="object")), "site_name", top_n=25)
    tables["top_pattern_departments"] = value_counts_table(pattern_records.get("department_name_filled", pd.Series(dtype="object")), "department_name", top_n=25)

    if "incident_month" in pattern_records.columns:
        pattern_month = pattern_records.dropna(subset=["incident_month"]).copy()
        monthly = (
            pattern_month.groupby(["incident_month", "incident_category_name"], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values("incident_month")
        )
        tables["pattern_records_by_month"] = monthly
    else:
        tables["pattern_records_by_month"] = pd.DataFrame(columns=["incident_month", "incident_category_name", "count"])

    severity_rows = []
    for col, label in [
        ("injury_count", "injury_records"),
        ("lost_time_any", "lost_time_incidents"),
        ("restricted_time_any", "restricted_time_incidents"),
        ("fatality_any", "fatality_incidents"),
        ("emergency_room_any", "emergency_room_incidents"),
        ("inpatient_any", "inpatient_incidents"),
        ("severe_actual", "severe_actual_incidents"),
    ]:
        if col in incident_enriched.columns:
            if col == "injury_count":
                val = int(pd.to_numeric(incident_enriched[col], errors="coerce").fillna(0).sum())
            else:
                val = int(incident_enriched[col].fillna(False).astype(bool).sum())
            severity_rows.append({"metric": label, "count": val})
    tables["injury_severity_summary"] = pd.DataFrame(severity_rows)

    tables["task_category_counts"] = value_counts_table(task_enriched["task_category_name"], "task_category_name")
    tables["task_status_counts"] = value_counts_table(task_enriched["task_status_name"], "task_status_name")
    task_snapshot = pd.DataFrame([
        {"metric": "task_rows", "count": len(task_enriched)},
        {"metric": "active_non_archived_tasks", "count": int(task_enriched.get("is_active_record", pd.Series(dtype=bool)).fillna(False).sum())},
        {"metric": "open_tasks", "count": int(task_enriched.get("is_open", pd.Series(dtype=bool)).fillna(False).sum())},
        {"metric": "overdue_tasks", "count": int(task_enriched.get("is_overdue", pd.Series(dtype=bool)).fillna(False).sum())},
    ])
    tables["task_snapshot_summary"] = task_snapshot

    tables["audit_category_counts"] = value_counts_table(audit_enriched["audit_category_name"], "audit_category_name")
    tables["audit_type_counts"] = value_counts_table(audit_enriched["audit_type_name"], "audit_type_name")

    location_coverage = pd.DataFrame([
        {"metric": "incident_rows_with_location_match", "count": int(incident_enriched.get("has_location_match", pd.Series(dtype=bool)).fillna(False).sum())},
        {"metric": "incident_rows_total", "count": len(incident_enriched)},
        {"metric": "pattern_rows_with_site", "count": int(pattern_records.get("site_name", pd.Series(index=pattern_records.index, dtype="object")).notna().sum())},
        {"metric": "pattern_rows_total", "count": len(pattern_records)},
        {"metric": "location_rows", "count": len(location_hierarchy)},
    ])
    location_coverage["percent"] = np.where(
        location_coverage["metric"].str.contains("with"),
        location_coverage["count"] / location_coverage["count"].shift(-1),
        np.nan,
    )
    tables["location_join_coverage"] = location_coverage

    dq_rows = []
    for col in ["incident_date_missing", "incident_date_after_reference", "incident_date_before_2000"]:
        if col in incident_enriched.columns:
            dq_rows.append({"metric": col, "count": int(incident_enriched[col].fillna(False).sum())})
    if "report_lag_days" in incident_enriched.columns:
        dq_rows.extend([
            {"metric": "negative_report_lag_rows", "count": int((incident_enriched["report_lag_days"] < 0).fillna(False).sum())},
            {"metric": "report_lag_gt_365_days_rows", "count": int((incident_enriched["report_lag_days"] > 365).fillna(False).sum())},
        ])
    tables["data_quality_summary"] = pd.DataFrame(dq_rows)

    missing = pattern_records.isna().mean().sort_values(ascending=False).rename("missing_rate").reset_index().rename(columns={"index": "column"})
    missing["missing_count"] = [int(pattern_records[c].isna().sum()) for c in missing["column"]]
    tables["pattern_missingness"] = missing

    # Site/month feature summary table for joined EDA.
    if not site_month_features.empty:
        tables["site_month_feature_summary"] = site_month_features.describe(include="all").transpose().reset_index().rename(columns={"index": "column"})
        risk_cols = [c for c in ["near_miss_count", "hazard_identification_count", "task_overdue_count", "unsafe_condition_count", "severe_actual_count"] if c in site_month_features.columns]
        if risk_cols:
            top_site_month = site_month_features.copy()
            top_site_month["simple_pattern_signal"] = top_site_month[risk_cols].sum(axis=1)
            keep = ["feature_site_name", "feature_department_name", "feature_month", "simple_pattern_signal"] + risk_cols
            tables["top_site_month_pattern_signals"] = top_site_month.sort_values("simple_pattern_signal", ascending=False)[keep].head(50)

    for name, table in tables.items():
        _save_table(table, table_dir / f"{name}.csv")

    # Plots.
    _plot_bar(tables["incident_category_counts"], "incident_category_name", "count", plot_dir / "incident_category_counts.png", "Incident records by category")
    _plot_bar(tables["pattern_category_counts"], "incident_category_name", "count", plot_dir / "pattern_category_counts.png", "Pattern-learning records by category")
    _plot_bar(tables["top_pattern_sites"], "site_name", "count", plot_dir / "top_pattern_sites.png", "Top sites by near-miss/hazard records", horizontal=True)
    _plot_bar(tables["task_status_counts"], "task_status_name", "count", plot_dir / "task_status_counts.png", "Task records by status")
    _plot_bar(tables["audit_category_counts"], "audit_category_name", "count", plot_dir / "audit_category_counts.png", "Audit/observation records by category")

    if not tables["pattern_records_by_month"].empty:
        pivot = tables["pattern_records_by_month"].pivot(index="incident_month", columns="incident_category_name", values="count").fillna(0)
        _plot_line(pivot, plot_dir / "pattern_records_by_month.png", "Near-miss and hazard records by month")

    if "text_early_word_count" in pattern_records.columns:
        _plot_hist(pattern_records["text_early_word_count"], plot_dir / "early_text_word_count_distribution.png", "Pattern-record early text word count distribution")

    missing_top = tables["pattern_missingness"].head(20).sort_values("missing_rate", ascending=True)
    if not missing_top.empty:
        plt.figure(figsize=(10, 7))
        plt.barh(missing_top["column"].astype(str), missing_top["missing_rate"])
        _finish_plot(plot_dir / "pattern_missingness_top20.png", "Top missingness rates in pattern-learning records", xlabel="Missing rate", ylabel="Column")

    _write_markdown_summary(tables, eda_dir / "eda_summary.md")
    return tables


def _df_to_markdown(df: pd.DataFrame, max_rows: int = 10) -> str:
    if df.empty:
        return "_No rows._"
    return df.head(max_rows).to_markdown(index=False)


def _write_markdown_summary(tables: Mapping[str, pd.DataFrame], path: Path) -> None:
    raw = tables.get("raw_row_counts", pd.DataFrame())
    pattern_counts = tables.get("pattern_category_counts", pd.DataFrame())
    injury = tables.get("injury_severity_summary", pd.DataFrame())
    dq = tables.get("data_quality_summary", pd.DataFrame())
    top_sites = tables.get("top_pattern_sites", pd.DataFrame())
    task_snapshot = tables.get("task_snapshot_summary", pd.DataFrame())
    audit_counts = tables.get("audit_category_counts", pd.DataFrame())
    top_site_month = tables.get("top_site_month_pattern_signals", pd.DataFrame())

    lines = [
        "# Pattern Learning EDA Summary",
        "",
        "## Raw dataset sizes",
        _df_to_markdown(raw, 20),
        "",
        "## Pattern-learning candidate records",
        "These are active, non-archived Near Miss and Hazard Identification records with usable early text.",
        _df_to_markdown(pattern_counts, 10),
        "",
        "## Injury and severe-actual outcome summary",
        "`severe_actual` is derived as fatality OR lost time OR restricted time OR inpatient. This is an outcome flag for downstream similarity/risk work, not the unsupervised target for the first clustering model.",
        _df_to_markdown(injury, 20),
        "",
        "## Task snapshot summary",
        _df_to_markdown(task_snapshot, 20),
        "",
        "## Audit/observation category counts",
        _df_to_markdown(audit_counts, 20),
        "",
        "## Data-quality flags",
        _df_to_markdown(dq, 20),
        "",
        "## Top sites by pattern-learning records",
        _df_to_markdown(top_sites, 15),
        "",
        "## Top joined site-month pattern signals",
        "This is a simple EDA signal only, not a trained risk score.",
        _df_to_markdown(top_site_month, 15),
        "",
        "## Recommended first ML table",
        "Use `outputs/processed/pattern_learning_records.csv` and start with `ml_text_early` for embeddings/clustering. Keep `ml_text_full` for analysis runs where post-investigation text is acceptable.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
