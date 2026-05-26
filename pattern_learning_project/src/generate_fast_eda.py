"""Fast EDA generator using a minimal set of columns.

This is useful for quick first-pass validation when the full enrichment pipeline is too heavy
for a constrained notebook/session. The production preparation script remains run_data_prep.py.
"""
from __future__ import annotations

from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lookups import prepare_listitems
from locations import build_location_hierarchy
from utils import read_csv_safely, standardize_columns, parse_datetime_utc


def save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def simple_text(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.replace(r"[\r\n\t]+", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()


def value_counts(series: pd.Series, name: str) -> pd.DataFrame:
    out = series.fillna("Unknown").value_counts().rename_axis(name).reset_index(name="count")
    out["percent"] = out["count"] / out["count"].sum()
    return out


def bar(df: pd.DataFrame, x: str, y: str, path: Path, title: str, horizontal: bool=False) -> None:
    plt.figure(figsize=(10, 6))
    if horizontal:
        plt.barh(df[x].astype(str), df[y])
        plt.gca().invert_yaxis()
        plt.xlabel(y)
        plt.ylabel(x)
    else:
        plt.bar(df[x].astype(str), df[y])
        plt.xticks(rotation=45, ha="right")
        plt.xlabel(x)
        plt.ylabel(y)
    plt.title(title)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def main(input_dir: Path, output_dir: Path, reference_date: str = "2026-05-20") -> None:
    table_dir = output_dir / "eda" / "tables"
    plot_dir = output_dir / "eda" / "plots"
    table_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    li_raw = read_csv_safely(input_dir / "LISTITEM_VIEW.csv")
    li = prepare_listitems(li_raw)
    item_map = dict(zip(li["list_item_id"].astype("Int64"), li["item"]))

    loc_raw = read_csv_safely(input_dir / "LOCATION_VIEW.csv")
    loc_h = build_location_hierarchy(loc_raw, li)

    # Incident fast EDA.
    inc_cols = ["INCIDENTID", "INCIDENTDATE", "REPORTDATE", "INCIDENTCATEGORYID", "INCIDENTSTATUSID", "LOCATIONID", "TITLE", "DESCRIPTION", "ACTIVE", "ARCHIVED"]
    inc = standardize_columns(pd.read_csv(input_dir / "INCIDENT_VIEW.csv", usecols=inc_cols, low_memory=False))
    inc["incident_date"] = parse_datetime_utc(inc["incident_date"])
    inc["report_date"] = parse_datetime_utc(inc["report_date"])
    inc["incident_category_name"] = inc["incident_category_id"].map(item_map)
    inc["incident_status_name"] = inc["incident_status_id"].map(item_map)
    inc["active"] = inc["active"].astype(str).str.lower().eq("true")
    inc["archived"] = inc["archived"].astype(str).str.lower().eq("true")
    inc = inc.merge(loc_h[["location_id", "site_name_filled", "department_name_filled", "business_unit_name_filled", "country_name_filled", "region_name_filled"]], on="location_id", how="left")
    inc["text"] = simple_text(inc["title"]) + " " + simple_text(inc["description"])
    inc["word_count"] = inc["text"].str.split().map(lambda x: len(x) if isinstance(x, list) else 0)
    inc["incident_month"] = inc["incident_date"].dt.tz_convert(None).dt.to_period("M").dt.to_timestamp()
    ref = pd.Timestamp(reference_date, tz="UTC")
    inc["incident_date_after_reference"] = inc["incident_date"].gt(ref)
    inc["incident_date_missing"] = inc["incident_date"].isna()
    inc["report_lag_days"] = (inc["report_date"] - inc["incident_date"]).dt.days
    pattern = inc[inc["incident_category_name"].isin(["Near Miss", "Hazard Identification"]) & inc["active"] & ~inc["archived"] & inc["word_count"].ge(3)].copy()

    raw_counts = []
    for fname in ["INCIDENT_VIEW.csv", "INCIDENTINJURY_VIEW.csv", "LISTITEM_VIEW.csv", "LOCATION_VIEW.csv", "TASK_VIEW.csv", "AUDIT_VIEW.csv"]:
        df_head = pd.read_csv(input_dir / fname, nrows=0)
        first_col = df_head.columns[0]
        row_probe = pd.read_csv(input_dir / fname, usecols=[first_col], low_memory=False)
        raw_counts.append({"dataset": fname, "rows": len(row_probe), "columns": len(df_head.columns)})
    raw_counts = pd.DataFrame(raw_counts)
    save(raw_counts, table_dir / "raw_row_counts.csv")

    inc_cat = value_counts(inc["incident_category_name"], "incident_category_name")
    inc_status = value_counts(inc["incident_status_name"], "incident_status_name")
    pattern_cat = value_counts(pattern["incident_category_name"], "incident_category_name")
    top_sites = value_counts(pattern["site_name_filled"], "site_name").head(25)
    top_departments = value_counts(pattern["department_name_filled"], "department_name").head(25)
    monthly = pattern.groupby(["incident_month", "incident_category_name"]).size().reset_index(name="count")
    dq = pd.DataFrame([
        {"metric": "incident_date_missing", "count": int(inc["incident_date_missing"].sum())},
        {"metric": "incident_date_after_reference", "count": int(inc["incident_date_after_reference"].sum())},
        {"metric": "negative_report_lag_rows", "count": int((inc["report_lag_days"] < 0).fillna(False).sum())},
        {"metric": "report_lag_gt_365_days_rows", "count": int((inc["report_lag_days"] > 365).fillna(False).sum())},
    ])
    save(inc_cat, table_dir / "incident_category_counts.csv")
    save(inc_status, table_dir / "incident_status_counts.csv")
    save(pattern_cat, table_dir / "pattern_category_counts.csv")
    save(top_sites, table_dir / "top_pattern_sites.csv")
    save(top_departments, table_dir / "top_pattern_departments.csv")
    save(monthly, table_dir / "pattern_records_by_month.csv")
    save(dq, table_dir / "data_quality_summary.csv")

    # Injury severity.
    inj_cols = ["INJURYID", "INCIDENTID", "LOSTTIME", "RESTRICTEDTIME", "FATALITY", "EMERGENCYROOM", "INPATIENT"]
    inj = standardize_columns(pd.read_csv(input_dir / "INCIDENTINJURY_VIEW.csv", usecols=inj_cols, low_memory=False))
    for col in ["lost_time", "restricted_time", "fatality", "emergency_room", "inpatient"]:
        inj[col] = inj[col].astype(str).str.lower().eq("true")
    agg = inj.groupby("incident_id").agg(
        injury_count=("injury_id", "count"),
        lost_time_any=("lost_time", "max"),
        restricted_time_any=("restricted_time", "max"),
        fatality_any=("fatality", "max"),
        emergency_room_any=("emergency_room", "max"),
        inpatient_any=("inpatient", "max"),
    ).reset_index()
    agg["severe_actual"] = agg["fatality_any"] | agg["lost_time_any"] | agg["restricted_time_any"] | agg["inpatient_any"]
    injury_summary = pd.DataFrame([
        {"metric": "injury_records", "count": int(agg["injury_count"].sum())},
        {"metric": "incidents_with_injury_record", "count": len(agg)},
        {"metric": "lost_time_incidents", "count": int(agg["lost_time_any"].sum())},
        {"metric": "restricted_time_incidents", "count": int(agg["restricted_time_any"].sum())},
        {"metric": "fatality_incidents", "count": int(agg["fatality_any"].sum())},
        {"metric": "emergency_room_incidents", "count": int(agg["emergency_room_any"].sum())},
        {"metric": "inpatient_incidents", "count": int(agg["inpatient_any"].sum())},
        {"metric": "severe_actual_incidents", "count": int(agg["severe_actual"].sum())},
    ])
    save(injury_summary, table_dir / "injury_severity_summary.csv")

    # Task fast EDA.
    task_cols = ["TASKID", "TASKCATEGORYID", "TASKSTATUSID", "SOURCETYPEID", "LOCATIONID", "ASSIGNEDDATE", "DUEDATE", "REVISEDDUEDATE", "COMPLETIONDATE", "MARKEDCOMPLETEDATE", "ACTIVE", "ARCHIVED"]
    task = standardize_columns(pd.read_csv(input_dir / "TASK_VIEW.csv", usecols=task_cols, low_memory=False))
    task["task_category_name"] = task["task_category_id"].map(item_map)
    task["task_status_name"] = task["task_status_id"].map(item_map)
    task["active"] = task["active"].astype(str).str.lower().eq("true")
    task["archived"] = task["archived"].astype(str).str.lower().eq("true")
    task["due_date"] = parse_datetime_utc(task["due_date"])
    task["revised_due_date"] = parse_datetime_utc(task["revised_due_date"])
    task["completion_date"] = parse_datetime_utc(task["completion_date"])
    task["marked_complete_date"] = parse_datetime_utc(task["marked_complete_date"])
    task["effective_due_date"] = task["revised_due_date"].fillna(task["due_date"])
    task["is_active_record"] = task["active"] & ~task["archived"]
    task["is_closed"] = task["task_status_name"].eq("Closed") | task["completion_date"].notna() | task["marked_complete_date"].notna()
    task["is_open"] = task["is_active_record"] & ~task["is_closed"]
    task["is_overdue"] = task["is_open"] & task["effective_due_date"].notna() & task["effective_due_date"].lt(ref)
    task_cat = value_counts(task["task_category_name"], "task_category_name")
    task_status = value_counts(task["task_status_name"], "task_status_name")
    task_snapshot = pd.DataFrame([
        {"metric": "task_rows", "count": len(task)},
        {"metric": "active_non_archived_tasks", "count": int(task["is_active_record"].sum())},
        {"metric": "open_tasks", "count": int(task["is_open"].sum())},
        {"metric": "overdue_tasks", "count": int(task["is_overdue"].sum())},
    ])
    save(task_cat, table_dir / "task_category_counts.csv")
    save(task_status, table_dir / "task_status_counts.csv")
    save(task_snapshot, table_dir / "task_snapshot_summary.csv")

    # Audit fast EDA.
    audit_cols = ["AUDITID", "AUDITCATEGORYID", "AUDITTYPEID", "AUDITSTATUSID", "SCHEDULEDLOCATIONID", "ACTUALSTART", "SCHEDULEDSTART", "ACTIVE", "ARCHIVED"]
    audit = standardize_columns(pd.read_csv(input_dir / "AUDIT_VIEW.csv", usecols=audit_cols, low_memory=False))
    audit["audit_category_name"] = audit["audit_category_id"].map(item_map)
    audit["audit_type_name"] = audit["audit_type_id"].map(item_map)
    audit["audit_status_name"] = audit["audit_status_id"].map(item_map)
    audit_cat = value_counts(audit["audit_category_name"], "audit_category_name")
    audit_type = value_counts(audit["audit_type_name"], "audit_type_name")
    audit_status = value_counts(audit["audit_status_name"], "audit_status_name")
    save(audit_cat, table_dir / "audit_category_counts.csv")
    save(audit_type, table_dir / "audit_type_counts.csv")
    save(audit_status, table_dir / "audit_status_counts.csv")

    # Plots.
    bar(inc_cat, "incident_category_name", "count", plot_dir / "incident_category_counts.png", "Incident records by category")
    bar(pattern_cat, "incident_category_name", "count", plot_dir / "pattern_category_counts.png", "Pattern-learning records by category")
    bar(top_sites, "site_name", "count", plot_dir / "top_pattern_sites.png", "Top sites by near-miss/hazard records", True)
    bar(task_status, "task_status_name", "count", plot_dir / "task_status_counts.png", "Task records by status")
    bar(audit_cat, "audit_category_name", "count", plot_dir / "audit_category_counts.png", "Audit/observation records by category")
    if not monthly.empty:
        pivot = monthly.pivot(index="incident_month", columns="incident_category_name", values="count").fillna(0)
        plt.figure(figsize=(12, 6))
        for col in pivot.columns:
            plt.plot(pivot.index, pivot[col], marker="o", markersize=2, label=str(col))
        plt.legend()
        plt.title("Near-miss and hazard records by month")
        plt.xlabel("Month")
        plt.ylabel("Record count")
        plt.tight_layout()
        plt.savefig(plot_dir / "pattern_records_by_month.png", dpi=150, bbox_inches="tight")
        plt.close()
    plt.figure(figsize=(10, 6))
    wc = pattern["word_count"]
    wc = wc[wc <= wc.quantile(0.99)]
    plt.hist(wc, bins=50)
    plt.title("Pattern-record title/description word count distribution")
    plt.xlabel("Word count")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(plot_dir / "early_text_word_count_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Markdown.
    def md(df, n=12):
        return df.head(n).to_markdown(index=False)
    summary = [
        "# Pattern Learning Fast EDA Summary", "",
        "## Raw dataset sizes", md(raw_counts, 10), "",
        "## Incident category counts", md(inc_cat), "",
        "## Active pattern-learning records", md(pattern_cat), "",
        "## Injury severity summary", md(injury_summary, 20), "",
        "## Task snapshot summary", md(task_snapshot, 20), "",
        "## Audit category counts", md(audit_cat), "",
        "## Data quality summary", md(dq, 20), "",
        "## Top sites by pattern-learning records", md(top_sites, 15), "",
        "Note: this fast EDA uses minimal columns. The full preparation pipeline creates the modeling tables.",
    ]
    (output_dir / "eda" / "eda_summary.md").write_text("\n".join(summary), encoding="utf-8")
    print('fast EDA complete', len(pattern), 'pattern rows')


if __name__ == "__main__":
    main(Path("/mnt/data"), Path("/mnt/data/pattern_learning_project/outputs"))
