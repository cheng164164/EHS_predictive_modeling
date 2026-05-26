"""Feature preparation functions for Pattern Learning."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from lookups import add_listitem_columns
from utils import (
    add_month_week_columns,
    add_text_quality_features,
    clean_text_series,
    combine_text_fields,
    ensure_bool_columns,
    parse_datetime_utc,
    safe_divide,
    standardize_columns,
)


INCIDENT_DATE_COLUMNS = [
    "report_date", "investigation_start_date", "insurance_notified_date", "incident_date", "end_date",
    "drug_test_date", "alcohol_test_date"
]
INCIDENT_BOOL_COLUMNS = [
    "on_premises", "downtime", "ppe_worn", "insurance_notified", "drug_test_performed",
    "alcohol_test_performed", "rca_performed", "preventable", "recurrence", "process_safety",
    "work_related", "litigable", "sensitive", "agency_sensitive", "media", "title_v",
    "include_in_statistics", "active", "archived"
]
INJURY_BOOL_COLUMNS = [
    "lost_time", "restricted_time", "fatality", "emergency_room", "inpatient", "repeated_motion",
    "lifted_from_floor", "lifting_aids_available", "lifting_aids_used", "sudden_pain", "non_pain_effect"
]
TASK_DATE_COLUMNS = [
    "verification_due_date", "start_date", "source_date", "revised_due_date", "marked_complete_date",
    "completion_date", "assigned_date", "due_date"
]
TASK_BOOL_COLUMNS = [
    "approval_required", "preventive_maintenance", "workflow_dependency", "task_verified", "active", "archived"
]
AUDIT_DATE_COLUMNS = ["actual_end", "actual_start", "scheduled_end", "scheduled_start"]
AUDIT_BOOL_COLUMNS = ["controlled", "scheduled", "offline_audit", "modular", "active", "archived"]


def prepare_injury_agg(injury_raw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate INCIDENTINJURY_VIEW to one row per incident."""
    injury = standardize_columns(injury_raw)
    if "fatality_date" in injury.columns:
        injury["fatality_date"] = parse_datetime_utc(injury["fatality_date"])
    injury = ensure_bool_columns(injury, INJURY_BOOL_COLUMNS)

    for col in ["incident_id", "injury_id"]:
        if col in injury.columns:
            injury[col] = pd.to_numeric(injury[col], errors="coerce").astype("Int64")

    bool_cols = [c for c in ["lost_time", "restricted_time", "fatality", "emergency_room", "inpatient"] if c in injury.columns]
    for col in bool_cols:
        injury[col] = injury[col].fillna(False).astype(bool)

    if "incident_id" not in injury.columns:
        return pd.DataFrame()

    agg_dict = {"injury_id": "count"} if "injury_id" in injury.columns else {}
    for col in bool_cols:
        agg_dict[col] = "max"
    if "fatality_date" in injury.columns:
        agg_dict["fatality_date"] = "min"

    injury_agg = injury.groupby("incident_id", dropna=False).agg(agg_dict).reset_index()
    if "injury_id" in injury_agg.columns:
        injury_agg = injury_agg.rename(columns={"injury_id": "injury_count"})
    else:
        injury_agg["injury_count"] = 1

    rename = {
        "lost_time": "lost_time_any",
        "restricted_time": "restricted_time_any",
        "fatality": "fatality_any",
        "emergency_room": "emergency_room_any",
        "inpatient": "inpatient_any",
    }
    injury_agg = injury_agg.rename(columns={k: v for k, v in rename.items() if k in injury_agg.columns})
    for col in ["lost_time_any", "restricted_time_any", "fatality_any", "emergency_room_any", "inpatient_any"]:
        if col not in injury_agg.columns:
            injury_agg[col] = False
        injury_agg[col] = injury_agg[col].fillna(False).astype(bool)

    injury_agg["severe_actual"] = (
        injury_agg["fatality_any"] |
        injury_agg["lost_time_any"] |
        injury_agg["restricted_time_any"] |
        injury_agg["inpatient_any"]
    )
    return injury_agg


def prepare_incidents(
    incident_raw: pd.DataFrame,
    listitems: pd.DataFrame,
    location_hierarchy: pd.DataFrame,
    injury_agg: pd.DataFrame,
    reference_date: pd.Timestamp,
) -> pd.DataFrame:
    """Prepare INCIDENT_VIEW with decoded names, injury flags, location hierarchy, and text fields."""
    inc = standardize_columns(incident_raw)
    for col in INCIDENT_DATE_COLUMNS:
        if col in inc.columns:
            inc[col] = parse_datetime_utc(inc[col])
    inc = ensure_bool_columns(inc, INCIDENT_BOOL_COLUMNS)

    id_cols = [
        "incident_id", "incident_category_id", "incident_status_id", "location_id", "offsite_location_id",
        "process_id", "activity_id"
    ]
    for col in id_cols:
        if col in inc.columns:
            inc[col] = pd.to_numeric(inc[col], errors="coerce").astype("Int64")

    inc = add_listitem_columns(inc, listitems, specs={
        "incident_category_id": ("incidentcategory", "incident_category"),
        "incident_status_id": ("incidentstatus", "incident_status"),
    })

    loc_cols = [c for c in location_hierarchy.columns if c != "location_id"]
    inc = inc.merge(location_hierarchy[["location_id"] + loc_cols], on="location_id", how="left", validate="m:1")

    if not injury_agg.empty:
        inc = inc.merge(injury_agg, on="incident_id", how="left", validate="1:1")
    for col in ["injury_count"]:
        if col not in inc.columns:
            inc[col] = 0
        inc[col] = inc[col].fillna(0).astype("Int64")
    for col in ["lost_time_any", "restricted_time_any", "fatality_any", "emergency_room_any", "inpatient_any", "severe_actual"]:
        if col not in inc.columns:
            inc[col] = False
        inc[col] = inc[col].fillna(False).astype(bool)

    # Clean source text columns.
    text_candidates = [
        "title", "description", "off_premises_location", "equipment", "vehicle", "other_process",
        "other_activity", "activity_during_incident", "immediate_action", "immediate_causes",
        "causal_factors", "best_practices", "risk_action", "risk_condition"
    ]
    for col in text_candidates:
        if col in inc.columns:
            inc[col] = clean_text_series(inc[col])

    early_text_cols = [
        "title", "description", "off_premises_location", "equipment", "vehicle", "other_process",
        "other_activity", "activity_during_incident"
    ]
    full_text_cols = early_text_cols + [
        "immediate_action", "immediate_causes", "causal_factors", "best_practices", "risk_action", "risk_condition"
    ]
    inc["ml_text_early"] = combine_text_fields(inc, early_text_cols)
    inc["ml_text_full"] = combine_text_fields(inc, full_text_cols)
    inc = add_text_quality_features(inc, "ml_text_early", "text_early")
    inc = add_text_quality_features(inc, "ml_text_full", "text_full")

    if "incident_date" in inc.columns:
        inc = add_month_week_columns(inc, "incident_date", "incident")
        inc["incident_date_missing"] = inc["incident_date"].isna()
        inc["incident_date_after_reference"] = inc["incident_date"].gt(reference_date)
        inc["incident_date_before_2000"] = inc["incident_date"].lt(pd.Timestamp("2000-01-01", tz="UTC"))
    if "report_date" in inc.columns:
        inc["report_lag_days"] = (inc["report_date"] - inc["incident_date"]).dt.days

    inc["is_active_record"] = inc.get("active", True).fillna(False).astype(bool) & ~inc.get("archived", False).fillna(False).astype(bool)
    inc["is_pattern_candidate"] = inc["incident_category_name"].isin(["Near Miss", "Hazard Identification"])
    inc["has_location_match"] = inc["location_name"].notna() if "location_name" in inc.columns else False
    inc["has_usable_early_text"] = inc["text_early_word_count"].fillna(0).ge(3)

    return inc


def prepare_incident_injury_all_records(incident_enriched: pd.DataFrame) -> pd.DataFrame:
    """Return all incident rows after incident/injury/location/listitem enrichment.

    This is the intermediate analytical table before pattern-learning filters are applied.

    Unlike prepare_pattern_learning_records(), this function does NOT:
    - filter to Near Miss / Hazard Identification only
    - filter to active records only
    - remove archived records
    - require usable early text

    It should include all INCIDENT_VIEW rows enriched with injury information.
    """
    preferred_cols = [
        "incident_id", "incident_number", "incident_date", "incident_month", "incident_week", "report_date",
        "incident_category_id", "incident_category_name", "incident_status_id", "incident_status_name",
        "location_id", "location_name", "location_type_name", "location_path", "business_unit_name",
        "region_name", "country_name", "site_name", "department_name", "department_2_name",
        "site_name_filled", "department_name_filled", "business_unit_name_filled", "country_name_filled",
        "region_name_filled",
        "title", "description", "equipment", "vehicle",
        "off_premises_location", "other_process", "other_activity", "activity_during_incident",
        "ml_text_early", "ml_text_full",
        "text_early_char_count", "text_early_word_count",
        "text_full_char_count", "text_full_word_count",
        "on_premises", "work_related", "process_safety", "preventable", "include_in_statistics",
        "injury_count", "lost_time_any", "restricted_time_any", "fatality_any",
        "emergency_room_any", "inpatient_any", "severe_actual",
        "is_active_record", "is_pattern_candidate", "has_usable_early_text",
        "incident_date_after_reference", "incident_date_before_2000", "report_lag_days"
    ]

    cols = [c for c in preferred_cols if c in incident_enriched.columns]
    return incident_enriched.loc[:, cols].reset_index(drop=True).copy()


def prepare_pattern_learning_records(incident_enriched: pd.DataFrame, active_only: bool = True) -> pd.DataFrame:
    """Filter incident_enriched to near-miss and hazard-identification records for clustering.

    The output intentionally keeps a focused set of columns so the first modeling table stays
    small, stable, and easy to feed into embeddings/clustering.
    """
    mask = incident_enriched["is_pattern_candidate"].fillna(False)
    if active_only and "is_active_record" in incident_enriched.columns:
        mask = mask & incident_enriched["is_active_record"].fillna(False)
    if "has_usable_early_text" in incident_enriched.columns:
        mask = mask & incident_enriched["has_usable_early_text"].fillna(False)

    preferred_cols = [
        "incident_id", "incident_number", "incident_date", "incident_month", "incident_week", "report_date",
        "incident_category_id", "incident_category_name", "incident_status_id", "incident_status_name",
        "location_id", "location_name", "location_type_name", "location_path", "business_unit_name",
        "region_name", "country_name", "site_name", "department_name", "department_2_name",
        "site_name_filled", "department_name_filled", "business_unit_name_filled", "country_name_filled",
        "region_name_filled", "title", "description", "equipment", "vehicle",
        "off_premises_location", "other_process", "other_activity", "activity_during_incident",
        "ml_text_early", "ml_text_full", "text_early_char_count", "text_early_word_count",
        "text_full_char_count", "text_full_word_count", "on_premises", "work_related",
        "process_safety", "preventable", "include_in_statistics", "injury_count", "lost_time_any",
        "restricted_time_any", "fatality_any", "emergency_room_any", "inpatient_any", "severe_actual",
        "is_active_record", "incident_date_after_reference", "incident_date_before_2000", "report_lag_days"
    ]
    cols = [c for c in preferred_cols if c in incident_enriched.columns]
    return incident_enriched.loc[mask, cols].reset_index(drop=True).copy()


def prepare_tasks(
    task_raw: pd.DataFrame,
    listitems: pd.DataFrame,
    location_hierarchy: pd.DataFrame,
    reference_date: pd.Timestamp,
) -> pd.DataFrame:
    """Prepare TASK_VIEW with decoded names, location hierarchy, and open/overdue features."""
    task = standardize_columns(task_raw)
    for col in TASK_DATE_COLUMNS:
        if col in task.columns:
            task[col] = parse_datetime_utc(task[col])
    task = ensure_bool_columns(task, TASK_BOOL_COLUMNS)
    for col in ["task_id", "parent_task_id", "task_category_id", "task_type_id", "task_status_id", "location_id", "source_type_id"]:
        if col in task.columns:
            task[col] = pd.to_numeric(task[col], errors="coerce").astype("Int64")

    task = add_listitem_columns(task, listitems, specs={
        "task_category_id": ("taskcategory", "task_category"),
        "task_type_id": ("tasktype", "task_type"),
        "task_status_id": ("taskstatus", "task_status"),
        "source_type_id": ("module", "source_type"),
    })
    task = task.merge(location_hierarchy, on="location_id", how="left", validate="m:1")

    text_cols = ["task", "description", "source", "equipment", "best_practices", "verification_reason"]
    for col in text_cols:
        if col in task.columns:
            task[col] = clean_text_series(task[col])
    task["task_text"] = combine_text_fields(task, text_cols)
    task = add_text_quality_features(task, "task_text", "task_text")

    task["effective_due_date"] = task.get("revised_due_date", pd.NaT).fillna(task.get("due_date", pd.NaT))
    task["is_active_record"] = task.get("active", True).fillna(False).astype(bool) & ~task.get("archived", False).fillna(False).astype(bool)
    task["is_closed"] = task["task_status_name"].eq("Closed") | task.get("completion_date", pd.NaT).notna() | task.get("marked_complete_date", pd.NaT).notna()
    task["is_open"] = task["is_active_record"] & ~task["is_closed"]
    task["is_overdue"] = task["is_open"] & task["effective_due_date"].notna() & task["effective_due_date"].lt(reference_date)
    task["days_open"] = np.where(
        task["assigned_date"].notna(),
        (reference_date - task["assigned_date"]).dt.days,
        np.nan,
    )
    task["days_overdue"] = np.where(
        task["is_overdue"],
        (reference_date - task["effective_due_date"]).dt.days,
        0,
    )
    task["days_until_due"] = np.where(
        task["effective_due_date"].notna(),
        (task["effective_due_date"] - reference_date).dt.days,
        np.nan,
    )
    date_for_month = task["assigned_date"].fillna(task.get("source_date", pd.NaT)).fillna(task.get("due_date", pd.NaT))
    task["task_event_date"] = date_for_month
    task = add_month_week_columns(task, "task_event_date", "task_event")
    return task


def prepare_audits(
    audit_raw: pd.DataFrame,
    listitems: pd.DataFrame,
    location_hierarchy: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare AUDIT_VIEW with decoded names, location hierarchy, and audit/observation flags."""
    audit = standardize_columns(audit_raw)
    for col in AUDIT_DATE_COLUMNS:
        if col in audit.columns:
            audit[col] = parse_datetime_utc(audit[col])
    audit = ensure_bool_columns(audit, AUDIT_BOOL_COLUMNS)
    for col in ["audit_id", "parent_id", "root_audit_id", "audit_category_id", "audit_type_id", "audit_status_id", "scheduled_location_id"]:
        if col in audit.columns:
            audit[col] = pd.to_numeric(audit[col], errors="coerce").astype("Int64")

    audit = add_listitem_columns(audit, listitems, specs={
        "audit_category_id": ("auditcategory", "audit_category"),
        "audit_type_id": ("audittype", "audit_type"),
        "audit_status_id": ("auditstatus", "audit_status"),
    })
    audit = audit.merge(
        location_hierarchy.add_prefix("scheduled_"),
        left_on="scheduled_location_id",
        right_on="scheduled_location_id",
        how="left",
        validate="m:1",
    )

    text_cols = ["title", "description", "comments", "associated_parties"]
    for col in text_cols:
        if col in audit.columns:
            audit[col] = clean_text_series(audit[col])
    audit["audit_text"] = combine_text_fields(audit, text_cols)
    audit = add_text_quality_features(audit, "audit_text", "audit_text")

    audit["audit_event_date"] = audit.get("actual_start", pd.NaT).fillna(audit.get("scheduled_start", pd.NaT)).fillna(audit.get("actual_end", pd.NaT))
    audit = add_month_week_columns(audit, "audit_event_date", "audit_event")
    audit["is_active_record"] = audit.get("active", True).fillna(False).astype(bool) & ~audit.get("archived", False).fillna(False).astype(bool)
    audit["is_observation"] = audit["audit_category_name"].eq("Observation")
    audit["is_inspection"] = audit["audit_category_name"].eq("Inspection")
    audit["is_risk_assessment"] = audit["audit_category_name"].eq("Risk Assessment")
    audit["is_unsafe_act"] = audit["audit_type_name"].eq("Unsafe Act")
    audit["is_unsafe_condition"] = audit["audit_type_name"].eq("Unsafe Condition")
    audit["is_safe_act"] = audit["audit_type_name"].eq("Safe Act")
    audit["is_safe_condition"] = audit["audit_type_name"].eq("Safe Condition")
    return audit


def _prepare_group_keys(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    """Create consistently named grouping keys for site/department/month aggregations."""
    out = df.copy()
    # Prefix is used for scheduled_ location columns in audit.
    site_col = f"{prefix}site_name_filled"
    dept_col = f"{prefix}department_name_filled"
    bu_col = f"{prefix}business_unit_name_filled"
    country_col = f"{prefix}country_name_filled"
    region_col = f"{prefix}region_name_filled"
    out["feature_site_name"] = out[site_col] if site_col in out.columns else "Unknown"
    out["feature_department_name"] = out[dept_col] if dept_col in out.columns else "Unknown"
    out["feature_business_unit_name"] = out[bu_col] if bu_col in out.columns else "Unknown"
    out["feature_country_name"] = out[country_col] if country_col in out.columns else "Unknown"
    out["feature_region_name"] = out[region_col] if region_col in out.columns else "Unknown"
    for col in ["feature_site_name", "feature_department_name", "feature_business_unit_name", "feature_country_name", "feature_region_name"]:
        out[col] = out[col].fillna("Unknown").astype(str)
    return out


def make_site_department_month_features(
    incident_enriched: pd.DataFrame,
    task_enriched: pd.DataFrame,
    audit_enriched: pd.DataFrame,
    active_only: bool = True,
) -> pd.DataFrame:
    """Create a joined monthly feature table by site and department.

    This feature table is useful for EDA now and can later be expanded into a time-aware
    supervised modeling table. Current open/overdue task fields are a snapshot as of the
    pipeline reference date, not a historical snapshot.
    """
    keys = ["feature_site_name", "feature_department_name", "feature_business_unit_name", "feature_country_name", "feature_region_name", "feature_month"]

    inc = _prepare_group_keys(incident_enriched)
    if active_only and "is_active_record" in inc.columns:
        inc = inc[inc["is_active_record"]].copy()
    inc = inc[inc["incident_month"].notna()].copy()
    inc["feature_month"] = inc["incident_month"]
    inc["incident_count"] = 1
    inc["near_miss_count"] = inc["incident_category_name"].eq("Near Miss").astype(int)
    inc["hazard_identification_count"] = inc["incident_category_name"].eq("Hazard Identification").astype(int)
    inc["actual_incident_count"] = inc["incident_category_name"].eq("Incident").astype(int)
    inc["injury_case_count"] = inc["injury_count"].fillna(0).astype(int)
    inc["severe_actual_count"] = inc["severe_actual"].fillna(False).astype(int)
    inc_agg = inc.groupby(keys, dropna=False).agg(
        incident_count=("incident_count", "sum"),
        near_miss_count=("near_miss_count", "sum"),
        hazard_identification_count=("hazard_identification_count", "sum"),
        actual_incident_count=("actual_incident_count", "sum"),
        injury_case_count=("injury_case_count", "sum"),
        severe_actual_count=("severe_actual_count", "sum"),
        unique_incident_locations=("location_id", "nunique"),
        median_early_text_words=("text_early_word_count", "median"),
    ).reset_index()

    task = _prepare_group_keys(task_enriched)
    if active_only and "is_active_record" in task.columns:
        task_active = task[task["is_active_record"]].copy()
    else:
        task_active = task.copy()
    task_active = task_active[task_active["task_event_month"].notna()].copy()
    task_active["feature_month"] = task_active["task_event_month"]
    task_agg = task_active.groupby(keys, dropna=False).agg(
        task_count=("task_id", "count"),
        task_open_count=("is_open", "sum"),
        task_overdue_count=("is_overdue", "sum"),
        task_closed_count=("is_closed", "sum"),
        task_avg_days_open=("days_open", "mean"),
        task_max_days_overdue=("days_overdue", "max"),
        corrective_action_count=("task_category_name", lambda s: s.eq("Standalone").sum()),
        incident_action_count=("task_category_name", lambda s: s.eq("Incident").sum()),
        audit_action_count=("task_category_name", lambda s: s.eq("Audit").sum()),
    ).reset_index()

    audit = _prepare_group_keys(audit_enriched, prefix="scheduled_")
    if active_only and "is_active_record" in audit.columns:
        audit = audit[audit["is_active_record"]].copy()
    audit = audit[audit["audit_event_month"].notna()].copy()
    audit["feature_month"] = audit["audit_event_month"]
    audit_agg = audit.groupby(keys, dropna=False).agg(
        audit_record_count=("audit_id", "count"),
        observation_count=("is_observation", "sum"),
        inspection_count=("is_inspection", "sum"),
        risk_assessment_count=("is_risk_assessment", "sum"),
        unsafe_act_count=("is_unsafe_act", "sum"),
        unsafe_condition_count=("is_unsafe_condition", "sum"),
        safe_act_count=("is_safe_act", "sum"),
        safe_condition_count=("is_safe_condition", "sum"),
    ).reset_index()

    # Outer joins preserve months/sites that are present in any source.
    features = inc_agg.merge(task_agg, on=keys, how="outer").merge(audit_agg, on=keys, how="outer")
    numeric_cols = features.select_dtypes(include=["number", "bool"]).columns
    features[numeric_cols] = features[numeric_cols].fillna(0)
    features["unsafe_to_safe_observation_ratio"] = safe_divide(
        features.get("unsafe_act_count", 0) + features.get("unsafe_condition_count", 0),
        features.get("safe_act_count", 0) + features.get("safe_condition_count", 0),
    )
    features["near_miss_hazard_ratio"] = safe_divide(features.get("near_miss_count", 0), features.get("hazard_identification_count", 0))
    return features.sort_values(keys).reset_index(drop=True)
