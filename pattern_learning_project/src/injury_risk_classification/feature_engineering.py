"""Feature engineering for the simplified any-injury risk ranking MVP.

The supervised table has one row per:

    site + department + month

Default target:

    future_any_injury_3m = 1 if the same site/department has at least one
    injury incident in the next 3 calendar months.

The output is intended to rank site/departments for EHS review, not to make a
hard guarantee that an injury will occur. The code also keeps the older
future_severe_actual target available for comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import re

import numpy as np
import pandas as pd

from .utils import (
    add_cyclical_month_features,
    clean_text_value,
    coerce_bool_columns,
    coerce_numeric_id,
    combine_text_fields,
    ensure_dir,
    month_start,
    parse_datetime,
    read_csv,
    safe_divide,
    standardize_columns,
)


INCIDENT_BOOL_COLUMNS = [
    "on_premises", "downtime", "ppe_worn", "insurance_notified", "drug_test_performed",
    "alcohol_test_performed", "rca_performed", "preventable", "recurrence", "process_safety",
    "work_related", "litigable", "sensitive", "agency_sensitive", "media", "title_v",
    "include_in_statistics", "active", "archived",
]
INJURY_BOOL_COLUMNS = [
    "lost_time", "restricted_time", "fatality", "emergency_room", "inpatient",
]
TASK_BOOL_COLUMNS = ["approval_required", "preventive_maintenance", "workflow_dependency", "task_verified", "active", "archived"]
AUDIT_BOOL_COLUMNS = ["controlled", "scheduled", "offline_audit", "modular", "active", "archived"]


@dataclass
class ClassificationDatasetBundle:
    """Container returned by build_classification_dataset.

    baseline_dataset and with_cluster_dataset are the modeling/training rows only
    (complete future target window + minimum history). The *_scoring_dataset
    tables keep the latest scoreable months, including months without a complete
    future label, so score_current_site_risk.py can score the actual current
    month instead of the last fully labeled month.
    """

    baseline_dataset: pd.DataFrame
    with_cluster_dataset: pd.DataFrame | None
    metadata: dict
    baseline_scoring_dataset: pd.DataFrame | None = None
    with_cluster_scoring_dataset: pd.DataFrame | None = None
    pattern_datasets: dict[str, pd.DataFrame] | None = None
    pattern_scoring_datasets: dict[str, pd.DataFrame] | None = None


def load_raw_tables(input_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load the raw CSV files needed for supervised risk modeling."""
    input_dir = Path(input_dir)
    required = {
        "incidents": "INCIDENT_VIEW.csv",
        "injuries": "INCIDENTINJURY_VIEW.csv",
        "locations": "LOCATION_VIEW.csv",
        "listitems": "LISTITEM_VIEW.csv",
    }
    optional = {
        "tasks": "TASK_VIEW.csv",
        "audits": "AUDIT_VIEW.csv",
    }
    tables: dict[str, pd.DataFrame] = {}
    for key, file_name in required.items():
        path = input_dir / file_name
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")
        tables[key] = read_csv(path)
    for key, file_name in optional.items():
        path = input_dir / file_name
        tables[key] = read_csv(path) if path.exists() else pd.DataFrame()
    return tables


def prepare_listitems(listitem_raw: pd.DataFrame) -> pd.DataFrame:
    """Prepare LISTITEM_VIEW as a lookup table."""
    li = standardize_columns(listitem_raw)
    if "list_item_id" in li.columns and "listitem_id" not in li.columns:
        li = li.rename(columns={"list_item_id": "listitem_id"})
    if "listitemid" in li.columns and "listitem_id" not in li.columns:
        li = li.rename(columns={"listitemid": "listitem_id"})
    if "listitem_id" not in li.columns and "listitemid" in li.columns:
        li = li.rename(columns={"listitemid": "listitem_id"})
    # standardize_columns keeps LISTITEMID as listitemid, so handle that explicitly.
    if "listitemid" in li.columns:
        li = li.rename(columns={"listitemid": "listitem_id"})
    for col in ["listitem_id", "parent_id", "location_id"]:
        if col in li.columns:
            li[col] = coerce_numeric_id(li[col])
    for col in ["list_type_code", "code", "item", "shortname", "description"]:
        if col in li.columns:
            li[col] = li[col].map(clean_text_value)
    return li


def add_listitem_name(
    df: pd.DataFrame,
    listitems: pd.DataFrame,
    id_col: str,
    output_prefix: str,
) -> pd.DataFrame:
    """Add a human-readable list item name for a single ID column.

    This function intentionally does not require knowing each LISTTYPECODE. The ID field is
    globally unique in the export, so joining to LISTITEMID is enough for decoding names.
    """
    out = df.copy()
    if id_col not in out.columns or "listitem_id" not in listitems.columns:
        out[f"{output_prefix}_name"] = pd.NA
        return out
    lookup_cols = ["listitem_id"] + [c for c in ["list_type_code", "code", "item", "shortname"] if c in listitems.columns]
    lookup = listitems[lookup_cols].drop_duplicates("listitem_id").rename(columns={
        "listitem_id": id_col,
        "list_type_code": f"{output_prefix}_list_type_code",
        "code": f"{output_prefix}_code",
        "item": f"{output_prefix}_name",
        "shortname": f"{output_prefix}_shortname",
    })
    out[id_col] = coerce_numeric_id(out[id_col])
    return out.merge(lookup, on=id_col, how="left", validate="m:1")


def build_location_hierarchy(location_raw: pd.DataFrame, listitems: pd.DataFrame) -> pd.DataFrame:
    """Build one row per location_id with best-effort site/department hierarchy."""
    loc = standardize_columns(location_raw)
    # Handle source column names after standardization.
    rename = {
        "locationid": "location_id",
        "parentlocationid": "parent_location_id",
        "locationtypeid": "location_type_id",
        "locationcategoryid": "location_category_id",
        "locationstatusid": "location_status_id",
        "locationtreelevel": "location_tree_level",
        "locationcode": "location_code",
    }
    loc = loc.rename(columns={k: v for k, v in rename.items() if k in loc.columns})
    for col in ["location_id", "parent_location_id", "location_type_id", "location_category_id", "location_status_id", "location_tree_level"]:
        if col in loc.columns:
            loc[col] = coerce_numeric_id(loc[col])
    loc = add_listitem_name(loc, listitems, "location_type_id", "location_type")
    loc = add_listitem_name(loc, listitems, "location_category_id", "location_category")
    loc = add_listitem_name(loc, listitems, "location_status_id", "location_status")
    for col in ["location", "location_code"]:
        if col in loc.columns:
            loc[col] = loc[col].map(clean_text_value)
    base_cols = [
        "location_id", "parent_location_id", "location", "location_code", "location_type_id",
        "location_type_name", "location_category_name", "location_status_name", "location_tree_level",
        "active", "archived",
    ]
    for col in base_cols:
        if col not in loc.columns:
            loc[col] = pd.NA
    loc_small = loc[base_cols].drop_duplicates("location_id").copy()
    loc_by_id = loc_small.set_index("location_id", drop=False).to_dict("index")

    type_to_col = {
        "Global": "global_name",
        "Corporate": "corporate_name",
        "Buisiness Unit": "business_unit_name",
        "Business Unit": "business_unit_name",
        "Region": "region_name",
        "Country": "country_name",
        "Site": "site_name",
        "Department": "department_name",
        "Department 2": "department_2_name",
        "Reporting location level 9": "reporting_location_level_9_name",
    }
    hierarchy_cols = sorted(set(type_to_col.values()))
    rows = []
    for raw_id in loc_small["location_id"].dropna().unique():
        location_id = int(raw_id)
        current = loc_by_id.get(location_id)
        ancestors = []
        visited = set()
        for _ in range(60):
            if current is None:
                break
            cid = current.get("location_id")
            if pd.isna(cid):
                break
            cid_int = int(cid)
            if cid_int in visited:
                break
            visited.add(cid_int)
            ancestors.append(current)
            parent_id = current.get("parent_location_id")
            if pd.isna(parent_id):
                break
            current = loc_by_id.get(int(parent_id))
        ancestors = list(reversed(ancestors))
        leaf = loc_by_id[location_id]
        row = {
            "location_id": location_id,
            "location_name": leaf.get("location"),
            "location_code": leaf.get("location_code"),
            "location_type_name": leaf.get("location_type_name"),
            "location_category_name": leaf.get("location_category_name"),
            "location_status_name": leaf.get("location_status_name"),
        }
        for col in hierarchy_cols:
            row[col] = pd.NA
        path_names = []
        path_ids = []
        for anc in ancestors:
            name = anc.get("location")
            typ = anc.get("location_type_name")
            anc_id = anc.get("location_id")
            if pd.notna(name) and str(name).strip():
                path_names.append(str(name))
            if pd.notna(anc_id):
                path_ids.append(str(int(anc_id)))
            if typ in type_to_col:
                row[type_to_col[typ]] = name
        row["location_path"] = " > ".join(path_names)
        row["location_id_path"] = " > ".join(path_ids)
        rows.append(row)
    hierarchy = pd.DataFrame(rows)
    if hierarchy.empty:
        return hierarchy
    hierarchy["site_name_filled"] = hierarchy.get("site_name").fillna(hierarchy["location_name"]).fillna("Unknown")
    hierarchy["department_name_filled"] = (
        hierarchy.get("department_name")
        .fillna(hierarchy.get("department_2_name"))
        .fillna(hierarchy.get("site_name"))
        .fillna(hierarchy["location_name"])
        .fillna("Unknown")
    )
    hierarchy["business_unit_name_filled"] = hierarchy.get("business_unit_name").fillna("Unknown")
    hierarchy["region_name_filled"] = hierarchy.get("region_name").fillna("Unknown")
    hierarchy["country_name_filled"] = hierarchy.get("country_name").fillna("Unknown")
    return hierarchy.sort_values("location_id").reset_index(drop=True)


def prepare_injury_agg(injury_raw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate INCIDENTINJURY_VIEW to one row per incident_id."""
    inj = standardize_columns(injury_raw)
    rename = {"incidentid": "incident_id", "injuryid": "injury_id", "fatalitydate": "fatality_date"}
    inj = inj.rename(columns={k: v for k, v in rename.items() if k in inj.columns})
    for col in ["incident_id", "injury_id"]:
        if col in inj.columns:
            inj[col] = coerce_numeric_id(inj[col])
    if "fatality_date" in inj.columns:
        inj["fatality_date"] = parse_datetime(inj["fatality_date"])
    inj = coerce_bool_columns(inj, INJURY_BOOL_COLUMNS)
    for col in INJURY_BOOL_COLUMNS:
        if col not in inj.columns:
            inj[col] = False
        inj[col] = inj[col].fillna(False).astype(bool)
    if "incident_id" not in inj.columns:
        return pd.DataFrame()
    agg = inj.groupby("incident_id", dropna=False).agg(
        injury_count=("injury_id", "count") if "injury_id" in inj.columns else ("incident_id", "size"),
        lost_time_any=("lost_time", "max"),
        restricted_time_any=("restricted_time", "max"),
        fatality_any=("fatality", "max"),
        emergency_room_any=("emergency_room", "max"),
        inpatient_any=("inpatient", "max"),
    ).reset_index()
    agg["severe_actual"] = (
        agg["lost_time_any"].astype(bool)
        | agg["restricted_time_any"].astype(bool)
        | agg["fatality_any"].astype(bool)
        | agg["inpatient_any"].astype(bool)
    )
    return agg


def prepare_incidents(incident_raw: pd.DataFrame, listitems: pd.DataFrame, location_hierarchy: pd.DataFrame, injury_agg: pd.DataFrame) -> pd.DataFrame:
    """Prepare incident records with decoded names, location hierarchy, text, and severe flags."""
    inc = standardize_columns(incident_raw)
    rename = {
        "incidentid": "incident_id",
        "incidentcategoryid": "incident_category_id",
        "incidentstatusid": "incident_status_id",
        "locationid": "location_id",
        "incidentdate": "incident_date",
        "reportdate": "report_date",
        "incidentnumber": "incident_number",
        "offpremiseslocation": "off_premises_location",
        "otherprocess": "other_process",
        "otheractivity": "other_activity",
        "activityduringincident": "activity_during_incident",
    }
    inc = inc.rename(columns={k: v for k, v in rename.items() if k in inc.columns})
    for col in ["incident_id", "incident_category_id", "incident_status_id", "location_id"]:
        if col in inc.columns:
            inc[col] = coerce_numeric_id(inc[col])
    for col in ["incident_date", "report_date"]:
        if col in inc.columns:
            inc[col] = parse_datetime(inc[col])
    inc = coerce_bool_columns(inc, INCIDENT_BOOL_COLUMNS)
    inc = add_listitem_name(inc, listitems, "incident_category_id", "incident_category")
    inc = add_listitem_name(inc, listitems, "incident_status_id", "incident_status")
    loc_cols = [c for c in location_hierarchy.columns if c != "location_id"]
    inc = inc.merge(location_hierarchy[["location_id"] + loc_cols], on="location_id", how="left", validate="m:1")
    if not injury_agg.empty:
        inc = inc.merge(injury_agg, on="incident_id", how="left", validate="m:1")
    for col in ["injury_count"]:
        if col not in inc.columns:
            inc[col] = 0
        inc[col] = inc[col].fillna(0).astype(int)
    for col in ["lost_time_any", "restricted_time_any", "fatality_any", "emergency_room_any", "inpatient_any", "severe_actual"]:
        if col not in inc.columns:
            inc[col] = False
        inc[col] = inc[col].fillna(False).astype(bool)
    text_cols = ["title", "description", "equipment", "vehicle", "off_premises_location", "other_process", "other_activity", "activity_during_incident"]
    for col in text_cols:
        if col in inc.columns:
            inc[col] = inc[col].map(clean_text_value)
    inc["ml_text_early"] = combine_text_fields(inc, text_cols)
    inc["text_early_word_count"] = inc["ml_text_early"].str.split().str.len().fillna(0).astype(int)
    inc["anchor_month"] = month_start(inc["incident_date"])
    inc["is_active_record"] = inc.get("active", True).fillna(False).astype(bool) & ~inc.get("archived", False).fillna(False).astype(bool)
    inc["incident_category_name"] = inc["incident_category_name"].fillna("Unknown")
    inc["site_name_filled"] = inc["site_name_filled"].fillna("Unknown")
    inc["department_name_filled"] = inc["department_name_filled"].fillna("Unknown")
    inc["region_name_filled"] = inc["region_name_filled"].fillna("Unknown")
    inc["business_unit_name_filled"] = inc["business_unit_name_filled"].fillna("Unknown")
    inc["country_name_filled"] = inc["country_name_filled"].fillna("Unknown")
    return inc


def _monthly_panel(incidents: pd.DataFrame, entity_cols: list[str]) -> pd.DataFrame:
    """Create a complete site/department/month panel for entities with incident history."""
    valid = incidents.dropna(subset=["anchor_month"]).copy()
    if valid.empty:
        raise ValueError("No incident rows with valid incident_date were found.")
    entity = valid[entity_cols].drop_duplicates()
    min_month = valid["anchor_month"].min()
    max_month = valid["anchor_month"].max()
    months = pd.DataFrame({"anchor_month": pd.date_range(min_month, max_month, freq="MS")})
    entity["__key"] = 1
    months["__key"] = 1
    panel = entity.merge(months, on="__key", how="outer").drop(columns="__key")
    return panel.sort_values(entity_cols + ["anchor_month"]).reset_index(drop=True)


def _add_monthly_counts(panel: pd.DataFrame, events: pd.DataFrame, entity_cols: list[str], count_cols: dict[str, pd.Series]) -> pd.DataFrame:
    """Add monthly count columns to panel from event-level boolean masks."""
    out = panel.copy()
    base = events.dropna(subset=["anchor_month"]).copy()
    if base.empty:
        for name in count_cols:
            out[name] = 0
        return out
    for name, mask in count_cols.items():
        if isinstance(mask, pd.Series):
            aligned_mask = mask.reindex(base.index).fillna(False).astype(bool)
        else:
            aligned_mask = pd.Series(mask, index=events.index).reindex(base.index).fillna(False).astype(bool)
        temp = base.loc[aligned_mask, entity_cols + ["anchor_month"]].copy()
        counts = temp.groupby(entity_cols + ["anchor_month"]).size().reset_index(name=name)
        out = out.merge(counts, on=entity_cols + ["anchor_month"], how="left")
        out[name] = out[name].fillna(0).astype(int)
    return out


def _add_rolling_features(df: pd.DataFrame, entity_cols: list[str], source_cols: list[str], windows: list[int]) -> pd.DataFrame:
    """Add rolling sums and growth features for monthly count columns."""
    out = df.sort_values(entity_cols + ["anchor_month"]).copy()
    group = out.groupby(entity_cols, dropna=False)
    for col in source_cols:
        if col not in out.columns:
            out[col] = 0
        for window in windows:
            roll_col = f"{col}_last_{window}m"
            prev_col = f"{col}_prev_{window}m"
            growth_col = f"{col}_growth_{window}m"
            out[roll_col] = group[col].transform(lambda s, w=window: s.rolling(w, min_periods=1).sum())
            out[prev_col] = group[col].transform(lambda s, w=window: s.shift(w).rolling(w, min_periods=1).sum())
            out[prev_col] = out[prev_col].fillna(0)
            out[growth_col] = safe_divide(out[roll_col] - out[prev_col], out[prev_col].replace(0, np.nan), default=0.0)
    return out


def build_incident_features(incidents: pd.DataFrame, entity_cols: list[str], rolling_windows: list[int]) -> pd.DataFrame:
    """Build baseline incident and injury-history features."""
    panel = _monthly_panel(incidents, entity_cols)
    active = incidents["is_active_record"].fillna(True) if "is_active_record" in incidents.columns else pd.Series(True, index=incidents.index)
    category = incidents["incident_category_name"].fillna("Unknown")
    count_masks = {
        "incident_record_count_m": active,
        "near_miss_count_m": active & category.eq("Near Miss"),
        "hazard_identification_count_m": active & category.eq("Hazard Identification"),
        "incident_category_incident_count_m": active & category.eq("Incident"),
        "injury_incident_count_m": active & incidents["injury_count"].fillna(0).gt(0),
        "severe_actual_count_m": active & incidents["severe_actual"].fillna(False).astype(bool),
    }
    panel = _add_monthly_counts(panel, incidents, entity_cols, count_masks)
    count_cols = list(count_masks.keys())
    panel = _add_rolling_features(panel, entity_cols, count_cols, rolling_windows)
    panel = add_cyclical_month_features(panel, "anchor_month")
    return panel


def prepare_tasks(task_raw: pd.DataFrame, listitems: pd.DataFrame, location_hierarchy: pd.DataFrame) -> pd.DataFrame:
    """Prepare task data for point-in-time corrective-action features."""
    if task_raw.empty:
        return pd.DataFrame()
    task = standardize_columns(task_raw)
    rename = {
        "taskid": "task_id",
        "taskcategoryid": "task_category_id",
        "taskstatusid": "task_status_id",
        "locationid": "location_id",
        "assigneddate": "assigned_date",
        "duedate": "due_date",
        "completiondate": "completion_date",
        "markedcompletedate": "marked_complete_date",
        "revisedduedate": "revised_due_date",
        "sourcedate": "source_date",
    }
    task = task.rename(columns={k: v for k, v in rename.items() if k in task.columns})
    for col in ["task_id", "task_category_id", "task_status_id", "location_id"]:
        if col in task.columns:
            task[col] = coerce_numeric_id(task[col])
    for col in ["assigned_date", "due_date", "completion_date", "marked_complete_date", "revised_due_date", "source_date"]:
        if col in task.columns:
            task[col] = parse_datetime(task[col])
    task = coerce_bool_columns(task, TASK_BOOL_COLUMNS)
    task = add_listitem_name(task, listitems, "task_category_id", "task_category")
    task = add_listitem_name(task, listitems, "task_status_id", "task_status")
    loc_cols = [c for c in location_hierarchy.columns if c != "location_id"]
    task = task.merge(location_hierarchy[["location_id"] + loc_cols], on="location_id", how="left", validate="m:1")
    task["site_name_filled"] = task["site_name_filled"].fillna("Unknown")
    task["department_name_filled"] = task["department_name_filled"].fillna("Unknown")
    def _coalesce_datetime(cols: list[str]) -> pd.Series:
        result = pd.Series(pd.NaT, index=task.index, dtype="datetime64[ns]")
        for c in cols:
            if c in task.columns:
                result = result.fillna(task[c])
        return result

    task["task_start_date"] = _coalesce_datetime(["assigned_date", "source_date", "due_date"])
    task["task_close_date"] = _coalesce_datetime(["completion_date", "marked_complete_date"])
    task["task_start_month"] = month_start(task["task_start_date"])
    task["task_close_month"] = month_start(task["task_close_date"])
    task["task_due_month"] = month_start(_coalesce_datetime(["revised_due_date", "due_date"]))
    task["is_active_record"] = task.get("active", True).fillna(False).astype(bool) & ~task.get("archived", False).fillna(False).astype(bool)
    task["days_to_close"] = (task["task_close_date"] - task["task_start_date"]).dt.days
    return task


def build_task_features(panel: pd.DataFrame, tasks: pd.DataFrame, entity_cols: list[str], rolling_windows: list[int]) -> pd.DataFrame:
    """Add corrective-action features to the site/department/month panel.

    Point-in-time open and overdue counts are approximated using monthly cumulative
    assigned/due/closed counts. This avoids leakage from future completion dates.
    """
    out = panel.copy()
    if tasks.empty:
        for col in ["task_assigned_count_m", "task_completed_count_m", "task_due_count_m", "open_task_count", "overdue_open_task_count", "task_closure_rate_last_3m"]:
            out[col] = 0.0
        return out
    t = tasks[tasks["is_active_record"].fillna(True)].copy()
    # monthly assigned, closed, due counts
    monthly_features = []
    for date_col, feature_name in [
        ("task_start_month", "task_assigned_count_m"),
        ("task_close_month", "task_completed_count_m"),
        ("task_due_month", "task_due_count_m"),
    ]:
        tmp = t.dropna(subset=[date_col]).groupby(entity_cols + [date_col]).size().reset_index(name=feature_name)
        tmp = tmp.rename(columns={date_col: "anchor_month"})
        monthly_features.append((feature_name, tmp))
    for feature_name, tmp in monthly_features:
        out = out.merge(tmp, on=entity_cols + ["anchor_month"], how="left")
        out[feature_name] = out[feature_name].fillna(0).astype(int)
    out = _add_rolling_features(out, entity_cols, ["task_assigned_count_m", "task_completed_count_m", "task_due_count_m"], rolling_windows)
    out = out.sort_values(entity_cols + ["anchor_month"])
    group = out.groupby(entity_cols, dropna=False)
    out["open_task_count"] = group["task_assigned_count_m"].cumsum() - group["task_completed_count_m"].cumsum()
    out["open_task_count"] = out["open_task_count"].clip(lower=0)
    out["overdue_open_task_count"] = group["task_due_count_m"].cumsum() - group["task_completed_count_m"].cumsum()
    out["overdue_open_task_count"] = out["overdue_open_task_count"].clip(lower=0)
    out["task_closure_rate_last_3m"] = safe_divide(out["task_completed_count_m_last_3m"], out["task_assigned_count_m_last_3m"], default=0.0)
    return out


def prepare_audits(audit_raw: pd.DataFrame, listitems: pd.DataFrame, location_hierarchy: pd.DataFrame) -> pd.DataFrame:
    """Prepare audit, inspection, and observation data."""
    if audit_raw.empty:
        return pd.DataFrame()
    audit = standardize_columns(audit_raw)
    rename = {
        "auditid": "audit_id",
        "auditcategoryid": "audit_category_id",
        "audittypeid": "audit_type_id",
        "auditstatusid": "audit_status_id",
        "scheduledlocationid": "scheduled_location_id",
        "actualstart": "actual_start",
        "actualend": "actual_end",
        "scheduledstart": "scheduled_start",
        "scheduledend": "scheduled_end",
    }
    audit = audit.rename(columns={k: v for k, v in rename.items() if k in audit.columns})
    for col in ["audit_id", "audit_category_id", "audit_type_id", "audit_status_id", "scheduled_location_id"]:
        if col in audit.columns:
            audit[col] = coerce_numeric_id(audit[col])
    for col in ["actual_start", "actual_end", "scheduled_start", "scheduled_end"]:
        if col in audit.columns:
            audit[col] = parse_datetime(audit[col])
    audit = coerce_bool_columns(audit, AUDIT_BOOL_COLUMNS)
    audit = add_listitem_name(audit, listitems, "audit_category_id", "audit_category")
    audit = add_listitem_name(audit, listitems, "audit_type_id", "audit_type")
    audit = add_listitem_name(audit, listitems, "audit_status_id", "audit_status")
    audit = audit.rename(columns={"scheduled_location_id": "location_id"})
    loc_cols = [c for c in location_hierarchy.columns if c != "location_id"]
    audit = audit.merge(location_hierarchy[["location_id"] + loc_cols], on="location_id", how="left", validate="m:1")
    audit["site_name_filled"] = audit["site_name_filled"].fillna("Unknown")
    audit["department_name_filled"] = audit["department_name_filled"].fillna("Unknown")
    def _coalesce_datetime(cols: list[str]) -> pd.Series:
        result = pd.Series(pd.NaT, index=audit.index, dtype="datetime64[ns]")
        for c in cols:
            if c in audit.columns:
                result = result.fillna(audit[c])
        return result

    audit["audit_event_date"] = _coalesce_datetime(["actual_start", "actual_end", "scheduled_start", "scheduled_end"])
    audit["anchor_month"] = month_start(audit["audit_event_date"])
    audit["is_active_record"] = audit.get("active", True).fillna(False).astype(bool) & ~audit.get("archived", False).fillna(False).astype(bool)
    audit["audit_category_name"] = audit["audit_category_name"].fillna("Unknown")
    return audit


def build_audit_features(panel: pd.DataFrame, audits: pd.DataFrame, entity_cols: list[str], rolling_windows: list[int]) -> pd.DataFrame:
    """Add audit/observation leading-indicator features."""
    out = panel.copy()
    if audits.empty:
        for col in ["audit_record_count_m", "observation_count_m", "inspection_count_m", "risk_assessment_count_m"]:
            out[col] = 0
        return out
    a = audits[audits["is_active_record"].fillna(True)].dropna(subset=["anchor_month"]).copy()
    cat = a["audit_category_name"].fillna("Unknown")
    masks = {
        "audit_record_count_m": pd.Series(True, index=a.index),
        "observation_count_m": cat.eq("Observation"),
        "inspection_count_m": cat.eq("Inspection"),
        "risk_assessment_count_m": cat.eq("Risk Assessment"),
    }
    out = _add_monthly_counts(out, a, entity_cols, masks)
    out = _add_rolling_features(out, entity_cols, list(masks.keys()), rolling_windows)
    return out


def build_target(panel: pd.DataFrame, entity_cols: list[str], horizon_months: int) -> pd.DataFrame:
    """Add future injury targets from monthly counts.

    The target excludes the current month and looks only into future months.
    For example, horizon_months=3 means the next 3 calendar months after
    anchor_month.

    This function creates both:
        future_any_injury_{horizon_months}m
        future_severe_actual_{horizon_months}m

    The simplified MVP trains on future_any_injury by default. The severe target
    is kept so you can compare later without rebuilding the feature pipeline.
    """
    out = panel.sort_values(entity_cols + ["anchor_month"]).copy()
    group = out.groupby(entity_cols, dropna=False)

    def _future_sum(source_col: str) -> pd.Series:
        if source_col not in out.columns:
            out[source_col] = 0
        return group[source_col].transform(
            lambda s: sum(s.shift(-i).fillna(0) for i in range(1, horizon_months + 1))
        ).astype(float)

    future_any_counts = _future_sum("injury_incident_count_m")
    out[f"future_any_injury_next_{horizon_months}m_count"] = future_any_counts
    out[f"future_any_injury_{horizon_months}m"] = future_any_counts.gt(0).astype(int)

    future_severe_counts = _future_sum("severe_actual_count_m")
    out[f"future_severe_actual_next_{horizon_months}m_count"] = future_severe_counts
    out[f"future_severe_actual_{horizon_months}m"] = future_severe_counts.gt(0).astype(int)

    # Rows too close to the end of history cannot have a complete future window.
    max_month = out["anchor_month"].max()
    last_allowed = max_month - pd.DateOffset(months=horizon_months)
    out["has_complete_future_window"] = out["anchor_month"].le(last_allowed)
    return out


def target_column_name(target_type: str, horizon_months: int) -> str:
    """Return the Boolean modeling target column for the configured target type."""
    target_type = str(target_type or "any_injury").lower()
    if target_type in {"any_injury", "any", "injury"}:
        return f"future_any_injury_{horizon_months}m"
    if target_type in {"severe_actual", "severe", "sif_proxy"}:
        return f"future_severe_actual_{horizon_months}m"
    raise ValueError(f"Unsupported target_type: {target_type!r}. Use 'any_injury' or 'severe_actual'.")



def load_clustered_records(clustered_records_path: str | Path | None) -> pd.DataFrame:
    """Load clustered HDBSCAN/theme output when available."""
    if clustered_records_path is None:
        return pd.DataFrame()
    path = Path(clustered_records_path)
    if not path.exists():
        return pd.DataFrame()
    return read_csv(path)


def _safe_feature_token(value: object, fallback: str, max_len: int = 48) -> str:
    """Convert a label into a stable, readable feature-name token."""
    text = clean_text_value(value)
    text = text.lower() if text else fallback.lower()
    text = "".join(ch if ch.isalnum() else "_" for ch in text)
    text = re.sub(r"_+", "_", text).strip("_")
    return (text[:max_len].strip("_") or fallback.lower())


def _safe_dataset_name(value: object, fallback: str = "pattern_features") -> str:
    """Create a filesystem-safe dataset/experiment name."""
    return _safe_feature_token(value, fallback=fallback, max_len=80)


def _normalize_pattern_levels(levels: object) -> list[str]:
    """Normalize pattern level choices to ['theme'], ['cluster'], or both."""
    if isinstance(levels, str):
        raw = [levels]
    elif isinstance(levels, Iterable):
        raw = list(levels)
    else:
        raw = ["cluster"]
    out: list[str] = []
    for level in raw:
        level_str = str(level).strip().lower()
        if level_str in {"theme", "themes"}:
            level_str = "theme"
        elif level_str in {"cluster", "clusters", "hdbscan"}:
            level_str = "cluster"
        else:
            raise ValueError(f"Unsupported pattern feature level: {level!r}. Use 'theme' or 'cluster'.")
        if level_str not in out:
            out.append(level_str)
    return out or ["cluster"]


def default_pattern_feature_config(top_n_clusters: int = 20) -> dict:
    """Default pattern feature configuration used for backward compatibility."""
    return {
        "name": "with_clusters",
        "pattern_levels": ["cluster"],
        "include_aggregate_features": True,
        "include_pattern_id_features": True,
        "include_diversity_features": True,
        "include_outlier_features": True,
        "include_membership_features": True,
        "include_severe_history_features": False,
        "top_n_themes": 20,
        "top_n_clusters": int(top_n_clusters),
        "min_pattern_records": 1,
        "id_selection": "frequency",
        "feature_prefix": "",
    }


def normalize_pattern_feature_config(config: dict | None, top_n_clusters: int = 20) -> dict:
    """Fill missing pattern-feature config keys and validate switches."""
    base = default_pattern_feature_config(top_n_clusters=top_n_clusters)
    if config:
        base.update(dict(config))
    base["name"] = _safe_dataset_name(base.get("name", "with_clusters"), fallback="with_clusters")
    base["pattern_levels"] = _normalize_pattern_levels(base.get("pattern_levels", ["cluster"]))
    for key in [
        "include_aggregate_features",
        "include_pattern_id_features",
        "include_diversity_features",
        "include_outlier_features",
        "include_membership_features",
        "include_severe_history_features",
    ]:
        base[key] = bool(base.get(key, False))
    base["top_n_themes"] = max(0, int(base.get("top_n_themes", 0)))
    base["top_n_clusters"] = max(0, int(base.get("top_n_clusters", top_n_clusters)))
    base["min_pattern_records"] = max(1, int(base.get("min_pattern_records", 1)))
    base["id_selection"] = str(base.get("id_selection", "frequency")).lower()
    if base["id_selection"] != "frequency":
        raise ValueError("Only PATTERN_ID_SELECTION='frequency' is currently supported to avoid target leakage.")

    # Optional production-scoring lock. Training stores selected top pattern IDs in
    # the model manifest. Scoring can pass them back here so the feature schema
    # stays stable even when new data changes global frequencies.
    fixed_raw = base.get("fixed_pattern_ids", {}) or {}
    fixed_clean: dict[str, list[int]] = {}
    if isinstance(fixed_raw, dict):
        for level, ids in fixed_raw.items():
            level_norm = _normalize_pattern_levels([level])[0]
            if ids is None:
                continue
            fixed_clean[level_norm] = [int(x) for x in list(ids)]
    base["fixed_pattern_ids"] = fixed_clean

    prefix = str(base.get("feature_prefix", "") or "")
    prefix = _safe_feature_token(prefix, fallback="", max_len=40) if prefix else ""
    base["feature_prefix"] = prefix
    return base


def _prefixed(name: str, prefix: str = "") -> str:
    return f"{prefix}_{name}" if prefix else name


def prepare_clustered_records(clustered_raw: pd.DataFrame, incidents: pd.DataFrame) -> pd.DataFrame:
    """Clean HDBSCAN clustered records and attach authoritative incident context.

    The updated unsupervised pipeline may include both cluster fields and theme
    fields. This function remains backward compatible with old cluster-only files.
    """
    if clustered_raw.empty:
        return pd.DataFrame()
    cl = standardize_columns(clustered_raw)
    rename = {
        "incidentid": "incident_id",
        "incidentdate": "incident_date",
        "clusterid": "cluster_id",
        "clusterlabel": "cluster_label",
        "topterms": "top_terms",
        "membershipstrength": "membership_strength",
        "isoutlier": "is_outlier",
        "themeid": "theme_id",
        "themelabel": "theme_label",
        "themetopterms": "theme_top_terms",
    }
    cl = cl.rename(columns={k: v for k, v in rename.items() if k in cl.columns})
    if "incident_id" not in cl.columns:
        raise ValueError("Clustered records must include incident_id so they can be linked to incident context.")

    cl["incident_id"] = coerce_numeric_id(cl["incident_id"])
    if "cluster_id" in cl.columns:
        cl["cluster_id"] = pd.to_numeric(cl["cluster_id"], errors="coerce").fillna(-1).astype(int)
    else:
        cl["cluster_id"] = -1

    if "cluster_label" not in cl.columns:
        cl["cluster_label"] = cl["cluster_id"].map(lambda x: f"cluster_{x}" if int(x) != -1 else "Outlier / unassigned")
    cl["cluster_label"] = cl["cluster_label"].fillna(cl["cluster_id"].map(lambda x: f"cluster_{x}"))

    if "membership_strength" not in cl.columns:
        cl["membership_strength"] = np.nan
    cl["membership_strength"] = pd.to_numeric(cl["membership_strength"], errors="coerce")

    if "is_outlier" not in cl.columns:
        cl["is_outlier"] = cl["cluster_id"].eq(-1)
    else:
        cl["is_outlier"] = cl["is_outlier"].fillna(cl["cluster_id"].eq(-1)).astype(bool)

    if "theme_id" in cl.columns:
        cl["theme_id"] = pd.to_numeric(cl["theme_id"], errors="coerce").fillna(-1).astype(int)
    else:
        # Keep theme_id unavailable for old unsupervised outputs. Theme-level
        # features will be skipped unless the file really has valid theme IDs.
        cl["theme_id"] = -1

    if "theme_label" not in cl.columns:
        cl["theme_label"] = cl["theme_id"].map(lambda x: f"theme_{x}" if int(x) != -1 else "Outlier / unassigned theme")
    cl["theme_label"] = cl["theme_label"].fillna(cl["theme_id"].map(lambda x: f"theme_{x}" if int(x) != -1 else "Outlier / unassigned theme"))

    needed_context = [
        "incident_id", "incident_date", "anchor_month", "site_name_filled", "department_name_filled",
        "region_name_filled", "business_unit_name_filled", "country_name_filled", "severe_actual",
    ]
    context_cols = [c for c in needed_context if c in incidents.columns]
    context = incidents[context_cols].drop_duplicates("incident_id")

    # Keep only model-output columns from the clustered file and use the incident
    # table as the source of truth for date/location. This prevents future records
    # excluded by reference_date from leaking back through the clustered file.
    cluster_cols = [
        c for c in [
            "incident_id", "cluster_id", "cluster_label", "top_terms", "theme_id", "theme_label",
            "theme_top_terms", "membership_strength", "is_outlier",
        ] if c in cl.columns
    ]
    out = cl[cluster_cols].merge(context, on="incident_id", how="left", validate="m:1")
    out["cluster_id"] = pd.to_numeric(out["cluster_id"], errors="coerce").fillna(-1).astype(int)
    out["theme_id"] = pd.to_numeric(out.get("theme_id", -1), errors="coerce").fillna(-1).astype(int)
    out["is_outlier"] = out["is_outlier"].fillna(out["cluster_id"].eq(-1)).astype(bool)
    out["cluster_label"] = out["cluster_label"].fillna(out["cluster_id"].map(lambda x: f"cluster_{x}"))
    out["theme_label"] = out["theme_label"].fillna(out["theme_id"].map(lambda x: f"theme_{x}" if int(x) != -1 else "Outlier / unassigned theme"))
    if "severe_actual" in out.columns:
        out["severe_actual"] = out["severe_actual"].fillna(False).astype(bool)
    else:
        out["severe_actual"] = False
    return out.dropna(subset=["anchor_month"])


def _add_rolling_unique_id_features(
    panel: pd.DataFrame,
    events: pd.DataFrame,
    entity_cols: list[str],
    id_col: str,
    monthly_col: str,
    rolling_base_col: str,
    windows: list[int],
) -> pd.DataFrame:
    """Add exact rolling distinct pattern-ID counts per entity/month."""
    out = panel.copy()
    if events.empty or id_col not in events.columns:
        out[monthly_col] = 0
        for window in windows:
            out[f"{rolling_base_col}_last_{window}m"] = 0
        return out

    valid = events.dropna(subset=["anchor_month"]).copy()
    valid[id_col] = pd.to_numeric(valid[id_col], errors="coerce").fillna(-1).astype(int)
    valid = valid[valid[id_col].ge(0)]
    if valid.empty:
        out[monthly_col] = 0
        for window in windows:
            out[f"{rolling_base_col}_last_{window}m"] = 0
        return out

    monthly = (
        valid.drop_duplicates(entity_cols + ["anchor_month", id_col])
        .groupby(entity_cols + ["anchor_month"], dropna=False)
        .size()
        .reset_index(name=monthly_col)
    )
    out = out.merge(monthly, on=entity_cols + ["anchor_month"], how="left")
    out[monthly_col] = out[monthly_col].fillna(0).astype(int)

    valid_small = valid[entity_cols + ["anchor_month", id_col]].drop_duplicates().copy()
    valid_small["anchor_month"] = pd.to_datetime(valid_small["anchor_month"])
    out["anchor_month"] = pd.to_datetime(out["anchor_month"])

    result_frames = []
    for _, entity_panel in out[entity_cols + ["anchor_month"]].drop_duplicates().groupby(entity_cols, dropna=False):
        key_values = entity_panel.iloc[0][entity_cols].to_dict()
        mask = pd.Series(True, index=valid_small.index)
        for col in entity_cols:
            mask &= valid_small[col].eq(key_values[col])
        entity_events = valid_small.loc[mask, ["anchor_month", id_col]].copy()
        rows = []
        for anchor in entity_panel["anchor_month"].sort_values():
            row = dict(key_values)
            row["anchor_month"] = anchor
            for window in windows:
                start = anchor - pd.DateOffset(months=int(window) - 1)
                ids = entity_events.loc[
                    entity_events["anchor_month"].between(start, anchor, inclusive="both"), id_col
                ]
                row[f"{rolling_base_col}_last_{window}m"] = int(ids.nunique())
            rows.append(row)
        if rows:
            result_frames.append(pd.DataFrame(rows))

    if result_frames:
        rolling_unique = pd.concat(result_frames, ignore_index=True)
        out = out.merge(rolling_unique, on=entity_cols + ["anchor_month"], how="left")
    for window in windows:
        col = f"{rolling_base_col}_last_{window}m"
        if col not in out.columns:
            out[col] = 0
        out[col] = out[col].fillna(0).astype(int)
    return out


def _add_membership_strength_features(
    panel: pd.DataFrame,
    clustered: pd.DataFrame,
    entity_cols: list[str],
    rolling_windows: list[int],
    prefix: str,
) -> pd.DataFrame:
    """Add monthly and rolling average membership-strength features."""
    out = panel.copy()
    base = clustered.dropna(subset=["anchor_month"]).copy()
    base = base[~base["is_outlier"].fillna(False)].copy()
    base["membership_strength"] = pd.to_numeric(base.get("membership_strength"), errors="coerce")
    base = base[base["membership_strength"].notna()]

    sum_col = _prefixed("pattern_membership_strength_sum_m", prefix)
    count_col = _prefixed("pattern_membership_strength_count_m", prefix)
    avg_col = _prefixed("avg_pattern_membership_strength_m", prefix)
    if base.empty:
        out[avg_col] = 0.0
        for window in rolling_windows:
            out[_prefixed(f"avg_pattern_membership_strength_last_{window}m", prefix)] = 0.0
        return out

    monthly = base.groupby(entity_cols + ["anchor_month"], dropna=False).agg(
        **{
            sum_col: ("membership_strength", "sum"),
            count_col: ("membership_strength", "count"),
        }
    ).reset_index()
    out = out.merge(monthly, on=entity_cols + ["anchor_month"], how="left")
    out[sum_col] = out[sum_col].fillna(0.0)
    out[count_col] = out[count_col].fillna(0.0)
    out[avg_col] = safe_divide(out[sum_col], out[count_col], default=0.0)

    out = out.sort_values(entity_cols + ["anchor_month"]).copy()
    group = out.groupby(entity_cols, dropna=False)
    for window in rolling_windows:
        sum_last = group[sum_col].transform(lambda s, w=window: s.rolling(w, min_periods=1).sum())
        count_last = group[count_col].transform(lambda s, w=window: s.rolling(w, min_periods=1).sum())
        out[_prefixed(f"avg_pattern_membership_strength_last_{window}m", prefix)] = safe_divide(sum_last, count_last, default=0.0)

    out = out.drop(columns=[sum_col, count_col], errors="ignore")
    return out


def _select_top_pattern_ids(
    clustered: pd.DataFrame,
    level: str,
    top_n: int,
    min_pattern_records: int,
) -> list[int]:
    """Select top theme/cluster IDs by global frequency among non-outliers."""
    if top_n <= 0:
        return []
    id_col = f"{level}_id"
    if id_col not in clustered.columns:
        return []
    valid = clustered.loc[~clustered["is_outlier"].fillna(False)].copy()
    valid[id_col] = pd.to_numeric(valid[id_col], errors="coerce").fillna(-1).astype(int)
    valid = valid[valid[id_col].ge(0)]
    counts = valid[id_col].value_counts()
    counts = counts[counts.ge(int(min_pattern_records))]
    return [int(x) for x in counts.head(int(top_n)).index.tolist()]


def _add_top_pattern_id_features(
    panel: pd.DataFrame,
    clustered: pd.DataFrame,
    entity_cols: list[str],
    rolling_windows: list[int],
    level: str,
    top_ids: list[int],
    feature_prefix: str,
) -> tuple[pd.DataFrame, dict]:
    """Add separate monthly/rolling count features for selected theme/cluster IDs."""
    out = panel.copy()
    id_col = f"{level}_id"
    label_col = f"{level}_label"
    feature_map: dict[int, dict] = {}
    if not top_ids or id_col not in clustered.columns:
        return out, feature_map

    valid = clustered.loc[~clustered["is_outlier"].fillna(False)].copy()
    valid[id_col] = pd.to_numeric(valid[id_col], errors="coerce").fillna(-1).astype(int)
    for pattern_id in top_ids:
        labels = valid.loc[valid[id_col].eq(pattern_id), label_col].dropna().astype(str) if label_col in valid.columns else pd.Series(dtype=str)
        label = labels.iloc[0] if len(labels) else f"{level}_{pattern_id}"
        safe_label = _safe_feature_token(label, fallback=f"{level}_{pattern_id}")
        feature_base = _prefixed(f"top_{level}_{pattern_id}_{safe_label}_count_m", feature_prefix)
        tmp = valid.loc[valid[id_col].eq(pattern_id), entity_cols + ["anchor_month"]]
        counts = tmp.groupby(entity_cols + ["anchor_month"], dropna=False).size().reset_index(name=feature_base)
        out = out.merge(counts, on=entity_cols + ["anchor_month"], how="left")
        out[feature_base] = out[feature_base].fillna(0).astype(int)
        feature_map[int(pattern_id)] = {
            "level": level,
            "label": label,
            "feature_base": feature_base,
            "record_count": int(len(tmp)),
        }
    if feature_map:
        out = _add_rolling_features(out, entity_cols, [v["feature_base"] for v in feature_map.values()], rolling_windows)
    return out, feature_map


def build_pattern_features(
    panel: pd.DataFrame,
    clustered: pd.DataFrame,
    entity_cols: list[str],
    rolling_windows: list[int],
    top_n_clusters: int = 20,
    pattern_feature_config: dict | None = None,
) -> pd.DataFrame:
    """Add optional HDBSCAN cluster/theme features to the site-month panel.

    This function supports the main experimental switches:
      - aggregate counts across all pattern records regardless of theme/cluster
      - per-theme count features
      - per-cluster count features
      - theme/cluster diversity
      - outlier/new-pattern candidate counts
      - rolling average membership strength
    """
    cfg = normalize_pattern_feature_config(pattern_feature_config, top_n_clusters=top_n_clusters)
    prefix = cfg.get("feature_prefix", "")
    out = panel.copy()
    if clustered.empty:
        out[_prefixed("pattern_feature_available", prefix)] = 0
        out.attrs["pattern_feature_config"] = cfg
        out.attrs["pattern_feature_map"] = {}
        return out

    cl = clustered.dropna(subset=["anchor_month"]).copy()
    cl["is_outlier"] = cl["is_outlier"].fillna(cl["cluster_id"].eq(-1)).astype(bool)
    cl["cluster_id"] = pd.to_numeric(cl.get("cluster_id", -1), errors="coerce").fillna(-1).astype(int)
    cl["theme_id"] = pd.to_numeric(cl.get("theme_id", -1), errors="coerce").fillna(-1).astype(int)
    cl["non_outlier_pattern_event"] = ~cl["is_outlier"]
    cl["outlier_pattern_event"] = cl["is_outlier"]

    feature_map: dict[str, object] = {
        "config": cfg,
        "available_levels": {
            "cluster": int(cl.loc[~cl["is_outlier"] & cl["cluster_id"].ge(0), "cluster_id"].nunique()),
            "theme": int(cl.loc[~cl["is_outlier"] & cl["theme_id"].ge(0), "theme_id"].nunique()),
        },
        "top_pattern_ids": {},
    }

    # Aggregate counts across all pattern records. These are the "regardless of
    # specific pattern" features requested by the user.
    need_base_counts = (
        cfg["include_aggregate_features"]
        or cfg["include_outlier_features"]
        or cfg["include_membership_features"]
        or cfg["include_severe_history_features"]
    )
    if need_base_counts:
        masks = {
            _prefixed("pattern_event_count_m", prefix): pd.Series(True, index=cl.index),
            _prefixed("assigned_pattern_count_m", prefix): cl["non_outlier_pattern_event"],
        }
        if cfg["include_outlier_features"]:
            masks[_prefixed("outlier_pattern_count_m", prefix)] = cl["outlier_pattern_event"]
        if cfg["include_severe_history_features"]:
            masks[_prefixed("pattern_severe_actual_count_m", prefix)] = cl["severe_actual"].fillna(False).astype(bool)
        out = _add_monthly_counts(out, cl, entity_cols, masks)
        out = _add_rolling_features(out, entity_cols, list(masks.keys()), rolling_windows)

        if cfg["include_outlier_features"]:
            for window in rolling_windows:
                out[_prefixed(f"outlier_pattern_rate_last_{window}m", prefix)] = safe_divide(
                    out.get(_prefixed(f"outlier_pattern_count_m_last_{window}m", prefix), 0),
                    out.get(_prefixed(f"pattern_event_count_m_last_{window}m", prefix), 0),
                    default=0.0,
                )

    # Diversity features: exact rolling distinct theme/cluster counts.
    if cfg["include_diversity_features"]:
        for level in cfg["pattern_levels"]:
            id_col = f"{level}_id"
            if id_col not in cl.columns:
                continue
            valid_level = cl.loc[~cl["is_outlier"] & pd.to_numeric(cl[id_col], errors="coerce").fillna(-1).astype(int).ge(0)].copy()
            if valid_level.empty:
                continue
            out = _add_rolling_unique_id_features(
                out,
                valid_level,
                entity_cols=entity_cols,
                id_col=id_col,
                monthly_col=_prefixed(f"unique_{level}_count_m", prefix),
                rolling_base_col=_prefixed(f"unique_{level}_count", prefix),
                windows=rolling_windows,
            )

    # Per theme/cluster count/trend features. These preserve pattern identity.
    if cfg["include_pattern_id_features"]:
        for level in cfg["pattern_levels"]:
            top_n = cfg["top_n_themes"] if level == "theme" else cfg["top_n_clusters"]
            fixed_ids = (cfg.get("fixed_pattern_ids") or {}).get(level)
            if fixed_ids is not None:
                top_ids = [int(x) for x in fixed_ids]
            else:
                top_ids = _select_top_pattern_ids(
                    cl,
                    level=level,
                    top_n=int(top_n),
                    min_pattern_records=int(cfg["min_pattern_records"]),
                )
            out, level_map = _add_top_pattern_id_features(
                out,
                cl,
                entity_cols=entity_cols,
                rolling_windows=rolling_windows,
                level=level,
                top_ids=top_ids,
                feature_prefix=prefix,
            )
            feature_map["top_pattern_ids"][level] = level_map

    if cfg["include_membership_features"]:
        out = _add_membership_strength_features(out, cl, entity_cols, rolling_windows, prefix=prefix)

    out[_prefixed("pattern_feature_available", prefix)] = 1
    # Backward-compatible alias used by old downstream checks.
    if not prefix:
        out["cluster_feature_available"] = 1
    out.attrs["pattern_feature_config"] = cfg
    out.attrs["pattern_feature_map"] = feature_map
    out.attrs["cluster_name_map"] = feature_map.get("top_pattern_ids", {}).get("cluster", {})
    return out


def build_cluster_features(
    panel: pd.DataFrame,
    clustered: pd.DataFrame,
    entity_cols: list[str],
    rolling_windows: list[int],
    top_n_clusters: int = 20,
    pattern_feature_config: dict | None = None,
) -> pd.DataFrame:
    """Backward-compatible wrapper around build_pattern_features."""
    return build_pattern_features(
        panel=panel,
        clustered=clustered,
        entity_cols=entity_cols,
        rolling_windows=rolling_windows,
        top_n_clusters=top_n_clusters,
        pattern_feature_config=pattern_feature_config,
    )


def save_dataframe_outputs(df: pd.DataFrame, output_base: Path, sample_rows: int = 5000) -> dict[str, str]:
    """Save a full dataframe efficiently plus a small CSV preview.

    Full feature matrices can be large and slow to write as CSV. The function prefers
    Parquet, falls back to pickle when Parquet engines are unavailable, and always writes
    a preview CSV for easy inspection in Excel/VS Code.
    """
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
    preview_path = output_base.with_name(output_base.name + "_preview_5000_rows").with_suffix(".csv")
    df.head(sample_rows).to_csv(preview_path, index=False)
    paths["preview_csv"] = str(preview_path)
    paths["n_rows"] = str(len(df))
    paths["n_cols"] = str(df.shape[1])
    return paths



def build_classification_dataset(
    input_dir: str | Path,
    output_dir: str | Path,
    clustered_records_path: str | Path | None = None,
    horizon_months: int = 3,
    target_type: str = "any_injury",
    rolling_windows: list[int] | None = None,
    top_n_clusters: int = 20,
    min_history_months: int = 6,
    reference_date: str | pd.Timestamp | None = None,
    write_outputs: bool = True,
    pattern_feature_config: dict | None = None,
    pattern_feature_experiments: list[dict] | None = None,
) -> ClassificationDatasetBundle:
    """Build baseline and optional pattern-feature classification datasets.

    The modeling datasets include only rows with a complete future target window.
    The scoring datasets include the latest scoreable months, even when the future
    target window is incomplete, so the current-risk scoring script can score the
    true current month.
    """
    rolling_windows = rolling_windows or [3, 6]
    output_dir = Path(output_dir)
    tables = load_raw_tables(input_dir)
    listitems = prepare_listitems(tables["listitems"])
    location_hierarchy = build_location_hierarchy(tables["locations"], listitems)
    injury_agg = prepare_injury_agg(tables["injuries"])
    incidents = prepare_incidents(tables["incidents"], listitems, location_hierarchy, injury_agg)
    reference_ts = pd.to_datetime(reference_date) if reference_date is not None else None
    if reference_ts is not None:
        incidents = incidents[incidents["incident_date"].notna() & incidents["incident_date"].le(reference_ts)].copy()
    entity_cols = ["site_name_filled", "department_name_filled"]

    panel = build_incident_features(incidents, entity_cols, rolling_windows)
    tasks = prepare_tasks(tables.get("tasks", pd.DataFrame()), listitems, location_hierarchy)
    if reference_ts is not None and not tasks.empty:
        tasks = tasks[tasks["task_start_date"].isna() | tasks["task_start_date"].le(reference_ts)].copy()
    panel = build_task_features(panel, tasks, entity_cols, rolling_windows)
    audits = prepare_audits(tables.get("audits", pd.DataFrame()), listitems, location_hierarchy)
    if reference_ts is not None and not audits.empty:
        audits = audits[audits["audit_event_date"].isna() | audits["audit_event_date"].le(reference_ts)].copy()
    panel = build_audit_features(panel, audits, entity_cols, rolling_windows)

    # Add static hierarchy metadata using latest observed location details for each entity.
    static_cols = entity_cols + ["region_name_filled", "business_unit_name_filled", "country_name_filled"]
    static_meta = incidents[static_cols].drop_duplicates(entity_cols).drop_duplicates(entity_cols, keep="last")
    panel = panel.merge(static_meta, on=entity_cols, how="left", validate="m:1")
    panel["region_name_filled"] = panel["region_name_filled"].fillna("Unknown")
    panel["business_unit_name_filled"] = panel["business_unit_name_filled"].fillna("Unknown")
    panel["country_name_filled"] = panel["country_name_filled"].fillna("Unknown")

    panel = build_target(panel, entity_cols, horizon_months=horizon_months)
    panel = panel.sort_values(entity_cols + ["anchor_month"])
    panel["months_since_entity_start"] = panel.groupby(entity_cols, dropna=False).cumcount()
    panel["eligible_for_modeling"] = panel["has_complete_future_window"] & panel["months_since_entity_start"].ge(min_history_months)
    panel["eligible_for_scoring"] = panel["months_since_entity_start"].ge(min_history_months)

    baseline = panel.loc[panel["eligible_for_modeling"]].reset_index(drop=True).copy()
    baseline_scoring = panel.loc[panel["eligible_for_scoring"]].reset_index(drop=True).copy()

    clustered_raw = load_clustered_records(clustered_records_path)
    clustered = prepare_clustered_records(clustered_raw, incidents) if not clustered_raw.empty else pd.DataFrame()

    default_cfg = normalize_pattern_feature_config(pattern_feature_config, top_n_clusters=top_n_clusters)
    with_clusters = None
    with_clusters_scoring = None
    pattern_feature_map: dict = {}
    if not clustered.empty:
        cluster_panel = build_cluster_features(
            panel,
            clustered,
            entity_cols,
            rolling_windows,
            top_n_clusters=top_n_clusters,
            pattern_feature_config=default_cfg,
        )
        pattern_feature_map = cluster_panel.attrs.get("pattern_feature_map", {})
        with_clusters = cluster_panel.loc[cluster_panel["eligible_for_modeling"]].reset_index(drop=True).copy()
        with_clusters_scoring = cluster_panel.loc[cluster_panel["eligible_for_scoring"]].reset_index(drop=True).copy()

    pattern_datasets: dict[str, pd.DataFrame] = {}
    pattern_scoring_datasets: dict[str, pd.DataFrame] = {}
    experiment_feature_maps: dict[str, dict] = {}
    if not clustered.empty and pattern_feature_experiments:
        for raw_cfg in pattern_feature_experiments:
            cfg = normalize_pattern_feature_config(raw_cfg, top_n_clusters=top_n_clusters)
            exp_name = cfg["name"]
            exp_panel = build_cluster_features(
                panel,
                clustered,
                entity_cols,
                rolling_windows,
                top_n_clusters=top_n_clusters,
                pattern_feature_config=cfg,
            )
            pattern_datasets[exp_name] = exp_panel.loc[exp_panel["eligible_for_modeling"]].reset_index(drop=True).copy()
            pattern_scoring_datasets[exp_name] = exp_panel.loc[exp_panel["eligible_for_scoring"]].reset_index(drop=True).copy()
            experiment_feature_maps[exp_name] = exp_panel.attrs.get("pattern_feature_map", {})

    target_name = target_column_name(target_type, horizon_months)
    valid_cluster_count = int(clustered.loc[~clustered.get("is_outlier", pd.Series(False, index=clustered.index)).fillna(False) & clustered.get("cluster_id", pd.Series(-1, index=clustered.index)).ge(0), "cluster_id"].nunique()) if not clustered.empty and "cluster_id" in clustered.columns else 0
    valid_theme_count = int(clustered.loc[~clustered.get("is_outlier", pd.Series(False, index=clustered.index)).fillna(False) & clustered.get("theme_id", pd.Series(-1, index=clustered.index)).ge(0), "theme_id"].nunique()) if not clustered.empty and "theme_id" in clustered.columns else 0

    metadata = {
        "task": "site_department_month_future_any_injury_risk_ranking",
        "row_definition": "one row per site + department + calendar month",
        "target_type": str(target_type),
        "target_column": target_name,
        "target_definition": f"{target_name} = 1 if at least one injury incident occurs in the next {horizon_months} months for the same site/department",
        "any_injury_definition": "injury_count > 0 on an incident record",
        "severe_actual_definition": "fatality_any OR lost_time_any OR restricted_time_any OR inpatient_any",
        "horizon_months": horizon_months,
        "rolling_windows_months": rolling_windows,
        "entity_columns": entity_cols,
        "min_history_months": min_history_months,
        "reference_date": str(reference_ts.date()) if reference_ts is not None else None,
        "n_incident_rows": int(len(incidents)),
        "n_injury_rows_aggregated": int(len(injury_agg)),
        "n_panel_rows_before_filter": int(len(panel)),
        "n_baseline_rows": int(len(baseline)),
        "n_baseline_scoring_rows": int(len(baseline_scoring)),
        "n_baseline_positive_rows": int(baseline[target_name].sum()) if len(baseline) else 0,
        "clustered_records_path": str(clustered_records_path) if clustered_records_path else None,
        "n_clustered_records": int(len(clustered)),
        "valid_cluster_count": valid_cluster_count,
        "valid_theme_count": valid_theme_count,
        "default_pattern_feature_config": default_cfg,
        "default_pattern_feature_map": pattern_feature_map,
        "pattern_feature_experiments": [normalize_pattern_feature_config(c, top_n_clusters=top_n_clusters) for c in (pattern_feature_experiments or [])],
        "experiment_pattern_feature_maps": experiment_feature_maps,
        "n_with_cluster_rows": int(len(with_clusters)) if with_clusters is not None else 0,
        "n_with_cluster_scoring_rows": int(len(with_clusters_scoring)) if with_clusters_scoring is not None else 0,
        "pattern_experiment_rows": {name: int(len(df)) for name, df in pattern_datasets.items()},
        "leakage_controls": [
            "Injury outcome fields are used to build future targets and historical aggregate features only.",
            "The target is shifted into the future and excludes the current month.",
            "Training uses rows with has_complete_future_window=True; current scoring uses rows with enough history even when future labels are incomplete.",
            "Temporal train/test split should be used downstream.",
            "Raw embeddings are not used directly in this site-level classifier; cluster/theme output is converted to interpretable monthly count/trend/diversity features.",
            "Top theme/cluster IDs are selected by frequency only, not by future target performance.",
        ],
    }

    if write_outputs:
        feature_dir = ensure_dir(output_dir / "ml" / "injury_risk_classification" / "features")
        saved_files = {
            "baseline_dataset": save_dataframe_outputs(baseline, feature_dir / "classification_dataset_baseline"),
            "baseline_scoring_dataset": save_dataframe_outputs(baseline_scoring, feature_dir / "classification_dataset_baseline_scoring"),
        }
        if with_clusters is not None:
            saved_files["with_cluster_dataset"] = save_dataframe_outputs(with_clusters, feature_dir / "classification_dataset_with_clusters")
        if with_clusters_scoring is not None:
            saved_files["with_cluster_scoring_dataset"] = save_dataframe_outputs(with_clusters_scoring, feature_dir / "classification_dataset_with_clusters_scoring")
        for name, df in pattern_datasets.items():
            saved_files[f"experiment_{name}_dataset"] = save_dataframe_outputs(df, feature_dir / f"classification_dataset_{name}")
        for name, df in pattern_scoring_datasets.items():
            saved_files[f"experiment_{name}_scoring_dataset"] = save_dataframe_outputs(df, feature_dir / f"classification_dataset_{name}_scoring")
        location_hierarchy.to_csv(feature_dir / "location_hierarchy_for_classification.csv", index=False)
        injury_agg.to_csv(feature_dir / "injury_agg_for_classification.csv", index=False)
        metadata["saved_feature_files"] = saved_files
        from .utils import save_json
        save_json(metadata, feature_dir / "classification_dataset_metadata.json")

    return ClassificationDatasetBundle(
        baseline_dataset=baseline,
        with_cluster_dataset=with_clusters,
        baseline_scoring_dataset=baseline_scoring,
        with_cluster_scoring_dataset=with_clusters_scoring,
        pattern_datasets=pattern_datasets,
        pattern_scoring_datasets=pattern_scoring_datasets,
        metadata=metadata,
    )
