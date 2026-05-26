"""Utility helpers for the Pattern Learning EHS data preparation pipeline."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


TRUE_VALUES = {"true", "1", "yes", "y", "t"}
FALSE_VALUES = {"false", "0", "no", "n", "f"}


COMMON_COLUMN_ALIASES = {
    # shared ids and metadata
    "clientid": "client_id",
    "locationid": "location_id",
    "historyid": "history_id",
    "parentid": "parent_id",
    "active": "active",
    "archived": "archived",
    "shortname": "short_name",
    # listitem
    "listitemid": "list_item_id",
    "listtypecode": "list_type_code",
    "listorder": "list_order",
    # incident
    "reportdate": "report_date",
    "investigationstartdate": "investigation_start_date",
    "insurancenotifieddate": "insurance_notified_date",
    "incidentdate": "incident_date",
    "enddate": "end_date",
    "drugtestdate": "drug_test_date",
    "alcoholtestdate": "alcohol_test_date",
    "incidentid": "incident_id",
    "incidentcategoryid": "incident_category_id",
    "incidentstatusid": "incident_status_id",
    "offsitelocationid": "offsite_location_id",
    "contractingcompanyid": "contracting_company_id",
    "addressid": "address_id",
    "processid": "process_id",
    "activityid": "activity_id",
    "onpremises": "on_premises",
    "downtime": "downtime",
    "ppeworn": "ppe_worn",
    "insurancenotified": "insurance_notified",
    "drugtestperformed": "drug_test_performed",
    "alcoholtestperformed": "alcohol_test_performed",
    "rcaperformed": "rca_performed",
    "processsafety": "process_safety",
    "workrelated": "work_related",
    "agencysensitive": "agency_sensitive",
    "titlev": "title_v",
    "includeinstatistics": "include_in_statistics",
    "downtimeamount": "downtime_amount",
    "windspeed": "wind_speed",
    "timetoextinguish": "time_to_extinguish",
    "downtimeunitcode": "downtime_unit_code",
    "windspeedunitcode": "wind_speed_unit_code",
    "timetoextinguishunitcode": "time_to_extinguish_unit_code",
    "incidentnumber": "incident_number",
    "reportedby": "reported_by",
    "reporterphone": "reporter_phone",
    "insuranceclaim": "insurance_claim",
    "offpremiseslocation": "off_premises_location",
    "otherlocation": "other_location",
    "othercontractingcompany": "other_contracting_company",
    "otherprocess": "other_process",
    "otheractivity": "other_activity",
    "otherppe": "other_ppe",
    "otherweather": "other_weather",
    "activityduringincident": "activity_during_incident",
    "immediateaction": "immediate_action",
    "immediatecauses": "immediate_causes",
    "causalfactors": "causal_factors",
    "bestpractices": "best_practices",
    "riskaction": "risk_action",
    "riskcondition": "risk_condition",
    "insuranceinfo": "insurance_info",
    # injury
    "fatalitydate": "fatality_date",
    "effectnoticed": "effect_noticed",
    "injuryid": "injury_id",
    "incidentinvolvedid": "incident_involved_id",
    "losttime": "lost_time",
    "restrictedtime": "restricted_time",
    "fatality": "fatality",
    "emergencyroom": "emergency_room",
    "inpatient": "inpatient",
    "repeatedmotion": "repeated_motion",
    "liftedfromfloor": "lifted_from_floor",
    "liftingaidsavailable": "lifting_aids_available",
    "liftingaidsused": "lifting_aids_used",
    "suddenpain": "sudden_pain",
    "nonpaineffect": "non_pain_effect",
    "fallheight": "fall_height",
    "losttimeestimate": "lost_time_estimate",
    "losttimeactual": "lost_time_actual",
    "restrictedtimeestimate": "restricted_time_estimate",
    "restrictedtimeactual": "restricted_time_actual",
    "distancemoved": "distance_moved",
    "repeatfrequency": "repeat_frequency",
    "objectsize": "object_size",
    "fallheightunitcode": "fall_height_unit_code",
    "distancemovedunitcode": "distance_moved_unit_code",
    "repeatfrequencyunitcode": "repeat_frequency_unit_code",
    "objectsizeunitcode": "object_size_unit_code",
    "firstaidresponder": "first_aid_responder",
    "otherobjectresponsible": "other_object_responsible",
    "otherinjurytype": "other_injury_type",
    "otherbodypart": "other_body_part",
    "otherinjurycause": "other_injury_cause",
    "othertreatment": "other_treatment",
    "physicaleffect": "physical_effect",
    "effectchanges": "effect_changes",
    "materialexposure": "material_exposure",
    "ingestedprior": "ingested_prior",
    "paincause": "pain_cause",
    # location
    "parentlocationid": "parent_location_id",
    "locationcategoryid": "location_category_id",
    "locationstatusid": "location_status_id",
    "locationtypeid": "location_type_id",
    "locationtreelevel": "location_tree_level",
    "locationcode": "location_code",
    "externalreference": "external_reference",
    "locationorder": "location_order",
    # task
    "verificationduedate": "verification_due_date",
    "startdate": "start_date",
    "sourcedate": "source_date",
    "revisedduedate": "revised_due_date",
    "markedcompletedate": "marked_complete_date",
    "completiondate": "completion_date",
    "assigneddate": "assigned_date",
    "duedate": "due_date",
    "taskid": "task_id",
    "parenttaskid": "parent_task_id",
    "taskcategoryid": "task_category_id",
    "tasktypeid": "task_type_id",
    "taskstatusid": "task_status_id",
    "recurrenceid": "recurrence_id",
    "otherlocationid": "other_location_id",
    "sourcetypeid": "source_type_id",
    "approvalrequired": "approval_required",
    "preventivemaintenance": "preventive_maintenance",
    "workflowdependency": "workflow_dependency",
    "taskverified": "task_verified",
    "percentcomplete": "percent_complete",
    "tasknumber": "task_number",
    "workordernumber": "work_order_number",
    "externalnumber": "external_number",
    "otherlocationname": "other_location_name",
    "permitsection": "permit_section",
    "verificationreason": "verification_reason",
    # audit
    "actualend": "actual_end",
    "actualstart": "actual_start",
    "scheduledend": "scheduled_end",
    "scheduledstart": "scheduled_start",
    "auditid": "audit_id",
    "rootauditid": "root_audit_id",
    "auditcategoryid": "audit_category_id",
    "audittypeid": "audit_type_id",
    "auditstatusid": "audit_status_id",
    "scheduledlocationid": "scheduled_location_id",
    "rootquestionid": "root_question_id",
    "offlineaudit": "offline_audit",
    "auditnumber": "audit_number",
    "associatedparties": "associated_parties",
}


def snake_case(name: str) -> str:
    """Convert a raw column name to snake_case."""
    name = str(name).strip()
    name = re.sub(r"[^0-9A-Za-z]+", "_", name)
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    return name.strip("_").lower()


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with snake_case column names and common EHS aliases."""
    out = df.copy()
    out.columns = [COMMON_COLUMN_ALIASES.get(snake_case(c), snake_case(c)) for c in out.columns]
    return out


def read_csv_safely(path: Path, **kwargs) -> pd.DataFrame:
    """Read a CSV with a UTF-8 fallback to latin1."""
    kwargs.setdefault("low_memory", False)
    try:
        return pd.read_csv(path, **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1", **kwargs)


def parse_datetime_utc(series: pd.Series) -> pd.Series:
    """Parse a datetime series as timezone-aware UTC datetimes."""
    return pd.to_datetime(series, errors="coerce", utc=True)


def parse_bool(series: pd.Series) -> pd.Series:
    """Parse common boolean representations into nullable boolean dtype."""
    if series.dtype == bool:
        return series.astype("boolean")
    s = series.astype("string").str.strip().str.lower()
    out = pd.Series(pd.NA, index=series.index, dtype="boolean")
    out[s.isin(TRUE_VALUES)] = True
    out[s.isin(FALSE_VALUES)] = False
    return out


def ensure_bool_columns(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    """Parse listed boolean columns when present."""
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = parse_bool(out[col])
    return out


def clean_text_value(value: object) -> str:
    """Clean a single text value by removing excess whitespace."""
    if pd.isna(value):
        return ""
    text = str(value)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if text.lower() in {"nan", "none", "null", "nat"}:
        return ""
    return text


def clean_text_series(series: pd.Series) -> pd.Series:
    """Clean a pandas text series using vectorized string operations."""
    s = series.fillna("").astype("string")
    s = s.str.replace(r"<[^>]+>", " ", regex=True)
    s = s.str.replace(r"[\r\n\t]+", " ", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    s = s.mask(s.str.lower().isin({"nan", "none", "null", "nat"}), "")
    return s.astype("string")


def combine_text_fields(df: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    """Combine and clean text fields into one ML-ready text column."""
    existing = [c for c in columns if c in df.columns]
    if not existing:
        return pd.Series("", index=df.index, dtype="string")
    cleaned = [clean_text_series(df[c]) for c in existing]
    combined = cleaned[0]
    for s in cleaned[1:]:
        combined = combined.str.cat(s, sep=" ")
    combined = combined.fillna("").map(clean_text_value).astype("string")
    return combined


def add_text_quality_features(df: pd.DataFrame, text_col: str, prefix: Optional[str] = None) -> pd.DataFrame:
    """Add character and word-count features for a text column."""
    out = df.copy(deep=False)
    prefix = prefix or text_col
    text = out[text_col].fillna("").astype("string")
    out[f"{prefix}_char_count"] = text.str.len().astype("Int64")
    out[f"{prefix}_word_count"] = text.str.split().map(lambda x: len(x) if isinstance(x, list) else 0).astype("Int64")
    out[f"{prefix}_has_text"] = out[f"{prefix}_char_count"].fillna(0).gt(0)
    return out


def add_month_week_columns(df: pd.DataFrame, date_col: str, prefix: str) -> pd.DataFrame:
    """Add normalized month and week date columns from a datetime column."""
    out = df.copy(deep=False)
    if date_col not in out.columns:
        return out
    dt = out[date_col]
    # Avoid overwriting the source datetime when date_col is already named like "incident_date".
    out[f"{prefix}_date_only"] = dt.dt.date
    out[f"{prefix}_year"] = dt.dt.year.astype("Int64")
    out[f"{prefix}_quarter"] = dt.dt.quarter.astype("Int64")
    # Use timezone-naive month/week starts for simple CSV compatibility.
    month_naive = dt.dt.tz_convert(None).dt.to_period("M").dt.to_timestamp()
    week_naive = dt.dt.tz_convert(None).dt.to_period("W-MON").dt.start_time
    out[f"{prefix}_month"] = month_naive
    out[f"{prefix}_week"] = week_naive
    return out


def safe_divide(num: pd.Series, den: pd.Series) -> pd.Series:
    """Divide safely, returning NaN where denominator is zero."""
    den = den.replace({0: np.nan})
    return num / den


def write_dataframe(df: pd.DataFrame, path_without_ext: Path, formats: Iterable[str] = ("csv",)) -> None:
    """Write a dataframe to requested formats. Parquet is skipped if dependencies fail."""
    path_without_ext.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fmt = fmt.lower().strip()
        if fmt == "csv":
            df.to_csv(path_without_ext.with_suffix(".csv"), index=False)
        elif fmt == "parquet":
            try:
                df.to_parquet(path_without_ext.with_suffix(".parquet"), index=False)
            except Exception as exc:  # pragma: no cover
                print(f"WARNING: could not write parquet for {path_without_ext.name}: {exc}")
        else:
            raise ValueError(f"Unsupported output format: {fmt}")


def value_counts_table(series: pd.Series, name: str, top_n: int | None = None) -> pd.DataFrame:
    """Return a value-count table with percentages."""
    counts = series.fillna("Unknown").value_counts(dropna=False)
    if top_n:
        counts = counts.head(top_n)
    out = counts.rename_axis(name).reset_index(name="count")
    total = counts.sum()
    out["percent"] = np.where(total > 0, out["count"] / total, np.nan)
    return out
