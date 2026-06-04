#!/usr/bin/env python
"""Build the unified safety text event file from raw Velocity/Accelerate exports.

This script is included so the Safety Retrieval Agent can start from raw source
CSV files when safety_text_event.csv.gz is not already available.

Configuration lives in src/safety_retrieval_agent/config.py. Run without args:

    python scripts/00_build_unified_text_events.py

Expected raw files by default under settings.raw_data_dir:
    INCIDENT_VIEW.csv
    INCIDENTINJURY_VIEW.csv
    AUDIT_VIEW.csv
    TASK_VIEW.csv
    LOCATION_VIEW.csv
    LISTITEM_VIEW.csv

Main output:
    outputs/safety retrieval agent/data/safety_text_event.csv.gz
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from _bootstrap import PROJECT_ROOT  # noqa: F401
from safety_retrieval_agent.config import get_settings
from safety_retrieval_agent.utils import clean_text_value, ensure_dir, save_json

TRUE_VALUES = {"true", "1", "yes", "y", "t"}


def log(message: str, start_time: float | None = None) -> None:
    if start_time is None:
        print(f"[00-unified] {message}", flush=True)
    else:
        print(f"[00-unified | {time.time() - start_time:,.1f}s] {message}", flush=True)


def read_csv(path: Path, nrows: int | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required raw source file not found: {path}")
    try:
        return pd.read_csv(path, low_memory=False, nrows=nrows)
    except UnicodeDecodeError:
        return pd.read_csv(path, low_memory=False, nrows=nrows, encoding="latin1")


def optional_read_csv(path: Path, nrows: int | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return read_csv(path, nrows=nrows)


def truthy_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False).astype(bool)
    return series.astype("string").fillna("").str.strip().str.lower().isin(TRUE_VALUES)


def parse_datetime_series(series: pd.Series | object) -> pd.Series:
    if isinstance(series, pd.Series):
        return pd.to_datetime(series, errors="coerce", utc=False, format="mixed")
    return pd.to_datetime(pd.Series(series), errors="coerce", utc=False, format="mixed")


def coalesce_datetime(df: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    out = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    for col in columns:
        if col in df.columns:
            parsed = parse_datetime_series(df[col])
            out = out.where(out.notna(), parsed)
    return out


def coalesce_string(df: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    out = pd.Series("", index=df.index, dtype="object")
    for col in columns:
        if col not in df.columns:
            continue
        values = df[col].fillna("").astype(str).map(clean_text_value)
        out = out.where(out.astype(str).str.strip().ne(""), values)
    return out.fillna("").astype(str)


def make_text_block(df: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    pieces: list[pd.Series] = []
    for col in columns:
        if col not in df.columns:
            continue
        values = df[col].fillna("").astype(str).map(clean_text_value)
        values = values.where(values.str.strip().eq(""), col.lower() + ": " + values)
        pieces.append(values)
    if not pieces:
        return pd.Series("", index=df.index, dtype="object")
    out = pieces[0]
    for piece in pieces[1:]:
        out = out.str.cat(piece, sep=" | ")
    return out.map(clean_text_value)


def _first_existing(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    lookup = {c.upper(): c for c in columns}
    for cand in candidates:
        if cand.upper() in lookup:
            return lookup[cand.upper()]
    return None


def load_listitem_lookup(path: Path) -> dict[str, str]:
    """Load ID -> display label mapping from LISTITEM_VIEW.csv.

    The exact Velocity export schema can vary. This function tries common ID and
    label column names and falls back gracefully if the file is unavailable.
    """
    if not path.exists():
        return {}
    df = read_csv(path)
    id_col = _first_existing(df.columns, ["LISTITEMID", "LISTITEM_ID", "ID", "ITEMID"])
    label_col = _first_existing(
        df.columns,
        ["ITEM", "NAME", "LISTITEM", "LISTITEMNAME", "SHORTNAME", "DESCRIPTION", "DISPLAYNAME", "LABEL", "VALUE"],
    )
    if id_col is None:
        return {}
    if label_col is None:
        # Use the first non-ID text-looking column as a last resort.
        for col in df.columns:
            if col != id_col and df[col].dtype == object:
                label_col = col
                break
    if label_col is None:
        return {}
    labels = df[[id_col, label_col]].dropna(subset=[id_col]).copy()
    labels[id_col] = labels[id_col].astype(str)
    labels[label_col] = labels[label_col].fillna("").astype(str).map(clean_text_value)
    return dict(zip(labels[id_col], labels[label_col]))


def add_listitem_fields(df: pd.DataFrame, id_col: str, prefix: str, lookup: dict[str, str]) -> pd.DataFrame:
    out = df.copy()
    if id_col not in out.columns:
        out[f"{prefix}_item"] = ""
        return out
    out[f"{prefix}_item"] = out[id_col].astype("string").map(lambda x: lookup.get(str(x), "") if pd.notna(x) else "")
    missing = out[f"{prefix}_item"].fillna("").astype(str).str.strip().eq("")
    out.loc[missing, f"{prefix}_item"] = out.loc[missing, id_col].fillna("").astype(str)
    return out


def normalize_source_type(value: object) -> str:
    text = clean_text_value(value).lower()
    if "near" in text and "miss" in text:
        return "near_miss"
    if "hazard" in text:
        return "hazard_identification"
    if "incident" in text:
        return "incident"
    if not text:
        return "incident"
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text or "incident"


def build_location_hierarchy(path: Path, lookup: dict[str, str]) -> pd.DataFrame:
    """Create a lightweight location dimension with site/department/path fields."""
    if not path.exists():
        return pd.DataFrame(columns=["LOCATIONID", "location_name", "location_path_clean", "site", "department"])
    loc = read_csv(path)
    id_col = _first_existing(loc.columns, ["LOCATIONID", "LOCATION_ID", "ID"])
    if id_col is None:
        return pd.DataFrame(columns=["LOCATIONID", "location_name", "location_path_clean", "site", "department"])
    name_col = _first_existing(
        loc.columns,
        ["LOCATIONNAME", "LOCATION", "NAME", "SHORTNAME", "DISPLAYNAME", "DESCRIPTION", "LOCATIONDESCRIPTION"],
    )
    parent_col = _first_existing(loc.columns, ["PARENTLOCATIONID", "PARENT_LOCATION_ID", "PARENTID", "PARENT_LOCATIONID"])

    loc = loc.copy()
    loc["LOCATIONID"] = loc[id_col]
    if name_col is not None:
        loc["location_name"] = loc[name_col].fillna("").astype(str).map(clean_text_value)
    else:
        loc["location_name"] = loc["LOCATIONID"].fillna("").astype(str)

    if parent_col is not None:
        parent_map = dict(zip(loc["LOCATIONID"].astype(str), loc[parent_col].fillna("").astype(str)))
    else:
        parent_map = {}
    name_map = dict(zip(loc["LOCATIONID"].astype(str), loc["location_name"].astype(str)))

    def path_for(location_id: object) -> list[str]:
        cur = "" if pd.isna(location_id) else str(location_id)
        parts: list[str] = []
        seen: set[str] = set()
        while cur and cur not in seen:
            seen.add(cur)
            name = clean_text_value(name_map.get(cur, cur))
            if name:
                parts.append(name)
            parent = clean_text_value(parent_map.get(cur, ""))
            if not parent or parent == cur or parent.lower() in {"nan", "none", "null"}:
                break
            cur = parent
        return list(reversed(parts)) if parts else []

    paths = [path_for(x) for x in loc["LOCATIONID"]]
    loc["location_path_clean"] = [" / ".join(p) for p in paths]
    loc["site"] = [p[0] if len(p) >= 1 else "" for p in paths]
    loc["department"] = [p[1] if len(p) >= 2 else "" for p in paths]
    for i in range(1, 7):
        loc[f"location_level_{i}"] = [p[i - 1] if len(p) >= i else "" for p in paths]
    cols = [
        "LOCATIONID", "location_name", "location_path_clean", "site", "department",
        "location_level_1", "location_level_2", "location_level_3", "location_level_4", "location_level_5", "location_level_6",
    ]
    return loc[[c for c in cols if c in loc.columns]].drop_duplicates(subset=["LOCATIONID"])


def attach_location(df: pd.DataFrame, location_dim: pd.DataFrame, id_col: str = "LOCATIONID") -> pd.DataFrame:
    out = df.copy()
    if id_col not in out.columns:
        out["LOCATIONID"] = np.nan
        id_col = "LOCATIONID"
    if location_dim.empty:
        out["site"] = ""
        out["department"] = ""
        out["location_path_clean"] = ""
        return out
    loc_cols = [
        "LOCATIONID", "location_name", "location_path_clean", "site", "department",
        "location_level_1", "location_level_2", "location_level_3", "location_level_4", "location_level_5", "location_level_6",
    ]
    loc_cols = [c for c in loc_cols if c in location_dim.columns]
    return out.merge(location_dim[loc_cols], left_on=id_col, right_on="LOCATIONID", how="left", suffixes=("", "_loc"))


def build_injury_flags(injury_path: Path) -> pd.DataFrame:
    inj = optional_read_csv(injury_path)
    if inj.empty or "INCIDENTID" not in inj.columns:
        return pd.DataFrame(columns=["INCIDENTID", "any_injury", "severe_actual", "injury_record_count"])
    bool_cols = ["FATALITY", "LOSTTIME", "RESTRICTEDTIME", "INPATIENT", "EMERGENCYROOM"]
    for col in bool_cols:
        if col not in inj.columns:
            inj[col] = False
        inj[col] = truthy_series(inj[col])
    if "INJURYID" not in inj.columns:
        inj["INJURYID"] = np.arange(len(inj))
    inj["any_injury"] = True
    inj["severe_actual"] = inj[["FATALITY", "LOSTTIME", "RESTRICTEDTIME", "INPATIENT"]].any(axis=1)
    return inj.groupby("INCIDENTID", dropna=False).agg(
        any_injury=("any_injury", "max"),
        severe_actual=("severe_actual", "max"),
        injury_record_count=("INJURYID", "count"),
        fatality=("FATALITY", "max"),
        losttime=("LOSTTIME", "max"),
        restrictedtime=("RESTRICTEDTIME", "max"),
        inpatient=("INPATIENT", "max"),
        emergencyroom=("EMERGENCYROOM", "max"),
    ).reset_index()


def _series(df: pd.DataFrame, col: str, default: object = "") -> pd.Series:
    if col in df.columns:
        return df[col]
    return pd.Series(default, index=df.index)


def build_incident_events(data_dir: Path, location_dim: pd.DataFrame, listitem_lookup: dict[str, str], sample_size: int | None) -> pd.DataFrame:
    df = read_csv(data_dir / "INCIDENT_VIEW.csv", nrows=sample_size)
    df = add_listitem_fields(df, "INCIDENTCATEGORYID", "incident_category", listitem_lookup)
    df = add_listitem_fields(df, "INCIDENTSTATUSID", "incident_status", listitem_lookup)
    injury = build_injury_flags(data_dir / "INCIDENTINJURY_VIEW.csv")
    if not injury.empty:
        df = df.merge(injury, on="INCIDENTID", how="left")
    for col in ["any_injury", "severe_actual", "fatality", "losttime", "restrictedtime", "inpatient", "emergencyroom"]:
        if col not in df.columns:
            df[col] = False
        df[col] = df[col].fillna(False).astype(bool)
    if "injury_record_count" not in df.columns:
        df["injury_record_count"] = 0
    df["injury_record_count"] = pd.to_numeric(df["injury_record_count"], errors="coerce").fillna(0).astype(int)
    df = attach_location(df, location_dim, "LOCATIONID")

    event_date = coalesce_datetime(df, ["INCIDENTDATE", "REPORTDATE", "INVESTIGATIONSTARTDATE"])
    source_subtype = _series(df, "incident_category_item", "Incident").map(clean_text_value)
    source_type = source_subtype.map(normalize_source_type)
    status = _series(df, "incident_status_item", "").map(clean_text_value)

    text_fields = [
        "TITLE", "DESCRIPTION", "ACTIVITYDURINGINCIDENT", "IMMEDIATEACTION",
        "IMMEDIATECAUSES", "CAUSALFACTORS", "BESTPRACTICES", "RISKACTION",
        "RISKCONDITION", "EQUIPMENT", "VEHICLE", "OTHERPROCESS", "OTHERACTIVITY",
        "OFFPREMISESLOCATION", "OTHERLOCATION",
    ]
    clean_text = make_text_block(df, text_fields)
    return pd.DataFrame({
        "event_id": "incident_" + _series(df, "INCIDENTID", "").astype(str),
        "source_type": source_type,
        "source_subtype": source_subtype,
        "source_id": _series(df, "INCIDENTID", pd.NA),
        "event_date": event_date,
        "location_id": _series(df, "LOCATIONID", pd.NA),
        "site": _series(df, "site", ""),
        "department": _series(df, "department", ""),
        "location_path": _series(df, "location_path_clean", ""),
        "title": coalesce_string(df, ["TITLE", "INCIDENTNUMBER"]),
        "description": coalesce_string(df, ["DESCRIPTION"]),
        "clean_text": clean_text,
        "status": status,
        "category": source_subtype,
        "audit_type": "",
        "is_open_task": False,
        "is_overdue_task": False,
        "due_date": pd.NaT,
        "completion_date": pd.NaT,
        "any_injury": _series(df, "any_injury", False),
        "severe_actual": _series(df, "severe_actual", False),
        "fatality": _series(df, "fatality", False),
        "losttime": _series(df, "losttime", False),
        "restrictedtime": _series(df, "restrictedtime", False),
        "inpatient": _series(df, "inpatient", False),
        "emergencyroom": _series(df, "emergencyroom", False),
        "injury_record_count": _series(df, "injury_record_count", 0),
        "raw_status_id": _series(df, "INCIDENTSTATUSID", pd.NA),
        "raw_category_id": _series(df, "INCIDENTCATEGORYID", pd.NA),
    })


def build_audit_events(data_dir: Path, location_dim: pd.DataFrame, listitem_lookup: dict[str, str], sample_size: int | None) -> pd.DataFrame:
    df = read_csv(data_dir / "AUDIT_VIEW.csv", nrows=sample_size)
    df = add_listitem_fields(df, "AUDITCATEGORYID", "audit_category", listitem_lookup)
    df = add_listitem_fields(df, "AUDITTYPEID", "audit_type", listitem_lookup)
    df = add_listitem_fields(df, "AUDITSTATUSID", "audit_status", listitem_lookup)
    loc_col = "SCHEDULEDLOCATIONID" if "SCHEDULEDLOCATIONID" in df.columns else "LOCATIONID"
    df = attach_location(df, location_dim, loc_col)

    event_date = coalesce_datetime(df, ["ACTUALSTART", "ACTUALEND", "SCHEDULEDSTART", "SCHEDULEDEND"])
    status = _series(df, "audit_status_item", "").map(clean_text_value)
    category = _series(df, "audit_category_item", "Audit").map(clean_text_value)
    audit_type = _series(df, "audit_type_item", "").map(clean_text_value)
    source_subtype = np.where(audit_type.astype(str).str.strip().ne(""), category + " - " + audit_type, category)
    text_fields = ["TITLE", "DESCRIPTION", "COMMENTS", "ASSOCIATEDPARTIES", "SHORTNAME", "OTHERLOCATIONNAME"]
    clean_text = make_text_block(df, text_fields)
    return pd.DataFrame({
        "event_id": "audit_" + _series(df, "AUDITID", "").astype(str),
        "source_type": "audit",
        "source_subtype": source_subtype,
        "source_id": _series(df, "AUDITID", pd.NA),
        "event_date": event_date,
        "location_id": df["SCHEDULEDLOCATIONID"] if "SCHEDULEDLOCATIONID" in df.columns else _series(df, "LOCATIONID", pd.NA),
        "site": _series(df, "site", ""),
        "department": _series(df, "department", ""),
        "location_path": _series(df, "location_path_clean", ""),
        "title": coalesce_string(df, ["TITLE", "SHORTNAME", "AUDITNUMBER"]),
        "description": coalesce_string(df, ["DESCRIPTION", "COMMENTS"]),
        "clean_text": clean_text,
        "status": status,
        "category": category,
        "audit_type": audit_type,
        "is_open_task": False,
        "is_overdue_task": False,
        "due_date": pd.NaT,
        "completion_date": pd.NaT,
        "any_injury": False,
        "severe_actual": False,
        "fatality": False,
        "losttime": False,
        "restrictedtime": False,
        "inpatient": False,
        "emergencyroom": False,
        "injury_record_count": 0,
        "raw_status_id": _series(df, "AUDITSTATUSID", pd.NA),
        "raw_category_id": _series(df, "AUDITCATEGORYID", pd.NA),
        "raw_type_id": _series(df, "AUDITTYPEID", pd.NA),
    })


def build_task_events(data_dir: Path, location_dim: pd.DataFrame, listitem_lookup: dict[str, str], sample_size: int | None) -> pd.DataFrame:
    df = read_csv(data_dir / "TASK_VIEW.csv", nrows=sample_size)
    df = add_listitem_fields(df, "TASKCATEGORYID", "task_category", listitem_lookup)
    df = add_listitem_fields(df, "TASKSTATUSID", "task_status", listitem_lookup)
    df = add_listitem_fields(df, "SOURCETYPEID", "source_module", listitem_lookup)
    df = attach_location(df, location_dim, "LOCATIONID")

    event_date = coalesce_datetime(df, ["ASSIGNEDDATE", "SOURCEDATE", "STARTDATE", "DUEDATE", "COMPLETIONDATE"])
    due_date = coalesce_datetime(df, ["REVISEDDUEDATE", "DUEDATE"])
    completion_date = coalesce_datetime(df, ["COMPLETIONDATE", "MARKEDCOMPLETEDATE"])
    status = _series(df, "task_status_item", "").map(clean_text_value)
    status_lower = status.str.lower()
    is_open_task = ~status_lower.isin(["closed", "deleted", "complete", "completed"]) & completion_date.isna()
    today = pd.Timestamp.utcnow().tz_localize(None).normalize()
    is_overdue_task = is_open_task & due_date.notna() & (due_date < today)
    category = _series(df, "task_category_item", "Task").map(clean_text_value)
    module = _series(df, "source_module_item", "").map(clean_text_value)
    source_subtype = np.where(module.astype(str).str.strip().ne(""), category + " - " + module, category)
    text_fields = ["TASK", "DESCRIPTION", "BESTPRACTICES", "VERIFICATIONREASON", "SOURCE", "EQUIPMENT", "OTHERLOCATIONNAME"]
    clean_text = make_text_block(df, text_fields)
    return pd.DataFrame({
        "event_id": "task_" + _series(df, "TASKID", "").astype(str),
        "source_type": "task",
        "source_subtype": source_subtype,
        "source_id": _series(df, "TASKID", pd.NA),
        "event_date": event_date,
        "location_id": _series(df, "LOCATIONID", pd.NA),
        "site": _series(df, "site", ""),
        "department": _series(df, "department", ""),
        "location_path": _series(df, "location_path_clean", ""),
        "title": coalesce_string(df, ["TASK", "TASKNUMBER"]),
        "description": coalesce_string(df, ["DESCRIPTION"]),
        "clean_text": clean_text,
        "status": status,
        "category": category,
        "audit_type": "",
        "task_source_module": module,
        "is_open_task": is_open_task,
        "is_overdue_task": is_overdue_task,
        "due_date": due_date,
        "completion_date": completion_date,
        "any_injury": False,
        "severe_actual": False,
        "fatality": False,
        "losttime": False,
        "restrictedtime": False,
        "inpatient": False,
        "emergencyroom": False,
        "injury_record_count": 0,
        "raw_status_id": _series(df, "TASKSTATUSID", pd.NA),
        "raw_category_id": _series(df, "TASKCATEGORYID", pd.NA),
        "raw_source_type_id": _series(df, "SOURCETYPEID", pd.NA),
    })


def main() -> dict:
    settings = get_settings()
    start = time.time()
    ensure_dir(settings.output_dir)
    ensure_dir(settings.unified_event_path().parent)
    log("Starting unified safety text event build.", start)
    log(f"Raw data directory: {settings.raw_data_dir}", start)

    required = [settings.raw_incident_path(), settings.raw_audit_path(), settings.raw_task_path()]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required raw source files for unified build: " + json.dumps(missing, indent=2)
        )

    listitem_lookup = load_listitem_lookup(settings.raw_listitem_path())
    location_dim = build_location_hierarchy(settings.raw_location_path(), listitem_lookup)
    location_dim.to_csv(settings.location_hierarchy_path(), index=False)
    log(f"Built location hierarchy with {len(location_dim):,} rows.", start)

    frames = [
        build_incident_events(settings.raw_data_dir, location_dim, listitem_lookup, settings.unified_sample_size),
        build_audit_events(settings.raw_data_dir, location_dim, listitem_lookup, settings.unified_sample_size),
        build_task_events(settings.raw_data_dir, location_dim, listitem_lookup, settings.unified_sample_size),
    ]
    events = pd.concat(frames, ignore_index=True, sort=False)
    log(f"Combined source events: {len(events):,} rows before filtering.", start)

    for col in ["event_date", "due_date", "completion_date"]:
        if col in events.columns:
            events[col] = parse_datetime_series(events[col])
    events["clean_text"] = events["clean_text"].fillna("").astype(str).map(clean_text_value)
    events["text_length"] = events["clean_text"].str.len()
    events["has_text"] = events["text_length"] > 0

    before_empty_filter = int(len(events))
    if settings.drop_empty_unified_text:
        events = events[events["has_text"]].copy()
        log(f"Dropped empty-text rows: {before_empty_filter - len(events):,}.", start)

    before_dedup = int(len(events))
    events = events.drop_duplicates(subset=["event_id"]).reset_index(drop=True)
    log(f"Dropped duplicate event IDs: {before_dedup - len(events):,}.", start)

    output_path = settings.unified_event_path()
    events.to_csv(output_path, index=False, compression="gzip")
    summary = {
        "output_path": str(output_path),
        "raw_data_dir": str(settings.raw_data_dir),
        "row_count": int(len(events)),
        "source_type_counts": {str(k): int(v) for k, v in events["source_type"].value_counts(dropna=False).to_dict().items()},
        "date_min": str(events["event_date"].min()) if "event_date" in events.columns else None,
        "date_max": str(events["event_date"].max()) if "event_date" in events.columns else None,
        "empty_text_count": int((~events["has_text"]).sum()),
        "location_hierarchy_path": str(settings.location_hierarchy_path()),
        "elapsed_seconds": round(time.time() - start, 2),
    }
    save_json(summary, settings.output_dir / "00_build_unified_text_events_summary.json")
    log(f"Saved unified event table to {output_path}", start)
    log("Unified build complete.", start)
    print(summary)
    return summary


if __name__ == "__main__":
    main()
