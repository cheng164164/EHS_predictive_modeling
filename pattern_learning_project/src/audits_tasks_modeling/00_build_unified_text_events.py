"""Step 00: build the unified safety text event table.

This script combines Incident, Audit, and Task exports into one text-event table
with the exact schema expected by downstream pattern-learning scripts.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for path in [SCRIPT_DIR, SRC_ROOT, PROJECT_ROOT]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import pandas as pd

try:
    import config as cfg
except Exception:  # pragma: no cover - lets the script import in isolated tests.
    cfg = object()

from utils import (
    OUTPUT_COLUMNS,
    add_listitem_fields,
    build_location_hierarchy,
    clean_text_value,
    coalesce_datetime,
    coalesce_string,
    ensure_dir,
    ensure_unified_event_schema,
    load_listitem_lookup,
    make_text_block,
    normalize_source_type,
    parse_datetime_series,
    read_csv,
    save_csv,
    save_json,
    truthy_series,
)


def cfg_value(name: str, default):
    return getattr(cfg, name, default)


def log(message: str, start_time: float | None = None) -> None:
    if start_time is None:
        print(f"[Step 00] {message}", flush=True)
    else:
        print(f"[Step 00 | {time.time() - start_time:,.1f}s] {message}", flush=True)


def detect_english_texts(texts: pd.Series) -> pd.DataFrame:
    """Detect English text when optional English-only filtering is enabled."""
    library = str(cfg_value("LANGUAGE_DETECTION_LIBRARY", "langdetect")).lower()
    if library != "langdetect":
        raise ValueError(
            f"Unsupported LANGUAGE_DETECTION_LIBRARY={library!r}. Currently supported: 'langdetect'."
        )
    try:
        from langdetect import DetectorFactory, LangDetectException, detect_langs
    except ImportError as exc:
        raise ImportError(
            "ENGLISH_ONLY_TEXT_FILTER is enabled, but 'langdetect' is not installed. "
            "Install it with: pip install langdetect"
        ) from exc

    DetectorFactory.seed = int(cfg_value("LANGUAGE_DETECTION_RANDOM_STATE", 42))
    min_prob = float(cfg_value("LANGUAGE_DETECTION_MIN_PROB", 0.80))
    min_chars = int(cfg_value("LANGUAGE_DETECTION_MIN_TEXT_CHARS", 20))
    max_chars = int(cfg_value("LANGUAGE_DETECTION_MAX_TEXT_CHARS", 1000))
    keep_short = bool(cfg_value("ENGLISH_ONLY_KEEP_SHORT_TEXT", True))
    keep_unknown = bool(cfg_value("ENGLISH_ONLY_KEEP_UNKNOWN_LANGUAGE", False))
    progress_every = int(cfg_value("LANGUAGE_DETECTION_PROGRESS_EVERY", 50000))

    detected_language: list[str] = []
    detected_score: list[float] = []
    language_status: list[str] = []
    is_english: list[bool] = []

    total = len(texts)
    start = time.time()
    for i, value in enumerate(texts.fillna("").astype(str), start=1):
        text = clean_text_value(value)
        if len(text) < min_chars:
            detected_language.append("short_text")
            detected_score.append(np.nan)
            language_status.append("short_text")
            is_english.append(bool(keep_short))
        else:
            try:
                candidates = detect_langs(text[:max_chars])
                top = candidates[0] if candidates else None
                lang = getattr(top, "lang", "unknown") if top is not None else "unknown"
                prob = float(getattr(top, "prob", np.nan)) if top is not None else np.nan
                ok = lang == "en" and prob >= min_prob
                detected_language.append(lang)
                detected_score.append(prob)
                language_status.append("english" if ok else "non_english")
                is_english.append(ok)
            except LangDetectException:
                detected_language.append("unknown")
                detected_score.append(np.nan)
                language_status.append("unknown")
                is_english.append(bool(keep_unknown))
        if progress_every > 0 and i % progress_every == 0:
            print(
                f"[Step 00 | {time.time() - start:,.1f}s] Language detection processed {i:,}/{total:,} rows...",
                flush=True,
            )

    return pd.DataFrame(
        {
            "detected_language": detected_language,
            "detected_language_score": detected_score,
            "language_detection_status": language_status,
            "is_english_text": is_english,
        },
        index=texts.index,
    )


def attach_location(df: pd.DataFrame, location_dim: pd.DataFrame, id_col: str = "LOCATIONID") -> pd.DataFrame:
    """Attach hierarchy fields by location id without changing source row count."""
    out = df.copy()
    if id_col not in out.columns:
        out[id_col] = np.nan
    loc_cols = [
        "LOCATIONID",
        "location_name",
        "location_path",
        "location_path_clean",
        "site",
        "department",
        "location_level_1",
        "location_level_2",
        "location_level_3",
        "location_level_4",
        "location_level_5",
        "location_level_6",
    ]
    loc_cols = [c for c in loc_cols if c in location_dim.columns]
    return out.merge(location_dim[loc_cols], left_on=id_col, right_on="LOCATIONID", how="left", suffixes=("", "_loc"))


def build_injury_flags(injury_path: Path) -> pd.DataFrame:
    inj = read_csv(injury_path)
    if "INCIDENTID" not in inj.columns:
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


def build_incident_events(data_dir: Path, location_dim: pd.DataFrame, listitem_lookup: dict, sample_size: int | None) -> pd.DataFrame:
    df = read_csv(data_dir / "INCIDENT_VIEW.csv", nrows=sample_size)
    df = add_listitem_fields(df, "INCIDENTCATEGORYID", "incident_category", listitem_lookup)
    df = add_listitem_fields(df, "INCIDENTSTATUSID", "incident_status", listitem_lookup)
    df = df.merge(build_injury_flags(data_dir / "INCIDENTINJURY_VIEW.csv"), on="INCIDENTID", how="left")
    for col in ["any_injury", "severe_actual", "fatality", "losttime", "restrictedtime", "inpatient", "emergencyroom"]:
        df[col] = truthy_series(df[col]) if col in df.columns else False
    df["injury_record_count"] = pd.to_numeric(df.get("injury_record_count", 0), errors="coerce").fillna(0).astype(int)
    df = attach_location(df, location_dim, "LOCATIONID")

    event_date = coalesce_datetime(df, ["INCIDENTDATE", "REPORTDATE", "INVESTIGATIONSTARTDATE"])
    source_subtype = df.get("incident_category_item", pd.Series("Incident", index=df.index)).map(clean_text_value)
    source_type = source_subtype.map(normalize_source_type)
    status = df.get("incident_status_item", pd.Series("", index=df.index)).map(clean_text_value)
    text_fields = [
        "TITLE", "DESCRIPTION", "ACTIVITYDURINGINCIDENT", "IMMEDIATEACTION", "IMMEDIATECAUSES",
        "CAUSALFACTORS", "BESTPRACTICES", "RISKACTION", "RISKCONDITION", "EQUIPMENT", "VEHICLE",
        "OTHERPROCESS", "OTHERACTIVITY", "OFFPREMISESLOCATION", "OTHERLOCATION",
    ]
    return pd.DataFrame({
        "event_id": "incident_" + df["INCIDENTID"].astype(str),
        "source_type": source_type,
        "source_subtype": source_subtype,
        "source_id": df["INCIDENTID"],
        "event_date": event_date,
        "location_id": df.get("LOCATIONID"),
        "site": df.get("site", ""),
        "department": df.get("department", ""),
        "location_path": df.get("location_path_clean", df.get("location_path", "")),
        "title": coalesce_string(df, ["TITLE", "INCIDENTNUMBER"]),
        "description": coalesce_string(df, ["DESCRIPTION"]),
        "clean_text": make_text_block(df, text_fields),
        "status": status,
        "category": source_subtype,
        "is_open_task": False,
        "is_overdue_task": False,
        "due_date": pd.NaT,
        "completion_date": pd.NaT,
        "any_injury": df.get("any_injury", False),
        "severe_actual": df.get("severe_actual", False),
        "fatality": df.get("fatality", False),
        "losttime": df.get("losttime", False),
        "restrictedtime": df.get("restrictedtime", False),
        "inpatient": df.get("inpatient", False),
        "emergencyroom": df.get("emergencyroom", False),
        "injury_record_count": df.get("injury_record_count", 0),
        "raw_status_id": df.get("INCIDENTSTATUSID"),
        "raw_category_id": df.get("INCIDENTCATEGORYID"),
    })


def build_audit_events(data_dir: Path, location_dim: pd.DataFrame, listitem_lookup: dict, sample_size: int | None) -> pd.DataFrame:
    df = read_csv(data_dir / "AUDIT_VIEW.csv", nrows=sample_size)
    df = add_listitem_fields(df, "AUDITCATEGORYID", "audit_category", listitem_lookup)
    df = add_listitem_fields(df, "AUDITTYPEID", "audit_type", listitem_lookup)
    df = add_listitem_fields(df, "AUDITSTATUSID", "audit_status", listitem_lookup)
    loc_col = "SCHEDULEDLOCATIONID" if "SCHEDULEDLOCATIONID" in df.columns else "LOCATIONID"
    df = attach_location(df, location_dim, loc_col)

    event_date = coalesce_datetime(df, ["ACTUALSTART", "ACTUALEND", "SCHEDULEDSTART", "SCHEDULEDEND"])
    status = df.get("audit_status_item", pd.Series("", index=df.index)).map(clean_text_value)
    category = df.get("audit_category_item", pd.Series("Audit", index=df.index)).map(clean_text_value)
    audit_type = df.get("audit_type_item", pd.Series("", index=df.index)).map(clean_text_value)
    source_subtype = np.where(audit_type.ne(""), category + " - " + audit_type, category)
    text_fields = ["TITLE", "DESCRIPTION", "COMMENTS", "ASSOCIATEDPARTIES", "SHORTNAME", "OTHERLOCATIONNAME"]
    return pd.DataFrame({
        "event_id": "audit_" + df["AUDITID"].astype(str),
        "source_type": "audit",
        "source_subtype": source_subtype,
        "source_id": df["AUDITID"],
        "event_date": event_date,
        "location_id": df.get(loc_col),
        "site": df.get("site", ""),
        "department": df.get("department", ""),
        "location_path": df.get("location_path_clean", df.get("location_path", "")),
        "title": coalesce_string(df, ["TITLE", "SHORTNAME", "AUDITNUMBER"]),
        "description": coalesce_string(df, ["DESCRIPTION", "COMMENTS"]),
        "clean_text": make_text_block(df, text_fields),
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
        "raw_status_id": df.get("AUDITSTATUSID"),
        "raw_category_id": df.get("AUDITCATEGORYID"),
        "raw_type_id": df.get("AUDITTYPEID"),
    })


def build_task_events(data_dir: Path, location_dim: pd.DataFrame, listitem_lookup: dict, sample_size: int | None) -> pd.DataFrame:
    df = read_csv(data_dir / "TASK_VIEW.csv", nrows=sample_size)
    df = add_listitem_fields(df, "TASKCATEGORYID", "task_category", listitem_lookup)
    df = add_listitem_fields(df, "TASKSTATUSID", "task_status", listitem_lookup)
    df = add_listitem_fields(df, "SOURCETYPEID", "source_module", listitem_lookup)
    df = attach_location(df, location_dim, "LOCATIONID")

    event_date = coalesce_datetime(df, ["ASSIGNEDDATE", "SOURCEDATE", "STARTDATE", "DUEDATE", "COMPLETIONDATE"])
    due_date = coalesce_datetime(df, ["REVISEDDUEDATE", "DUEDATE"])
    completion_date = coalesce_datetime(df, ["COMPLETIONDATE", "MARKEDCOMPLETEDATE"])
    status = df.get("task_status_item", pd.Series("", index=df.index)).map(clean_text_value)
    status_lower = status.str.lower()
    is_open_task = ~status_lower.isin(["closed", "deleted", "complete", "completed"]) & completion_date.isna()
    today = pd.Timestamp.utcnow().tz_localize(None).normalize()
    is_overdue_task = is_open_task & due_date.notna() & (due_date < today)
    category = df.get("task_category_item", pd.Series("Task", index=df.index)).map(clean_text_value)
    module = df.get("source_module_item", pd.Series("", index=df.index)).map(clean_text_value)
    source_subtype = np.where(module.ne(""), category + " - " + module, category)
    text_fields = ["TASK", "DESCRIPTION", "BESTPRACTICES", "VERIFICATIONREASON", "SOURCE", "EQUIPMENT", "OTHERLOCATIONNAME"]
    return pd.DataFrame({
        "event_id": "task_" + df["TASKID"].astype(str),
        "source_type": "task",
        "source_subtype": source_subtype,
        "source_id": df["TASKID"],
        "event_date": event_date,
        "location_id": df.get("LOCATIONID"),
        "site": df.get("site", ""),
        "department": df.get("department", ""),
        "location_path": df.get("location_path_clean", df.get("location_path", "")),
        "title": coalesce_string(df, ["TASK", "TASKNUMBER"]),
        "description": coalesce_string(df, ["DESCRIPTION"]),
        "clean_text": make_text_block(df, text_fields),
        "status": status,
        "category": category,
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
        "raw_status_id": df.get("TASKSTATUSID"),
        "raw_category_id": df.get("TASKCATEGORYID"),
        "raw_source_type_id": df.get("SOURCETYPEID"),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description="Build unified safety text event table from incident, audit, task, injury, location, and list item CSV files.")
    parser.add_argument("--data-dir", default=cfg_value("DATA_DIR", PROJECT_ROOT / "data" / "raw"))
    parser.add_argument("--output-dir", default=cfg_value("STEP_00_DIR", PROJECT_ROOT / "outputs" / "step_00_unified_text_events"))
    parser.add_argument("--sample-size", type=int, default=cfg_value("SAMPLE_SIZE", None), help="Optional rows per main source table for testing.")
    parser.add_argument("--drop-empty-text", action="store_true", default=cfg_value("DROP_EMPTY_TEXT", False), help="Drop records with empty clean_text.")
    parser.add_argument("--keep-empty-text", dest="drop_empty_text", action="store_false", help="Keep records with empty clean_text.")
    parser.add_argument("--english-only", action="store_true", default=cfg_value("ENGLISH_ONLY_TEXT_FILTER", False), help="Keep only records detected as English.")
    parser.add_argument("--keep-all-languages", dest="english_only", action="store_false", help="Disable English-only language filtering.")
    args = parser.parse_args()

    start_time = time.time()
    data_dir = Path(args.data_dir)
    output_dir = ensure_dir(args.output_dir)
    log("Starting unified safety text event build.", start_time)
    log(f"Data directory: {data_dir}", start_time)

    lookup, _ = load_listitem_lookup(data_dir / "LISTITEM_VIEW.csv")
    location_dim = build_location_hierarchy(data_dir / "LOCATION_VIEW.csv", lookup)
    save_csv(location_dim, output_dir / "location_hierarchy.csv")
    log(f"Built location hierarchy with {len(location_dim):,} rows.", start_time)

    frames = [
        build_incident_events(data_dir, location_dim, lookup, args.sample_size),
        build_audit_events(data_dir, location_dim, lookup, args.sample_size),
        build_task_events(data_dir, location_dim, lookup, args.sample_size),
    ]
    events = pd.concat(frames, ignore_index=True, sort=False)
    log(f"Combined source events: {len(events):,} rows before text/language filtering.", start_time)

    events["event_date"] = parse_datetime_series(events["event_date"])
    events["due_date"] = parse_datetime_series(events["due_date"])
    events["completion_date"] = parse_datetime_series(events["completion_date"])
    events["clean_text"] = events["clean_text"].fillna("").map(clean_text_value)
    events["text_length"] = events["clean_text"].str.len()
    events["has_text"] = events["text_length"] > 0

    row_count_before_empty_filter = int(len(events))
    if args.drop_empty_text:
        events = events[events["has_text"]].copy()
        log(f"Dropped empty-text rows: {row_count_before_empty_filter - len(events):,}.", start_time)

    language_summary = {
        "english_only_filter_enabled": bool(args.english_only),
        "language_detection_library": cfg_value("LANGUAGE_DETECTION_LIBRARY", "langdetect"),
    }
    if args.english_only:
        before_language_filter = int(len(events))
        log("Running library-based language detection for English-only filtering...", start_time)
        lang_df = detect_english_texts(events["clean_text"])
        events = pd.concat([events, lang_df], axis=1)
        language_counts = events["detected_language"].value_counts(dropna=False).to_dict()
        language_status_counts = events["language_detection_status"].value_counts(dropna=False).to_dict()
        keep_mask = events["is_english_text"].fillna(False).astype(bool)
        filtered_out = events.loc[~keep_mask].copy()
        if len(filtered_out) > 0:
            save_csv(
                filtered_out[[c for c in [
                    "event_id", "source_type", "source_subtype", "source_id", "detected_language",
                    "detected_language_score", "language_detection_status", "title", "description", "clean_text",
                ] if c in filtered_out.columns]].head(10000),
                output_dir / "non_english_filtered_sample.csv.gz",
            )
        events = events.loc[keep_mask].copy()
        language_summary.update({
            "rows_before_language_filter": before_language_filter,
            "rows_after_language_filter": int(len(events)),
            "rows_removed_by_language_filter": int(before_language_filter - len(events)),
            "pct_removed_by_language_filter": float((before_language_filter - len(events)) / before_language_filter) if before_language_filter else 0.0,
            "detected_language_counts_before_filter": language_counts,
            "language_detection_status_counts_before_filter": language_status_counts,
            "language_detection_min_prob": cfg_value("LANGUAGE_DETECTION_MIN_PROB", None),
            "language_detection_min_text_chars": cfg_value("LANGUAGE_DETECTION_MIN_TEXT_CHARS", None),
            "language_detection_max_text_chars": cfg_value("LANGUAGE_DETECTION_MAX_TEXT_CHARS", None),
            "english_only_keep_short_text": cfg_value("ENGLISH_ONLY_KEEP_SHORT_TEXT", None),
            "english_only_keep_unknown_language": cfg_value("ENGLISH_ONLY_KEEP_UNKNOWN_LANGUAGE", None),
            "non_english_filtered_sample_path": str(output_dir / "non_english_filtered_sample.csv.gz"),
        })
        log(f"English-only filter removed {before_language_filter - len(events):,} rows; kept {len(events):,} rows.", start_time)
    else:
        events["detected_language"] = "not_run"
        events["detected_language_score"] = np.nan
        events["language_detection_status"] = "not_run"
        events["is_english_text"] = np.nan

    before_dedup = int(len(events))
    events = events.drop_duplicates(subset=["event_id"]).reset_index(drop=True)
    log(f"Dropped duplicate event IDs: {before_dedup - len(events):,}.", start_time)

    events = ensure_unified_event_schema(events)
    missing = [c for c in OUTPUT_COLUMNS if c not in events.columns]
    if missing:
        raise RuntimeError(f"Output schema enforcement failed. Missing columns: {missing}")

    output_path = output_dir / "safety_text_event.csv.gz"
    save_csv(events, output_path)
    summary = {
        "output_path": str(output_path),
        "schema_columns": OUTPUT_COLUMNS,
        "row_count": int(len(events)),
        "source_type_counts": events["source_type"].value_counts(dropna=False).to_dict(),
        "date_min": str(events["event_date"].min()),
        "date_max": str(events["event_date"].max()),
        "empty_text_count": int((~events["has_text"]).sum()),
        **language_summary,
    }
    save_json(summary, output_dir / "00_unified_text_events_summary.json")
    log(f"Saved unified event table to {output_path}", start_time)
    log("Step 00 complete.", start_time)
    print(summary)


if __name__ == "__main__":
    main()
