#!/usr/bin/env python
"""Prepare the unified safety text table for local retrieval.

Run without args:

    python scripts/00_prepare_knowledge_base.py

Configuration lives in src/safety_retrieval_agent/config.py.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from _bootstrap import PROJECT_ROOT  # noqa: F401
from safety_retrieval_agent.config import get_settings
from safety_retrieval_agent.utils import (
    bool_series,
    ensure_dir,
    extract_year_month,
    parse_datetime,
    read_csv,
    save_json,
)


def main() -> dict:
    settings = get_settings()
    start = time.time()
    ensure_dir(settings.output_dir)
    ensure_dir(settings.knowledge_base_path().parent)

    print(f"[00-prepare] Loading unified input file: {settings.input_event_file}", flush=True)
    if not settings.input_event_file.exists():
        raise FileNotFoundError(
            f"Unified event file not found: {settings.input_event_file}. "
            "Run scripts/00_build_unified_text_events.py first or update input_event_file in config.py."
        )
    df = read_csv(settings.input_event_file, nrows=settings.max_records)
    original_count = len(df)

    required_cols = ["event_id", "source_type", "event_date", "title", "description", "clean_text"]
    missing_required = [c for c in required_cols if c not in df.columns]
    if missing_required:
        raise ValueError(f"Input file is missing required columns: {missing_required}")

    for col in ["site", "department", "location_path", "source_subtype", "category", "status", "audit_type"]:
        if col not in df.columns:
            df[col] = ""
    bool_cols = [
        "any_injury", "severe_actual", "fatality", "losttime", "restrictedtime", "inpatient",
        "emergencyroom", "is_open_task", "is_overdue_task", "has_text",
    ]
    for col in bool_cols:
        if col not in df.columns:
            df[col] = False
        df[col] = bool_series(df[col])
    if "injury_record_count" not in df.columns:
        df["injury_record_count"] = 0
    df["injury_record_count"] = pd.to_numeric(df["injury_record_count"], errors="coerce").fillna(0).astype(int)

    df["event_date_dt"] = parse_datetime(df["event_date"])
    min_date = pd.Timestamp(settings.min_event_date)
    max_date = pd.Timestamp(settings.max_event_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    missing_date_count = int(df["event_date_dt"].isna().sum())
    out_of_range_count = int((df["event_date_dt"].notna() & ~df["event_date_dt"].between(min_date, max_date)).sum())
    df = df[df["event_date_dt"].notna() & df["event_date_dt"].between(min_date, max_date)].copy()

    print(f"[00-prepare] Rows after date filter {settings.min_event_date} to {settings.max_event_date}: {len(df):,}", flush=True)
    df["description_text"] = df["description"].fillna("").astype(str).str.strip()
    df["description_text_length"] = df["description_text"].str.len().astype(int)
    df["description_nonempty"] = df["description_text_length"].gt(0)
    df["retrieval_text"] = df["clean_text"].fillna("").astype(str)
    blank_text = df["retrieval_text"].str.strip().eq("")
    if blank_text.any():
        df.loc[blank_text, "retrieval_text"] = (
            df.loc[blank_text, "title"].fillna("").astype(str)
            + " | " + df.loc[blank_text, "description"].fillna("").astype(str)
            + " | " + df.loc[blank_text, "source_subtype"].fillna("").astype(str)
        )
    if "text_length" in df.columns:
        df["text_length"] = pd.to_numeric(df["text_length"], errors="coerce").fillna(df["retrieval_text"].str.len()).astype(int)
    else:
        df["text_length"] = df["retrieval_text"].str.len()
    df["retrieval_text"] = df["retrieval_text"].str.slice(0, settings.max_retrieval_text_chars)
    too_short_count = int((df["text_length"] < settings.min_text_chars).sum())
    df = df[df["text_length"] >= settings.min_text_chars].copy()
    df["text_word_count"] = np.maximum(1, (df["text_length"] // 6).astype(int))

    source_type_l = df["source_type"].fillna("").astype(str).str.lower()
    audit_label_l = (df["audit_type"].fillna("").astype(str) + " " + df["source_subtype"].fillna("").astype(str)).str.lower()
    role = source_type_l.copy()
    any_injury_mask = df["any_injury"] | df["injury_record_count"].gt(0)
    severe_mask = df["severe_actual"]
    role = role.mask(any_injury_mask & severe_mask, "severe_injury")
    role = role.mask(any_injury_mask & ~severe_mask, "injury")
    role = role.mask(source_type_l.eq("hazard_identification"), "hazard_identification")
    role = role.mask(source_type_l.eq("near_miss"), "near_miss")
    task_mask = source_type_l.eq("task")
    role = role.mask(task_mask, "corrective_action")
    role = role.mask(task_mask & df["is_open_task"], "open_corrective_action")
    role = role.mask(task_mask & df["is_overdue_task"], "overdue_corrective_action")
    audit_mask = source_type_l.eq("audit")
    role = role.mask(audit_mask, "audit_observation")
    role = role.mask(audit_mask & audit_label_l.str.contains("inspection", na=False), "inspection")
    role = role.mask(audit_mask & audit_label_l.str.contains("unsafe", na=False), "unsafe_observation")
    role = role.mask(audit_mask & audit_label_l.str.contains("safe", na=False) & ~audit_label_l.str.contains("unsafe", na=False), "safe_observation")
    df["source_role"] = role
    df["event_year_month"] = extract_year_month(df["event_date_dt"])

    df = df.reset_index(drop=True)
    df["row_id"] = np.arange(len(df))
    df["event_date_iso"] = df["event_date_dt"].dt.strftime("%Y-%m-%dT%H:%M:%S")

    keep_cols = [
        "row_id", "event_id", "source_type", "source_role", "source_subtype", "source_id",
        "event_date", "event_date_dt", "event_date_iso", "event_year_month",
        "location_id", "site", "department", "location_path",
        "title", "description_text", "description_text_length", "description_nonempty",
        "retrieval_text", "status", "category", "audit_type",
        "is_open_task", "is_overdue_task", "due_date", "completion_date",
        "any_injury", "severe_actual", "fatality", "losttime", "restrictedtime",
        "inpatient", "emergencyroom", "injury_record_count",
        "text_length", "text_word_count",
    ]
    df = df[[c for c in keep_cols if c in df.columns]].copy()

    print(f"[00-prepare] Saving compact knowledge base to {settings.knowledge_base_path()}", flush=True)
    df.to_pickle(settings.knowledge_base_path())
    sample_csv_path = settings.knowledge_base_path().with_name("safety_knowledge_base_sample.csv")
    df.head(1000).to_csv(sample_csv_path, index=False)
    full_csv_path = None
    if settings.prepare_save_full_csv:
        full_csv_path = settings.knowledge_base_csv_path()
        df.to_csv(full_csv_path, index=False, compression="gzip")

    summary = {
        "input_file": str(settings.input_event_file),
        "output_dir": str(settings.output_dir),
        "original_row_count": int(original_count),
        "missing_event_date_count": missing_date_count,
        "out_of_range_date_count": out_of_range_count,
        "too_short_text_count_after_date_filter": too_short_count,
        "final_row_count": int(len(df)),
        "main_table_path": str(settings.knowledge_base_path()),
        "sample_csv_path": str(sample_csv_path),
        "full_csv_path": str(full_csv_path) if full_csv_path is not None else None,
        "date_filter": {"min_event_date": settings.min_event_date, "max_event_date": settings.max_event_date},
        "source_type_counts": {str(k): int(v) for k, v in df["source_type"].value_counts(dropna=False).to_dict().items()},
        "source_role_counts": {str(k): int(v) for k, v in df["source_role"].value_counts(dropna=False).to_dict().items()},
        "elapsed_seconds": round(time.time() - start, 2),
    }
    save_json(summary, settings.output_dir / "00_prepare_knowledge_base_summary.json")
    print(f"[00-prepare] Complete: {summary}", flush=True)
    return summary


if __name__ == "__main__":
    main()
