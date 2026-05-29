#!/usr/bin/env python
from __future__ import annotations

import argparse

import config as cfg
import numpy as np
import pandas as pd

from utils import ensure_dir, load_table, parse_datetime_series, save_csv, save_json


def add_source_flags(df: pd.DataFrame) -> pd.DataFrame:
    s = df["source_type"].fillna("").astype(str)
    df["is_incident_record"] = s.eq("incident").astype(int)
    df["is_hazard_record"] = s.eq("hazard_identification").astype(int)
    df["is_near_miss_record"] = s.eq("near_miss").astype(int)
    df["is_audit_record"] = s.eq("audit").astype(int)
    df["is_task_record"] = s.eq("task").astype(int)
    df["is_leading_record"] = s.isin(["hazard_identification", "near_miss", "audit"]).astype(int)
    return df


def compute_event_burden(df: pd.DataFrame) -> pd.Series:
    s = df["source_type"].fillna("").astype(str)
    base = np.select(
        [s.eq("incident"), s.eq("near_miss"), s.eq("hazard_identification"), s.eq("audit")],
        [2.0, 2.0, 1.0, 1.0],
        default=0.0,
    )
    consequence = pd.to_numeric(df.get("consequence_score", 0), errors="coerce").fillna(0).to_numpy(float)
    severe = df.get("severe_actual", pd.Series(False, index=df.index)).fillna(False).astype(bool).to_numpy()
    injury = df.get("any_injury", pd.Series(False, index=df.index)).fillna(False).astype(bool).to_numpy()
    unsafe_audit = (
        df.get("audit_type", pd.Series("", index=df.index)).fillna("").astype(str).str.lower().str.contains("unsafe", na=False).to_numpy()
        if "audit_type" in df.columns else np.zeros(len(df), dtype=bool)
    )
    burden = base + consequence + 10.0 * severe + 3.0 * injury + 2.0 * unsafe_audit
    burden[s.eq("task").to_numpy()] = 0.0
    return pd.Series(burden, index=df.index)


def prepare_events(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["event_date"] = parse_datetime_series(df["event_date"])
    df["due_date"] = parse_datetime_series(df.get("due_date", pd.Series(pd.NaT, index=df.index)))
    df["completion_date"] = parse_datetime_series(df.get("completion_date", pd.Series(pd.NaT, index=df.index)))
    df = df[df["event_date"].notna()].copy()
    df = df[df.get("risk_theme_id", "").fillna("").ne("UNASSIGNED")].copy()
    df["site"] = df.get("site", "").fillna("").replace("", "Unknown Site")
    df["department"] = df.get("department", "").fillna("").replace("", "Unknown Department")
    df["risk_theme_name"] = df.get("risk_theme_name", df["risk_theme_id"]).fillna(df["risk_theme_id"])
    for c in ["any_injury", "severe_actual"]:
        df[c] = df.get(c, False).fillna(False).astype(bool)
    df["consequence_score"] = pd.to_numeric(df.get("consequence_score", 0), errors="coerce").fillna(0)
    df["theme_similarity_score"] = pd.to_numeric(df.get("theme_similarity_score", 0), errors="coerce").fillna(0)
    df = add_source_flags(df)
    df["event_burden"] = compute_event_burden(df)
    return df.reset_index(drop=True)


def aggregate_lookback(df: pd.DataFrame, as_of: pd.Timestamp, lookback_days: int, keys: list[str]) -> pd.DataFrame:
    start = as_of - pd.Timedelta(days=lookback_days)
    win = df[(df["event_date"] > start) & (df["event_date"] <= as_of)].copy()
    if win.empty:
        return pd.DataFrame(columns=keys)
    task_open = win["is_task_record"].eq(1) & (win["completion_date"].isna() | (win["completion_date"] > as_of))
    task_overdue = task_open & win["due_date"].notna() & (win["due_date"] < as_of)
    win["task_open_asof"] = task_open.astype(int)
    win["task_overdue_asof"] = task_overdue.astype(int)
    win["task_days_overdue_asof"] = np.where(task_overdue, (as_of - win["due_date"]).dt.days, 0)
    win["serious_or_above"] = win["consequence_score"].ge(3).astype(int)
    win["fatality_potential_flag"] = win.get("consequence_potential", "").fillna("").astype(str).eq("fatality_potential").astype(int)
    win["has_control_failure"] = win.get("control_failure_tags", "").fillna("").astype(str).ne("").astype(int)
    win["has_hazard_tag"] = win.get("hazard_tags", "").fillna("").astype(str).ne("").astype(int)
    p = f"lb{lookback_days}d"
    return win.groupby(keys, dropna=False).agg(
        **{
            f"{p}_event_count": ("event_id", "count"),
            f"{p}_incident_count": ("is_incident_record", "sum"),
            f"{p}_hazard_count": ("is_hazard_record", "sum"),
            f"{p}_near_miss_count": ("is_near_miss_record", "sum"),
            f"{p}_audit_count": ("is_audit_record", "sum"),
            f"{p}_task_count": ("is_task_record", "sum"),
            f"{p}_open_task_count": ("task_open_asof", "sum"),
            f"{p}_overdue_task_count": ("task_overdue_asof", "sum"),
            f"{p}_max_task_days_overdue": ("task_days_overdue_asof", "max"),
            f"{p}_serious_or_above_count": ("serious_or_above", "sum"),
            f"{p}_fatality_potential_count": ("fatality_potential_flag", "sum"),
            f"{p}_severe_actual_count": ("severe_actual", "sum"),
            f"{p}_any_injury_count": ("any_injury", "sum"),
            f"{p}_control_failure_count": ("has_control_failure", "sum"),
            f"{p}_hazard_tagged_count": ("has_hazard_tag", "sum"),
            f"{p}_mean_consequence_score": ("consequence_score", "mean"),
            f"{p}_max_consequence_score": ("consequence_score", "max"),
            f"{p}_mean_theme_similarity": ("theme_similarity_score", "mean"),
            f"{p}_days_since_last_event": ("event_date", lambda x: (as_of - x.max()).days),
        }
    ).reset_index()


def aggregate_future(df: pd.DataFrame, as_of: pd.Timestamp, horizon_days: int, keys: list[str]) -> pd.DataFrame:
    end = as_of + pd.Timedelta(days=horizon_days)
    fut = df[(df["event_date"] > as_of) & (df["event_date"] <= end) & df["source_type"].ne("task")].copy()
    cols = keys + [f"future_risk_burden_{horizon_days}d", f"future_event_count_{horizon_days}d", f"future_severe_actual_count_{horizon_days}d", f"future_any_injury_count_{horizon_days}d"]
    if fut.empty:
        return pd.DataFrame(columns=cols)
    return fut.groupby(keys, dropna=False).agg(
        **{
            f"future_risk_burden_{horizon_days}d": ("event_burden", "sum"),
            f"future_event_count_{horizon_days}d": ("event_id", "count"),
            f"future_severe_actual_count_{horizon_days}d": ("severe_actual", "sum"),
            f"future_any_injury_count_{horizon_days}d": ("any_injury", "sum"),
        }
    ).reset_index()


def main():
    parser = argparse.ArgumentParser(description="Build theme-specific time-window risk-state modeling table.")
    parser.add_argument("--events", default=cfg.THEMED_EVENTS_PATH)
    parser.add_argument("--embeddings", default=cfg.TEXT_EMBEDDINGS_PATH, help="Accepted for interface consistency; not used by default.")
    parser.add_argument("--output-dir", default=cfg.STEP_05_DIR)
    parser.add_argument("--asof-frequency", default=cfg.ASOF_FREQUENCY)
    parser.add_argument("--lookbacks", default=cfg.LOOKBACK_WINDOWS)
    parser.add_argument("--horizons", default=cfg.PREDICTION_HORIZONS)
    parser.add_argument("--embedding-components", type=int, default=0, help="Reserved for later; current version uses text-derived themes/tags.")
    parser.add_argument("--min-history-events", type=int, default=cfg.MIN_HISTORY_EVENTS)
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    df = prepare_events(load_table(args.events))
    lookbacks = [int(x) for x in args.lookbacks.split(",") if x.strip()]
    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    max_lb, max_h = max(lookbacks), max(horizons)
    start = (df["event_date"].min() + pd.Timedelta(days=max_lb)).normalize()
    end = (df["event_date"].max() - pd.Timedelta(days=max_h)).normalize()
    as_of_dates = pd.date_range(start=start, end=end, freq=args.asof_frequency)
    keys = ["site", "department", "risk_theme_id", "risk_theme_name"]
    all_rows = []
    for as_of in as_of_dates:
        base = None
        for lb in lookbacks:
            agg = aggregate_lookback(df, as_of, lb, keys)
            base = agg if base is None else base.merge(agg, on=keys, how="outer")
        if base is None or base.empty:
            continue
        for h in horizons:
            base = base.merge(aggregate_future(df, as_of, h, keys), on=keys, how="left")
        base["as_of_date"] = as_of
        all_rows.append(base)
        print(f"built as_of={as_of.date()} rows={len(base)}")
    if not all_rows:
        raise ValueError("No risk-state rows created. Check dates and theme assignments.")
    out = pd.concat(all_rows, ignore_index=True, sort=False)
    target_prefixes = tuple(["lb", "future"])
    for c in out.columns:
        if c.startswith(target_prefixes):
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)
    hist_col = f"lb{max_lb}d_event_count"
    if hist_col in out.columns and args.min_history_events > 0:
        out = out[out[hist_col] >= args.min_history_events].copy()
    if "lb30d_event_count" in out.columns and "lb180d_event_count" in out.columns:
        out["trend_30_vs_180_daily_rate"] = (out["lb30d_event_count"] / 30.0) / ((out["lb180d_event_count"] / 180.0) + 1e-6)
    if "lb90d_overdue_task_count" in out.columns and "lb90d_event_count" in out.columns:
        out["overdue_task_per_event_90d"] = out["lb90d_overdue_task_count"] / (out["lb90d_event_count"] + 1.0)
    output_path = output_dir / "risk_state_training_data.csv.gz"
    save_csv(out, output_path)
    summary = {
        "output_path": str(output_path),
        "row_count": int(len(out)),
        "as_of_count": int(out["as_of_date"].nunique()),
        "lookbacks": lookbacks,
        "horizons": horizons,
        "date_min": str(out["as_of_date"].min()),
        "date_max": str(out["as_of_date"].max()),
    }
    save_json(summary, output_dir / "05_risk_state_dataset_summary.json")
    print(summary)


if __name__ == "__main__":
    main()
