#!/usr/bin/env python3
"""Build location-period-theme profile tables for dashboard/review.

This converts event-level source-specific theme assignments into complete
counts and trends by location, time period, and theme.
"""
from __future__ import annotations

try:
    import config as cfg
    from theme_utils import ProgressLogger, assign_period, compact_text, ensure_dir, read_csv, safe_value_counts, save_json, write_csv
except ImportError:  # pragma: no cover
    from . import config as cfg
    from .theme_utils import ProgressLogger, assign_period, compact_text, ensure_dir, read_csv, safe_value_counts, save_json, write_csv

import numpy as np
import pandas as pd


EVENT_KIND_COLUMNS = {
    "serious_injury": "serious_injury_count",
    "normal_injury": "normal_injury_count",
    "near_miss": "near_miss_count",
    "hazard_identification": "hazard_identification_count",
    "audit_unsafe_condition": "audit_unsafe_condition_count",
    "audit_unsafe_act": "audit_unsafe_act_count",
    "audit_safe_condition": "audit_safe_condition_count",
    "audit_safe_act": "audit_safe_act_count",
    "audit_positive_observation": "audit_positive_observation_count",
    "audit_observation": "audit_observation_count",
    "audit_other": "audit_other_count",
    "task_overdue": "task_overdue_count",
    "task_open": "task_open_count",
    "task_other": "task_other_count",
}


def _load_event_theme_table(log: ProgressLogger) -> pd.DataFrame:
    if not cfg.THEME_ASSIGNMENTS_FILE.exists():
        raise FileNotFoundError(f"Missing assignments file: {cfg.THEME_ASSIGNMENTS_FILE}. Run 03_cluster_by_family.py first.")
    if not cfg.THEME_INPUT_ALL_FILE.exists():
        raise FileNotFoundError(f"Missing theme input file: {cfg.THEME_INPUT_ALL_FILE}. Run 01_prepare_theme_text.py first.")
    assignments = read_csv(cfg.THEME_ASSIGNMENTS_FILE)
    events = read_csv(cfg.THEME_INPUT_ALL_FILE)
    merge_keys = ["theme_row_id", "event_id"] if "theme_row_id" in assignments.columns and "theme_row_id" in events.columns else ["event_id"]
    df = assignments.merge(events, on=merge_keys, how="left", suffixes=("", "_event"))
    for col in [
        "source_type", "event_kind", "event_date", "location_id", "location_path", "site", "department",
        "title", "clean_text", "theme_text", "category", "status", "raw_source_family", "audit_signal_type", "audit_cluster_family", "review_priority", "is_open_task", "is_overdue_task",
    ]:
        ec = f"{col}_event"
        if col not in df.columns and ec in df.columns:
            df[col] = df[ec]
        elif ec in df.columns:
            df[col] = df[col].where(df[col].notna(), df[ec])
    if cfg.THEME_CATALOG_FILE.exists():
        catalog = read_csv(cfg.THEME_CATALOG_FILE)[["theme_id", "theme_name", "top_terms"]].drop_duplicates("theme_id")
        df = df.merge(catalog, on="theme_id", how="left")
    else:
        df["theme_name"] = df["theme_id"]
        df["top_terms"] = ""
    log.log(f"loaded event-theme table rows={len(df):,}")
    return df


def _kind_count(g: pd.DataFrame, kind: str) -> int:
    return int(g.get("event_kind", pd.Series("", index=g.index)).fillna("").astype(str).eq(kind).sum())


def _make_profile(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["location_id", "location_path", "site", "department", "period", "source_family", "theme_id", "theme_name"]
    for keys, g in df.groupby(group_cols, dropna=False):
        key_dict = dict(zip(group_cols, keys))
        dates = pd.to_datetime(g.get("event_date", pd.Series(pd.NaT, index=g.index)), errors="coerce")
        row = {
            **key_dict,
            "event_count": int(len(g)),
            "first_event_date": str(dates.min()) if dates.notna().any() else "",
            "last_event_date": str(dates.max()) if dates.notna().any() else "",
            "mean_theme_confidence": float(pd.to_numeric(g.get("theme_confidence", pd.Series(np.nan, index=g.index)), errors="coerce").mean()),
            "strong_cluster_count": int(g.get("assignment_type", pd.Series("", index=g.index)).astype(str).eq("strong_cluster").sum()),
            "weak_nearest_theme_count": int(g.get("assignment_type", pd.Series("", index=g.index)).astype(str).eq("weak_nearest_theme").sum()),
            "source_type_mix": safe_value_counts(g.get("source_type", pd.Series("", index=g.index)), 10),
            "event_kind_mix": safe_value_counts(g.get("event_kind", pd.Series("", index=g.index)), 15),
            "category_mix": safe_value_counts(g.get("category", pd.Series("", index=g.index)), 10),
            "representative_event_ids": ";".join(g.sort_values(["review_priority", "theme_confidence"], ascending=[False, False]).get("event_id", pd.Series(dtype=str)).astype(str).head(10).tolist()),
            "top_terms": compact_text(g.get("top_terms", pd.Series("", index=g.index)).dropna().astype(str).iloc[0] if "top_terms" in g and g["top_terms"].notna().any() else "", 1000),
        }
        for kind, col in EVENT_KIND_COLUMNS.items():
            row[col] = _kind_count(g, kind)
        row["injury_count"] = row.get("serious_injury_count", 0) + row.get("normal_injury_count", 0)
        row["audit_count"] = int(g.get("source_type", pd.Series("", index=g.index)).fillna("").astype(str).str.lower().eq("audit").sum())
        row["task_count"] = int(g.get("source_type", pd.Series("", index=g.index)).fillna("").astype(str).str.lower().eq("task").sum())
        row["open_action_count"] = int(g.get("is_open_task", pd.Series(False, index=g.index)).fillna(False).astype(bool).sum()) if "is_open_task" in g else 0
        row["overdue_action_count"] = int(g.get("is_overdue_task", pd.Series(False, index=g.index)).fillna(False).astype(bool).sum()) if "is_overdue_task" in g else 0
        row["theme_review_score"] = (
            10 * row.get("serious_injury_count", 0)
            + 4 * row.get("normal_injury_count", 0)
            + 3 * row.get("near_miss_count", 0)
            + 1 * row.get("hazard_identification_count", 0)
            + 2 * row.get("audit_unsafe_condition_count", 0)
            + 2 * row.get("audit_unsafe_act_count", 0)
            + 0.8 * row.get("audit_safe_condition_count", 0)
            + 0.8 * row.get("audit_safe_act_count", 0)
            + 0.8 * row.get("audit_positive_observation_count", 0)
            + 2 * row.get("overdue_action_count", 0)
            + 0.5 * row.get("open_action_count", 0)
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _make_top_themes(profile: pd.DataFrame) -> pd.DataFrame:
    if profile.empty:
        return profile
    rows = []
    for keys, g in profile.groupby(["location_id", "location_path", "site", "department", "period"], dropna=False):
        g = g.sort_values(["theme_review_score", "event_count"], ascending=[False, False])
        top = g.head(int(cfg.TOP_THEMES_PER_LOCATION_PERIOD))
        rows.append({
            "location_id": keys[0],
            "location_path": keys[1],
            "site": keys[2],
            "department": keys[3],
            "period": keys[4],
            "theme_count": int(g["theme_id"].nunique()),
            "event_count": int(g["event_count"].sum()),
            "review_score": float(g["theme_review_score"].sum()),
            "serious_injury_count": int(g.get("serious_injury_count", pd.Series(dtype=int)).sum()),
            "normal_injury_count": int(g.get("normal_injury_count", pd.Series(dtype=int)).sum()),
            "near_miss_count": int(g.get("near_miss_count", pd.Series(dtype=int)).sum()),
            "hazard_identification_count": int(g.get("hazard_identification_count", pd.Series(dtype=int)).sum()),
            "audit_count": int(g.get("audit_count", pd.Series(dtype=int)).sum()),
            "audit_safe_condition_count": int(g.get("audit_safe_condition_count", pd.Series(dtype=int)).sum()),
            "audit_safe_act_count": int(g.get("audit_safe_act_count", pd.Series(dtype=int)).sum()),
            "audit_positive_observation_count": int(g.get("audit_positive_observation_count", pd.Series(dtype=int)).sum()),
            "task_count": int(g.get("task_count", pd.Series(dtype=int)).sum()),
            "open_action_count": int(g.get("open_action_count", pd.Series(dtype=int)).sum()),
            "overdue_action_count": int(g.get("overdue_action_count", pd.Series(dtype=int)).sum()),
            "top_themes": " || ".join([f"{r.theme_id}: {r.theme_name} (events={int(r.event_count)}, score={float(r.theme_review_score):.1f})" for r in top.itertuples()]),
            "top_theme_ids": ";".join(top["theme_id"].astype(str).tolist()),
        })
    return pd.DataFrame(rows).sort_values(["review_score", "event_count"], ascending=[False, False])


def _make_theme_trends(profile: pd.DataFrame) -> pd.DataFrame:
    if profile.empty:
        return profile
    agg = profile.groupby(["source_family", "theme_id", "theme_name", "period"], dropna=False).agg(
        locations_active=("location_id", "nunique"),
        event_count=("event_count", "sum"),
        theme_review_score=("theme_review_score", "sum"),
        serious_injury_count=("serious_injury_count", "sum"),
        normal_injury_count=("normal_injury_count", "sum"),
        near_miss_count=("near_miss_count", "sum"),
        hazard_identification_count=("hazard_identification_count", "sum"),
        audit_count=("audit_count", "sum"),
        audit_safe_condition_count=("audit_safe_condition_count", "sum"),
        audit_safe_act_count=("audit_safe_act_count", "sum"),
        audit_positive_observation_count=("audit_positive_observation_count", "sum"),
        task_count=("task_count", "sum"),
        open_action_count=("open_action_count", "sum"),
        overdue_action_count=("overdue_action_count", "sum"),
    ).reset_index()
    return agg.sort_values(["theme_review_score", "event_count"], ascending=[False, False])


def _make_location_theme_rollup(profile: pd.DataFrame) -> pd.DataFrame:
    if profile.empty:
        return profile
    agg = profile.groupby(["location_id", "location_path", "site", "department", "source_family", "theme_id", "theme_name"], dropna=False).agg(
        periods_active=("period", "nunique"),
        event_count=("event_count", "sum"),
        theme_review_score=("theme_review_score", "sum"),
        serious_injury_count=("serious_injury_count", "sum"),
        normal_injury_count=("normal_injury_count", "sum"),
        near_miss_count=("near_miss_count", "sum"),
        hazard_identification_count=("hazard_identification_count", "sum"),
        audit_count=("audit_count", "sum"),
        audit_safe_condition_count=("audit_safe_condition_count", "sum"),
        audit_safe_act_count=("audit_safe_act_count", "sum"),
        audit_positive_observation_count=("audit_positive_observation_count", "sum"),
        task_count=("task_count", "sum"),
        open_action_count=("open_action_count", "sum"),
        overdue_action_count=("overdue_action_count", "sum"),
        representative_event_ids=("representative_event_ids", lambda s: ";".join([x for x in ";".join(s.dropna().astype(str)).split(";") if x][:20])),
    ).reset_index()
    return agg.sort_values(["theme_review_score", "event_count"], ascending=[False, False])


def main() -> None:
    log = ProgressLogger("05_build_location_theme_period_profiles")
    ensure_dir(cfg.THEME_PROFILE_DIR)
    df = _load_event_theme_table(log)
    df["event_date"] = pd.to_datetime(df.get("event_date"), errors="coerce")
    df["period"] = assign_period(df["event_date"], str(cfg.PROFILE_PERIOD_FREQ))
    for col in ["location_id", "location_path", "site", "department", "source_family", "theme_id", "theme_name"]:
        if col not in df.columns:
            df[col] = "unknown"
        df[col] = df[col].fillna("unknown").astype(str)

    log.log("building location-theme-period profile")
    profile = _make_profile(df)
    write_csv(profile, cfg.LOCATION_THEME_PERIOD_FILE)

    log.log("building location-period top themes")
    top = _make_top_themes(profile)
    write_csv(top, cfg.LOCATION_PERIOD_TOP_THEMES_FILE)

    log.log("building theme-period trends")
    trends = _make_theme_trends(profile)
    write_csv(trends, cfg.THEME_PERIOD_TRENDS_FILE)

    log.log("building location-theme rollup")
    rollup = _make_location_theme_rollup(profile)
    write_csv(rollup, cfg.LOCATION_THEME_ROLLUP_FILE)

    save_json({
        "profile_file": str(cfg.LOCATION_THEME_PERIOD_FILE),
        "top_themes_file": str(cfg.LOCATION_PERIOD_TOP_THEMES_FILE),
        "theme_trends_file": str(cfg.THEME_PERIOD_TRENDS_FILE),
        "location_theme_rollup_file": str(cfg.LOCATION_THEME_ROLLUP_FILE),
        "profile_rows": int(len(profile)),
        "top_rows": int(len(top)),
        "trend_rows": int(len(trends)),
        "rollup_rows": int(len(rollup)),
        "period_freq": str(cfg.PROFILE_PERIOD_FREQ),
    }, cfg.THEME_PROFILE_DIR / "location_theme_profile_summary.json")
    log.done("location/theme period profiles complete")


if __name__ == "__main__":
    main()
