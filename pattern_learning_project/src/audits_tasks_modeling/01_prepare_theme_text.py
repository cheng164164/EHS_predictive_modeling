#!/usr/bin/env python3
"""Prepare source-aware theme-mining input from the unified event table.

The important change in this version is audit-specific filtering:
  * incident/hazard and task/action records are passed through normally
  * audit records are split into three paths:
      - audit_risk: unsafe acts/conditions and risk observations for clustering
      - audit_positive: safe acts/conditions and positive-control observations for clustering
      - audit_activity: routine inspections/templates/admin records for accounting only

This prevents audit clusters from mixing "safe" and "unsafe" wording, while
still preserving safe observations as positive-control evidence. Routine form
names such as forklift inspections, vehicle inspections, generic safety
observations, and checklists are counted but not clustered.

Outputs are saved under outputs/audits_tasks_modeling/01_theme_input.
Runs without command-line arguments; edit config.py for settings.
"""
from __future__ import annotations

try:
    import config as cfg
    from theme_utils import (
        ProgressLogger,
        compact_text,
        ensure_dir,
        read_csv,
        save_json,
        write_csv,
    )
except ImportError:  # pragma: no cover
    from . import config as cfg
    from .theme_utils import (
        ProgressLogger,
        compact_text,
        ensure_dir,
        read_csv,
        save_json,
        write_csv,
    )

from typing import Any
import re

import numpy as np
import pandas as pd


KEEP_COLUMNS = [
    "event_id", "source_type", "source_subtype", "source_id", "event_date",
    "location_id", "site", "department", "location_path", "title", "description",
    "clean_text", "status", "category", "audit_type", "task_source_module",
    "is_open_task", "is_overdue_task", "due_date", "completion_date",
    "any_injury", "severe_actual", "fatality", "losttime", "restrictedtime",
    "inpatient", "emergencyroom", "injury_record_count", "text_length", "has_text",
    "detected_language", "detected_language_score", "language_detection_status", "is_english_text",
]

COUNT_COLUMNS = [
    "serious_injury", "normal_injury", "near_miss", "hazard_identification",
    "audit_unsafe_condition", "audit_unsafe_act", "audit_safe_condition", "audit_safe_act",
    "audit_positive_observation", "audit_observation", "audit_other",
    "task_overdue", "task_open", "task_other",
]

REVIEW_PRIORITY_BY_KIND = {
    "serious_injury": 10.0,
    "normal_injury": 4.0,
    "near_miss": 3.0,
    "hazard_identification": 1.0,
    "audit_unsafe_condition": 2.5,
    "audit_unsafe_act": 2.5,
    "audit_safe_condition": 1.4,
    "audit_safe_act": 1.4,
    "audit_positive_observation": 1.2,
    "audit_observation": 1.0,
    "task_overdue": 2.0,
    "task_open": 0.5,
}


def _s(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return df[col].fillna("").astype(str)
    return pd.Series("", index=df.index, dtype="object")


def _truthy_series(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})


def _contains_any(series: pd.Series, patterns: list[str] | tuple[str, ...] | set[str]) -> pd.Series:
    pats = [str(p) for p in patterns if str(p).strip()]
    if not pats:
        return pd.Series(False, index=series.index)
    text = series.fillna("").astype(str)
    # Single union regex is much faster than scanning the same text once per pattern.
    union = "(?:" + ")|(?:".join(pats) + ")"
    try:
        return text.str.contains(union, case=False, regex=True, na=False)
    except re.error:
        out = pd.Series(False, index=series.index)
        for pat in pats:
            try:
                out = out | text.str.contains(str(pat), case=False, regex=True, na=False)
            except re.error:
                out = out | text.str.contains(str(pat), case=False, regex=False, na=False)
        return out


def _clean_pattern_text(series: pd.Series) -> pd.Series:
    out = series.fillna("").astype(str).str.lower()
    for old in ["\r", "\n", "\t", "\u2022", "â€¢", "â€“", "â€™", "â€œ", "â€\x9d"]:
        out = out.str.replace(old, " ", regex=False)
    out = out.str.replace(r"\s+", " ", regex=True).str.strip()
    return out


def _classify_source_family_vectorized(df: pd.DataFrame) -> pd.Series:
    source_type = _s(df, "source_type").str.lower()
    combined = (
        source_type + " "
        + _s(df, "source_subtype").str.lower() + " "
        + _s(df, "category").str.lower() + " "
        + _s(df, "audit_type").str.lower() + " "
        + _s(df, "task_source_module").str.lower()
    )
    is_task = source_type.eq("task") | combined.str.contains(r"\b(?:task|action)\b", regex=True, na=False)
    is_audit = source_type.eq("audit") | combined.str.contains(r"\b(?:audit|observation|inspection)\b", regex=True, na=False)
    return pd.Series(
        np.select(
            [is_task, is_audit],
            [cfg.FAMILY_TASK_ACTION, cfg.FAMILY_AUDIT_OBSERVATION],
            default=cfg.FAMILY_INCIDENT_HAZARD,
        ),
        index=df.index,
    )


def _classify_event_kind_vectorized(df: pd.DataFrame, source_family: pd.Series) -> pd.Series:
    source_type = _s(df, "source_type").str.lower()
    combined = (
        source_type + " "
        + _s(df, "category").str.lower() + " "
        + _s(df, "source_subtype").str.lower() + " "
        + _s(df, "audit_type").str.lower() + " "
        + _s(df, "status").str.lower()
    )
    any_injury = _truthy_series(df["any_injury"]) if "any_injury" in df.columns else pd.Series(False, index=df.index)
    serious = pd.Series(False, index=df.index)
    for c in ["severe_actual", "fatality", "losttime", "restrictedtime", "inpatient"]:
        if c in df.columns:
            serious = serious | _truthy_series(df[c])

    is_near_miss = combined.str.contains("near miss", regex=False, na=False) | source_type.eq("near_miss")
    is_hazard = combined.str.contains("hazard", regex=False, na=False)
    is_audit = source_family.eq(cfg.FAMILY_AUDIT_OBSERVATION)
    is_task = source_family.eq(cfg.FAMILY_TASK_ACTION)
    is_unsafe_condition = combined.str.contains("unsafe condition", regex=False, na=False)
    is_unsafe_act = combined.str.contains("unsafe act", regex=False, na=False)
    is_safe_condition = combined.str.contains("safe condition", regex=False, na=False)
    is_safe_act = combined.str.contains("safe act", regex=False, na=False)
    is_observation = combined.str.contains("observation", regex=False, na=False)
    is_overdue_task = _truthy_series(df["is_overdue_task"]) if "is_overdue_task" in df.columns else pd.Series(False, index=df.index)
    is_open_task = _truthy_series(df["is_open_task"]) if "is_open_task" in df.columns else pd.Series(False, index=df.index)

    conditions = [
        serious,
        any_injury,
        is_near_miss,
        is_hazard,
        is_audit & is_unsafe_condition,
        is_audit & is_unsafe_act,
        is_audit & is_safe_condition,
        is_audit & is_safe_act,
        is_audit & is_observation,
        is_audit,
        is_task & is_overdue_task,
        is_task & is_open_task,
        is_task,
    ]
    choices = [
        "serious_injury",
        "normal_injury",
        "near_miss",
        "hazard_identification",
        "audit_unsafe_condition",
        "audit_unsafe_act",
        "audit_safe_condition",
        "audit_safe_act",
        "audit_observation",
        "audit_other",
        "task_overdue",
        "task_open",
        "task_other",
    ]
    return pd.Series(np.select(conditions, choices, default=source_type.replace("", "other")), index=df.index)


def _location_label_from_path(path: Any, site: Any = "") -> str:
    path_s = compact_text(path)
    if path_s:
        parts = [p.strip() for p in path_s.replace("\\", "/").split("/") if p.strip()]
        if parts:
            return parts[-1]
    return compact_text(site) or "unknown"


def _location_key(df: pd.DataFrame) -> pd.Series:
    loc_id = _s(df, "location_id").str.strip()
    loc_path = _s(df, "location_path").str.strip()
    return pd.Series(np.where(loc_id.ne("") & loc_id.ne("nan"), loc_id, "path:" + loc_path), index=df.index)


def _add_location_profile_fields(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["location_key"] = _location_key(out)
    paths = _s(out, "location_path")
    sites = _s(out, "site")
    out["location_label"] = [_location_label_from_path(path, site) for path, site in zip(paths, sites)]
    return out


def _audit_text_for_rules(df: pd.DataFrame) -> pd.Series:
    """Text used only to decide audit clustering eligibility.

    This is intentionally truncated for speed; rule-based eligibility only needs
    the beginning of title/description/clean_text, not the whole audit record.
    """
    limits = {
        "title": 250,
        "description": 500,
        "clean_text": 900,
        "category": 120,
        "source_subtype": 160,
        "audit_type": 120,
        "status": 80,
    }
    parts = [
        _s(df, "title").str.slice(0, limits["title"]),
        _s(df, "description").str.slice(0, limits["description"]),
        _s(df, "clean_text").str.slice(0, limits["clean_text"]),
        _s(df, "category").str.slice(0, limits["category"]),
        _s(df, "source_subtype").str.slice(0, limits["source_subtype"]),
        _s(df, "audit_type").str.slice(0, limits["audit_type"]),
        _s(df, "status").str.slice(0, limits["status"]),
    ]
    combined = parts[0]
    for p in parts[1:]:
        combined = combined + " | " + p
    return _clean_pattern_text(combined)


def _audit_specific_text_for_rules(df: pd.DataFrame) -> pd.Series:
    """Text excluding classification/status fields, used for information quality."""
    parts = [
        _s(df, "title").str.slice(0, 250),
        _s(df, "description").str.slice(0, 500),
        _s(df, "clean_text").str.slice(0, 900),
    ]
    combined = parts[0]
    for p in parts[1:]:
        combined = combined + " | " + p
    out = _clean_pattern_text(combined)
    for pat in getattr(cfg, "AUDIT_THEME_TEXT_REMOVE_PATTERNS", []):
        try:
            out = out.str.replace(str(pat), " ", case=False, regex=True)
        except re.error:
            out = out.str.replace(str(pat), " ", case=False, regex=False)
    out = out.str.replace(r"\s+", " ", regex=True).str.strip()
    return out


def _classify_audit_clustering_eligibility(df: pd.DataFrame, log: ProgressLogger) -> pd.DataFrame:
    """Split audits into risk, positive-control, and activity-only paths.

    Output semantics:
      audit_risk      -> embedded/clustered separately
      audit_positive  -> embedded/clustered separately
      audit_activity  -> accounting only, not embedded/clustered

    Non-audit records remain eligible and keep their original source family.
    """
    out = df.copy()
    out["audit_signal_type"] = "not_audit"
    out["audit_cluster_eligible"] = True
    out["audit_cluster_exclusion_reason"] = ""
    out["audit_cluster_family"] = out["source_family"].astype(str)
    out["raw_source_family"] = out.get("raw_source_family", out["source_family"]).astype(str)

    is_audit = out["raw_source_family"].astype(str).eq(cfg.FAMILY_AUDIT_OBSERVATION) | out["source_type"].astype(str).str.lower().eq("audit")
    if not is_audit.any():
        return out

    audit = out.loc[is_audit].copy()
    all_text = _audit_text_for_rules(audit)
    specific_text = _audit_specific_text_for_rules(audit)
    title = _clean_pattern_text(_s(audit, "title"))
    category = _clean_pattern_text(_s(audit, "category"))
    subtype = _clean_pattern_text(_s(audit, "source_subtype"))
    audit_type = _clean_pattern_text(_s(audit, "audit_type"))
    status = _clean_pattern_text(_s(audit, "status"))
    event_kind = audit["event_kind"].fillna("").astype(str)

    is_unsafe_kind = event_kind.isin(["audit_unsafe_condition", "audit_unsafe_act"])
    is_safe_kind = event_kind.isin(["audit_safe_condition", "audit_safe_act"])

    has_risk_keyword = _contains_any(all_text, getattr(cfg, "AUDIT_RISK_KEYWORD_PATTERNS", [])) | _contains_any(specific_text, getattr(cfg, "AUDIT_RISK_KEYWORD_PATTERNS", []))
    has_positive_keyword = _contains_any(all_text, getattr(cfg, "AUDIT_POSITIVE_KEYWORD_PATTERNS", [])) | _contains_any(specific_text, getattr(cfg, "AUDIT_POSITIVE_KEYWORD_PATTERNS", []))
    has_unsafe_act_text = _contains_any(all_text, getattr(cfg, "AUDIT_UNSAFE_ACT_PATTERNS", []))
    has_unsafe_condition_text = _contains_any(all_text, getattr(cfg, "AUDIT_UNSAFE_CONDITION_PATTERNS", []))
    has_safe_act_text = _contains_any(all_text, getattr(cfg, "AUDIT_SAFE_ACT_PATTERNS", []))
    has_safe_condition_text = _contains_any(all_text, getattr(cfg, "AUDIT_SAFE_CONDITION_PATTERNS", []))

    is_scheduled_status = _contains_any(status, getattr(cfg, "AUDIT_ROUTINE_STATUS_PATTERNS", []))
    is_routine_text = _contains_any(all_text, getattr(cfg, "AUDIT_ROUTINE_EXCLUDE_PATTERNS", []))
    is_template_admin = (
        category.str.contains(r"risk\s+assessment|assessment|checklist|inspection", regex=True, na=False)
        | subtype.str.contains(r"risk\s+assessment|checklist|scheduled", regex=True, na=False)
        | audit_type.str.contains(r"risk\s+assessment|checklist", regex=True, na=False)
    )
    is_generic_title = _contains_any(title, getattr(cfg, "AUDIT_GENERIC_TITLE_PATTERNS", [])) | (title.str.len() <= 12)
    specific_len = specific_text.str.len()

    min_chars = int(getattr(cfg, "AUDIT_MIN_CLUSTER_TEXT_CHARS", 25))
    min_obs_chars = int(getattr(cfg, "AUDIT_MIN_MEANINGFUL_OBSERVATION_CHARS", 60))
    include_generic_unsafe = bool(getattr(cfg, "AUDIT_INCLUDE_UNSAFE_FINDINGS_WITH_GENERIC_TEXT", False))
    include_general_obs = bool(getattr(cfg, "AUDIT_INCLUDE_GENERAL_OBSERVATIONS_WITH_RISK_KEYWORDS", True))
    include_safe = bool(getattr(cfg, "AUDIT_INCLUDE_SAFE_ACT_CONDITION_CLUSTERING", True))

    # Unsafe findings or specific risk observations become audit_risk.
    unsafe_specific = (
        (is_unsafe_kind | has_unsafe_act_text | has_unsafe_condition_text)
        & (specific_len >= min_chars)
        & (~is_generic_title | has_risk_keyword | (specific_len >= min_obs_chars))
    )
    unsafe_generic_but_allowed = (is_unsafe_kind | has_unsafe_act_text | has_unsafe_condition_text) & include_generic_unsafe & has_risk_keyword
    meaningful_risk_observation = (
        include_general_obs
        & ~is_unsafe_kind
        & ~is_safe_kind
        & ~has_safe_act_text
        & ~has_safe_condition_text
        & ~is_template_admin
        & ~is_scheduled_status
        & has_risk_keyword
        & ((specific_len >= min_obs_chars) | (~is_generic_title & specific_len >= min_chars))
    )
    # If the source system explicitly labels a record as Safe Act / Safe Condition,
    # keep it in the positive-control audit path. This prevents semantically
    # similar words like "PPE", "guard", or "condition" from moving safe
    # observations back into the audit_risk family.
    risk_eligible = (unsafe_specific | unsafe_generic_but_allowed | meaningful_risk_observation) & ~is_template_admin & ~is_safe_kind

    # Safe findings or specific positive-control observations become audit_positive.
    safe_specific = (
        include_safe
        & (is_safe_kind | has_safe_act_text | has_safe_condition_text)
        & (specific_len >= min_chars)
        & (~is_generic_title | has_positive_keyword | (specific_len >= min_obs_chars))
    )
    meaningful_positive_observation = (
        include_safe
        & ~risk_eligible
        & ~is_unsafe_kind
        & ~has_unsafe_act_text
        & ~has_unsafe_condition_text
        & ~is_template_admin
        & ~is_scheduled_status
        & has_positive_keyword
        & ((specific_len >= min_obs_chars) | (~is_generic_title & specific_len >= min_chars))
    )
    positive_eligible = (safe_specific | meaningful_positive_observation) & ~is_template_admin & ~risk_eligible

    eligible = risk_eligible | positive_eligible
    routine_or_activity = (is_scheduled_status | is_routine_text | is_template_admin) & ~eligible

    signal_type = pd.Series("audit_other_accounting", index=audit.index, dtype="object")
    signal_type[is_template_admin & ~eligible] = "template_admin_or_risk_assessment"
    signal_type[routine_or_activity] = "routine_inspection_activity"
    signal_type[(specific_len < min_chars) & ~eligible] = "generic_or_low_information"
    signal_type[meaningful_risk_observation] = "meaningful_risk_observation"
    signal_type[meaningful_positive_observation] = "meaningful_positive_observation"
    signal_type[risk_eligible & (event_kind.eq("audit_unsafe_condition") | has_unsafe_condition_text)] = "unsafe_condition_finding"
    signal_type[risk_eligible & (event_kind.eq("audit_unsafe_act") | has_unsafe_act_text)] = "unsafe_act_finding"
    signal_type[positive_eligible & (event_kind.eq("audit_safe_condition") | has_safe_condition_text)] = "safe_condition_observation"
    signal_type[positive_eligible & (event_kind.eq("audit_safe_act") | has_safe_act_text)] = "safe_act_observation"
    signal_type[is_safe_kind & ~positive_eligible & ~eligible] = "safe_positive_accounting_only"

    family = pd.Series(cfg.FAMILY_AUDIT_ACTIVITY, index=audit.index, dtype="object")
    family[risk_eligible] = cfg.FAMILY_AUDIT_RISK
    family[positive_eligible] = cfg.FAMILY_AUDIT_POSITIVE

    reason = pd.Series("eligible_for_audit_theme_clustering", index=audit.index, dtype="object")
    reason[~eligible & is_template_admin] = "template_admin_or_risk_assessment"
    reason[~eligible & is_scheduled_status] = "scheduled_status"
    reason[~eligible & is_routine_text] = "routine_inspection_or_checklist"
    reason[~eligible & is_safe_kind] = "safe_positive_but_low_information"
    reason[~eligible & is_generic_title & ~(has_risk_keyword | has_positive_keyword)] = "generic_title_without_specific_risk_or_control_text"
    reason[~eligible & (specific_len < min_chars)] = "too_little_specific_text"
    reason[~eligible & reason.eq("eligible_for_audit_theme_clustering")] = "not_meaningful_for_clustering"

    # Write audit-specific fields back.
    out.loc[audit.index, "audit_signal_type"] = signal_type
    out.loc[audit.index, "audit_cluster_eligible"] = eligible.astype(bool)
    out.loc[audit.index, "audit_cluster_exclusion_reason"] = np.where(eligible, "", reason)
    out.loc[audit.index, "audit_cluster_family"] = family
    out.loc[audit.index, "source_family"] = family
    out.loc[audit.index, "audit_specific_text_length"] = specific_len
    out.loc[audit.index, "audit_has_risk_keyword"] = has_risk_keyword.astype(bool)
    out.loc[audit.index, "audit_has_positive_keyword"] = has_positive_keyword.astype(bool)
    out.loc[audit.index, "audit_is_routine_pattern"] = is_routine_text.astype(bool)
    out.loc[audit.index, "audit_is_generic_title"] = is_generic_title.astype(bool)

    # Upgrade review priority. This is only a sampling/sorting score, not a risk model.
    out.loc[audit.index[risk_eligible & event_kind.eq("audit_unsafe_condition")], "review_priority"] = 2.8
    out.loc[audit.index[risk_eligible & event_kind.eq("audit_unsafe_act")], "review_priority"] = 2.8
    out.loc[audit.index[meaningful_risk_observation], "review_priority"] = 1.6
    out.loc[audit.index[positive_eligible & event_kind.eq("audit_safe_condition")], "review_priority"] = 1.4
    out.loc[audit.index[positive_eligible & event_kind.eq("audit_safe_act")], "review_priority"] = 1.4
    out.loc[audit.index[meaningful_positive_observation], "review_priority"] = 1.2
    out.loc[audit.index[family.eq(cfg.FAMILY_AUDIT_ACTIVITY)], "review_priority"] = 0.1

    log.log(
        "audit split for clustering/accounting: "
        f"audit_rows={int(is_audit.sum()):,}; "
        f"audit_risk={int(risk_eligible.sum()):,}; "
        f"audit_positive={int(positive_eligible.sum()):,}; "
        f"audit_activity_only={int((~eligible).sum()):,}"
    )
    return out


def _write_audit_accounting_outputs(df: pd.DataFrame, log: ProgressLogger) -> None:
    raw_family = df.get("raw_source_family", df.get("source_family", pd.Series("", index=df.index))).astype(str)
    audit_families = {cfg.FAMILY_AUDIT_OBSERVATION, cfg.FAMILY_AUDIT_RISK, cfg.FAMILY_AUDIT_POSITIVE, cfg.FAMILY_AUDIT_ACTIVITY}
    audit = df[raw_family.eq(cfg.FAMILY_AUDIT_OBSERVATION) | df.get("source_family", pd.Series("", index=df.index)).astype(str).isin(audit_families) | df.get("source_type", pd.Series("", index=df.index)).astype(str).str.lower().eq("audit")].copy()
    if audit.empty:
        return
    audit["event_date"] = pd.to_datetime(audit.get("event_date"), errors="coerce")
    audit["period_y"] = audit["event_date"].dt.to_period("Y").astype(str).replace("NaT", "unknown")

    eligible = audit[audit["audit_cluster_eligible"].fillna(False).astype(bool)].copy()
    excluded = audit[~audit["audit_cluster_eligible"].fillna(False).astype(bool)].copy()
    keep_cols = [
        "event_id", "source_type", "source_family", "raw_source_family", "source_subtype", "event_kind", "audit_signal_type",
        "audit_cluster_family", "audit_cluster_exclusion_reason", "event_date", "period_y", "location_id",
        "location_key", "location_label", "location_path", "title", "category", "audit_type",
        "status", "clean_text", "review_priority", "audit_specific_text_length",
        "audit_has_risk_keyword", "audit_is_routine_pattern", "audit_is_generic_title",
    ]
    keep_cols = [c for c in keep_cols if c in audit.columns]
    if bool(getattr(cfg, "AUDIT_KEEP_ROUTINE_INSPECTIONS_FOR_ACCOUNTING", True)):
        max_excluded = int(getattr(cfg, "AUDIT_EXCLUDED_SAMPLE_ROWS", 0) or 0)
        excluded_out = excluded[keep_cols].copy()
        if max_excluded > 0 and len(excluded_out) > max_excluded:
            excluded_out = excluded_out.sample(n=max_excluded, random_state=int(cfg.RANDOM_STATE)).copy()
        write_csv(excluded_out, cfg.AUDIT_CLUSTER_EXCLUDED_FILE)
    eligible_path = getattr(cfg, "AUDIT_ELIGIBLE_FOR_CLUSTERING_FILE", None)
    if eligible_path is not None:
        max_eligible = int(getattr(cfg, "AUDIT_ELIGIBLE_SAMPLE_ROWS", 0) or 0)
        eligible_out = eligible[keep_cols].copy()
        if max_eligible > 0 and len(eligible_out) > max_eligible:
            eligible_out = eligible_out.sample(n=max_eligible, random_state=int(cfg.RANDOM_STATE)).copy()
        write_csv(eligible_out, eligible_path)

    # Separate review samples for the two clustered audit paths.
    for fam_attr, path_attr in [("FAMILY_AUDIT_RISK", "AUDIT_RISK_FOR_CLUSTERING_FILE"), ("FAMILY_AUDIT_POSITIVE", "AUDIT_POSITIVE_FOR_CLUSTERING_FILE")]:
        fam = getattr(cfg, fam_attr, "")
        path = getattr(cfg, path_attr, None)
        if path is not None and fam:
            sub = audit[audit.get("source_family", pd.Series("", index=audit.index)).astype(str).eq(str(fam))][keep_cols].copy()
            max_rows = int(getattr(cfg, "AUDIT_ELIGIBLE_SAMPLE_ROWS", 0) or 0)
            if max_rows > 0 and len(sub) > max_rows:
                sub = sub.sample(n=max_rows, random_state=int(cfg.RANDOM_STATE)).copy()
            write_csv(sub, path)

    profile_cols = ["source_family", "audit_signal_type", "audit_cluster_family", "audit_cluster_eligible", "audit_cluster_exclusion_reason", "event_kind", "source_subtype", "category", "audit_type", "status"]
    profile_cols = [c for c in profile_cols if c in audit.columns]
    profile = (
        audit.groupby(profile_cols, dropna=False)
        .size()
        .reset_index(name="row_count")
        .sort_values("row_count", ascending=False)
    )
    write_csv(profile, cfg.AUDIT_CLUSTER_ELIGIBILITY_PROFILE_FILE)

    acct_group_cols = ["location_key", "location_label", "location_path", "period_y", "source_family", "audit_signal_type", "event_kind"]
    acct_group_cols = [c for c in acct_group_cols if c in audit.columns]
    accounting = (
        audit.groupby(acct_group_cols, dropna=False)
        .size()
        .reset_index(name="audit_record_count")
        .sort_values(["audit_record_count"], ascending=False)
    )
    write_csv(accounting, cfg.AUDIT_ACTIVITY_ACCOUNTING_FILE)
    log.log("wrote audit accounting/eligibility outputs")


def _score_locations(df: pd.DataFrame, log: ProgressLogger) -> tuple[pd.DataFrame, set[str]]:
    weights = dict(getattr(cfg, "POC_LOCATION_SCORE_WEIGHTS", {}))
    score = df["event_kind"].map(lambda k: float(weights.get(str(k), 0.1))).astype(float)
    loc_df = df[["location_key", "location_id", "location_label", "location_path", "site", "department", "source_family", "event_kind"]].copy()
    loc_df["location_signal_score_component"] = score

    def first_nonempty(s: pd.Series) -> str:
        for x in s:
            val = compact_text(x)
            if val:
                return val
        return ""

    base = loc_df.groupby("location_key", dropna=False).agg(
        location_id=("location_id", first_nonempty),
        location_label=("location_label", first_nonempty),
        location_path=("location_path", first_nonempty),
        site=("site", first_nonempty),
        department=("department", first_nonempty),
        total_records=("event_kind", "size"),
        location_signal_score=("location_signal_score_component", "sum"),
        families_present=("source_family", lambda s: int(s.nunique(dropna=True))),
    ).reset_index()

    family_counts = loc_df.pivot_table(index="location_key", columns="source_family", values="event_kind", aggfunc="size", fill_value=0).rename_axis(None, axis=1).reset_index()
    for fam in getattr(cfg, "SOURCE_FAMILIES", []):
        if fam not in family_counts.columns:
            family_counts[fam] = 0
        family_counts = family_counts.rename(columns={fam: f"{fam}_records"})

    kind_counts = loc_df.pivot_table(index="location_key", columns="event_kind", values="source_family", aggfunc="size", fill_value=0).rename_axis(None, axis=1).reset_index()
    for kind in COUNT_COLUMNS:
        if kind not in kind_counts.columns:
            kind_counts[kind] = 0
        kind_counts = kind_counts.rename(columns={kind: f"{kind}_count"})

    profile = base.merge(family_counts, on="location_key", how="left").merge(kind_counts, on="location_key", how="left")
    numeric_cols = [c for c in profile.columns if c.endswith("_records") or c.endswith("_count")]
    profile[numeric_cols] = profile[numeric_cols].fillna(0).astype(int)

    profile["balanced_family_bonus"] = profile["families_present"].astype(float) * 25.0
    profile["poc_location_score"] = profile["location_signal_score"] + profile["balanced_family_bonus"] + np.log1p(profile["total_records"]) * 2.0

    min_records = int(getattr(cfg, "POC_MIN_LOCATION_RECORDS", 0) or 0)
    min_families = int(getattr(cfg, "POC_MIN_FAMILIES_PRESENT", 1) or 1)
    candidates = profile[(profile["total_records"] >= min_records) & (profile["families_present"] >= min_families)].copy()

    exclude_ids = {str(x) for x in getattr(cfg, "POC_EXCLUDE_LOCATION_IDS", [])}
    if exclude_ids and "location_id" in candidates.columns:
        candidates = candidates[~candidates["location_id"].astype(str).isin(exclude_ids)].copy()

    top_n = int(getattr(cfg, "POC_TOP_LOCATIONS", 0) or 0)
    candidates = candidates.sort_values(["poc_location_score", "total_records"], ascending=False)
    selected = candidates.head(top_n).copy() if top_n > 0 else candidates.copy()

    include_ids = {str(x) for x in getattr(cfg, "POC_INCLUDE_LOCATION_IDS", [])}
    if include_ids:
        manual = profile[profile["location_id"].astype(str).isin(include_ids)].copy()
        selected = pd.concat([selected, manual], ignore_index=True).drop_duplicates("location_key")

    selected_keys = set(selected["location_key"].astype(str).tolist())
    profile["selected_for_poc"] = profile["location_key"].astype(str).isin(selected_keys)
    profile = profile.sort_values(["selected_for_poc", "poc_location_score", "total_records"], ascending=[False, False, False])

    write_csv(profile, cfg.POC_LOCATION_PROFILE_FILE)
    write_csv(profile[profile["selected_for_poc"]].copy(), cfg.POC_SELECTED_LOCATIONS_FILE)
    log.log(f"selected {len(selected_keys):,} major locations for POC sample from {len(profile):,} locations")
    return profile, selected_keys


def _scaled_family_quotas(max_total: int) -> dict[str, int]:
    quotas = {str(k): int(v) for k, v in dict(getattr(cfg, "POC_FAMILY_QUOTAS", {})).items()}
    families = [str(f) for f in getattr(cfg, "SOURCE_FAMILIES", [])]
    if not quotas:
        each = max_total // max(len(families), 1)
        quotas = {f: each for f in families}
    for fam in families:
        quotas.setdefault(fam, 0)
    total = sum(max(v, 0) for v in quotas.values())
    if total <= 0:
        each = max_total // max(len(families), 1)
        quotas = {f: each for f in families}
        total = sum(quotas.values())
    if total != max_total:
        scaled = {fam: int(round(max(0, q) * max_total / total)) for fam, q in quotas.items()}
        diff = max_total - sum(scaled.values())
        ordered = sorted(scaled, key=scaled.get, reverse=True)
        for i in range(abs(diff)):
            fam = ordered[i % len(ordered)]
            scaled[fam] += 1 if diff > 0 else -1
        quotas = scaled
    return quotas


def _sample_family(g: pd.DataFrame, n: int, random_state: int) -> pd.DataFrame:
    if n <= 0 or len(g) == 0:
        return g.iloc[0:0].copy()
    if len(g) <= n:
        return g.copy()

    keep_kinds = {str(x) for x in getattr(cfg, "POC_ALWAYS_KEEP_EVENT_KINDS", set())}
    min_pri = float(getattr(cfg, "POC_ALWAYS_KEEP_MIN_REVIEW_PRIORITY", 99.0))
    high_mask = g["event_kind"].astype(str).isin(keep_kinds) | (g["review_priority"].astype(float) >= min_pri)
    high = g[high_mask].copy()
    rest = g[~high_mask].copy()

    if len(high) >= n:
        weights = high["review_priority"].astype(float).clip(lower=0.1) + 1.0
        return high.sample(n=n, replace=False, weights=weights, random_state=random_state).copy()

    remaining_n = n - len(high)
    if len(rest) <= remaining_n:
        return pd.concat([high, rest], ignore_index=False).copy()

    samples = [high]
    rest_counts = rest["event_kind"].astype(str).value_counts()
    total_rest = int(rest_counts.sum())
    allocations = {str(kind): int(np.floor(remaining_n * count / total_rest)) for kind, count in rest_counts.items()}
    for kind in rest_counts.index:
        if remaining_n >= len(rest_counts) and allocations.get(str(kind), 0) == 0:
            allocations[str(kind)] = 1
    diff = remaining_n - sum(allocations.values())
    ordered_kinds = list(rest_counts.index)
    for i in range(abs(diff)):
        kind = str(ordered_kinds[i % len(ordered_kinds)])
        allocations[kind] = allocations.get(kind, 0) + (1 if diff > 0 else -1)

    for idx, (kind, kg) in enumerate(rest.groupby(rest["event_kind"].astype(str), dropna=False)):
        k = max(0, min(int(allocations.get(str(kind), 0)), len(kg)))
        if k > 0:
            weights = kg["review_priority"].astype(float).clip(lower=0.1) + 1.0
            samples.append(kg.sample(n=k, replace=False, weights=weights, random_state=random_state + idx + 1))

    out = pd.concat(samples, ignore_index=False)
    if len(out) > n:
        weights = out["review_priority"].astype(float).clip(lower=0.1) + 1.0
        out = out.sample(n=n, replace=False, weights=weights, random_state=random_state)
    return out.copy()


def _apply_poc_sampling(df: pd.DataFrame, log: ProgressLogger) -> tuple[pd.DataFrame, dict[str, Any]]:
    log.log("POC major-location sampling is enabled")
    before = int(len(df))
    _, selected_keys = _score_locations(df, log)
    if not selected_keys:
        raise RuntimeError("POC sampling selected zero locations. Lower POC_MIN_LOCATION_RECORDS or POC_MIN_FAMILIES_PRESENT.")

    selected_df = df[df["location_key"].astype(str).isin(selected_keys)].copy()
    after_location = int(len(selected_df))
    log.log(f"location filter kept {after_location:,}/{before:,} cluster-eligible rows across selected major locations")

    max_total = int(getattr(cfg, "POC_MAX_TOTAL_RECORDS", 0) or 0)
    if max_total <= 0 or after_location <= max_total:
        sampled = selected_df.copy()
        quotas = {}
    else:
        quotas = _scaled_family_quotas(max_total)
        pieces = []
        for offset, family in enumerate(getattr(cfg, "SOURCE_FAMILIES", [])):
            g = selected_df[selected_df["source_family"] == family].copy()
            fam_quota = int(quotas.get(str(family), 0))
            sampled_g = _sample_family(g, fam_quota, int(cfg.RANDOM_STATE) + offset * 1000)
            pieces.append(sampled_g)
            log.log(f"sampled family={family}: eligible_available={len(g):,}; quota={fam_quota:,}; kept={len(sampled_g):,}")
        sampled = pd.concat(pieces, ignore_index=False).copy()
        if len(sampled) > max_total:
            sampled = sampled.sample(n=max_total, random_state=int(cfg.RANDOM_STATE)).copy()

    sampled = sampled.sort_values(["source_family", "location_label", "event_date", "event_id"]).reset_index(drop=True)
    log.log(f"POC sample final size before text build: {len(sampled):,} rows")

    summary = {
        "poc_sampling_enabled": True,
        "rows_before_location_sampling_cluster_eligible": before,
        "rows_after_location_filter_cluster_eligible": after_location,
        "rows_after_record_sampling_before_text_filter": int(len(sampled)),
        "selected_location_count": len(selected_keys),
        "poc_top_locations": int(getattr(cfg, "POC_TOP_LOCATIONS", 0) or 0),
        "poc_max_total_records": max_total,
        "family_quotas_scaled": quotas,
        "family_counts_after_sampling_before_text_filter": sampled["source_family"].value_counts(dropna=False).to_dict(),
        "event_kind_counts_after_sampling_before_text_filter": sampled["event_kind"].value_counts(dropna=False).to_dict(),
        "audit_signal_type_counts_after_sampling_before_text_filter": sampled.get("audit_signal_type", pd.Series(dtype=object)).value_counts(dropna=False).to_dict(),
        "selected_locations_file": str(cfg.POC_SELECTED_LOCATIONS_FILE),
        "location_profile_file": str(cfg.POC_LOCATION_PROFILE_FILE),
    }
    save_json(summary, cfg.POC_SAMPLING_SUMMARY_FILE)
    return sampled, summary


def _apply_legacy_family_sample(df: pd.DataFrame, log: ProgressLogger) -> pd.DataFrame:
    max_per_family = int(getattr(cfg, "MAX_RECORDS_PER_FAMILY", 0) or 0)
    if max_per_family <= 0:
        return df
    before = len(df)
    pieces = []
    for family, g in df.groupby("source_family", dropna=False):
        if len(g) > max_per_family:
            high = g[g["review_priority"] >= 3.0]
            remaining_n = max(max_per_family - len(high), 0)
            rest = g.drop(index=high.index)
            if remaining_n > 0 and len(rest) > 0:
                rest = rest.sample(n=min(remaining_n, len(rest)), random_state=int(cfg.RANDOM_STATE))
            g = pd.concat([high, rest], ignore_index=False).head(max_per_family)
        pieces.append(g)
    out = pd.concat(pieces, ignore_index=True)
    log.log(f"legacy MAX_RECORDS_PER_FAMILY={max_per_family:,}: {before:,} -> {len(out):,}")
    return out


def _light_text_series(df: pd.DataFrame, col: str, limit: int) -> pd.Series:
    if col not in df.columns:
        return pd.Series("", index=df.index, dtype="object")
    out = df[col].fillna("").astype(str).str.slice(0, limit)
    for old, new in [("\r", " "), ("\n", " "), ("\t", " "), ("\u2022", " "), ("â€¢", " "), ("â€“", "-"), ("â€™", "'")]:
        out = out.str.replace(old, new, regex=False)
    return out.str.replace(r"\s+", " ", regex=True).str.strip()


def _clean_audit_theme_text(text: pd.Series, max_chars: int) -> pd.Series:
    out = _clean_pattern_text(text)
    for pat in getattr(cfg, "AUDIT_THEME_TEXT_REMOVE_PATTERNS", []):
        try:
            out = out.str.replace(str(pat), " ", case=False, regex=True)
        except re.error:
            out = out.str.replace(str(pat), " ", case=False, regex=False)
    # Remove repeated source-field labels and obvious audit status/form language.
    out = out.str.replace(r"\b(?:title|description|comments|status|category|audit type)\s*:\s*", " ", regex=True)
    out = out.str.replace(r"\s+", " ", regex=True).str.strip()
    return out.str.slice(0, max_chars)


def _prepare_text_for_sample(df: pd.DataFrame, max_chars: int, log: ProgressLogger) -> pd.DataFrame:
    """Create compact model text for sampled rows only.

    Audit rows use source-specific text construction that avoids classification
    fields such as status/category/source_subtype, because those caused previous
    audit clusters to learn form structure rather than actual findings.
    """
    limits = {
        "title": 300,
        "description": 700,
        "clean_text": max_chars,
        "category": 180,
        "source_subtype": 180,
        "audit_type": 180,
        "task_source_module": 180,
        "status": 120,
        "source_type": 80,
        "site": 180,
        "department": 180,
        "location_path": 500,
        "location_id": 100,
    }
    for col, limit in limits.items():
        df[col] = _light_text_series(df, col, limit)
    log.log("compact metadata/text columns prepared")

    raw_family = df.get("raw_source_family", df.get("source_family", pd.Series("", index=df.index))).astype(str)
    audit_families = {cfg.FAMILY_AUDIT_OBSERVATION, cfg.FAMILY_AUDIT_RISK, cfg.FAMILY_AUDIT_POSITIVE, cfg.FAMILY_AUDIT_ACTIVITY}
    is_audit = raw_family.eq(cfg.FAMILY_AUDIT_OBSERVATION) | df["source_family"].astype(str).isin(audit_families) | df.get("source_type", pd.Series("", index=df.index)).astype(str).str.lower().eq("audit")

    # Default non-audit text keeps source context.
    pieces = []
    for col in ["title", "description", "clean_text", "category", "source_subtype", "audit_type", "task_source_module", "status"]:
        pieces.append(df[col].fillna("").astype(str))
    combined = pieces[0]
    for val in pieces[1:]:
        combined = combined + " | " + val
    combined = combined.str.slice(0, max_chars)

    # Audit text uses only finding/observation content and removes audit boilerplate.
    audit_pieces = df["title"].fillna("").astype(str) + " | " + df["description"].fillna("").astype(str) + " | " + df["clean_text"].fillna("").astype(str)
    audit_text = _clean_audit_theme_text(audit_pieces, max_chars=max_chars)
    combined.loc[is_audit] = audit_text.loc[is_audit]

    df["theme_text"] = combined.fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df["theme_text_length"] = df["theme_text"].str.len()
    log.log("theme_text built")
    return df


def main() -> None:
    log = ProgressLogger("01_prepare_theme_text")
    ensure_dir(cfg.THEME_INPUT_DIR)

    if not cfg.UNIFIED_EVENTS_FILE.exists():
        raise FileNotFoundError(
            f"Unified event table not found: {cfg.UNIFIED_EVENTS_FILE}. "
            "Run 00_build_unified_text_events.py first or set UNIFIED_EVENTS_FILE in config.py."
        )

    log.log(f"reading unified events: {cfg.UNIFIED_EVENTS_FILE}")
    header = pd.read_csv(cfg.UNIFIED_EVENTS_FILE, nrows=0).columns.tolist()
    usecols = [c for c in KEEP_COLUMNS if c in header]
    df = read_csv(cfg.UNIFIED_EVENTS_FILE, usecols=usecols)
    log.log(f"loaded {len(df):,} rows")

    if "event_date" in df.columns:
        df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
        if cfg.MIN_EVENT_DATE:
            before = len(df)
            df = df[df["event_date"] >= pd.Timestamp(cfg.MIN_EVENT_DATE)].copy()
            log.log(f"date filter min removed {before - len(df):,} rows")
        if cfg.MAX_EVENT_DATE:
            before = len(df)
            df = df[df["event_date"] <= pd.Timestamp(cfg.MAX_EVENT_DATE)].copy()
            log.log(f"date filter max removed {before - len(df):,} rows")

    raw_text = _s(df, "clean_text")
    before = len(df)
    df = df[raw_text.str.strip().str.len() >= int(cfg.MIN_TEXT_LENGTH)].copy()
    log.log(f"raw clean_text length filter removed {before - len(df):,}; kept {len(df):,}")

    log.log("classifying source family and event kind with vectorized rules")
    df["source_family"] = _classify_source_family_vectorized(df)
    df["raw_source_family"] = df["source_family"].astype(str)
    df["event_kind"] = _classify_event_kind_vectorized(df, df["source_family"])
    df["review_priority"] = df["event_kind"].map(lambda k: REVIEW_PRIORITY_BY_KIND.get(str(k), 0.1)).astype(float)
    df = _add_location_profile_fields(df)

    if bool(getattr(cfg, "AUDIT_CLUSTER_ONLY_MEANINGFUL_FINDINGS", True)):
        df = _classify_audit_clustering_eligibility(df, log)
    else:
        df["audit_signal_type"] = np.where(df["source_family"].eq(cfg.FAMILY_AUDIT_OBSERVATION), "audit_clustering_unfiltered", "not_audit")
        df["audit_cluster_eligible"] = True
        df["audit_cluster_exclusion_reason"] = ""
        df["audit_cluster_family"] = df["source_family"].astype(str)

    _write_audit_accounting_outputs(df, log)

    # Only cluster audit records that pass the eligibility filter. Non-audit
    # records remain eligible. Routine/scheduled audits are still available in
    # audit_activity_accounting.csv and audit_records_excluded_from_clustering.csv.
    before = len(df)
    df_cluster = df[df["audit_cluster_eligible"].fillna(True).astype(bool)].copy()
    log.log(f"audit eligibility filter for clustering removed {before - len(df_cluster):,}; kept {len(df_cluster):,}")

    sampling_summary: dict[str, Any] = {"poc_sampling_enabled": False}
    if bool(getattr(cfg, "ENABLE_POC_MAJOR_LOCATION_SAMPLE", False)):
        df_cluster, sampling_summary = _apply_poc_sampling(df_cluster, log)
    else:
        df_cluster = _apply_legacy_family_sample(df_cluster, log)

    max_chars = int(getattr(cfg, "MAX_TEXT_CHARS_FOR_MODEL", 1800))
    log.log(f"building compact theme_text for {len(df_cluster):,} clustering rows; max_chars={max_chars:,}")
    df_cluster = _prepare_text_for_sample(df_cluster, max_chars=max_chars, log=log)

    before = len(df_cluster)
    df_cluster = df_cluster[df_cluster["theme_text_length"] >= int(cfg.MIN_TEXT_LENGTH)].copy()
    log.log(f"final theme_text length filter removed {before - len(df_cluster):,}; kept {len(df_cluster):,}")

    if bool(getattr(cfg, "DROP_DUPLICATE_THEME_TEXT_WITHIN_FAMILY", False)):
        before = len(df_cluster)
        df_cluster = df_cluster.sort_values(["source_family", "review_priority"], ascending=[True, False])
        df_cluster = df_cluster.drop_duplicates(subset=["source_family", "theme_text"]).copy()
        log.log(f"dropped duplicate theme_text within family: {before - len(df_cluster):,}")

    sort_cols = [c for c in ["source_family", "location_label", "event_date", "event_id"] if c in df_cluster.columns]
    if sort_cols:
        df_cluster = df_cluster.sort_values(sort_cols).reset_index(drop=True)

    df_cluster["theme_row_id"] = range(len(df_cluster))

    output_cols = [
        "theme_row_id", "event_id", "source_type", "source_family", "raw_source_family", "source_subtype", "source_id",
        "event_kind", "audit_signal_type", "audit_cluster_family", "audit_cluster_eligible", "audit_cluster_exclusion_reason",
        "audit_specific_text_length", "audit_has_risk_keyword", "audit_has_positive_keyword", "audit_is_routine_pattern", "audit_is_generic_title",
        "event_date", "location_id", "location_key", "location_label", "location_path",
        "site", "department", "title", "clean_text", "theme_text", "category", "status",
        "audit_type", "task_source_module", "is_open_task", "is_overdue_task", "due_date",
        "completion_date", "any_injury", "severe_actual", "fatality", "losttime", "restrictedtime",
        "inpatient", "emergencyroom", "injury_record_count", "review_priority", "theme_text_length",
    ]
    output_cols = [c for c in output_cols if c in df_cluster.columns]
    df_cluster = df_cluster[output_cols].copy()

    write_csv(df_cluster, cfg.THEME_INPUT_ALL_FILE)
    for family, path in cfg.THEME_INPUT_FILE_BY_FAMILY.items():
        sub = df_cluster[df_cluster["source_family"] == family].copy()
        write_csv(sub, path)

    profile_group_cols = ["source_family", "source_type", "event_kind"]
    if "audit_signal_type" in df_cluster.columns:
        profile_group_cols.append("audit_signal_type")
    profile = (
        df_cluster.groupby(profile_group_cols, dropna=False)
        .size()
        .reset_index(name="row_count")
        .sort_values(["source_family", "row_count"], ascending=[True, False])
    )
    write_csv(profile, cfg.THEME_INPUT_PROFILE_FILE)

    raw_all = df.get("raw_source_family", df.get("source_family", pd.Series("", index=df.index))).astype(str)
    raw_cluster = df_cluster.get("raw_source_family", df_cluster.get("source_family", pd.Series("", index=df_cluster.index))).astype(str)
    audit_all_count = int(raw_all.eq(cfg.FAMILY_AUDIT_OBSERVATION).sum())
    audit_eligible_count = int(raw_cluster.eq(cfg.FAMILY_AUDIT_OBSERVATION).sum())
    summary = {
        "input_file": str(cfg.UNIFIED_EVENTS_FILE),
        "output_all": str(cfg.THEME_INPUT_ALL_FILE),
        "row_count": int(len(df_cluster)),
        "family_counts": df_cluster["source_family"].value_counts(dropna=False).to_dict(),
        "event_kind_counts": df_cluster["event_kind"].value_counts(dropna=False).to_dict(),
        "audit_rows_before_clustering_filter": audit_all_count,
        "audit_rows_after_clustering_filter": audit_eligible_count,
        "audit_rows_excluded_from_clustering": int(audit_all_count - audit_eligible_count),
        "audit_signal_type_counts_cluster_input": df_cluster.get("audit_signal_type", pd.Series(dtype=object)).value_counts(dropna=False).to_dict(),
        "min_text_length": int(cfg.MIN_TEXT_LENGTH),
        "max_text_chars_for_model": int(max_chars),
        **sampling_summary,
    }
    if sampling_summary.get("poc_sampling_enabled"):
        summary["rows_after_record_sampling_after_text_filter"] = int(len(df_cluster))
        summary["family_counts_after_text_filter"] = df_cluster["source_family"].value_counts(dropna=False).to_dict()
        summary["event_kind_counts_after_text_filter"] = df_cluster["event_kind"].value_counts(dropna=False).to_dict()
        save_json(summary, cfg.POC_SAMPLING_SUMMARY_FILE)
    save_json(summary, cfg.THEME_INPUT_DIR / "theme_input_summary.json")
    log.done("theme input preparation complete")


if __name__ == "__main__":
    main()
