#!/usr/bin/env python3
"""Create interpretable theme catalog and representative examples.

This step does not use an LLM. It labels clusters using top terms, source/event
mix, counts, and representative records closest/highest-confidence in each theme.
"""
from __future__ import annotations

try:
    import config as cfg
    from theme_utils import (
        ProgressLogger,
        compact_text,
        ensure_dir,
        read_csv,
        representative_text,
        safe_value_counts,
        save_json,
        top_terms_from_texts,
        write_csv,
    )
except ImportError:  # pragma: no cover
    from . import config as cfg
    from .theme_utils import (
        ProgressLogger,
        compact_text,
        ensure_dir,
        read_csv,
        representative_text,
        safe_value_counts,
        save_json,
        top_terms_from_texts,
        write_csv,
    )

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


def _family_label(family: str) -> str:
    return {
        cfg.FAMILY_INCIDENT_HAZARD: "Incident / hazard / near-miss / injury theme",
        getattr(cfg, "FAMILY_AUDIT_RISK", "audit_risk"): "Audit risk / unsafe finding theme",
        getattr(cfg, "FAMILY_AUDIT_POSITIVE", "audit_positive"): "Audit positive-control / safe observation theme",
        cfg.FAMILY_AUDIT_OBSERVATION: "Audit / observation theme",
        cfg.FAMILY_TASK_ACTION: "Task / corrective-action theme",
    }.get(family, family)


def _make_theme_name(top_terms: str, family: str, theme_id: str) -> str:
    if theme_id.endswith("UNASSIGNED"):
        return "Unassigned / low-confidence records"
    terms = [compact_text(t) for t in str(top_terms or "").split("|")]
    terms = [t for t in terms if t]
    # Prefer phrases first; use short readable label.
    phrase_terms = [t for t in terms if " " in t]
    chosen = phrase_terms[: int(cfg.LABEL_TOP_TERM_COUNT)]
    if len(chosen) < int(cfg.LABEL_TOP_TERM_COUNT):
        chosen += [t for t in terms if t not in chosen][: int(cfg.LABEL_TOP_TERM_COUNT) - len(chosen)]
    if not chosen:
        return f"{_family_label(family)} {theme_id}"
    return " / ".join(chosen)


def _count_event_kinds(g: pd.DataFrame) -> dict:
    counts = g["event_kind"].fillna("unknown").astype(str).value_counts().to_dict()
    out = {col: int(counts.get(kind, 0)) for kind, col in EVENT_KIND_COLUMNS.items()}
    out["injury_count"] = out.get("serious_injury_count", 0) + out.get("normal_injury_count", 0)
    out["audit_count"] = int(g["source_type"].fillna("").astype(str).str.lower().eq("audit").sum())
    out["task_count"] = int(g["source_type"].fillna("").astype(str).str.lower().eq("task").sum())
    out["open_action_count"] = int(g.get("is_open_task", pd.Series(False, index=g.index)).fillna(False).astype(bool).sum()) if "is_open_task" in g else 0
    out["overdue_action_count"] = int(g.get("is_overdue_task", pd.Series(False, index=g.index)).fillna(False).astype(bool).sum()) if "is_overdue_task" in g else 0
    return out


def _representatives(g: pd.DataFrame, theme_id: str, mode: str, n: int) -> list[dict]:
    if mode == "representative":
        sort_cols = [c for c in ["theme_confidence", "review_priority", "event_date"] if c in g.columns]
        ascending = [False if c in {"theme_confidence", "review_priority"} else True for c in sort_cols]
        sample = g.sort_values(sort_cols, ascending=ascending).head(n) if sort_cols else g.head(n)
    else:
        sample = g.sample(n=min(n, len(g)), random_state=int(cfg.RANDOM_STATE)) if len(g) > 0 else g
    rows = []
    for rank, (_, r) in enumerate(sample.iterrows(), start=1):
        rows.append({
            "theme_id": theme_id,
            "source_family": r.get("source_family", ""),
            "example_type": mode,
            "example_rank": rank,
            "event_id": r.get("event_id", ""),
            "event_date": r.get("event_date", ""),
            "location_id": r.get("location_id", ""),
            "location_path": r.get("location_path", ""),
            "source_type": r.get("source_type", ""),
            "event_kind": r.get("event_kind", ""),
            "category": r.get("category", ""),
            "status": r.get("status", ""),
            "theme_confidence": r.get("theme_confidence", np.nan),
            "assignment_type": r.get("assignment_type", ""),
            "title": compact_text(r.get("title", ""), 250),
            "representative_text": representative_text(r, int(cfg.MAX_REPRESENTATIVE_TEXT_CHARS)),
            "clean_text": compact_text(r.get("clean_text", ""), int(cfg.MAX_REPRESENTATIVE_TEXT_CHARS)),
        })
    return rows


def main() -> None:
    log = ProgressLogger("04_label_theme_clusters")
    ensure_dir(cfg.THEME_CATALOG_DIR)

    if not cfg.THEME_ASSIGNMENTS_FILE.exists():
        raise FileNotFoundError(f"Missing assignments file: {cfg.THEME_ASSIGNMENTS_FILE}. Run 03_cluster_by_family.py first.")
    if not cfg.THEME_INPUT_ALL_FILE.exists():
        raise FileNotFoundError(f"Missing theme input file: {cfg.THEME_INPUT_ALL_FILE}. Run 01_prepare_theme_text.py first.")

    log.log("reading assignments and theme input")
    assignments = read_csv(cfg.THEME_ASSIGNMENTS_FILE)
    events = read_csv(cfg.THEME_INPUT_ALL_FILE)
    merge_keys = ["theme_row_id", "event_id"] if "theme_row_id" in assignments.columns and "theme_row_id" in events.columns else ["event_id"]
    df = assignments.merge(events, on=merge_keys, how="left", suffixes=("", "_event"))

    # Resolve duplicate columns after merge.
    for col in ["source_type", "event_kind", "event_date", "location_id", "location_path", "title", "clean_text", "theme_text", "category", "status", "raw_source_family", "audit_signal_type", "audit_cluster_family", "audit_cluster_eligible", "audit_cluster_exclusion_reason", "audit_has_positive_keyword", "review_priority", "is_open_task", "is_overdue_task"]:
        event_col = f"{col}_event"
        if col not in df.columns and event_col in df.columns:
            df[col] = df[event_col]
        elif event_col in df.columns:
            df[col] = df[col].where(df[col].notna(), df[event_col])

    log.log(f"building catalog for {df['theme_id'].nunique():,} themes")
    catalog_rows = []
    example_rows = []
    for theme_id, g in df.groupby("theme_id", dropna=False):
        g = g.copy()
        if len(g) < int(cfg.MIN_THEME_SIZE_FOR_CATALOG) and not str(theme_id).endswith("UNASSIGNED"):
            continue
        family = compact_text(g["source_family"].dropna().iloc[0]) if "source_family" in g and g["source_family"].notna().any() else "unknown"
        audit_fams = {cfg.FAMILY_AUDIT_OBSERVATION, getattr(cfg, "FAMILY_AUDIT_RISK", "audit_risk"), getattr(cfg, "FAMILY_AUDIT_POSITIVE", "audit_positive")}
        extra_stopwords = set(getattr(cfg, "AUDIT_CLUSTER_STOPWORDS", set())) if family in audit_fams else set()
        top_terms = top_terms_from_texts(
            g.get("theme_text", g.get("clean_text", pd.Series("", index=g.index))).fillna(""),
            int(cfg.TOP_TERMS_PER_THEME),
            extra_stopwords=extra_stopwords,
        )
        theme_name = _make_theme_name(top_terms, family, str(theme_id))
        count_cols = _count_event_kinds(g)
        assigned_strong = int(g["assignment_type"].astype(str).eq("strong_cluster").sum()) if "assignment_type" in g else 0
        assigned_weak = int(g["assignment_type"].astype(str).eq("weak_nearest_theme").sum()) if "assignment_type" in g else 0
        unassigned = int(g["assignment_type"].astype(str).str.contains("unassigned", case=False, na=False).sum()) if "assignment_type" in g else 0
        dates = pd.to_datetime(g.get("event_date", pd.Series(pd.NaT, index=g.index)), errors="coerce")
        rep_rows = _representatives(g, str(theme_id), "representative", int(cfg.REPRESENTATIVE_EXAMPLES_PER_THEME))
        rand_rows = _representatives(g, str(theme_id), "random", int(cfg.RANDOM_EXAMPLES_PER_THEME))
        example_rows.extend(rep_rows)
        example_rows.extend(rand_rows)
        representative_event_ids = ";".join([str(x["event_id"]) for x in rep_rows[: int(cfg.REPRESENTATIVE_EXAMPLES_PER_THEME)]])
        representative_texts = " || ".join([str(x["representative_text"]) for x in rep_rows[:5]])
        row = {
            "theme_id": theme_id,
            "source_family": family,
            "theme_name": theme_name,
            "theme_description": f"{_family_label(family)} characterized by: {top_terms}",
            "theme_size": int(len(g)),
            "strong_cluster_count": assigned_strong,
            "weak_nearest_theme_count": assigned_weak,
            "unassigned_count": unassigned,
            "mean_theme_confidence": float(pd.to_numeric(g.get("theme_confidence", pd.Series(np.nan, index=g.index)), errors="coerce").mean()),
            "min_event_date": str(dates.min()) if dates.notna().any() else "",
            "max_event_date": str(dates.max()) if dates.notna().any() else "",
            "source_type_mix": safe_value_counts(g.get("source_type", pd.Series("", index=g.index)), 10),
            "event_kind_mix": safe_value_counts(g.get("event_kind", pd.Series("", index=g.index)), 15),
            "audit_signal_type_mix": safe_value_counts(g.get("audit_signal_type", pd.Series("", index=g.index)), 12),
            "audit_exclusion_reason_mix": safe_value_counts(g.get("audit_cluster_exclusion_reason", pd.Series("", index=g.index)), 8),
            "category_mix": safe_value_counts(g.get("category", pd.Series("", index=g.index)), 10),
            "top_locations": safe_value_counts(g.get("location_path", pd.Series("", index=g.index)), 8),
            "top_terms": top_terms,
            "representative_event_ids": representative_event_ids,
            "representative_texts": compact_text(representative_texts, 4000),
            **count_cols,
        }
        catalog_rows.append(row)

    catalog = pd.DataFrame(catalog_rows)
    if len(catalog) > 0:
        catalog = catalog.sort_values(["source_family", "theme_size"], ascending=[True, False]).reset_index(drop=True)
    examples = pd.DataFrame(example_rows)
    if len(examples) > 0:
        examples = examples.sort_values(["source_family", "theme_id", "example_type", "example_rank"])

    write_csv(catalog, cfg.THEME_CATALOG_FILE)
    write_csv(examples, cfg.THEME_REPRESENTATIVE_EXAMPLES_FILE)

    review_cols = [
        "theme_id", "source_family", "theme_name", "theme_size", "mean_theme_confidence",
        "event_kind_mix", "source_type_mix", "category_mix", "audit_signal_type_mix", "audit_exclusion_reason_mix", "top_locations", "top_terms",
        "serious_injury_count", "normal_injury_count", "near_miss_count", "hazard_identification_count",
        "audit_unsafe_condition_count", "audit_unsafe_act_count", "audit_safe_condition_count", "audit_safe_act_count", "audit_positive_observation_count", "task_overdue_count", "task_open_count",
        "representative_event_ids", "representative_texts",
    ]
    review_cols = [c for c in review_cols if c in catalog.columns]
    write_csv(catalog[review_cols], cfg.THEME_CATALOG_REVIEW_FILE)

    save_json({
        "assignment_file": str(cfg.THEME_ASSIGNMENTS_FILE),
        "theme_catalog_file": str(cfg.THEME_CATALOG_FILE),
        "theme_count": int(len(catalog)),
        "example_count": int(len(examples)),
    }, cfg.THEME_CATALOG_DIR / "theme_catalog_summary.json")
    log.done("theme catalog complete")


if __name__ == "__main__":
    main()
