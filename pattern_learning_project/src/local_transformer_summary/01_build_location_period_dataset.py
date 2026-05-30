#!/usr/bin/env python3
"""Build location-period facts and prioritized evidence examples.

Runs without command-line arguments. All settings are in config.py.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from typing import Dict, Tuple

# Allow direct execution.
try:
    from . import config
    from .utils import (
        ProgressLogger,
        blank_fact,
        classify_row,
        compact_text,
        ensure_dir,
        facts_to_frame,
        location_leaf,
        make_event_detail,
        open_text_csv,
        parse_bool,
        parse_dt,
        period_info,
        update_min_max,
    )
except ImportError:  # pragma: no cover
    import config  # type: ignore
    from utils import (  # type: ignore
        ProgressLogger,
        blank_fact,
        classify_row,
        compact_text,
        ensure_dir,
        facts_to_frame,
        location_leaf,
        make_event_detail,
        open_text_csv,
        parse_bool,
        parse_dt,
        period_info,
        update_min_max,
    )

import pandas as pd


def _date_filters():
    min_dt = parse_dt(config.MIN_DATE) if config.MIN_DATE else None
    max_dt = parse_dt(config.MAX_DATE) if config.MAX_DATE else None
    if max_dt and len(str(config.MAX_DATE)) == 10:
        max_dt = max_dt.replace(hour=23, minute=59, second=59)
    return min_dt, max_dt


def _row_in_scope(row: dict, event_dt, min_dt, max_dt) -> bool:
    if event_dt is None:
        return False
    if min_dt and event_dt < min_dt:
        return False
    if max_dt and event_dt > max_dt:
        return False
    location_id = str(row.get("location_id") or "").strip()
    if config.LOCATION_ID and location_id != str(config.LOCATION_ID):
        return False
    location_path = row.get("location_path") or ""
    if config.LOCATION_CONTAINS and str(config.LOCATION_CONTAINS).lower() not in location_path.lower():
        return False
    return True


def scan_facts(min_dt, max_dt):
    log = ProgressLogger("01_build_facts_pass")
    facts: Dict[Tuple[str, str, str, str, str, str], dict] = {}
    scanned = 0
    kept = 0
    with open_text_csv(config.INPUT_FILE) as f:
        reader = csv.DictReader(f)
        for row in reader:
            scanned += 1
            event_dt = parse_dt(row.get("event_date"))
            if not _row_in_scope(row, event_dt, min_dt, max_dt):
                if scanned % config.PROGRESS_EVERY_ROWS == 0:
                    log.log(f"scanned {scanned:,}; kept {kept:,}; groups={len(facts):,}")
                continue
            kept += 1
            location_id = str(row.get("location_id") or "").strip()
            location_path = row.get("location_path") or ""
            location_label = location_leaf(location_path, "Unknown Location")
            period, period_start, period_end = period_info(event_dt, config.PERIOD)
            key = (location_id, location_label, location_path, period, period_start, period_end)
            if key not in facts:
                facts[key] = blank_fact()
            fact = facts[key]
            flags = classify_row(row, include_safety_keywords=config.INCLUDE_SAFETY_KEYWORD_COUNT)

            fact["event_count"] += 1
            fact["text_event_count"] += int(flags["has_text"])
            fact["serious_injury_count"] += int(flags["is_serious"])
            fact["normal_injury_count"] += int(flags["is_normal"])
            fact["near_miss_count"] += int(flags["is_near_miss"])
            fact["hazard_identification_count"] += int(flags["is_hazard"])
            fact["audit_count"] += int(flags["is_audit"])
            fact["unsafe_condition_audit_count"] += int(flags["is_unsafe_condition"])
            fact["unsafe_act_audit_count"] += int(flags["is_unsafe_act"])
            fact["safe_condition_audit_count"] += int(flags["is_safe_condition"])
            fact["safe_act_audit_count"] += int(flags["is_safe_act"])
            fact["task_count"] += int(flags["is_task"])
            fact["open_action_count"] += int(flags["is_task"] and flags["is_open_task"])
            fact["overdue_action_count"] += int(flags["is_task"] and flags["is_overdue_task"])
            fact["completed_or_closed_task_count"] += int(flags["is_completed_task"])
            fact["safety_keyword_text_count"] += int(flags["has_safety_keyword"])
            update_min_max(fact, event_dt)

            if scanned % config.PROGRESS_EVERY_ROWS == 0:
                log.log(f"scanned {scanned:,}; kept {kept:,}; groups={len(facts):,}")
    log.done(f"scanned {scanned:,}; kept {kept:,}; groups={len(facts):,}")
    return facts, scanned, kept


def scan_examples(min_dt, max_dt, target_keys: set[tuple[str, str]]) -> pd.DataFrame:
    log = ProgressLogger("01_build_examples_pass")
    examples = defaultdict(list)
    scanned = 0
    matched = 0
    with open_text_csv(config.INPUT_FILE) as f:
        reader = csv.DictReader(f)
        for row in reader:
            scanned += 1
            event_dt = parse_dt(row.get("event_date"))
            if not _row_in_scope(row, event_dt, min_dt, max_dt):
                if scanned % config.PROGRESS_EVERY_ROWS == 0:
                    log.log(f"scanned {scanned:,}; matched={matched:,}; groups_with_examples={len(examples):,}")
                continue
            location_id = str(row.get("location_id") or "").strip()
            period, _, _ = period_info(event_dt, config.PERIOD)
            target_pair = (location_id, period)
            if target_keys and target_pair not in target_keys:
                if scanned % config.PROGRESS_EVERY_ROWS == 0:
                    log.log(f"scanned {scanned:,}; matched={matched:,}; groups_with_examples={len(examples):,}")
                continue
            matched += 1
            flags = classify_row(row, include_safety_keywords=False)
            if flags["review_priority"] > 11 or not flags["has_text"]:
                if scanned % config.PROGRESS_EVERY_ROWS == 0:
                    log.log(f"scanned {scanned:,}; matched={matched:,}; groups_with_examples={len(examples):,}")
                continue

            event_id = row.get("event_id") or ""
            candidate_key = (flags["review_priority"], event_dt.isoformat(), event_id)
            ex_key = (location_id, period, flags["review_event_type"])
            bucket = examples[ex_key]
            replace_idx = None
            if len(bucket) < config.MAX_EXAMPLES_PER_EVENT_TYPE:
                keep_candidate = True
            else:
                worst_idx, worst_value = max(enumerate(bucket), key=lambda item: item[1][:3])
                keep_candidate = candidate_key < worst_value[:3]
                replace_idx = worst_idx
            if not keep_candidate:
                continue

            location_path = row.get("location_path") or ""
            ex = {
                "event_id": event_id,
                "source_type": row.get("source_type") or "",
                "source_subtype": row.get("source_subtype") or "",
                "source_id": row.get("source_id") or "",
                "event_dt": event_dt.isoformat(sep=" "),
                "period": period,
                "location_id": location_id,
                "location_label": location_leaf(location_path, "Unknown Location"),
                "location_path": location_path,
                "review_event_type": flags["review_event_type"],
                "review_priority": flags["review_priority"],
                "status": row.get("status") or "",
                "category": row.get("category") or "",
                "audit_type": row.get("audit_type") or "",
                "task_source_module": row.get("task_source_module") or "",
                "is_open_task": flags["is_open_task"],
                "is_overdue_task": flags["is_overdue_task"],
                "due_dt": row.get("due_date") or "",
                "completion_dt": row.get("completion_date") or "",
                "any_injury": flags["any_injury"],
                "severe_actual": flags["severe_actual"],
                "fatality": parse_bool(row.get("fatality")),
                "losttime": parse_bool(row.get("losttime")),
                "restrictedtime": parse_bool(row.get("restrictedtime")),
                "inpatient": parse_bool(row.get("inpatient")),
                "emergencyroom": parse_bool(row.get("emergencyroom")),
                "clean_text_full": row.get("clean_text") or "",
                "event_dt_obj": event_dt,
            }
            candidate = (candidate_key[0], candidate_key[1], candidate_key[2], ex)
            if replace_idx is None:
                bucket.append(candidate)
            else:
                bucket[replace_idx] = candidate

            if scanned % config.PROGRESS_EVERY_ROWS == 0:
                log.log(f"scanned {scanned:,}; matched={matched:,}; groups_with_examples={len(examples):,}")

    example_rows = []
    for _, items in examples.items():
        top_items = sorted(items, key=lambda x: x[:3])[: config.MAX_EXAMPLES_PER_EVENT_TYPE]
        for _, _, _, ex in top_items:
            raw_text = ex.pop("clean_text_full", "")
            event_dt_obj = ex.pop("event_dt_obj", None)
            fake_row = {
                "event_id": ex.get("event_id", ""),
                "source_subtype": ex.get("source_subtype", ""),
                "category": ex.get("category", ""),
                "status": ex.get("status", ""),
                "clean_text": raw_text,
            }
            ex["clean_text"] = compact_text(raw_text, 1500)
            ex["event_detail"] = make_event_detail(fake_row, event_dt_obj, {"review_event_type": ex["review_event_type"]}, config.MAX_CHARS_PER_EXAMPLE)
            example_rows.append(ex)

    examples_df = pd.DataFrame(example_rows)
    if not examples_df.empty:
        examples_df = examples_df.sort_values(["location_id", "period", "review_priority", "event_dt"])
    log.done(f"example rows={len(examples_df):,}; matched target rows={matched:,}")
    return examples_df


def main() -> None:
    ensure_dir(config.AGGREGATE_DIR)
    if not config.INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {config.INPUT_FILE}")

    min_dt, max_dt = _date_filters()
    facts, scanned, kept = scan_facts(min_dt, max_dt)
    facts_df = facts_to_frame(facts)
    if facts_df.empty:
        raise SystemExit("No rows after filtering. Check config.MIN_DATE, config.MAX_DATE, and location filters.")

    facts_df.to_csv(config.FACTS_FILE, index=False)

    if config.EXAMPLE_TOP_GROUPS and config.EXAMPLE_TOP_GROUPS > 0:
        target_df = facts_df.head(config.EXAMPLE_TOP_GROUPS)
    else:
        target_df = facts_df
    target_keys = set(zip(target_df["location_id"].astype(str), target_df["period"].astype(str)))
    examples_df = scan_examples(min_dt, max_dt, target_keys)
    examples_df.to_csv(config.EXAMPLES_FILE, index=False)

    rollup = (
        facts_df.groupby(["location_id", "location_label", "location_path"], dropna=False)
        .agg(
            period_count=("period", "nunique"),
            first_event_date=("first_event_date", "min"),
            last_event_date=("last_event_date", "max"),
            event_count=("event_count", "sum"),
            serious_injury_count=("serious_injury_count", "sum"),
            normal_injury_count=("normal_injury_count", "sum"),
            near_miss_count=("near_miss_count", "sum"),
            hazard_identification_count=("hazard_identification_count", "sum"),
            audit_count=("audit_count", "sum"),
            unsafe_condition_audit_count=("unsafe_condition_audit_count", "sum"),
            unsafe_act_audit_count=("unsafe_act_audit_count", "sum"),
            task_count=("task_count", "sum"),
            open_action_count=("open_action_count", "sum"),
            overdue_action_count=("overdue_action_count", "sum"),
            review_priority_score=("review_priority_score", "sum"),
        )
        .reset_index()
        .sort_values(["review_priority_score", "event_count"], ascending=False)
    )
    rollup.to_csv(config.ROLLUP_FILE, index=False)

    print(f"Wrote {config.FACTS_FILE}", flush=True)
    print(f"Wrote {config.EXAMPLES_FILE}", flush=True)
    print(f"Wrote {config.ROLLUP_FILE}", flush=True)
    print(f"Source rows scanned: {scanned:,}; rows kept: {kept:,}", flush=True)
    print(f"Location-period rows: {len(facts_df):,}; example rows: {len(examples_df):,}; example target groups: {len(target_keys):,}", flush=True)


if __name__ == "__main__":
    main()
