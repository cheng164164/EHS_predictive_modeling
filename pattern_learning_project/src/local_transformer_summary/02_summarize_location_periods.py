#!/usr/bin/env python3
"""Summarize selected location-period groups.

Runs without command-line arguments. All settings are in config.py.
"""
from __future__ import annotations

# Allow direct execution.
try:
    from . import config
    from .transformer_summarizer import get_summarizer
    from .utils import ProgressLogger, SUMMARY_FIELDS, ensure_dir
except ImportError:  # pragma: no cover
    import config  # type: ignore
    from transformer_summarizer import get_summarizer  # type: ignore
    from utils import ProgressLogger, SUMMARY_FIELDS, ensure_dir  # type: ignore

import pandas as pd


def _load_inputs():
    missing = []
    if not config.FACTS_FILE.exists():
        missing.append(("facts", config.FACTS_FILE))
    if not config.EXAMPLES_FILE.exists():
        missing.append(("examples", config.EXAMPLES_FILE))

    if missing:
        print("\nRequired input files for summarization are missing:", flush=True)
        for name, path in missing:
            print(f"  - {name}: {path}", flush=True)
        print("\nThis usually means one of two things:", flush=True)
        print("  1. Step 01 has not been run yet.", flush=True)
        print("  2. config.py is pointing to a different output folder than step 01 used.", flush=True)
        print("\nRun this first:", flush=True)
        print("  python pattern_learning_project/src/local_transformer_summary/01_build_location_period_dataset.py", flush=True)
        print("\nCurrent configured folders:", flush=True)
        print(f"  OUTPUT_DIR: {config.OUTPUT_DIR}", flush=True)
        print(f"  AGGREGATE_DIR: {config.AGGREGATE_DIR}", flush=True)
        print(f"  SUMMARY_DIR: {config.SUMMARY_DIR}", flush=True)
        raise FileNotFoundError("Missing required aggregate input files for summarization.")

    facts = pd.read_csv(config.FACTS_FILE, low_memory=False)
    examples = pd.read_csv(config.EXAMPLES_FILE, low_memory=False)
    facts["location_id"] = facts["location_id"].astype(str)
    facts["period"] = facts["period"].astype(str)
    if not examples.empty:
        examples["location_id"] = examples["location_id"].astype(str)
        examples["period"] = examples["period"].astype(str)
    return facts, examples


def _filter_facts_examples(facts: pd.DataFrame, examples: pd.DataFrame):
    if config.LOCATION_ID:
        facts = facts[facts["location_id"].eq(str(config.LOCATION_ID))].copy()
        examples = examples[examples["location_id"].eq(str(config.LOCATION_ID))].copy() if not examples.empty else examples
    if config.LOCATION_CONTAINS:
        needle = str(config.LOCATION_CONTAINS)
        facts = facts[facts["location_path"].str.contains(needle, case=False, regex=False, na=False)].copy()
        examples = examples[examples["location_path"].str.contains(needle, case=False, regex=False, na=False)].copy() if not examples.empty else examples
    if config.MIN_REVIEW_SCORE_FOR_SUMMARY > 0:
        facts = facts[facts["review_priority_score"] >= config.MIN_REVIEW_SCORE_FOR_SUMMARY].copy()

    # Old behavior required examples and silently dropped many locations. New
    # behavior keeps all facts rows by default and produces empty/no-sampled-
    # evidence messages where examples were not retained.
    if getattr(config, "SUMMARY_REQUIRE_EXAMPLES", False) and not examples.empty:
        example_keys = set(zip(examples["location_id"].astype(str), examples["period"].astype(str)))
        facts = facts[facts.apply(lambda r: (str(r["location_id"]), str(r["period"])) in example_keys, axis=1)].copy()

    facts = facts.sort_values(["review_priority_score", "event_count"], ascending=False)
    if config.SUMMARY_MAX_GROUPS and config.SUMMARY_MAX_GROUPS > 0:
        facts = facts.head(config.SUMMARY_MAX_GROUPS).copy()
    return facts, examples


def _write_coverage_report(facts: pd.DataFrame, examples: pd.DataFrame, selected: pd.DataFrame) -> None:
    ensure_dir(config.SUMMARY_DIR)
    if examples.empty:
        groups_with_examples = 0
        locs_with_examples = 0
    else:
        groups_with_examples = examples[["location_id", "period"]].drop_duplicates().shape[0]
        locs_with_examples = examples["location_id"].nunique()
    report = pd.DataFrame(
        [
            {"metric": "facts_location_period_rows", "value": len(facts)},
            {"metric": "facts_unique_locations", "value": facts["location_id"].nunique() if not facts.empty else 0},
            {"metric": "examples_rows", "value": len(examples)},
            {"metric": "example_location_period_groups", "value": groups_with_examples},
            {"metric": "example_unique_locations", "value": locs_with_examples},
            {"metric": "selected_summary_rows", "value": len(selected)},
            {"metric": "selected_unique_locations", "value": selected["location_id"].nunique() if not selected.empty else 0},
            {"metric": "summary_max_groups_config", "value": config.SUMMARY_MAX_GROUPS},
            {"metric": "example_top_groups_config", "value": config.EXAMPLE_TOP_GROUPS},
            {"metric": "summary_require_examples_config", "value": getattr(config, "SUMMARY_REQUIRE_EXAMPLES", False)},
            {"metric": "summarizer_backend", "value": config.SUMMARIZER_BACKEND},
            {"metric": "transformer_model", "value": config.TRANSFORMER_MODEL_NAME},
        ]
    )
    report.to_csv(config.SUMMARY_COVERAGE_FILE, index=False)


def main() -> None:
    log = ProgressLogger("02_summarize_location_periods")
    ensure_dir(config.SUMMARY_DIR)

    all_facts, examples = _load_inputs()
    log.log(f"loaded facts={len(all_facts):,}; examples={len(examples):,}")
    facts, examples = _filter_facts_examples(all_facts, examples)
    log.log(
        f"selected groups for summarization={len(facts):,}; "
        f"unique_locations={facts['location_id'].nunique() if not facts.empty else 0:,}; "
        f"backend={config.SUMMARIZER_BACKEND}; model={config.TRANSFORMER_MODEL_NAME}"
    )

    summarizer = get_summarizer()
    log.log(f"initialized summarizer: {getattr(summarizer, 'model_used', 'unknown')}")

    grouped_examples = {k: g for k, g in examples.groupby(["location_id", "period"], dropna=False)} if not examples.empty else {}
    rows = []
    total = len(facts)

    for i, fact in facts.reset_index(drop=True).iterrows():
        key = (str(fact["location_id"]), str(fact["period"]))
        group_examples = grouped_examples.get(key, pd.DataFrame())
        summary = summarizer.summarize(fact, group_examples)
        row = fact.to_dict()
        row.update(summary)
        row["evidence_event_ids"] = ";".join(group_examples.get("event_id", pd.Series(dtype=str)).astype(str).head(25).tolist()) if not group_examples.empty else ""
        row["evidence_record_count_used"] = int(len(group_examples))
        rows.append(row)

        current = i + 1
        if current == 1 or current % config.SUMMARY_PROGRESS_EVERY_GROUPS == 0 or current == total:
            loc = str(fact.get("location_label", ""))[:60]
            log.log(f"summarized {current:,}/{total:,} groups; latest={loc} period={fact.get('period', '')}")

    summaries = pd.DataFrame(rows)
    summaries.to_csv(config.SUMMARY_FILE, index=False)

    lead_cols = [
        "location_id", "location_label", "location_path", "period", "period_start", "period_end",
        "review_priority_score", "event_count", "serious_injury_count", "normal_injury_count",
        "near_miss_count", "hazard_identification_count", "audit_count",
        "unsafe_condition_audit_count", "unsafe_act_audit_count", "task_count",
        "open_action_count", "overdue_action_count",
    ]
    compact_cols = [c for c in lead_cols + SUMMARY_FIELDS + ["evidence_record_count_used", "evidence_event_ids", "summary_model_used"] if c in summaries.columns]
    summaries[compact_cols].to_csv(config.SUMMARY_REVIEW_FILE, index=False)
    _write_coverage_report(all_facts, examples, facts)

    print(f"Wrote {config.SUMMARY_FILE}", flush=True)
    print(f"Wrote {config.SUMMARY_REVIEW_FILE}", flush=True)
    print(f"Wrote {config.SUMMARY_COVERAGE_FILE}", flush=True)
    print(f"Model used: {getattr(summarizer, 'model_used', 'unknown')}", flush=True)
    print(f"Rows summarized: {len(summaries):,}", flush=True)
    print(f"Unique locations summarized: {summaries['location_id'].nunique() if not summaries.empty else 0:,}", flush=True)
    log.done("summarization complete")


if __name__ == "__main__":
    main()
