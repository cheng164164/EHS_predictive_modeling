#!/usr/bin/env python3
"""Summarize selected location-period groups using a free local transformer by default.

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
    if not config.FACTS_FILE.exists():
        raise FileNotFoundError(f"Facts file not found: {config.FACTS_FILE}. Run 01_build_location_period_dataset.py first.")
    if not config.EXAMPLES_FILE.exists():
        raise FileNotFoundError(f"Examples file not found: {config.EXAMPLES_FILE}. Run 01_build_location_period_dataset.py first.")
    facts = pd.read_csv(config.FACTS_FILE, low_memory=False)
    examples = pd.read_csv(config.EXAMPLES_FILE, low_memory=False)
    facts["location_id"] = facts["location_id"].astype(str)
    examples["location_id"] = examples["location_id"].astype(str)
    facts["period"] = facts["period"].astype(str)
    examples["period"] = examples["period"].astype(str)
    return facts, examples


def _filter_facts_examples(facts: pd.DataFrame, examples: pd.DataFrame):
    if config.LOCATION_ID:
        facts = facts[facts["location_id"].eq(str(config.LOCATION_ID))].copy()
        examples = examples[examples["location_id"].eq(str(config.LOCATION_ID))].copy()
    if config.LOCATION_CONTAINS:
        needle = str(config.LOCATION_CONTAINS)
        facts = facts[facts["location_path"].str.contains(needle, case=False, regex=False, na=False)].copy()
        examples = examples[examples["location_path"].str.contains(needle, case=False, regex=False, na=False)].copy()
    if config.MIN_REVIEW_SCORE_FOR_SUMMARY > 0:
        facts = facts[facts["review_priority_score"] >= config.MIN_REVIEW_SCORE_FOR_SUMMARY].copy()

    # Summarize only groups where examples were actually retained.
    example_keys = set(zip(examples["location_id"].astype(str), examples["period"].astype(str)))
    facts = facts[facts.apply(lambda r: (str(r["location_id"]), str(r["period"])) in example_keys, axis=1)].copy()

    facts = facts.sort_values(["review_priority_score", "event_count"], ascending=False)
    if config.SUMMARY_MAX_GROUPS and config.SUMMARY_MAX_GROUPS > 0:
        facts = facts.head(config.SUMMARY_MAX_GROUPS).copy()
    return facts, examples


def main() -> None:
    log = ProgressLogger("02_summarize_location_periods")
    ensure_dir(config.SUMMARY_DIR)

    facts, examples = _load_inputs()
    log.log(f"loaded facts={len(facts):,}; examples={len(examples):,}")
    facts, examples = _filter_facts_examples(facts, examples)
    log.log(f"selected groups for summarization={len(facts):,}; backend={config.SUMMARIZER_BACKEND}; model={config.TRANSFORMER_MODEL_NAME}")

    summarizer = get_summarizer()
    log.log(f"initialized summarizer: {getattr(summarizer, 'model_used', 'unknown')}")

    grouped_examples = {k: g for k, g in examples.groupby(["location_id", "period"], dropna=False)}
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

    print(f"Wrote {config.SUMMARY_FILE}", flush=True)
    print(f"Wrote {config.SUMMARY_REVIEW_FILE}", flush=True)
    print(f"Model used: {getattr(summarizer, 'model_used', 'unknown')}", flush=True)
    print(f"Rows summarized: {len(summaries):,}", flush=True)
    log.done("summarization complete")


if __name__ == "__main__":
    main()
