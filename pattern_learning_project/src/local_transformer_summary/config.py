"""Central configuration for the local safety review summarization project.

All scripts in this folder run without command-line arguments. Change parameters
here, or set SAFETY_EVENT_INPUT for the input file path.
"""
from __future__ import annotations

import os
from pathlib import Path

# Project paths.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "audits_tasks_modeling"
    / "00_unified_text_events"
    / "safety_text_event.csv.gz"
)
INPUT_FILE = Path(os.environ.get("SAFETY_EVENT_INPUT", str(DEFAULT_INPUT_FILE)))

# Keep this project isolated under outputs/local_transformer_summary.
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "local_transformer_summary"
PROFILE_DIR = OUTPUT_DIR / "profile"
AGGREGATE_DIR = OUTPUT_DIR / "aggregates"
SUMMARY_DIR = OUTPUT_DIR / "summaries"
LOG_DIR = OUTPUT_DIR / "logs"

# Main time/location filters.
# The source file may contain old/future placeholder dates, so keep this configurable.
MIN_DATE = "1990-01-01"
MAX_DATE = "2026-12-31"
PERIOD = "Y"  # Y, Q, M, or W
LOCATION_ID = None  # example: "1613.0"; None means all locations
LOCATION_CONTAINS = None  # example: "Tianjin"; None means all locations

# Evidence extraction settings.
# Use all location-period groups by default. This is why the summary output will
# now cover all locations from the facts table rather than only the top 150.
MAX_EXAMPLES_PER_EVENT_TYPE = 3
MAX_CHARS_PER_EXAMPLE = 450
EXAMPLE_TOP_GROUPS = 0  # 0 means build examples for all location-period groups
INCLUDE_SAFETY_KEYWORD_COUNT = False
PROGRESS_EVERY_ROWS = 250_000

# Summary settings.
# "hybrid" is the recommended default:
#   1. Build deterministic, structured evidence summaries from sampled records.
#   2. Optionally ask a free local transformer to polish/condense section text.
#   3. Reject weak transformer output and keep the structured summary.
# This avoids the prompt leakage and repeated generic text seen with pure BART.
SUMMARIZER_BACKEND = "hybrid"  # hybrid, structured, transformers, extractive
ALLOW_EXTRACTIVE_FALLBACK = True

# FLAN-T5 is instruction-tuned and is usually better for this task than a CNN/news
# summarizer such as distilbart-cnn. CPU is OK but can be slow.
TRANSFORMER_MODEL_NAME = "google/flan-t5-base"
TRANSFORMER_TASK = "text2text-generation"
TRANSFORMER_DEVICE = -1  # -1 means CPU; 0 means first GPU if available
TRANSFORMER_LOCAL_FILES_ONLY = False  # True requires the model to be cached locally
PRINT_TRANSFORMER_DIAGNOSTICS = True
TRANSFORMER_BATCH_SIZE = 2
TRANSFORMER_MAX_INPUT_CHARS = 2800
TRANSFORMER_MAX_SUMMARY_TOKENS = 140
TRANSFORMER_MIN_SUMMARY_TOKENS = 20

# For full all-location runs, local transformer generation can be slow. The hybrid
# summarizer always creates structured summaries for every row. Set this to a
# positive number to let the transformer polish only the top N groups; set to 0 to
# apply transformer polishing to every group with evidence.
HYBRID_TRANSFORMER_MAX_GROUPS = 150
HYBRID_TRANSFORMER_FIELDS = {
    "unsafe_conditions_summary",
    "near_miss_summary",
    "hazards_summary",
    "audits_summary",
    "actions_summary",
}

# Summarize all facts rows by default. Set SUMMARY_MAX_GROUPS to a positive number
# if you want a small high-priority sample for testing.
SUMMARY_MAX_GROUPS = 0  # 0 means all rows after filtering
MIN_REVIEW_SCORE_FOR_SUMMARY = 0.0
SUMMARY_REQUIRE_EXAMPLES = False
MAX_CONTEXT_CHARS_PER_GROUP = 12000
SUMMARY_PROGRESS_EVERY_GROUPS = 50
STRUCTURED_SUMMARY_MAX_EVENTS_PER_SECTION = 5
STRUCTURED_SUMMARY_MAX_CHARS = 1600

# CSV output filenames. Normally you do not need to edit these.
FACTS_FILE = AGGREGATE_DIR / f"location_period_facts_{PERIOD}.csv"
EXAMPLES_FILE = AGGREGATE_DIR / f"location_period_event_examples_{PERIOD}.csv"
ROLLUP_FILE = AGGREGATE_DIR / f"location_rollup_from_{PERIOD}.csv"
SUMMARY_FILE = SUMMARY_DIR / "location_period_summaries.csv"
SUMMARY_REVIEW_FILE = SUMMARY_DIR / "location_period_summaries_review.csv"
SUMMARY_COVERAGE_FILE = SUMMARY_DIR / "summary_coverage_report.csv"
