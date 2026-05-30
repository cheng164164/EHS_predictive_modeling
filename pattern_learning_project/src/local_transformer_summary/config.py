"""Central configuration for the local transformer safety review project.

All scripts in this folder run without command-line arguments. Change parameters
here, or set SAFETY_EVENT_INPUT for the input file path.
"""
from __future__ import annotations

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_FILE = PROJECT_ROOT / "outputs" / "audits_tasks_modeling" / "00_unified_text_events" / "safety_text_event.csv.gz"
INPUT_FILE = Path(os.environ.get("SAFETY_EVENT_INPUT", str(DEFAULT_INPUT_FILE)))
OUTPUT_DIR = PROJECT_ROOT / "outputs"/ "local_transformer_summary"
PROFILE_DIR = OUTPUT_DIR / "profile"
AGGREGATES_DIR = OUTPUT_DIR / "aggregates"
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
MAX_EXAMPLES_PER_EVENT_TYPE = 5
MAX_CHARS_PER_EXAMPLE = 500
# Build evidence examples for only the top N location-period groups by review score.
# Set to 0 for all groups. Top groups are usually enough for first review.
EXAMPLE_TOP_GROUPS = 250
INCLUDE_SAFETY_KEYWORD_COUNT = False
PROGRESS_EVERY_ROWS = 250_000

# Summary settings.
# "transformers" is the default free local transformer backend.
# "extractive" is deterministic and does not require any model package.
SUMMARIZER_BACKEND = "transformers"
ALLOW_EXTRACTIVE_FALLBACK = True
TRANSFORMER_MODEL_NAME = "sshleifer/distilbart-cnn-12-6"
TRANSFORMER_TASK = "summarization"
TRANSFORMER_DEVICE = -1  # -1 means CPU; 0 means first GPU if available
TRANSFORMER_LOCAL_FILES_ONLY = False  # True requires the model to be already cached locally
PRINT_TRANSFORMER_DIAGNOSTICS = True
TRANSFORMER_BATCH_SIZE = 4
TRANSFORMER_MAX_INPUT_CHARS = 3500
TRANSFORMER_MAX_SUMMARY_TOKENS = 150
TRANSFORMER_MIN_SUMMARY_TOKENS = 25

# Summarize top N highest-priority location-period rows.
# Set to 0 for all facts rows that have evidence examples.
SUMMARY_MAX_GROUPS = 150
MIN_REVIEW_SCORE_FOR_SUMMARY = 0.0
MAX_CONTEXT_CHARS_PER_GROUP = 12000
SUMMARY_PROGRESS_EVERY_GROUPS = 10

# CSV output filenames. Normally you do not need to edit these.
FACTS_FILE = AGGREGATES_DIR / f"location_period_facts_{PERIOD}.csv"
EXAMPLES_FILE = AGGREGATES_DIR / f"location_period_event_examples_{PERIOD}.csv"
LOCATION_ROLLUP_FILE = AGGREGATES_DIR / f"location_rollup_from_{PERIOD}.csv"

SUMMARY_FILE = SUMMARY_DIR / "location_period_summaries.csv"
SUMMARY_REVIEW_FILE = SUMMARY_DIR / "location_period_summaries_review.csv"
