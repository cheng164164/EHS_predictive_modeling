"""Configuration for the simplified any-injury risk ranking MVP.

Run from the project root after placing these files under:
    src/injury_risk_classification/

    python src/injury_risk_classification/build_classification_dataset.py
    python src/injury_risk_classification/train_injury_risk_classifier.py
    python src/injury_risk_classification/score_current_site_risk.py

MVP definition
--------------
Row: one site + department + calendar month.
Target: future_any_injury_3m.
Meaning: 1 if the same site/department has at least one injury incident in the
next 3 calendar months, excluding the current month.
Output: a risk score used for ranking, not a hard guarantee that an injury will occur.

Default model behavior
----------------------
- Train baseline and baseline + simple pattern/theme features.
- Use class-weighted LightGBM when available.
- Select an operational top-percent threshold, default top 10%.
- Score the latest month and produce ranked site/department risk output.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

# -----------------------------------------------------------------------------
# Project paths
# -----------------------------------------------------------------------------
# This assumes this file lives in:
#   pattern_learning_project/src/injury_risk_classification/config.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Unsupervised pattern-learning output. Keep this aligned with the unsupervised
# package config. The newer unsupervised scripts write to outputs/ml/hdbscan_patterns.
HDBSCAN_OUTPUT_DIR = OUTPUT_DIR / "ml" / "hdbscan_patterns"
CLUSTERED_PATTERN_RECORDS_PATH = HDBSCAN_OUTPUT_DIR / "final" / "pattern_learning_clustered_records.csv"
CLUSTERED_RECORDS_PATH = CLUSTERED_PATTERN_RECORDS_PATH
REQUIRE_CLUSTERED_RECORDS_FOR_CLUSTER_FEATURES = True

FEATURE_OUTPUT_DIR = OUTPUT_DIR / "ml" / "injury_risk_classification" / "features"
RUN_OUTPUT_DIR = OUTPUT_DIR / "ml" / "injury_risk_classification" / "runs"
PREDICTION_OUTPUT_DIR = OUTPUT_DIR / "ml" / "injury_risk_classification" / "predictions"

# -----------------------------------------------------------------------------
# Simplified MVP target and feature-set switches
# -----------------------------------------------------------------------------
# Supported target types in this package:
#   "any_injury"     -> future_any_injury_{HORIZON_MONTHS}m
#   "severe_actual"  -> future_severe_actual_{HORIZON_MONTHS}m, kept for comparison
TARGET_TYPE = "any_injury"

# Options:
#   "baseline"      -> only operational, audit, task, and injury-history features
#   "with_clusters" -> baseline + simple aggregate/theme pattern features
#   "both"          -> train baseline and with_clusters for comparison
#   "experiments"   -> baseline + small experiment matrix below
#   "all"           -> baseline, with_clusters, and experiment matrix
FEATURE_SET = "baseline"

# Future prediction horizon. 3 months is the recommended MVP horizon.
HORIZON_MONTHS = 3
ROLLING_WINDOWS = [3, 6]
MIN_HISTORY_MONTHS = 6

# Optional as-of date. Set None to use all available records.
# Example: "2026-05-20"
REFERENCE_DATE = "2026-05-20"

# -----------------------------------------------------------------------------
# Simple pattern/theme feature configuration
# -----------------------------------------------------------------------------
# The unsupervised layer can produce: record -> cluster -> theme.
# For the MVP, use broad themes and aggregate pattern activity. Do not use
# detailed per-cluster ID features by default because they add dimensionality and
# can overfit.
PATTERN_FEATURE_LEVELS = ["theme"]
PATTERN_INCLUDE_AGGREGATE_FEATURES = True
PATTERN_INCLUDE_PATTERN_ID_FEATURES = True
PATTERN_INCLUDE_DIVERSITY_FEATURES = True
PATTERN_INCLUDE_OUTLIER_FEATURES = True
PATTERN_INCLUDE_MEMBERSHIP_FEATURES = True
PATTERN_INCLUDE_SEVERE_HISTORY_FEATURES = False
PATTERN_TOP_N_THEMES = 20
PATTERN_TOP_N_CLUSTERS = 0
TOP_N_CLUSTERS = PATTERN_TOP_N_CLUSTERS
PATTERN_MIN_PATTERN_RECORDS = 50
PATTERN_ID_SELECTION = "frequency"
PATTERN_FEATURE_PREFIX = ""


def get_pattern_feature_config() -> dict:
    """Default MVP pattern feature config: aggregate activity + top themes."""
    return {
        "name": "with_clusters",  # kept for backward compatibility with old folder names
        "pattern_levels": list(PATTERN_FEATURE_LEVELS),
        "include_aggregate_features": bool(PATTERN_INCLUDE_AGGREGATE_FEATURES),
        "include_pattern_id_features": bool(PATTERN_INCLUDE_PATTERN_ID_FEATURES),
        "include_diversity_features": bool(PATTERN_INCLUDE_DIVERSITY_FEATURES),
        "include_outlier_features": bool(PATTERN_INCLUDE_OUTLIER_FEATURES),
        "include_membership_features": bool(PATTERN_INCLUDE_MEMBERSHIP_FEATURES),
        "include_severe_history_features": bool(PATTERN_INCLUDE_SEVERE_HISTORY_FEATURES),
        "top_n_themes": int(PATTERN_TOP_N_THEMES),
        "top_n_clusters": int(PATTERN_TOP_N_CLUSTERS),
        "min_pattern_records": int(PATTERN_MIN_PATTERN_RECORDS),
        "id_selection": PATTERN_ID_SELECTION,
        "feature_prefix": PATTERN_FEATURE_PREFIX,
    }


# Small optional experiment matrix. The MVP does not require running this, but it
# lets you verify whether patterns help without creating a large research project.
PATTERN_FEATURE_EXPERIMENTS = [
    {
        "name": "patterns_aggregate_only",
        "pattern_levels": ["theme"],
        "include_aggregate_features": True,
        "include_pattern_id_features": False,
        "include_diversity_features": True,
        "include_outlier_features": True,
        "include_membership_features": True,
        "include_severe_history_features": False,
        "top_n_themes": 0,
        "top_n_clusters": 0,
        "min_pattern_records": PATTERN_MIN_PATTERN_RECORDS,
        "id_selection": "frequency",
        "feature_prefix": "",
    },
    {
        "name": "themes_aggregate_plus_top_ids",
        "pattern_levels": ["theme"],
        "include_aggregate_features": True,
        "include_pattern_id_features": True,
        "include_diversity_features": True,
        "include_outlier_features": True,
        "include_membership_features": True,
        "include_severe_history_features": False,
        "top_n_themes": 20,
        "top_n_clusters": 0,
        "min_pattern_records": PATTERN_MIN_PATTERN_RECORDS,
        "id_selection": "frequency",
        "feature_prefix": "",
    },
]


def get_pattern_feature_experiments() -> list[dict]:
    """Return independent copies so scripts can add fixed IDs safely."""
    return [deepcopy(x) for x in PATTERN_FEATURE_EXPERIMENTS]

# -----------------------------------------------------------------------------
# Train/validation/test design
# -----------------------------------------------------------------------------
TEST_MONTHS = 6
CV_SPLITS = 4
MAX_TRAIN_ROWS = None
RUN_ID = None
SAVE_FEATURE_DATASETS_DURING_TRAINING = False

# -----------------------------------------------------------------------------
# Model selection
# -----------------------------------------------------------------------------
MODEL_TYPE = "auto"  # auto, lightgbm, hist_gradient_boosting, logistic_regression, sgd_logistic
RANDOM_STATE = 42
MIN_CATEGORY_FREQUENCY = 5

LIGHTGBM_PARAMS = {
    "objective": "binary",
    "n_estimators": 500,
    "learning_rate": 0.04,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 30,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "class_weight": "balanced",
    "n_jobs": -1,
    "verbose": -1,
}

HIST_GRADIENT_BOOSTING_PARAMS = {
    "max_iter": 300,
    "learning_rate": 0.05,
    "max_leaf_nodes": 31,
    "l2_regularization": 0.1,
    "early_stopping": True,
}

LOGISTIC_REGRESSION_PARAMS = {
    "max_iter": 500,
    "class_weight": "balanced",
    "solver": "saga",
    "n_jobs": -1,
}

SGD_LOGISTIC_PARAMS = {
    "loss": "log_loss",
    "penalty": "elasticnet",
    "alpha": 0.0001,
    "l1_ratio": 0.05,
    "class_weight": "balanced",
    "max_iter": 200,
    "tol": 1e-2,
    "n_jobs": -1,
}

# -----------------------------------------------------------------------------
# Operational ranking threshold
# -----------------------------------------------------------------------------
# For the MVP, use the model primarily as a ranking tool. This threshold means
# "flag the top 10% highest-risk site/departments".
THRESHOLD_STRATEGY = "top_percent"  # top_percent, f2, fixed
TOP_PERCENT_THRESHOLD = 0.10
FIXED_THRESHOLD = 0.50

# -----------------------------------------------------------------------------
# Current-risk scoring configuration
# -----------------------------------------------------------------------------
MODEL_DIR = None
SCORE_FEATURE_SET = "with_clusters"
CURRENT_RISK_OUTPUT_FILE = PREDICTION_OUTPUT_DIR / "current_any_injury_risk_scores.csv"
PREVIEW_ROWS = 5000

# -----------------------------------------------------------------------------
# Validation, feature-audit, dashboard, and operational workflow outputs
# -----------------------------------------------------------------------------
# Save the exact raw feature tables used for model fitting before sklearn one-hot
# encoding/scaling. These files are useful for QA, stakeholder review, and Power BI.
SAVE_MODEL_INPUT_FEATURE_TABLES = True
MODEL_INPUT_PREVIEW_ROWS = 10000

# Leakage validation runs during training for every feature set. Set
# FAIL_ON_LEAKAGE=True only after you are comfortable with the checks because the
# validator is intentionally conservative and may flag features for review.
LEAKAGE_VALIDATION_ENABLED = True
FAIL_ON_LEAKAGE = False
LEAKAGE_CORRELATION_WARNING_THRESHOLD = 0.98
LEAKAGE_AUC_WARNING_THRESHOLD = 0.995

# Dashboard/explanation settings used by score_current_site_risk.py.
SAVE_DASHBOARD_OUTPUTS = True
DASHBOARD_TOP_N_DRIVERS = 6
DASHBOARD_TOP_N_THEMES = 5
DASHBOARD_ROLLING_WINDOW_MONTHS = 3
DASHBOARD_OUTPUT_FILE = PREDICTION_OUTPUT_DIR / "current_any_injury_risk_dashboard.csv"
OPERATIONAL_REVIEW_QUEUE_OUTPUT_FILE = PREDICTION_OUTPUT_DIR / "operational_review_queue.csv"
DASHBOARD_TIER_SUMMARY_OUTPUT_FILE = PREDICTION_OUTPUT_DIR / "risk_tier_summary.csv"
DASHBOARD_SITE_ROLLUP_OUTPUT_FILE = PREDICTION_OUTPUT_DIR / "site_risk_rollup.csv"

# Operational review tiers. The score script ranks every row, assigns tiers, and
# writes a queue for EHS follow-up.
RISK_TIER_CRITICAL_FRACTION = 0.05
RISK_TIER_HIGH_FRACTION = 0.10
RISK_TIER_WATCHLIST_FRACTION = 0.25
OPERATIONAL_QUEUE_TIERS = ["Critical", "High", "Watchlist"]
CRITICAL_REVIEW_DUE_DAYS = 7
HIGH_REVIEW_DUE_DAYS = 14
WATCHLIST_REVIEW_DUE_DAYS = 30
MONITOR_REVIEW_DUE_DAYS = 60
