"""Configuration for the injury-similarity machine-learning workflow.

This module intentionally keeps every project path and tunable modeling
parameter in one place. The rest of the code imports values from here instead
of hardcoding constants in multiple scripts.

Workflow reminder
-----------------
The best-practice workflow is:

1. Temporal validation
   - Fit a validation-only TF-IDF + nearest-neighbor model on older injury
     records.
   - Evaluate that model on newer held-out injury records.

2. Final production model fitting
   - After reviewing validation results, fit the final model on all available
     historical injury records.

3. Prediction
   - Load only the saved final production model and score near-miss/hazard
     candidate records.

Most parameters below control either feature construction, validation splitting,
retrieval depth, or threshold calibration. Start with the defaults. Tune only
when validation outputs show a clear reason to do so.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

# Random seed used by all random sampling logic in the ML workflow.
#
# Where it is used:
# - temporal split fallback when there are not enough dated records
# - background similarity sampling for threshold calibration
# - leave-one-out sampling when the reference library is very large
#
# How to tune:
# - Keep fixed for reproducible validation metrics.
# - Change only if you want to test whether metrics are stable across random
#   samples. In that case, run validation with several seeds and compare.
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Retrieval behavior
# ---------------------------------------------------------------------------

# Number of historical injury records returned for each query record.
#
# Business meaning:
# - A query is a near-miss/hazard/new incident candidate.
# - The model retrieves the top K most similar historical injury cases.
#
# How to tune:
# - 3: easier for reviewers; fewer rows in output; may miss useful context.
# - 5: good default for EHS review workflows.
# - 10: better for analysis and dashboards; more records to inspect.
# - Higher than 10 is usually noisy unless you are building a broad evidence
#   panel rather than a focused recommendation.
DEFAULT_TOP_K = 5

# ---------------------------------------------------------------------------
# Temporal validation split
# ---------------------------------------------------------------------------

# Fraction of dated historical injury records used as the older training/reference
# library during temporal holdout validation.
#
# Example:
# - 0.80 means the oldest 80% of dated injury records become the validation
#   training/reference library, and the newest 20% are treated as future held-out
#   queries.
#
# How to tune:
# - 0.70 gives a larger holdout set and more reliable validation metrics, but a
#   smaller reference library.
# - 0.80 is a balanced default.
# - 0.90 gives a larger reference library but fewer holdout examples.
#
# Do not use a random split as the primary validation method unless dates are
# unavailable. The goal is to simulate a future incident being compared with past
# injury history.
TEMPORAL_TRAIN_FRACTION = 0.80

# Minimum number of dated injury records required before attempting a temporal
# validation split.
#
# Why this exists:
# - If only a small number of dated injury records exist, a temporal split can be
#   unstable and may produce a tiny holdout set.
#
# How to tune:
# - Lower this only for small demo datasets.
# - Increase this if your validation metrics are unstable because the holdout set
#   is too small.
MIN_TEMPORAL_RECORDS = 50

# Minimum number of held-out injury records required after the temporal split.
#
# Why this exists:
# - Metrics such as same-site match rate or severe-query match rate are not
#   meaningful if the holdout set has only a few records.
#
# How to tune:
# - Use 10 for a small MVP.
# - Use 30+ if you have enough injury data and want more stable validation.
MIN_HOLDOUT_RECORDS = 10

# ---------------------------------------------------------------------------
# TF-IDF text vectorization parameters
# ---------------------------------------------------------------------------

# Maximum vocabulary size retained by the TF-IDF vectorizer.
#
# Business/model meaning:
# - Each retained word or phrase becomes one model feature.
# - A larger value captures more specific phrases, such as equipment names or
#   detailed hazard wording.
#
# How to tune:
# - Lower, e.g. 10,000-25,000: faster, less memory, more general matching.
# - Default 50,000: good balance for medium EHS text datasets.
# - Higher, e.g. 100,000+: may improve specificity but can overfit rare wording
#   and increase memory/time.
TFIDF_MAX_FEATURES = 50000

# Minimum document frequency for a token/phrase to enter the vocabulary.
#
# Business/model meaning:
# - min_df=2 means a word/phrase must appear in at least two reference injury
#   records to be used.
# - This removes one-off typos and extremely rare text that is unlikely to help
#   similarity search.
#
# How to tune:
# - 1: use for very small datasets; keeps rare but possibly important terms.
# - 2: good default; removes many typos.
# - 3-5: more aggressive cleanup; may remove rare hazard/equipment names.
TFIDF_MIN_DF = 2

# Maximum document frequency for a token/phrase to enter the vocabulary.
#
# Business/model meaning:
# - max_df=0.95 removes words/phrases appearing in more than 95% of records.
# - Very common words often carry little meaning for similarity.
#
# How to tune:
# - 0.90: more aggressive removal of common language.
# - 0.95: balanced default.
# - 1.00: keep all common terms; sometimes useful if the dataset is tiny.
TFIDF_MAX_DF = 0.95

# N-gram range used by TF-IDF.
#
# Business/model meaning:
# - (1, 1): single words only, e.g. "forklift", "pedestrian".
# - (1, 2): single words plus two-word phrases, e.g. "forklift pedestrian",
#   "lost balance", "loading dock".
#
# How to tune:
# - (1, 1): faster and more general; useful for small datasets.
# - (1, 2): recommended default; captures important EHS phrases.
# - (1, 3): can capture longer phrases but increases vocabulary size and can
#   become sparse/noisy.
TFIDF_NGRAM_RANGE = (1, 2)

# ---------------------------------------------------------------------------
# Threshold calibration parameters
# ---------------------------------------------------------------------------

# Number of record pairs sampled to estimate background similarity.
#
# Background similarity answers:
# - "How similar do records look by chance or generic safety-language overlap?"
#
# How to tune:
# - 2,000-5,000: faster for development.
# - 10,000: good default.
# - 50,000+: more stable on large datasets but slower.
#
# Note:
# - The sampling logic tries to choose records that differ by metadata such as
#   category/site/department/date gap when those fields exist. This makes the
#   background distribution cleaner than fully random pairs.
BACKGROUND_SAMPLE_SIZE = 10000

# Maximum number of reference injury records used for leave-one-out top-1
# calibration.
#
# Leave-one-out calibration answers:
# - "When an injury record searches for similar injury records, what top-1
#   similarity scores are typical?"
#
# Why a cap exists:
# - Exact nearest-neighbor search over every record can be slower on large
#   datasets.
#
# How to tune:
# - Lower, e.g. 1,000-2,000: faster but noisier calibration.
# - Default 5,000: stable for most projects.
# - Higher: more stable if runtime is acceptable.
MAX_LOO_RECORDS = 5000

# Minimum date gap used when constructing metadata-constrained background pairs.
#
# Business/model meaning:
# - If two records are close in time, especially at similar sites/categories,
#   they may belong to the same recurring issue. Requiring a date gap helps make
#   sampled background pairs less likely to be true duplicates or same-event
#   clusters.
#
# How to tune:
# - 90: less strict; more background pairs available.
# - 180: balanced default.
# - 365: stricter; may be cleaner but harder to sample enough pairs.
BACKGROUND_MIN_DATE_GAP_DAYS = 180

# ---------------------------------------------------------------------------
# Text fields used for early-lifecycle similarity
# ---------------------------------------------------------------------------

# Columns used to build the text representation if ml_text_early is unavailable.
#
# Important modeling rule:
# - Use only information available early in the incident lifecycle.
# - Do NOT add outcome/post-investigation fields such as lost time, restricted
#   time, inpatient status, immediate causes, causal factors, final severity, or
#   treatment fields as input text. Those are outcomes or later investigation
#   details and can leak label information into the model.
#
# How to tune:
# - Add early descriptive fields if they are available in the prepared dataset.
# - Remove fields if they are too noisy or mostly blank.
# - Keep the list stable once the model is used in production so scores remain
#   comparable over time.
EARLY_TEXT_FIELDS = [
    "title",
    "description",
    "activity_during_incident",
    "equipment",
    "vehicle",
    "off_premises_location",
    "other_process",
    "other_activity",
]





# ---------------------------------------------------------------------------
# User-facing match return policy
# ---------------------------------------------------------------------------

# Minimum score band required before the system treats a query as having a
# user-facing historical injury match.
#
# Why this exists:
# - The calibrated weak_match threshold is intentionally low. It only means the
#   score is above ordinary background overlap.
# - In practice, weak matches can be too noisy for business users and can make
#   almost every query appear to have a match.
# - This setting separates diagnostic bands from user-facing output. A record can
#   have raw_similarity_band = weak_match, but if this parameter is set to
#   "possible_match", the user-facing similarity_band becomes no_match.
#
# Allowed values:
# - "weak_match": most permissive. Returns weak, possible, and strong matches.
# - "possible_match": recommended default. Returns only possible and strong
#   matches; weak evidence is treated as no_match in output files.
# - "strong_match": most conservative. Returns only strong matches.
#
# How to tune:
# - Use "possible_match" when you want more no_match examples and fewer noisy
#   low-confidence matches.
# - Use "strong_match" if stakeholders only want very confident historical
#   analogs.
# - Use "weak_match" only for analysis/debugging, not for production displays.
RETURN_MATCH_MIN_BAND = "possible_match"

# Optional absolute score override for the user-facing return threshold.
#
# Default None means: derive the return threshold from RETURN_MATCH_MIN_BAND and
# the calibrated thresholds.json. For example, if RETURN_MATCH_MIN_BAND is
# "possible_match", then the return threshold is possible_match_threshold.
#
# How to tune:
# - Keep None for data-driven thresholding.
# - Set a numeric value such as 0.20 if you want a fixed operational threshold
#   after reviewing validation/no-match-control results.
# - If this is set, it overrides RETURN_MATCH_MIN_BAND.
RETURN_MATCH_MIN_SCORE = None

# Whether to keep the raw diagnostic band in output files.
#
# If True, output CSVs include:
# - raw_similarity_band: calibrated diagnostic band using weak/possible/strong
# - similarity_band: user-facing band after applying RETURN_MATCH_MIN_BAND
#
# This is useful because reviewers can see that a query had weak evidence but was
# intentionally suppressed as no_match for business use.
INCLUDE_RAW_SIMILARITY_BAND = True

# ---------------------------------------------------------------------------
# No-match validation controls
# ---------------------------------------------------------------------------

# Whether temporal validation should add a small negative-control validation set.
#
# Why this exists:
# - The regular temporal holdout set contains injury records only. Because every
#   holdout query is itself an injury case, it is expected to resemble at least
#   some historical injury language, so almost all records may receive weak_match
#   or better.
# - This option adds extra validation queries that are expected to have no strong
#   relationship to historical injury records. These controls demonstrate that
#   the scoring/threshold logic can produce "no_match" when the input text is not
#   meaningfully related to the injury reference library.
#
# How to tune:
# - Keep True for validation demos and sanity checks.
# - Set False if you want validation outputs to contain only real held-out injury
#   records and no negative controls.
ENABLE_NO_MATCH_VALIDATION_CONTROLS = True

# Number of real non-injury control rows to add when enough low-similarity
# non-injury records exist.
#
# Important interpretation:
# - A non-injury record is NOT automatically irrelevant. A near miss or hazard
#   can be highly relevant if it resembles past injuries. Therefore the code does
#   not blindly label all non-injury rows as expected no_match.
# - Instead, it scores non-injury rows against the train/reference injury library
#   and keeps only the lowest-similarity rows that fall below the weak-match
#   threshold. These are useful negative controls, not proof that all non-injury
#   records should be no_match.
#
# How to tune:
# - 10-25: enough examples for validation reports without making files large.
# - 50+: useful if you want more examples in a dashboard or slide deck.
# - 0: disable real non-injury controls while keeping synthetic controls.
NO_MATCH_REAL_CONTROL_SAMPLE_SIZE = 25

# Maximum number of candidate non-injury records to score when looking for real
# no-match controls.
#
# Why this exists:
# - Scoring every non-injury record can be slow if the dataset is large.
# - The script samples a reproducible pool, scores that pool, and then keeps the
#   lowest-similarity records.
#
# How to tune:
# - 500-1,000: faster development runs.
# - 2,000: balanced default.
# - 10,000+: more likely to find many real no-match controls, but slower.
NO_MATCH_REAL_CONTROL_CANDIDATE_POOL_SIZE = 2000

# If True, real non-injury controls are sampled from the holdout/future time
# period when incident dates are available.
#
# Why this exists:
# - It mirrors the temporal validation setup: older injury history is the
#   reference library and newer records are treated as future queries.
#
# How to tune:
# - Keep True for the cleanest validation story.
# - Set False if too few non-injury records exist in the holdout date range.
NO_MATCH_REAL_CONTROLS_HOLDOUT_PERIOD_ONLY = True

# Synthetic off-domain controls used to prove that clearly irrelevant text can
# return no_match.
#
# Why synthetic controls are included:
# - Real non-injury EHS records may still be relevant to injuries, so they are
#   not guaranteed no_match cases.
# - These synthetic controls are intentionally unrelated administrative or daily
#   life examples. They should have very low TF-IDF overlap with the injury
#   reference library and therefore should fall below the weak-match threshold.
#
# How to tune:
# - Keep a small number of clearly irrelevant examples.
# - Do not make these look like safety incidents; the purpose is to test the
#   rejection behavior for irrelevant inputs.
SYNTHETIC_NO_MATCH_CONTROL_RECORDS = [
    {
        "incident_id": "synthetic_no_match_001",
        "incident_category_name": "Negative Control",
        "title": "Astronomy lecture schedule",
        "description": "The planetarium agenda includes nebula photography, telescope calibration, and constellation viewing notes.",
    },
    {
        "incident_id": "synthetic_no_match_002",
        "incident_category_name": "Negative Control",
        "title": "Sourdough recipe catalog",
        "description": "The cookbook index lists focaccia, rosemary bread, citrus marmalade, and pastry glazing instructions.",
    },
    {
        "incident_id": "synthetic_no_match_003",
        "incident_category_name": "Negative Control",
        "title": "Watercolor gallery exhibit",
        "description": "The art exhibit features landscape sketches, pigment palettes, canvas framing, and ceramic sculpture labels.",
    },
    {
        "incident_id": "synthetic_no_match_004",
        "incident_category_name": "Negative Control",
        "title": "Classical music recital program",
        "description": "The evening program includes violin sonatas, piano accompaniment, orchestra seating, and composer biographies.",
    },
    {
        "incident_id": "synthetic_no_match_005",
        "incident_category_name": "Negative Control",
        "title": "Travel itinerary planning",
        "description": "The itinerary lists museum reservations, hotel confirmation numbers, passport reminders, and train schedules.",
    },
]

# ---------------------------------------------------------------------------
# Project path helpers
# ---------------------------------------------------------------------------


def get_project_root() -> Path:
    """Return the project root folder.

    The file layout is expected to be:

        project_root/
          src/
            injury_similarity/
              config.py

    Because this file lives in src/injury_similarity, parents[2] points back to
    project_root. Keeping this calculation here lets the training scripts run
    without requiring the user to pass paths at the command line.
    """
    return Path(__file__).resolve().parents[2]


def get_outputs_dir() -> Path:
    """Return the base outputs folder used by the whole project.

    Default:
        <project_root>/outputs

    Optional override:
        PATTERN_LEARNING_OUTPUT_DIR=/custom/output/path

    This mirrors the existing data-preparation pipeline, so all ML outputs land
    under the same original outputs folder rather than creating a new project
    location.
    """
    return Path(os.getenv("PATTERN_LEARNING_OUTPUT_DIR", get_project_root() / "outputs"))


def get_processed_dir() -> Path:
    """Return the folder containing prepared CSV files from run_data_prep.py.

    Expected inputs for this ML task include:
    - incident_injury_all_records.csv
    - pattern_learning_records.csv, for prediction candidates when available
    """
    return get_outputs_dir() / "processed"


def get_ml_dir() -> Path:
    """Return the root folder for injury-similarity ML outputs."""
    return get_outputs_dir() / "ml" / "injury_similarity"


def get_validation_dir() -> Path:
    """Return the folder for temporal holdout validation outputs.

    This folder stores validation metrics, validation-only predictions, and the
    temporary validation-only model fit on the older train split.
    """
    return get_ml_dir() / "validation"


def get_temporal_validation_dir() -> Path:
    """Return a named subfolder for temporal-holdout validation artifacts.

    This helper is retained for compatibility and clarity if you later split
    validation into multiple strategies. The current core workflow writes
    directly under get_validation_dir().
    """
    return get_validation_dir() / "temporal_holdout"


def get_validation_model_dir() -> Path:
    """Return the folder for the validation-only train-split model.

    This is NOT the final production model. It exists so you can inspect exactly
    what was fit during temporal holdout validation.
    """
    return get_temporal_validation_dir() / "validation_model"


def get_model_dir() -> Path:
    """Return the folder for the final production model.

    The final production model is fit on all eligible historical injury records
    after validation. Prediction scripts load artifacts from this folder.
    """
    return get_ml_dir() / "final_model"


def get_prediction_dir() -> Path:
    """Return the folder for batch prediction/scoring outputs."""
    return get_ml_dir() / "predictions"
