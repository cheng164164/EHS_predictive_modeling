"""Configuration for audits_tasks_modeling theme-mining pipeline.

This file keeps the existing Step 00 unified-event settings and adds a
source-aware clustering pipeline for:
  1) incident/hazard/near-miss/injury records
  2) audit/observation records
  3) task/action records

All scripts can be run without command-line arguments. Change settings here.

Quick POC mode is enabled by default to avoid embedding every record. It first
chooses major case-heavy locations, then draws a capped stratified sample from
those locations. Increase/decrease the caps below for faster or broader tests.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
# config.py path: pattern_learning_project/src/audits_tasks_modeling/config.py
PROJECT_ROOT = THIS_FILE.parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "audits_tasks_modeling"

# Existing Step 00 unified table output folder.
STEP_00_DIR = OUTPUT_ROOT / "00_unified_text_events"
UNIFIED_EVENTS_FILE = STEP_00_DIR / "safety_text_event.csv.gz"

# New theme-mining output folders.
THEME_INPUT_DIR = OUTPUT_ROOT / "01_theme_input"
THEME_EMBEDDING_DIR = OUTPUT_ROOT / "02_theme_embeddings"
THEME_CLUSTER_DIR = OUTPUT_ROOT / "03_theme_clusters"
THEME_CATALOG_DIR = OUTPUT_ROOT / "04_theme_catalog"
THEME_PROFILE_DIR = OUTPUT_ROOT / "05_location_theme_profiles"
THEME_LINK_DIR = OUTPUT_ROOT / "06_theme_links"
THEME_LOG_DIR = OUTPUT_ROOT / "logs"

# ---------------------------------------------------------------------------
# Existing Step 00 settings. These are used by 00_build_unified_text_events.py.
# ---------------------------------------------------------------------------
SAMPLE_SIZE = None
DROP_EMPTY_TEXT = True
ENGLISH_ONLY_TEXT_FILTER = False
LANGUAGE_DETECTION_LIBRARY = "langdetect"
LANGUAGE_DETECTION_RANDOM_STATE = 42
LANGUAGE_DETECTION_MIN_PROB = 0.80
LANGUAGE_DETECTION_MIN_TEXT_CHARS = 20
LANGUAGE_DETECTION_MAX_TEXT_CHARS = 1000
ENGLISH_ONLY_KEEP_SHORT_TEXT = True
ENGLISH_ONLY_KEEP_UNKNOWN_LANGUAGE = False
LANGUAGE_DETECTION_PROGRESS_EVERY = 50000

# Set True in run_theme_mining_end_to_end.py if you want to rebuild unified data
# from raw data. If False, the pipeline uses UNIFIED_EVENTS_FILE if it exists.
RUN_STEP_00_IN_END_TO_END = False

# ---------------------------------------------------------------------------
# Source family names used throughout the project.
# ---------------------------------------------------------------------------
FAMILY_INCIDENT_HAZARD = "incident_hazard"

# Raw audit rows are first detected as FAMILY_AUDIT_OBSERVATION, then split
# into separate clustering families. Do not cluster the raw family directly.
FAMILY_AUDIT_OBSERVATION = "audit_observation"  # raw/legacy audit family
FAMILY_AUDIT_RISK = "audit_risk"                # unsafe acts/conditions + risk observations
FAMILY_AUDIT_POSITIVE = "audit_positive"        # safe acts/conditions + positive controls
FAMILY_AUDIT_ACTIVITY = "audit_activity"        # routine/admin/accounting only, not clustered

FAMILY_TASK_ACTION = "task_action"

# Families in SOURCE_FAMILIES are embedded and clustered. Audit activity is
# kept in accounting files but intentionally excluded from clustering.
SOURCE_FAMILIES = [
    FAMILY_INCIDENT_HAZARD,
    FAMILY_AUDIT_RISK,
    FAMILY_AUDIT_POSITIVE,
    FAMILY_TASK_ACTION,
]
AUDIT_CLUSTER_FAMILIES = [FAMILY_AUDIT_RISK, FAMILY_AUDIT_POSITIVE]

# ---------------------------------------------------------------------------
# Theme input preparation
# ---------------------------------------------------------------------------
MIN_TEXT_LENGTH = 20
MAX_TEXT_CHARS_FOR_MODEL = 600
DROP_DUPLICATE_THEME_TEXT_WITHIN_FAMILY = False

# Optional date limits. Use None to include all dates.
MIN_EVENT_DATE = None  # Example: "2015-01-01"
MAX_EVENT_DATE = None  # Example: "2026-12-31"

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Quick POC sampling by major case-heavy locations
# ---------------------------------------------------------------------------
# This is the recommended setting for a fast concept proof. Step 01 still reads
# the full unified file to identify the largest/most relevant locations, but it
# only passes the selected sample into the embedding/clustering steps.
ENABLE_POC_MAJOR_LOCATION_SAMPLE = True

# Select locations with large amounts of hazards, near misses, injuries,
# audits/observations, and tasks/actions. Use 10-15 for a quick POC.
POC_TOP_LOCATIONS = 12
POC_MIN_LOCATION_RECORDS = 100
POC_MIN_FAMILIES_PRESENT = 2

# Optional manual override. If non-empty, these location_id values are used in
# addition to the score-selected locations.
POC_INCLUDE_LOCATION_IDS: list[str] = []
POC_EXCLUDE_LOCATION_IDS: list[str] = []

# Maximum total records passed to embedding/clustering. Set to 100000 for a
# broader POC; keep 30000-60000 for quick iteration on CPU.
POC_MAX_TOTAL_RECORDS = 20000

# Family quotas should sum to POC_MAX_TOTAL_RECORDS. If they do not, the code
# scales them automatically. Equal-ish quotas prevent audits/tasks from drowning
# out incident/hazard records.
POC_FAMILY_QUOTAS = {
    FAMILY_INCIDENT_HAZARD: 8000,
    FAMILY_AUDIT_RISK: 3500,
    FAMILY_AUDIT_POSITIVE: 3500,
    FAMILY_TASK_ACTION: 5000,
}

# Always try to preserve high-value records from selected locations before
# random sampling lower-priority records.
POC_ALWAYS_KEEP_EVENT_KINDS = {
    "serious_injury",
    "normal_injury",
    "near_miss",
    "task_overdue",
    "task_open",
}
POC_ALWAYS_KEEP_MIN_REVIEW_PRIORITY = 4.0

# Location selection weights. These do not become model features; they only
# choose high-signal locations for the concept-proof sample.
POC_LOCATION_SCORE_WEIGHTS = {
    "serious_injury": 30.0,
    "normal_injury": 10.0,
    "near_miss": 8.0,
    "hazard_identification": 2.0,
    "audit_unsafe_condition": 3.0,
    "audit_unsafe_act": 3.0,
    "audit_safe_condition": 1.8,
    "audit_safe_act": 1.8,
    "audit_positive_observation": 1.5,
    "audit_observation": 1.0,
    "audit_other": 0.7,
    "task_overdue": 3.0,
    "task_open": 1.5,
    "task_other": 0.5,
}

# Optional old development sample per family after all other filtering. Keep at
# 0 when POC_MAJOR_LOCATION_SAMPLE is enabled. This is retained for debugging.
MAX_RECORDS_PER_FAMILY = 0

THEME_INPUT_ALL_FILE = THEME_INPUT_DIR / "theme_input_all.csv"
THEME_INPUT_FILE_BY_FAMILY = {
    FAMILY_INCIDENT_HAZARD: THEME_INPUT_DIR / "theme_input_incident_hazard.csv",
    FAMILY_AUDIT_RISK: THEME_INPUT_DIR / "theme_input_audit_risk.csv",
    FAMILY_AUDIT_POSITIVE: THEME_INPUT_DIR / "theme_input_audit_positive.csv",
    FAMILY_TASK_ACTION: THEME_INPUT_DIR / "theme_input_task_action.csv",
}
THEME_INPUT_PROFILE_FILE = THEME_INPUT_DIR / "theme_input_profile.csv"
POC_LOCATION_PROFILE_FILE = THEME_INPUT_DIR / "poc_major_location_profile.csv"
POC_SELECTED_LOCATIONS_FILE = THEME_INPUT_DIR / "poc_selected_locations.csv"
POC_SAMPLING_SUMMARY_FILE = THEME_INPUT_DIR / "poc_sampling_summary.json"


# ---------------------------------------------------------------------------
# Audit-specific clustering controls
# ---------------------------------------------------------------------------
# Routine scheduled inspections/checklists often dominate audit text and create
# meaningless clusters. Keep them for accounting, but exclude them from audit
# text clustering by default.
# Split audits into: audit_risk, audit_positive, and audit_activity.
# audit_risk and audit_positive are clustered separately. audit_activity is
# retained only for accounting.
AUDIT_CLUSTER_ONLY_MEANINGFUL_FINDINGS = True
AUDIT_SPLIT_SAFE_AND_UNSAFE_CLUSTERS = True
AUDIT_INCLUDE_SAFE_ACT_CONDITION_CLUSTERING = True
AUDIT_WRITE_ACTIVITY_ACCOUNTING = True
AUDIT_MEANINGFUL_OBSERVATION_MIN_CHARS = 60
AUDIT_KEEP_NONROUTINE_LONG_OBSERVATIONS = False
AUDIT_EXCLUDED_SAMPLE_ROWS = 10000
AUDIT_ELIGIBLE_SAMPLE_ROWS = 10000

# If status/title/text indicates a scheduled/routine inspection and no unsafe
# signal is present, the record is counted but not embedded/clustered.
AUDIT_ROUTINE_STATUS_PATTERNS = [
    r"\bscheduled\b",
    r"\bprogramad[ao]\b",
    r"\bplanead[ao]\b",
]

AUDIT_ROUTINE_TEXT_PATTERNS = [
    r"\bscheduled\b",
    r"\bin progress\b",
    r"\bweekly\b",
    r"\bmonthly\b",
    r"\bdaily\b",
    r"\bcheck\s*list\b",
    r"\bchecklist\b",
    r"\binspec(?:ci[oó]n|cion|ion|ci[oó]|ci)\b",
    r"\binspecci[oó]n\b",
    r"\binspecion\b",
    r"\binspecci\b",
    r"\bauditor[ií]a\b",
    r"\bmontacargas\b.*\bscheduled\b",
    r"\bvehicular\b.*\bscheduled\b",
    r"\binspec(?:ci[oó]n|ion|ci[oó])\s+montacargas\b",
    r"\binspec(?:ci[oó]n|ion|ci[oó])\s+vehicular\b",
    r"\bforklift\s+inspection\b",
    r"\bvehicle\s+inspection\b",
    r"\btruck\s+inspection\b",
    r"\bservice\s+truck\b",
    r"\bsafety\s+observation\s*$",
    r"\bobservation\s+act\s*$",
]

# Broad, multilingual indicators that an audit/observation text contains a
# specific safety finding rather than just an inspection form name. This is not
# used to define clusters; it is only used to keep meaningful audit records.
AUDIT_FINDING_SIGNAL_PATTERNS = [
    r"\bunsafe\b", r"\bhazard\b", r"\brisk\b",
    r"\bnot\s+wearing\b", r"\bwithout\b", r"\bmissing\b", r"\bno\s+safety\b",
    r"\bdamaged\b", r"\bbroken\b", r"\bdefective\b", r"\bloose\b", r"\bexposed\b",
    r"\bspill\b", r"\bleak\b", r"\bblocked\b", r"\bobstruct", r"\btrip\b", r"\bslip\b",
    r"\bguard\b", r"\bunguarded\b", r"\bcable\b", r"\bcord\b", r"\bwire\b",
    r"\bppe\b", r"\bepp\b", r"\bsafety\s+glasses\b", r"\bgloves?\b",
    r"\bhard\s+hat\b", r"\bhelmet\b", r"\bface\s+shield\b", r"\bvest\b",
    r"\bfalta\b", r"\bsin\b", r"\bno\s+usa", r"\bno\s+usar", r"\blentes\b",
    r"\bguantes\b", r"\bcasco\b", r"\bcareta\b", r"\bchaleco\b", r"\bderrame\b",
    r"\bobstru", r"\bdañ", r"\bdanad", r"\broto\b", r"\brota\b", r"\bmal\s+estado\b",
]

AUDIT_UNSAFE_ACT_PATTERNS = [
    r"\bunsafe\s+act\b", r"\bnot\s+wearing\b", r"\bwithout\s+(?:ppe|gloves?|safety\s+glasses|hard\s+hat|helmet|vest)\b",
    r"\bno\s+safety\s+(?:glasses|vest|gloves?)\b", r"\bsin\s+(?:lentes|guantes|casco|chaleco|careta)\b",
    r"\bfalta\s+de\s+(?:lentes|guantes|casco|chaleco|careta)\b", r"\bno\s+usa", r"\bno\s+usar",
]

AUDIT_UNSAFE_CONDITION_PATTERNS = [
    r"\bunsafe\s+condition\b", r"\bdamaged\b", r"\bbroken\b", r"\bdefective\b",
    r"\bmissing\b", r"\bloose\b", r"\bexposed\b", r"\bspill\b", r"\bleak\b",
    r"\bblocked\b", r"\bobstruct", r"\btrip\b", r"\bslip\b", r"\bunguarded\b",
    r"\bderrame\b", r"\bobstru", r"\bdañ", r"\bdanad", r"\broto\b", r"\brota\b",
]

# Safe/positive observations are valuable control evidence. They are clustered
# separately from unsafe findings to avoid mixing "wearing PPE" with "not
# wearing PPE" in the same audit theme.
AUDIT_SAFE_ACT_PATTERNS = [
    r"\bsafe\s+act\b", r"\bpositive\s+behavior\b", r"\bproper(?:ly)?\s+(?:wearing|using)\b",
    r"\bwearing\s+(?:proper\s+)?(?:ppe|gloves?|safety\s+glasses|hard\s+hat|helmet|vest)\b",
    r"\busing\s+(?:proper\s+)?(?:ppe|fall\s+protection|lockout|loto|guard)\b",
    r"\bcompliant\b", r"\bcompliance\b", r"\bcorrectly\b", r"\bproper\b",
    r"\buso\s+correcto\b", r"\busando\b", r"\butiliza\b", r"\bcumple\b", r"\bcumplimiento\b",
]

AUDIT_SAFE_CONDITION_PATTERNS = [
    r"\bsafe\s+condition\b", r"\bgood\s+housekeeping\b", r"\bclean\s+(?:area|workplace|floor)\b",
    r"\bclear\s+(?:walkway|access|aisle)\b", r"\bguard(?:ing)?\s+(?:in\s+place|installed)\b",
    r"\bbarrier(?:s)?\s+(?:in\s+place|installed)\b", r"\bproper\s+storage\b",
    r"\bwell\s+maintained\b", r"\bin\s+good\s+condition\b",
    r"\borden\s+y\s+limpieza\b", r"\blimpio\b", r"\bbuena\s+condici",
]

AUDIT_POSITIVE_SIGNAL_PATTERNS = AUDIT_SAFE_ACT_PATTERNS + AUDIT_SAFE_CONDITION_PATTERNS

# Remove these from audit model text after eligibility is decided. This prevents
# clusters from being named after form/template language.
AUDIT_MODEL_TEXT_REMOVE_PATTERNS = [
    r"\btitle\s*:", r"\bdescription\s*:", r"\bstatus\s*:",
    r"\bobservation\s+act\b", r"\bsafety\s+observation\b",
    r"\bobservation\b", r"\binspection\b", r"\baudit\b", r"\bscheduled\b",
    r"\binspec(?:ci[oó]n|cion|ion|ci[oó]|ci)\b", r"\binspecion\b", r"\binspecci\b",
    r"\bclosed\b", r"\bin\s+progress\b", r"\bpending\b",
]

AUDIT_CLUSTER_STOPWORDS = {
    "observation", "observations", "act", "acts", "safety", "audit", "audits",
    "inspection", "inspections", "inspecion", "inspeccion", "inspecci", "scheduled",
    "closed", "progress", "title", "description", "weekly", "monthly", "daily",
}

AUDIT_ACTIVITY_ACCOUNTING_FILE = THEME_INPUT_DIR / "audit_activity_accounting.csv"
AUDIT_CLUSTER_ELIGIBILITY_SUMMARY_FILE = THEME_INPUT_DIR / "audit_cluster_eligibility_summary.csv"
AUDIT_EXCLUDED_FROM_CLUSTERING_FILE = THEME_INPUT_DIR / "audit_excluded_from_clustering_sample.csv"
AUDIT_ELIGIBLE_FOR_CLUSTERING_FILE = THEME_INPUT_DIR / "audit_eligible_for_clustering_sample.csv"
AUDIT_RISK_FOR_CLUSTERING_FILE = THEME_INPUT_DIR / "audit_risk_for_clustering_sample.csv"
AUDIT_POSITIVE_FOR_CLUSTERING_FILE = THEME_INPUT_DIR / "audit_positive_for_clustering_sample.csv"

# ---------------------------------------------------------------------------
# Embedding settings
# ---------------------------------------------------------------------------
# Preferred backend is sentence_transformers. If unavailable, the code falls
# back to TF-IDF + SVD embeddings so the pipeline can still run.
EMBEDDING_BACKEND = "sentence_transformers"  # sentence_transformers or tfidf_svd
# SENTENCE_TRANSFORMER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Better quality but sometimes slower to download/run:
SENTENCE_TRANSFORMER_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_BATCH_SIZE = 128
EMBEDDING_NORMALIZE = True
EMBEDDING_DEVICE = None  # None=auto, or "cpu", "cuda"

TFIDF_MAX_FEATURES = 50000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.95
SVD_COMPONENTS = 256

EMBEDDING_FILE_BY_FAMILY = {
    FAMILY_INCIDENT_HAZARD: THEME_EMBEDDING_DIR / "embeddings_incident_hazard.npy",
    FAMILY_AUDIT_RISK: THEME_EMBEDDING_DIR / "embeddings_audit_risk.npy",
    FAMILY_AUDIT_POSITIVE: THEME_EMBEDDING_DIR / "embeddings_audit_positive.npy",
    FAMILY_TASK_ACTION: THEME_EMBEDDING_DIR / "embeddings_task_action.npy",
}
EMBEDDING_META_FILE_BY_FAMILY = {
    FAMILY_INCIDENT_HAZARD: THEME_EMBEDDING_DIR / "embedding_meta_incident_hazard.csv",
    FAMILY_AUDIT_RISK: THEME_EMBEDDING_DIR / "embedding_meta_audit_risk.csv",
    FAMILY_AUDIT_POSITIVE: THEME_EMBEDDING_DIR / "embedding_meta_audit_positive.csv",
    FAMILY_TASK_ACTION: THEME_EMBEDDING_DIR / "embedding_meta_task_action.csv",
}
EMBEDDING_SUMMARY_FILE = THEME_EMBEDDING_DIR / "embedding_summary.json"

# ---------------------------------------------------------------------------
# Clustering settings
# ---------------------------------------------------------------------------
# hdbscan is recommended for discovery. minibatch_kmeans is useful when you
# need full assignment or the family has very repetitive short text.
CLUSTER_METHOD_BY_FAMILY = {
    FAMILY_INCIDENT_HAZARD: "hdbscan",
    FAMILY_AUDIT_RISK: "hdbscan",
    FAMILY_AUDIT_POSITIVE: "hdbscan",
    FAMILY_TASK_ACTION: "minibatch_kmeans",
}

# Dimensionality reduction before clustering. UMAP works best for HDBSCAN if
# installed. PCA fallback is automatic if UMAP is not available.
USE_DIMENSION_REDUCTION = True
REDUCTION_METHOD = "umap"  # umap or pca
UMAP_N_NEIGHBORS = 30
UMAP_N_COMPONENTS = 15
UMAP_MIN_DIST = 0.0
UMAP_METRIC = "cosine"
PCA_N_COMPONENTS = 50

# HDBSCAN parameters. Smaller values are suitable for sampled POC runs.
HDBSCAN_MIN_CLUSTER_SIZE_BY_FAMILY = {
    FAMILY_INCIDENT_HAZARD: 30,
    FAMILY_AUDIT_RISK: 35,
    FAMILY_AUDIT_POSITIVE: 35,
    FAMILY_TASK_ACTION: 50,
}
HDBSCAN_MIN_SAMPLES_BY_FAMILY = {
    FAMILY_INCIDENT_HAZARD: 8,
    FAMILY_AUDIT_RISK: 8,
    FAMILY_AUDIT_POSITIVE: 8,
    FAMILY_TASK_ACTION: 10,
}
HDBSCAN_CLUSTER_SELECTION_METHOD = "eom"

# KMeans parameters. Used for families configured as minibatch_kmeans or as a
# fallback when HDBSCAN is unavailable.
KMEANS_N_CLUSTERS_BY_FAMILY = {
    FAMILY_INCIDENT_HAZARD: 60,
    FAMILY_AUDIT_RISK: 50,
    FAMILY_AUDIT_POSITIVE: 50,
    FAMILY_TASK_ACTION: 70,
}
KMEANS_BATCH_SIZE = 4096
KMEANS_MAX_ITER = 200

# If HDBSCAN labels records as noise, optionally assign them to the nearest
# strong cluster centroid if the cosine similarity is high enough.
ASSIGN_HDBSCAN_NOISE_TO_NEAREST_THEME = True
NEAREST_THEME_MIN_COSINE_SIMILARITY = 0.55

# If a family is still large after POC sampling, fit HDBSCAN on a sample and
# assign the rest by nearest centroid. 0 means fit on all family records.
CLUSTER_FIT_MAX_RECORDS_BY_FAMILY = {
    FAMILY_INCIDENT_HAZARD: 0,
    FAMILY_AUDIT_RISK: 0,
    FAMILY_AUDIT_POSITIVE: 0,
    FAMILY_TASK_ACTION: 0,
}

THEME_ASSIGNMENTS_FILE = THEME_CLUSTER_DIR / "event_theme_assignments.csv"
CLUSTER_SUMMARY_FILE = THEME_CLUSTER_DIR / "cluster_run_summary.json"
ASSIGNMENT_FILE_BY_FAMILY = {
    FAMILY_INCIDENT_HAZARD: THEME_CLUSTER_DIR / "event_theme_assignments_incident_hazard.csv",
    FAMILY_AUDIT_RISK: THEME_CLUSTER_DIR / "event_theme_assignments_audit_risk.csv",
    FAMILY_AUDIT_POSITIVE: THEME_CLUSTER_DIR / "event_theme_assignments_audit_positive.csv",
    FAMILY_TASK_ACTION: THEME_CLUSTER_DIR / "event_theme_assignments_task_action.csv",
}
THEME_CENTROID_FILE = THEME_CLUSTER_DIR / "theme_centroids.npy"
THEME_CENTROID_META_FILE = THEME_CLUSTER_DIR / "theme_centroid_meta.csv"

# ---------------------------------------------------------------------------
# Labeling / catalog settings
# ---------------------------------------------------------------------------
TOP_TERMS_PER_THEME = 20
REPRESENTATIVE_EXAMPLES_PER_THEME = 10
RANDOM_EXAMPLES_PER_THEME = 5
MAX_REPRESENTATIVE_TEXT_CHARS = 700
LABEL_TOP_TERM_COUNT = 4
MIN_THEME_SIZE_FOR_CATALOG = 3

THEME_CATALOG_FILE = THEME_CATALOG_DIR / "theme_catalog.csv"
THEME_REPRESENTATIVE_EXAMPLES_FILE = THEME_CATALOG_DIR / "theme_representative_examples.csv"
THEME_CATALOG_REVIEW_FILE = THEME_CATALOG_DIR / "theme_catalog_review.csv"

# ---------------------------------------------------------------------------
# Location/theme period profiles
# ---------------------------------------------------------------------------
PROFILE_PERIOD_FREQ = "Y"  # Y, Q, M, W
LOCATION_THEME_PERIOD_FILE = THEME_PROFILE_DIR / f"location_theme_period_profile_{PROFILE_PERIOD_FREQ}.csv"
LOCATION_PERIOD_TOP_THEMES_FILE = THEME_PROFILE_DIR / f"location_period_top_themes_{PROFILE_PERIOD_FREQ}.csv"
THEME_PERIOD_TRENDS_FILE = THEME_PROFILE_DIR / f"theme_period_trends_{PROFILE_PERIOD_FREQ}.csv"
LOCATION_THEME_ROLLUP_FILE = THEME_PROFILE_DIR / "location_theme_rollup.csv"
TOP_THEMES_PER_LOCATION_PERIOD = 10

# ---------------------------------------------------------------------------
# Cross-family link settings
# ---------------------------------------------------------------------------
# Candidate links are not causal links. They are review candidates based on
# theme centroid similarity and location-period co-occurrence.
ENABLE_CROSS_FAMILY_LINKS = True
LINK_MIN_COSINE_SIMILARITY = 0.45
LINK_MIN_COOCCURRENCE_COUNT = 2
LINK_MAX_PAIRS_PER_SOURCE_THEME = 10
LINK_SCORE_SIMILARITY_WEIGHT = 0.65
LINK_SCORE_COOCCURRENCE_WEIGHT = 0.35
CROSS_FAMILY_LINKS_FILE = THEME_LINK_DIR / f"cross_family_theme_links_{PROFILE_PERIOD_FREQ}.csv"
LOCATION_DOMAIN_CANDIDATE_FILE = THEME_LINK_DIR / f"location_period_cross_family_candidates_{PROFILE_PERIOD_FREQ}.csv"

# ---------------------------------------------------------------------------
# Text cleaning / generic words
# ---------------------------------------------------------------------------
CUSTOM_STOPWORDS = {
    "title", "description", "activity", "activityduringincident", "immediateaction",
    "immediatecauses", "causalfactors", "bestpractices", "riskaction", "riskcondition",
    "status", "closed", "pending", "closure", "investigation", "record", "records",
    "employee", "worker", "person", "people", "reported", "report", "taken", "action",
    "area", "work", "working", "process", "incident", "near", "miss", "hazard",
    "audit", "task", "inspection", "observation", "observations", "condition", "safe", "unsafe", "komatsu",
    "immediately", "immediate", "pictures", "photo", "photos", "supervisor", "management",
    "approximately", "relevant", "stakeholder", "stakeholders", "department",
}

# Keep these strings out of theme text because they are source-system boilerplate.

CUSTOM_STOPWORDS |= {
    "observation", "observations", "observed", "observe", "observacion",
    "observación", "observa", "observação", "seguridad", "seguranca",
    "segurança", "act", "acts", "scheduled", "schedule", "weekly",
    "monthly", "daily", "checklist", "template", "inspeccion",
    "inspección", "inspecion", "inspecci", "consigna", "buena",
    "vista", "progress", "review", "prior", "version", "published",
}

# Keep these strings out of theme text because they are source-system boilerplate.
BOILERPLATE_PATTERNS = [
    "title:", "description:", "activityduringincident:", "immediateaction:",
    "immediatecauses:", "causalfactors:", "bestpractices:", "riskaction:",
    "riskcondition:", "offpremiseslocation:", "otherlocation:", "equipment:",
    "vehicle:", "source:", "task:", "comments:", "associatedparties:",
]

# Backward-compatible names used by the audit filtering code.
AUDIT_ROUTINE_EXCLUDE_PATTERNS = AUDIT_ROUTINE_TEXT_PATTERNS + AUDIT_ROUTINE_STATUS_PATTERNS
AUDIT_GENERIC_TITLE_PATTERNS = [
    r"^\s*(?:safety\s+observation|observation\s+act|observation|inspection|inspeccion|inspecion|inspecci)\s*$",
    r"^\s*(?:weekly|monthly|daily)\s+(?:inspection|observation)\s*$",
]
AUDIT_RISK_KEYWORD_PATTERNS = AUDIT_FINDING_SIGNAL_PATTERNS
AUDIT_POSITIVE_KEYWORD_PATTERNS = AUDIT_POSITIVE_SIGNAL_PATTERNS
AUDIT_THEME_TEXT_REMOVE_PATTERNS = AUDIT_MODEL_TEXT_REMOVE_PATTERNS
AUDIT_MIN_CLUSTER_TEXT_CHARS = 25
AUDIT_MIN_MEANINGFUL_OBSERVATION_CHARS = AUDIT_MEANINGFUL_OBSERVATION_MIN_CHARS
AUDIT_INCLUDE_UNSAFE_FINDINGS_WITH_GENERIC_TEXT = False
AUDIT_INCLUDE_GENERAL_OBSERVATIONS_WITH_RISK_KEYWORDS = True
AUDIT_EXCLUDE_SAFE_POSITIVE_OBSERVATIONS_FROM_CLUSTERING = False  # safe observations are clustered separately as audit_positive
AUDIT_KEEP_ROUTINE_INSPECTIONS_FOR_ACCOUNTING = True
AUDIT_CLUSTER_EXCLUDED_FILE = AUDIT_EXCLUDED_FROM_CLUSTERING_FILE
AUDIT_CLUSTER_ELIGIBILITY_PROFILE_FILE = AUDIT_CLUSTER_ELIGIBILITY_SUMMARY_FILE
