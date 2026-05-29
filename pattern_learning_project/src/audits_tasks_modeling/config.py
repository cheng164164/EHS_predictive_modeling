"""Central configuration for the audits/tasks text-first risk modeling pipeline.

Edit this file once, then run any step directly, for example:

    python 00_build_unified_text_events.py
    python 01_generate_embeddings.py
    python run_end_to_end.py

The defaults assume the CSV exports are stored in the project root. In the
ChatGPT sandbox that project root is /mnt/data. In your repo it is the folder
that contains src/ and outputs/.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def _optional_int(value: str | None) -> Optional[int]:
    if value is None or str(value).strip() == "":
        return None
    return int(value)


# ---------------------------------------------------------------------------
# Base folders
# ---------------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get("SAFETY_PROJECT_ROOT", SRC_DIR.parents[1])).resolve()


def _detect_data_dir(project_root: Path) -> Path:
    """Return the folder that contains the required raw CSV exports.

    Priority:
      1. SAFETY_DATA_DIR environment variable, if set
      2. <project_root>/data/raw
      3. <project_root>/data
      4. <project_root>
    """
    env_value = os.environ.get("SAFETY_DATA_DIR")
    if env_value:
        return Path(env_value).expanduser().resolve()

    candidates = [
        project_root / "data" / "raw",
        project_root / "data",
        project_root,
    ]

    required_names = {
        "AUDIT_VIEW.csv",
        "INCIDENT_VIEW.csv",
        "INCIDENTINJURY_VIEW.csv",
        "LISTITEM_VIEW.csv",
        "LOCATION_VIEW.csv",
        "TASK_VIEW.csv",
    }

    for candidate in candidates:
        if candidate.exists():
            existing_names = {p.name for p in candidate.glob("*.csv")}
            if required_names.issubset(existing_names):
                return candidate.resolve()

    return (project_root / "data" / "raw").resolve()


DATA_DIR = _detect_data_dir(PROJECT_ROOT)
OUTPUT_ROOT = Path(
    os.environ.get("SAFETY_OUTPUT_ROOT", PROJECT_ROOT / "outputs" / "audits_tasks_modeling")
).resolve()

# Raw input files. Change DATA_DIR above if these CSVs live elsewhere.
AUDIT_VIEW_PATH = DATA_DIR / "AUDIT_VIEW.csv"
INCIDENT_VIEW_PATH = DATA_DIR / "INCIDENT_VIEW.csv"
INCIDENTINJURY_VIEW_PATH = DATA_DIR / "INCIDENTINJURY_VIEW.csv"
LISTITEM_VIEW_PATH = DATA_DIR / "LISTITEM_VIEW.csv"
LOCATION_VIEW_PATH = DATA_DIR / "LOCATION_VIEW.csv"
TASK_VIEW_PATH = DATA_DIR / "TASK_VIEW.csv"

# Clear step output structure under OUTPUT_ROOT.
STEP_00_DIR = OUTPUT_ROOT / "00_unified_text_events"
STEP_01_DIR = OUTPUT_ROOT / "01_embeddings"
STEP_02_DIR = OUTPUT_ROOT / "02_safety_tags"
STEP_03_DIR = OUTPUT_ROOT / "03_risk_theme_discovery"
STEP_04_DIR = OUTPUT_ROOT / "04_theme_assignment"
STEP_05_DIR = OUTPUT_ROOT / "05_risk_state_dataset"
STEP_06_DIR = OUTPUT_ROOT / "06_risk_burden_model"
STEP_07_DIR = OUTPUT_ROOT / "07_elevated_risk_classifier"
STEP_08_DIR = OUTPUT_ROOT / "08_risk_driver_explanations"

# Main intermediate files.
LOCATION_HIERARCHY_PATH = STEP_00_DIR / "location_hierarchy.csv"
SAFETY_TEXT_EVENT_PATH = STEP_00_DIR / "safety_text_event.csv.gz"
TEXT_EMBEDDINGS_PATH = STEP_01_DIR / "text_embeddings.npy"
TEXT_EMBEDDING_EVENT_IDS_PATH = STEP_01_DIR / "text_embedding_event_ids.csv.gz"
TAGGED_EVENTS_PATH = STEP_02_DIR / "safety_text_event_tagged.csv.gz"
THEME_MEMBERSHIPS_PATH = STEP_03_DIR / "discovered_theme_memberships.csv.gz"
THEME_LIBRARY_PATH = STEP_03_DIR / "risk_theme_library.csv"
THEME_CENTROIDS_PATH = STEP_03_DIR / "risk_theme_centroids.npy"
THEMED_EVENTS_PATH = STEP_04_DIR / "safety_text_event_themed.csv.gz"
THEME_ASSIGNMENTS_PATH = STEP_04_DIR / "risk_theme_assignments.csv.gz"
RISK_STATE_DATA_PATH = STEP_05_DIR / "risk_state_training_data.csv.gz"

# ---------------------------------------------------------------------------
# Step 0: unified event table
# ---------------------------------------------------------------------------
SAMPLE_SIZE = _optional_int(os.environ.get("SAFETY_SAMPLE_SIZE"))
DROP_EMPTY_TEXT = os.environ.get("DROP_EMPTY_TEXT", "false").lower() in {"1", "true", "yes", "y"}
ENGLISH_ONLY_TEXT_FILTER=False
# ---------------------------------------------------------------------------
# Step 1: embeddings
# ---------------------------------------------------------------------------
TEXT_COLUMN = os.environ.get("TEXT_COLUMN", "clean_text")
# Default changed to sentence_transformer as requested.
EMBEDDING_PROVIDER = os.environ.get("EMBEDDING_PROVIDER", "tfidf_svd")
SENTENCE_TRANSFORMER_MODEL = os.environ.get(
    "SENTENCE_TRANSFORMER_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.environ.get(
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large"
)
# Used by the CLI default. Change this if you switch EMBEDDING_PROVIDER.
EMBEDDING_MODEL_NAME = os.environ.get(
    "EMBEDDING_MODEL_NAME",
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT if EMBEDDING_PROVIDER == "azure_openai" else SENTENCE_TRANSFORMER_MODEL,
)
EMBEDDING_BATCH_SIZE = int(os.environ.get("EMBEDDING_BATCH_SIZE", "256"))
TFIDF_SVD_COMPONENTS = int(os.environ.get("TFIDF_SVD_COMPONENTS", "128"))
TFIDF_MAX_FEATURES = int(os.environ.get("TFIDF_MAX_FEATURES", "50000"))
TFIDF_MIN_DF = int(os.environ.get("TFIDF_MIN_DF", "3"))
TFIDF_NGRAM_MAX = int(os.environ.get("TFIDF_NGRAM_MAX", "2"))

# ---------------------------------------------------------------------------
# Step 2: safety tag extraction
# ---------------------------------------------------------------------------
# Default is local-only: regex rules + optional embedding fallback. LLM extraction
# remains available through TAG_BACKEND=azure_openai, but is intentionally not
# the default for simplicity, cost control, and repeatability.
TAG_BACKEND = os.environ.get("TAG_BACKEND", "rules")  # rules or azure_openai
AZURE_OPENAI_CHAT_DEPLOYMENT = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o")
TAG_LLM_SLEEP_SECONDS = float(os.environ.get("TAG_LLM_SLEEP_SECONDS", "0"))

# Layer 2: use Step 01 text embeddings to assign a fallback tag when regex rules
# do not match a category. This reduces unassigned rows without relying on an LLM.
TAG_EMBEDDING_FALLBACK_ENABLED = os.environ.get("TAG_EMBEDDING_FALLBACK_ENABLED", "true").lower() in {"1", "true", "yes", "y"}
TAG_EMBEDDING_FALLBACK_FILL_ONLY_EMPTY = os.environ.get("TAG_EMBEDDING_FALLBACK_FILL_ONLY_EMPTY", "true").lower() in {"1", "true", "yes", "y"}
TAG_EMBEDDING_FALLBACK_TOP_K = int(os.environ.get("TAG_EMBEDDING_FALLBACK_TOP_K", "1"))
TAG_EMBEDDING_FALLBACK_THRESHOLD = float(os.environ.get("TAG_EMBEDDING_FALLBACK_THRESHOLD", "0.30"))
TAG_EMBEDDING_FALLBACK_HAZARD_THRESHOLD = float(os.environ.get("TAG_EMBEDDING_FALLBACK_HAZARD_THRESHOLD", "0.30"))
TAG_EMBEDDING_FALLBACK_CONTROL_THRESHOLD = float(os.environ.get("TAG_EMBEDDING_FALLBACK_CONTROL_THRESHOLD", "0.30"))
TAG_EMBEDDING_FALLBACK_ENERGY_THRESHOLD = float(os.environ.get("TAG_EMBEDDING_FALLBACK_ENERGY_THRESHOLD", "0.30"))
TAG_EMBEDDING_FALLBACK_ACTION_THRESHOLD = float(os.environ.get("TAG_EMBEDDING_FALLBACK_ACTION_THRESHOLD", "0.30"))
TAG_EMBEDDING_FALLBACK_HIGH_CONFIDENCE_THRESHOLD = float(os.environ.get("TAG_EMBEDDING_FALLBACK_HIGH_CONFIDENCE_THRESHOLD", "0.46"))

# Layer 3: save unmatched records and cluster a sample of still-unassigned text so
# you can discover missing dictionary terms/patterns.
TAG_UNKNOWN_DISCOVERY_ENABLED = os.environ.get("TAG_UNKNOWN_DISCOVERY_ENABLED", "true").lower() in {"1", "true", "yes", "y"}
TAG_UNKNOWN_EXPORT_SAMPLE_SIZE = int(os.environ.get("TAG_UNKNOWN_EXPORT_SAMPLE_SIZE", "5000"))
TAG_UNKNOWN_DISCOVERY_SAMPLE_SIZE = int(os.environ.get("TAG_UNKNOWN_DISCOVERY_SAMPLE_SIZE", "50000"))
TAG_UNKNOWN_CLUSTER_COUNT = int(os.environ.get("TAG_UNKNOWN_CLUSTER_COUNT", "25"))

# ---------------------------------------------------------------------------
# Step 3: risk theme discovery
# ---------------------------------------------------------------------------
EXCLUDE_TASKS_FROM_THEME_DISCOVERY = os.environ.get("EXCLUDE_TASKS_FROM_THEME_DISCOVERY", "true").lower() in {
    "1",
    "true",
    "yes",
    "y",
}
# Discover risk themes on a representative sample, then Step 4 assigns all records
# to the discovered theme centroids by cosine similarity. This keeps UMAP/HDBSCAN fast.
THEME_DISCOVERY_SAMPLE_SIZE = _optional_int(os.environ.get("THEME_DISCOVERY_SAMPLE_SIZE", "100000"))
THEME_DISCOVERY_SAMPLE_STRATEGY = os.environ.get("THEME_DISCOVERY_SAMPLE_STRATEGY", "stratified")
THEME_DISCOVERY_RANDOM_STATE = _optional_int(os.environ.get("THEME_DISCOVERY_RANDOM_STATE", "42"))
CLUSTER_ALGORITHM = os.environ.get("CLUSTER_ALGORITHM", "hdbscan").lower()  # hdbscan or kmeans
MIN_CLUSTER_SIZE = int(os.environ.get("MIN_CLUSTER_SIZE", "500"))
MIN_SAMPLES = int(os.environ.get("MIN_SAMPLES", "25"))
# HDBSCAN-only parameter. Increase slightly, e.g. 0.05-0.30, to merge nearby dense clusters.
CLUSTER_SELECTION_EPSILON = float(os.environ.get("CLUSTER_SELECTION_EPSILON", "0.25"))
UMAP_NEIGHBORS = int(os.environ.get("UMAP_NEIGHBORS", "75"))
UMAP_COMPONENTS = int(os.environ.get("UMAP_COMPONENTS", "10"))
UMAP_MIN_DIST = float(os.environ.get("UMAP_MIN_DIST", "0.0"))
# Use None for faster parallel UMAP, or set an integer seed for reproducibility.
UMAP_RANDOM_STATE = _optional_int(os.environ.get("UMAP_RANDOM_STATE", ""))
UMAP_N_JOBS = int(os.environ.get("UMAP_N_JOBS", "-1"))
KMEANS_CLUSTERS = int(os.environ.get("KMEANS_CLUSTERS", "80"))

# ---------------------------------------------------------------------------
# Step 4: theme assignment
# ---------------------------------------------------------------------------
THEME_SIMILARITY_THRESHOLD = float(os.environ.get("THEME_SIMILARITY_THRESHOLD", "0.25"))
ASSIGNMENT_BATCH_SIZE = int(os.environ.get("ASSIGNMENT_BATCH_SIZE", "10000"))

# ---------------------------------------------------------------------------
# Step 5: risk-state dataset
# ---------------------------------------------------------------------------
ASOF_FREQUENCY = os.environ.get("ASOF_FREQUENCY", "MS")
LOOKBACK_WINDOWS = os.environ.get("LOOKBACK_WINDOWS", "30,90,180")
PREDICTION_HORIZONS = os.environ.get("PREDICTION_HORIZONS", "30,90")
MIN_HISTORY_EVENTS = int(os.environ.get("MIN_HISTORY_EVENTS", "1"))

# ---------------------------------------------------------------------------
# Steps 6-8: model training and explanations
# ---------------------------------------------------------------------------
DEFAULT_HORIZON = int(os.environ.get("DEFAULT_HORIZON", "90"))
TEST_FRAC = float(os.environ.get("TEST_FRAC", "0.20"))
POSITIVE_QUANTILE = float(os.environ.get("POSITIVE_QUANTILE", "0.90"))
FIXED_RISK_THRESHOLD = os.environ.get("FIXED_RISK_THRESHOLD")
FIXED_RISK_THRESHOLD = None if FIXED_RISK_THRESHOLD in {None, ""} else float(FIXED_RISK_THRESHOLD)
CALIBRATION_FRAC = float(os.environ.get("CALIBRATION_FRAC", "0.20"))
CALIBRATION_METHOD = os.environ.get("CALIBRATION_METHOD", "isotonic")
EXPLANATION_SAMPLE_SIZE = int(os.environ.get("EXPLANATION_SAMPLE_SIZE", "500"))
EXPLANATION_TOP_N = int(os.environ.get("EXPLANATION_TOP_N", "8"))
EXPLAIN_TOP_PREDICTIONS = os.environ.get("EXPLAIN_TOP_PREDICTIONS", "false").lower() in {"1", "true", "yes", "y"}
RANDOM_STATE = int(os.environ.get("RANDOM_STATE", "42"))


def all_step_dirs() -> list[Path]:
    return [
        STEP_00_DIR,
        STEP_01_DIR,
        STEP_02_DIR,
        STEP_03_DIR,
        STEP_04_DIR,
        STEP_05_DIR,
        STEP_06_DIR,
        STEP_07_DIR,
        STEP_08_DIR,
    ]


def print_config_summary() -> None:
    print("Audits/tasks modeling configuration")
    print(f"  PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"  DATA_DIR: {DATA_DIR}")
    print(f"  OUTPUT_ROOT: {OUTPUT_ROOT}")
    print(f"  EMBEDDING_PROVIDER: {EMBEDDING_PROVIDER}")
    print(f"  EMBEDDING_MODEL_NAME: {EMBEDDING_MODEL_NAME}")
    print(f"  TAG_BACKEND: {TAG_BACKEND}")
    print(f"  DEFAULT_HORIZON: {DEFAULT_HORIZON}")
