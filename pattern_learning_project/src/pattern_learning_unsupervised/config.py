"""Configuration for the unsupervised Pattern Learning pipeline.

Edit this file to change paths, model choices, validation settings, or output
behavior. The scripts in this folder can be run directly from the project root:

    python src/pattern_learning_unsupervised/train_pattern_clusters_hdbscan.py
    python src/pattern_learning_unsupervised/score_pattern_clusters_hdbscan.py

This package is the explicit bridge between the original data preparation output
and the supervised injury-risk classification package:

    outputs/processed/pattern_learning_records.csv
        -> HDBSCAN pattern learning
        -> outputs/modeling/hdbscan_patterns/final/pattern_learning_clustered_records.csv
        -> src/injury_risk_classification feature engineering
"""

from __future__ import annotations

from pathlib import Path

# This file lives at:
#   pattern_learning_project/src/pattern_learning_unsupervised/config.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Input created by the original data-preparation pipeline.
# This is the clean, one-row-per-near-miss/hazard table used for unsupervised
# text clustering.
PATTERN_LEARNING_RECORDS_PATH = OUTPUT_DIR / "processed" / "pattern_learning_records.csv"

# Official output of the unsupervised clustering task. This file is consumed by
# src/injury_risk_classification when clustering features are enabled.
HDBSCAN_OUTPUT_DIR = OUTPUT_DIR / "ml" / "hdbscan_patterns"
CLUSTERED_PATTERN_RECORDS_PATH = HDBSCAN_OUTPUT_DIR / "final" / "pattern_learning_clustered_records.csv"

# Text/ID columns.
TEXT_COL = "ml_text_early"
ID_COL = "incident_id"
MIN_WORDS = 3

# Sentence embedding model. Use a local path here if your runtime cannot access
# Hugging Face directly.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEVICE = "auto"  # auto, cpu, cuda, cuda:0
BATCH_SIZE = 128

# Optional sample size for fast testing. Use None for full production training.
MAX_RECORDS = None

# Validation split for unsupervised production scoring behavior.
# "time" means train on older records and approximate-predict newer records.
SPLIT_MODE = "time"  # "time" or "random"
TEST_SIZE = 0.20
RANDOM_STATE = 42

# ------------------------------------------------------------------
# UMAP dimensionality reduction before HDBSCAN
# ------------------------------------------------------------------

USE_UMAP = True
# Enable/disable UMAP preprocessing.
# True  -> reduce embedding dimensions before clustering.
# False -> run HDBSCAN directly on the original embeddings.

UMAP_N_NEIGHBORS = 50
# Number of neighboring points UMAP considers when learning
# the manifold structure.
# Larger values:
#   - preserve more global structure
#   - produce broader, smoother clusters
# Smaller values:
#   - preserve more local detail
#   - may create smaller or fragmented clusters

UMAP_N_COMPONENTS = 15
# Target number of dimensions after reduction.
# Lower values:
#   - faster clustering
#   - more compression
#   - possible information loss
# Higher values:
#   - preserve more embedding information
#   - slower clustering
# Typical range for text embeddings: 5–50.

UMAP_MIN_DIST = 0.0
# Controls how tightly UMAP packs points together.
# Lower values (closer to 0):
#   - tighter and denser clusters
#   - better separation for clustering tasks
# Higher values:
#   - more spread-out embeddings
#   - smoother manifold representation

UMAP_METRIC = "cosine"
# Distance metric used by UMAP when comparing embeddings.
# "cosine" is commonly used for text embeddings because
# semantic similarity is angle-based rather than magnitude-based.


# ------------------------------------------------------------------
# HDBSCAN density clustering
# ------------------------------------------------------------------

MIN_CLUSTER_SIZE = 100
# Minimum number of points required to form a cluster.
# Larger values:
#   - fewer but more stable clusters
#   - more points labeled as noise/outliers
# Smaller values:
#   - more granular clusters
#   - may introduce unstable or noisy clusters

MIN_SAMPLES = 3
# Controls cluster conservativeness and outlier sensitivity.
# Higher values:
#   - stricter density requirements
#   - more outliers/noise points
# Lower values:
#   - easier cluster formation
#   - potentially noisier results
# If None, HDBSCAN defaults to MIN_CLUSTER_SIZE.

HDBSCAN_METRIC = "euclidean"
# Distance metric used by HDBSCAN after dimensionality reduction.
# "euclidean" is typically recommended when using UMAP outputs,
# since UMAP projects embeddings into Euclidean space.

CLUSTER_SELECTION_METHOD = "eom"
# Strategy for extracting clusters from the hierarchy.
#
# "eom"  (Excess of Mass):
#   - prefers larger, stable clusters
#   - good default for most applications
#
# "leaf":
#   - returns smaller, more fine-grained clusters
#   - useful when detailed subtopics are desired

CLUSTER_SELECTION_EPSILON = 0.15
# Distance threshold for merging nearby clusters.
# Larger values:
#   - merge more clusters together
#   - fewer overall clusters
# Smaller values:
#   - preserve finer cluster separation
#   - more distinct clusters
# Often left at 0.0 unless clusters are overly fragmented.

# Second-level theme grouping on top of HDBSCAN clusters.
#
# Purpose:
#   HDBSCAN can intentionally produce detailed sub-pattern clusters. Theme
#   grouping learns a broader layer above those clusters:
#
#       record -> cluster -> theme
#
# Generic design:
#   Themes are learned from cluster centroids in vector space. This avoids
#   hand-curated keyword rules, result-specific stop-word tweaks, or supervised
#   labels. The label text is created only after grouping for explanation; it
#   does not influence the grouping itself.
ENABLE_THEMES = True

# Currently supported: "agglomerative". This uses hierarchical clustering on
# cluster centroids, which is deterministic for a fixed training dataset and
# config.
THEME_METHOD = "agglomerative"

# Which vector space to use when computing cluster centroids for theme grouping.
# "embedding" is recommended because it uses the original semantic sentence
# embedding space. "umap" groups clusters in the reduced UMAP space.
THEME_CENTROID_SPACE = "embedding"  # "embedding" or "umap"

# Hierarchical theme tuning. Smaller distance threshold -> more themes. Larger
# distance threshold -> fewer, broader themes. With cosine distance, 0.25 means
# clusters are merged when their centroid similarity is roughly 0.75 or higher.
# Set THEME_N_CLUSTERS to an integer to force a fixed number of themes; leave as
# None to let the distance threshold determine the number of themes.
THEME_DISTANCE_THRESHOLD = 0.35
THEME_N_CLUSTERS = None
THEME_METRIC = "cosine"
THEME_LINKAGE = "average"

# Use HDBSCAN membership strength as a generic reliability weight when computing
# each cluster centroid. This weights clear cluster members slightly more than
# weak/borderline members. It does not weight words or labels.
THEME_USE_MEMBERSHIP_WEIGHTS = True

# Ignore clusters smaller than this when fitting themes. Normally keep this at 1
# because HDBSCAN MIN_CLUSTER_SIZE already controls minimum cluster size.
THEME_MIN_RECORDS_PER_CLUSTER = 1

# Labeling and reporting only. These values do not affect model fitting.
THEME_TOP_TERMS_N = 12
THEME_TOP_CLUSTERS_N = 8

# ------------------------------------------------------------------
# Contextual cluster/theme descriptions
# ------------------------------------------------------------------
# These settings only affect the reporting layer. They do not change embeddings,
# UMAP, HDBSCAN, or theme grouping. The training script will still output the
# original keyword labels, but it will also add phrase-based labels and
# short stakeholder-readable descriptions.
ENABLE_CONTEXTUAL_LABELS = True

# If True, cluster_label/theme_label use the richer phrase-based label. The old
# keyword-only labels are preserved in keyword_cluster_label and
# keyword_theme_label.
USE_CONTEXTUAL_LABEL_AS_PRIMARY = True

# Multi-word phrase extraction settings. These phrases are used to generate
# richer labels/descriptions from the cluster text itself, without applying
# any manually curated hazard taxonomy.
PHRASE_TOP_N = 20
PHRASE_MIN_DF = 2
PHRASE_MAX_FEATURES = 30000
PHRASE_NGRAM_MAX = 4

# Representative records included in cluster/theme descriptions.
REPRESENTATIVE_TEXTS_PER_SUMMARY = 5
SUMMARY_MAX_SNIPPET_CHARS = 220

# Metrics and output behavior.
METRIC_SAMPLE_SIZE = 10000
FIT_FINAL = True
SAVE_EMBEDDINGS = False
SAVE_SENTENCE_MODEL = False

# Scoring script defaults. This can be changed to a file containing new records.
SCORE_INPUT_FILE = PATTERN_LEARNING_RECORDS_PATH
SCORE_OUTPUT_FILE = HDBSCAN_OUTPUT_DIR / "scored_new_records.csv"
SCORE_REJECTED_OUTPUT_FILE = HDBSCAN_OUTPUT_DIR / "scored_rejected_records.csv"
