# Pattern Learning Unsupervised Pipeline

This folder contains the unsupervised text-pattern learning pipeline for the first modeling task.

## Why this folder exists

The original data preparation process creates:

```text
outputs/processed/pattern_learning_records.csv
```

That file is the clean incident-level input table for text clustering. This unsupervised pipeline reads that file and writes the official clustered output:

```text
outputs/modeling/hdbscan_patterns/final/pattern_learning_clustered_records.csv
```

The supervised injury-risk classification pipeline reads that clustered file when clustering features are enabled.

## Run order

From the project root:

```bash
python src/run_data_prep.py
python src/pattern_learning_unsupervised/train_pattern_clusters_hdbscan.py
python src/injury_risk_classification/train_injury_risk_classifier.py
```

## Files

```text
config.py
```

Central place for all tunable parameters and paths.

```text
train_pattern_clusters_hdbscan.py
```

Trains Sentence Embeddings + UMAP + HDBSCAN, validates with a train/test split, and writes the final clustered record file used by the classification pipeline.

```text
score_pattern_clusters_hdbscan.py
```

Scores prepared records with a trained HDBSCAN model using `approximate_predict`.

```text
pattern_hdbscan_utils.py
```

Reusable functions for cleaning, embedding, clustering, validation metrics, output tables, and plots.

## Main outputs

```text
outputs/modeling/hdbscan_patterns/final/pattern_learning_clustered_records.csv
outputs/modeling/hdbscan_patterns/final/cluster_summary.csv
outputs/modeling/hdbscan_patterns/final/cluster_site_summary.csv
outputs/modeling/hdbscan_patterns/final/cluster_monthly_trend.csv
outputs/modeling/hdbscan_patterns/validation/validation_metrics.csv
outputs/modeling/hdbscan_patterns/artifacts/hdbscan_model.joblib
outputs/modeling/hdbscan_patterns/artifacts/umap_model.joblib
```

## Downstream connection

`src/injury_risk_classification/config.py` points to:

```text
outputs/modeling/hdbscan_patterns/final/pattern_learning_clustered_records.csv
```

When `FEATURE_SET = "with_clusters"` or `FEATURE_SET = "both"`, the classification pipeline uses that file to create clustering-derived features such as pattern counts, outlier rates, unique cluster counts, and top-cluster growth features.
