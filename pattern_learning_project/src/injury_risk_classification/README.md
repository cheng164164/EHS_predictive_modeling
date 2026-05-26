# Any-Injury Risk Ranking MVP

This package builds a site + department + month risk ranking model.

Default target:

```text
future_any_injury_3m
```

Meaning:

```text
1 = the same site/department has at least one injury incident in the next 3 calendar months
0 = no injury incident occurs in the next 3 calendar months
```

The model output should be used as an operational **risk ranking**, not as a hard guarantee that an injury will occur.

## Recommended run order

From the project root:

```bash
python src/injury_risk_classification/build_classification_dataset.py
python src/injury_risk_classification/train_injury_risk_classifier.py
python src/injury_risk_classification/score_current_site_risk.py
```

## Feature-set comparison

`FEATURE_SET = "both"` trains:

```text
baseline
with_clusters
```

`FEATURE_SET = "experiments"` trains baseline plus the experiment matrix in `config.py`.

Training now writes these comparison files under:

```text
outputs/ml/injury_risk_classification/runs/<run_id>/
```

Important files:

```text
model_comparison_holdout_test.csv
feature_set_comparison_ranked.csv
feature_set_recommendation.json
run_summary.json
```

Recommended comparison metrics:

```text
PR-AUC
recall_at_top_10pct
precision_at_top_10pct
lift_at_top_10pct
false_negative
```

Do not rely on accuracy as the main metric because the target is imbalanced.

## Leakage validation

Training runs leakage checks for every feature set and writes:

```text
<run_id>/<feature_set>/leakage_validation/leakage_validation_report.csv
<run_id>/<feature_set>/leakage_validation/leakage_validation_summary.json
```

The checks flag selected model features that:

```text
match forbidden future/target column-name patterns
exactly equal the target
have near-perfect target correlation
have near-perfect single-feature AUC
```

`FAIL_ON_LEAKAGE = False` by default so the run completes and gives you a report. Set it to `True` after reviewing the checks.

## Clean pre-encoding model input tables

Training now saves the exact raw feature dataframe used before sklearn encoding/scaling:

```text
<run_id>/<feature_set>/raw_model_input_features/model_input_features_raw_all_eligible_rows.parquet
<run_id>/<feature_set>/raw_model_input_features/model_input_features_raw_train_period.parquet
<run_id>/<feature_set>/raw_model_input_features/model_input_features_raw_holdout_test.parquet
```

CSV previews are saved beside them. These files contain the selected raw model features plus ID columns and target. They do **not** contain one-hot encoded categorical columns.

The selected feature catalog is also saved:

```text
<run_id>/<feature_set>/model_feature_catalog_raw_columns.csv
```

## Dashboard and operational ranking outputs

Scoring writes the usual ranked output plus dashboard-ready files under:

```text
outputs/ml/injury_risk_classification/predictions/
```

Important files:

```text
current_any_injury_risk_scores.csv
current_any_injury_risk_dashboard.csv
operational_review_queue.csv
risk_tier_summary.csv
site_risk_rollup.csv
```

Dashboard columns include:

```text
risk_score
risk_rank
risk_percentile
risk_tier
top_driver_features
top_themes_last_3m
recent_pattern_increases
pattern_trend_direction
near_miss_trend_direction
hazard_trend_direction
injury_trend_direction
overdue_open_task_count
recommended_action
notification_audience
review_due_days
```

Default tiers:

```text
Critical  = top 5%
High      = top 10%
Watchlist = top 25%
Monitor   = all others
```

The operational queue includes Critical, High, and Watchlist rows by default.

## Main configuration switches

```python
TARGET_TYPE = "any_injury"
FEATURE_SET = "both"
THRESHOLD_STRATEGY = "top_percent"
TOP_PERCENT_THRESHOLD = 0.10
SAVE_MODEL_INPUT_FEATURE_TABLES = True
LEAKAGE_VALIDATION_ENABLED = True
SAVE_DASHBOARD_OUTPUTS = True
```

Pattern feature defaults use aggregate pattern activity and broad theme features rather than detailed per-cluster IDs.
