# Injury Similarity ML Workflow

Copy `src/injury_similarity` into your existing project. Your current data-preparation scripts are not changed.

The scripts read from:

```text
outputs/processed/incident_injury_all_records.csv
outputs/processed/pattern_learning_records.csv  # optional default scoring source
```

All ML outputs are written under your original `outputs` folder.

## Recommended end-to-end run

```bash
python src/injury_similarity/run_end_to_end.py
```

This performs the best-practice sequence:

```text
1. Temporal validation
   Fit on older historical injury records and evaluate on newer injury records.

2. Final model fitting
   After validation, fit the final production model on all historical injury records.

3. Prediction
   Score non-injury near-miss/hazard candidates using only the saved final model.
```

## Individual commands

```bash
# Validation only: train on older records, evaluate on newer held-out records
python src/injury_similarity/evaluate_injury_similarity.py

# Validation + final model training, but no prediction
python src/injury_similarity/train_injury_similarity.py

# Final model training only
python src/injury_similarity/train_final_model.py

# Prediction only using saved final model
python src/injury_similarity/predict_injury_similarity.py
```

## Python functions

```python
from injury_similarity.core import (
    run_temporal_validation,
    train_final_model,
    predict_injury_similarity,
    run_training_workflow,
)

validation_metrics = run_temporal_validation()
final_metadata = train_final_model()
prediction_summary = predict_injury_similarity()
```

## Output structure

```text
outputs/ml/injury_similarity/
  validation/
    temporal_validation_metrics.json
    thresholds_from_train_split.json
    temporal_holdout_query_summary.csv
    temporal_holdout_top_matches.csv
    model_train_split/
      tfidf_vectorizer.joblib
      nearest_neighbors.joblib
      reference_matrix.joblib
      reference_records.csv
      thresholds.json
      metadata.json

  final_model/
    tfidf_vectorizer.joblib
    nearest_neighbors.joblib
    reference_matrix.joblib
    reference_records.csv
    thresholds.json
    metadata.json
    training_summary.json

  predictions/
    injury_similarity_query_summary.csv
    injury_similarity_top_matches.csv
    prediction_run_summary.json

  workflow_summary.json
```

## Important distinction

The validation model in `validation/model_train_split` is not the production model. It is fitted only on older training records and exists to document the temporal holdout test.

The final model in `final_model` is fitted after validation using all available historical injury records. Prediction loads only this final model and does not refit anything.

## Algorithm

```text
TF-IDF vectorizer + cosine nearest-neighbor retrieval
```

This is a similarity/retrieval model. It is not a supervised PSIF classifier.

## Feature usage

The vectorizer uses early incident text, primarily:

```text
title
description
activity_during_incident
equipment
vehicle
off_premises_location
other_process
other_activity
```

Outcome fields such as `injury_count`, `severe_actual`, `lost_time_any`, `restricted_time_any`, `fatality_any`, `inpatient_any`, and `emergency_room_any` are not used as model text features. They are used only as context on matched historical injury records.



# No-match validation control update

This update keeps the existing temporal injury holdout validation unchanged and adds an optional negative-control validation set.

## Why this was added

The original temporal validation set contains only injury records. Because every query is an injury record, it is expected that most or all queries have at least a weak historical injury match. That validates injury-to-injury retrieval, but it does not demonstrate rejection behavior for irrelevant inputs.

## What changed

Only `src/injury_similarity/config.py` and `src/injury_similarity/core.py` changed.

### config.py

Added no-match validation parameters:

- `ENABLE_NO_MATCH_VALIDATION_CONTROLS`
- `NO_MATCH_REAL_CONTROL_SAMPLE_SIZE`
- `NO_MATCH_REAL_CONTROL_CANDIDATE_POOL_SIZE`
- `NO_MATCH_REAL_CONTROLS_HOLDOUT_PERIOD_ONLY`
- `SYNTHETIC_NO_MATCH_CONTROL_RECORDS`

### core.py

Added helper functions:

- `tag_validation_queries()`
- `make_synthetic_no_match_controls()`
- `make_real_non_injury_no_match_controls()`
- `make_no_match_validation_controls()`
- `no_match_control_metrics()`

Updated `retrieve()` to carry validation-control metadata into output files.

Updated `run_temporal_validation()` to write additional files:

- `no_match_control_query_summary.csv`
- `no_match_control_top_matches.csv`
- `temporal_holdout_query_summary_with_no_match_controls.csv`
- `temporal_holdout_top_matches_with_no_match_controls.csv`
- `no_match_control_metrics.json`

The original injury-only files are still generated:

- `temporal_holdout_query_summary.csv`
- `temporal_holdout_top_matches.csv`
- `temporal_validation_metrics.json`
- `thresholds_from_train_split.json`

## Important interpretation

Real non-injury records are not automatically no-match records. Near misses and hazards may be highly relevant if they resemble historical injuries. For that reason, the real non-injury controls are selected only after scoring a candidate pool and keeping records below the weak-match threshold.

Synthetic controls are intentionally off-domain and are included only to demonstrate that irrelevant text can produce `no_match`.
