# Injury Event Similarity Modeling Add-on

Copy the `src/injury_similarity` folder into your existing `pattern_learning_project/src` folder.

This add-on does not modify your current data preparation scripts. It reuses the prepared file that your existing pipeline already writes:

```text
outputs/processed/incident_injury_all_records.csv
```

If that file is missing, the training script attempts to run your existing:

```text
src/run_data_prep.py
```

## ML algorithm

The recommended first model is:

```text
TF-IDF text vectorization + cosine nearest-neighbor similarity search
```

The searchable reference library is:

```text
all historical injury records where injury_count > 0
```

That means it includes both severe and non-severe injuries. The model is not a PSIF classifier and does not use injury outcome fields as input features. It only carries outcome fields such as `severe_actual`, `lost_time_any`, and `restricted_time_any` as context on returned matches.

## Run end-to-end

From the project root:

```bash
python src/injury_similarity/run_end_to_end.py
```

This runs training, validation/testing, and batch prediction with no required path arguments.

## Run only training

```bash
python src/injury_similarity/train_injury_similarity.py
```

## Run only evaluation/testing

```bash
python src/injury_similarity/evaluate_injury_similarity.py
```

## Run only batch prediction

```bash
python src/injury_similarity/predict_injury_similarity.py
```

## Outputs

All outputs are written under your existing `outputs` folder:

```text
outputs/ml/injury_similarity/
  model/
    tfidf_vectorizer.joblib
    nearest_neighbors.joblib
    reference_records.csv
    thresholds.json
    metadata.json
    training_summary.json
  evaluation/
    metrics.json
    leave_one_out_top_matches.csv
    temporal_holdout_top_matches.csv
    candidate_top1_distribution.csv
  predictions/
    injury_similarity_query_summary.csv
    injury_similarity_top_matches.csv
    prediction_run_summary.json
```

## Prediction output logic

The prediction script scores non-injury near-miss and hazard records by default. It first tries to use:

```text
outputs/processed/pattern_learning_records.csv
```

If that file is not available, it falls back to `incident_injury_all_records.csv` and filters to non-injury near-miss/hazard candidates.

A query receives:

```text
strong_match
possible_match
weak_match
no_match
```

For a `no_match` query, the top matches file does not return historical injury rows for that query. The query summary still keeps one row so you can see that the record was scored.

## Notebook usage

From a notebook, after making sure the project `src` folder is importable:

```python
from injury_similarity.run_end_to_end import run_end_to_end

result = run_end_to_end()
```

Training only:

```python
from injury_similarity.train_injury_similarity import train_model

training_summary = train_model()
```

Scoring a dataframe later:

```python
from injury_similarity.predict_injury_similarity import score_records

summary_df, matches_df = score_records(new_records_df)
```
