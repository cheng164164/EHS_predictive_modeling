# Pattern Learning - Safety Incident Data Preparation

This starter project prepares Velocity/Accelerate EHS exports for the first modeling task: near-miss and hazard pattern discovery.

## Main objective

Create reusable analytical datasets that can later feed text embeddings, clustering, severe-injury similarity scoring, and site/department risk features.

## Expected raw files

Place these files in `data/raw/`, or place them in the same folder where you run the script. You can also point `--input-dir` to another folder if needed:

- `INCIDENT_VIEW.csv`
- `INCIDENTINJURY_VIEW.csv`
- `LISTITEM_VIEW.csv`
- `LOCATION_VIEW.csv`
- `TASK_VIEW.csv`
- `AUDIT_VIEW.csv`

## Run

No arguments are required. From the project folder, run:

```bash
python src/run_data_prep.py
```

By default, the script auto-detects the six raw CSV files from this search order:

1. `PATTERN_LEARNING_INPUT_DIR` environment variable
2. `data/raw/` under the project folder
3. `data/raw/` under the current working directory
4. the current working directory
5. `/mnt/data` for this ChatGPT sandbox

The default output folder is:

```text
outputs/
```

You can still override the defaults when needed:

```bash
python src/run_data_prep.py --input-dir data/raw --output-dir outputs --reference-date 2026-05-20
```

The default run writes the focused modeling tables and EDA outputs only. To also write the very large enriched source tables, run:

```bash
python src/run_data_prep.py --write-large-processed
```

## Key outputs

Processed data:

- `outputs/processed/incident_enriched.csv`
- `outputs/processed/injury_agg.csv`
- `outputs/processed/location_hierarchy.csv`
- `outputs/processed/task_enriched.csv`
- `outputs/processed/audit_enriched.csv`
- `outputs/processed/pattern_learning_records.csv`
- `outputs/processed/site_department_month_features.csv`

EDA:

- `outputs/eda/eda_summary.md`
- `outputs/eda/tables/*.csv`
- `outputs/eda/plots/*.png`

## Modeling handoff

The first ML model should start from `pattern_learning_records.csv`. Recommended initial columns:

- `incident_id`
- `incident_date`
- `incident_category_name`
- `location_id`
- `site_name`
- `department_name`
- `title`
- `description`
- `ml_text_early`
- `ml_text_full`
- `text_word_count_early`
- `incident_month`

For unsupervised pattern discovery, use `ml_text_early` first to avoid relying on post-investigation fields that may not exist when a new record is created.

## Fast EDA option

For a quick first-pass EDA without building the full enriched modeling tables, run:

```bash
PYTHONPATH=src python src/generate_fast_eda.py
```

This produces `outputs/eda/eda_summary.md`, EDA tables, and plots using a minimal set of columns. It is intended for quick validation; use `run_data_prep.py` for the full modeling data foundation.

## Recommended first ML implementation after this prep

1. Read `outputs/processed/pattern_learning_records.csv`.
2. Use `ml_text_early` as the text field for embeddings or TF-IDF.
3. Keep `incident_id`, `incident_date`, `incident_category_name`, `site_name`, and `department_name` as metadata for explainability.
4. Start with a baseline TF-IDF + MiniBatchKMeans model before moving to sentence embeddings + HDBSCAN/BERTopic.
5. Use time-based validation for trend outputs; do not tune cluster decisions on future months.
