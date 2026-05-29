# Audits and Tasks Modeling

This folder contains the text-first safety risk modeling pipeline for incident, hazard, near-miss, audit, and task descriptions.

The scripts are designed so that every step can be run individually **without passing command-line arguments**. Edit `config.py` to change paths, model settings, horizons, clustering settings, or output folders.

## 1. Folder layout

```text
src/audits_tasks_modeling/
  config.py
  utils.py
  00_build_unified_text_events.py
  01_generate_embeddings.py
  02_extract_safety_tags.py
  03_discover_risk_themes.py
  04_assign_risk_themes.py
  05_build_risk_state_dataset.py
  06_train_risk_burden_model.py
  07_train_elevated_risk_classifier.py
  08_explain_risk_drivers.py
  run_end_to_end.py
  run_end_to_end.sh
  requirements.txt
```

## 2. Expected input CSVs

By default, `config.py` expects these files under the project root folder:

```text
AUDIT_VIEW.csv
INCIDENT_VIEW.csv
INCIDENTINJURY_VIEW.csv
LISTITEM_VIEW.csv
LOCATION_VIEW.csv
TASK_VIEW.csv
```

In the sandbox, the default project root is `/mnt/data`. In your own repo, the default project root is the folder that contains `src/` and `outputs/`.

To change the input location, edit this line in `config.py`:

```python
DATA_DIR = Path(os.environ.get("SAFETY_DATA_DIR", PROJECT_ROOT)).resolve()
```

or set an environment variable:

```bash
export SAFETY_DATA_DIR="/path/to/csv/folder"
```

## 3. Output folder structure

All outputs are saved under:

```text
outputs/audits_tasks_modeling/
```

with one clear subfolder per step:

```text
outputs/audits_tasks_modeling/
  00_unified_text_events/
    safety_text_event.csv.gz
    location_hierarchy.csv
    00_unified_text_events_summary.json

  01_embeddings/
    text_embeddings.npy
    text_embedding_event_ids.csv.gz
    01_embedding_summary.json
    models/

  02_safety_tags/
    safety_text_event_tagged.csv.gz
    02_safety_tag_summary.json

  03_risk_theme_discovery/
    risk_theme_library.csv
    risk_theme_centroids.npy
    discovered_theme_memberships.csv.gz
    03_risk_theme_discovery_summary.json
    models/

  04_theme_assignment/
    safety_text_event_themed.csv.gz
    risk_theme_assignments.csv.gz
    04_theme_assignment_summary.json

  05_risk_state_dataset/
    risk_state_training_data.csv.gz
    05_risk_state_dataset_summary.json

  06_risk_burden_model/
    risk_burden_predictions_h90.csv.gz
    model_evaluation_risk_burden_h90.csv
    06_risk_burden_model_summary_h90.json
    models/

  07_elevated_risk_classifier/
    elevated_risk_predictions_h90.csv.gz
    model_evaluation_elevated_risk_h90.csv
    07_elevated_risk_classifier_summary_h90.json
    models/

  08_risk_driver_explanations/
    risk_driver_explanations_h90.csv.gz
    global_feature_importance_h90.csv
    08_risk_driver_explanations_summary_h90.json
```

## 4. Install requirements

```bash
cd src/audits_tasks_modeling
pip install -r requirements.txt
```

The default embedding provider is now `sentence_transformer`, using:

```text
sentence-transformers/all-mpnet-base-v2
```

If your environment does not already have this model cached, it will need internet/model access the first time it runs.

## 5. Run one step at a time

From this folder:

```bash
cd src/audits_tasks_modeling
```

Run any step directly:

```bash
python 00_build_unified_text_events.py
python 01_generate_embeddings.py
python 02_extract_safety_tags.py
python 03_discover_risk_themes.py
python 04_assign_risk_themes.py
python 05_build_risk_state_dataset.py
python 06_train_risk_burden_model.py
python 07_train_elevated_risk_classifier.py
python 08_explain_risk_drivers.py
```

No arguments are required. Each script reads inputs and writes outputs based on `config.py`.

## 6. Run the full pipeline

Use either command:

```bash
python run_end_to_end.py
```

or:

```bash
bash run_end_to_end.sh
```

This runs steps 0 through 8 in order.

You can also run a partial range when needed:

```bash
python run_end_to_end.py --start-step 3 --end-step 8
```

## 7. Main settings in `config.py`

Common settings to change:

```python
DATA_DIR
OUTPUT_ROOT
EMBEDDING_PROVIDER
SENTENCE_TRANSFORMER_MODEL
TAG_BACKEND
MIN_CLUSTER_SIZE
MIN_SAMPLES
THEME_SIMILARITY_THRESHOLD
LOOKBACK_WINDOWS
PREDICTION_HORIZONS
DEFAULT_HORIZON
POSITIVE_QUANTILE
```

### Use SentenceTransformer embeddings, default

```python
EMBEDDING_PROVIDER = "sentence_transformer"
SENTENCE_TRANSFORMER_MODEL = "sentence-transformers/all-mpnet-base-v2"
```

### Use Azure OpenAI embeddings

```bash
export AZURE_OPENAI_ENDPOINT="..."
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_API_VERSION="2024-02-01"
export AZURE_OPENAI_EMBEDDING_DEPLOYMENT="text-embedding-3-large"
export EMBEDDING_PROVIDER="azure_openai"
python 01_generate_embeddings.py
```

### Use local TF-IDF/SVD fallback for quick testing

```bash
export EMBEDDING_PROVIDER="tfidf_svd"
python 01_generate_embeddings.py
```

## 8. Step summary

| Step | Script | Main output |
|---:|---|---|
| 0 | `00_build_unified_text_events.py` | unified text event table |
| 1 | `01_generate_embeddings.py` | text embeddings |
| 2 | `02_extract_safety_tags.py` | hazard/control/consequence tags |
| 3 | `03_discover_risk_themes.py` | risk theme library and centroids |
| 4 | `04_assign_risk_themes.py` | all records assigned to risk themes |
| 5 | `05_build_risk_state_dataset.py` | site/department/theme/as-of-date training table |
| 6 | `06_train_risk_burden_model.py` | future risk-burden regression model |
| 7 | `07_train_elevated_risk_classifier.py` | calibrated elevated-risk probability model |
| 8 | `08_explain_risk_drivers.py` | SHAP driver explanations |

## 9. Notes

- Step 2 defaults to rule-based tag extraction so it can run without an LLM key.
- Step 3 tries UMAP + HDBSCAN if installed. If they are unavailable, it falls back to MiniBatch KMeans.
- Step 6 uses LightGBM Tweedie regression when LightGBM is installed. Otherwise, it falls back to scikit-learn gradient boosting.
- Step 8 explains the Step 6 model by default. It reads the model from `06_risk_burden_model/models/` and writes explanations to `08_risk_driver_explanations/`.
