# Safety Retrieval Agent MVP1

This is a local, non-Azure-Search implementation of the **Safety Retrieval Agent**.
It uses a free/open transformer embedding model plus FAISS for fast similarity search.

Default output folder:

```text
outputs/safety retrieval agent
```

## MVP1 features implemented

1. Risk pattern / theme classification
2. Historical severe-injury similarity
3. Similar historical event recall
4. Risk factor extraction
5. Recommended prevention actions
6. Missing-information prompt

## Important change: scripts run without command-line arguments

All runtime paths and tunable parameters live in:

```text
src/safety_retrieval_agent/config.py
```

Each script can be run directly:

```bash
python scripts/00_build_unified_text_events.py
python scripts/00_prepare_knowledge_base.py
python scripts/01_build_faiss_indexes.py
python scripts/02_run_mvp_recommendations.py
python scripts/predict_single_event.py
python scripts/run_end_to_end.py
```

To change input paths, model name, sample size, date range, FAISS type, or query text, edit `config.py` or set the documented environment variables used by `config.py`.

## Default embedding model

The default model is:

```text
BAAI/bge-m3
```

It was selected because it is a strong free/open retrieval model and supports multilingual safety text. The code will use `FlagEmbedding` for BGE-M3 if installed. It can also run with `sentence-transformers` models such as:

```text
BAAI/bge-large-en-v1.5
BAAI/bge-base-en-v1.5
intfloat/multilingual-e5-base
```

Change the model in `config.py`:

```python
embedding_model_name = "BAAI/bge-m3"
embedding_backend = "auto"
```

## Install

```bash
pip install -r requirements.txt
```

For CPU-only local builds, `faiss-cpu` is enough. If your environment already has PyTorch but not `sentence-transformers`, install only the missing packages.

## Input options

### Option A: unified event file already exists

Set this path in `config.py`:

```python
input_event_file = Path("path/to/safety_text_event.csv.gz")
```

Then run:

```bash
python scripts/00_prepare_knowledge_base.py
python scripts/01_build_faiss_indexes.py
python scripts/02_run_mvp_recommendations.py
```

### Option B: unified event file is not available

Put the raw Velocity/Accelerate export files in the folder configured by:

```python
raw_data_dir = project_root / "data" / "raw"
```

Default expected file names:

```text
INCIDENT_VIEW.csv
INCIDENTINJURY_VIEW.csv
AUDIT_VIEW.csv
TASK_VIEW.csv
LOCATION_VIEW.csv
LISTITEM_VIEW.csv
```

Then run:

```bash
python scripts/00_build_unified_text_events.py
```

This creates:

```text
outputs/safety retrieval agent/data/safety_text_event.csv.gz
outputs/safety retrieval agent/data/location_hierarchy.csv
```

`run_end_to_end.py` will automatically call the unified builder if `input_event_file` is missing and `run_unified_builder_if_missing = True` in `config.py`.

## Run a smoke test

For a smoke test, edit `config.py` first:

```python
max_records = 5000
recommendation_sample_size = 20
```

Then run:

```bash
python scripts/run_end_to_end.py
```

## Run the full build

In `config.py`, set:

```python
max_records = None
recommendation_sample_size = 100
```

Then run:

```bash
python scripts/run_end_to_end.py
```

## Run one configured event

After indexes are built, edit these values in `config.py`:

```python
single_event_text = "Forklift reversed near a loading dock and almost struck a pedestrian walking through the area."
single_event_site = None
single_event_department = "Warehouse"
single_event_source_type = "near_miss"
single_event_id = "manual_query_001"
```

Then run:

```bash
python scripts/predict_single_event.py
```

## Main outputs

```text
outputs/safety retrieval agent/data/safety_text_event.csv.gz
outputs/safety retrieval agent/data/location_hierarchy.csv
outputs/safety retrieval agent/data/safety_knowledge_base.pkl
outputs/safety retrieval agent/data/safety_knowledge_base_sample.csv
outputs/safety retrieval agent/data/safety_knowledge_base_with_themes.pkl
outputs/safety retrieval agent/data/safety_knowledge_base_with_themes.csv.gz
outputs/safety retrieval agent/data/safety_theme_profiles.pkl
outputs/safety retrieval agent/data/safety_theme_profiles.csv
outputs/safety retrieval agent/faiss_indexes/*.faiss
outputs/safety retrieval agent/recommendations/mvp1_recommendation_summary.csv
outputs/safety retrieval agent/recommendations/mvp1_recommendation_results.jsonl
outputs/safety retrieval agent/recommendations/single_event_analysis.json
```

## Design notes

- This project does **not** use Azure AI Search.
- Similarity search is local FAISS.
- Existing theme columns are reused if available.
- If no theme columns exist, themes are discovered with MiniBatchKMeans over transformer embeddings.
- Risk factors are extracted from the new report plus retrieved evidence using local TF-IDF keyphrase mining.
- Prevention actions are recommended from retrieved historical task/corrective-action records and safe observations.
- Missing-information prompts are generated locally to improve incident form quality.
- The unified builder is a self-contained script adapted from the prior unified-event step so this project can run even when the unified file is not prebuilt.

## Production cautions

- Review similarity thresholds before deployment. Embedding cosine scores vary by model.
- Full BGE-M3 embeddings for hundreds of thousands of records may require several GB of memory/disk.
- If the model is not cached, the first run needs access to Hugging Face to download model files.
- For Velocity/Accelerate integration, wrap `SafetyRetrievalAgent.analyze_event()` in an API endpoint.
