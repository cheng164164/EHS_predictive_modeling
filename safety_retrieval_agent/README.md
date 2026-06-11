# Safety Retrieval Agent MVP1

This is a local, non-Azure-Search implementation of the **Safety Retrieval Agent**. It uses local retrieval only: FAISS semantic vector search, BM25 keyword search, or hybrid FAISS+BM25 search. No Azure AI Search service is required.

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

## Retrieval modes

Set the mode in `src/safety_retrieval_agent/config.py`:

```python
retrieval_mode = "hybrid"  # "faiss", "bm25", or "hybrid"
```

Available modes:

```text
faiss   = transformer embeddings + FAISS semantic search only
bm25    = BM25 keyword search only
hybrid  = FAISS + BM25 merged with reciprocal-rank fusion
```

Hybrid mode is the recommended default for EHS text because it captures both semantic matches, such as `lift truck almost struck employee`, and exact terms, such as `LOTO`, `PPE`, `forklift`, `confined space`, or chemical/equipment names.

Candidate-pool settings are also in `config.py`:

```python
faiss_candidate_k = 75
bm25_candidate_k = 75
hybrid_rrf_k = 60
```

The final number of returned records is still controlled by:

```python
top_k_severe_injuries = 5
top_k_similar_events = 15
top_k_corrective_actions = 8
top_k_safe_practices = 5
```

`01_build_faiss_indexes.py` now builds both FAISS artifacts and BM25 artifacts. The script name is retained for compatibility with previous versions.

## Scripts run without command-line arguments

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

## Embedding model behavior

Primary model:

```text
BAAI/bge-m3
```

Fallback model:

```text
Qwen/Qwen3-Embedding-0.6B
```

The default backend is:

```python
embedding_backend = "sentence_transformers"
```

This keeps BGE-M3 as the primary model but avoids the `FlagEmbedding` loader path that caused the error:

```text
XLMRobertaModel.__init__() got an unexpected keyword argument 'dtype'
```

If BGE-M3 fails to load or fails on the first encode batch, the index builder automatically switches to Qwen3-Embedding-0.6B. The actual model used is saved in:

```text
outputs/safety retrieval agent/embeddings/embedding_model_metadata.json
outputs/safety retrieval agent/models/embedding_model_metadata.json
outputs/safety retrieval agent/faiss_indexes/*_metadata.json
```

At query time, `SafetyRetrievalAgent` loads the embedding model recorded in FAISS metadata. This prevents invalid searches caused by building indexes with one model and querying with another model.

Important: after changing embedding model settings, either set:

```python
build_reuse_embeddings = False
```

or delete these folders before rebuilding:

```text
outputs/safety retrieval agent/embeddings
outputs/safety retrieval agent/faiss_indexes
outputs/safety retrieval agent/models
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

## Batch recommendations: where query records come from

`02_run_mvp_recommendations.py` uses this logic:

1. If `recommendation_query_file` in `config.py` points to a CSV, it loads that file.
2. Otherwise, it samples records from the prepared knowledge base.

A manual query CSV should contain at least one of these text columns:

```text
query_text
retrieval_text
clean_text
description
title
```

Recommended columns:

```text
query_id,query_text,site,department,source_type
```

If no query file is configured, the script samples hazard/near-miss/unsafe-observation records from:

```text
outputs/safety retrieval agent/data/safety_knowledge_base_with_themes.pkl
```

or, if that does not exist yet:

```text
outputs/safety retrieval agent/data/safety_knowledge_base.pkl
```

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
outputs/safety retrieval agent/embeddings/event_embeddings.npy
outputs/safety retrieval agent/embeddings/embedding_model_metadata.json
outputs/safety retrieval agent/faiss_indexes/*.faiss
outputs/safety retrieval agent/faiss_indexes/*_metadata.json
outputs/safety retrieval agent/bm25_indexes/bm25_vectorizer.joblib
outputs/safety retrieval agent/bm25_indexes/bm25_matrix_csc.joblib
outputs/safety retrieval agent/bm25_indexes/*_row_ids.npy
outputs/safety retrieval agent/recommendations/mvp1_recommendation_summary.csv
outputs/safety retrieval agent/recommendations/mvp1_recommendation_results.jsonl
outputs/safety retrieval agent/recommendations/single_event_analysis.json
```

## Design notes

- This project does **not** use Azure AI Search.
- Similarity search is local and configurable: FAISS only, BM25 only, or hybrid FAISS+BM25 reciprocal-rank fusion.
- Existing theme columns are reused if available.
- If no theme columns exist, themes are discovered with MiniBatchKMeans over transformer embeddings.
- Risk factors are extracted from the new report plus retrieved evidence using local TF-IDF keyphrase mining.
- Prevention actions are recommended from retrieved historical task/corrective-action records and safe observations.
- Missing-information prompts are generated locally to improve incident form quality.
- The unified builder is a self-contained script adapted from the prior unified-event step so this project can run even when the unified file is not prebuilt.

## Production cautions

- Review similarity thresholds before deployment. Embedding cosine scores vary by model.
- Full BGE-M3 embeddings for hundreds of thousands of records may require several GB of memory/disk. BM25 artifacts also require disk space because they store a sparse term matrix.
- If BGE-M3 falls back to Qwen3, keep the generated metadata with the index and query with the same model.
- If the model is not cached, the first run needs access to Hugging Face to download model files.
- For Velocity/Accelerate integration, wrap `SafetyRetrievalAgent.analyze_event()` in an API endpoint.


### Embedding scope filter

`01_build_faiss_indexes.py` does not embed every row in the unified table. It first filters the prepared knowledge base to the configured MVP retrieval roles in `config.py`:

- injuries and severe injuries
- hazard identifications
- near misses
- unsafe observations
- safe observations
- corrective actions / open corrective actions / overdue corrective actions
- generic `audit_observation` rows only when `description` is non-empty by default

The filter summary is saved to:

```text
outputs/safety retrieval agent/data/embedding_scope_summary.json
outputs/safety retrieval agent/data/embedding_scope_counts_by_role.csv
```

Use these settings to change the scope:

```python
embedding_source_roles = (...)
include_generic_audit_observations = True
require_generic_audit_description = True
generic_audit_description_min_chars = 1
```
