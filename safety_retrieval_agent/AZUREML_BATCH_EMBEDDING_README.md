# Azure ML batch embedding/index build update

This patch adds a resumable Azure ML CPU workflow for the Safety Retrieval Agent.
It does **not** modify these existing scripts:

- `scripts/00_build_unified_text_events.py`
- `scripts/00_prepare_knowledge_base.py`
- `scripts/01_build_faiss_indexes.py`
- `scripts/02_run_mvp_recommendations.py`

## New workflow

### 1. Prepare data locally / interactively once

Run your existing prep scripts if needed:

```bash
python scripts/00_build_unified_text_events.py
python scripts/00_prepare_knowledge_base.py
```

This creates the prepared knowledge base under your local/project output folder,
for example:

```text
outputs/safety retrieval agent/data/safety_knowledge_base.pkl
```

### 2. Submit Azure ML embedding job from VS Code

Install submit-only dependencies in the environment where you run the submitter:

```bash
pip install -r requirements_azureml_submit.txt
```

Then submit the long-running embedding job:

```bash
python scripts/submit_azureml_embedding_job.py
```

This job runs `scripts/01a_generate_embedding_chunks.py` on the Azure ML compute
cluster configured in `config.py`.

The job writes resumable chunk files:

```text
<configured Azure ML output URI>/embeddings/chunks/chunk_00000.npy
<configured Azure ML output URI>/embeddings/chunks/chunk_00001.npy
...
<configured Azure ML output URI>/embeddings/embedding_chunks_manifest.csv
<configured Azure ML output URI>/embeddings/embedding_chunk_run_summary.json
```

If the job fails or is interrupted, rerun the same submit script. Completed chunk
files are skipped.

### 3. Submit Azure ML index job after embeddings are complete

```bash
python scripts/submit_azureml_index_job.py
```

This job runs `scripts/01b_build_indexes_from_chunks.py`. It merges the chunked
embeddings and builds:

```text
<configured Azure ML output URI>/embeddings/event_embeddings.npy
<configured Azure ML output URI>/faiss_indexes/*.faiss
<configured Azure ML output URI>/bm25_indexes/*
<configured Azure ML output URI>/data/safety_knowledge_base_with_themes.pkl
<configured Azure ML output URI>/data/safety_theme_profiles.pkl
```

## Main config settings

All settings are in:

```text
src/safety_retrieval_agent/config.py
```

Key Azure ML defaults added by this patch:

```python
aml_subscription_id = "7f07baf7-8bba-4b88-b300-74ba5b15f52d"
aml_resource_group = "EHS-Safety"
aml_workspace_name = "ehs-safety-aml"
aml_compute_name = "Tan-dev-cluster"
aml_output_uri = "azureml://datastores/workspaceblobstore/paths/Users/tan.cheng/EHS_predictive_modeling/safety_retrieval_agent/outputs/safety retrieval agent/"
aml_cpu_cores_per_node = 8
embedding_chunk_size = 5000
```

If your workspace does not accept spaces in datastore URI paths, change only
`aml_output_uri` to a space-free path, for example:

```python
aml_output_uri = "azureml://datastores/workspaceblobstore/paths/safety-retrieval-agent/"
```

## Output paths

The Azure ML jobs write to the stable datastore URI configured by `aml_output_uri`.
With the default patch settings, the target URI is:

```text
azureml://datastores/workspaceblobstore/paths/Users/tan.cheng/EHS_predictive_modeling/safety_retrieval_agent/outputs/safety retrieval agent/
```

Inside that folder:

```text
embeddings/
faiss_indexes/
bm25_indexes/
data/
models/
logs/
```

## CPU usage

The submit scripts export:

```text
OMP_NUM_THREADS=8
MKL_NUM_THREADS=8
OPENBLAS_NUM_THREADS=8
NUMEXPR_NUM_THREADS=8
SAFETY_RETRIEVAL_EMBEDDING_DEVICE=cpu
```

At runtime, the scripts also call `torch.set_num_threads(os.cpu_count())` and
`faiss.omp_set_num_threads(os.cpu_count())`, then write CPU diagnostics into the
summary JSON files.
