# Datastore-direct runtime artifact access

This patch removes the need to download large FAISS/BM25/embedding artifacts into the VS Code workspace. The build job keeps artifacts in `workspaceblobstore`, and the runtime agent reads them directly through `azureml-fsspec`.

## Install the small extra runtime dependency

```bash
pip install -r requirements_datastore_runtime.txt
```

Your existing `requirements.txt` can remain as-is. This patch only adds Azure ML datastore file-system support.

## Default artifact path

The runtime artifact root is configured in `src/safety_retrieval_agent/config.py`:

```text
azureml://subscriptions/7f07baf7-8bba-4b88-b300-74ba5b15f52d/resourcegroups/EHS-Safety/workspaces/ehs-safety-aml/datastores/workspaceblobstore/paths/safety-retrieval-agent/managed-batch-artifacts/
```

This matches the current datastore folder shown in Azure ML Studio:

```text
workspaceblobstore/safety-retrieval-agent/managed-batch-artifacts/
```

Expected subfolders:

```text
data/
faiss_indexes/
bm25_indexes/
models/
embeddings/
```

## How to run direct tests

```bash
python scripts/predict_single_event.py
python scripts/02_run_mvp_recommendations.py
```

The agent prints the artifact root it is reading from. Small recommendation outputs are still written locally under:

```text
outputs/safety retrieval agent/recommendations/
```

The large retrieval artifacts stay in the datastore.

## To switch back to local artifacts

Set this environment variable before running the scripts:

```bash
export SAFETY_RETRIEVAL_ARTIFACT_READ_MODE=local
```

Then the agent reads local files from:

```text
outputs/safety retrieval agent/
```

## Build job behavior

`submit_azureml_full_batch_job.py` now runs only:

```text
01a_generate_embedding_chunks.py
01b_build_indexes_from_chunks.py
```

The old sync-to-workspace step is disabled. Artifacts remain in the stable AML datastore path.
