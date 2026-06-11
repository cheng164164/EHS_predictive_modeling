#!/usr/bin/env python
"""Submit the resumable embedding-chunk job to Azure ML.

Run this from VS Code / Azure ML notebooks with no arguments:

    python scripts/submit_azureml_embedding_job.py

The job keeps running in Azure ML after your browser/PC disconnects. All
workspace, compute, environment, input, and output settings are in config.py.

Optional local download:
    Set SAFETY_RETRIEVAL_DOWNLOAD_AFTER_SUBMIT=true to stream/wait for the job
    and then run the local runtime-artifact download step. This is mainly useful
    after a full build/index already exists; the embedding-only job by itself does
    not create FAISS/BM25 indexes.
"""
from __future__ import annotations

from _bootstrap import PROJECT_ROOT  # noqa: F401
from safety_retrieval_agent.azureml_submit import submit_embedding_chunks_job, stream_job_until_complete
from safety_retrieval_agent.config import get_settings, _bool_from_env
from safety_retrieval_agent.runtime_artifact_cache import download_runtime_artifacts


if __name__ == "__main__":
    settings = get_settings()
    job = submit_embedding_chunks_job(settings)
    if _bool_from_env("SAFETY_RETRIEVAL_DOWNLOAD_AFTER_SUBMIT", False):
        stream_job_until_complete(settings, job.name)
        download_runtime_artifacts(settings)
    else:
        print(
            "[AML] Embedding job submitted. After embeddings and indexes are built, "
            "run `python scripts/01c_download_runtime_artifacts.py` to cache indexes/models locally for faster testing.",
            flush=True,
        )
