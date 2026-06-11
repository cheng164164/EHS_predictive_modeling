#!/usr/bin/env python
"""Submit one Azure ML job for embedding generation and index creation.

Run from VS Code / Azure ML Notebooks terminal:

    python scripts/submit_azureml_full_batch_job.py

After the job is submitted, the long-running work happens on the Azure ML
compute cluster and does not depend on your browser or PC staying awake. Large
artifacts remain in the configured Azure ML datastore. For faster interactive
query testing, run scripts/01c_download_runtime_artifacts.py after the job
completes, or set SAFETY_RETRIEVAL_DOWNLOAD_AFTER_SUBMIT=true to wait and
download automatically.
"""
from __future__ import annotations

from _bootstrap import PROJECT_ROOT  # noqa: F401
from safety_retrieval_agent.azureml_submit import submit_full_batch_job, stream_job_until_complete
from safety_retrieval_agent.config import get_settings, _bool_from_env
from safety_retrieval_agent.runtime_artifact_cache import download_runtime_artifacts


if __name__ == "__main__":
    settings = get_settings()
    job = submit_full_batch_job(settings)
    if _bool_from_env("SAFETY_RETRIEVAL_DOWNLOAD_AFTER_SUBMIT", False):
        stream_job_until_complete(settings, job.name)
        download_runtime_artifacts(settings)
    else:
        print(
            "[AML] Full batch job submitted. After it completes, run "
            "`python scripts/01c_download_runtime_artifacts.py` to cache runtime artifacts locally for faster agent queries.",
            flush=True,
        )
