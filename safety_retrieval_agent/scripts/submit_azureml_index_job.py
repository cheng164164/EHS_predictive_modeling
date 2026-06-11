#!/usr/bin/env python
"""Submit the FAISS/BM25 index-build job to Azure ML.

Run this after the embedding-chunk job has completed:

    python scripts/submit_azureml_index_job.py

All workspace, compute, environment, input, and output settings are in config.py.
"""
from __future__ import annotations

from _bootstrap import PROJECT_ROOT  # noqa: F401
from safety_retrieval_agent.azureml_submit import submit_index_build_job
from safety_retrieval_agent.config import get_settings


if __name__ == "__main__":
    submit_index_build_job(get_settings())
