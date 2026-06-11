#!/usr/bin/env python
"""Submit an Azure ML job that copies runtime artifacts into a job output.

For interactive VS Code testing, prefer running this local command instead:

    python scripts/01c_download_runtime_artifacts.py

That command downloads artifacts into the current VS Code workspace. This submit
script runs remotely on Azure ML compute, so it cannot write directly into your
active VS Code filesystem unless that filesystem is explicitly mounted in the
job environment. It is included for users who want a cloud-side copy/check job.
"""
from __future__ import annotations

from _bootstrap import PROJECT_ROOT  # noqa: F401
from safety_retrieval_agent.azureml_submit import submit_download_runtime_artifacts_job
from safety_retrieval_agent.config import get_settings


if __name__ == "__main__":
    submit_download_runtime_artifacts_job(get_settings())
