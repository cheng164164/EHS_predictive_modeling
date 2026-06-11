#!/usr/bin/env python
"""Download runtime artifacts from Azure ML datastore to the VS Code workspace.

Run this manually from the Azure ML Notebooks / VS Code terminal after the
Azure ML embedding/index build has completed:

    python scripts/01c_download_runtime_artifacts.py

It downloads only the artifacts needed by the agent at query time:
- data/
- faiss indexes -> outputs/safety retrieval agent/indexes/faiss_indexes/
- bm25 indexes  -> outputs/safety retrieval agent/indexes/bm25_indexes/
- models/

It intentionally does not download embedding chunks or event_embeddings.npy.
"""
from __future__ import annotations

from _bootstrap import PROJECT_ROOT  # noqa: F401
from safety_retrieval_agent.config import get_settings
from safety_retrieval_agent.runtime_artifact_cache import download_runtime_artifacts


if __name__ == "__main__":
    download_runtime_artifacts(get_settings())
