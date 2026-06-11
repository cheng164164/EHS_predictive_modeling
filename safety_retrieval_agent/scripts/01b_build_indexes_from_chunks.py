#!/usr/bin/env python
"""Build FAISS/BM25/theme artifacts from completed embedding chunks.

Run locally or as an Azure ML command job with no arguments:

    python scripts/01b_build_indexes_from_chunks.py

All configuration is in src/safety_retrieval_agent/config.py.
"""
from __future__ import annotations

from _bootstrap import PROJECT_ROOT  # noqa: F401
from safety_retrieval_agent.config import get_settings
from safety_retrieval_agent.index_pipeline import build_indexes_from_chunks
from safety_retrieval_agent.runtime import configure_cpu_runtime
from safety_retrieval_agent.utils import ensure_dir, save_json


def main() -> dict:
    settings = get_settings()
    ensure_dir(settings.output_dir)
    diagnostics = configure_cpu_runtime(settings, include_faiss=True)
    summary = build_indexes_from_chunks(settings)
    summary["runtime_diagnostics"] = diagnostics
    save_json(summary, settings.output_dir / "01b_build_indexes_from_chunks_summary.json")
    return summary


if __name__ == "__main__":
    main()
