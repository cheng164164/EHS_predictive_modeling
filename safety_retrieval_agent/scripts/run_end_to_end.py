#!/usr/bin/env python
"""Run the full MVP1 local pipeline.

Execution order:
    1. Build unified event file from raw exports when needed
    2. Prepare retrieval knowledge base
    3. Build transformer embeddings, risk themes, and FAISS indexes
    4. Generate MVP1 recommendation examples

Run without args:

    python scripts/run_end_to_end.py

Configuration lives in src/safety_retrieval_agent/config.py.
"""
from __future__ import annotations

import subprocess
import sys

from _bootstrap import PROJECT_ROOT
from safety_retrieval_agent.config import get_settings


def _run(script: str) -> None:
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / script)]
    print("Running:", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    settings = get_settings()
    if settings.force_rebuild_unified_event_file or (settings.run_unified_builder_if_missing and not settings.input_event_file.exists()):
        _run("00_build_unified_text_events.py")
    else:
        print(f"Unified event file exists; skipping unified build: {settings.input_event_file}", flush=True)
    _run("00_prepare_knowledge_base.py")
    _run("01_build_faiss_indexes.py")
    _run("02_run_mvp_recommendations.py")


if __name__ == "__main__":
    main()
