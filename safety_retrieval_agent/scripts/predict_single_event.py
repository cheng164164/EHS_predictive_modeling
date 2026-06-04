#!/usr/bin/env python
"""Analyze one configured incident/hazard description.

Run without args:

    python scripts/predict_single_event.py

Edit single_event_* values in src/safety_retrieval_agent/config.py to change the
query text, site, department, source type, or event ID.
"""
from __future__ import annotations

import json

from _bootstrap import PROJECT_ROOT  # noqa: F401
from safety_retrieval_agent.agent import SafetyRetrievalAgent
from safety_retrieval_agent.config import get_settings
from safety_retrieval_agent.utils import ensure_dir, save_json


def main() -> dict:
    settings = get_settings()
    ensure_dir(settings.recommendations_dir())
    agent = SafetyRetrievalAgent(settings)
    result = agent.analyze_event(
        query_text=settings.single_event_text,
        site=settings.single_event_site,
        department=settings.single_event_department,
        source_type=settings.single_event_source_type,
        event_id=settings.single_event_id,
    )
    out_path = settings.recommendations_dir() / "single_event_analysis.json"
    save_json(result, out_path)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    print(f"Saved: {out_path}")
    return result


if __name__ == "__main__":
    main()
