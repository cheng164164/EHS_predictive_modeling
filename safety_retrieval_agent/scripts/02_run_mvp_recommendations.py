#!/usr/bin/env python
"""Run MVP1 recommendations for configured sample/query records.

Run without args:

    python scripts/02_run_mvp_recommendations.py

Configuration lives in src/safety_retrieval_agent/config.py.
"""
from __future__ import annotations

import json
import time

import pandas as pd

from _bootstrap import PROJECT_ROOT  # noqa: F401
from safety_retrieval_agent.agent import SafetyRetrievalAgent, build_analysis_test_outputs, flatten_analysis_for_csv
from safety_retrieval_agent.artifact_io import artifact_exists, read_pickle
from safety_retrieval_agent.config import get_settings
from safety_retrieval_agent.utils import clean_text_value, ensure_dir, json_default, save_json


def _load_queries(settings) -> pd.DataFrame:
    if getattr(settings, "use_configured_test_queries", False):
        df = pd.DataFrame(list(getattr(settings, "configured_test_queries", [])))
        if df.empty:
            raise ValueError("use_configured_test_queries=True, but configured_test_queries is empty in config.py.")
        if "event_id" not in df.columns and "query_id" in df.columns:
            df["event_id"] = df["query_id"]
        return df

    query_file = settings.recommendation_query_file
    if query_file is not None:
        df = pd.read_csv(query_file, low_memory=False)
        if "query_text" not in df.columns:
            for col in ["retrieval_text", "clean_text", "description", "title"]:
                if col in df.columns:
                    df["query_text"] = df[col]
                    break
        if "query_text" not in df.columns:
            raise ValueError("Query file must contain query_text, retrieval_text, clean_text, description, or title.")
        return df

    kb_path = settings.artifact_enriched_knowledge_base_path() if hasattr(settings, "artifact_enriched_knowledge_base_path") else settings.enriched_knowledge_base_path()
    if not artifact_exists(kb_path):
        kb_path = settings.artifact_knowledge_base_path() if hasattr(settings, "artifact_knowledge_base_path") else settings.knowledge_base_path()
    if not artifact_exists(kb_path):
        raise FileNotFoundError(f"Knowledge base not found at artifact path: {kb_path}. Check config.py artifact_azureml_uri.")
    print(f"[02] Loading query source records from: {kb_path}", flush=True)
    df = read_pickle(kb_path)
    # Default batch examples come from leading/prevention records only. If the
    # enriched knowledge base exists, it has already been filtered to the embedding
    # scope by scripts/01_build_faiss_indexes.py.
    allowed = {"hazard_identification", "near_miss", "unsafe_observation", "safe_observation", "audit_observation"}
    if settings.recommendation_source_role:
        df = df[df["source_role"].astype(str).eq(settings.recommendation_source_role)].copy()
    else:
        df = df[df["source_role"].astype(str).isin(allowed)].copy()
    if df.empty:
        raise ValueError("No records available for recommendation examples after filtering.")
    n = min(settings.recommendation_sample_size, len(df))
    if settings.recommendation_recent and "event_date_dt" in df.columns:
        df = df.sort_values("event_date_dt", ascending=False).head(n).copy()
    else:
        df = df.sample(n=n, random_state=settings.random_seed).copy()
    df["query_text"] = df["retrieval_text"]
    return df.reset_index(drop=True)


def main() -> dict:
    settings = get_settings()
    start = time.time()
    ensure_dir(settings.recommendations_dir())

    queries = _load_queries(settings)
    print(f"[02] Loaded {len(queries):,} query records", flush=True)
    agent = SafetyRetrievalAgent(settings)

    jsonl_path = settings.recommendations_dir() / "mvp1_recommendation_results.jsonl"

    test_dir = ensure_dir(settings.tests_dir() / "mvp_recommendations")
    raw_debug_path = test_dir / "raw_retrieval_debug.jsonl"
    evidence_summary_path = test_dir / "structured_evidence_summary.jsonl"
    response_plan_path = test_dir / "structured_response_plan.jsonl"
    user_response_jsonl_path = test_dir / "user_facing_final_response.jsonl"
    user_response_csv_path = test_dir / "user_facing_final_response.csv"

    summary_rows = []
    user_response_rows = []
    with (
        jsonl_path.open("w", encoding="utf-8") as f,
        raw_debug_path.open("w", encoding="utf-8") as raw_f,
        evidence_summary_path.open("w", encoding="utf-8") as evidence_f,
        response_plan_path.open("w", encoding="utf-8") as plan_f,
        user_response_jsonl_path.open("w", encoding="utf-8") as user_f,
    ):
        for i, row in queries.iterrows():
            query_text = clean_text_value(row.get("query_text"))
            if not query_text:
                continue
            result = agent.analyze_event(
                query_text=query_text,
                site=row.get("site"),
                department=row.get("department"),
                source_type=row.get("source_type"),
                event_id=row.get("event_id"),
            )
            f.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")
            test_outputs = build_analysis_test_outputs(result)
            raw_f.write(json.dumps(test_outputs["raw_retrieval_debug"], ensure_ascii=False, default=json_default) + "\n")
            evidence_f.write(json.dumps(test_outputs["structured_evidence_summary"], ensure_ascii=False, default=json_default) + "\n")
            plan_f.write(json.dumps(test_outputs["structured_response_plan"], ensure_ascii=False, default=json_default) + "\n")
            user_f.write(json.dumps(test_outputs["user_facing_final_response"], ensure_ascii=False, default=json_default) + "\n")
            summary_rows.append(flatten_analysis_for_csv(result))
            user_response_rows.append({
                "event_id": result.get("query", {}).get("event_id"),
                "source_type": result.get("query", {}).get("source_type"),
                "site": result.get("query", {}).get("site"),
                "department": result.get("query", {}).get("department"),
                "risk_theme_id": result.get("user_facing_final_response", {}).get("risk_theme_id"),
                "risk_theme_name": result.get("user_facing_final_response", {}).get("risk_theme_name"),
                "response_status": result.get("user_facing_final_response", {}).get("response_status"),
                "llm_model_name": result.get("user_facing_final_response", {}).get("llm_model_name"),
                "response_text": result.get("user_facing_final_response", {}).get("response_text"),
            })
            if (i + 1) % 10 == 0:
                print(f"[02] Analyzed {i + 1:,}/{len(queries):,}", flush=True)

    summary_df = pd.DataFrame(summary_rows)
    csv_path = settings.recommendations_dir() / "mvp1_recommendation_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    pd.DataFrame(user_response_rows).to_csv(user_response_csv_path, index=False)
    summary = {
        "query_count": int(len(queries)),
        "analyzed_count": int(len(summary_rows)),
        "jsonl_output": str(jsonl_path),
        "csv_output": str(csv_path),
        "tests_output_dir": str(test_dir),
        "raw_retrieval_debug_jsonl": str(raw_debug_path),
        "structured_evidence_summary_jsonl": str(evidence_summary_path),
        "structured_response_plan_jsonl": str(response_plan_path),
        "user_facing_final_response_jsonl": str(user_response_jsonl_path),
        "user_facing_final_response_csv": str(user_response_csv_path),
        "elapsed_seconds": round(time.time() - start, 2),
    }
    save_json(summary, settings.recommendations_dir() / "mvp1_recommendation_run_summary.json")
    print(f"[02] Complete: {summary}", flush=True)
    return summary


if __name__ == "__main__":
    main()
