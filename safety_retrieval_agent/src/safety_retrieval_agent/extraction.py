"""Local, evidence-based extraction helpers for MVP1.

No LLM is required here. Risk factors and missing-information prompts are derived
from the query text, retrieved evidence, and structured form fields.
"""
from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .utils import clean_text_value, preview, word_count


def extract_keyphrases(query_text: str, evidence_texts: Iterable[str] | None = None, top_n: int = 12) -> list[dict]:
    """Extract data-driven risk-factor phrases from the query plus evidence.

    The query is intentionally repeated so terms in the user's new report are
    weighted more strongly than terms from retrieved historical records.
    """
    query_text = clean_text_value(query_text)
    evidence = [clean_text_value(t) for t in (evidence_texts or []) if clean_text_value(t)]
    docs = [query_text, query_text, query_text] + evidence[:50]
    docs = [d for d in docs if word_count(d) >= 2]
    if not docs:
        return []
    try:
        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 3),
            min_df=1,
            max_df=0.95,
            max_features=3000,
        )
        matrix = vectorizer.fit_transform(docs)
        # Weight the first three rows because they are the new report.
        row_weights = np.ones(matrix.shape[0], dtype="float32")
        row_weights[: min(3, len(row_weights))] = 2.5
        weighted = matrix.multiply(row_weights[:, None])
        scores = np.asarray(weighted.sum(axis=0)).ravel()
        terms = np.asarray(vectorizer.get_feature_names_out())
        order = scores.argsort()[::-1]
        out: list[dict] = []
        for idx in order:
            phrase = clean_text_value(terms[idx])
            if not _useful_phrase(phrase):
                continue
            out.append({"risk_factor": phrase, "score": float(scores[idx])})
            if len(out) >= top_n:
                break
        return out
    except Exception:
        return []


def _useful_phrase(phrase: str) -> bool:
    if not phrase or len(phrase) < 3:
        return False
    if phrase.isnumeric():
        return False
    blocked = {
        "title", "description", "comments", "record", "sims", "employee", "employees",
        "incident", "event", "observation", "observed", "safe", "unsafe", "work", "working",
    }
    parts = phrase.lower().split()
    if all(p in blocked for p in parts):
        return False
    if phrase.lower() in blocked:
        return False
    return True


def recommend_from_actions(action_matches: pd.DataFrame, safe_matches: pd.DataFrame | None = None, max_actions: int = 8) -> list[dict]:
    """Convert retrieved action/safe-practice records into evidence-backed recommendations."""
    recommendations: list[dict] = []
    if action_matches is not None and not action_matches.empty:
        for _, row in action_matches.head(max_actions).iterrows():
            text = _best_text_from_match(row)
            if not text:
                continue
            recommendations.append({
                "recommendation": f"Review similar historical action: {preview(text, 180)}",
                "evidence_type": "historical_corrective_action",
                "supporting_event_id": row.get("matched_event_id", row.get("event_id", "")),
                "similarity_score": float(row.get("similarity_score", 0.0) or 0.0),
            })
    if safe_matches is not None and not safe_matches.empty:
        for _, row in safe_matches.head(max(1, max_actions // 3)).iterrows():
            text = _best_text_from_match(row)
            if not text:
                continue
            recommendations.append({
                "recommendation": f"Use related safe-practice example: {preview(text, 180)}",
                "evidence_type": "historical_safe_observation",
                "supporting_event_id": row.get("matched_event_id", row.get("event_id", "")),
                "similarity_score": float(row.get("similarity_score", 0.0) or 0.0),
            })
    # De-duplicate by recommendation text.
    seen = set()
    deduped = []
    for item in recommendations:
        key = item["recommendation"].lower()
        if key not in seen:
            deduped.append(item)
            seen.add(key)
    return deduped[:max_actions]


def _best_text_from_match(row: pd.Series) -> str:
    for col in ["matched_title", "matched_description", "matched_retrieval_text", "title", "description", "retrieval_text"]:
        if col in row and clean_text_value(row.get(col)):
            return clean_text_value(row.get(col))
    return ""


def missing_information_prompt(
    query_text: str,
    site: str | None = None,
    department: str | None = None,
    source_type: str | None = None,
    detected_theme: str | None = None,
    severe_similarity_band: str | None = None,
) -> list[dict]:
    """Suggest missing details to improve the incident/hazard report.

    This is deliberately generic and form-quality oriented. It does not depend on
    a hard-coded hazard taxonomy.
    """
    text = clean_text_value(query_text)
    lower = text.lower()
    questions: list[dict] = []
    wc = word_count(text)

    def add(field: str, question: str, why: str) -> None:
        questions.append({"missing_area": field, "prompt": question, "why_it_matters": why})

    if wc < 25:
        add(
            "event narrative",
            "Add a fuller description of what happened, what activity was being performed, and what changed from normal conditions.",
            "Short descriptions make pattern recognition and severe-injury similarity less reliable.",
        )
    if not clean_text_value(site) or not clean_text_value(department):
        add(
            "location context",
            "Confirm the exact site, department, and specific area where the event occurred.",
            "Location context allows the agent to check whether this is a recurring pattern in the same area.",
        )
    if not _mentions_equipment_or_material(lower):
        add(
            "equipment / material involved",
            "Identify any equipment, vehicle, tool, material, energy source, or process involved.",
            "Similar historical cases are often driven by the equipment or energy source involved.",
        )
    if not _mentions_controls(lower):
        add(
            "controls in place",
            "Describe what controls were present or missing, such as barriers, guards, PPE, procedures, spotters, signage, or isolation.",
            "Control details help convert the report into prevention actions instead of only describing the event.",
        )
    if not _mentions_exposure_or_consequence(lower):
        add(
            "exposure / potential consequence",
            "Clarify who was exposed and what the credible worst outcome could have been.",
            "Potential consequence helps prioritize review even when no injury occurred.",
        )
    if not _mentions_immediate_action(lower):
        add(
            "immediate action",
            "Add what was done immediately to make the condition safe or prevent recurrence.",
            "Immediate actions help the system retrieve comparable corrective actions and avoid duplicate tasks.",
        )
    if severe_similarity_band in {"high", "medium"}:
        add(
            "high-severity review detail",
            "Because this resembles historical higher-severity records, confirm whether any safeguards failed, were bypassed, or were not available.",
            "This improves EHS review quality for events with elevated historical similarity.",
        )
    if detected_theme:
        add(
            "theme confirmation",
            f"Review whether the detected pattern '{detected_theme}' accurately describes the event, and correct it if needed.",
            "User feedback on the detected pattern improves future retrieval and recommendations.",
        )
    return questions


def _mentions_equipment_or_material(text: str) -> bool:
    # Broad lexical cues, not a hazard-family taxonomy.
    cues = ["equipment", "machine", "vehicle", "truck", "forklift", "tool", "material", "chemical", "hose", "crane", "ladder", "conveyor", "panel", "power", "energy"]
    return any(re.search(rf"\b{re.escape(cue)}\b", text) for cue in cues)


def _mentions_controls(text: str) -> bool:
    cues = ["ppe", "guard", "barrier", "procedure", "permit", "lock", "tag", "spotter", "sign", "training", "inspection", "control", "isolation", "walkway", "harness"]
    return any(re.search(rf"\b{re.escape(cue)}\b", text) for cue in cues)


def _mentions_exposure_or_consequence(text: str) -> bool:
    cues = ["injury", "hurt", "struck", "caught", "fall", "burn", "cut", "exposed", "near miss", "almost", "potential", "could have", "line of fire"]
    return any(cue in text for cue in cues)


def _mentions_immediate_action(text: str) -> bool:
    cues = ["corrected", "removed", "stopped", "repaired", "notified", "reported", "isolated", "cleaned", "blocked", "barricaded", "trained", "reviewed", "action"]
    return any(re.search(rf"\b{re.escape(cue)}\b", text) for cue in cues)
