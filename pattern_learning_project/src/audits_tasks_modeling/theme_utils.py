"""Shared utilities for source-aware safety text theme mining."""
from __future__ import annotations

import json
import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
except Exception:  # pragma: no cover
    ENGLISH_STOP_WORDS = frozenset()

try:
    import config as cfg
except Exception:  # pragma: no cover
    cfg = None


class ProgressLogger:
    def __init__(self, step_name: str) -> None:
        self.step_name = step_name
        self.start = time.time()
        self.log("START")

    def log(self, message: str) -> None:
        elapsed = time.time() - self.start
        print(f"[{self.step_name} | {elapsed:,.1f}s] {message}", flush=True)

    def done(self, message: str = "DONE") -> None:
        self.log(message)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def read_csv(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False, **kwargs)


def write_csv(df: pd.DataFrame, path: str | Path, **kwargs: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(path, index=False, **kwargs)
    print(f"Wrote {path} ({len(df):,} rows)", flush=True)


def compact_text(value: Any, max_chars: int | None = None) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    s = str(value)
    s = s.replace("\u2022", " ")
    s = s.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    # Common mojibake bullet/dash artifacts in the current exports.
    s = s.replace("â€¢", " ").replace("â€“", "-").replace("â€™", "'").replace("â€œ", '"').replace("â€\x9d", '"')
    s = re.sub(r"\s+", " ", s).strip()
    if max_chars and len(s) > max_chars:
        return s[: max_chars - 3].rstrip() + "..."
    return s


def normalize_for_terms(text: str) -> str:
    text = compact_text(text).lower()
    text = re.sub(r"[^a-z0-9\-/ ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_boilerplate(text: str) -> str:
    out = compact_text(text)
    patterns = getattr(cfg, "BOILERPLATE_PATTERNS", []) if cfg is not None else []
    for pat in patterns:
        out = re.sub(re.escape(str(pat)), " ", out, flags=re.IGNORECASE)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def source_family_from_row(row: pd.Series) -> str:
    source_type = compact_text(row.get("source_type", "")).lower()
    source_subtype = compact_text(row.get("source_subtype", "")).lower()
    category = compact_text(row.get("category", "")).lower()
    audit_type = compact_text(row.get("audit_type", "")).lower()
    module = compact_text(row.get("task_source_module", "")).lower()
    combined = " ".join([source_type, source_subtype, category, audit_type, module])

    if source_type == "task" or "task" in combined or "action" in combined:
        return getattr(cfg, "FAMILY_TASK_ACTION", "task_action")
    if source_type == "audit" or "audit" in combined or "observation" in combined or "inspection" in combined:
        return getattr(cfg, "FAMILY_AUDIT_OBSERVATION", "audit_observation")
    return getattr(cfg, "FAMILY_INCIDENT_HAZARD", "incident_hazard")


def truthy(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    if isinstance(value, float):
        if np.isnan(value):
            return False
        return bool(value)
    s = str(value).strip().lower()
    return s in {"true", "1", "yes", "y", "t"}


def event_kind_from_row(row: pd.Series) -> str:
    """Normalized event kind used for aggregate counts."""
    source_type = compact_text(row.get("source_type", "")).lower()
    category = compact_text(row.get("category", "")).lower()
    subtype = compact_text(row.get("source_subtype", "")).lower()
    status = compact_text(row.get("status", "")).lower()
    combined = " ".join([source_type, category, subtype, status])

    any_injury = truthy(row.get("any_injury", False))
    serious = truthy(row.get("severe_actual", False)) or truthy(row.get("fatality", False)) or truthy(row.get("losttime", False)) or truthy(row.get("restrictedtime", False)) or truthy(row.get("inpatient", False))
    if serious:
        return "serious_injury"
    if any_injury:
        return "normal_injury"
    if "near miss" in combined or source_type == "near_miss":
        return "near_miss"
    if "hazard" in combined:
        return "hazard_identification"
    if source_type == "audit":
        if "unsafe condition" in combined:
            return "audit_unsafe_condition"
        if "unsafe act" in combined:
            return "audit_unsafe_act"
        if "observation" in combined:
            return "audit_observation"
        return "audit_other"
    if source_type == "task":
        if truthy(row.get("is_overdue_task", False)):
            return "task_overdue"
        if truthy(row.get("is_open_task", False)):
            return "task_open"
        return "task_other"
    return source_type or "other"


def build_theme_text(row: pd.Series, max_chars: int | None = None) -> str:
    pieces = []
    for col in ["title", "description", "clean_text", "category", "source_subtype", "audit_type", "task_source_module", "status"]:
        if col in row.index:
            val = compact_text(row.get(col, ""))
            if val:
                pieces.append(val)
    text = " | ".join(pieces)
    text = remove_boilerplate(text)
    return compact_text(text, max_chars=max_chars)


def top_terms_from_texts(texts: Iterable[str], n: int = 20, extra_stopwords: set[str] | None = None) -> str:
    stop = set(ENGLISH_STOP_WORDS)
    if cfg is not None:
        stop |= set(getattr(cfg, "CUSTOM_STOPWORDS", set()))
    if extra_stopwords:
        stop |= set(extra_stopwords)

    counter: Counter[str] = Counter()
    phrase_counter: Counter[str] = Counter()
    for raw in texts:
        text = normalize_for_terms(str(raw))
        words = [w for w in re.findall(r"[a-z][a-z0-9\-/]{2,}", text) if w not in stop and not w.isdigit()]
        counter.update(words)
        for i in range(len(words) - 1):
            phrase = f"{words[i]} {words[i+1]}"
            if words[i] != words[i + 1]:
                phrase_counter[phrase] += 1
    # Prefer strong phrases but include words too.
    phrases = [p for p, c in phrase_counter.most_common(n) if c >= 2]
    words = [w for w, c in counter.most_common(n * 2)]
    terms: list[str] = []
    for t in phrases + words:
        if t not in terms:
            terms.append(t)
        if len(terms) >= n:
            break
    return " | ".join(terms)


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    a_norm[a_norm == 0] = 1.0
    b_norm[b_norm == 0] = 1.0
    return (a / a_norm) @ (b / b_norm).T


def assign_period(date_series: pd.Series, freq: str) -> pd.Series:
    dates = pd.to_datetime(date_series, errors="coerce")
    return dates.dt.to_period(freq).astype(str).replace("NaT", "unknown")


def safe_value_counts(series: pd.Series, top_n: int = 10) -> str:
    if series is None or len(series) == 0:
        return ""
    counts = series.fillna("unknown").astype(str).value_counts().head(top_n)
    return "; ".join([f"{idx}={int(val)}" for idx, val in counts.items()])


def build_review_priority(row: pd.Series) -> float:
    kind = compact_text(row.get("event_kind", "")).lower()
    if kind == "serious_injury":
        return 10.0
    if kind == "normal_injury":
        return 4.0
    if kind == "near_miss":
        return 3.0
    if kind == "hazard_identification":
        return 1.0
    if kind in {"audit_unsafe_condition", "audit_unsafe_act"}:
        return 2.0
    if kind == "task_overdue":
        return 2.0
    if kind == "task_open":
        return 0.5
    return 0.1


def representative_text(row: pd.Series, max_chars: int = 700) -> str:
    date = compact_text(row.get("event_date", ""))[:10]
    eid = compact_text(row.get("event_id", ""))
    kind = compact_text(row.get("event_kind", ""))
    status = compact_text(row.get("status", ""))
    title = compact_text(row.get("title", ""), 180)
    text = compact_text(row.get("clean_text", row.get("theme_text", "")), max_chars)
    prefix = " | ".join([x for x in [date, eid, kind, f"status={status}" if status else "", title] if x])
    return f"{prefix} - {text}" if text else prefix
