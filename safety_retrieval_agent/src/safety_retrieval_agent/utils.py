"""Shared helpers for the Safety Retrieval Agent MVP."""
from __future__ import annotations

import gzip
import json
import math
import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

TRUE_VALUES = {"true", "1", "yes", "y", "t"}
FALSE_VALUES = {"false", "0", "no", "n", "f", ""}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if math.isnan(float(value)) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat() if pd.notna(value) else None
    if isinstance(value, Path):
        return str(value)
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return str(value)


def save_json(payload: dict, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default), encoding="utf-8")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path, nrows: int | None = None, usecols: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    try:
        return pd.read_csv(path, low_memory=False, nrows=nrows, usecols=usecols)
    except UnicodeDecodeError:
        return pd.read_csv(path, low_memory=False, nrows=nrows, usecols=usecols, encoding="latin1")


def clean_text_value(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return "" if text.lower() in {"nan", "none", "null", "nat"} else text


def clean_text_series(series: pd.Series) -> pd.Series:
    return series.fillna("").map(clean_text_value).astype("string")


def normalize_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or pd.isna(value):
        return False
    text = str(value).strip().lower()
    if text in TRUE_VALUES:
        return True
    if text in FALSE_VALUES:
        return False
    return False


def bool_series(series: pd.Series) -> pd.Series:
    return series.map(normalize_bool).astype(bool)


def parse_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=False, format="mixed")


def preview(text: object, n: int = 320) -> str:
    text = clean_text_value(text)
    return text if len(text) <= n else text[: n - 3].rstrip() + "..."


def word_count(text: object) -> int:
    text = clean_text_value(text)
    if not text:
        return 0
    return len(re.findall(r"\b\w+\b", text))


def choose_text(row: pd.Series) -> str:
    """Create the text used for embedding/retrieval.

    The unified table already has clean_text. We prefer it because it contains a
    labelled combination of source fields. If it is too short, we fall back to
    title + description.
    """
    clean_text = clean_text_value(row.get("clean_text", ""))
    if word_count(clean_text) >= 3:
        return clean_text
    pieces = [row.get("title", ""), row.get("description", ""), row.get("source_subtype", "")]
    return clean_text_value(" | ".join(clean_text_value(p) for p in pieces if clean_text_value(p)))


def safe_join(values: Iterable[object], sep: str = " | ", max_items: int | None = None) -> str:
    items: list[str] = []
    for value in values:
        text = clean_text_value(value)
        if text:
            items.append(text)
        if max_items is not None and len(items) >= max_items:
            break
    return sep.join(items)


def list_top_counts(series: pd.Series, n: int = 10) -> list[dict]:
    if series is None or series.empty:
        return []
    counts = series.fillna("Unknown").astype(str).replace("", "Unknown").value_counts().head(n)
    return [{"value": str(k), "count": int(v)} for k, v in counts.items()]


def similarity_band(score: float | None, high: float, medium: float, low: float) -> str:
    if score is None or pd.isna(score):
        return "no_match"
    score = float(score)
    if score >= high:
        return "high"
    if score >= medium:
        return "medium"
    if score >= low:
        return "low"
    return "no_match"


def extract_year_month(date_series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(date_series, errors="coerce")
    return dt.dt.to_period("M").astype(str).replace("NaT", "")


def normalize_vectors(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    vectors = np.asarray(vectors, dtype="float32")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, eps)


def chunk_ranges(n: int, chunk_size: int) -> Iterable[tuple[int, int]]:
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        yield start, end
        start = end


def compress_json_field(value: object) -> str:
    """Store nested evidence in CSV-friendly JSON text."""
    return json.dumps(value, ensure_ascii=False, default=json_default)


def source_role(row: pd.Series) -> str:
    """Map each event to its retrieval role.

    This is based on source system fields, not on hazard keyword mappings.
    """
    source_type = clean_text_value(row.get("source_type", "")).lower()
    audit_type = clean_text_value(row.get("audit_type", "")).lower()
    source_subtype = clean_text_value(row.get("source_subtype", "")).lower()
    any_injury = normalize_bool(row.get("any_injury", False)) or int(row.get("injury_record_count", 0) or 0) > 0
    severe = normalize_bool(row.get("severe_actual", False))
    is_open_task = normalize_bool(row.get("is_open_task", False))
    is_overdue_task = normalize_bool(row.get("is_overdue_task", False))

    if any_injury and severe:
        return "severe_injury"
    if any_injury:
        return "injury"
    if source_type == "hazard_identification":
        return "hazard_identification"
    if source_type == "near_miss":
        return "near_miss"
    if source_type == "task":
        if is_overdue_task:
            return "overdue_corrective_action"
        if is_open_task:
            return "open_corrective_action"
        return "corrective_action"
    if source_type == "audit":
        label = audit_type or source_subtype
        if "unsafe" in label:
            return "unsafe_observation"
        if "safe" in label:
            return "safe_observation"
        if "inspection" in label:
            return "inspection"
        return "audit_observation"
    return source_type or "unknown"


def is_blank(value: object) -> bool:
    return clean_text_value(value) == ""
