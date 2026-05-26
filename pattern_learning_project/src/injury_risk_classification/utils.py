"""Shared utilities for injury-risk classification scripts."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


_BOOL_TRUE = {"true", "1", "yes", "y", "t"}
_BOOL_FALSE = {"false", "0", "no", "n", "f"}


def now_run_id() -> str:
    """Return a filesystem-safe UTC timestamp for run folders."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_utc")


def ensure_dir(path: str | Path) -> Path:
    """Create a folder if it does not exist and return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: dict, path: str | Path) -> None:
    """Write a dictionary as pretty JSON."""
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def read_json(path: str | Path) -> dict:
    """Read JSON from disk."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with lower snake_case columns.

    Velocity exports have uppercase names such as INCIDENTID and SCHEDULEDLOCATIONID.
    Standardizing early keeps downstream feature code consistent.
    """
    out = df.copy()
    out.columns = [_to_snake(c) for c in out.columns]
    return out


def _to_snake(name: object) -> str:
    text = str(name).strip()
    text = re.sub(r"[^0-9a-zA-Z]+", "_", text)
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text.lower()


def read_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    """Read a CSV with practical defaults for mixed-type operational exports."""
    return pd.read_csv(path, low_memory=False, **kwargs)


def parse_datetime(series: pd.Series) -> pd.Series:
    """Parse a pandas Series into timezone-naive UTC timestamps.

    The source files often use ISO strings ending in Z. We parse as UTC and then drop the
    timezone so all comparisons and month periods are straightforward.
    """
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    return parsed.dt.tz_convert(None)


def coerce_bool(series: pd.Series) -> pd.Series:
    """Coerce common boolean representations to pandas BooleanDtype."""
    if series.dtype == bool:
        return series.astype("boolean")
    s = series.astype("string").str.strip().str.lower()
    mapped = s.map(lambda x: True if x in _BOOL_TRUE else (False if x in _BOOL_FALSE else pd.NA))
    return mapped.astype("boolean")


def coerce_bool_columns(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    """Coerce a set of columns to pandas BooleanDtype when present."""
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = coerce_bool(out[col])
    return out


def coerce_numeric_id(series: pd.Series) -> pd.Series:
    """Coerce ID-like values to nullable integer IDs."""
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def month_start(series: pd.Series) -> pd.Series:
    """Convert dates to the first day of their calendar month."""
    return series.dt.to_period("M").dt.to_timestamp()


def clean_text_value(value: object) -> str:
    """Clean one text value for robust downstream feature generation."""
    if pd.isna(value):
        return ""
    text = str(value)
    text = re.sub(r"\s+", " ", text.replace("\r", " ").replace("\n", " ")).strip()
    return text


def combine_text_fields(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Combine available text columns into one modeling text field."""
    present = [c for c in cols if c in df.columns]
    if not present:
        return pd.Series([""] * len(df), index=df.index)
    clean = df[present].apply(lambda col: col.map(clean_text_value))
    return clean.apply(lambda row: " | ".join([x for x in row.tolist() if x]), axis=1)


def safe_divide(numer: pd.Series | np.ndarray | float, denom: pd.Series | np.ndarray | float, default: float = 0.0):
    """Vectorized safe division that returns default where denominator is zero/missing."""
    n = np.asarray(numer, dtype=float)
    d = np.asarray(denom, dtype=float)
    out = np.full_like(n, fill_value=default, dtype=float)
    mask = np.isfinite(n) & np.isfinite(d) & (d != 0)
    out[mask] = n[mask] / d[mask]
    return out


def add_cyclical_month_features(df: pd.DataFrame, month_col: str = "anchor_month") -> pd.DataFrame:
    """Add month number and cyclical month encoding."""
    out = df.copy()
    month_num = pd.to_datetime(out[month_col]).dt.month.fillna(0).astype(int)
    out["calendar_month"] = month_num
    out["calendar_month_sin"] = np.sin(2 * np.pi * month_num / 12.0)
    out["calendar_month_cos"] = np.cos(2 * np.pi * month_num / 12.0)
    return out


def top_k_capture(y_true: np.ndarray, y_score: np.ndarray, top_fraction: float) -> dict:
    """Compute how many positives are captured in the highest-risk top fraction.

    This is often more operationally useful than accuracy because EHS teams typically
    investigate a limited number of highest-risk site/departments.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    n = len(y_true)
    if n == 0:
        return {"top_fraction": top_fraction, "top_n": 0, "positives_in_top": 0, "total_positives": 0, "recall_at_top": np.nan, "precision_at_top": np.nan}
    top_n = max(1, int(np.ceil(n * top_fraction)))
    order = np.argsort(-y_score)
    top_idx = order[:top_n]
    total_pos = int(y_true.sum())
    pos_top = int(y_true[top_idx].sum())
    base_rate = float(total_pos / n) if n else np.nan
    precision = float(pos_top / top_n) if top_n else np.nan
    return {
        "top_fraction": top_fraction,
        "top_n": int(top_n),
        "positives_in_top": pos_top,
        "total_positives": total_pos,
        "recall_at_top": float(pos_top / total_pos) if total_pos else np.nan,
        "precision_at_top": precision,
        "base_positive_rate": base_rate,
        "lift_at_top": float(precision / base_rate) if base_rate and np.isfinite(base_rate) else np.nan,
    }
