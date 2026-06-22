"""Utility helpers for Step 00 unified safety text event build.

This module intentionally supports the raw Velocity/Accelerate CSV column names
used by ``00_build_unified_text_events.py``.  It keeps the Step 00 output schema
stable while avoiding dependencies on other helper modules.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

TRUE_VALUES = {"true", "1", "yes", "y", "t", "x"}
FALSE_VALUES = {"false", "0", "no", "n", "f", ""}

OUTPUT_COLUMNS = [
    "event_id",
    "source_type",
    "source_subtype",
    "source_id",
    "event_date",
    "location_id",
    "site",
    "department",
    "location_path",
    "title",
    "description",
    "clean_text",
    "status",
    "category",
    "is_open_task",
    "is_overdue_task",
    "due_date",
    "completion_date",
    "any_injury",
    "severe_actual",
    "fatality",
    "losttime",
    "restrictedtime",
    "inpatient",
    "emergencyroom",
    "injury_record_count",
    "raw_status_id",
    "raw_category_id",
    "audit_type",
    "raw_type_id",
    "task_source_module",
    "raw_source_type_id",
    "text_length",
    "has_text",
    "detected_language",
    "detected_language_score",
    "language_detection_status",
    "is_english_text",
]

BOOL_OUTPUT_COLUMNS = [
    "is_open_task",
    "is_overdue_task",
    "any_injury",
    "severe_actual",
    "fatality",
    "losttime",
    "restrictedtime",
    "inpatient",
    "emergencyroom",
    "has_text",
]


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    """Read a CSV file with UTF-8 first and latin1 fallback."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Required CSV file does not exist: {path}")
    kwargs.setdefault("low_memory", False)
    try:
        return pd.read_csv(path, **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1", **kwargs)


def save_csv(df: pd.DataFrame, path: str | Path) -> None:
    """Save CSV, automatically using gzip when the suffix is .gz."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    compression = "gzip" if path.suffix.lower() == ".gz" else None
    df.to_csv(path, index=False, compression=compression)


def save_json(payload: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if pd.isna(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat() if pd.notna(value) else None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return str(value)


def clean_text_value(value: object) -> str:
    """Clean a single text value without changing the business meaning."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return "" if text.lower() in {"nan", "none", "null", "nat"} else text


def truthy_series(series: pd.Series | object) -> pd.Series:
    """Parse common true/false values to bool."""
    if not isinstance(series, pd.Series):
        return pd.Series(series).fillna(False).astype(bool)
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    s = series.fillna("").astype(str).str.strip().str.lower()
    return s.isin(TRUE_VALUES)


def parse_datetime_series(series: pd.Series | object) -> pd.Series:
    """Parse datetimes and return timezone-naive timestamps for CSV compatibility."""
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    out = pd.to_datetime(series, errors="coerce", utc=True)
    try:
        out = out.dt.tz_convert(None)
    except Exception:
        pass
    return out


def coalesce_datetime(df: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    """Return first non-null parsed datetime from the listed columns."""
    out = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    for col in columns:
        if col not in df.columns:
            continue
        parsed = parse_datetime_series(df[col])
        out = out.where(out.notna(), parsed)
    return out


def coalesce_string(df: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    """Return first nonblank cleaned string from the listed columns."""
    out = pd.Series("", index=df.index, dtype="object")
    for col in columns:
        if col not in df.columns:
            continue
        values = df[col].map(clean_text_value)
        out = out.where(out.astype(str).str.len().gt(0), values)
    return out.fillna("").astype(str)


def make_text_block(df: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    """Build labeled text like 'title: ... | description: ...'."""
    parts = []
    for col in columns:
        if col not in df.columns:
            continue
        label = col.lower()
        values = df[col].map(clean_text_value)
        labeled = values.map(lambda v, label=label: f"{label}: {v}" if v else "")
        parts.append(labeled)
    if not parts:
        return pd.Series("", index=df.index, dtype="object")
    out = parts[0]
    for p in parts[1:]:
        out = out.str.cat(p, sep=" | ")
    out = out.map(lambda x: " | ".join([p for p in str(x).split(" | ") if clean_text_value(p)]))
    return out.map(clean_text_value)


def normalize_source_type(source_subtype: object) -> str:
    """Map incident subtypes to the high-level source_type field."""
    text = clean_text_value(source_subtype).lower()
    if "near miss" in text:
        return "near_miss"
    if "hazard" in text:
        return "hazard_identification"
    return "incident"


def _first_existing_column(df: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    upper_map = {c.upper(): c for c in df.columns}
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
        if candidate.upper() in upper_map:
            return upper_map[candidate.upper()]
    return None


def load_listitem_lookup(path: str | Path) -> tuple[dict, pd.DataFrame]:
    """Load LISTITEM_VIEW into a dict keyed by LISTITEMID.

    The lookup value includes the readable item text and optional list type code.
    """
    df = read_csv(path)
    id_col = _first_existing_column(df, ["LISTITEMID", "ListItemID", "list_item_id"])
    if id_col is None:
        raise ValueError("LISTITEM_VIEW.csv must contain LISTITEMID")
    item_col = _first_existing_column(
        df,
        [
            "LISTITEM",
            "LISTITEMNAME",
            "NAME",
            "TITLE",
            "SHORTNAME",
            "DESCRIPTION",
            "VALUE",
            "ITEM",
        ],
    )
    if item_col is None:
        # Fallback: choose the first non-id object column.
        candidates = [c for c in df.columns if c != id_col and pd.api.types.is_object_dtype(df[c])]
        item_col = candidates[0] if candidates else id_col
    type_col = _first_existing_column(df, ["LISTTYPECODE", "LISTTYPE", "TYPE", "CATEGORY"])

    lookup = {}
    for _, row in df.iterrows():
        key = row.get(id_col)
        if pd.isna(key):
            continue
        try:
            key = int(key)
        except Exception:
            key = str(key)
        lookup[key] = {
            "item": clean_text_value(row.get(item_col)),
            "list_type": clean_text_value(row.get(type_col)) if type_col else "",
        }
    return lookup, df


def add_listitem_fields(df: pd.DataFrame, id_col: str, prefix: str, listitem_lookup: dict) -> pd.DataFrame:
    """Attach '<prefix>_item' and '<prefix>_list_type' from LISTITEM_VIEW."""
    out = df.copy()
    if id_col not in out.columns:
        out[f"{prefix}_item"] = ""
        out[f"{prefix}_list_type"] = ""
        return out

    def lookup_item(value: object) -> str:
        if pd.isna(value):
            return ""
        keys = []
        try:
            keys.append(int(value))
        except Exception:
            pass
        keys.append(str(value))
        for key in keys:
            if key in listitem_lookup:
                return listitem_lookup[key].get("item", "")
        return ""

    def lookup_type(value: object) -> str:
        if pd.isna(value):
            return ""
        keys = []
        try:
            keys.append(int(value))
        except Exception:
            pass
        keys.append(str(value))
        for key in keys:
            if key in listitem_lookup:
                return listitem_lookup[key].get("list_type", "")
        return ""

    out[f"{prefix}_item"] = out[id_col].map(lookup_item)
    out[f"{prefix}_list_type"] = out[id_col].map(lookup_type)
    return out


def _location_lookup_key(value: object) -> object:
    """Return a stable lookup key for numeric or text location IDs."""
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    text = clean_text_value(value)
    if not text:
        return None
    try:
        number = float(text)
        if number.is_integer():
            return int(number)
    except Exception:
        pass
    return text


def _is_location_id_like_name(name: str, loc_id: object | None = None) -> bool:
    """True when a candidate location name is only an ID/code value."""
    text = clean_text_value(name)
    if not text:
        return True
    if re.fullmatch(r"\d+(?:\.0)?", text):
        return True
    loc_key = _location_lookup_key(loc_id)
    if loc_key is not None and text == str(loc_key):
        return True
    return False


def _clean_location_levels(levels: Iterable[str]) -> list[str]:
    """Remove blank/ID-only levels and collapse consecutive duplicates."""
    cleaned: list[str] = []
    previous = ""
    for value in levels:
        text = clean_text_value(value)
        if _is_location_id_like_name(text):
            continue
        if text.casefold() == previous:
            continue
        cleaned.append(text)
        previous = text.casefold()
    return cleaned


def _derive_department_and_site(levels: Sequence[str]) -> tuple[str, str]:
    """Department is level 5; site is the last two location levels."""
    department = levels[4] if len(levels) >= 5 else (levels[-1] if levels else "")
    site = " > ".join(levels[-2:]) if len(levels) >= 2 else (levels[0] if levels else "")
    return department, site


def build_location_hierarchy(location_path: str | Path, listitem_lookup: dict | None = None) -> pd.DataFrame:
    """Build decoded location hierarchy columns used by Step 00.

    Department is the fifth hierarchy level. Site is the last two hierarchy
    levels. The display columns contain decoded names only, not location IDs.
    """
    del listitem_lookup  # Kept for backward-compatible function signature.
    loc = read_csv(location_path)
    id_col = _first_existing_column(loc, ["LOCATIONID", "location_id"])
    parent_col = _first_existing_column(loc, ["PARENTLOCATIONID", "parent_location_id", "PARENTID", "parent_id"])
    name_cols = [
        c for c in [
            _first_existing_column(loc, ["LOCATIONNAME", "location_name"]),
            _first_existing_column(loc, ["NAME", "name"]),
            _first_existing_column(loc, ["SHORTNAME", "short_name"]),
            _first_existing_column(loc, ["LOCATION", "location"]),
            _first_existing_column(loc, ["TITLE", "title"]),
            _first_existing_column(loc, ["DESCRIPTION", "description"]),
        ] if c is not None
    ]
    if id_col is None:
        raise ValueError("LOCATION_VIEW.csv must contain LOCATIONID")

    work = loc.copy()
    work["_loc_id"] = work[id_col]
    work["_loc_key"] = work[id_col].map(_location_lookup_key)
    work["_parent_key"] = work[parent_col].map(_location_lookup_key) if parent_col else None

    def decoded_name(row: pd.Series) -> str:
        for col in name_cols:
            name = clean_text_value(row.get(col))
            if name and not _is_location_id_like_name(name, row.get("_loc_id")):
                return name
        return ""

    work["_name"] = work.apply(decoded_name, axis=1)
    by_key = {row["_loc_key"]: row for _, row in work.iterrows() if row["_loc_key"] is not None}

    def parent_of(key):
        row = by_key.get(key)
        if row is None:
            return None
        return row.get("_parent_key")

    def name_of(key) -> str:
        row = by_key.get(key)
        if row is None:
            return ""
        return clean_text_value(row.get("_name"))

    def path_for(value) -> list[str]:
        levels: list[str] = []
        seen = set()
        cur = _location_lookup_key(value)
        while cur is not None and cur not in seen and cur in by_key:
            seen.add(cur)
            name = name_of(cur)
            if name:
                levels.append(name)
            cur = parent_of(cur)
        return _clean_location_levels(reversed(levels))

    rows = []
    for _, row in work.iterrows():
        loc_id = row["_loc_id"]
        levels = path_for(loc_id)
        if not levels:
            fallback_name = clean_text_value(row.get("_name"))
            levels = [fallback_name] if fallback_name and not _is_location_id_like_name(fallback_name, loc_id) else []
        department, site = _derive_department_and_site(levels)
        path_text = " > ".join(levels)
        record = {
            "LOCATIONID": loc_id,
            "location_name": clean_text_value(row.get("_name")),
            "location_path": path_text,
            "location_path_clean": path_text,
            "site": site,
            "department": department,
        }
        for i in range(1, 7):
            record[f"location_level_{i}"] = levels[i - 1] if len(levels) >= i else ""
        rows.append(record)
    return pd.DataFrame(rows)


def ensure_unified_event_schema(events: pd.DataFrame) -> pd.DataFrame:
    """Add missing Step 00 columns and order exactly as the posted schema."""
    out = events.copy()
    defaults = {
        "audit_type": "",
        "raw_type_id": np.nan,
        "task_source_module": "",
        "raw_source_type_id": np.nan,
        "detected_language": "not_run",
        "detected_language_score": np.nan,
        "language_detection_status": "not_run",
        "is_english_text": np.nan,
    }
    for col in OUTPUT_COLUMNS:
        if col not in out.columns:
            out[col] = defaults.get(col, "")
    for col in BOOL_OUTPUT_COLUMNS:
        if col in out.columns:
            out[col] = out[col].fillna(False).astype(bool)
    for col in ["event_date", "due_date", "completion_date"]:
        if col in out.columns:
            out[col] = parse_datetime_series(out[col])
    if "injury_record_count" in out.columns:
        out["injury_record_count"] = pd.to_numeric(out["injury_record_count"], errors="coerce").fillna(0).astype(int)
    out["clean_text"] = out["clean_text"].fillna("").map(clean_text_value)
    out["text_length"] = out["clean_text"].str.len().astype(int)
    out["has_text"] = out["text_length"].gt(0)
    return out[OUTPUT_COLUMNS].copy()
