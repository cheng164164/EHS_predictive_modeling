from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_csv(path: str | Path, nrows: Optional[int] = None, usecols: Optional[Sequence[str]] = None) -> pd.DataFrame:
    return pd.read_csv(path, nrows=nrows, usecols=usecols, low_memory=False)


def save_csv(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(path, index=False)
    return path


def load_table(path: str | Path, **kwargs) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    if path.suffix == ".parquet":
        return pd.read_parquet(path, **kwargs)
    return pd.read_csv(path, low_memory=False, **kwargs)


def parse_datetime_series(s: pd.Series) -> pd.Series:
    out = pd.to_datetime(s, errors="coerce", utc=True)
    try:
        out = out.dt.tz_convert(None)
    except Exception:
        try:
            out = out.dt.tz_localize(None)
        except Exception:
            pass
    return out


def coalesce_datetime(df: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    out = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    for col in columns:
        if col in df.columns:
            parsed = parse_datetime_series(df[col])
            out = out.fillna(parsed)
    return out


def coalesce_string(df: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    out = pd.Series("", index=df.index, dtype="object")
    for col in columns:
        if col in df.columns:
            vals = df[col].fillna("").astype(str).str.strip()
            vals = vals.replace({"nan": "", "None": "", "NaT": ""})
            out = out.mask(out.eq(""), vals)
    return out.fillna("")


def clean_text_value(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value)
    if text.lower() in {"nan", "none", "nat"}:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def make_text_block(df: pd.DataFrame, fields: Sequence[str], labels: Optional[Dict[str, str]] = None) -> pd.Series:
    labels = labels or {}
    out = pd.Series("", index=df.index, dtype="object")
    for col in fields:
        if col not in df.columns:
            continue
        vals = df[col].map(clean_text_value)
        vals = vals.where(vals.ne(""), "")
        label = labels.get(col, col.lower())
        piece = np.where(vals.ne(""), label + ": " + vals, "")
        piece = pd.Series(piece, index=df.index, dtype="object")
        out = np.where((out != "") & (piece != ""), out + " | " + piece, np.where(piece != "", piece, out))
        out = pd.Series(out, index=df.index, dtype="object")
    return out.map(clean_text_value)


def truthy_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    return s.fillna(False).astype(str).str.lower().isin({"true", "1", "yes", "y", "t"})


def load_listitem_lookup(listitem_path: str | Path) -> Tuple[Dict[int, dict], pd.DataFrame]:
    li = read_csv(listitem_path)
    li["LISTITEMID"] = pd.to_numeric(li["LISTITEMID"], errors="coerce")
    lookup = {}
    for _, row in li.dropna(subset=["LISTITEMID"]).iterrows():
        lookup[int(row["LISTITEMID"])] = {
            "list_type": row.get("LISTTYPECODE"),
            "code": row.get("CODE"),
            "item": row.get("ITEM"),
            "description": row.get("DESCRIPTION"),
        }
    return lookup, li


def add_listitem_fields(df: pd.DataFrame, id_col: str, prefix: str, lookup: Dict[int, dict]) -> pd.DataFrame:
    if id_col not in df.columns:
        df[f"{prefix}_item"] = ""
        df[f"{prefix}_code"] = ""
        return df
    ids = pd.to_numeric(df[id_col], errors="coerce")
    df[f"{prefix}_item"] = ids.map(lambda x: lookup.get(int(x), {}).get("item", "") if pd.notna(x) else "")
    df[f"{prefix}_code"] = ids.map(lambda x: lookup.get(int(x), {}).get("code", "") if pd.notna(x) else "")
    return df


def build_location_hierarchy(location_path: str | Path, listitem_lookup: Optional[Dict[int, dict]] = None) -> pd.DataFrame:
    loc = read_csv(location_path)
    loc["LOCATIONID"] = pd.to_numeric(loc["LOCATIONID"], errors="coerce").astype("Int64")
    loc["PARENTLOCATIONID"] = pd.to_numeric(loc.get("PARENTLOCATIONID"), errors="coerce").astype("Int64")
    loc["location_name"] = coalesce_string(loc, ["LOCATION", "SHORTNAME", "LOCATIONCODE"])

    records = {}
    for _, r in loc.iterrows():
        if pd.isna(r["LOCATIONID"]):
            continue
        records[int(r["LOCATIONID"])] = {
            "parent": None if pd.isna(r["PARENTLOCATIONID"]) else int(r["PARENTLOCATIONID"]),
            "name": clean_text_value(r.get("location_name")),
            "type_id": r.get("LOCATIONTYPEID"),
            "category_id": r.get("LOCATIONCATEGORYID"),
        }

    root_names = {"root", "default client root", "client root"}

    def path_for(location_id: object) -> List[str]:
        if pd.isna(location_id):
            return []
        try:
            cur = int(location_id)
        except Exception:
            return []
        seen = set()
        names = []
        for _ in range(50):
            if cur in seen or cur not in records:
                break
            seen.add(cur)
            name = records[cur]["name"]
            if name:
                names.append(name)
            parent = records[cur]["parent"]
            if parent is None:
                break
            cur = parent
        names.reverse()
        return names

    paths = loc["LOCATIONID"].map(path_for)
    cleaned_paths = paths.map(lambda xs: [x for x in xs if x.strip().lower() not in root_names])
    loc["location_path"] = paths.map(lambda xs: " > ".join(xs))
    loc["location_path_clean"] = cleaned_paths.map(lambda xs: " > ".join(xs))
    max_levels = 8
    for i in range(max_levels):
        loc[f"location_level_{i+1}"] = cleaned_paths.map(lambda xs, i=i: xs[i] if len(xs) > i else "")
    loc["site"] = cleaned_paths.map(lambda xs: xs[0] if len(xs) >= 1 else "")
    loc["department"] = cleaned_paths.map(lambda xs: xs[1] if len(xs) >= 2 else (xs[0] if len(xs) == 1 else ""))
    if listitem_lookup and "LOCATIONTYPEID" in loc.columns:
        add_listitem_fields(loc, "LOCATIONTYPEID", "location_type", listitem_lookup)
    return loc


def normalize_source_type(value: object) -> str:
    text = clean_text_value(value).lower()
    text = text.replace("/", " ").replace("-", " ")
    text = re.sub(r"\s+", "_", text).strip("_")
    if text in {"hazard_identification", "hazardid"}:
        return "hazard_identification"
    if text in {"near_miss", "nearmiss"}:
        return "near_miss"
    if text == "":
        return "unknown"
    return text


def safe_numeric(s: pd.Series, fill_value: float = 0.0) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(fill_value)


def save_json(obj: dict, path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    return path


def load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_feature_columns(df: pd.DataFrame, target_cols: Sequence[str], extra_exclude: Optional[Sequence[str]] = None) -> Tuple[List[str], List[str]]:
    exclude = set(target_cols) | set(extra_exclude or [])
    numeric_cols = []
    categorical_cols = []
    for col in df.columns:
        if col in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        elif pd.api.types.is_bool_dtype(df[col]):
            numeric_cols.append(col)
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        else:
            nunique = df[col].nunique(dropna=True)
            if nunique <= 500:
                categorical_cols.append(col)
    return numeric_cols, categorical_cols


def time_based_split(df: pd.DataFrame, date_col: str = "as_of_date", test_frac: float = 0.2):
    dates = pd.to_datetime(df[date_col], errors="coerce")
    valid_dates = dates.dropna().sort_values()
    if valid_dates.empty:
        raise ValueError(f"No valid dates in {date_col}")
    cutoff = valid_dates.quantile(1 - test_frac)
    train_idx = dates <= cutoff
    test_idx = dates > cutoff
    if test_idx.sum() == 0:
        cutoff = valid_dates.iloc[int(len(valid_dates) * (1 - test_frac))]
        train_idx = dates <= cutoff
        test_idx = dates > cutoff
    return train_idx, test_idx, cutoff


def top_decile_lift(y_true: np.ndarray, y_score: np.ndarray, top_frac: float = 0.10) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    if len(y_true) == 0 or np.nansum(y_true) <= 0:
        return np.nan
    n_top = max(1, int(np.ceil(len(y_true) * top_frac)))
    order = np.argsort(-y_score)[:n_top]
    captured = np.nansum(y_true[order]) / np.nansum(y_true)
    return captured / top_frac


def precision_at_top_frac(y_true: np.ndarray, y_score: np.ndarray, top_frac: float = 0.10) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if len(y_true) == 0:
        return np.nan
    n_top = max(1, int(np.ceil(len(y_true) * top_frac)))
    order = np.argsort(-y_score)[:n_top]
    return float(np.mean(y_true[order]))


def normalize_embeddings(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms
