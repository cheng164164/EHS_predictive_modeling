"""Utilities for Sentence Embeddings + HDBSCAN pattern learning.

The functions in this module are intentionally separated from the CLI scripts so the
same cleaning, validation, reporting, and plotting logic can be reused in notebooks,
Azure ML jobs, and scheduled retraining pipelines.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import joblib
import numpy as np
import pandas as pd


NULL_TEXT_VALUES = {
    "",
    " ",
    "nan",
    "none",
    "null",
    "n/a",
    "na",
    "not applicable",
    "unknown",
    "undefined",
    "-",
    "--",
}

DEFAULT_CONTEXT_COLUMNS = [
    # Only event narrative/context fields are used as fallback text.
    # Location and category fields are deliberately excluded so the model does
    # not cluster records only because they came from the same site/category.
    "title",
    "description",
    "off_premises_location",
    "equipment",
    "vehicle",
    "other_process",
    "other_activity",
    "activity_during_incident",
]

BUSINESS_OUTPUT_COLUMNS = [
    "incident_id",
    "incident_number",
    "incident_date",
    "incident_month",
    "incident_category_name",
    "incident_status_name",
    "location_id",
    "site_name_filled",
    "department_name_filled",
    "business_unit_name_filled",
    "region_name_filled",
    "country_name_filled",
    "title",
    "description",
    "equipment",
    "vehicle",
    "ml_text_early",
    "ml_text_full",
    "text_early_word_count",
    "injury_count",
    "lost_time_any",
    "restricted_time_any",
    "fatality_any",
    "emergency_room_any",
    "inpatient_any",
    "severe_actual",
]


@dataclass
class ClusteringMetrics:
    split_name: str
    row_count: int
    clustered_count: int
    outlier_count: int
    cluster_count: int
    clustered_rate: float
    outlier_rate: float
    largest_cluster_size: int | None
    largest_cluster_share: float | None
    median_cluster_size: float | None
    mean_membership_strength: float | None
    silhouette_cosine_sample: float | None
    davies_bouldin_sample: float | None
    calinski_harabasz_sample: float | None


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(obj: object, path: Path) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def read_table(path: Path) -> pd.DataFrame:
    """Read a CSV or parquet file using the file extension."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file extension for {path}. Use .csv or .parquet.")


def write_table(df: pd.DataFrame, path: Path) -> None:
    """Write a CSV or parquet file using the file extension."""
    path = Path(path)
    ensure_parent(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=False)
    elif suffix in {".parquet", ".pq"}:
        df.to_parquet(path, index=False)
    else:
        raise ValueError(f"Unsupported output extension for {path}. Use .csv or .parquet.")


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to snake_case while preserving column values."""
    rename = {}
    for col in df.columns:
        new = str(col).strip()
        new = re.sub(r"[^0-9A-Za-z]+", "_", new)
        new = re.sub(r"_+", "_", new).strip("_").lower()
        rename[col] = new
    return df.rename(columns=rename)


def clean_scalar_text(value: object) -> str:
    """Clean a single text value for embedding input."""
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value)
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if text.lower() in NULL_TEXT_VALUES:
        return ""
    return text


def clean_text_series(series: pd.Series) -> pd.Series:
    return series.map(clean_scalar_text).astype("string")


def word_count_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.findall(r"\b\w+\b").map(len)


def parse_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    truthy = {"true", "t", "yes", "y", "1", "x"}
    return series.fillna(False).map(lambda x: str(x).strip().lower() in truthy).astype(bool)


def build_fallback_text(df: pd.DataFrame, text_col: str, context_columns: Sequence[str] | None = None) -> pd.Series:
    """Create a fallback text field if the requested text column is missing or blank."""
    context_columns = list(context_columns or DEFAULT_CONTEXT_COLUMNS)
    source_cols = []
    for col in context_columns:
        if col in df.columns and col not in source_cols:
            source_cols.append(col)

    if text_col in df.columns:
        base = clean_text_series(df[text_col])
    else:
        base = pd.Series([""] * len(df), index=df.index, dtype="string")

    if source_cols:
        fallback = (
            df[source_cols]
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1)
            .map(clean_scalar_text)
            .astype("string")
        )
        base = base.mask(base.fillna("").str.len().eq(0), fallback)
    return base


def clean_pattern_records(
    raw_df: pd.DataFrame,
    text_col: str = "ml_text_early",
    min_words: int = 3,
    id_col: str = "incident_id",
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Clean pattern records before embedding and clustering.

    Returns:
        clean_df: Records eligible for embedding/clustering.
        rejected_df: Records removed with a rejection reason.
        profile: Data-quality and filtering summary.
    """
    df = normalize_column_names(raw_df.copy())
    original_rows = len(df)

    if id_col not in df.columns:
        id_col = "row_id"
        df[id_col] = np.arange(len(df), dtype=np.int64)

    df["row_uid"] = np.arange(len(df), dtype=np.int64)
    df["source_index"] = df.index.astype(str)

    if "incident_date" in df.columns:
        df["incident_date"] = pd.to_datetime(df["incident_date"], errors="coerce", utc=True)
    else:
        df["incident_date"] = pd.NaT

    if "incident_month" not in df.columns:
        df["incident_month"] = df["incident_date"].dt.to_period("M").astype(str).replace("NaT", np.nan)

    for col in ["incident_category_name", "incident_status_name"]:
        if col not in df.columns:
            df[col] = "Unknown"
        df[col] = df[col].fillna("Unknown").astype(str).map(clean_scalar_text).replace("", "Unknown")

    for col in [
        "site_name_filled",
        "department_name_filled",
        "business_unit_name_filled",
        "region_name_filled",
        "country_name_filled",
    ]:
        if col not in df.columns:
            df[col] = "Unknown"
        df[col] = df[col].fillna("Unknown").astype(str).map(clean_scalar_text).replace("", "Unknown")

    for col in ["title", "description", "equipment", "vehicle", "ml_text_full"]:
        if col in df.columns:
            df[col] = clean_text_series(df[col])
        else:
            df[col] = ""

    df[text_col] = build_fallback_text(df, text_col=text_col)
    df["model_text"] = clean_text_series(df[text_col])
    df["model_text_word_count"] = word_count_series(df["model_text"])
    df["model_text_char_count"] = df["model_text"].fillna("").astype(str).str.len()

    for col in [
        "lost_time_any",
        "restricted_time_any",
        "fatality_any",
        "emergency_room_any",
        "inpatient_any",
        "severe_actual",
    ]:
        if col in df.columns:
            df[col] = parse_bool_series(df[col])
        else:
            df[col] = False

    if "injury_count" in df.columns:
        df["injury_count"] = pd.to_numeric(df["injury_count"], errors="coerce").fillna(0).astype(int)
    else:
        df["injury_count"] = 0

    reject_reason = pd.Series("", index=df.index, dtype="string")
    reject_reason = reject_reason.mask(df["model_text"].fillna("").str.len().eq(0), "missing_text")
    reject_reason = reject_reason.mask(
        reject_reason.eq("") & df["model_text_word_count"].fillna(0).lt(min_words),
        "text_too_short",
    )

    duplicate_mask = df.duplicated(subset=[id_col], keep="first") if id_col in df.columns else pd.Series(False, index=df.index)
    reject_reason = reject_reason.mask(reject_reason.eq("") & duplicate_mask, "duplicate_record_id")

    eligible_mask = reject_reason.eq("")
    clean_df = df.loc[eligible_mask].copy().reset_index(drop=True)
    rejected_df = df.loc[~eligible_mask].copy()
    if not rejected_df.empty:
        rejected_df["rejection_reason"] = reject_reason.loc[~eligible_mask].to_numpy()
        keep_cols = [c for c in [id_col, "row_uid", "model_text", "model_text_word_count", "rejection_reason"] if c in rejected_df.columns]
        rejected_df = rejected_df[keep_cols].reset_index(drop=True)
    else:
        rejected_df = pd.DataFrame(columns=[id_col, "row_uid", "model_text", "model_text_word_count", "rejection_reason"])

    clean_df["record_id"] = clean_df[id_col].astype(str)
    clean_df["model_text_hash"] = pd.util.hash_pandas_object(clean_df["model_text"], index=False).astype(str)

    profile = {
        "original_rows": int(original_rows),
        "eligible_rows": int(len(clean_df)),
        "rejected_rows": int(len(rejected_df)),
        "min_words": int(min_words),
        "text_column_requested": text_col,
        "id_column_used": id_col,
        "rejection_counts": rejected_df["rejection_reason"].value_counts(dropna=False).to_dict()
        if not rejected_df.empty
        else {},
        "category_counts_after_cleaning": clean_df["incident_category_name"].value_counts(dropna=False).to_dict(),
        "missing_incident_date_after_cleaning": int(clean_df["incident_date"].isna().sum()),
    }
    return clean_df, rejected_df, profile


def optionally_sample_records(
    df: pd.DataFrame,
    max_records: int | None,
    random_state: int,
    stratify_col: str = "incident_category_name",
) -> pd.DataFrame:
    """Optional reproducible stratified sampling for rapid experiments."""
    if max_records is None or max_records <= 0 or len(df) <= max_records:
        return df.copy().reset_index(drop=True)
    if stratify_col not in df.columns:
        return df.sample(n=max_records, random_state=random_state).reset_index(drop=True)

    sampled_indices: list[int] = []
    counts = df[stratify_col].fillna("Unknown").value_counts()
    for value, count in counts.items():
        n = max(1, int(round(max_records * (count / len(df)))))
        candidates = df.index[df[stratify_col].fillna("Unknown").eq(value)].to_numpy()
        selected = pd.Series(candidates).sample(n=min(n, len(candidates)), random_state=random_state).tolist()
        sampled_indices.extend(selected)

    sampled_indices = list(dict.fromkeys(sampled_indices))
    if len(sampled_indices) > max_records:
        sampled_indices = pd.Series(sampled_indices).sample(n=max_records, random_state=random_state).tolist()
    elif len(sampled_indices) < max_records:
        remaining = [idx for idx in df.index if idx not in set(sampled_indices)]
        if remaining:
            add = pd.Series(remaining).sample(n=min(max_records - len(sampled_indices), len(remaining)), random_state=random_state).tolist()
            sampled_indices.extend(add)

    sampled = df.loc[sampled_indices].sample(frac=1.0, random_state=random_state)
    return sampled.reset_index(drop=True)


def make_train_test_split(
    df: pd.DataFrame,
    split_mode: str = "time",
    test_size: float = 0.2,
    random_state: int = 42,
    min_test_rows: int = 100,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Create train/test indices for validation.

    For safety event data, a time split is preferred because it simulates training on
    history and applying the model to newer records.
    """
    if not 0 < test_size < 0.9:
        raise ValueError("test_size must be between 0 and 0.9")
    n = len(df)
    if n < 10:
        raise ValueError("At least 10 cleaned records are required for clustering validation")

    rng = np.random.default_rng(random_state)
    all_idx = np.arange(n)
    min_test_rows = min(min_test_rows, max(1, n // 5))

    if split_mode == "time" and "incident_date" in df.columns and df["incident_date"].notna().sum() >= max(10, min_test_rows):
        dated = df.reset_index().dropna(subset=["incident_date"]).sort_values("incident_date")
        test_n = max(min_test_rows, int(round(len(dated) * test_size)))
        test_n = min(test_n, len(dated) - 2)
        test_idx = dated.tail(test_n)["index"].to_numpy(dtype=int)
        train_idx = np.setdiff1d(all_idx, test_idx)
        split_info = {
            "split_mode_requested": split_mode,
            "split_mode_used": "time",
            "test_size_requested": test_size,
            "train_rows": int(len(train_idx)),
            "test_rows": int(len(test_idx)),
            "cutoff_date_exclusive": str(dated.tail(test_n)["incident_date"].min()),
        }
    else:
        shuffled = rng.permutation(all_idx)
        test_n = max(min_test_rows, int(round(n * test_size)))
        test_n = min(test_n, n - 2)
        test_idx = np.sort(shuffled[:test_n])
        train_idx = np.sort(shuffled[test_n:])
        split_info = {
            "split_mode_requested": split_mode,
            "split_mode_used": "random",
            "test_size_requested": test_size,
            "train_rows": int(len(train_idx)),
            "test_rows": int(len(test_idx)),
        }
    return train_idx, test_idx, split_info


def load_sentence_model(model_name_or_path: str, device: str | None = None):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required. Install with: pip install -r requirements-ml.txt"
        ) from exc

    kwargs = {}
    if device and device.lower() != "auto":
        kwargs["device"] = device
    return SentenceTransformer(model_name_or_path, **kwargs)


def encode_texts(
    texts: Sequence[str],
    model_name_or_path: str,
    batch_size: int = 128,
    device: str = "auto",
    normalize_embeddings: bool = True,
    show_progress_bar: bool = True,
) -> np.ndarray:
    model = load_sentence_model(model_name_or_path, device=device)
    embeddings = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
    )
    return embeddings.astype(np.float32)


def fit_umap(train_embeddings: np.ndarray, config: dict, random_state: int):
    try:
        import umap
    except ImportError as exc:
        raise ImportError("umap-learn is required. Install with: pip install -r requirements-ml.txt") from exc

    model = umap.UMAP(
        n_neighbors=int(config.get("n_neighbors", 30)),
        n_components=int(config.get("n_components", 15)),
        min_dist=float(config.get("min_dist", 0.0)),
        metric=str(config.get("metric", "cosine")),
        random_state=random_state,
        low_memory=True,
    )
    train_reduced = model.fit_transform(train_embeddings)
    return model, train_reduced.astype(np.float32)


def transform_umap(model, embeddings: np.ndarray) -> np.ndarray:
    return model.transform(embeddings).astype(np.float32)


def fit_hdbscan(train_vectors: np.ndarray, config: dict):
    try:
        import hdbscan
    except ImportError as exc:
        raise ImportError("hdbscan is required. Install with: pip install -r requirements-ml.txt") from exc

    model = hdbscan.HDBSCAN(
        min_cluster_size=int(config.get("min_cluster_size", 50)),
        min_samples=int(config.get("min_samples", 10)),
        metric=str(config.get("metric", "euclidean")),
        cluster_selection_method=str(config.get("cluster_selection_method", "eom")),
        cluster_selection_epsilon=float(config.get("cluster_selection_epsilon", 0.0)),
        prediction_data=True,
        core_dist_n_jobs=int(config.get("core_dist_n_jobs", -1)),
    )
    labels = model.fit_predict(train_vectors)
    probabilities = getattr(model, "probabilities_", np.ones(len(labels), dtype=np.float32))
    return model, labels.astype(int), probabilities.astype(np.float32)


def approximate_hdbscan_predict(model, vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    try:
        import hdbscan
    except ImportError as exc:
        raise ImportError("hdbscan is required. Install with: pip install -r requirements-ml.txt") from exc
    labels, strengths = hdbscan.approximate_predict(model, vectors)
    return labels.astype(int), strengths.astype(np.float32)


def _sample_for_metric(vectors: np.ndarray, labels: np.ndarray, max_points: int, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    mask = labels != -1
    vectors = vectors[mask]
    labels = labels[mask]
    if len(labels) > max_points:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(np.arange(len(labels)), size=max_points, replace=False)
        vectors = vectors[idx]
        labels = labels[idx]
    return vectors, labels


def compute_clustering_metrics(
    split_name: str,
    vectors: np.ndarray,
    labels: np.ndarray,
    membership_strength: np.ndarray | None = None,
    max_metric_points: int = 10000,
    random_state: int = 42,
) -> ClusteringMetrics:
    labels = np.asarray(labels).astype(int)
    row_count = int(len(labels))
    clustered_mask = labels != -1
    clustered_count = int(clustered_mask.sum())
    outlier_count = int((~clustered_mask).sum())
    cluster_labels = sorted(set(labels[clustered_mask].tolist()))
    cluster_count = int(len(cluster_labels))

    cluster_sizes = pd.Series(labels[clustered_mask]).value_counts() if clustered_count > 0 else pd.Series(dtype=int)
    largest_cluster_size = int(cluster_sizes.max()) if not cluster_sizes.empty else None
    largest_cluster_share = float(largest_cluster_size / row_count) if largest_cluster_size is not None and row_count else None
    median_cluster_size = float(cluster_sizes.median()) if not cluster_sizes.empty else None

    mean_strength = None
    if membership_strength is not None and len(membership_strength) == row_count:
        mean_strength = float(np.nanmean(np.asarray(membership_strength, dtype=float)))

    silhouette = None
    davies_bouldin = None
    calinski = None
    if cluster_count >= 2 and clustered_count > cluster_count:
        try:
            from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

            sample_vectors, sample_labels = _sample_for_metric(vectors, labels, max_metric_points, random_state)
            if len(set(sample_labels.tolist())) >= 2 and len(sample_labels) > len(set(sample_labels.tolist())):
                silhouette = float(silhouette_score(sample_vectors, sample_labels, metric="cosine"))
                davies_bouldin = float(davies_bouldin_score(sample_vectors, sample_labels))
                calinski = float(calinski_harabasz_score(sample_vectors, sample_labels))
        except Exception:
            silhouette = None
            davies_bouldin = None
            calinski = None

    return ClusteringMetrics(
        split_name=split_name,
        row_count=row_count,
        clustered_count=clustered_count,
        outlier_count=outlier_count,
        cluster_count=cluster_count,
        clustered_rate=float(clustered_count / row_count) if row_count else 0.0,
        outlier_rate=float(outlier_count / row_count) if row_count else 0.0,
        largest_cluster_size=largest_cluster_size,
        largest_cluster_share=largest_cluster_share,
        median_cluster_size=median_cluster_size,
        mean_membership_strength=mean_strength,
        silhouette_cosine_sample=silhouette,
        davies_bouldin_sample=davies_bouldin,
        calinski_harabasz_sample=calinski,
    )


def compute_top_terms(
    df: pd.DataFrame,
    labels: Sequence[int],
    text_col: str = "model_text",
    max_features: int = 20000,
    top_n: int = 10,
    min_df: int = 2,
) -> pd.DataFrame:
    """Extract top TF-IDF terms for each cluster for explainability."""
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

    labels = np.asarray(labels).astype(int)
    valid_mask = labels != -1
    if valid_mask.sum() == 0:
        return pd.DataFrame(columns=["cluster_id", "top_terms"])

    SPANISH_STOP_WORDS = {
    "a", "al", "algo", "algunos", "ante", "antes", "como", "con", "contra",
    "cual", "cuando", "de", "del", "desde", "donde", "durante", "e", "el",
    "ella", "ellas", "ellos", "en", "entre", "era", "eran", "es", "esa",
    "esas", "ese", "eso", "esos", "esta", "estaba", "estado", "estan",
    "estar", "este", "esto", "estos", "fue", "fueron", "ha", "han",
    "hasta", "hay", "la", "las", "le", "les", "lo", "los", "mas",
    "me", "mi", "mis", "muy", "no", "nos", "o", "para", "pero", "por",
    "que", "se", "sin", "sobre", "su", "sus", "tambien", "te", "tiene",
    "un", "una", "unas", "uno", "unos", "y", "ya"
    }

    custom_stop = set(ENGLISH_STOP_WORDS).union(SPANISH_STOP_WORDS).union({
        "employee",
        "employees",
        "near",
        "miss",
        "hazard",
        "incident",
        "area",
        "observed",
        "identified",
        "reported",
        "work",
        "working",
        "safe",
        "safety",
    })
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        ngram_range=(1, 2),
        stop_words=list(custom_stop),
    )
    texts = df[text_col].fillna("").astype(str).tolist()
    try:
        matrix = vectorizer.fit_transform(texts)
    except ValueError:
        return pd.DataFrame(columns=["cluster_id", "top_terms"])
    feature_names = np.asarray(vectorizer.get_feature_names_out())

    rows = []
    for cluster_id in sorted(set(labels.tolist())):
        if cluster_id == -1:
            continue
        idx = np.where(labels == cluster_id)[0]
        if len(idx) == 0:
            continue
        mean_scores = np.asarray(matrix[idx].mean(axis=0)).ravel()
        top_idx = mean_scores.argsort()[::-1][:top_n]
        terms = [feature_names[i] for i in top_idx if mean_scores[i] > 0]
        rows.append({
            "cluster_id": int(cluster_id),
            "top_terms": ", ".join(terms),
            "suggested_cluster_label": " / ".join(terms[:3]) if terms else f"Cluster {cluster_id}",
        })
    return pd.DataFrame(rows)


def representative_records(
    df: pd.DataFrame,
    labels: Sequence[int],
    membership_strength: Sequence[float] | None = None,
    top_k: int = 5,
) -> pd.DataFrame:
    """Select representative records per cluster using membership strength when available."""
    labels = np.asarray(labels).astype(int)
    out = df.copy()
    out["cluster_id"] = labels
    if membership_strength is None:
        out["membership_strength"] = 1.0
    else:
        out["membership_strength"] = np.asarray(membership_strength, dtype=float)

    keep = [c for c in [
        "cluster_id",
        "membership_strength",
        "incident_id",
        "incident_number",
        "incident_date",
        "incident_category_name",
        "site_name_filled",
        "department_name_filled",
        "title",
        "description",
        "model_text",
        "severe_actual",
    ] if c in out.columns]
    reps = []
    for cluster_id, part in out[out["cluster_id"] != -1].groupby("cluster_id"):
        reps.append(part.sort_values("membership_strength", ascending=False).head(top_k)[keep])
    if not reps:
        return pd.DataFrame(columns=keep)
    return pd.concat(reps, ignore_index=True)


def build_cluster_summary(
    df: pd.DataFrame,
    labels: Sequence[int],
    membership_strength: Sequence[float] | None = None,
    top_terms: pd.DataFrame | None = None,
) -> pd.DataFrame:
    labels = np.asarray(labels).astype(int)
    temp = df.copy()
    temp["cluster_id"] = labels
    if membership_strength is None:
        temp["membership_strength"] = np.where(labels == -1, 0.0, 1.0)
    else:
        temp["membership_strength"] = np.asarray(membership_strength, dtype=float)

    rows = []
    for cluster_id, part in temp.groupby("cluster_id", dropna=False):
        cluster_id_int = int(cluster_id)
        if cluster_id_int == -1:
            label = "Outlier / unassigned"
        else:
            label = f"Cluster {cluster_id_int}"
        row = {
            "cluster_id": cluster_id_int,
            "cluster_label": label,
            "record_count": int(len(part)),
            "near_miss_count": int(part["incident_category_name"].eq("Near Miss").sum()) if "incident_category_name" in part.columns else 0,
            "hazard_identification_count": int(part["incident_category_name"].eq("Hazard Identification").sum()) if "incident_category_name" in part.columns else 0,
            "severe_actual_count": int(part["severe_actual"].fillna(False).astype(bool).sum()) if "severe_actual" in part.columns else 0,
            "severe_actual_rate": float(part["severe_actual"].fillna(False).astype(bool).mean()) if "severe_actual" in part.columns else 0.0,
            "mean_membership_strength": float(part["membership_strength"].mean()) if len(part) else None,
            "top_site": part["site_name_filled"].mode().iat[0] if "site_name_filled" in part.columns and not part["site_name_filled"].mode().empty else None,
            "top_department": part["department_name_filled"].mode().iat[0] if "department_name_filled" in part.columns and not part["department_name_filled"].mode().empty else None,
            "representative_incident_ids": ", ".join(part.sort_values("membership_strength", ascending=False).head(5)["record_id"].astype(str).tolist()) if "record_id" in part.columns else "",
        }
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values(["cluster_id"], ascending=True).reset_index(drop=True)

    if top_terms is not None and not top_terms.empty:
        summary = summary.merge(top_terms, on="cluster_id", how="left")
        mask = summary["cluster_id"].ne(-1) & summary.get("suggested_cluster_label", pd.Series(index=summary.index)).notna()
        summary.loc[mask, "cluster_label"] = summary.loc[mask, "suggested_cluster_label"]
    else:
        summary["top_terms"] = ""
        summary["suggested_cluster_label"] = summary["cluster_label"]

    return summary



def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """Normalize matrix rows to unit length, leaving all-zero rows unchanged."""
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (matrix / norms).astype(np.float32)


def compute_cluster_centroids(
    vectors: np.ndarray,
    labels: Sequence[int],
    membership_strength: Sequence[float] | None = None,
    use_membership_weights: bool = True,
    min_records_per_cluster: int = 1,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Compute one centroid vector per non-outlier HDBSCAN cluster.

    The centroid is a generic representation of the cluster in vector space. It is
    used to learn higher-level themes without any hand-selected keywords or
    supervised target labels.

    Args:
        vectors: Record-level vectors. This can be original sentence embeddings
            or UMAP-reduced vectors.
        labels: HDBSCAN cluster labels aligned to vectors.
        membership_strength: Optional HDBSCAN membership strengths aligned to
            vectors.
        use_membership_weights: If True, clear cluster members contribute more
            to the centroid than weak/borderline members.
        min_records_per_cluster: Clusters smaller than this are not used to fit
            the theme model. Normally this should stay at 1 because HDBSCAN's
            MIN_CLUSTER_SIZE already controls cluster size.

    Returns:
        cluster_ids: Array of cluster IDs represented by the centroids.
        centroids: Array of normalized centroid vectors.
        centroid_summary: Per-cluster record counts and mean membership strength.
    """
    vectors = np.asarray(vectors, dtype=np.float32)
    labels = np.asarray(labels).astype(int)
    if len(vectors) != len(labels):
        raise ValueError("vectors and labels must have the same length")

    if membership_strength is None:
        strengths = np.ones(len(labels), dtype=np.float32)
    else:
        strengths = np.asarray(membership_strength, dtype=np.float32)
        if len(strengths) != len(labels):
            raise ValueError("membership_strength and labels must have the same length")

    rows: list[dict] = []
    centroids: list[np.ndarray] = []
    cluster_ids: list[int] = []
    min_records_per_cluster = max(1, int(min_records_per_cluster))

    for cluster_id in sorted(set(labels.tolist())):
        cluster_id_int = int(cluster_id)
        if cluster_id_int == -1:
            continue
        idx = np.where(labels == cluster_id_int)[0]
        if len(idx) < min_records_per_cluster:
            continue

        part_vectors = vectors[idx]
        part_strengths = strengths[idx]
        if use_membership_weights:
            weights = np.clip(part_strengths, 0.0, None)
            if not np.isfinite(weights).all() or float(weights.sum()) <= 0:
                centroid = part_vectors.mean(axis=0)
            else:
                centroid = np.average(part_vectors, axis=0, weights=weights)
        else:
            centroid = part_vectors.mean(axis=0)

        centroids.append(np.asarray(centroid, dtype=np.float32))
        cluster_ids.append(cluster_id_int)
        rows.append({
            "cluster_id": cluster_id_int,
            "cluster_record_count": int(len(idx)),
            "cluster_mean_membership_strength": float(np.nanmean(part_strengths)),
        })

    if not centroids:
        return (
            np.asarray([], dtype=int),
            np.empty((0, vectors.shape[1] if vectors.ndim == 2 else 0), dtype=np.float32),
            pd.DataFrame(columns=["cluster_id", "cluster_record_count", "cluster_mean_membership_strength"]),
        )

    centroid_array = _normalize_rows(np.vstack(centroids))
    return np.asarray(cluster_ids, dtype=int), centroid_array, pd.DataFrame(rows)


def _make_agglomerative_model(
    n_clusters: int | None,
    distance_threshold: float | None,
    metric: str,
    linkage: str,
):
    """Create an AgglomerativeClustering model across sklearn versions."""
    from sklearn.cluster import AgglomerativeClustering

    kwargs = {
        "n_clusters": n_clusters,
        "distance_threshold": distance_threshold,
        "linkage": linkage,
        "compute_full_tree": True,
    }
    try:
        return AgglomerativeClustering(metric=metric, **kwargs)
    except TypeError:
        # Older sklearn releases used affinity instead of metric.
        return AgglomerativeClustering(affinity=metric, **kwargs)


def fit_cluster_theme_model(
    cluster_ids: Sequence[int],
    centroids: np.ndarray,
    cluster_summary: pd.DataFrame | None,
    config: dict,
    random_state: int = 42,
) -> tuple[dict | None, pd.DataFrame, dict]:
    """Learn a second-level theme grouping from cluster centroid vectors.

    This groups clusters into broader themes using only vector-space distance
    between cluster centroids. No result-specific keywords, site names, labels,
    severe outcomes, or future information are used to fit themes.
    """
    del random_state  # Agglomerative clustering is deterministic for fixed input.

    cluster_ids = np.asarray(cluster_ids).astype(int)
    centroids = np.asarray(centroids, dtype=np.float32)
    if len(cluster_ids) != len(centroids):
        raise ValueError("cluster_ids and centroids must have the same length")

    empty_map = pd.DataFrame(columns=[
        "cluster_id",
        "theme_id",
        "raw_theme_id",
        "cluster_record_count",
    ])
    if len(cluster_ids) == 0:
        return None, empty_map, {"theme_count": 0, "theme_method": config.get("method", "agglomerative")}

    if cluster_summary is not None and not cluster_summary.empty and "record_count" in cluster_summary.columns:
        count_map = cluster_summary.set_index("cluster_id")["record_count"].to_dict()
        cluster_record_counts = np.asarray([int(count_map.get(int(cid), 1)) for cid in cluster_ids], dtype=int)
    else:
        cluster_record_counts = np.ones(len(cluster_ids), dtype=int)

    if len(cluster_ids) == 1:
        mapping = pd.DataFrame({
            "cluster_id": cluster_ids,
            "theme_id": np.asarray([0], dtype=int),
            "raw_theme_id": np.asarray([0], dtype=int),
            "cluster_record_count": cluster_record_counts,
        })
        info = {
            "theme_method": config.get("method", "agglomerative"),
            "theme_count": 1,
            "theme_distance_threshold": None,
            "theme_n_clusters": 1,
        }
        return None, mapping, info

    method = str(config.get("method", "agglomerative")).lower()
    if method != "agglomerative":
        raise ValueError(f"Unsupported theme method: {method}. Supported: agglomerative")

    configured_n_clusters = config.get("n_clusters", None)
    n_clusters = None
    if configured_n_clusters is not None:
        try:
            configured_n_clusters = int(configured_n_clusters)
        except (TypeError, ValueError):
            configured_n_clusters = None
    if configured_n_clusters is not None and configured_n_clusters > 0:
        n_clusters = min(configured_n_clusters, len(cluster_ids))
        distance_threshold = None
    else:
        n_clusters = None
        distance_threshold = float(config.get("distance_threshold", 0.25))
        if distance_threshold <= 0:
            distance_threshold = 0.25

    metric = str(config.get("metric", "cosine"))
    linkage = str(config.get("linkage", "average"))
    if linkage == "ward" and metric != "euclidean":
        raise ValueError("Agglomerative linkage='ward' requires metric='euclidean'")

    model = _make_agglomerative_model(
        n_clusters=n_clusters,
        distance_threshold=distance_threshold,
        metric=metric,
        linkage=linkage,
    )
    raw_labels = model.fit_predict(centroids).astype(int)

    mapping = pd.DataFrame({
        "cluster_id": cluster_ids.astype(int),
        "raw_theme_id": raw_labels.astype(int),
        "cluster_record_count": cluster_record_counts.astype(int),
    })

    # Re-number themes by descending total record count for easier reporting.
    theme_order = (
        mapping.groupby("raw_theme_id", dropna=False)["cluster_record_count"]
        .sum()
        .sort_values(ascending=False)
        .index
        .tolist()
    )
    raw_to_theme = {int(raw_id): int(i) for i, raw_id in enumerate(theme_order)}
    mapping["theme_id"] = mapping["raw_theme_id"].map(raw_to_theme).astype(int)
    mapping = mapping[["cluster_id", "theme_id", "raw_theme_id", "cluster_record_count"]].sort_values(
        ["theme_id", "cluster_record_count", "cluster_id"],
        ascending=[True, False, True],
    ).reset_index(drop=True)

    info = {
        "theme_method": method,
        "theme_count": int(mapping["theme_id"].nunique()),
        "theme_distance_threshold": distance_threshold,
        "theme_n_clusters": n_clusters,
        "theme_metric": metric,
        "theme_linkage": linkage,
        "cluster_count_used_for_theme_fit": int(len(cluster_ids)),
    }
    bundle = {
        "model": model,
        "cluster_ids": cluster_ids.astype(int),
        "cluster_centroids": centroids.astype(np.float32),
        "raw_to_theme": raw_to_theme,
        "config": dict(config),
        "info": info,
    }
    return bundle, mapping, info


def attach_theme_ids(scored_df: pd.DataFrame, cluster_theme_map: pd.DataFrame) -> pd.DataFrame:
    """Attach theme_id to records from a cluster-to-theme mapping."""
    out = scored_df.copy()
    if cluster_theme_map is None or cluster_theme_map.empty:
        out["theme_id"] = np.where(out.get("cluster_id", -1).eq(-1), -1, np.nan)
        return out
    mapping = cluster_theme_map[["cluster_id", "theme_id"]].drop_duplicates("cluster_id")
    out = out.merge(mapping, on="cluster_id", how="left")
    out["theme_id"] = out["theme_id"].fillna(-1).astype(int)
    return out


def compute_theme_top_terms(
    scored_df: pd.DataFrame,
    text_col: str = "model_text",
    top_n: int = 12,
) -> pd.DataFrame:
    """Create theme-level explanatory terms after themes have already been fit."""
    if "theme_id" not in scored_df.columns or scored_df.empty:
        return pd.DataFrame(columns=["theme_id", "theme_top_terms", "suggested_theme_label"])
    temp = scored_df.copy()
    labels = temp["theme_id"].fillna(-1).astype(int).to_numpy()
    terms = compute_top_terms(temp, labels, text_col=text_col, top_n=top_n)
    if terms.empty:
        return pd.DataFrame(columns=["theme_id", "theme_top_terms", "suggested_theme_label"])
    terms = terms.rename(columns={
        "cluster_id": "theme_id",
        "top_terms": "theme_top_terms",
        "suggested_cluster_label": "suggested_theme_label",
    })
    return terms[["theme_id", "theme_top_terms", "suggested_theme_label"]]


def build_theme_summary(
    scored_df: pd.DataFrame,
    theme_top_terms: pd.DataFrame | None = None,
    top_clusters_n: int = 8,
) -> pd.DataFrame:
    """Build one summary row per theme."""
    if "theme_id" not in scored_df.columns:
        return pd.DataFrame()

    temp = scored_df.copy()
    temp["theme_id"] = temp["theme_id"].fillna(-1).astype(int)
    rows = []
    for theme_id, part in temp.groupby("theme_id", dropna=False):
        theme_id_int = int(theme_id)
        if theme_id_int == -1:
            label = "Outlier / unassigned theme"
        else:
            label = f"Theme {theme_id_int}"

        non_outlier_clusters = part.loc[part["cluster_id"].ne(-1), "cluster_id"] if "cluster_id" in part.columns else pd.Series(dtype=int)
        top_cluster_labels = ""
        top_cluster_ids = ""
        if "cluster_label" in part.columns and "cluster_id" in part.columns:
            cluster_counts = (
                part[part["cluster_id"] != -1]
                .groupby(["cluster_id", "cluster_label"], dropna=False)
                .size()
                .reset_index(name="record_count")
                .sort_values("record_count", ascending=False)
                .head(int(top_clusters_n))
            )
            if not cluster_counts.empty:
                top_cluster_labels = " | ".join(cluster_counts["cluster_label"].astype(str).tolist())
                top_cluster_ids = ", ".join(cluster_counts["cluster_id"].astype(str).tolist())

        row = {
            "theme_id": theme_id_int,
            "theme_label": label,
            "record_count": int(len(part)),
            "cluster_count": int(non_outlier_clusters.nunique()) if len(non_outlier_clusters) else 0,
            "near_miss_count": int(part["incident_category_name"].eq("Near Miss").sum()) if "incident_category_name" in part.columns else 0,
            "hazard_identification_count": int(part["incident_category_name"].eq("Hazard Identification").sum()) if "incident_category_name" in part.columns else 0,
            "severe_actual_count": int(part["severe_actual"].fillna(False).astype(bool).sum()) if "severe_actual" in part.columns else 0,
            "severe_actual_rate": float(part["severe_actual"].fillna(False).astype(bool).mean()) if "severe_actual" in part.columns and len(part) else 0.0,
            "mean_membership_strength": float(part["membership_strength"].mean()) if "membership_strength" in part.columns and len(part) else None,
            "top_site": part["site_name_filled"].mode().iat[0] if "site_name_filled" in part.columns and not part["site_name_filled"].mode().empty else None,
            "top_department": part["department_name_filled"].mode().iat[0] if "department_name_filled" in part.columns and not part["department_name_filled"].mode().empty else None,
            "top_cluster_ids": top_cluster_ids,
            "top_cluster_labels": top_cluster_labels,
            "representative_incident_ids": ", ".join(part.sort_values("membership_strength", ascending=False).head(5)["record_id"].astype(str).tolist()) if "record_id" in part.columns and "membership_strength" in part.columns else "",
        }
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values(["theme_id"], ascending=True).reset_index(drop=True)
    if theme_top_terms is not None and not theme_top_terms.empty:
        summary = summary.merge(theme_top_terms, on="theme_id", how="left")
        mask = summary["theme_id"].ne(-1) & summary.get("suggested_theme_label", pd.Series(index=summary.index)).notna()
        summary.loc[mask, "theme_label"] = summary.loc[mask, "suggested_theme_label"]
    else:
        summary["theme_top_terms"] = ""
        summary["suggested_theme_label"] = summary["theme_label"]
    return summary


def attach_themes_to_cluster_summary(
    cluster_summary: pd.DataFrame,
    cluster_theme_map: pd.DataFrame,
    theme_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Attach theme fields to the cluster summary / label map."""
    out = cluster_summary.copy()
    if cluster_theme_map is not None and not cluster_theme_map.empty:
        out = out.merge(cluster_theme_map[["cluster_id", "theme_id"]].drop_duplicates("cluster_id"), on="cluster_id", how="left")
    else:
        out["theme_id"] = np.nan
    out["theme_id"] = out["theme_id"].fillna(-1).astype(int)

    if theme_summary is not None and not theme_summary.empty:
        theme_cols = [c for c in ["theme_id", "theme_label", "theme_top_terms"] if c in theme_summary.columns]
        out = out.merge(theme_summary[theme_cols].drop_duplicates("theme_id"), on="theme_id", how="left")
    else:
        out["theme_label"] = np.where(out["theme_id"].eq(-1), "Outlier / unassigned theme", out["theme_id"].map(lambda x: f"Theme {x}"))
        out["theme_top_terms"] = ""
    return out


def build_theme_site_summary(scored_df: pd.DataFrame) -> pd.DataFrame:
    """Build site/department counts by learned theme."""
    keys = ["theme_id", "theme_label", "site_name_filled", "department_name_filled"]
    keys = [c for c in keys if c in scored_df.columns]
    if not keys:
        return pd.DataFrame()
    agg = scored_df.groupby(keys, dropna=False).agg(
        record_count=("record_id", "count"),
        cluster_count=("cluster_id", lambda s: int(pd.Series(s).loc[pd.Series(s).ne(-1)].nunique())) if "cluster_id" in scored_df.columns else ("record_id", "size"),
        severe_actual_count=("severe_actual", lambda s: int(s.fillna(False).astype(bool).sum())) if "severe_actual" in scored_df.columns else ("record_id", "size"),
    ).reset_index()
    if "incident_category_name" in scored_df.columns:
        cat = pd.crosstab(
            [scored_df[c] for c in keys],
            scored_df["incident_category_name"],
        ).reset_index()
        agg = agg.merge(cat, on=keys, how="left")
    return agg.sort_values("record_count", ascending=False).reset_index(drop=True)


def build_theme_monthly_trend(scored_df: pd.DataFrame) -> pd.DataFrame:
    """Build monthly theme trends using the same generic logic as cluster trends."""
    if "incident_month" not in scored_df.columns or "theme_id" not in scored_df.columns:
        return pd.DataFrame()
    df = scored_df.copy()
    df["incident_month"] = df["incident_month"].fillna("Unknown").astype(str)
    group_cols = ["theme_id", "theme_label", "incident_month"]
    group_cols = [c for c in group_cols if c in df.columns]
    trend = df.groupby(group_cols, dropna=False).agg(record_count=("record_id", "count")).reset_index()
    trend = trend.sort_values(["theme_id", "incident_month"])
    trend["rolling_3_month_count"] = trend.groupby("theme_id")["record_count"].transform(lambda s: s.rolling(3, min_periods=1).sum())

    def flag(group: pd.DataFrame) -> pd.DataFrame:
        values = group["record_count"].to_numpy(dtype=float)
        if len(values) < 4:
            group["trend_flag"] = "Insufficient history"
            return group
        recent = np.nanmean(values[-3:])
        prior = np.nanmean(values[:-3]) if len(values[:-3]) else np.nan
        if not np.isfinite(prior) or prior == 0:
            trend_value = "New or sparse pattern" if recent > 0 else "Stable"
        elif recent >= prior * 1.25 and recent - prior >= 2:
            trend_value = "Increasing"
        elif recent <= prior * 0.75 and prior - recent >= 2:
            trend_value = "Decreasing"
        else:
            trend_value = "Stable"
        group["trend_flag"] = trend_value
        return group

    return trend.groupby("theme_id", group_keys=False).apply(flag).reset_index(drop=True)

def attach_cluster_labels(scored_df: pd.DataFrame, cluster_summary: pd.DataFrame) -> pd.DataFrame:
    """Attach cluster and optional theme labels to scored records.

    Older model artifacts only contain cluster columns. Newer artifacts may also
    contain theme columns. This function handles both cases so the scoring script
    remains backward compatible.
    """
    cols = [
        "cluster_id",
        "cluster_label",
        "top_terms",
        "theme_id",
        "theme_label",
        "theme_top_terms",
    ]
    available = [c for c in cols if c in cluster_summary.columns]
    return scored_df.merge(cluster_summary[available], on="cluster_id", how="left")


def build_cluster_site_summary(scored_df: pd.DataFrame) -> pd.DataFrame:
    keys = ["cluster_id", "cluster_label", "site_name_filled", "department_name_filled"]
    keys = [c for c in keys if c in scored_df.columns]
    if not keys:
        return pd.DataFrame()
    agg = scored_df.groupby(keys, dropna=False).agg(
        record_count=("record_id", "count"),
        severe_actual_count=("severe_actual", lambda s: int(s.fillna(False).astype(bool).sum())) if "severe_actual" in scored_df.columns else ("record_id", "size"),
    ).reset_index()
    if "incident_category_name" in scored_df.columns:
        cat = pd.crosstab(
            [scored_df[c] for c in keys],
            scored_df["incident_category_name"],
        ).reset_index()
        agg = agg.merge(cat, on=keys, how="left")
    return agg.sort_values("record_count", ascending=False).reset_index(drop=True)


def build_cluster_monthly_trend(scored_df: pd.DataFrame) -> pd.DataFrame:
    if "incident_month" not in scored_df.columns:
        return pd.DataFrame()
    df = scored_df.copy()
    df["incident_month"] = df["incident_month"].fillna("Unknown").astype(str)
    group_cols = ["cluster_id", "cluster_label", "incident_month"]
    trend = df.groupby(group_cols, dropna=False).agg(record_count=("record_id", "count")).reset_index()
    trend = trend.sort_values(["cluster_id", "incident_month"])
    trend["rolling_3_month_count"] = trend.groupby("cluster_id")["record_count"].transform(lambda s: s.rolling(3, min_periods=1).sum())

    def flag(group: pd.DataFrame) -> pd.DataFrame:
        values = group["record_count"].to_numpy(dtype=float)
        if len(values) < 4:
            group["trend_flag"] = "Insufficient history"
            return group
        recent = np.nanmean(values[-3:])
        prior = np.nanmean(values[:-3]) if len(values[:-3]) else np.nan
        if not np.isfinite(prior) or prior == 0:
            trend_value = "New or sparse pattern" if recent > 0 else "Stable"
        elif recent >= prior * 1.25 and recent - prior >= 2:
            trend_value = "Increasing"
        elif recent <= prior * 0.75 and prior - recent >= 2:
            trend_value = "Decreasing"
        else:
            trend_value = "Stable"
        group["trend_flag"] = trend_value
        return group

    return trend.groupby("cluster_id", group_keys=False).apply(flag).reset_index(drop=True)


def plot_cluster_size_distribution(cluster_summary: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    ensure_parent(path)
    data = cluster_summary[cluster_summary["cluster_id"] != -1].copy()
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    data.sort_values("record_count", ascending=False)["record_count"].head(50).plot(kind="bar", ax=ax)
    ax.set_title("Top Cluster Sizes")
    ax.set_xlabel("Cluster rank")
    ax.set_ylabel("Record count")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_membership_strength(strengths: Sequence[float], path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    ensure_parent(path)
    values = np.asarray(strengths, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(values, bins=30)
    ax.set_title(title)
    ax.set_xlabel("Membership strength")
    ax.set_ylabel("Record count")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_outlier_rate_by_month(scored_df: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    if "incident_month" not in scored_df.columns:
        return
    ensure_parent(path)
    temp = scored_df.copy()
    temp = temp[temp["incident_month"].notna()]
    if temp.empty:
        return
    month = temp.groupby("incident_month").agg(
        record_count=("record_id", "count"),
        outlier_count=("cluster_id", lambda s: int((s == -1).sum())),
    ).reset_index()
    month["outlier_rate"] = month["outlier_count"] / month["record_count"].replace(0, np.nan)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(month["incident_month"], month["outlier_rate"], marker="o")
    ax.set_title("Outlier Rate by Incident Month")
    ax.set_xlabel("Incident month")
    ax.set_ylabel("Outlier rate")
    ax.tick_params(axis="x", rotation=90)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_top_clusters_by_severe(cluster_summary: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    ensure_parent(path)
    data = cluster_summary[(cluster_summary["cluster_id"] != -1) & (cluster_summary["severe_actual_count"] > 0)].copy()
    if data.empty:
        return
    data = data.sort_values("severe_actual_count", ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(10, 6))
    data.set_index("cluster_label")["severe_actual_count"].plot(kind="barh", ax=ax)
    ax.set_title("Top Clusters by Historical Severe Actual Count")
    ax.set_xlabel("Severe actual count")
    ax.set_ylabel("Cluster label")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def compact_business_columns(df: pd.DataFrame) -> pd.DataFrame:
    base = [c for c in BUSINESS_OUTPUT_COLUMNS if c in df.columns]
    model_cols = [
        "record_id",
        "row_uid",
        "model_text",
        "model_text_word_count",
        "model_text_char_count",
        "cluster_id",
        "cluster_label",
        "top_terms",
        "theme_id",
        "theme_label",
        "theme_top_terms",
        "membership_strength",
        "is_outlier",
        "split",
    ]
    cols = []
    for col in base + model_cols:
        if col in df.columns and col not in cols:
            cols.append(col)
    return df[cols].copy()


def save_model_artifacts(
    output_dir: Path,
    sentence_model_name: str,
    umap_model,
    hdbscan_model,
    config: dict,
    cluster_summary: pd.DataFrame,
    theme_summary: pd.DataFrame | None = None,
    cluster_theme_map: pd.DataFrame | None = None,
    theme_model_bundle: dict | None = None,
) -> None:
    artifact_dir = output_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "sentence_model_name.txt").write_text(sentence_model_name, encoding="utf-8")
    if umap_model is not None:
        joblib.dump(umap_model, artifact_dir / "umap_model.joblib")
    joblib.dump(hdbscan_model, artifact_dir / "hdbscan_model.joblib")
    if theme_model_bundle is not None:
        joblib.dump(theme_model_bundle, artifact_dir / "theme_model.joblib")
    write_json(config, artifact_dir / "model_config.json")
    cluster_summary.to_csv(artifact_dir / "cluster_label_map.csv", index=False)
    if theme_summary is not None:
        theme_summary.to_csv(artifact_dir / "theme_summary.csv", index=False)
    if cluster_theme_map is not None:
        cluster_theme_map.to_csv(artifact_dir / "cluster_theme_map.csv", index=False)


def load_model_artifacts(artifact_dir: Path) -> dict:
    artifact_dir = Path(artifact_dir)
    hdbscan_path = artifact_dir / "hdbscan_model.joblib"
    if not hdbscan_path.exists():
        raise FileNotFoundError(f"Missing HDBSCAN model: {hdbscan_path}")
    sentence_path = artifact_dir / "sentence_model_name.txt"
    if not sentence_path.exists():
        raise FileNotFoundError(f"Missing sentence model name: {sentence_path}")
    config_path = artifact_dir / "model_config.json"
    label_map_path = artifact_dir / "cluster_label_map.csv"
    theme_summary_path = artifact_dir / "theme_summary.csv"
    cluster_theme_map_path = artifact_dir / "cluster_theme_map.csv"
    theme_model_path = artifact_dir / "theme_model.joblib"
    return {
        "sentence_model_name": sentence_path.read_text(encoding="utf-8").strip(),
        "umap_model": joblib.load(artifact_dir / "umap_model.joblib") if (artifact_dir / "umap_model.joblib").exists() else None,
        "hdbscan_model": joblib.load(hdbscan_path),
        "theme_model_bundle": joblib.load(theme_model_path) if theme_model_path.exists() else None,
        "config": json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {},
        "cluster_summary": pd.read_csv(label_map_path) if label_map_path.exists() else pd.DataFrame(),
        "theme_summary": pd.read_csv(theme_summary_path) if theme_summary_path.exists() else pd.DataFrame(),
        "cluster_theme_map": pd.read_csv(cluster_theme_map_path) if cluster_theme_map_path.exists() else pd.DataFrame(),
    }


def metrics_to_frame(metrics: Sequence[ClusteringMetrics]) -> pd.DataFrame:
    return pd.DataFrame([asdict(m) for m in metrics])
