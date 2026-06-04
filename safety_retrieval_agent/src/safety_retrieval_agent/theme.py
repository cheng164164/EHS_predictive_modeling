"""Data-driven risk-theme discovery and theme profile generation."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import Settings
from .utils import clean_text_value, compress_json_field, ensure_dir, list_top_counts, preview, save_json

THEME_ID_COL = "risk_theme_id"
THEME_NAME_COL = "risk_theme_name"


def _has_existing_theme_columns(df: pd.DataFrame) -> bool:
    candidates = [
        ("risk_theme_id", "risk_theme_name"),
        ("theme_id", "theme_name"),
        ("cluster_id", "cluster_label"),
        ("cluster_label", "risk_theme_name"),
    ]
    for id_col, name_col in candidates:
        if id_col in df.columns and name_col in df.columns:
            valid = df[id_col].notna() & df[name_col].notna() & df[name_col].astype(str).str.strip().ne("")
            if valid.mean() > 0.20:
                return True
    return False


def standardize_existing_theme_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "risk_theme_id" in out.columns and "risk_theme_name" in out.columns:
        return out
    mapping_options = [
        ("theme_id", "theme_name"),
        ("cluster_id", "cluster_label"),
        ("cluster_label", "risk_theme_name"),
    ]
    for id_col, name_col in mapping_options:
        if id_col in out.columns and name_col in out.columns:
            out[THEME_ID_COL] = out[id_col].astype(str)
            out[THEME_NAME_COL] = out[name_col].astype(str)
            return out
    return out


def _theme_training_mask(df: pd.DataFrame) -> np.ndarray:
    """Prefer narrative safety records for theme discovery.

    Tasks are useful for action recommendations but can dominate clusters with
    action wording, so they are not the default training population. They are
    still assigned to themes after the clustering model is trained.
    """
    source_type = df.get("source_type", pd.Series("", index=df.index)).fillna("").astype(str).str.lower()
    role = df.get("source_role", pd.Series("", index=df.index)).fillna("").astype(str).str.lower()
    mask = source_type.isin(["hazard_identification", "near_miss", "incident", "audit"])
    mask &= ~role.isin(["inspection"])
    return mask.to_numpy()


def discover_themes(df: pd.DataFrame, vectors: np.ndarray, settings: Settings) -> tuple[pd.DataFrame, MiniBatchKMeans | None]:
    """Assign data-driven risk themes.

    If existing theme columns are present, they are reused. Otherwise, embeddings
    are clustered with MiniBatchKMeans and theme names are generated from top
    TF-IDF terms in each cluster.
    """
    if _has_existing_theme_columns(df):
        out = standardize_existing_theme_columns(df)
        out[THEME_ID_COL] = out[THEME_ID_COL].fillna("theme_unknown").astype(str)
        out[THEME_NAME_COL] = out[THEME_NAME_COL].fillna("Unknown theme").astype(str)
        return out, None

    out = df.copy()
    if vectors.shape[0] != len(out):
        raise ValueError("vectors row count must match dataframe row count")

    mask = _theme_training_mask(out)
    train_idx = np.flatnonzero(mask)
    if len(train_idx) < 2:
        out[THEME_ID_COL] = "RT0001"
        out[THEME_NAME_COL] = "general safety records"
        return out, None

    n_clusters = max(2, min(int(settings.n_themes), len(train_idx)))
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=4096,
        random_state=settings.random_seed,
        n_init="auto",
        reassignment_ratio=0.01,
    )
    kmeans.fit(vectors[train_idx])
    labels = kmeans.predict(vectors)

    out[THEME_ID_COL] = [f"RT{int(label) + 1:04d}" for label in labels]

    names = _generate_theme_names(out, settings)
    out[THEME_NAME_COL] = out[THEME_ID_COL].map(names).fillna(out[THEME_ID_COL])
    return out, kmeans


def _generate_theme_names(df: pd.DataFrame, settings: Settings) -> dict[str, str]:
    names: dict[str, str] = {}
    # Sample text for label generation if the dataset is very large.
    work = df[[THEME_ID_COL, "retrieval_text"]].copy()
    if len(work) > settings.theme_sample_size_for_labels:
        work = work.sample(n=settings.theme_sample_size_for_labels, random_state=settings.random_seed)
    for theme_id, group in work.groupby(THEME_ID_COL):
        texts = group["retrieval_text"].fillna("").astype(str).tolist()
        if len(texts) < settings.theme_min_cluster_size:
            names[theme_id] = f"small theme {theme_id}"
            continue
        try:
            vectorizer = TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 3),
                min_df=2 if len(texts) >= 20 else 1,
                max_df=0.80,
                max_features=2000,
            )
            matrix = vectorizer.fit_transform(texts)
            weights = np.asarray(matrix.sum(axis=0)).ravel()
            terms = np.asarray(vectorizer.get_feature_names_out())
            order = weights.argsort()[::-1]
            selected: list[str] = []
            for idx in order:
                term = clean_text_value(terms[idx])
                if not term or term.isnumeric():
                    continue
                # Avoid theme labels that are only source-field labels.
                if term in {"title", "description", "comments", "activityduringincident", "immediateaction"}:
                    continue
                selected.append(term)
                if len(selected) >= settings.theme_top_terms:
                    break
            names[theme_id] = " / ".join(selected[:4]) if selected else f"theme {theme_id}"
        except Exception:
            names[theme_id] = f"theme {theme_id}"
    return names


def build_theme_profiles(df: pd.DataFrame, vectors: np.ndarray | None, settings: Settings) -> pd.DataFrame:
    """Create one evidence-backed profile row per risk theme."""
    rows = []
    for theme_id, group in df.groupby(THEME_ID_COL, dropna=False):
        group = group.copy()
        theme_name = clean_text_value(group[THEME_NAME_COL].dropna().astype(str).mode().iloc[0]) if THEME_NAME_COL in group.columns and not group[THEME_NAME_COL].dropna().empty else str(theme_id)
        source_mix = list_top_counts(group.get("source_role", group.get("source_type", pd.Series(dtype=str))), n=12)
        top_sites = list_top_counts(group.get("site", pd.Series(dtype=str)), n=10)
        top_departments = list_top_counts(group.get("department", pd.Series(dtype=str)), n=10)
        representative_event_ids = _representative_event_ids(group, vectors, settings)
        common_hazards = _top_phrases(group, roles=["hazard_identification", "near_miss", "injury", "severe_injury", "unsafe_observation"], settings=settings)
        common_corrective_actions = _top_phrases(group, roles=["corrective_action", "open_corrective_action", "overdue_corrective_action"], settings=settings)
        common_safe_practices = _top_phrases(group, roles=["safe_observation"], settings=settings)
        # Control gaps are derived from unsafe observations and hazard/near-miss
        # evidence. This is a keyphrase summary, not a hard-coded gap taxonomy.
        common_control_gaps = _top_phrases(group, roles=["unsafe_observation", "near_miss", "hazard_identification"], settings=settings)
        inspection_focus = _make_inspection_focus(common_hazards, common_control_gaps, common_corrective_actions, common_safe_practices)
        rows.append({
            "risk_theme_id": str(theme_id),
            "risk_theme_name": theme_name,
            "record_count": int(len(group)),
            "source_type_mix": compress_json_field(source_mix),
            "top_sites": compress_json_field(top_sites),
            "top_departments": compress_json_field(top_departments),
            "common_hazards": compress_json_field(common_hazards),
            "common_control_gaps": compress_json_field(common_control_gaps),
            "common_corrective_actions": compress_json_field(common_corrective_actions),
            "common_safe_practices": compress_json_field(common_safe_practices),
            "recommended_inspection_focus": compress_json_field(inspection_focus),
            "representative_event_ids": compress_json_field(representative_event_ids),
        })
    profiles = pd.DataFrame(rows).sort_values("record_count", ascending=False).reset_index(drop=True)
    ensure_dir(settings.theme_profiles_path().parent)
    profiles.to_pickle(settings.theme_profiles_path())
    profiles.to_csv(settings.theme_profiles_path().with_suffix(".csv"), index=False)
    return profiles


def _representative_event_ids(group: pd.DataFrame, vectors: np.ndarray | None, settings: Settings) -> list[str]:
    # Prefer source diversity and non-empty text. If vectors are available, select
    # records closest to the theme centroid inside each role group.
    candidate = group[group["retrieval_text"].fillna("").astype(str).str.len().ge(settings.min_text_chars)].copy()
    if candidate.empty:
        candidate = group.copy()
    selected: list[str] = []
    roles = ["severe_injury", "injury", "near_miss", "hazard_identification", "unsafe_observation", "safe_observation", "corrective_action", "open_corrective_action", "overdue_corrective_action"]
    if vectors is not None and "row_id" in candidate.columns:
        row_ids = candidate["row_id"].astype(int).to_numpy()
        centroid = vectors[row_ids].mean(axis=0)
        centroid = centroid / max(np.linalg.norm(centroid), 1e-12)
        candidate["_centroid_score"] = vectors[row_ids].dot(centroid)
    else:
        candidate["_centroid_score"] = candidate["text_length"].fillna(0)
    for role in roles:
        sub = candidate[candidate["source_role"].eq(role)].sort_values("_centroid_score", ascending=False)
        if not sub.empty:
            event_id = str(sub.iloc[0].get("event_id", ""))
            if event_id and event_id not in selected:
                selected.append(event_id)
        if len(selected) >= settings.theme_representative_events:
            break
    if len(selected) < settings.theme_representative_events:
        for _, row in candidate.sort_values("_centroid_score", ascending=False).head(settings.theme_representative_events * 2).iterrows():
            event_id = str(row.get("event_id", ""))
            if event_id and event_id not in selected:
                selected.append(event_id)
            if len(selected) >= settings.theme_representative_events:
                break
    return selected


def _top_phrases(group: pd.DataFrame, roles: Iterable[str], settings: Settings, top_n: int = 10) -> list[dict]:
    if "source_role" not in group.columns:
        return []
    sub = group[group["source_role"].isin(list(roles))].copy()
    sub = sub[sub["retrieval_text"].fillna("").astype(str).str.len().ge(settings.min_text_chars)]
    if sub.empty:
        return []
    texts = sub["retrieval_text"].fillna("").astype(str).tolist()
    try:
        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 3),
            min_df=2 if len(texts) >= 20 else 1,
            max_df=0.85,
            max_features=3000,
        )
        matrix = vectorizer.fit_transform(texts)
        weights = np.asarray(matrix.sum(axis=0)).ravel()
        terms = np.asarray(vectorizer.get_feature_names_out())
        order = weights.argsort()[::-1]
        out = []
        for idx in order:
            phrase = clean_text_value(terms[idx])
            if not phrase or phrase.isnumeric() or phrase in {"title", "description", "comments"}:
                continue
            supporting = _supporting_event_ids(sub, phrase, max_ids=5)
            out.append({"phrase": phrase, "score": float(weights[idx]), "supporting_event_ids": supporting})
            if len(out) >= top_n:
                break
        return out
    except Exception:
        return []


def _supporting_event_ids(df: pd.DataFrame, phrase: str, max_ids: int = 5) -> list[str]:
    pattern = re_escape_words(phrase)
    mask = df["retrieval_text"].fillna("").astype(str).str.contains(pattern, case=False, regex=True, na=False)
    return [str(x) for x in df.loc[mask, "event_id"].head(max_ids).tolist()]


def re_escape_words(phrase: str) -> str:
    parts = [p for p in phrase.split() if p]
    return r"\b" + r"\s+".join([__import__("re").escape(p) for p in parts]) + r"\b"


def _make_inspection_focus(common_hazards: list[dict], common_control_gaps: list[dict], common_actions: list[dict], common_safe: list[dict]) -> list[dict]:
    focus = []
    combined = []
    for source, label in [
        (common_hazards, "hazard evidence"),
        (common_control_gaps, "control-gap evidence"),
        (common_actions, "historical action evidence"),
        (common_safe, "safe-practice evidence"),
    ]:
        for item in source[:4]:
            phrase = clean_text_value(item.get("phrase", ""))
            if phrase:
                combined.append((phrase, label, item.get("supporting_event_ids", [])))
    seen = set()
    for phrase, label, ids in combined:
        short = phrase[:120]
        if short.lower() in seen:
            continue
        seen.add(short.lower())
        focus.append({
            "focus_area": f"Review {short}",
            "basis": label,
            "supporting_event_ids": ids,
        })
        if len(focus) >= 8:
            break
    return focus


def save_theme_model(kmeans: MiniBatchKMeans | None, settings: Settings) -> None:
    ensure_dir(settings.models_dir())
    if kmeans is not None:
        joblib.dump(kmeans, settings.models_dir() / "theme_kmeans.joblib")
        save_json({"theme_model_type": "MiniBatchKMeans", "n_clusters": int(kmeans.n_clusters)}, settings.models_dir() / "theme_model_metadata.json")
    else:
        save_json({"theme_model_type": "existing_theme_columns", "n_clusters": None}, settings.models_dir() / "theme_model_metadata.json")


def load_theme_model(settings: Settings):
    path = settings.models_dir() / "theme_kmeans.joblib"
    if path.exists():
        return joblib.load(path)
    return None
