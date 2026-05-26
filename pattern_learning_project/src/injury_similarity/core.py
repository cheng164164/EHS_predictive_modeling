"""Core functions for the injury-similarity machine-learning workflow.

This file contains the end-to-end logic for a retrieval-style ML task:

    Query record, usually a near miss or hazard
        -> convert early incident text to TF-IDF vector
        -> search historical injury reference records
        -> return top similar injury examples and a calibrated match band

Why this is retrieval instead of a standard classifier
------------------------------------------------------
The business question is not simply "will this become severe?". The current
CSV data does not contain PSIF labels, accepted/rejected PSIF decisions, or a
manual potential-severity label. Instead, the defensible task is:

    "Does this near miss / hazard resemble historical injury cases?"

Therefore the model is implemented as:

    TF-IDF vectorizer + cosine nearest-neighbor search

rather than a supervised classifier. Outcome fields such as severe_actual,
lost_time_any, restricted_time_any, fatality_any, inpatient_any, and
emergency_room_any are not used as text features. They are returned only as
context on the matched historical injury records.

Best-practice workflow
----------------------
1. Temporal validation
   Fit a validation-only model on older injury records and evaluate on newer
   held-out injury records. This estimates future retrieval behavior.

2. Final production fitting
   After reviewing validation results, fit the final production model on all
   eligible historical injury records so production has the largest reference
   library.

3. Prediction
   Load the saved final production model and score candidate near-miss/hazard
   records. Prediction does not refit the model.

Important leakage rule
----------------------
The input representation is built from early incident text only. Injury outcome
fields are allowed in outputs and validation metrics, but not as features used
by the vectorizer.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from injury_similarity import config

TRUE_VALUES = {"true", "1", "yes", "y", "t"}


def ensure_dir(path: Path) -> Path:
    """Create a directory if it does not already exist and return it.
    
    This small helper keeps all file-writing functions concise. It is safe to call
    even when the directory already exists.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def _json_default(value):
    """Convert pandas/numpy objects into JSON-serializable Python values.
    
    JSON cannot directly serialize numpy integers, numpy floats, pandas timestamps,
    or pandas missing values. This function is passed to json.dumps(default=...) so
    metrics and metadata files can be written without manual conversion in every
    caller.
    """
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if math.isnan(float(value)) else float(value)
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


def save_json(payload: dict, path: Path) -> None:
    """Write a dictionary to a pretty-printed JSON file.
    
    The parent folder is created automatically. This is used for validation metrics,
    thresholds, model metadata, prediction summaries, and workflow summaries.
    """
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")


def read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV file with a latin1 fallback.
    
    The prepared project outputs are expected to be UTF-8, but some raw exports can
    contain special characters. The fallback makes the ML script more tolerant. If
    the required processed file is missing, the error message tells the user to run
    the data-preparation pipeline first.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}. Run python src/run_data_prep.py first."
        )
    try:
        return pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, low_memory=False, encoding="latin1")


def clean_text_value(value: object) -> str:
    """Clean one text cell for ML use.
    
    The cleaning is intentionally simple and conservative:
    - missing values become an empty string
    - HTML tags are removed
    - newlines/tabs become spaces
    - repeated whitespace is collapsed
    - text placeholders such as "nan", "none", and "null" are removed
    
    This avoids changing the meaning of incident descriptions while making TF-IDF
    features less noisy.
    """
    if pd.isna(value):
        return ""
    text = str(value)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return "" if text.lower() in {"nan", "none", "null", "nat"} else text


def clean_text_series(series: pd.Series) -> pd.Series:
    """Apply clean_text_value to an entire pandas Series.
    
    Returns pandas string dtype so downstream string operations such as split() and
    str.cat() behave consistently.
    """
    return series.fillna("").map(clean_text_value).astype("string")


def to_bool(series: pd.Series) -> pd.Series:
    """Convert common boolean encodings to a normal bool Series.
    
    The data-prep outputs may contain real booleans or strings such as "True",
    "1", "yes", or blanks. This helper standardizes those values before filters
    and validation metrics use them.
    """
    if series.dtype == bool:
        return series.fillna(False).astype(bool)
    return series.astype("string").fillna("").str.strip().str.lower().isin(TRUE_VALUES)


def build_early_text(df: pd.DataFrame) -> pd.Series:
    """Build the early-lifecycle incident text used as the ML representation.
    
    Priority:
    1. Use the prepared ml_text_early column if it exists and has usable text.
    2. Otherwise combine configured early text fields from config.EARLY_TEXT_FIELDS.
    
    Why early text only:
    The goal is to score near misses/hazards using information that would be
    available early. Post-investigation fields and injury outcomes are excluded to
    avoid leakage.
    """
    if "ml_text_early" in df.columns:
        text = clean_text_series(df["ml_text_early"])
        if text.str.split().map(len).fillna(0).ge(3).any():
            return text
    pieces = [clean_text_series(df[c]) for c in config.EARLY_TEXT_FIELDS if c in df.columns]
    if not pieces:
        return pd.Series("", index=df.index, dtype="string")
    combined = pieces[0]
    for piece in pieces[1:]:
        combined = combined.str.cat(piece, sep=" ")
    return combined.map(clean_text_value).astype("string")


def add_working_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add normalized internal columns needed by training, validation, and prediction.
    
    Creates:
    - _ml_text: cleaned early incident text
    - _ml_word_count: simple text-quality filter
    - _incident_date_dt: parsed UTC date for temporal validation
    - standardized injury/severity booleans
    
    The leading underscore indicates internal helper columns that are not raw source
    fields.
    """
    out = df.copy()
    out["_ml_text"] = build_early_text(out)
    out["_ml_word_count"] = out["_ml_text"].str.split().map(lambda x: len(x) if isinstance(x, list) else 0)
    out["_incident_date_dt"] = pd.to_datetime(out["incident_date"], errors="coerce", utc=True) if "incident_date" in out.columns else pd.NaT
    out["injury_count"] = pd.to_numeric(out.get("injury_count", 0), errors="coerce").fillna(0).astype(int)
    for col in ["lost_time_any", "restricted_time_any", "fatality_any", "emergency_room_any", "inpatient_any", "severe_actual", "is_pattern_candidate"]:
        if col in out.columns:
            out[col] = to_bool(out[col])
    if "severe_actual" not in out.columns:
        out["severe_actual"] = False
    return out


def load_all_records() -> pd.DataFrame:
    """Load the main prepared incident/injury analytical table.
    
    Expected file:
        outputs/processed/incident_injury_all_records.csv
    
    This file is created by the existing data-preparation pipeline and includes all
    incident rows enriched with injury aggregates, location fields, list item names,
    and early text fields.
    """
    return read_csv(config.get_processed_dir() / "incident_injury_all_records.csv")


def load_prediction_candidates() -> pd.DataFrame:
    """Load records to score with the final production model.
    
    Preferred source:
        outputs/processed/pattern_learning_records.csv
    
    That file contains prepared near-miss and hazard-identification records. If it
    does not exist, the function falls back to incident_injury_all_records.csv and
    filters to pattern candidates when possible.
    
    The final candidate filter keeps non-injury records with usable text because the
    model is designed to ask: "which near misses/hazards resemble historical injury
    records?"
    """
    pattern_path = config.get_processed_dir() / "pattern_learning_records.csv"
    if pattern_path.exists():
        candidates = read_csv(pattern_path)
        source = "pattern_learning_records.csv"
    else:
        candidates = read_csv(config.get_processed_dir() / "incident_injury_all_records.csv")
        source = "incident_injury_all_records.csv"
    candidates = add_working_columns(candidates)
    if source == "incident_injury_all_records.csv" and "is_pattern_candidate" in candidates.columns:
        candidates = candidates[candidates["is_pattern_candidate"].fillna(False)].copy()
    candidates = candidates[candidates["injury_count"].eq(0) & candidates["_ml_word_count"].ge(3)].copy()
    candidates = candidates.reset_index(drop=True)
    candidates["_query_row_id"] = np.arange(len(candidates))
    candidates["_candidate_source"] = source
    return candidates


def make_reference_records(all_records: pd.DataFrame) -> pd.DataFrame:
    """Create the historical injury reference library.
    
    Reference records are the searchable historical examples. The function keeps
    records where:
    - injury_count > 0
    - early text has at least 3 words
    
    Both severe and non-severe injuries are retained. severe_actual is not the
    training target here; it is context returned with matches and used in validation
    metrics.
    """
    records = add_working_columns(all_records)
    ref = records[records["injury_count"].gt(0) & records["_ml_word_count"].ge(3)].copy()
    ref = ref.reset_index(drop=True)
    if ref.empty:
        raise ValueError("No historical injury records with injury_count > 0 and usable early text were found.")
    ref["_reference_row_id"] = np.arange(len(ref))
    return ref


def fit_vectorizer(texts: Iterable[str]) -> TfidfVectorizer:
    """Fit the TF-IDF vectorizer on reference texts.
    
    This is the model's text embedding step. In validation it is fit only on the
    older train/reference split. In final production fitting it is fit on all
    historical injury reference records.
    
    Tunable parameters come from config.py:
    - TFIDF_MIN_DF removes rare tokens/phrases
    - TFIDF_MAX_DF removes very common tokens/phrases
    - TFIDF_MAX_FEATURES caps vocabulary size
    - TFIDF_NGRAM_RANGE controls single words vs short phrases
    
    The fallback block is for very small datasets where the configured parameters
    would remove all vocabulary terms.
    """
    texts = list(texts)
    min_df = config.TFIDF_MIN_DF if len(texts) >= 10 else 1
    try:
        vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            stop_words="english",
            ngram_range=config.TFIDF_NGRAM_RANGE,
            min_df=min_df,
            max_df=config.TFIDF_MAX_DF,
            max_features=config.TFIDF_MAX_FEATURES,
            sublinear_tf=True,
            norm="l2",
        )
        vectorizer.fit(texts)
        return vectorizer
    except ValueError:
        vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            stop_words="english",
            ngram_range=(1, 1),
            min_df=1,
            max_df=1.0,
            max_features=config.TFIDF_MAX_FEATURES,
            sublinear_tf=True,
            norm="l2",
        )
        vectorizer.fit(texts)
        return vectorizer


def fit_nn(matrix, top_k: int) -> NearestNeighbors:
    """Fit a cosine nearest-neighbor index on reference vectors.
    
    The TF-IDF matrix is already L2-normalized, so cosine distance is appropriate.
    The brute-force algorithm is simple and reliable for sparse TF-IDF matrices.
    For much larger datasets, this could later be replaced with an approximate
    nearest-neighbor index.
    """
    nn = NearestNeighbors(n_neighbors=max(1, min(top_k, matrix.shape[0])), metric="cosine", algorithm="brute")
    nn.fit(matrix)
    return nn


def _nonnull(value: object) -> str | None:
    """Return a stripped string value, or None for missing/blank values.
    
    Used by background-pair filtering so missing metadata does not create false
    "different site" or "different category" evidence.
    """
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text if text else None


def _background_pair_ok(records: pd.DataFrame, i: int, j: int) -> bool:
    """Decide whether two records are acceptable as a background/noise pair.
    
    Background pairs are used to estimate ordinary or weak similarity. To reduce
    contamination from genuinely related repeated cases, this function prefers pairs
    that differ on available metadata such as category, site, department, business
    unit, or incident date gap.
    
    The rule is intentionally not too strict. If too few metadata checks are
    available, the pair is accepted. If enough checks exist, at least two must
    suggest the records are different.
    """
    checks = []
    for col in ["incident_category_name", "site_name_filled", "department_name_filled", "business_unit_name_filled"]:
        if col in records.columns:
            a = _nonnull(records.iloc[i].get(col))
            b = _nonnull(records.iloc[j].get(col))
            if a is not None and b is not None:
                checks.append(a != b)
    if "_incident_date_dt" in records.columns:
        a_date = records.iloc[i].get("_incident_date_dt")
        b_date = records.iloc[j].get("_incident_date_dt")
        if pd.notna(a_date) and pd.notna(b_date):
            checks.append(abs((a_date - b_date).days) >= config.BACKGROUND_MIN_DATE_GAP_DAYS)
    return True if len(checks) < 2 else sum(checks) >= 2


def sample_background_scores(matrix, records: pd.DataFrame, sample_size: int = config.BACKGROUND_SAMPLE_SIZE) -> np.ndarray:
    """Sample metadata-constrained pairwise similarity scores.
    
    Purpose:
    Estimate the "background" similarity distribution: scores that can happen
    from generic safety language, common workplace vocabulary, or weak overlap.
    
    Why not assume random pairs are unrelated?
    EHS data contains repeated patterns, so a fully random pair can accidentally be
    a real repeated hazard. The helper _background_pair_ok tries to reduce this by
    prefering pairs from different categories/sites/departments/business units or
    with enough date separation.
    
    Output:
    A numpy array of cosine similarities. These scores are used by
    calibrate_thresholds() to set weak/possible/strong match cutoffs.
    """
    n = matrix.shape[0]
    if n < 2:
        return np.array([], dtype=float)
    rng = np.random.default_rng(config.RANDOM_SEED)
    target = int(min(sample_size, max(1, n * (n - 1) // 2)))
    max_trials = max(target * 30, 1000)
    left, right = [], []
    trials = 0
    while len(left) < target and trials < max_trials:
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n))
        trials += 1
        if i != j and _background_pair_ok(records, i, j):
            left.append(i)
            right.append(j)
    while len(left) < target:
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n))
        if i != j:
            left.append(i)
            right.append(j)
    scores = np.asarray(matrix[np.asarray(left)].multiply(matrix[np.asarray(right)]).sum(axis=1)).ravel()
    return scores[np.isfinite(scores)].astype(float)


def leave_one_out_top1_scores(matrix) -> np.ndarray:
    """Compute leave-one-out nearest-neighbor top-1 scores.
    
    For each sampled injury reference record:
    1. Search the reference library.
    2. Ignore the record itself, which would have similarity 1.0.
    3. Keep the best non-self similarity score.
    
    Purpose:
    Estimate what meaningful injury-to-injury similarity usually looks like in this
    dataset. This is a calibration diagnostic, not an unbiased final test. The true
    validation test is temporal holdout.
    """
    n = matrix.shape[0]
    if n < 2:
        return np.array([], dtype=float)
    rng = np.random.default_rng(config.RANDOM_SEED)
    query_idx = np.arange(n) if n <= config.MAX_LOO_RECORDS else np.sort(rng.choice(n, size=config.MAX_LOO_RECORDS, replace=False))
    nn = NearestNeighbors(n_neighbors=min(10, n), metric="cosine", algorithm="brute")
    nn.fit(matrix)
    distances, indices = nn.kneighbors(matrix[query_idx], n_neighbors=min(10, n))
    scores = []
    for pos, qid in enumerate(query_idx):
        for dist, idx in zip(distances[pos], indices[pos]):
            if int(idx) != int(qid):
                scores.append(float(1.0 - dist))
                break
    return np.asarray(scores, dtype=float)


def q(scores: np.ndarray, quantile: float, default: float) -> float:
    """Return a quantile from a score array, or a default when the array is empty.
    
    This keeps threshold calibration robust on tiny datasets where background or
    leave-one-out samples may be unavailable.
    """
    return default if scores.size == 0 else float(np.quantile(scores, quantile))


def calibrate_thresholds(matrix, records: pd.DataFrame) -> dict:
    """Convert similarity score distributions into operational match thresholds.
    
    Inputs:
    - matrix: TF-IDF vectors for the current reference library
    - records: metadata for the same reference records
    
    Logic:
    1. Estimate background/noise similarity using metadata-constrained pairs.
    2. Estimate meaningful injury-to-injury similarity using leave-one-out top-1.
    3. Set three thresholds:
       - weak_match: above ordinary background similarity
       - possible_match: above stronger background and low-end meaningful scores
       - strong_match: above very high background and median meaningful scores
    
    The exact thresholds are data-driven, so they adapt when the dataset grows or
    writing style changes.
    """
    # Background scores represent weak/random/generic overlap. Higher background
    # percentiles are used as a guardrail so common EHS wording does not create
    # too many false positive matches.
    background = sample_background_scores(matrix, records)

    # Leave-one-out scores represent how strongly real injury records tend to
    # match other injury records in the same historical library. This gives the
    # signal side of calibration.
    loo = leave_one_out_top1_scores(matrix)

    # Background percentiles: increasing strictness.
    bg90 = q(background, 0.90, 0.10)
    bg95 = q(background, 0.95, 0.15)
    bg99 = q(background, 0.99, 0.25)

    # Leave-one-out percentiles: low, lower-middle, and median meaningful-match
    # behavior. Defaults depend partly on background scores so tiny datasets do
    # not produce thresholds that are too permissive.
    loo10 = q(loo, 0.10, max(bg95, 0.20))
    loo25 = q(loo, 0.25, max(bg99, 0.30))
    loo50 = q(loo, 0.50, max(bg99, 0.40))

    # Weak match: must beat most background overlap, but stays within [0, 0.98].
    weak = min(max(max(0.05, bg90), 0.0), 0.98)

    # Possible match: must beat stronger background overlap and low-end
    # meaningful injury-to-injury similarity. Also enforce a small margin above
    # weak so the bands are ordered.
    possible = min(max(max(bg95, loo10), weak + 0.02), 0.99)

    # Strong match: must beat very high background overlap and typical
    # meaningful injury-to-injury similarity. Also enforce a small margin above
    # possible.
    strong = min(max(max(bg99, loo50), possible + 0.02), 1.0)
    return {
        "weak_match_threshold": float(weak),
        "possible_match_threshold": float(possible),
        "strong_match_threshold": float(strong),
        "background_score_count": int(background.size),
        "leave_one_out_score_count": int(loo.size),
        "background_p50": q(background, 0.50, np.nan),
        "background_p90": bg90,
        "background_p95": bg95,
        "background_p99": bg99,
        "leave_one_out_top1_p10": loo10,
        "leave_one_out_top1_p25": loo25,
        "leave_one_out_top1_p50": loo50,
        "leave_one_out_top1_p75": q(loo, 0.75, np.nan),
        "leave_one_out_top1_p90": q(loo, 0.90, np.nan),
        "calibration_note": "Validation thresholds use only the older train split; final thresholds use all historical injury records.",
    }


def match_band(score: float, thresholds: dict) -> str:
    """Assign a readable match band from a numeric similarity score.
    
    Bands are ordered from strongest to weakest:
    - strong_match
    - possible_match
    - weak_match
    - no_match
    
    These labels are easier for EHS reviewers and downstream dashboards than raw
    cosine scores alone.
    """
    if score is None or pd.isna(score):
        return "no_match"
    if score >= thresholds["strong_match_threshold"]:
        return "strong_match"
    if score >= thresholds["possible_match_threshold"]:
        return "possible_match"
    if score >= thresholds["weak_match_threshold"]:
        return "weak_match"
    return "no_match"


def preview(text: object, n: int = 300) -> str:
    """Return a short text preview for CSV outputs.
    
    This makes prediction and validation files reviewable without writing full long
    incident descriptions into every output row.
    """
    text = clean_text_value(text)
    return text if len(text) <= n else text[: n - 3].rstrip() + "..."


def retrieve(query_records: pd.DataFrame, query_matrix, reference_records: pd.DataFrame, nn: NearestNeighbors, thresholds: dict, top_k: int, include_no_match_rows: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retrieve similar historical injury records for each query record.
    
    Inputs:
    - query_records: records being scored, such as holdout injuries or near misses
    - query_matrix: TF-IDF vectors for those query records
    - reference_records: historical injury library
    - nn: fitted nearest-neighbor model over reference_records
    - thresholds: calibrated match thresholds
    - top_k: number of neighbors to return
    - include_no_match_rows: if True, return rows even when top score is no_match
    
    Outputs:
    - query summary: one row per query with top-1 score and match band
    - top matches: one row per returned historical match
    
    In production prediction, no_match queries remain in the summary file but do not
    produce top-match rows unless include_no_match_rows is True.
    """
    if len(query_records) == 0:
        return pd.DataFrame(), pd.DataFrame()
    n_neighbors = min(top_k, len(reference_records))
    distances, indices = nn.kneighbors(query_matrix, n_neighbors=n_neighbors)
    summary_rows = []
    match_rows = []
    for qpos, qrow in query_records.reset_index(drop=True).iterrows():
        sims = 1.0 - distances[qpos]
        top1 = float(sims[0]) if len(sims) else np.nan
        band = match_band(top1, thresholds)
        qid = qrow.get("incident_id", qrow.get("_query_row_id", qpos))
        summary_rows.append({
            "query_row_id": int(qrow.get("_query_row_id", qpos)),
            "query_incident_id": qid,
            "query_incident_number": qrow.get("incident_number", pd.NA),
            "query_incident_date": qrow.get("incident_date", pd.NA),
            "query_incident_category_name": qrow.get("incident_category_name", pd.NA),
            "query_site_name_filled": qrow.get("site_name_filled", pd.NA),
            "query_department_name_filled": qrow.get("department_name_filled", pd.NA),
            "query_title": qrow.get("title", pd.NA),
            "top1_similarity_score": top1,
            "similarity_band": band,
            "returned_match_count": 0 if band == "no_match" else int(n_neighbors),
            "query_text_preview": preview(qrow.get("_ml_text", "")),
        })
        if band == "no_match" and not include_no_match_rows:
            continue
        for rank, (score, ridx) in enumerate(zip(sims, indices[qpos]), start=1):
            ref = reference_records.iloc[int(ridx)]
            score = float(score)
            if not include_no_match_rows and score < thresholds["weak_match_threshold"]:
                continue
            match_rows.append({
                "query_row_id": int(qrow.get("_query_row_id", qpos)),
                "query_incident_id": qid,
                "query_incident_number": qrow.get("incident_number", pd.NA),
                "query_incident_date": qrow.get("incident_date", pd.NA),
                "query_incident_category_name": qrow.get("incident_category_name", pd.NA),
                "query_site_name_filled": qrow.get("site_name_filled", pd.NA),
                "query_department_name_filled": qrow.get("department_name_filled", pd.NA),
                "query_severe_actual": qrow.get("severe_actual", pd.NA),
                "query_title": qrow.get("title", pd.NA),
                "rank": rank,
                "similarity_score": score,
                "similarity_band": match_band(score, thresholds),
                "matched_reference_row_id": int(ref.get("_reference_row_id", ridx)),
                "matched_reference_incident_id": ref.get("incident_id", pd.NA),
                "matched_reference_incident_number": ref.get("incident_number", pd.NA),
                "matched_reference_incident_date": ref.get("incident_date", pd.NA),
                "matched_reference_incident_category_name": ref.get("incident_category_name", pd.NA),
                "matched_reference_site_name_filled": ref.get("site_name_filled", pd.NA),
                "matched_reference_department_name_filled": ref.get("department_name_filled", pd.NA),
                "matched_reference_title": ref.get("title", pd.NA),
                "matched_reference_injury_count": ref.get("injury_count", pd.NA),
                "matched_reference_severe_actual": ref.get("severe_actual", pd.NA),
                "matched_reference_lost_time_any": ref.get("lost_time_any", pd.NA),
                "matched_reference_restricted_time_any": ref.get("restricted_time_any", pd.NA),
                "matched_reference_fatality_any": ref.get("fatality_any", pd.NA),
                "matched_reference_inpatient_any": ref.get("inpatient_any", pd.NA),
                "matched_reference_emergency_room_any": ref.get("emergency_room_any", pd.NA),
                "matched_reference_text_preview": preview(ref.get("_ml_text", ref.get("ml_text_early", ""))),
            })
    return pd.DataFrame(summary_rows), pd.DataFrame(match_rows)


def save_model(model_dir: Path, vectorizer, nn, matrix, ref: pd.DataFrame, thresholds: dict, metadata: dict) -> None:
    """Persist all artifacts needed to reuse a fitted retrieval model.
    
    Saved artifacts:
    - tfidf_vectorizer.joblib: text-to-vector transformer
    - nearest_neighbors.joblib: fitted cosine nearest-neighbor index
    - reference_matrix.joblib: sparse TF-IDF matrix of reference records
    - reference_records.csv: metadata/context for matched historical injuries
    - thresholds.json: weak/possible/strong thresholds
    - metadata.json: human-readable model information
    """
    ensure_dir(model_dir)
    joblib.dump(vectorizer, model_dir / "tfidf_vectorizer.joblib")
    joblib.dump(nn, model_dir / "nearest_neighbors.joblib")
    joblib.dump(matrix, model_dir / "reference_matrix.joblib")
    out = ref.copy()
    if "_incident_date_dt" in out.columns:
        out["_incident_date_dt"] = out["_incident_date_dt"].astype(str)
    out.to_csv(model_dir / "reference_records.csv", index=False)
    save_json(thresholds, model_dir / "thresholds.json")
    save_json(metadata, model_dir / "metadata.json")


def load_model(model_dir: Path | None = None) -> dict:
    """Load a saved model artifact bundle.
    
    Prediction uses this function to load the final production model from
    config.get_model_dir(). It does not refit any model components.
    """
    model_dir = model_dir or config.get_model_dir()
    return {
        "vectorizer": joblib.load(model_dir / "tfidf_vectorizer.joblib"),
        "nearest_neighbors": joblib.load(model_dir / "nearest_neighbors.joblib"),
        "reference_matrix": joblib.load(model_dir / "reference_matrix.joblib"),
        "reference_records": pd.read_csv(model_dir / "reference_records.csv", low_memory=False),
        "thresholds": json.loads((model_dir / "thresholds.json").read_text(encoding="utf-8")),
        "metadata": json.loads((model_dir / "metadata.json").read_text(encoding="utf-8")),
    }


def make_temporal_split(ref: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Split injury reference records into older train and newer holdout sets.
    
    This is the preferred validation split because it simulates production:
    future/newer records are compared against past/older injury history.
    
    If there are not enough dated records, the function falls back to a reproducible
    random split. That fallback is less ideal and is clearly marked in split_info.
    """
    dated = ref[ref["_incident_date_dt"].notna()].sort_values("_incident_date_dt").copy()
    info = {"train_fraction_requested": config.TEMPORAL_TRAIN_FRACTION}
    if len(dated) >= config.MIN_TEMPORAL_RECORDS:
        split = int(np.floor(len(dated) * config.TEMPORAL_TRAIN_FRACTION))
        split = max(1, min(split, len(dated) - config.MIN_HOLDOUT_RECORDS))
        if split > 0 and len(dated) - split >= config.MIN_HOLDOUT_RECORDS:
            train = dated.iloc[:split].copy().reset_index(drop=True)
            holdout = dated.iloc[split:].copy().reset_index(drop=True)
            info.update({
                "split_type": "temporal_by_incident_date",
                "train_start_date": str(train["_incident_date_dt"].min()),
                "train_end_date": str(train["_incident_date_dt"].max()),
                "holdout_start_date": str(holdout["_incident_date_dt"].min()),
                "holdout_end_date": str(holdout["_incident_date_dt"].max()),
            })
            return train, holdout, info
    rng = np.random.default_rng(config.RANDOM_SEED)
    order = rng.permutation(len(ref))
    split = int(np.floor(len(ref) * config.TEMPORAL_TRAIN_FRACTION))
    split = max(1, min(split, len(ref) - 1))
    train = ref.iloc[order[:split]].copy().reset_index(drop=True)
    holdout = ref.iloc[order[split:]].copy().reset_index(drop=True)
    info.update({"split_type": "fixed_random_fallback", "note": "Not enough dated records for temporal split."})
    return train, holdout, info


def equality_rate(df: pd.DataFrame, left: str, right: str) -> float | None:
    """Calculate the rate at which two columns have equal nonblank values.
    
    Used for validation diagnostics such as same-category, same-site, and
    same-department top-1 match rates. Missing/blank pairs are excluded from the
    denominator.
    """
    if df.empty or left not in df.columns or right not in df.columns:
        return None
    l = df[left].fillna("").astype(str)
    r = df[right].fillna("").astype(str)
    valid = l.ne("") & r.ne("")
    return None if valid.sum() == 0 else float(l[valid].eq(r[valid]).mean())


def bool_rate(series: pd.Series) -> float | None:
    """Return the mean of a boolean Series, or None if no values exist.
    
    Used for validation metrics such as same-severity rate.
    """
    series = series.dropna()
    return None if series.empty else float(series.mean())


def validation_metrics(train: pd.DataFrame, holdout: pd.DataFrame, summary: pd.DataFrame, matches: pd.DataFrame, thresholds: dict, split_info: dict) -> dict:
    """Compute retrieval-quality metrics for temporal holdout validation.
    
    These are not classifier accuracy metrics. They describe whether held-out injury
    queries retrieve plausible historical injury examples. Metrics include score
    distribution, match-band counts, same-category/site/department rates, and
    severity-context diagnostics.
    """
    top1 = matches[matches["rank"].eq(1)].copy() if not matches.empty else pd.DataFrame()
    scores = pd.to_numeric(summary.get("top1_similarity_score", pd.Series(dtype=float)), errors="coerce")
    metrics = {
        "n_train_reference_records": int(len(train)),
        "n_holdout_query_records": int(len(holdout)),
        "split_info": split_info,
        "thresholds_used_for_holdout": thresholds,
        "top1_similarity_mean": float(scores.mean()) if not scores.empty else None,
        "top1_similarity_median": float(scores.median()) if not scores.empty else None,
        "top1_similarity_p10": float(scores.quantile(0.10)) if not scores.empty else None,
        "top1_similarity_p25": float(scores.quantile(0.25)) if not scores.empty else None,
        "top1_similarity_p75": float(scores.quantile(0.75)) if not scores.empty else None,
        "top1_similarity_p90": float(scores.quantile(0.90)) if not scores.empty else None,
        "same_category_top1_rate": equality_rate(top1, "query_incident_category_name", "matched_reference_incident_category_name"),
        "same_site_top1_rate": equality_rate(top1, "query_site_name_filled", "matched_reference_site_name_filled"),
        "same_department_top1_rate": equality_rate(top1, "query_department_name_filled", "matched_reference_department_name_filled"),
    }
    if not summary.empty:
        metrics["match_band_counts"] = {str(k): int(v) for k, v in summary["similarity_band"].value_counts(dropna=False).to_dict().items()}
        metrics["strong_or_possible_match_rate"] = float(summary["similarity_band"].isin(["strong_match", "possible_match"]).mean())
        metrics["weak_or_better_match_rate"] = float(summary["similarity_band"].ne("no_match").mean())
    if not top1.empty and "query_severe_actual" in top1.columns:
        qsev = to_bool(top1["query_severe_actual"])
        rsev = to_bool(top1["matched_reference_severe_actual"])
        metrics["same_severity_top1_rate"] = bool_rate(qsev.eq(rsev))
        severe_top1 = top1[qsev]
        if not severe_top1.empty:
            metrics["n_severe_holdout_queries_with_top1"] = int(len(severe_top1))
            metrics["severe_query_top1_is_severe_reference_rate"] = bool_rate(to_bool(severe_top1["matched_reference_severe_actual"]))
        if not matches.empty:
            work = matches.copy()
            work["query_severe_bool"] = to_bool(work["query_severe_actual"])
            work["ref_severe_bool"] = to_bool(work["matched_reference_severe_actual"])
            severe_queries = work[work["query_severe_bool"]]
            if not severe_queries.empty:
                metrics["severe_query_topk_contains_severe_reference_rate"] = float(severe_queries.groupby("query_row_id")["ref_severe_bool"].max().mean())
    return metrics


def run_temporal_validation(top_k: int = config.DEFAULT_TOP_K) -> dict:
    """Run the validation-only training/testing workflow.
    
    Steps:
    1. Load all prepared incident/injury records.
    2. Build the historical injury reference table.
    3. Split records by time into older train/reference and newer holdout queries.
    4. Fit vectorizer and NN index only on older train/reference records.
    5. Transform held-out records using the train-fitted vectorizer.
    6. Retrieve matches and compute validation metrics.
    7. Save validation outputs under outputs/ml/injury_similarity/validation.
    
    This function intentionally does not create the final production model.
    """
    validation_dir = ensure_dir(config.get_validation_dir())
    ref = make_reference_records(load_all_records())
    train, holdout, split_info = make_temporal_split(ref)
    # Fit only on the older train/reference split. This is the key anti-leakage
    # step: the held-out newer records must not influence the vocabulary, IDF
    # weights, nearest-neighbor library, or thresholds.
    vectorizer = fit_vectorizer(train["_ml_text"].tolist())
    train_matrix = vectorizer.transform(train["_ml_text"].tolist())

    # Transform holdout records with the train-fitted vectorizer. Unknown words
    # in the newer holdout records are ignored, which mirrors future production
    # behavior.
    holdout_matrix = vectorizer.transform(holdout["_ml_text"].tolist())

    # Build the searchable library using only older train/reference records.
    nn = fit_nn(train_matrix, top_k=max(top_k, config.DEFAULT_TOP_K))

    # Calibrate thresholds using only train/reference records.
    thresholds = calibrate_thresholds(train_matrix, train)
    summary, matches = retrieve(holdout, holdout_matrix, train, nn, thresholds, top_k, include_no_match_rows=True)
    metrics = validation_metrics(train, holdout, summary, matches, thresholds, split_info)
    summary.to_csv(validation_dir / "temporal_holdout_query_summary.csv", index=False)
    matches.to_csv(validation_dir / "temporal_holdout_top_matches.csv", index=False)
    save_json(metrics, validation_dir / "temporal_validation_metrics.json")
    save_json(thresholds, validation_dir / "thresholds_from_train_split.json")
    save_model(
        validation_dir / "model_train_split",
        vectorizer,
        nn,
        train_matrix,
        train,
        thresholds,
        {
            "model_purpose": "validation_only_train_split_model",
            "important_note": "Fit only on older train/reference records. Not the final production model.",
            "top_k": top_k,
            "n_train_reference_records": int(len(train)),
            "n_holdout_query_records": int(len(holdout)),
            "split_info": split_info,
        },
    )
    return metrics


def train_final_model(top_k: int = config.DEFAULT_TOP_K) -> dict:
    """Fit the final production model on all eligible historical injury records.
    
    This should be done after reviewing temporal validation results. Unlike
    validation, this final fit uses all available historical injury reference records
    so production prediction has the most complete injury history.
    """
    model_dir = ensure_dir(config.get_model_dir())
    ref = make_reference_records(load_all_records())
    # Final production fitting intentionally uses all eligible historical injury
    # records. This is different from validation. Once validation is complete,
    # the production model should have the largest available reference library.
    vectorizer = fit_vectorizer(ref["_ml_text"].tolist())
    matrix = vectorizer.transform(ref["_ml_text"].tolist())
    nn = fit_nn(matrix, top_k=max(top_k, config.DEFAULT_TOP_K))

    # Final thresholds are calibrated on the full production reference library.
    thresholds = calibrate_thresholds(matrix, ref)
    severe_count = int(ref["severe_actual"].fillna(False).astype(bool).sum()) if "severe_actual" in ref.columns else 0
    metadata = {
        "model_purpose": "final_production_model_all_historical_injury_records",
        "important_note": "Temporal validation is saved separately. This final model is fit after validation on all historical injury records.",
        "algorithm": "TF-IDF vectorizer plus cosine nearest-neighbor retrieval",
        "top_k_default": top_k,
        "feature_text": "early incident text only",
        "outcome_fields_usage": "injury/severity fields are context on matched reference records only; they are not vectorizer features",
        "n_reference_records": int(len(ref)),
        "n_severe_actual_reference_records": severe_count,
        "n_non_severe_injury_reference_records": int(len(ref) - severe_count),
        "n_features": int(matrix.shape[1]),
    }
    save_model(model_dir, vectorizer, nn, matrix, ref, thresholds, metadata)
    save_json({"metadata": metadata, "thresholds": thresholds}, model_dir / "training_summary.json")
    return metadata


def predict_injury_similarity(top_k: int = config.DEFAULT_TOP_K) -> dict:
    """Score near-miss/hazard candidates using the saved final production model.
    
    This function loads artifacts from final_model and does not refit the vectorizer
    or nearest-neighbor index. It writes a query-level summary and a top-match file
    under outputs/ml/injury_similarity/predictions.
    """
    prediction_dir = ensure_dir(config.get_prediction_dir())
    artifacts = load_model(config.get_model_dir())
    candidates = load_prediction_candidates()
    if candidates.empty:
        summary = {"n_candidate_records": 0, "n_queries_scored": 0, "message": "No candidate records with usable text were found."}
        save_json(summary, prediction_dir / "prediction_run_summary.json")
        pd.DataFrame().to_csv(prediction_dir / "injury_similarity_query_summary.csv", index=False)
        pd.DataFrame().to_csv(prediction_dir / "injury_similarity_top_matches.csv", index=False)
        return summary
    qmatrix = artifacts["vectorizer"].transform(candidates["_ml_text"].tolist())
    summary, matches = retrieve(candidates, qmatrix, artifacts["reference_records"], artifacts["nearest_neighbors"], artifacts["thresholds"], top_k, include_no_match_rows=False)
    summary.to_csv(prediction_dir / "injury_similarity_query_summary.csv", index=False)
    matches.to_csv(prediction_dir / "injury_similarity_top_matches.csv", index=False)
    band_counts = summary["similarity_band"].value_counts(dropna=False).to_dict() if not summary.empty else {}
    run_summary = {
        "n_candidate_records": int(len(candidates)),
        "n_queries_scored": int(len(summary)),
        "n_top_match_rows_returned": int(len(matches)),
        "candidate_source": str(candidates["_candidate_source"].iloc[0]) if len(candidates) else None,
        "top_k": top_k,
        "match_band_counts": {str(k): int(v) for k, v in band_counts.items()},
        "strong_or_possible_match_count": int(summary["similarity_band"].isin(["strong_match", "possible_match"]).sum()) if not summary.empty else 0,
        "weak_or_better_match_count": int(summary["similarity_band"].ne("no_match").sum()) if not summary.empty else 0,
        "important_note": "No-match queries remain in query_summary. Top-match rows are returned only for weak_match or better.",
    }
    save_json(run_summary, prediction_dir / "prediction_run_summary.json")
    return run_summary


def run_training_workflow(include_predictions: bool = True) -> dict:
    """Run the full recommended workflow in the correct order.
    
    Order:
    1. temporal validation
    2. final model fitting on all historical injury records
    3. optional batch prediction using the final saved model
    
    The workflow summary records where major outputs are saved.
    """
    ml_dir = ensure_dir(config.get_ml_dir())
    validation = run_temporal_validation(config.DEFAULT_TOP_K)
    final_model = train_final_model(config.DEFAULT_TOP_K)
    prediction = predict_injury_similarity(config.DEFAULT_TOP_K) if include_predictions else None
    workflow_summary = {
        "workflow_order": [
            "1_temporal_validation_train_on_older_test_on_newer",
            "2_final_model_fit_on_all_historical_injury_records",
            "3_optional_batch_prediction_using_final_model",
        ],
        "validation_metrics_file": str(config.get_validation_dir() / "temporal_validation_metrics.json"),
        "final_model_dir": str(config.get_model_dir()),
        "prediction_summary_file": str(config.get_prediction_dir() / "prediction_run_summary.json") if include_predictions else None,
        "validation_metrics_snapshot": validation,
        "final_model_metadata_snapshot": final_model,
        "prediction_summary_snapshot": prediction,
    }
    save_json(workflow_summary, ml_dir / "workflow_summary.json")
    return workflow_summary
