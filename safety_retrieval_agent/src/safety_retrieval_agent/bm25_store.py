"""Local BM25 keyword retrieval utilities.

This module implements a lightweight Okapi BM25 index with scikit-learn's
CountVectorizer plus sparse matrices. It is designed to complement FAISS semantic
retrieval without requiring Azure AI Search or any external search service.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

from .config import Settings
from .utils import clean_text_value, ensure_dir, load_json, save_json


@dataclass
class BM25Result:
    row_id: int
    score: float
    rank: int


@dataclass
class BM25Store:
    vectorizer: CountVectorizer
    matrix_csc: sparse.csc_matrix
    doc_len: np.ndarray
    avg_doc_len: float
    idf: np.ndarray
    metadata: dict


def _bm25_token_pattern() -> str:
    # Keeps common EHS acronyms/short terms such as PPE and LOTO. Single-letter
    # tokens are ignored to reduce noise.
    return r"(?u)\b[\w\-/]{2,}\b"


def build_bm25_store(records: pd.DataFrame, settings: Settings) -> BM25Store:
    """Fit a BM25 keyword index over the full knowledge base.

    The returned store covers all rows in ``records``. Source-specific searches
    are handled by passing subset row IDs at query time, so we do not need to fit
    one BM25 vectorizer per subset.
    """
    texts = records["retrieval_text"].fillna("").astype(str).map(clean_text_value).tolist()
    max_features = int(getattr(settings, "bm25_max_features", 250000))
    min_df = int(getattr(settings, "bm25_min_df", 2))
    max_df = float(getattr(settings, "bm25_max_df", 0.98))
    ngram_range = tuple(getattr(settings, "bm25_ngram_range", (1, 2)))

    vectorizer = CountVectorizer(
        lowercase=True,
        strip_accents="unicode",
        token_pattern=_bm25_token_pattern(),
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
    )
    try:
        matrix_csr = vectorizer.fit_transform(texts).astype(np.float32)
    except ValueError:
        # Small smoke-test samples can be too sparse for min_df/max_df. Fall
        # back to permissive unigram/bigram settings so the script can still run.
        vectorizer = CountVectorizer(
            lowercase=True,
            strip_accents="unicode",
            token_pattern=_bm25_token_pattern(),
            ngram_range=ngram_range,
            min_df=1,
            max_df=1.0,
            max_features=max_features,
        )
        matrix_csr = vectorizer.fit_transform(texts).astype(np.float32)
        min_df = 1
        max_df = 1.0
    matrix_csc = matrix_csr.tocsc()
    doc_len = np.asarray(matrix_csr.sum(axis=1)).ravel().astype(np.float32)
    avg_doc_len = float(doc_len.mean()) if len(doc_len) else 0.0

    n_docs = int(matrix_csr.shape[0])
    # Number of documents containing each term.
    df = np.diff(matrix_csc.indptr).astype(np.float32)
    # Standard Okapi BM25 IDF with +1 inside log to keep values positive.
    idf = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)

    metadata = {
        "bm25_algorithm": "okapi_bm25",
        "row_count": n_docs,
        "vocabulary_size": int(matrix_csr.shape[1]),
        "bm25_k1": float(getattr(settings, "bm25_k1", 1.5)),
        "bm25_b": float(getattr(settings, "bm25_b", 0.75)),
        "bm25_min_df": min_df,
        "bm25_max_df": max_df,
        "bm25_max_features": max_features,
        "bm25_ngram_range": list(ngram_range),
        "avg_doc_len": avg_doc_len,
    }
    return BM25Store(vectorizer=vectorizer, matrix_csc=matrix_csc, doc_len=doc_len, avg_doc_len=avg_doc_len, idf=idf, metadata=metadata)


def save_bm25_store(store: BM25Store, output_dir: Path) -> None:
    ensure_dir(output_dir)
    joblib.dump(store.vectorizer, output_dir / "bm25_vectorizer.joblib")
    joblib.dump(store.matrix_csc, output_dir / "bm25_matrix_csc.joblib")
    np.save(output_dir / "bm25_doc_len.npy", store.doc_len.astype(np.float32))
    np.save(output_dir / "bm25_idf.npy", store.idf.astype(np.float32))
    save_json(store.metadata, output_dir / "bm25_metadata.json")


def load_bm25_store(output_dir: Path) -> BM25Store:
    metadata_path = output_dir / "bm25_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"BM25 index metadata not found: {metadata_path}. Run scripts/01_build_faiss_indexes.py first.")
    vectorizer = joblib.load(output_dir / "bm25_vectorizer.joblib")
    matrix_csc = joblib.load(output_dir / "bm25_matrix_csc.joblib")
    doc_len = np.load(output_dir / "bm25_doc_len.npy")
    idf = np.load(output_dir / "bm25_idf.npy")
    metadata = load_json(metadata_path)
    return BM25Store(
        vectorizer=vectorizer,
        matrix_csc=matrix_csc,
        doc_len=doc_len,
        avg_doc_len=float(metadata.get("avg_doc_len", float(doc_len.mean()) if len(doc_len) else 0.0)),
        idf=idf,
        metadata=metadata,
    )


def save_bm25_subset(output_dir: Path, name: str, row_ids: np.ndarray, metadata: dict | None = None) -> dict:
    ensure_dir(output_dir)
    row_ids = np.asarray(row_ids, dtype="int64")
    np.save(output_dir / f"{name}_row_ids.npy", row_ids)
    meta = {"name": name, "row_count": int(len(row_ids)), **(metadata or {})}
    save_json(meta, output_dir / f"{name}_metadata.json")
    return meta


def load_bm25_subset(output_dir: Path, name: str) -> tuple[np.ndarray, dict]:
    row_id_path = output_dir / f"{name}_row_ids.npy"
    metadata_path = output_dir / f"{name}_metadata.json"
    if not row_id_path.exists():
        raise FileNotFoundError(f"BM25 subset row IDs not found: {row_id_path}")
    row_ids = np.load(row_id_path)
    metadata = load_json(metadata_path) if metadata_path.exists() else {"name": name, "row_count": int(len(row_ids))}
    return row_ids, metadata


def _top_k_from_scores(scores: np.ndarray, row_ids: np.ndarray, top_k: int) -> list[BM25Result]:
    if len(row_ids) == 0:
        return []
    top_k = max(1, min(int(top_k), int(len(row_ids))))
    subset_scores = scores[row_ids]
    positive = subset_scores > 0
    if not np.any(positive):
        return []
    positive_positions = np.flatnonzero(positive)
    positive_scores = subset_scores[positive_positions]
    if len(positive_positions) > top_k:
        # Pick top K without sorting the whole subset.
        keep = np.argpartition(-positive_scores, top_k - 1)[:top_k]
        positive_positions = positive_positions[keep]
        positive_scores = positive_scores[keep]
    order = np.argsort(-positive_scores)
    results: list[BM25Result] = []
    for rank, idx in enumerate(order, start=1):
        pos = int(positive_positions[idx])
        results.append(BM25Result(row_id=int(row_ids[pos]), score=float(positive_scores[idx]), rank=rank))
    return results


def search_bm25(store: BM25Store, query_text: str, top_k: int, row_ids: Iterable[int] | np.ndarray | None = None, settings: Settings | None = None) -> list[BM25Result]:
    """Search the BM25 index and return top rows.

    If ``row_ids`` is supplied, ranking is restricted to that subset. The BM25
    scores are computed from the global corpus statistics, which keeps scores
    comparable across source-specific retrieval routes.
    """
    query_text = clean_text_value(query_text)
    if not query_text:
        return []
    q = store.vectorizer.transform([query_text])
    if q.nnz == 0:
        return []

    k1 = float(getattr(settings, "bm25_k1", store.metadata.get("bm25_k1", 1.5))) if settings is not None else float(store.metadata.get("bm25_k1", 1.5))
    b = float(getattr(settings, "bm25_b", store.metadata.get("bm25_b", 0.75))) if settings is not None else float(store.metadata.get("bm25_b", 0.75))
    avgdl = float(store.avg_doc_len) if store.avg_doc_len > 0 else 1.0
    doc_len = store.doc_len
    scores = np.zeros(store.matrix_csc.shape[0], dtype=np.float32)

    q_csr = q.tocsr()
    q_indices = q_csr.indices
    q_counts = q_csr.data.astype(np.float32)
    for term_idx, qtf in zip(q_indices, q_counts):
        start = store.matrix_csc.indptr[term_idx]
        end = store.matrix_csc.indptr[term_idx + 1]
        docs = store.matrix_csc.indices[start:end]
        tf = store.matrix_csc.data[start:end].astype(np.float32)
        if len(docs) == 0:
            continue
        denom = tf + k1 * (1.0 - b + b * (doc_len[docs] / avgdl))
        term_scores = store.idf[term_idx] * ((tf * (k1 + 1.0)) / np.maximum(denom, 1e-9))
        # Query-term frequency is usually 1 for short incident descriptions, but
        # using log1p(qtf) gives repeated query terms a mild boost.
        scores[docs] += term_scores * (1.0 + np.log1p(float(qtf)))

    if row_ids is None:
        row_ids = np.arange(store.matrix_csc.shape[0], dtype="int64")
    else:
        row_ids = np.asarray(list(row_ids) if not isinstance(row_ids, np.ndarray) else row_ids, dtype="int64")
    return _top_k_from_scores(scores, row_ids=row_ids, top_k=top_k)


def bm25_results_to_frame(results: list[BM25Result], records: pd.DataFrame, prefix: str = "matched") -> pd.DataFrame:
    rows = []
    for r in results:
        rec = records.iloc[int(r.row_id)].to_dict()
        row = {f"{prefix}_row_id": int(r.row_id), "rank": int(r.rank), "bm25_score": float(r.score)}
        for key, value in rec.items():
            row[f"{prefix}_{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows)
