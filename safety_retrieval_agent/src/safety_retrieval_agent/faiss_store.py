"""FAISS index build/load/search utilities.

The project now defaults to HNSW for faster interactive retrieval. All vectors
are L2-normalized before indexing/search, so inner product scores can be read as
cosine similarity when the FAISS index uses METRIC_INNER_PRODUCT. If an older
FAISS build falls back to L2 HNSW, returned L2 distances are converted back to a
cosine-equivalent score for downstream quality gating.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import Settings
from .artifact_io import artifact_exists, artifact_join, load_faiss_index, load_numpy, read_json
from .utils import ensure_dir, normalize_vectors, save_json


@dataclass
class SearchResult:
    row_id: int
    score: float
    rank: int


def _import_faiss():
    try:
        import faiss
    except ImportError as exc:
        raise ImportError("faiss-cpu is required. Install it with: pip install faiss-cpu") from exc
    return faiss


def _set_hnsw_runtime_params(index: Any, settings: Settings | None = None, metadata: dict | None = None) -> None:
    """Set HNSW efSearch after build/load when the index supports it."""
    if not hasattr(index, "hnsw"):
        return
    ef_search = None
    if settings is not None:
        ef_search = getattr(settings, "hnsw_ef_search", None)
    if ef_search is None and metadata:
        ef_search = metadata.get("hnsw_ef_search")
    if ef_search is not None:
        try:
            index.hnsw.efSearch = int(ef_search)
        except Exception:
            pass


def _index_metric_name(index: Any) -> str:
    faiss = _import_faiss()
    metric = getattr(index, "metric_type", None)
    if metric == getattr(faiss, "METRIC_INNER_PRODUCT", None):
        return "inner_product"
    if metric == getattr(faiss, "METRIC_L2", None):
        return "l2"
    return str(metric or "unknown")


def _score_from_faiss_output(index: Any, raw_score: float) -> float:
    """Return a similarity-style score from FAISS output.

    For inner-product indexes over normalized vectors, FAISS scores are cosine
    similarities. For L2 indexes over normalized vectors, FAISS returns squared
    L2 distances, so cosine = 1 - distance / 2.
    """
    faiss = _import_faiss()
    metric = getattr(index, "metric_type", None)
    value = float(raw_score)
    if metric == getattr(faiss, "METRIC_L2", None):
        return float(1.0 - (value / 2.0))
    return value


def build_faiss_index(vectors: np.ndarray, settings: Settings):
    """Build a FAISS index over normalized vectors.

    Supported index types:
    - flat: exact IndexFlatIP search.
    - hnsw: approximate IndexHNSWFlat search, using inner product when supported.

    HNSW is faster for interactive retrieval over large indexes. Use flat for
    exact-search validation/debugging by setting SAFETY_RETRIEVAL_FAISS_INDEX_TYPE=flat.
    """
    faiss = _import_faiss()
    vectors = normalize_vectors(np.asarray(vectors, dtype="float32"))
    if vectors.ndim != 2 or vectors.shape[0] == 0:
        raise ValueError("vectors must be a non-empty 2D array")
    dim = int(vectors.shape[1])
    index_type = str(settings.faiss_index_type).strip().lower()
    if index_type == "flat":
        index = faiss.IndexFlatIP(dim)
    elif index_type == "hnsw":
        try:
            index = faiss.IndexHNSWFlat(dim, int(settings.hnsw_m), faiss.METRIC_INNER_PRODUCT)
        except TypeError:
            # Older FAISS builds may not accept metric in the constructor. Try to
            # set metric_type explicitly; if that is not supported, downstream
            # search converts L2 distances to cosine-equivalent scores.
            index = faiss.IndexHNSWFlat(dim, int(settings.hnsw_m))
            try:
                index.metric_type = faiss.METRIC_INNER_PRODUCT
            except Exception:
                pass
        if hasattr(index, "hnsw"):
            index.hnsw.efConstruction = int(settings.hnsw_ef_construction)
            index.hnsw.efSearch = int(settings.hnsw_ef_search)
    else:
        raise ValueError("faiss_index_type must be 'hnsw' or 'flat'.")
    index.add(vectors)
    return index


def save_index_bundle(index, row_ids: np.ndarray, output_dir: Path, name: str, metadata: dict | None = None) -> None:
    faiss = _import_faiss()
    ensure_dir(output_dir)
    faiss.write_index(index, str(output_dir / f"{name}.faiss"))
    np.save(output_dir / f"{name}_row_ids.npy", np.asarray(row_ids, dtype="int64"))
    meta = dict(metadata or {})
    meta.update(
        {
            "faiss_metric": _index_metric_name(index),
            "faiss_ntotal": int(index.ntotal),
            "faiss_index_class": type(index).__name__,
        }
    )
    save_json(meta, output_dir / f"{name}_metadata.json")


def load_index_bundle(index_dir: Path | str, name: str, settings: Settings | None = None):
    """Load a FAISS index bundle from a local folder or Azure ML datastore URI."""
    index_path = artifact_join(index_dir, f"{name}.faiss")
    row_ids_path = artifact_join(index_dir, f"{name}_row_ids.npy")
    metadata_path = artifact_join(index_dir, f"{name}_metadata.json")
    index = load_faiss_index(index_path)
    row_ids = load_numpy(row_ids_path)
    metadata = read_json(metadata_path) if artifact_exists(metadata_path) else {}
    _set_hnsw_runtime_params(index, settings=settings, metadata=metadata)
    return index, row_ids, metadata


def search_index(index, row_ids: np.ndarray, query_vectors: np.ndarray, top_k: int) -> list[list[SearchResult]]:
    query_vectors = normalize_vectors(np.asarray(query_vectors, dtype="float32"))
    if query_vectors.ndim == 1:
        query_vectors = query_vectors.reshape(1, -1)
    k = max(1, min(int(top_k), int(index.ntotal)))
    raw_scores, positions = index.search(query_vectors, k)
    output: list[list[SearchResult]] = []
    for q_scores, q_positions in zip(raw_scores, positions):
        rows: list[SearchResult] = []
        for rank, (raw_score, pos) in enumerate(zip(q_scores, q_positions), start=1):
            if int(pos) < 0:
                continue
            score = _score_from_faiss_output(index, float(raw_score))
            rows.append(SearchResult(row_id=int(row_ids[int(pos)]), score=float(score), rank=rank))
        output.append(rows)
    return output


def build_and_save_subset_index(
    vectors: np.ndarray,
    row_mask: np.ndarray,
    settings: Settings,
    name: str,
    metadata: dict | None = None,
) -> dict:
    row_ids = np.flatnonzero(row_mask)
    if len(row_ids) == 0:
        return {"name": name, "row_count": 0, "skipped": True}
    subset_vectors = vectors[row_ids]
    index = build_faiss_index(subset_vectors, settings)
    bundle_meta = {
        "name": name,
        "row_count": int(len(row_ids)),
        "faiss_index_type": settings.faiss_index_type,
        "hnsw_m": settings.hnsw_m,
        "hnsw_ef_construction": settings.hnsw_ef_construction,
        "hnsw_ef_search": settings.hnsw_ef_search,
        "score_interpretation": "cosine_similarity_over_l2_normalized_vectors",
        **(metadata or {}),
    }
    save_index_bundle(index, row_ids, settings.indexes_dir(), name, bundle_meta)
    return bundle_meta


def results_to_frame(results: list[SearchResult], records: pd.DataFrame, prefix: str = "matched") -> pd.DataFrame:
    rows = []
    for r in results:
        rec = records.iloc[r.row_id].to_dict()
        row = {f"{prefix}_row_id": r.row_id, "rank": r.rank, "similarity_score": r.score}
        for key, value in rec.items():
            row[f"{prefix}_{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows)
