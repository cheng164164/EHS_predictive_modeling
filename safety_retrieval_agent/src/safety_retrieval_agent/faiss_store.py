"""FAISS index build/load/search utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Settings
from .utils import ensure_dir, load_json, normalize_vectors, save_json


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


def build_faiss_index(vectors: np.ndarray, settings: Settings):
    """Build a FAISS index over normalized vectors.

    Scores are inner products. Because vectors are L2-normalized, inner product is
    cosine similarity.
    """
    faiss = _import_faiss()
    vectors = normalize_vectors(np.asarray(vectors, dtype="float32"))
    if vectors.ndim != 2 or vectors.shape[0] == 0:
        raise ValueError("vectors must be a non-empty 2D array")
    dim = int(vectors.shape[1])
    index_type = settings.faiss_index_type.lower()
    if index_type == "flat":
        index = faiss.IndexFlatIP(dim)
    elif index_type == "hnsw":
        try:
            index = faiss.IndexHNSWFlat(dim, int(settings.hnsw_m), faiss.METRIC_INNER_PRODUCT)
        except TypeError:
            # Older FAISS builds may not accept metric in constructor. With
            # normalized vectors, L2 ranking is equivalent to cosine ranking.
            index = faiss.IndexHNSWFlat(dim, int(settings.hnsw_m))
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
    save_json(metadata or {}, output_dir / f"{name}_metadata.json")


def load_index_bundle(index_dir: Path, name: str):
    faiss = _import_faiss()
    index = faiss.read_index(str(index_dir / f"{name}.faiss"))
    row_ids = np.load(index_dir / f"{name}_row_ids.npy")
    metadata_path = index_dir / f"{name}_metadata.json"
    metadata = load_json(metadata_path) if metadata_path.exists() else {}
    return index, row_ids, metadata


def search_index(index, row_ids: np.ndarray, query_vectors: np.ndarray, top_k: int) -> list[list[SearchResult]]:
    query_vectors = normalize_vectors(np.asarray(query_vectors, dtype="float32"))
    if query_vectors.ndim == 1:
        query_vectors = query_vectors.reshape(1, -1)
    k = max(1, min(int(top_k), int(index.ntotal)))
    scores, positions = index.search(query_vectors, k)
    output: list[list[SearchResult]] = []
    for q_scores, q_positions in zip(scores, positions):
        rows: list[SearchResult] = []
        for rank, (score, pos) in enumerate(zip(q_scores, q_positions), start=1):
            if int(pos) < 0:
                continue
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
