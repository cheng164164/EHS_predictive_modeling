#!/usr/bin/env python
"""Build local transformer embeddings, data-driven themes, FAISS indexes, and BM25 indexes.

Run without args:

    python scripts/01_build_faiss_indexes.py

Configuration lives in src/safety_retrieval_agent/config.py.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from _bootstrap import PROJECT_ROOT  # noqa: F401
from safety_retrieval_agent.config import get_settings
from safety_retrieval_agent.embedding import TextEmbedder
from safety_retrieval_agent.bm25_store import build_bm25_store, save_bm25_store, save_bm25_subset
from safety_retrieval_agent.faiss_store import build_and_save_subset_index
from safety_retrieval_agent.theme import build_theme_profiles, discover_themes, save_theme_model
from safety_retrieval_agent.index_pipeline import build_retrieval_index_masks
from safety_retrieval_agent.utils import chunk_ranges, ensure_dir, load_json, save_json


def _embedding_path(settings):
    return settings.embeddings_dir() / "event_embeddings.npy"


def _embedding_shape_path(settings):
    return settings.embeddings_dir() / "embedding_shape.npy"


def _embedding_metadata_path(settings):
    return settings.embeddings_dir() / "embedding_model_metadata.json"


def _embedding_scope_sample_path(settings):
    return settings.embedding_scope_path().with_name("safety_embedding_scope_sample.csv")


def _embedding_scope_full_csv_path(settings):
    return settings.embedding_scope_csv_path()


def _metadata_looks_compatible(metadata: dict, settings, record_count: int, shape: tuple[int, ...]) -> bool:
    if not metadata:
        return False
    if int(metadata.get("record_count", -1)) != int(record_count):
        return False
    if len(shape) != 2 or int(shape[0]) != int(record_count):
        return False
    # The actual model can be primary or fallback. Reuse is only safe when the
    # metadata tells us what model created the vectors and the stored requested
    # primary model/backend match the current config.
    if not (metadata.get("embedding_model_name") and metadata.get("embedding_backend")):
        return False
    stored_primary_model = str(metadata.get("primary_embedding_model_name") or metadata.get("embedding_model_name") or "")
    stored_primary_backend = str(metadata.get("primary_embedding_backend") or metadata.get("embedding_backend") or "")
    return stored_primary_model == str(settings.embedding_model_name) and stored_primary_backend == str(settings.embedding_backend)


def _nonempty_description_mask(records: pd.DataFrame, settings) -> pd.Series:
    """Return True for rows with a usable description field.

    Generic audit observations often have title/source-subtype text but blank
    descriptions. For the MVP embedding scope, those generic audit rows should be
    included only when they have a real description. Unsafe/safe audit rows are
    controlled separately by source_role and are not affected by this mask.
    """
    min_chars = int(getattr(settings, "generic_audit_description_min_chars", 1))
    if "description_nonempty" in records.columns and "description_text_length" in records.columns:
        length = pd.to_numeric(records["description_text_length"], errors="coerce").fillna(0)
        return records["description_nonempty"].fillna(False).astype(bool) & length.ge(min_chars)
    if "description_text" in records.columns:
        return records["description_text"].fillna("").astype(str).str.strip().str.len().ge(min_chars)
    if "description" in records.columns:
        return records["description"].fillna("").astype(str).str.strip().str.len().ge(min_chars)
    # Be conservative when older prepared files do not contain description fields.
    return pd.Series(False, index=records.index)


def _filter_embedding_scope(records_all: pd.DataFrame, settings) -> tuple[pd.DataFrame, dict]:
    """Filter the full knowledge base to the MVP retrieval/embedding scope.

    The earlier v6 build embedded every prepared record, then created subset
    indexes afterward. That was wasteful because generic incidents/inspections
    were not part of the MVP use case. This function filters before embedding.
    """
    records_all = records_all.copy()
    if "source_role" not in records_all.columns:
        raise ValueError("Knowledge base is missing source_role. Run scripts/00_prepare_knowledge_base.py first.")
    if "retrieval_text" not in records_all.columns:
        raise ValueError("Knowledge base is missing retrieval_text. Run scripts/00_prepare_knowledge_base.py first.")

    role = records_all["source_role"].fillna("").astype(str)
    configured_roles = set(str(v) for v in getattr(settings, "embedding_source_roles", ()))
    non_generic_roles = configured_roles - {"audit_observation"}

    # Always handle generic audit_observation through the explicit audit settings
    # so blank-description generic audits are not accidentally included.
    standard_scope_mask = role.isin(non_generic_roles)

    generic_audit_mask = role.eq("audit_observation")
    generic_audit_desc_mask = _nonempty_description_mask(records_all, settings)
    if bool(getattr(settings, "include_generic_audit_observations", True)):
        if bool(getattr(settings, "require_generic_audit_description", True)):
            generic_include_mask = generic_audit_mask & generic_audit_desc_mask
        else:
            generic_include_mask = generic_audit_mask
    else:
        generic_include_mask = pd.Series(False, index=records_all.index)

    scope_mask = standard_scope_mask | generic_include_mask
    records = records_all.loc[scope_mask].copy().reset_index(drop=True)
    records["row_id"] = np.arange(len(records), dtype="int64")

    excluded_generic_empty = int((generic_audit_mask & ~generic_audit_desc_mask).sum())
    included_generic = int(generic_include_mask.sum())
    total_generic = int(generic_audit_mask.sum())

    summary = {
        "full_knowledge_base_rows": int(len(records_all)),
        "embedding_scope_rows": int(len(records)),
        "excluded_from_embedding_scope_rows": int(len(records_all) - len(records)),
        "configured_embedding_source_roles": sorted(non_generic_roles),
        "include_generic_audit_observations": bool(getattr(settings, "include_generic_audit_observations", True)),
        "require_generic_audit_description": bool(getattr(settings, "require_generic_audit_description", True)),
        "generic_audit_description_min_chars": int(getattr(settings, "generic_audit_description_min_chars", 1)),
        "generic_audit_total_rows_in_knowledge_base": total_generic,
        "generic_audit_rows_included": included_generic,
        "generic_audit_empty_description_rows_excluded": excluded_generic_empty,
        "full_source_role_counts": {str(k): int(v) for k, v in role.value_counts(dropna=False).to_dict().items()},
        "embedding_scope_counts_by_role": {
            str(k): int(v) for k, v in records["source_role"].value_counts(dropna=False).to_dict().items()
        },
        "embedding_scope_counts_by_source_type": {
            str(k): int(v) for k, v in records["source_type"].value_counts(dropna=False).to_dict().items()
        } if "source_type" in records.columns else {},
    }
    return records, summary


def _save_embedding_scope_reports(records: pd.DataFrame, summary: dict, settings) -> None:
    ensure_dir(settings.embedding_scope_path().parent)
    records.to_pickle(settings.embedding_scope_path())
    records.head(1000).to_csv(_embedding_scope_sample_path(settings), index=False)
    if bool(getattr(settings, "prepare_save_full_csv", False)):
        records.to_csv(_embedding_scope_full_csv_path(settings), index=False, compression="gzip")
    save_json(summary, settings.embedding_scope_summary_path())
    counts = records["source_role"].value_counts(dropna=False).rename_axis("source_role").reset_index(name="count")
    counts.to_csv(settings.embedding_scope_counts_path(), index=False)


def _build_embeddings(records: pd.DataFrame, settings, reuse: bool) -> tuple[np.ndarray, dict]:
    ensure_dir(settings.embeddings_dir())
    path = _embedding_path(settings)
    shape_path = _embedding_shape_path(settings)
    metadata_path = _embedding_metadata_path(settings)
    texts = records["retrieval_text"].fillna("").astype(str).tolist()

    if reuse and path.exists() and shape_path.exists() and metadata_path.exists():
        shape = tuple(int(x) for x in np.load(shape_path))
        metadata = load_json(metadata_path)
        if _metadata_looks_compatible(metadata, settings, len(records), shape):
            print(
                f"[01] Reusing embeddings: {path} shape={shape} "
                f"model={metadata.get('embedding_model_name')}",
                flush=True,
            )
            return np.load(path, mmap_mode="r"), metadata
        print("[01] Existing embeddings were found but metadata/shape was incompatible. Rebuilding.", flush=True)

    embedder = TextEmbedder(settings)
    print(
        f"[01] Embedding model requested: {settings.embedding_model_name} "
        f"backend={settings.embedding_backend}",
        flush=True,
    )

    if len(texts) == 0:
        raise ValueError("No records available for embedding after applying embedding-scope filters.")

    first_end = min(settings.embedding_batch_size, len(texts))
    first_vectors = embedder.encode(texts[:first_end], is_query=False, batch_size=settings.embedding_batch_size)
    # Once the first batch succeeds, lock the actual model. Later batches should
    # not silently fall back to a different model/dimension.
    embedder.disable_fallback()

    dim = int(first_vectors.shape[1])
    memmap = np.lib.format.open_memmap(path, mode="w+", dtype="float32", shape=(len(records), dim))
    memmap[:first_end] = first_vectors

    for start, end in chunk_ranges(len(records), settings.embedding_batch_size):
        if start == 0:
            continue
        batch = texts[start:end]
        memmap[start:end] = embedder.encode(batch, is_query=False, batch_size=settings.embedding_batch_size)
        if settings.show_progress and (end % max(settings.embedding_batch_size * 50, 1) == 0 or end == len(records)):
            print(f"[01] Embedded {end:,}/{len(records):,} records", flush=True)

    memmap.flush()
    np.save(shape_path, np.array([len(records), dim], dtype="int64"))

    metadata = embedder.metadata()
    metadata.update(
        {
            "record_count": int(len(records)),
            "embedding_shape": [int(len(records)), int(dim)],
            "embedding_file": str(path),
            "embedding_scope_summary_file": str(settings.embedding_scope_summary_path()),
            "important_note": "All FAISS indexes in this output folder must be queried with this same embedding model.",
        }
    )
    save_json(metadata, metadata_path)
    # Also save under models for easier review.
    ensure_dir(settings.models_dir())
    save_json(metadata, settings.models_dir() / "embedding_model_metadata.json")
    return np.load(path, mmap_mode="r"), metadata


def _index_metadata(name: str, settings, embedding_metadata: dict, row_count: int) -> dict:
    return {
        "name": name,
        "embedding": embedding_metadata,
        "embedding_model_name": embedding_metadata.get("embedding_model_name"),
        "embedding_backend": embedding_metadata.get("embedding_backend"),
        "used_fallback_embedding_model": embedding_metadata.get("used_fallback_embedding_model"),
        "row_count": int(row_count),
        "faiss_index_type": settings.faiss_index_type,
        "embedding_scope_summary_file": str(settings.embedding_scope_summary_path()),
    }


def main() -> dict:
    settings = get_settings()
    start = time.time()
    ensure_dir(settings.output_dir)
    ensure_dir(settings.indexes_dir())
    ensure_dir(settings.bm25_dir())
    ensure_dir(settings.models_dir())

    if not settings.knowledge_base_path().exists():
        raise FileNotFoundError(
            f"Knowledge base not found: {settings.knowledge_base_path()}. Run scripts/00_prepare_knowledge_base.py first."
        )
    records_all = pd.read_pickle(settings.knowledge_base_path()).reset_index(drop=True)
    records_all["row_id"] = np.arange(len(records_all), dtype="int64")
    print(f"[01] Loaded full knowledge base: {len(records_all):,} records", flush=True)

    records, scope_summary = _filter_embedding_scope(records_all, settings)
    _save_embedding_scope_reports(records, scope_summary, settings)
    print(
        f"[01] Embedding-scope records: {len(records):,} "
        f"(excluded {scope_summary['excluded_from_embedding_scope_rows']:,})",
        flush=True,
    )
    print(
        f"[01] Generic audit_observation included: {scope_summary['generic_audit_rows_included']:,}; "
        f"empty-description generic audits excluded: {scope_summary['generic_audit_empty_description_rows_excluded']:,}",
        flush=True,
    )

    vectors, embedding_metadata = _build_embeddings(records, settings, reuse=settings.build_reuse_embeddings)
    print(
        f"[01] Embeddings shape: {vectors.shape}; actual model={embedding_metadata.get('embedding_model_name')} "
        f"fallback_used={embedding_metadata.get('used_fallback_embedding_model')}",
        flush=True,
    )

    print("[01] Discovering/reusing risk themes on embedding scope...", flush=True)
    records, theme_model = discover_themes(records, vectors, settings)
    save_theme_model(theme_model, settings)
    profiles = build_theme_profiles(records, vectors, settings)
    print(f"[01] Theme profiles: {len(profiles):,}", flush=True)

    print("[01] Saving enriched embedding-scope knowledge base with themes...", flush=True)
    records.to_pickle(settings.enriched_knowledge_base_path())
    records.to_csv(settings.enriched_knowledge_base_path().with_suffix(".csv.gz"), index=False, compression="gzip")

    masks = build_retrieval_index_masks(records)
    configured_index_names = list(getattr(settings, "retrieval_index_names", masks.keys()))
    masks = {name: masks[name] for name in configured_index_names if name in masks}

    index_summaries = []
    bm25_subset_summaries = []
    for name, mask in masks.items():
        row_count = int(mask.sum())
        print(f"[01] Building FAISS index: {name} rows={row_count:,}", flush=True)
        meta = build_and_save_subset_index(
            vectors,
            mask,
            settings,
            name,
            metadata=_index_metadata(name, settings, embedding_metadata, row_count),
        )
        index_summaries.append(meta)

        row_ids = np.flatnonzero(mask)
        bm25_meta = save_bm25_subset(
            settings.bm25_dir(),
            name,
            row_ids,
            metadata={
                "name": name,
                "row_count": int(row_count),
                "source": "safety_knowledge_base_with_themes_embedding_scope",
                "embedding_scope_summary_file": str(settings.embedding_scope_summary_path()),
            },
        )
        bm25_subset_summaries.append(bm25_meta)

    print("[01] Building BM25 keyword index over embedding scope...", flush=True)
    bm25_store = build_bm25_store(records, settings)
    save_bm25_store(bm25_store, settings.bm25_dir())
    print(
        f"[01] BM25 vocabulary size: {bm25_store.metadata.get('vocabulary_size'):,}; "
        f"rows={bm25_store.metadata.get('row_count'):,}",
        flush=True,
    )

    summary = {
        "output_dir": str(settings.output_dir),
        "full_knowledge_base_rows": int(len(records_all)),
        "embedding_scope_rows": int(len(records)),
        "embedding_scope_summary": scope_summary,
        "embedding_scope_summary_file": str(settings.embedding_scope_summary_path()),
        "embedding_model_requested": settings.embedding_model_name,
        "embedding_backend_requested": settings.embedding_backend,
        "embedding_model_used": embedding_metadata.get("embedding_model_name"),
        "embedding_backend_used": embedding_metadata.get("embedding_backend"),
        "used_fallback_embedding_model": bool(embedding_metadata.get("used_fallback_embedding_model")),
        "fallback_embedding_model_name": embedding_metadata.get("fallback_embedding_model_name"),
        "embedding_shape": [int(vectors.shape[0]), int(vectors.shape[1])],
        "retrieval_mode_default": settings.retrieval_mode,
        "faiss_index_type": settings.faiss_index_type,
        "bm25_index_dir": str(settings.bm25_dir()),
        "bm25_metadata": bm25_store.metadata,
        "theme_profile_count": int(len(profiles)),
        "faiss_indexes": index_summaries,
        "bm25_subsets": bm25_subset_summaries,
        "elapsed_seconds": round(time.time() - start, 2),
    }
    save_json(summary, settings.output_dir / "01_build_faiss_indexes_summary.json")
    print(f"[01] Complete: {summary}", flush=True)
    return summary


if __name__ == "__main__":
    main()
