"""Shared split embedding/index-building pipeline utilities.

These functions are used by:
- scripts/01a_generate_embedding_chunks.py
- scripts/01b_build_indexes_from_chunks.py

The existing scripts/00_*.py, scripts/01_build_faiss_indexes.py, and
scripts/02_run_mvp_recommendations.py are intentionally left unchanged. This
module provides a resumable alternative for long Azure ML CPU jobs.
"""
from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .bm25_store import build_bm25_store, save_bm25_store, save_bm25_subset
from .config import Settings
from .embedding import TextEmbedder
from .faiss_store import build_and_save_subset_index
from .theme import build_theme_profiles, discover_themes, save_theme_model
from .artifact_io import artifact_exists, artifact_join, load_numpy, read_json as read_artifact_json
from .utils import clean_text_value, ensure_dir, load_json, save_json, word_count


def embedding_path(settings: Settings) -> Path:
    return settings.embeddings_dir() / "event_embeddings.npy"


def embedding_shape_path(settings: Settings) -> Path:
    return settings.embeddings_dir() / "embedding_shape.npy"


def embedding_metadata_path(settings: Settings) -> Path:
    return settings.embeddings_dir() / "embedding_model_metadata.json"




def _datastore_embeddings_dir_for_merge(settings: Settings) -> str | Path | None:
    """Return the remote datastore embeddings directory for 01b merge reads.

    Do not use settings.artifact_embeddings_dir() here. In auto mode, that
    method intentionally switches to local runtime artifacts when local outputs
    exist. For the 01b rebuild workflow we often want local updated data/scope
    files while reading only the heavy embedding artifacts from the Azure ML
    datastore. Therefore this function uses artifact_azureml_uri directly.
    """
    base = clean_text_value(getattr(settings, "artifact_azureml_uri", ""))
    if base and base.startswith(("azureml://", "abfss://", "wasbs://", "https://")):
        return artifact_join(base.rstrip("/"), "embeddings")
    return None


def datastore_embedding_metadata_path(settings: Settings) -> str | Path | None:
    """Return the configured remote embedding metadata path, when available."""
    remote_embeddings = _datastore_embeddings_dir_for_merge(settings)
    if remote_embeddings is not None:
        return artifact_join(remote_embeddings, "embedding_model_metadata.json")
    return None


def datastore_chunk_file_path(settings: Settings, chunk_index: int) -> str | Path | None:
    """Return the configured remote embedding chunk path, when available."""
    remote_embeddings = _datastore_embeddings_dir_for_merge(settings)
    if remote_embeddings is not None:
        return artifact_join(remote_embeddings, "chunks", f"chunk_{int(chunk_index):05d}.npy")
    return None

def embedding_scope_sample_path(settings: Settings) -> Path:
    return settings.embedding_scope_path().with_name("safety_embedding_scope_sample.csv")


def embedding_scope_full_csv_path(settings: Settings) -> Path:
    return settings.embedding_scope_csv_path()


def chunk_file_path(settings: Settings, chunk_index: int) -> Path:
    return settings.embedding_chunks_dir() / f"chunk_{int(chunk_index):05d}.npy"


def chunk_ranges_by_index(n_records: int, chunk_size: int) -> list[dict]:
    if chunk_size <= 0:
        raise ValueError("embedding_chunk_size must be positive")
    chunks = []
    n_chunks = int(math.ceil(n_records / chunk_size)) if n_records else 0
    for chunk_index in range(n_chunks):
        start = int(chunk_index * chunk_size)
        end = int(min(start + chunk_size, n_records))
        chunks.append({"chunk_index": chunk_index, "start_row": start, "end_row": end, "n_records": end - start})
    return chunks


def _description_text(records: pd.DataFrame) -> pd.Series:
    for col in ["description_text", "description", "matched_description"]:
        if col in records.columns:
            return records[col].fillna("").astype(str).map(clean_text_value)
    return pd.Series("", index=records.index, dtype="string")


def _nonempty_description_mask(records: pd.DataFrame, settings: Settings) -> pd.Series:
    min_chars = int(getattr(settings, "generic_audit_description_min_chars", 1))
    min_words = int(getattr(settings, "generic_audit_description_min_words", 1))
    if "description_nonempty" in records.columns and "description_text_length" in records.columns:
        length = pd.to_numeric(records["description_text_length"], errors="coerce").fillna(0)
        desc = _description_text(records)
        words = desc.map(word_count)
        return records["description_nonempty"].fillna(False).astype(bool) & length.ge(min_chars) & words.ge(min_words)
    desc = _description_text(records)
    return desc.str.len().ge(min_chars) & desc.map(word_count).ge(min_words)


def _apply_text_quality_filter(records: pd.DataFrame, settings: Settings) -> tuple[pd.DataFrame, dict]:
    if "retrieval_text" not in records.columns:
        raise ValueError("Dataframe is missing retrieval_text")
    text = records["retrieval_text"].fillna("").astype(str).map(clean_text_value)
    text_len = text.str.len()
    text_words = text.map(word_count)
    mask = text_len.ge(int(settings.min_text_chars)) & text_words.ge(int(settings.min_text_words))
    out = records.loc[mask].copy().reset_index(drop=True)
    out["retrieval_text"] = text.loc[mask].tolist()
    out["text_length"] = text_len.loc[mask].astype(int).tolist()
    out["text_word_count"] = text_words.loc[mask].astype(int).tolist()
    return out, {
        "min_text_chars": int(settings.min_text_chars),
        "min_text_words": int(settings.min_text_words),
        "rows_before_text_quality_filter": int(len(records)),
        "rows_removed_by_text_quality_filter": int((~mask).sum()),
        "rows_after_text_quality_filter": int(len(out)),
    }


def filter_embedding_scope(records_all: pd.DataFrame, settings: Settings) -> tuple[pd.DataFrame, dict]:
    """Filter the full knowledge base to the configured MVP embedding scope."""
    records_all = records_all.copy()
    if "source_role" not in records_all.columns:
        raise ValueError("Knowledge base is missing source_role. Run scripts/00_prepare_knowledge_base.py first.")
    if "retrieval_text" not in records_all.columns:
        raise ValueError("Knowledge base is missing retrieval_text. Run scripts/00_prepare_knowledge_base.py first.")

    role = records_all["source_role"].fillna("").astype(str)
    configured_roles = set(str(v) for v in getattr(settings, "embedding_source_roles", ()))
    non_generic_roles = configured_roles - {"audit_observation"}

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
    scoped = records_all.loc[scope_mask].copy().reset_index(drop=True)
    scoped, text_summary = _apply_text_quality_filter(scoped, settings)

    if settings.max_records is not None and len(scoped) > int(settings.max_records):
        scoped = scoped.head(int(settings.max_records)).copy().reset_index(drop=True)
        max_records_note = f"Capped to max_records={settings.max_records}"
    else:
        max_records_note = None

    scoped["row_id"] = np.arange(len(scoped), dtype="int64")

    summary = {
        "full_knowledge_base_rows": int(len(records_all)),
        "embedding_scope_rows": int(len(scoped)),
        "excluded_from_embedding_scope_rows": int(len(records_all) - len(scoped)),
        "configured_embedding_source_roles": sorted(non_generic_roles),
        "include_generic_audit_observations": bool(getattr(settings, "include_generic_audit_observations", True)),
        "require_generic_audit_description": bool(getattr(settings, "require_generic_audit_description", True)),
        "generic_audit_description_min_chars": int(getattr(settings, "generic_audit_description_min_chars", 1)),
        "generic_audit_description_min_words": int(getattr(settings, "generic_audit_description_min_words", 1)),
        "generic_audit_total_rows_in_knowledge_base": int(generic_audit_mask.sum()),
        "generic_audit_rows_included_before_text_quality_filter": int(generic_include_mask.sum()),
        "generic_audit_empty_or_short_description_rows_excluded": int((generic_audit_mask & ~generic_audit_desc_mask).sum()),
        "full_source_role_counts": {str(k): int(v) for k, v in role.value_counts(dropna=False).to_dict().items()},
        "embedding_scope_counts_by_role": {str(k): int(v) for k, v in scoped["source_role"].value_counts(dropna=False).to_dict().items()},
        "embedding_scope_counts_by_source_type": {str(k): int(v) for k, v in scoped["source_type"].value_counts(dropna=False).to_dict().items()} if "source_type" in scoped.columns else {},
        "text_quality_filter": text_summary,
        "max_records_note": max_records_note,
    }
    return scoped, summary


def save_embedding_scope_reports(records: pd.DataFrame, summary: dict, settings: Settings) -> None:
    ensure_dir(settings.embedding_scope_path().parent)
    records.to_pickle(settings.embedding_scope_path())
    records.head(1000).to_csv(embedding_scope_sample_path(settings), index=False)
    if bool(getattr(settings, "prepare_save_full_csv", False)):
        records.to_csv(embedding_scope_full_csv_path(settings), index=False, compression="gzip")
    save_json(summary, settings.embedding_scope_summary_path())
    counts = records["source_role"].value_counts(dropna=False).rename_axis("source_role").reset_index(name="count")
    counts.to_csv(settings.embedding_scope_counts_path(), index=False)


def load_or_create_embedding_scope(settings: Settings) -> tuple[pd.DataFrame, dict]:
    """Load existing output scope, copy prepared scope, or build from knowledge base."""
    if settings.embedding_scope_path().exists():
        records = pd.read_pickle(settings.embedding_scope_path()).reset_index(drop=True)
        records["row_id"] = np.arange(len(records), dtype="int64")
        summary = load_json(settings.embedding_scope_summary_path()) if settings.embedding_scope_summary_path().exists() else {}
        print(f"[Scope] Loaded existing embedding scope from output_dir: {settings.embedding_scope_path()} rows={len(records):,}", flush=True)
        return records, summary

    if settings.prepared_embedding_scope_path().exists():
        records = pd.read_pickle(settings.prepared_embedding_scope_path()).reset_index(drop=True)
        records, text_summary = _apply_text_quality_filter(records, settings)
        records["row_id"] = np.arange(len(records), dtype="int64")
        summary = {
            "source": str(settings.prepared_embedding_scope_path()),
            "embedding_scope_rows": int(len(records)),
            "text_quality_filter": text_summary,
            "note": "Copied/prepared from mounted prepared_embedding_scope_path.",
        }
        save_embedding_scope_reports(records, summary, settings)
        print(f"[Scope] Copied prepared embedding scope to output_dir: {settings.embedding_scope_path()} rows={len(records):,}", flush=True)
        return records, summary

    kb_path = settings.knowledge_base_path() if settings.knowledge_base_path().exists() else settings.prepared_knowledge_base_path()
    if not kb_path.exists():
        raise FileNotFoundError(
            "Knowledge base not found. Expected one of: "
            f"{settings.knowledge_base_path()} or {settings.prepared_knowledge_base_path()}. "
            "Run scripts/00_prepare_knowledge_base.py locally first, or configure SAFETY_RETRIEVAL_PREPARED_OUTPUT_DIR."
        )
    records_all = pd.read_pickle(kb_path).reset_index(drop=True)
    records_all["row_id"] = np.arange(len(records_all), dtype="int64")
    print(f"[Scope] Loaded knowledge base: {kb_path} rows={len(records_all):,}", flush=True)
    records, summary = filter_embedding_scope(records_all, settings)
    summary["knowledge_base_source"] = str(kb_path)
    save_embedding_scope_reports(records, summary, settings)
    print(f"[Scope] Created embedding scope rows={len(records):,}; summary={settings.embedding_scope_summary_path()}", flush=True)
    return records, summary


def _load_existing_embedding_metadata(settings: Settings) -> dict | None:
    path = embedding_metadata_path(settings)
    if path.exists():
        try:
            return load_json(path)
        except Exception:
            return None
    return None


def _load_embedding_metadata_for_merge(settings: Settings) -> tuple[dict | None, str | Path | None]:
    """Load embedding metadata for 01b.

    01b may be run locally with updated local data/scope files while the heavy
    embedding chunks remain in the Azure ML datastore. Prefer the local metadata
    when present; otherwise read the configured datastore/runtime artifact path.
    """
    local_path = embedding_metadata_path(settings)
    if local_path.exists():
        try:
            return load_json(local_path), local_path
        except Exception:
            pass

    remote_path = datastore_embedding_metadata_path(settings)
    if remote_path is not None and artifact_exists(remote_path):
        try:
            return read_artifact_json(remote_path), remote_path
        except Exception:
            pass
    return None, None


def _resolve_chunk_file_for_read(settings: Settings, chunk_index: int) -> str | Path:
    """Return local chunk path when present, otherwise datastore chunk path."""
    local_path = chunk_file_path(settings, chunk_index)
    if local_path.exists():
        return local_path
    remote_path = datastore_chunk_file_path(settings, chunk_index)
    if remote_path is not None and artifact_exists(remote_path):
        return remote_path
    return local_path


def _existing_chunk_ok(path: Path, expected_rows: int, expected_dim: int | None = None) -> bool:
    if not path.exists():
        return False
    try:
        arr = np.load(path, mmap_mode="r")
        if arr.ndim != 2 or int(arr.shape[0]) != int(expected_rows):
            return False
        if expected_dim is not None and int(arr.shape[1]) != int(expected_dim):
            return False
        return True
    except Exception:
        return False


def _write_manifest(records: pd.DataFrame, chunks: list[dict], settings: Settings, metadata: dict | None = None) -> pd.DataFrame:
    rows = []
    expected_dim = int(metadata.get("embedding_dimension")) if metadata and metadata.get("embedding_dimension") else None
    for ch in chunks:
        path = chunk_file_path(settings, ch["chunk_index"])
        complete = _existing_chunk_ok(path, ch["n_records"], expected_dim)
        rows.append(
            {
                **ch,
                "chunk_file": str(path),
                "complete": bool(complete),
                "file_exists": bool(path.exists()),
                "embedding_dimension": expected_dim,
            }
        )
    manifest = pd.DataFrame(rows)
    ensure_dir(settings.embedding_chunks_manifest_path().parent)
    manifest.to_csv(settings.embedding_chunks_manifest_path(), index=False)
    return manifest




def _metadata_matches_requested_model(metadata: dict, settings: Settings) -> bool:
    """Return True if existing chunks belong to the currently requested model setup.

    If a previous run fell back from BGE-M3 to Qwen, metadata["embedding_model_name"]
    is the actual fallback model. That is still compatible as long as the stored
    primary model/backend matches the current requested primary model/backend.
    """
    if not metadata:
        return False
    requested_model = str(settings.embedding_model_name)
    requested_backend = str(settings.embedding_backend)
    stored_primary_model = str(metadata.get("primary_embedding_model_name") or metadata.get("embedding_model_name") or "")
    stored_primary_backend = str(metadata.get("primary_embedding_backend") or metadata.get("embedding_backend") or "")
    return stored_primary_model == requested_model and stored_primary_backend == requested_backend


def _clear_existing_embedding_artifacts(settings: Settings) -> None:
    """Delete local/output-mount embedding artifacts so a different model can rebuild."""
    for path in [embedding_path(settings), embedding_shape_path(settings), embedding_metadata_path(settings), settings.models_dir() / "embedding_model_metadata.json"]:
        try:
            if path.exists():
                path.unlink()
        except Exception:
            pass
    chunk_dir = settings.embedding_chunks_dir()
    if chunk_dir.exists():
        for path in chunk_dir.glob("chunk_*.npy"):
            try:
                path.unlink()
            except Exception:
                pass
    for path in [settings.embedding_chunks_manifest_path(), settings.embedding_chunk_run_summary_path()]:
        try:
            if path.exists():
                path.unlink()
        except Exception:
            pass


def generate_embedding_chunks(settings: Settings) -> dict:
    """Generate chunked embeddings with resume support."""
    start_time = time.time()
    ensure_dir(settings.embeddings_dir())
    ensure_dir(settings.embedding_chunks_dir())
    ensure_dir(settings.models_dir())

    records, scope_summary = load_or_create_embedding_scope(settings)
    texts = records["retrieval_text"].fillna("").astype(str).tolist()
    if not texts:
        raise ValueError("No records available for embedding after scope/text filters.")

    chunks = chunk_ranges_by_index(len(records), int(settings.embedding_chunk_size))
    start_idx = max(0, int(settings.embedding_chunk_start_index))
    end_idx = settings.embedding_chunk_end_index if settings.embedding_chunk_end_index is not None else len(chunks)
    end_idx = min(int(end_idx), len(chunks))
    selected_chunks = [ch for ch in chunks if start_idx <= int(ch["chunk_index"]) < end_idx]

    existing_metadata = _load_existing_embedding_metadata(settings)
    if existing_metadata and not _metadata_matches_requested_model(existing_metadata, settings):
        message = (
            "Existing embedding chunks were built with a different requested embedding model/backend. "
            f"Existing primary={existing_metadata.get('primary_embedding_model_name') or existing_metadata.get('embedding_model_name')} "
            f"backend={existing_metadata.get('primary_embedding_backend') or existing_metadata.get('embedding_backend')}; "
            f"requested primary={settings.embedding_model_name} backend={settings.embedding_backend}."
        )
        if bool(getattr(settings, "force_rebuild_embeddings", False)):
            print(f"[01a] {message} Clearing old chunks because force_rebuild_embeddings=True.", flush=True)
            _clear_existing_embedding_artifacts(settings)
            existing_metadata = None
        else:
            raise RuntimeError(message + " Set SAFETY_RETRIEVAL_FORCE_REBUILD_EMBEDDINGS=true or use a new output/datastore path.")

    if existing_metadata and existing_metadata.get("embedding_model_name") and existing_metadata.get("embedding_backend"):
        print(
            "[01a] Existing embedding metadata found. Resuming with locked actual model: "
            f"{existing_metadata.get('embedding_model_name')} backend={existing_metadata.get('embedding_backend')}",
            flush=True,
        )
        embedder = TextEmbedder(
            settings,
            model_name=str(existing_metadata["embedding_model_name"]),
            backend=str(existing_metadata["embedding_backend"]),
            enable_fallback=False,
            query_instruction=str(existing_metadata.get("query_instruction") or settings.query_instruction),
        )
        embedding_metadata = dict(existing_metadata)
    else:
        embedder = TextEmbedder(settings)
        embedding_metadata = None

    completed_before = 0
    completed_now = 0
    skipped_existing = 0
    for ch in selected_chunks:
        idx = int(ch["chunk_index"])
        start = int(ch["start_row"])
        end = int(ch["end_row"])
        n = int(ch["n_records"])
        path = chunk_file_path(settings, idx)
        expected_dim = int(embedding_metadata.get("embedding_dimension")) if embedding_metadata and embedding_metadata.get("embedding_dimension") else None

        if bool(settings.skip_existing_embedding_chunks) and _existing_chunk_ok(path, n, expected_dim):
            skipped_existing += 1
            completed_before += 1
            print(f"[01a] Skipping existing complete chunk {idx:05d} rows={start:,}:{end:,}", flush=True)
            continue

        print(f"[01a] Embedding chunk {idx:05d} rows={start:,}:{end:,} n={n:,}", flush=True)
        vectors = embedder.encode(texts[start:end], is_query=False, batch_size=int(settings.embedding_batch_size))
        if vectors.shape[0] != n:
            raise ValueError(f"Chunk {idx} returned {vectors.shape[0]} vectors for {n} records")

        # First successful new chunk locks the model and writes metadata. This
        # prevents BGE and fallback vectors from being mixed across later chunks.
        if embedding_metadata is None:
            embedder.disable_fallback()
            embedding_metadata = embedder.metadata()
            embedding_metadata.update(
                {
                    "record_count": int(len(records)),
                    "embedding_dimension": int(vectors.shape[1]),
                    "embedding_chunk_size": int(settings.embedding_chunk_size),
                    "embedding_chunks_dir": str(settings.embedding_chunks_dir()),
                    "embedding_scope_summary_file": str(settings.embedding_scope_summary_path()),
                    "important_note": "All chunks and FAISS indexes in this output folder must use this same embedding model.",
                }
            )
            save_json(embedding_metadata, embedding_metadata_path(settings))
            save_json(embedding_metadata, settings.models_dir() / "embedding_model_metadata.json")
        else:
            expected_dim = int(embedding_metadata.get("embedding_dimension") or vectors.shape[1])
            if int(vectors.shape[1]) != expected_dim:
                raise ValueError(
                    f"Chunk {idx} dimension {vectors.shape[1]} differs from existing embedding dimension {expected_dim}. "
                    "Delete incompatible chunks/metadata before rebuilding."
                )

        tmp_path = path.with_suffix(".tmp.npy")
        np.save(tmp_path, vectors.astype("float32", copy=False))
        os.replace(tmp_path, path)
        completed_now += 1
        print(f"[01a] Saved chunk {idx:05d}: {path}", flush=True)

    # Refresh manifest after all selected chunks.
    metadata = _load_existing_embedding_metadata(settings) or embedding_metadata or {}
    manifest = _write_manifest(records, chunks, settings, metadata)
    complete_count = int(manifest["complete"].sum()) if not manifest.empty else 0
    incomplete_count = int(len(manifest) - complete_count)
    summary = {
        "script": "01a_generate_embedding_chunks.py",
        "record_count": int(len(records)),
        "chunk_size": int(settings.embedding_chunk_size),
        "total_chunks": int(len(chunks)),
        "selected_chunk_start_index": int(start_idx),
        "selected_chunk_end_index": int(end_idx),
        "selected_chunks": int(len(selected_chunks)),
        "chunks_skipped_existing_in_selected_range": int(skipped_existing),
        "chunks_completed_this_run": int(completed_now),
        "complete_chunks_total": int(complete_count),
        "incomplete_chunks_total": int(incomplete_count),
        "all_chunks_complete": bool(incomplete_count == 0),
        "embedding_chunks_manifest": str(settings.embedding_chunks_manifest_path()),
        "embedding_chunks_dir": str(settings.embedding_chunks_dir()),
        "embedding_model_metadata": str(embedding_metadata_path(settings)),
        "embedding_scope_path": str(settings.embedding_scope_path()),
        "embedding_scope_summary": str(settings.embedding_scope_summary_path()),
        "scope_summary_snapshot": scope_summary,
        "elapsed_seconds": round(time.time() - start_time, 2),
    }
    save_json(summary, settings.embedding_chunk_run_summary_path())
    print(f"[01a] Summary saved: {settings.embedding_chunk_run_summary_path()}", flush=True)
    if bool(settings.merge_embedding_chunks_after_generation) and incomplete_count == 0:
        merge_embedding_chunks(settings=settings, records=records)
    return summary


def merge_embedding_chunks(settings: Settings, records: pd.DataFrame | None = None) -> tuple[np.ndarray, dict]:
    if records is None:
        records, _ = load_or_create_embedding_scope(settings)
    chunks = chunk_ranges_by_index(len(records), int(settings.embedding_chunk_size))
    metadata, metadata_source = _load_embedding_metadata_for_merge(settings)
    if not metadata:
        remote_path = datastore_embedding_metadata_path(settings)
        raise FileNotFoundError(
            "Embedding metadata not found locally or in the configured datastore artifact root. "
            f"Checked local: {embedding_metadata_path(settings)}; remote: {remote_path}. "
            "Set SAFETY_RETRIEVAL_ARTIFACT_AZUREML_URI to the managed-batch-artifacts folder if needed."
        )
    print(f"[Merge] Loaded embedding metadata from: {metadata_source}", flush=True)

    dim = int(metadata.get("embedding_dimension") or 0)
    if dim <= 0:
        # Infer from first available local or datastore chunk.
        for ch in chunks:
            path = _resolve_chunk_file_for_read(settings, ch["chunk_index"])
            if artifact_exists(path):
                arr = load_numpy(path, allow_pickle=False)
                dim = int(arr.shape[1])
                break
    if dim <= 0:
        raise ValueError("Could not determine embedding dimension from metadata or chunks.")

    missing = []
    for ch in chunks:
        path = _resolve_chunk_file_for_read(settings, ch["chunk_index"])
        if not artifact_exists(path):
            missing.append({**ch, "chunk_file": str(path)})
    if missing:
        if bool(settings.fail_if_embedding_chunks_incomplete):
            raise RuntimeError(f"Cannot merge embeddings: {len(missing)} chunks are missing. First missing: {missing[:3]}")
        print(f"[Merge] Warning: {len(missing)} chunks are missing.", flush=True)

    ensure_dir(settings.embeddings_dir())
    out = np.lib.format.open_memmap(embedding_path(settings), mode="w+", dtype="float32", shape=(len(records), dim))
    for ch in chunks:
        path = _resolve_chunk_file_for_read(settings, ch["chunk_index"])
        arr = load_numpy(path, allow_pickle=False)
        expected_rows = int(ch["n_records"])
        if arr.ndim != 2 or int(arr.shape[0]) != expected_rows or int(arr.shape[1]) != dim:
            raise RuntimeError(
                f"Chunk {int(ch['chunk_index']):05d} has shape {getattr(arr, 'shape', None)}, "
                f"expected ({expected_rows}, {dim}). Source: {path}"
            )
        out[int(ch["start_row"]):int(ch["end_row"])] = arr.astype("float32", copy=False)
        print(
            f"[Merge] Copied chunk {int(ch['chunk_index']):05d} rows={int(ch['start_row']):,}:{int(ch['end_row']):,} from {path}",
            flush=True,
        )
    out.flush()
    np.save(embedding_shape_path(settings), np.array([len(records), dim], dtype="int64"))
    metadata.update(
        {
            "record_count": int(len(records)),
            "embedding_shape": [int(len(records)), int(dim)],
            "embedding_file": str(embedding_path(settings)),
            "embedding_shape_file": str(embedding_shape_path(settings)),
            "embedding_metadata_source": str(metadata_source),
            "merged_from_chunks": True,
        }
    )
    save_json(metadata, embedding_metadata_path(settings))
    save_json(metadata, settings.models_dir() / "embedding_model_metadata.json")
    return np.load(embedding_path(settings), mmap_mode="r"), metadata




def _normalized_text_series(records: pd.DataFrame, column: str) -> pd.Series:
    if column not in records.columns:
        return pd.Series("", index=records.index, dtype="string")
    return records[column].fillna("").astype(str).str.strip().str.lower()


def _raw_id_series(records: pd.DataFrame, column: str) -> pd.Series:
    if column not in records.columns:
        return pd.Series(np.nan, index=records.index)
    return pd.to_numeric(records[column], errors="coerce")


def _contains_any(records: pd.DataFrame, columns: list[str], phrases: list[str]) -> pd.Series:
    combined = pd.Series("", index=records.index, dtype="string")
    for column in columns:
        combined = combined.str.cat(_normalized_text_series(records, column), sep=" | ")
    mask = pd.Series(False, index=records.index)
    for phrase in phrases:
        mask = mask | combined.str.contains(phrase.lower(), regex=False, na=False)
    return mask


def _contains_regex(records: pd.DataFrame, columns: list[str], patterns: list[str]) -> pd.Series:
    combined = pd.Series("", index=records.index, dtype="string")
    for column in columns:
        combined = combined.str.cat(_normalized_text_series(records, column), sep=" | ")
    mask = pd.Series(False, index=records.index)
    for pattern in patterns:
        mask = mask | combined.str.contains(pattern, case=False, regex=True, na=False)
    return mask


def build_retrieval_index_masks(records: pd.DataFrame) -> dict[str, np.ndarray]:
    """Return purpose-specific index masks used by both local and AML index builds.

    The broad ``all_events`` index is intentionally not created. Each index has a
    business purpose so retrieval can gather clean evidence for the final agent
    response.
    """
    role = records["source_role"].fillna("").astype(str) if "source_role" in records.columns else pd.Series("", index=records.index)
    source_type = records["source_type"].fillna("").astype(str) if "source_type" in records.columns else pd.Series("", index=records.index)
    source_type_l = source_type.str.lower()

    raw_type_id = _raw_id_series(records, "raw_type_id")
    audit_label_columns = ["source_subtype", "category", "audit_type", "status", "title", "description"]
    safe_action = (
        role.isin(["safe_action"])
        | raw_type_id.eq(700)
        | _contains_regex(records, audit_label_columns, [r"(?<!un)\bsafe act\b", r"(?<!un)\bsafe action\b", r"observation\s*-\s*safe act"])
    )
    unsafe_action = (
        role.isin(["unsafe_action"])
        | raw_type_id.eq(701)
        | _contains_any(records, audit_label_columns, ["unsafe act", "unsafe action", "observation - unsafe act", "at risk act", "at-risk act"])
    )
    unsafe_condition = (
        role.isin(["unsafe_condition"])
        | raw_type_id.eq(702)
        | _contains_any(records, audit_label_columns, ["unsafe condition", "observation - unsafe condition", "at risk condition", "at-risk condition"])
    )
    safe_condition = (
        role.isin(["safe_condition"])
        | raw_type_id.eq(703)
        | _contains_regex(records, audit_label_columns, [r"(?<!un)\bsafe condition\b", r"observation\s*-\s*safe condition"])
    )

    safe_observation = role.eq("safe_observation") | safe_action | safe_condition
    unsafe_observation = role.eq("unsafe_observation") | unsafe_action | unsafe_condition
    any_audit = source_type_l.eq("audit") | role.isin(["audit_observation", "safe_observation", "unsafe_observation"])
    other_audit = any_audit & ~(safe_observation | unsafe_observation)

    corrective = role.isin(["corrective_action", "open_corrective_action", "overdue_corrective_action"]) | source_type_l.eq("task")

    masks = {
        "severe_injuries": role.eq("severe_injury").to_numpy(),
        "all_injuries": role.isin(["injury", "severe_injury"]).to_numpy(),
        "hazard_identifications": role.eq("hazard_identification").to_numpy(),
        "near_misses": role.eq("near_miss").to_numpy(),
        "audit_observations": any_audit.to_numpy(),
        "other_audit_observations": other_audit.to_numpy(),
        "safe_observations": safe_observation.to_numpy(),
        "unsafe_observations": unsafe_observation.to_numpy(),
        "safe_actions": safe_action.to_numpy(),
        "unsafe_actions": unsafe_action.to_numpy(),
        "safe_conditions": safe_condition.to_numpy(),
        "unsafe_conditions": unsafe_condition.to_numpy(),
        "corrective_actions": corrective.to_numpy(),
        "open_corrective_actions": role.eq("open_corrective_action").to_numpy(),
        "overdue_corrective_actions": role.eq("overdue_corrective_action").to_numpy(),
    }
    return masks

def _index_metadata(name: str, settings: Settings, embedding_metadata: dict, row_count: int) -> dict:
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


def build_indexes_from_chunks(settings: Settings) -> dict:
    """Merge chunked embeddings and build FAISS/BM25/theme artifacts."""
    start_time = time.time()
    ensure_dir(settings.indexes_dir())
    ensure_dir(settings.bm25_dir())
    ensure_dir(settings.models_dir())

    records, _ = load_or_create_embedding_scope(settings)
    vectors, embedding_metadata = merge_embedding_chunks(settings, records=records)
    print(
        f"[01b] Merged embeddings shape={vectors.shape}; model={embedding_metadata.get('embedding_model_name')} "
        f"fallback_used={embedding_metadata.get('used_fallback_embedding_model')}",
        flush=True,
    )

    print("[01b] Discovering/reusing risk themes on embedding scope...", flush=True)
    records, theme_model = discover_themes(records, vectors, settings)
    save_theme_model(theme_model, settings)
    profiles = build_theme_profiles(records, vectors, settings)
    print(f"[01b] Theme profiles: {len(profiles):,}", flush=True)

    records.to_pickle(settings.enriched_knowledge_base_path())
    records.to_csv(settings.enriched_knowledge_base_path().with_suffix(".csv.gz"), index=False, compression="gzip")

    masks = build_retrieval_index_masks(records)
    configured_index_names = list(getattr(settings, "retrieval_index_names", masks.keys()))
    masks = {name: masks[name] for name in configured_index_names if name in masks}

    index_summaries = []
    bm25_subset_summaries = []
    for name, mask in masks.items():
        row_count = int(mask.sum())
        print(f"[01b] Building FAISS index: {name} rows={row_count:,}", flush=True)
        meta = build_and_save_subset_index(
            vectors,
            mask,
            settings,
            name,
            metadata=_index_metadata(name, settings, embedding_metadata, row_count),
        )
        index_summaries.append(meta)

        bm25_meta = save_bm25_subset(
            settings.bm25_dir(),
            name,
            np.flatnonzero(mask),
            metadata={
                "name": name,
                "row_count": int(row_count),
                "source": "safety_embedding_scope_with_themes",
                "embedding_scope_summary_file": str(settings.embedding_scope_summary_path()),
            },
        )
        bm25_subset_summaries.append(bm25_meta)

    print("[01b] Building BM25 keyword index over embedding scope...", flush=True)
    bm25_store = build_bm25_store(records, settings)
    save_bm25_store(bm25_store, settings.bm25_dir())
    print(
        f"[01b] BM25 vocabulary size: {bm25_store.metadata.get('vocabulary_size'):,}; "
        f"rows={bm25_store.metadata.get('row_count'):,}",
        flush=True,
    )

    summary = {
        "script": "01b_build_indexes_from_chunks.py",
        "record_count": int(len(records)),
        "embedding_shape": [int(vectors.shape[0]), int(vectors.shape[1])],
        "embedding_metadata": embedding_metadata,
        "faiss_indexes_dir": str(settings.indexes_dir()),
        "bm25_indexes_dir": str(settings.bm25_dir()),
        "embeddings_dir": str(settings.embeddings_dir()),
        "index_summaries": index_summaries,
        "bm25_subset_summaries": bm25_subset_summaries,
        "theme_profile_count": int(len(profiles)),
        "elapsed_seconds": round(time.time() - start_time, 2),
    }
    save_json(summary, settings.output_dir / "01b_build_indexes_from_chunks_summary.json")
    print(f"[01b] Saved summary: {settings.output_dir / '01b_build_indexes_from_chunks_summary.json'}", flush=True)
    return summary
