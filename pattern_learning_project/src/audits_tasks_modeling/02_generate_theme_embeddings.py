#!/usr/bin/env python3
"""Generate embeddings for each source family.

Default backend uses a free local sentence-transformer. If unavailable, falls
back to TF-IDF + SVD embeddings so clustering can still run without an API key.
"""
from __future__ import annotations

try:
    import config as cfg
    from theme_utils import ProgressLogger, ensure_dir, read_csv, save_json, write_csv
except ImportError:  # pragma: no cover
    from . import config as cfg
    from .theme_utils import ProgressLogger, ensure_dir, read_csv, save_json, write_csv

import pickle
import traceback
from pathlib import Path

import numpy as np


def _normalize(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def _sentence_transformer_embed(texts: list[str], log: ProgressLogger) -> tuple[np.ndarray, dict]:
    from sentence_transformers import SentenceTransformer

    kwargs = {}
    if getattr(cfg, "EMBEDDING_DEVICE", None):
        kwargs["device"] = cfg.EMBEDDING_DEVICE
    log.log(f"loading sentence-transformer model: {cfg.SENTENCE_TRANSFORMER_MODEL}")
    model = SentenceTransformer(cfg.SENTENCE_TRANSFORMER_MODEL, **kwargs)
    log.log(f"encoding {len(texts):,} records; batch_size={cfg.EMBEDDING_BATCH_SIZE}")
    emb = model.encode(
        texts,
        batch_size=int(cfg.EMBEDDING_BATCH_SIZE),
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=bool(cfg.EMBEDDING_NORMALIZE),
    ).astype(np.float32)
    if bool(cfg.EMBEDDING_NORMALIZE):
        emb = _normalize(emb)
    return emb, {"backend_used": "sentence_transformers", "model": cfg.SENTENCE_TRANSFORMER_MODEL}


def _tfidf_svd_embed(texts: list[str], family: str, log: ProgressLogger) -> tuple[np.ndarray, dict]:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import Normalizer

    stop_words = sorted(list(getattr(cfg, "CUSTOM_STOPWORDS", set())))
    n_components = min(int(cfg.SVD_COMPONENTS), max(2, len(texts) - 1))
    log.log(
        f"fitting TF-IDF + SVD fallback for {family}; "
        f"texts={len(texts):,}; n_components={n_components}"
    )
    vectorizer = TfidfVectorizer(
        max_features=int(cfg.TFIDF_MAX_FEATURES),
        ngram_range=tuple(cfg.TFIDF_NGRAM_RANGE),
        min_df=int(cfg.TFIDF_MIN_DF),
        max_df=float(cfg.TFIDF_MAX_DF),
        stop_words="english",
    )
    svd = TruncatedSVD(n_components=n_components, random_state=int(cfg.RANDOM_STATE))
    normalizer = Normalizer(copy=False)
    pipe = Pipeline([("tfidf", vectorizer), ("svd", svd), ("norm", normalizer)])
    emb = pipe.fit_transform(texts).astype(np.float32)
    model_path = cfg.THEME_EMBEDDING_DIR / f"tfidf_svd_model_{family}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(pipe, f)
    return emb, {"backend_used": "tfidf_svd", "model_path": str(model_path), "n_components": n_components}


def embed_family(family: str, log: ProgressLogger) -> dict:
    input_file = cfg.THEME_INPUT_FILE_BY_FAMILY[family]
    if not input_file.exists():
        raise FileNotFoundError(f"Missing theme input file for {family}: {input_file}. Run 01_prepare_theme_text.py first.")
    df = read_csv(input_file)
    if len(df) == 0:
        log.log(f"family={family} has no rows; writing empty metadata")
        np.save(cfg.EMBEDDING_FILE_BY_FAMILY[family], np.empty((0, 0), dtype=np.float32))
        write_csv(df, cfg.EMBEDDING_META_FILE_BY_FAMILY[family])
        return {"family": family, "row_count": 0, "status": "empty"}

    texts = df["theme_text"].fillna("").astype(str).tolist()
    backend = str(getattr(cfg, "EMBEDDING_BACKEND", "sentence_transformers")).lower()
    backend_info = {}
    try:
        if backend == "sentence_transformers":
            emb, backend_info = _sentence_transformer_embed(texts, log)
        elif backend == "tfidf_svd":
            emb, backend_info = _tfidf_svd_embed(texts, family, log)
        else:
            raise ValueError(f"Unsupported EMBEDDING_BACKEND={backend!r}")
    except Exception as exc:
        log.log("sentence-transformer backend failed; falling back to TF-IDF + SVD")
        error_path = cfg.THEME_EMBEDDING_DIR / f"embedding_error_{family}.txt"
        error_path.write_text(traceback.format_exc(), encoding="utf-8")
        emb, backend_info = _tfidf_svd_embed(texts, family, log)
        backend_info["fallback_reason"] = repr(exc)
        backend_info["error_traceback_path"] = str(error_path)

    if bool(getattr(cfg, "EMBEDDING_NORMALIZE", True)):
        emb = _normalize(emb)
    emb_file = cfg.EMBEDDING_FILE_BY_FAMILY[family]
    np.save(emb_file, emb)

    meta_cols = [
        "theme_row_id", "event_id", "source_family", "raw_source_family", "source_type", "event_kind",
        "event_date", "location_id", "location_path", "title", "category", "status",
        "audit_signal_type", "audit_cluster_family", "audit_cluster_eligible", "audit_cluster_exclusion_reason",
        "audit_specific_text_length", "audit_has_risk_keyword", "audit_has_positive_keyword", "audit_is_routine_pattern",
        "audit_is_generic_title", "review_priority", "theme_text_length",
    ]
    meta_cols = [c for c in meta_cols if c in df.columns]
    meta = df[meta_cols].copy()
    meta["embedding_row_id"] = range(len(meta))
    write_csv(meta, cfg.EMBEDDING_META_FILE_BY_FAMILY[family])
    log.log(f"saved embeddings for {family}: shape={emb.shape} -> {emb_file}")
    return {
        "family": family,
        "row_count": int(len(df)),
        "embedding_shape": list(emb.shape),
        "embedding_file": str(emb_file),
        "metadata_file": str(cfg.EMBEDDING_META_FILE_BY_FAMILY[family]),
        **backend_info,
    }


def main() -> None:
    log = ProgressLogger("02_generate_theme_embeddings")
    ensure_dir(cfg.THEME_EMBEDDING_DIR)
    results = []
    for family in cfg.SOURCE_FAMILIES:
        log.log(f"processing family={family}")
        results.append(embed_family(family, log))
    save_json({"families": results}, cfg.EMBEDDING_SUMMARY_FILE)
    log.done("embedding generation complete")


if __name__ == "__main__":
    main()
