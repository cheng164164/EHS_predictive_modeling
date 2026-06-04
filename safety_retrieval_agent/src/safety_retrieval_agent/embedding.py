"""Embedding backends for local semantic search.

Default recommendation:
- Use BAAI/bge-m3 through sentence-transformers for stable dense embeddings.
- Automatically fall back to Qwen/Qwen3-Embedding-0.6B if the primary model
  fails to load or fails on the first encode batch.
- Save metadata for the actual model used so FAISS indexes are always queried
  with the same embedding model that built them.

The import of heavy optional dependencies is delayed until runtime so the rest of
this project can be inspected or syntax-checked without installing them first.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Iterable

import numpy as np

from .config import Settings
from .utils import clean_text_value, normalize_vectors


class TextEmbedder:
    """Wrapper around local/free embedding models used for semantic search.

    Parameters
    ----------
    settings:
        Project settings object.
    model_name:
        Optional explicit model to load. Used at query time to force the same
        model that was recorded in FAISS index metadata.
    backend:
        Optional explicit backend. Currently ``sentence_transformers`` is the
        recommended backend for both BGE-M3 and Qwen3 fallback.
    enable_fallback:
        Optional override. When loading a model from index metadata, set this to
        False so prediction fails rather than silently using a different model.
    """

    def __init__(
        self,
        settings: Settings,
        model_name: str | None = None,
        backend: str | None = None,
        enable_fallback: bool | None = None,
        allow_fallback: bool | None = None,
        query_instruction: str | None = None,
    ):
        self.settings = settings
        self.requested_backend = backend or settings.embedding_backend
        self.backend = self._resolve_backend(self.requested_backend)

        self.primary_model_name = str(model_name or settings.embedding_model_name)
        self.fallback_model_name = str(getattr(settings, "embedding_fallback_model_name", "") or "").strip()
        self.fallback_backend_requested = str(getattr(settings, "embedding_fallback_backend", "sentence_transformers") or "sentence_transformers")
        self.fallback_backend = self._resolve_backend(self.fallback_backend_requested)
        fallback_override = enable_fallback if enable_fallback is not None else allow_fallback
        self.enable_fallback = bool(getattr(settings, "enable_embedding_fallback", True) if fallback_override is None else fallback_override)
        self.query_instruction = settings.query_instruction if query_instruction is None else str(query_instruction)

        # If the caller forced a specific model from index metadata, do not fall
        # back to another model by default. Mixed embedding models invalidate
        # FAISS similarity scores.
        if model_name is not None and enable_fallback is None and allow_fallback is None:
            self.enable_fallback = False

        self.model_name = self.primary_model_name
        self.actual_backend = self.backend
        self.used_fallback = False
        self.primary_load_error: str | None = None
        self.primary_encode_error: str | None = None
        self.embedding_dimension: int | None = None
        self.model = self._load_with_fallback()

    def _resolve_backend(self, requested: str | None) -> str:
        requested = (requested or "auto").strip().lower().replace("-", "_")
        if requested in {"auto", "sentence_transformers", "sentence_transformer", "st"}:
            return "sentence_transformers"
        if requested in {"flagembedding", "flag_embedding", "bge_m3_flag"}:
            return "flagembedding"
        raise ValueError(
            "embedding_backend must be 'auto', 'sentence_transformers', or 'flagembedding'. "
            f"Got: {requested!r}"
        )

    def _load_sentence_transformer(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for embedding_backend='sentence_transformers'. "
                "Install it with: pip install sentence-transformers"
            ) from exc

        kwargs = {}
        device = str(getattr(self.settings, "embedding_device", "") or "").strip()
        if device:
            kwargs["device"] = device
        if bool(getattr(self.settings, "trust_remote_code", True)):
            kwargs["trust_remote_code"] = True

        try:
            model = SentenceTransformer(model_name, **kwargs)
        except TypeError:
            # Older sentence-transformers versions may not accept
            # trust_remote_code/device in the constructor. Retry with a minimal
            # call so the error is not caused by wrapper kwargs.
            minimal_kwargs = {}
            if device:
                minimal_kwargs["device"] = device
            try:
                model = SentenceTransformer(model_name, **minimal_kwargs)
            except TypeError:
                model = SentenceTransformer(model_name)

        max_len = int(getattr(self.settings, "embedding_max_length", 512))
        if max_len > 0 and hasattr(model, "max_seq_length"):
            try:
                model.max_seq_length = max_len
            except Exception:
                pass
        return model

    def _load_flagembedding(self, model_name: str):
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as exc:
            raise ImportError(
                "FlagEmbedding is required for embedding_backend='flagembedding'. "
                "Install it with: pip install FlagEmbedding"
            ) from exc
        try:
            return BGEM3FlagModel(model_name, use_fp16=bool(self.settings.use_fp16))
        except TypeError as exc:
            raise RuntimeError(
                "Failed to load BGE-M3 with FlagEmbedding. This is usually a "
                "FlagEmbedding/transformers compatibility issue. Use "
                "embedding_backend='sentence_transformers' or the default auto backend. "
                "Original error: " + str(exc)
            ) from exc

    def _load_one_model(self, model_name: str, backend: str):
        if backend == "sentence_transformers":
            return self._load_sentence_transformer(model_name)
        if backend == "flagembedding":
            return self._load_flagembedding(model_name)
        raise ValueError(f"Unsupported embedding backend: {backend}")

    def _load_with_fallback(self):
        try:
            print(
                f"[Embedding] Loading primary model: {self.primary_model_name} "
                f"backend={self.backend}",
                flush=True,
            )
            self.model_name = self.primary_model_name
            self.actual_backend = self.backend
            self.used_fallback = False
            return self._load_one_model(self.primary_model_name, self.backend)
        except Exception as exc:
            self.primary_load_error = repr(exc)
            if not self.enable_fallback or not self.fallback_model_name or self.fallback_model_name == self.primary_model_name:
                raise
            print(
                "[Embedding] Primary model failed to load. "
                f"Falling back to {self.fallback_model_name} backend={self.fallback_backend}.\n"
                f"[Embedding] Primary load error: {exc}",
                flush=True,
            )
            self.model_name = self.fallback_model_name
            self.actual_backend = self.fallback_backend
            self.used_fallback = True
            return self._load_one_model(self.fallback_model_name, self.fallback_backend)

    def disable_fallback(self) -> None:
        """Prevent later batches from silently switching models.

        The index builder calls this after the first successful batch. If a later
        batch fails, the run should stop rather than mixing BGE and Qwen vectors
        inside one FAISS index.
        """
        self.enable_fallback = False

    def _encode_with_current_model(self, texts: list[str], is_query: bool, batch_size: int) -> np.ndarray:
        if self.actual_backend == "flagembedding":
            payload = texts
            if is_query and self.query_instruction:
                payload = [self.query_instruction + t for t in payload]
            result = self.model.encode(
                payload,
                batch_size=batch_size,
                max_length=int(self.settings.embedding_max_length),
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
            vectors = result["dense_vecs"] if isinstance(result, dict) else result
            return normalize_vectors(np.asarray(vectors, dtype="float32"))

        if self.actual_backend == "sentence_transformers":
            payload = texts
            if is_query and self.query_instruction:
                payload = [self.query_instruction + t for t in payload]
            try:
                vectors = self.model.encode(
                    payload,
                    batch_size=batch_size,
                    show_progress_bar=bool(self.settings.show_progress),
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                )
            except TypeError:
                # Compatibility fallback for older sentence-transformers builds.
                vectors = self.model.encode(
                    payload,
                    batch_size=batch_size,
                    show_progress_bar=bool(self.settings.show_progress),
                    convert_to_numpy=True,
                )
            return normalize_vectors(np.asarray(vectors, dtype="float32"))

        raise ValueError(f"Unsupported embedding backend: {self.actual_backend}")

    def encode(self, texts: Iterable[str], is_query: bool = False, batch_size: int | None = None) -> np.ndarray:
        clean_texts = [clean_text_value(t) for t in texts]
        if not clean_texts:
            return np.empty((0, 0), dtype="float32")
        batch_size = int(batch_size or self.settings.embedding_batch_size)

        try:
            vectors = self._encode_with_current_model(clean_texts, is_query=is_query, batch_size=batch_size)
        except Exception as exc:
            self.primary_encode_error = repr(exc)
            if self.used_fallback or not self.enable_fallback or not self.fallback_model_name or self.fallback_model_name == self.model_name:
                raise
            print(
                "[Embedding] Primary model failed during encoding. "
                f"Falling back to {self.fallback_model_name} backend={self.fallback_backend}.\n"
                f"[Embedding] Primary encode error: {exc}",
                flush=True,
            )
            self.model_name = self.fallback_model_name
            self.actual_backend = self.fallback_backend
            self.used_fallback = True
            self.model = self._load_one_model(self.fallback_model_name, self.fallback_backend)
            vectors = self._encode_with_current_model(clean_texts, is_query=is_query, batch_size=batch_size)

        if vectors.ndim != 2:
            raise ValueError(f"Embedding model returned non-2D vectors with shape={vectors.shape}")
        self.embedding_dimension = int(vectors.shape[1])
        return vectors.astype("float32", copy=False)

    def metadata(self) -> dict:
        return {
            "embedding_model_name": self.model_name,
            "embedding_backend": self.actual_backend,
            "primary_embedding_model_name": self.primary_model_name,
            "primary_embedding_backend": self.backend,
            "fallback_embedding_model_name": self.fallback_model_name or None,
            "fallback_embedding_backend": self.fallback_backend,
            "enable_embedding_fallback": bool(self.enable_fallback),
            "used_fallback_embedding_model": bool(self.used_fallback),
            "primary_load_error": self.primary_load_error,
            "primary_encode_error": self.primary_encode_error,
            "embedding_dimension": self.embedding_dimension,
            "query_instruction": self.query_instruction,
            "embedding_max_length": int(self.settings.embedding_max_length),
            "embedding_batch_size": int(self.settings.embedding_batch_size),
            "settings": asdict(self.settings),
        }
