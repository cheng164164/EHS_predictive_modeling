"""MVP1 retrieval agent.

The agent loads local FAISS and/or BM25 indexes and produces evidence-backed
outputs for:
1. Risk pattern / theme classification
2. Historical severe-injury similarity
3. Similar historical event recall
4. Risk factor extraction
5. Recommended prevention actions
6. Missing-information prompts

Retrieval mode is controlled in config.py:
- retrieval_mode = "faiss"  -> semantic vector search only
- retrieval_mode = "bm25"   -> keyword/BM25 search only
- retrieval_mode = "hybrid" -> FAISS + BM25 with reciprocal-rank fusion
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from .bm25_store import load_bm25_store, load_bm25_subset, search_bm25
from .config import Settings, get_settings
from .embedding import TextEmbedder
from .extraction import extract_keyphrases, missing_information_prompt, recommend_from_actions
from .faiss_store import load_index_bundle, search_index
from .local_llm import LocalLLMResponder
from .theme import load_theme_model
from .utils import clean_text_value, compress_json_field, preview, similarity_band


_INDEX_NAMES = [
    "all_events",
    "severe_injuries",
    "all_injuries",
    "leading_events",
    "corrective_actions",
    "safe_observations",
    "unsafe_observations",
]


class SafetyRetrievalAgent:
    """Local safety retrieval agent with configurable retrieval mode.

    Important model-safety rule:
    When FAISS is used, the query embedding model is loaded from FAISS index
    metadata when available. This prevents invalid results if the index was built
    with the Qwen3 fallback but config.py still lists BGE-M3 as the primary model.
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.retrieval_mode = self._normalize_retrieval_mode(self.settings.retrieval_mode)

        kb_path = self.settings.enriched_knowledge_base_path()
        if not kb_path.exists():
            kb_path = self.settings.knowledge_base_path()
        if not kb_path.exists():
            raise FileNotFoundError(
                f"Knowledge base not found: {kb_path}. Run scripts/00_prepare_knowledge_base.py and scripts/01_build_faiss_indexes.py first."
            )
        self.records = pd.read_pickle(kb_path).reset_index(drop=True)
        self.records["row_id"] = np.arange(len(self.records))
        self.theme_profiles = self._load_theme_profiles()
        self.theme_model = load_theme_model(self.settings)

        self.indexes: dict[str, tuple[Any, np.ndarray, dict]] = {}
        self.bm25_subsets: dict[str, tuple[np.ndarray, dict]] = {}
        self.bm25_store = None
        self.embedder: TextEmbedder | None = None
        self.index_embedding_metadata: dict = {}
        self.llm_responder = LocalLLMResponder(self.settings) if bool(getattr(self.settings, "enable_llm_response", True)) else None

        if self._uses_faiss:
            self._load_faiss_indexes()
            self.index_embedding_metadata = self._resolve_index_embedding_metadata()
            self.embedder = self._load_query_embedder()

        if self._uses_bm25:
            self._load_bm25_indexes()

    @staticmethod
    def _normalize_retrieval_mode(value: str | None) -> str:
        mode = (value or "hybrid").strip().lower().replace("-", "_")
        aliases = {
            "vector": "faiss",
            "semantic": "faiss",
            "keyword": "bm25",
            "keywords": "bm25",
            "bm25_only": "bm25",
            "faiss_only": "faiss",
            "hybrid_search": "hybrid",
        }
        mode = aliases.get(mode, mode)
        if mode not in {"faiss", "bm25", "hybrid"}:
            raise ValueError("retrieval_mode must be 'faiss', 'bm25', or 'hybrid'.")
        return mode

    @property
    def _uses_faiss(self) -> bool:
        return self.retrieval_mode in {"faiss", "hybrid"}

    @property
    def _uses_bm25(self) -> bool:
        return self.retrieval_mode in {"bm25", "hybrid"}

    def _load_faiss_indexes(self) -> None:
        for name in _INDEX_NAMES:
            if (self.settings.indexes_dir() / f"{name}.faiss").exists():
                self.indexes[name] = load_index_bundle(self.settings.indexes_dir(), name)
        if not self.indexes:
            raise FileNotFoundError(
                f"No FAISS indexes found in {self.settings.indexes_dir()}. Run scripts/01_build_faiss_indexes.py first, "
                "or set retrieval_mode='bm25'."
            )

    def _load_bm25_indexes(self) -> None:
        self.bm25_store = load_bm25_store(self.settings.bm25_dir())
        for name in _INDEX_NAMES:
            row_path = self.settings.bm25_dir() / f"{name}_row_ids.npy"
            if row_path.exists():
                self.bm25_subsets[name] = load_bm25_subset(self.settings.bm25_dir(), name)
        if not self.bm25_subsets:
            raise FileNotFoundError(
                f"No BM25 subset files found in {self.settings.bm25_dir()}. Run scripts/01_build_faiss_indexes.py first, "
                "or set retrieval_mode='faiss'."
            )

    def _load_theme_profiles(self) -> pd.DataFrame:
        path = self.settings.theme_profiles_path()
        if path.exists():
            return pd.read_pickle(path)
        return pd.DataFrame(columns=["risk_theme_id", "risk_theme_name"])

    @staticmethod
    def _extract_embedding_metadata(index_metadata: dict | None) -> dict:
        if not index_metadata:
            return {}
        nested = index_metadata.get("embedding")
        if isinstance(nested, dict) and nested.get("embedding_model_name"):
            return nested
        if index_metadata.get("embedding_model_name"):
            return {
                "embedding_model_name": index_metadata.get("embedding_model_name"),
                "embedding_backend": index_metadata.get("embedding_backend", "sentence_transformers"),
                "used_fallback_embedding_model": index_metadata.get("used_fallback_embedding_model"),
                "embedding_dimension": index_metadata.get("embedding_dimension"),
            }
        return {}

    def _resolve_index_embedding_metadata(self) -> dict:
        """Read and validate embedding metadata from loaded FAISS indexes."""
        metas: list[dict] = []
        for _name, (_index, _row_ids, metadata) in self.indexes.items():
            meta = self._extract_embedding_metadata(metadata)
            if meta:
                metas.append(meta)
        if not metas:
            return {}

        first = metas[0]
        first_model = str(first.get("embedding_model_name") or "")
        first_backend = str(first.get("embedding_backend") or "sentence_transformers")
        for meta in metas[1:]:
            model = str(meta.get("embedding_model_name") or "")
            backend = str(meta.get("embedding_backend") or "sentence_transformers")
            if model != first_model or backend != first_backend:
                raise ValueError(
                    "Loaded FAISS indexes were built with inconsistent embedding models/backends. "
                    f"First={first_model}/{first_backend}; found={model}/{backend}. "
                    "Delete the FAISS index folder and rebuild with scripts/01_build_faiss_indexes.py."
                )
        return first

    def _load_query_embedder(self) -> TextEmbedder:
        if self.index_embedding_metadata.get("embedding_model_name"):
            model_name = str(self.index_embedding_metadata["embedding_model_name"])
            backend = str(self.index_embedding_metadata.get("embedding_backend") or "sentence_transformers")
            print(
                f"[Agent] Loading query embedding model from FAISS metadata: {model_name} backend={backend}",
                flush=True,
            )
            return TextEmbedder(
                self.settings,
                model_name=model_name,
                backend=backend,
                enable_fallback=False,
                query_instruction=self.settings.query_instruction,
            )
        print(
            "[Agent] No embedding metadata found in indexes. Loading model from config.py. "
            "Rebuild indexes if you previously changed embedding models.",
            flush=True,
        )
        return TextEmbedder(self.settings)

    def analyze_event(
        self,
        query_text: str,
        site: str | None = None,
        department: str | None = None,
        source_type: str | None = None,
        event_id: str | None = None,
        top_k_events: int | None = None,
    ) -> dict:
        """Analyze a new or existing safety event description."""
        query_text = clean_text_value(query_text)
        if not query_text:
            raise ValueError("query_text is required")

        query_vector = self._encode_query(query_text) if self._uses_faiss else None
        top_k_events = top_k_events or self.settings.top_k_similar_events

        similar_events = self._retrieve("all_events", query_text, query_vector, top_k_events + 5, exclude_event_id=event_id)
        severe_matches = self._retrieve("severe_injuries", query_text, query_vector, self.settings.top_k_severe_injuries, exclude_event_id=event_id)
        injury_matches = self._retrieve("all_injuries", query_text, query_vector, self.settings.top_k_severe_injuries, exclude_event_id=event_id)
        action_matches = self._retrieve("corrective_actions", query_text, query_vector, self.settings.top_k_corrective_actions, exclude_event_id=event_id)
        safe_matches = self._retrieve("safe_observations", query_text, query_vector, self.settings.top_k_safe_practices, exclude_event_id=event_id)

        theme = self._classify_theme(query_vector, similar_events)
        top_severe_score = self._top_retrieval_score(severe_matches)
        top_severe_faiss_score = self._top_faiss_score(severe_matches)
        severe_band = self._severe_similarity_band(severe_matches, top_severe_faiss_score)

        evidence_texts = []
        for df in [similar_events, severe_matches, action_matches, safe_matches]:
            if df is not None and not df.empty and "matched_retrieval_text" in df.columns:
                evidence_texts.extend(df["matched_retrieval_text"].dropna().astype(str).head(20).tolist())
        risk_factors = extract_keyphrases(query_text, evidence_texts, top_n=12)
        prevention_actions = recommend_from_actions(action_matches, safe_matches, max_actions=self.settings.top_k_corrective_actions)
        missing_prompts = missing_information_prompt(
            query_text=query_text,
            site=site,
            department=department,
            source_type=source_type,
            detected_theme=theme.get("risk_theme_name"),
            severe_similarity_band=severe_band,
        )
        theme_profile = self._theme_profile(theme.get("risk_theme_id"))

        result = {
            "query": {
                "event_id": event_id,
                "source_type": source_type,
                "site": site,
                "department": department,
                "text_preview": preview(query_text, 500),
            },
            "runtime": {
                "retrieval_mode": self.retrieval_mode,
                "embedding_model_name": self.embedder.model_name if self.embedder is not None else None,
                "embedding_backend": self.embedder.actual_backend if self.embedder is not None else None,
                "index_embedding_model_name": self.index_embedding_metadata.get("embedding_model_name"),
                "index_embedding_backend": self.index_embedding_metadata.get("embedding_backend"),
                "bm25_algorithm": self.bm25_store.metadata.get("bm25_algorithm") if self.bm25_store is not None else None,
                "llm_response_enabled": bool(getattr(self.settings, "enable_llm_response", True)),
                "llm_model_name": getattr(self.settings, "llm_model_name", None),
            },
            "risk_pattern_classification": theme,
            "theme_profile": theme_profile,
            "historical_severe_injury_similarity": {
                "top_score": top_severe_score,
                "top_faiss_cosine_score": top_severe_faiss_score,
                "similarity_band": severe_band,
                "score_note": self._score_note(),
                "matches": self._records_for_json(severe_matches),
            },
            "historical_injury_similarity": {
                "matches": self._records_for_json(injury_matches),
            },
            "similar_historical_event_recall": {
                "matches": self._records_for_json(similar_events.head(top_k_events)),
                "source_role_counts": similar_events["matched_source_role"].value_counts().to_dict() if not similar_events.empty and "matched_source_role" in similar_events.columns else {},
            },
            "risk_factor_extraction": risk_factors,
            "recommended_prevention_actions": prevention_actions,
            "missing_information_prompt": missing_prompts,
        }

        result["llm_final_response"] = self._generate_llm_final_response(result)
        return result

    def _generate_llm_final_response(self, result: dict) -> dict:
        """Use a local/free LLM to organize the final user-facing response.

        Retrieval output remains available in the structured fields. This method
        adds a concise response_text generated from those fields only.
        """
        if self.llm_responder is None:
            return {
                "status": "disabled",
                "model_name": None,
                "response_text": "LLM response generation is disabled in config.py.",
                "error": None,
            }
        generation = self.llm_responder.generate(result)
        return {
            "status": generation.status,
            "model_name": generation.model_name,
            "response_text": generation.response_text,
            "error": generation.error,
            "prompt_chars": generation.prompt_chars,
            "note": (
                "Generated from retrieved historical evidence only. Review by an EHS professional is still required; "
                "the agent does not predict that an injury will occur."
            ),
        }

    def _encode_query(self, query_text: str) -> np.ndarray:
        if self.embedder is None:
            raise RuntimeError("FAISS retrieval requested but query embedder was not loaded.")
        return self.embedder.encode([query_text], is_query=True)

    def _retrieve(
        self,
        index_name: str,
        query_text: str,
        query_vector: np.ndarray | None,
        top_k: int,
        exclude_event_id: str | None = None,
    ) -> pd.DataFrame:
        if self.retrieval_mode == "faiss":
            return self._faiss_search(index_name, query_vector, top_k, exclude_event_id=exclude_event_id)
        if self.retrieval_mode == "bm25":
            return self._bm25_search(index_name, query_text, top_k, exclude_event_id=exclude_event_id)
        return self._hybrid_search(index_name, query_text, query_vector, top_k, exclude_event_id=exclude_event_id)

    def _faiss_search(self, index_name: str, query_vector: np.ndarray | None, top_k: int, exclude_event_id: str | None = None) -> pd.DataFrame:
        if query_vector is None or index_name not in self.indexes:
            return pd.DataFrame()
        index, row_ids, _metadata = self.indexes[index_name]
        result_lists = search_index(index, row_ids, query_vector, top_k=top_k)
        results = result_lists[0] if result_lists else []
        return self._rows_from_ranked_ids(
            [(r.row_id, r.rank, {"faiss_score": float(r.score), "similarity_score": float(r.score), "retrieval_method": "faiss"}) for r in results],
            exclude_event_id=exclude_event_id,
        )

    def _bm25_search(self, index_name: str, query_text: str, top_k: int, exclude_event_id: str | None = None) -> pd.DataFrame:
        if self.bm25_store is None or index_name not in self.bm25_subsets:
            return pd.DataFrame()
        row_ids, _metadata = self.bm25_subsets[index_name]
        results = search_bm25(self.bm25_store, query_text, top_k=top_k, row_ids=row_ids, settings=self.settings)
        return self._rows_from_ranked_ids(
            [(r.row_id, r.rank, {"bm25_score": float(r.score), "similarity_score": float(r.score), "retrieval_method": "bm25"}) for r in results],
            exclude_event_id=exclude_event_id,
        )

    def _hybrid_search(self, index_name: str, query_text: str, query_vector: np.ndarray | None, top_k: int, exclude_event_id: str | None = None) -> pd.DataFrame:
        faiss_k = max(int(self.settings.faiss_candidate_k), int(top_k))
        bm25_k = max(int(self.settings.bm25_candidate_k), int(top_k))
        faiss_df = self._faiss_search(index_name, query_vector, faiss_k, exclude_event_id=exclude_event_id)
        bm25_df = self._bm25_search(index_name, query_text, bm25_k, exclude_event_id=exclude_event_id)
        return self._fuse_results(faiss_df, bm25_df, top_k=top_k)

    def _fuse_results(self, faiss_df: pd.DataFrame, bm25_df: pd.DataFrame, top_k: int) -> pd.DataFrame:
        if (faiss_df is None or faiss_df.empty) and (bm25_df is None or bm25_df.empty):
            return pd.DataFrame()
        rrf_k = float(self.settings.hybrid_rrf_k)
        candidates: dict[int, dict] = {}

        def add(df: pd.DataFrame, source: str) -> None:
            if df is None or df.empty:
                return
            for _, row in df.iterrows():
                rid = int(row["matched_row_id"])
                item = candidates.setdefault(rid, {"matched_row_id": rid, "faiss_rank": None, "bm25_rank": None, "faiss_score": None, "bm25_score": None, "hybrid_score": 0.0})
                rank = int(row["rank"])
                if source == "faiss":
                    item["faiss_rank"] = rank
                    item["faiss_score"] = float(row.get("faiss_score", row.get("similarity_score", np.nan)))
                else:
                    item["bm25_rank"] = rank
                    item["bm25_score"] = float(row.get("bm25_score", row.get("similarity_score", np.nan)))
                item["hybrid_score"] += 1.0 / (rrf_k + rank)

        add(faiss_df, "faiss")
        add(bm25_df, "bm25")
        if not candidates:
            return pd.DataFrame()
        ranked = sorted(candidates.values(), key=lambda x: x["hybrid_score"], reverse=True)[: max(1, int(top_k))]
        ranked_rows = []
        for rank, item in enumerate(ranked, start=1):
            extras = {
                "hybrid_score": float(item["hybrid_score"]),
                "faiss_score": item["faiss_score"],
                "bm25_score": item["bm25_score"],
                "faiss_rank": item["faiss_rank"],
                "bm25_rank": item["bm25_rank"],
                "similarity_score": float(item["hybrid_score"]),
                "retrieval_method": "hybrid_rrf",
            }
            ranked_rows.append((int(item["matched_row_id"]), rank, extras))
        return self._rows_from_ranked_ids(ranked_rows, exclude_event_id=None)

    def _rows_from_ranked_ids(self, ranked_items: list[tuple[int, int, dict]], exclude_event_id: str | None = None) -> pd.DataFrame:
        rows = []
        for row_id, rank, extras in ranked_items:
            rec = self.records.iloc[int(row_id)].to_dict()
            if exclude_event_id and str(rec.get("event_id")) == str(exclude_event_id):
                continue
            row = {
                "rank": int(rank),
                "matched_row_id": int(row_id),
                **extras,
            }
            for key, value in rec.items():
                row[f"matched_{key}"] = value
            rows.append(row)
        return pd.DataFrame(rows)

    def _classify_theme(self, query_vector: np.ndarray | None, similar_events: pd.DataFrame) -> dict:
        if query_vector is not None and self.theme_model is not None:
            label = int(self.theme_model.predict(query_vector)[0])
            theme_id = f"RT{label + 1:04d}"
            profile = self._theme_profile(theme_id)
            return {
                "risk_theme_id": theme_id,
                "risk_theme_name": profile.get("risk_theme_name", theme_id),
                "classification_method": "embedding_cluster_model",
                "confidence_note": "Theme assigned by nearest MiniBatchKMeans embedding cluster.",
            }
        # BM25-only or existing-theme-column fallback: infer from weighted nearest neighbors.
        if similar_events is not None and not similar_events.empty and "matched_risk_theme_id" in similar_events.columns:
            work = similar_events.dropna(subset=["matched_risk_theme_id"]).copy()
            if not work.empty:
                score_col = "faiss_score" if "faiss_score" in work.columns and work["faiss_score"].notna().any() else "similarity_score"
                weights = work.groupby("matched_risk_theme_id")[score_col].sum().sort_values(ascending=False)
                theme_id = str(weights.index[0])
                names = work.loc[work["matched_risk_theme_id"].astype(str).eq(theme_id), "matched_risk_theme_name"].dropna().astype(str)
                theme_name = names.mode().iloc[0] if not names.empty else theme_id
                return {
                    "risk_theme_id": theme_id,
                    "risk_theme_name": theme_name,
                    "classification_method": "weighted_nearest_neighbor_theme_vote",
                    "weighted_similarity": float(weights.iloc[0]),
                    "confidence_note": "Theme inferred from retrieved historical records.",
                }
        return {
            "risk_theme_id": None,
            "risk_theme_name": "Unknown theme",
            "classification_method": "not_available",
            "confidence_note": "No theme model or theme columns were available.",
        }

    def _top_retrieval_score(self, matches: pd.DataFrame) -> float | None:
        if matches is None or matches.empty or "similarity_score" not in matches.columns:
            return None
        value = pd.to_numeric(matches["similarity_score"], errors="coerce").max()
        return None if pd.isna(value) else float(value)

    def _top_faiss_score(self, matches: pd.DataFrame) -> float | None:
        if matches is None or matches.empty or "faiss_score" not in matches.columns:
            return None
        value = pd.to_numeric(matches["faiss_score"], errors="coerce").max()
        return None if pd.isna(value) else float(value)

    def _severe_similarity_band(self, matches: pd.DataFrame, top_faiss_score: float | None) -> str:
        if top_faiss_score is not None:
            return similarity_band(
                top_faiss_score,
                high=self.settings.high_similarity_threshold,
                medium=self.settings.medium_similarity_threshold,
                low=self.settings.low_similarity_threshold,
            )
        if matches is not None and not matches.empty and self.retrieval_mode == "bm25":
            return "bm25_retrieved_keyword_match"
        if matches is not None and not matches.empty:
            return "retrieved_match"
        return "no_match"

    def _score_note(self) -> str:
        if self.retrieval_mode == "faiss":
            return "top_score is FAISS cosine similarity over normalized transformer embeddings."
        if self.retrieval_mode == "bm25":
            return "top_score is BM25 keyword score; BM25 scores are not cosine similarities and are not calibrated to FAISS thresholds."
        return "top_score is hybrid reciprocal-rank-fusion score; top_faiss_cosine_score is included when available for cosine-based severity banding."

    def _theme_profile(self, theme_id: str | None) -> dict:
        if not theme_id or self.theme_profiles.empty:
            return {}
        work = self.theme_profiles[self.theme_profiles["risk_theme_id"].astype(str).eq(str(theme_id))]
        if work.empty:
            return {}
        row = work.iloc[0].to_dict()
        decoded = {}
        for key, value in row.items():
            if isinstance(value, str) and value.startswith("["):
                try:
                    decoded[key] = json.loads(value)
                except Exception:
                    decoded[key] = value
            else:
                decoded[key] = value
        return decoded

    def _records_for_json(self, matches: pd.DataFrame, max_text: int = 300) -> list[dict]:
        if matches is None or matches.empty:
            return []
        cols = [
            "rank", "retrieval_method", "similarity_score", "faiss_score", "bm25_score", "hybrid_score", "faiss_rank", "bm25_rank",
            "matched_event_id", "matched_source_type", "matched_source_role", "matched_event_date", "matched_site",
            "matched_department", "matched_title", "matched_description", "matched_retrieval_text", "matched_risk_theme_id",
            "matched_risk_theme_name", "matched_severe_actual", "matched_any_injury", "matched_is_open_task", "matched_is_overdue_task",
        ]
        out = []
        for _, row in matches.iterrows():
            item = {}
            for col in cols:
                if col in row.index:
                    value = row[col]
                    if col.endswith("text") or col.endswith("description") or col.endswith("title"):
                        value = preview(value, max_text)
                    try:
                        if pd.isna(value):
                            value = None
                    except Exception:
                        pass
                    item[col.replace("matched_", "")] = value
            out.append(item)
        return out


def flatten_analysis_for_csv(result: dict) -> dict:
    """Create one row per analyzed query for CSV exports."""
    severe = result.get("historical_severe_injury_similarity", {})
    theme = result.get("risk_pattern_classification", {})
    recall = result.get("similar_historical_event_recall", {})
    runtime = result.get("runtime", {})
    return {
        "event_id": result.get("query", {}).get("event_id"),
        "source_type": result.get("query", {}).get("source_type"),
        "site": result.get("query", {}).get("site"),
        "department": result.get("query", {}).get("department"),
        "query_text_preview": result.get("query", {}).get("text_preview"),
        "retrieval_mode": runtime.get("retrieval_mode"),
        "embedding_model_name": runtime.get("embedding_model_name"),
        "risk_theme_id": theme.get("risk_theme_id"),
        "risk_theme_name": theme.get("risk_theme_name"),
        "theme_classification_method": theme.get("classification_method"),
        "severe_injury_top_score": severe.get("top_score"),
        "severe_injury_top_faiss_cosine_score": severe.get("top_faiss_cosine_score"),
        "severe_injury_similarity_band": severe.get("similarity_band"),
        "risk_factors": compress_json_field(result.get("risk_factor_extraction", [])),
        "recommended_prevention_actions": compress_json_field(result.get("recommended_prevention_actions", [])),
        "missing_information_prompt": compress_json_field(result.get("missing_information_prompt", [])),
        "llm_response_status": result.get("llm_final_response", {}).get("status"),
        "llm_model_name": result.get("llm_final_response", {}).get("model_name"),
        "llm_response_text": result.get("llm_final_response", {}).get("response_text"),
        "llm_response_error": result.get("llm_final_response", {}).get("error"),
        "similar_event_source_role_counts": compress_json_field(recall.get("source_role_counts", {})),
        "top_severe_injury_matches": compress_json_field(severe.get("matches", [])),
        "top_similar_event_matches": compress_json_field(recall.get("matches", [])),
    }
