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
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .artifact_io import artifact_exists, artifact_join, read_pickle
from .bm25_store import load_bm25_store, load_bm25_subset, search_bm25
from .config import Settings, get_settings
from .embedding import TextEmbedder
from .extraction import extract_keyphrases, missing_information_prompt, recommend_from_actions
from .faiss_store import load_index_bundle, search_index
from .local_llm import LocalLLMResponder
from .theme import load_theme_model
from .utils import clean_text_value, compress_json_field, ensure_dir, preview, save_json, similarity_band


_INDEX_NAMES = [
    "severe_injuries",
    "all_injuries",
    "hazard_identifications",
    "near_misses",
    "audit_observations",
    "other_audit_observations",
    "safe_observations",
    "unsafe_observations",
    "safe_actions",
    "unsafe_actions",
    "safe_conditions",
    "unsafe_conditions",
    "corrective_actions",
    "open_corrective_actions",
    "overdue_corrective_actions",
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

        kb_path = self.settings.artifact_enriched_knowledge_base_path() if hasattr(self.settings, "artifact_enriched_knowledge_base_path") else self.settings.enriched_knowledge_base_path()
        if not artifact_exists(kb_path):
            kb_path = self.settings.artifact_knowledge_base_path() if hasattr(self.settings, "artifact_knowledge_base_path") else self.settings.knowledge_base_path()
        if not artifact_exists(kb_path):
            raise FileNotFoundError(
                f"Knowledge base not found at artifact path: {kb_path}. Confirm the Azure ML job output exists or switch artifact_read_mode='local'."
            )
        print(f"[Agent] Loading knowledge base from: {kb_path}", flush=True)
        self.records = read_pickle(kb_path).reset_index(drop=True)
        self.records["row_id"] = np.arange(len(self.records))
        self.theme_profiles = self._load_theme_profiles()
        self.theme_model = load_theme_model(self.settings)

        self.indexes: dict[str, tuple[Any, np.ndarray, dict]] = {}
        self.bm25_subsets: dict[str, tuple[np.ndarray, dict]] = {}
        self.bm25_store = None
        self.embedder: TextEmbedder | None = None
        self.index_embedding_metadata: dict = {}
        self.llm_responder = LocalLLMResponder(self.settings) if bool(getattr(self.settings, "enable_llm_response", True)) else None
        self.cross_encoder_guard = self._load_cross_encoder_guard()

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

    def _load_cross_encoder_guard(self):
        """Optionally load an open-source cross-encoder relevance guard.

        This is disabled by default because it adds latency. When enabled, it
        provides a second-stage query/evidence relevance score after FAISS/BM25
        retrieval and before structured evidence is sent to the LLM.
        """
        if not bool(getattr(self.settings, "enable_cross_encoder_relevance_check", False)):
            return None
        try:
            from sentence_transformers import CrossEncoder

            model_name = str(getattr(self.settings, "cross_encoder_model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
            print(f"[Agent] Loading optional cross-encoder relevance guard: {model_name}", flush=True)
            return CrossEncoder(model_name)
        except Exception as exc:
            print(
                "[Agent] Cross-encoder relevance guard could not be loaded; continuing with score/overlap gate only. "
                f"Error: {exc}",
                flush=True,
            )
            return None

    @property
    def _uses_faiss(self) -> bool:
        return self.retrieval_mode in {"faiss", "hybrid"}

    @property
    def _uses_bm25(self) -> bool:
        return self.retrieval_mode in {"bm25", "hybrid"}

    def _load_faiss_indexes(self) -> None:
        index_dir = self.settings.artifact_indexes_dir() if hasattr(self.settings, "artifact_indexes_dir") else self.settings.indexes_dir()
        print(f"[Agent] Loading FAISS indexes from: {index_dir}", flush=True)
        for name in _INDEX_NAMES:
            if artifact_exists(artifact_join(index_dir, f"{name}.faiss")):
                self.indexes[name] = load_index_bundle(index_dir, name, settings=self.settings)
        if not self.indexes:
            raise FileNotFoundError(
                f"No FAISS indexes found in {index_dir}. Confirm datastore artifacts exist, or set retrieval_mode='bm25'."
            )

    def _load_bm25_indexes(self) -> None:
        bm25_dir = self.settings.artifact_bm25_dir() if hasattr(self.settings, "artifact_bm25_dir") else self.settings.bm25_dir()
        print(f"[Agent] Loading BM25 index from: {bm25_dir}", flush=True)
        self.bm25_store = load_bm25_store(bm25_dir)
        for name in _INDEX_NAMES:
            row_path = artifact_join(bm25_dir, f"{name}_row_ids.npy")
            if artifact_exists(row_path):
                self.bm25_subsets[name] = load_bm25_subset(bm25_dir, name)
        if not self.bm25_subsets:
            raise FileNotFoundError(
                f"No BM25 subset files found in {bm25_dir}. Confirm datastore artifacts exist, or set retrieval_mode='faiss'."
            )

    def _load_theme_profiles(self) -> pd.DataFrame:
        path = self.settings.artifact_theme_profiles_path() if hasattr(self.settings, "artifact_theme_profiles_path") else self.settings.theme_profiles_path()
        if artifact_exists(path):
            return read_pickle(path)
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
        """Analyze a new or existing safety event description.

        The runtime evidence package is now built from purpose-specific indexes:
        injuries first, then leading events, then controls/prevention evidence.
        The broad all_events index is intentionally not used.
        """
        query_text = clean_text_value(query_text)
        if not query_text:
            raise ValueError("query_text is required")

        query_vector = self._encode_query(query_text) if self._uses_faiss else None
        top_k_events = top_k_events or self.settings.top_k_similar_events
        per_source_k = max(5, int(top_k_events))

        # 1) Injury evidence: retrieve raw candidates first, then apply the
        # route-specific quality gate. Raw candidates are preserved in
        # raw_retrieval_debug; only accepted evidence is used for structured output
        # and for the final LLM response.
        raw_severe_matches = self._retrieve_raw(
            "severe_injuries",
            query_text,
            query_vector,
            self.settings.top_k_severe_injuries,
            exclude_event_id=event_id,
        )
        severe_matches = self._filter_meaningful_matches(
            raw_severe_matches,
            "severe_injuries",
            top_k=self.settings.top_k_severe_injuries,
        )
        top_severe_score = self._top_retrieval_score(severe_matches)
        top_severe_faiss_score = self._top_faiss_score(severe_matches)
        severe_band = self._severe_similarity_band(severe_matches, top_severe_faiss_score)
        severe_evidence_found = self._has_meaningful_severe_evidence(severe_matches, severe_band)

        if severe_evidence_found:
            raw_injury_matches = pd.DataFrame()
            injury_matches = pd.DataFrame()
            injury_search_note = "Skipped all_injuries search because meaningful severe-injury evidence was found."
        else:
            raw_injury_matches = self._retrieve_raw(
                "all_injuries",
                query_text,
                query_vector,
                max(self.settings.top_k_severe_injuries * 4, self.settings.top_k_severe_injuries + 5),
                exclude_event_id=event_id,
            )
            injury_matches = self._filter_meaningful_matches(
                raw_injury_matches,
                "all_injuries",
                top_k=self.settings.top_k_severe_injuries,
            )
            injury_search_note = "Searched all_injuries because no meaningful severe-injury evidence met the response threshold."

        # 2) Leading-event evidence from purpose-specific source indexes.
        raw_hazard_matches = self._retrieve_raw("hazard_identifications", query_text, query_vector, per_source_k, exclude_event_id=event_id)
        raw_near_miss_matches = self._retrieve_raw("near_misses", query_text, query_vector, per_source_k, exclude_event_id=event_id)
        raw_audit_matches = self._retrieve_raw("audit_observations", query_text, query_vector, per_source_k, exclude_event_id=event_id)
        raw_other_audit_matches = self._retrieve_raw("other_audit_observations", query_text, query_vector, per_source_k, exclude_event_id=event_id)
        raw_unsafe_action_matches = self._retrieve_raw("unsafe_actions", query_text, query_vector, per_source_k, exclude_event_id=event_id)
        raw_unsafe_condition_matches = self._retrieve_raw("unsafe_conditions", query_text, query_vector, per_source_k, exclude_event_id=event_id)
        raw_safe_action_matches = self._retrieve_raw("safe_actions", query_text, query_vector, self.settings.top_k_safe_practices, exclude_event_id=event_id)
        raw_safe_condition_matches = self._retrieve_raw("safe_conditions", query_text, query_vector, self.settings.top_k_safe_practices, exclude_event_id=event_id)

        hazard_matches = self._filter_meaningful_matches(raw_hazard_matches, "hazard_identifications", top_k=per_source_k)
        near_miss_matches = self._filter_meaningful_matches(raw_near_miss_matches, "near_misses", top_k=per_source_k)
        audit_matches = self._filter_meaningful_matches(raw_audit_matches, "audit_observations", top_k=per_source_k)
        other_audit_matches = self._filter_meaningful_matches(raw_other_audit_matches, "other_audit_observations", top_k=per_source_k)
        unsafe_action_matches = self._filter_meaningful_matches(raw_unsafe_action_matches, "unsafe_actions", top_k=per_source_k)
        unsafe_condition_matches = self._filter_meaningful_matches(raw_unsafe_condition_matches, "unsafe_conditions", top_k=per_source_k)
        safe_action_matches = self._filter_meaningful_matches(raw_safe_action_matches, "safe_actions", top_k=self.settings.top_k_safe_practices)
        safe_condition_matches = self._filter_meaningful_matches(raw_safe_condition_matches, "safe_conditions", top_k=self.settings.top_k_safe_practices)

        similar_events = self._concat_matches(
            [hazard_matches, near_miss_matches, audit_matches, unsafe_action_matches, unsafe_condition_matches],
            top_k=max(top_k_events, int(top_k_events)),
        )

        # 3) Corrective-action/prevention evidence. Open/overdue routes use
        # stricter quality thresholds because their indexes are smaller and can
        # otherwise return the nearest available but unrelated task.
        raw_action_matches = self._retrieve_raw("corrective_actions", query_text, query_vector, self.settings.top_k_corrective_actions, exclude_event_id=event_id)
        raw_open_action_matches = self._retrieve_raw("open_corrective_actions", query_text, query_vector, self.settings.top_k_corrective_actions, exclude_event_id=event_id)
        raw_overdue_action_matches = self._retrieve_raw("overdue_corrective_actions", query_text, query_vector, self.settings.top_k_corrective_actions, exclude_event_id=event_id)
        action_matches = self._filter_meaningful_matches(raw_action_matches, "corrective_actions", top_k=self.settings.top_k_corrective_actions)
        open_action_matches = self._filter_meaningful_matches(raw_open_action_matches, "open_corrective_actions", top_k=self.settings.top_k_corrective_actions)
        overdue_action_matches = self._filter_meaningful_matches(raw_overdue_action_matches, "overdue_corrective_actions", top_k=self.settings.top_k_corrective_actions)
        safe_matches = self._concat_matches([safe_action_matches, safe_condition_matches], top_k=self.settings.top_k_safe_practices)

        theme = self._classify_theme(query_vector, similar_events)

        evidence_texts = []
        for df in [hazard_matches, near_miss_matches, audit_matches, unsafe_action_matches, unsafe_condition_matches, severe_matches, injury_matches, action_matches, safe_matches]:
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
        display_injury_evidence = self._select_injury_evidence_for_response(
            severe_matches=severe_matches,
            injury_matches=injury_matches,
            severe_band=severe_band,
        )

        raw_retrieval_frames = {
            "severe_injuries": raw_severe_matches,
            "all_injuries": raw_injury_matches,
            "hazard_identifications": raw_hazard_matches,
            "near_misses": raw_near_miss_matches,
            "audit_observations": raw_audit_matches,
            "other_audit_observations": raw_other_audit_matches,
            "unsafe_actions": raw_unsafe_action_matches,
            "unsafe_conditions": raw_unsafe_condition_matches,
            "safe_actions": raw_safe_action_matches,
            "safe_conditions": raw_safe_condition_matches,
            "corrective_actions": raw_action_matches,
            "open_corrective_actions": raw_open_action_matches,
            "overdue_corrective_actions": raw_overdue_action_matches,
        }
        retrieval_frames = {
            "severe_injuries": severe_matches,
            "all_injuries": injury_matches,
            "hazard_identifications": hazard_matches,
            "near_misses": near_miss_matches,
            "audit_observations": audit_matches,
            "other_audit_observations": other_audit_matches,
            "unsafe_actions": unsafe_action_matches,
            "unsafe_conditions": unsafe_condition_matches,
            "safe_actions": safe_action_matches,
            "safe_conditions": safe_condition_matches,
            "corrective_actions": action_matches,
            "open_corrective_actions": open_action_matches,
            "overdue_corrective_actions": overdue_action_matches,
        }

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
                "artifact_read_mode": getattr(self.settings, "artifact_read_mode", "local"),
                "artifact_root": str(self.settings.artifact_root()) if hasattr(self.settings, "artifact_root") else str(self.settings.output_dir),
                "embedding_model_name": self.embedder.model_name if self.embedder is not None else None,
                "embedding_backend": self.embedder.actual_backend if self.embedder is not None else None,
                "index_embedding_model_name": self.index_embedding_metadata.get("embedding_model_name"),
                "index_embedding_backend": self.index_embedding_metadata.get("embedding_backend"),
                "bm25_algorithm": self.bm25_store.metadata.get("bm25_algorithm") if self.bm25_store is not None else None,
                "llm_response_enabled": bool(getattr(self.settings, "enable_llm_response", True)),
                "llm_model_name": getattr(self.settings, "llm_model_name", None),
                "index_strategy": "purpose_specific_no_all_events",
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
                "searched": not severe_evidence_found,
                "search_note": injury_search_note,
                "matches": self._records_for_json(injury_matches),
            },
            "injury_evidence_for_response": display_injury_evidence,
            "similar_historical_event_recall": {
                "matches": self._records_for_json(similar_events.head(top_k_events)),
                "source_role_counts": similar_events["matched_source_role"].value_counts().to_dict() if not similar_events.empty and "matched_source_role" in similar_events.columns else {},
                "retrieval_indexes": ["hazard_identifications", "near_misses", "audit_observations"],
            },
            "leading_event_evidence": {
                "hazard_identification_matches": self._records_for_json(hazard_matches.head(self.settings.top_k_similar_events)),
                "near_miss_matches": self._records_for_json(near_miss_matches.head(self.settings.top_k_similar_events)),
                "audit_observation_matches": self._records_for_json(audit_matches.head(self.settings.top_k_similar_events)),
                "other_audit_observation_matches": self._records_for_json(other_audit_matches.head(self.settings.top_k_similar_events)),
                "unsafe_action_matches": self._records_for_json(unsafe_action_matches.head(self.settings.top_k_similar_events)),
                "unsafe_condition_matches": self._records_for_json(unsafe_condition_matches.head(self.settings.top_k_similar_events)),
                "safe_action_matches": self._records_for_json(safe_action_matches.head(self.settings.top_k_safe_practices)),
                "safe_condition_matches": self._records_for_json(safe_condition_matches.head(self.settings.top_k_safe_practices)),
            },
            "corrective_action_recall": {
                "matches": self._records_for_json(action_matches),
                "open_action_matches": self._records_for_json(open_action_matches),
                "overdue_action_matches": self._records_for_json(overdue_action_matches),
            },
            "safe_practice_recall": {
                "matches": self._records_for_json(safe_matches),
                "safe_action_matches": self._records_for_json(safe_action_matches),
                "safe_condition_matches": self._records_for_json(safe_condition_matches),
            },
            "risk_factor_extraction": risk_factors,
            "recommended_prevention_actions": prevention_actions,
            "missing_information_prompt": missing_prompts,
        }

        # Separate retrieval layers for testing/debugging. The raw layer keeps scores,
        # ranks, row IDs, and site codes. The structured layers remove technical fields
        # and describe which indexes support each response section.
        result["raw_retrieval_debug"] = self._build_raw_retrieval_debug(raw_retrieval_frames)
        result["structured_evidence_summary"] = self._build_structured_evidence_summary(
            result=result,
            retrieval_frames=retrieval_frames,
            injury_evidence=display_injury_evidence,
            risk_factors=risk_factors,
            missing_prompts=missing_prompts,
            prevention_actions=prevention_actions,
            severe_evidence_found=severe_evidence_found,
        )
        result["structured_response_plan"] = self._build_structured_response_plan(result["structured_evidence_summary"])
        result["llm_final_response"] = self._generate_llm_final_response(result)
        result["user_facing_final_response"] = self._build_user_facing_final_response(result)
        return result


    def _concat_matches(self, frames: list[pd.DataFrame], top_k: int | None = None) -> pd.DataFrame:
        """Combine retrieval results from multiple purpose-specific indexes.

        Each route (hazards, near misses, audit observations, safe/unsafe actions,
        etc.) is retrieved separately. This helper merges those DataFrames, removes
        duplicate event IDs, sorts by the best available retrieval score, and returns
        the top rows for downstream evidence packaging.
        """
        valid_frames = []
        for frame in frames or []:
            if frame is not None and not frame.empty:
                valid_frames.append(frame.copy())

        if not valid_frames:
            return pd.DataFrame()

        combined = pd.concat(valid_frames, ignore_index=True, sort=False)

        # Create a common ranking score. Hybrid score is preferred when available,
        # then FAISS/cosine similarity, then the generic similarity score. BM25 is
        # intentionally a later fallback because its scale is corpus-dependent.
        score_cols = [
            "hybrid_score",
            "faiss_score",
            "similarity_score",
            "bm25_score",
        ]
        combined["_sort_score"] = 0.0
        for col in score_cols:
            if col in combined.columns:
                vals = pd.to_numeric(combined[col], errors="coerce")
                combined["_sort_score"] = combined["_sort_score"].where(
                    combined["_sort_score"].notna() & (combined["_sort_score"] != 0),
                    vals,
                )

        # Keep one row per event when the same record is retrieved from multiple
        # routes. Sort first so the strongest route is retained.
        if "matched_event_id" in combined.columns:
            combined = combined.sort_values("_sort_score", ascending=False, na_position="last")
            combined = combined.drop_duplicates(subset=["matched_event_id"], keep="first")
        else:
            combined = combined.sort_values("_sort_score", ascending=False, na_position="last")

        if top_k is not None:
            combined = combined.head(int(top_k))

        combined = combined.drop(columns=["_sort_score"], errors="ignore").reset_index(drop=True)
        combined["rank"] = range(1, len(combined) + 1)
        return combined


    @staticmethod
    def _is_numeric_like(value: object) -> bool:
        text = clean_text_value(value)
        if not text:
            return False
        try:
            float(text)
            return True
        except Exception:
            return False

    @staticmethod
    def _safe_display_text(value: object, max_len: int = 300) -> str | None:
        text = clean_text_value(value)
        if not text:
            return None
        return preview(text, max_len)

    def _display_site(self, value: object) -> str | None:
        """Return a business-readable site label, suppressing numeric-only site codes."""
        text = clean_text_value(value)
        if not text or self._is_numeric_like(text):
            return None
        return text

    def _raw_records_for_debug(self, matches: pd.DataFrame, max_text: int = 700) -> list[dict]:
        """Return raw retrieval rows for technical debugging.

        This intentionally preserves ranking, score, row-id, and site-code fields.
        These records are saved under outputs/tests/raw_*.json/jsonl and should not
        be shown directly to business users.
        """
        if matches is None or matches.empty:
            return []
        raw_cols = [
            "rank", "retrieval_method", "similarity_score", "faiss_score", "bm25_score", "hybrid_score",
            "faiss_rank", "bm25_rank", "matched_row_id", "matched_event_id", "matched_source_id",
            "matched_source_type", "matched_source_role", "matched_event_date", "matched_site", "matched_department",
            "matched_location_path", "matched_title", "matched_description", "matched_retrieval_text",
            "matched_risk_theme_id", "matched_risk_theme_name", "matched_severe_actual", "matched_any_injury",
            "matched_is_open_task", "matched_is_overdue_task", "matched_status", "matched_category", "matched_audit_type",
            "matched_raw_category_id", "matched_raw_type_id", "matched_raw_status_id",
            "retrieval_quality_pass", "retrieval_quality_reason", "retrieval_quality_band",
            "faiss_quality_band", "bm25_quality_band", "hybrid_quality_band",
            "lexical_overlap_count", "lexical_overlap_ratio", "shared_quality_terms",
            "query_term_count", "evidence_term_count", "retriever_agreement",
            "min_return_band_required", "min_overlap_terms_required",
            "min_overlap_ratio_required", "cross_encoder_score", "cross_encoder_pass",
        ]
        out: list[dict] = []
        for _, row in matches.iterrows():
            item: dict[str, Any] = {}
            for col in raw_cols:
                if col not in row.index:
                    continue
                value = row[col]
                if col.endswith("text") or col.endswith("description") or col.endswith("title"):
                    value = preview(value, max_text)
                try:
                    if pd.isna(value):
                        value = None
                except Exception:
                    pass
                item[col] = value
            out.append(item)
        return out

    def _compact_records_for_summary(
        self,
        matches: pd.DataFrame,
        index_name: str,
        purpose: str,
        max_records: int | None = None,
        max_text: int = 220,
    ) -> list[dict]:
        """Return business-readable evidence records without retrieval scores."""
        if matches is None or matches.empty:
            return []
        if max_records is not None:
            matches = matches.head(int(max_records))
        out: list[dict] = []
        for _, row in matches.iterrows():
            title = self._safe_display_text(row.get("matched_title"), max_text)
            description = self._safe_display_text(row.get("matched_description"), max_text)
            retrieval_text = self._safe_display_text(row.get("matched_retrieval_text"), max_text)
            summary_text = title or description or retrieval_text
            event_id = clean_text_value(row.get("matched_event_id")) or "unknown_event"
            role = clean_text_value(row.get("matched_source_role")) or clean_text_value(row.get("matched_source_type")) or "historical_record"
            item = {
                "evidence_id": event_id,
                "index_used": index_name,
                "evidence_purpose": purpose,
                "source_type": clean_text_value(row.get("matched_source_type")) or None,
                "source_role": role,
                "event_date": clean_text_value(row.get("matched_event_date")) or None,
                "site_label": self._display_site(row.get("matched_site")),
                "department_label": self._display_site(row.get("matched_department")),
                "title": title,
                "summary": summary_text,
                "risk_theme_id": clean_text_value(row.get("matched_risk_theme_id")) or None,
                "risk_theme_name": clean_text_value(row.get("matched_risk_theme_name")) or None,
            }
            out.append(item)
        return out

    def _build_raw_retrieval_debug(self, retrieval_frames: dict[str, pd.DataFrame]) -> dict:
        route_purpose = {
            "severe_injuries": "High-severity historical injury similarity route.",
            "all_injuries": "Fallback normal/all-injury similarity route used only when severe evidence is not meaningful.",
            "hazard_identifications": "Leading-event route for historical hazard identification records.",
            "near_misses": "Leading-event route for historical near-miss records.",
            "audit_observations": "Leading-event route for generic audit observation records.",
            "other_audit_observations": "Leading-event route for audit records not classified as safe/unsafe action or condition.",
            "unsafe_actions": "Leading-event route for unsafe action observations.",
            "unsafe_conditions": "Leading-event route for unsafe condition observations.",
            "safe_actions": "Safe-practice route for safe action observations.",
            "safe_conditions": "Safe-practice route for safe condition observations.",
            "corrective_actions": "Prevention evidence route for historical corrective/action task records.",
            "open_corrective_actions": "Prevention evidence route for currently open corrective/action task records.",
            "overdue_corrective_actions": "Prevention evidence route for overdue corrective/action task records.",
        }
        routes: dict[str, dict] = {}
        for index_name, frame in retrieval_frames.items():
            accepted_count = 0
            if frame is not None and not frame.empty and "retrieval_quality_pass" in frame.columns:
                accepted_count = int(frame["retrieval_quality_pass"].fillna(False).astype(bool).sum())
            routes[index_name] = {
                "index_used": index_name,
                "purpose": route_purpose.get(index_name, "Retrieval route."),
                "raw_candidate_count": int(0 if frame is None or frame.empty else len(frame)),
                "accepted_count_after_quality_gate": accepted_count,
                "records": self._raw_records_for_debug(frame),
            }
        return {
            "retrieval_mode": self.retrieval_mode,
            "note": "Technical debug output. Contains raw scores, ranks, row IDs, source codes, and quality-gate diagnostics. Do not show this directly to business users.",
            "routes": routes,
        }

    def _normalise_risk_factor(self, item: object) -> str | None:
        if isinstance(item, dict):
            text = item.get("risk_factor") or item.get("phrase") or item.get("text") or item.get("value")
        else:
            text = item
        text = clean_text_value(text)
        return text or None

    def _normalise_missing_prompt(self, item: object) -> dict | None:
        if isinstance(item, dict):
            prompt = clean_text_value(item.get("prompt") or item.get("question") or "")
            area = clean_text_value(item.get("missing_area") or item.get("area") or "")
        else:
            prompt = clean_text_value(item)
            area = ""
        if not prompt:
            return None
        return {"area": area or "additional detail", "question": prompt}

    def _normalise_action_candidate(self, item: object) -> dict | None:
        if isinstance(item, dict):
            rec = clean_text_value(item.get("recommendation") or item.get("action") or item.get("text") or "")
            evidence_id = clean_text_value(item.get("supporting_event_id") or item.get("event_id") or "")
        else:
            rec = clean_text_value(item)
            evidence_id = ""
        if not rec:
            return None
        return {"suggested_action": preview(rec, 220), "supporting_evidence_id": evidence_id or None}

    @staticmethod
    def _evidence_ids(records: list[dict]) -> list[str]:
        out: list[str] = []
        for item in records or []:
            evid = clean_text_value(item.get("evidence_id"))
            if evid and evid not in out:
                out.append(evid)
        return out

    def _build_structured_evidence_summary(
        self,
        result: dict,
        retrieval_frames: dict[str, pd.DataFrame],
        injury_evidence: dict,
        risk_factors: list,
        missing_prompts: list,
        prevention_actions: list,
        severe_evidence_found: bool,
    ) -> dict:
        """Build cleaned evidence used by the LLM and by test/debug outputs."""
        max_records = int(getattr(self.settings, "llm_max_evidence_records_per_section", 5))
        theme = result.get("risk_pattern_classification", {}) or {}
        query = result.get("query", {}) or {}

        selected_injury_index = "severe_injuries" if severe_evidence_found else "all_injuries"
        selected_injury_records = injury_evidence.get("matches", []) or []
        # injury_evidence already uses compact field names from _records_for_json;
        # convert them to the same compact format used elsewhere.
        injury_records = []
        for item in selected_injury_records[:max_records]:
            injury_records.append({
                "evidence_id": clean_text_value(item.get("event_id")) or "unknown_event",
                "index_used": selected_injury_index,
                "evidence_purpose": "injury similarity evidence",
                "source_type": item.get("source_type"),
                "source_role": item.get("source_role"),
                "event_date": item.get("event_date"),
                "site_label": self._display_site(item.get("site")),
                "department_label": self._display_site(item.get("department")),
                "title": self._safe_display_text(item.get("title"), 220),
                "summary": self._safe_display_text(item.get("description") or item.get("retrieval_text") or item.get("title"), 220),
                "risk_theme_id": item.get("risk_theme_id"),
                "risk_theme_name": item.get("risk_theme_name"),
            })

        leading = {
            "hazard_identifications": self._compact_records_for_summary(retrieval_frames.get("hazard_identifications"), "hazard_identifications", "leading event: hazard identification", max_records),
            "near_misses": self._compact_records_for_summary(retrieval_frames.get("near_misses"), "near_misses", "leading event: near miss", max_records),
            "audit_observations": self._compact_records_for_summary(retrieval_frames.get("audit_observations"), "audit_observations", "leading event: audit observation", max_records),
            "other_audit_observations": self._compact_records_for_summary(retrieval_frames.get("other_audit_observations"), "other_audit_observations", "leading event: other audit observation", max_records),
            "unsafe_actions": self._compact_records_for_summary(retrieval_frames.get("unsafe_actions"), "unsafe_actions", "leading event: unsafe action", max_records),
            "unsafe_conditions": self._compact_records_for_summary(retrieval_frames.get("unsafe_conditions"), "unsafe_conditions", "leading event: unsafe condition", max_records),
            "safe_actions": self._compact_records_for_summary(retrieval_frames.get("safe_actions"), "safe_actions", "safe practice: safe action", max_records),
            "safe_conditions": self._compact_records_for_summary(retrieval_frames.get("safe_conditions"), "safe_conditions", "safe practice: safe condition", max_records),
        }
        prevention = {
            "corrective_actions": self._compact_records_for_summary(retrieval_frames.get("corrective_actions"), "corrective_actions", "historical corrective action / prevention evidence", max_records),
            "open_corrective_actions": self._compact_records_for_summary(retrieval_frames.get("open_corrective_actions"), "open_corrective_actions", "open corrective action / prevention evidence", max_records),
            "overdue_corrective_actions": self._compact_records_for_summary(retrieval_frames.get("overdue_corrective_actions"), "overdue_corrective_actions", "overdue corrective action / prevention evidence", max_records),
        }
        action_candidates = [self._normalise_action_candidate(x) for x in (prevention_actions or [])]
        action_candidates = [x for x in action_candidates if x]
        factor_candidates = [self._normalise_risk_factor(x) for x in (risk_factors or [])]
        factor_candidates = [x for x in factor_candidates if x]
        missing = [self._normalise_missing_prompt(x) for x in (missing_prompts or [])]
        missing = [x for x in missing if x]

        evidence_ids_by_section = {
            "injury_similarity_evidence": self._evidence_ids(injury_records),
            "hazard_identifications": self._evidence_ids(leading["hazard_identifications"]),
            "near_misses": self._evidence_ids(leading["near_misses"]),
            "audit_observations": self._evidence_ids(leading["audit_observations"]),
            "unsafe_actions": self._evidence_ids(leading["unsafe_actions"]),
            "unsafe_conditions": self._evidence_ids(leading["unsafe_conditions"]),
            "safe_actions": self._evidence_ids(leading["safe_actions"]),
            "safe_conditions": self._evidence_ids(leading["safe_conditions"]),
            "corrective_actions": self._evidence_ids(prevention["corrective_actions"]),
            "open_corrective_actions": self._evidence_ids(prevention["open_corrective_actions"]),
            "overdue_corrective_actions": self._evidence_ids(prevention["overdue_corrective_actions"]),
        }
        return {
            "query": {
                "event_id": query.get("event_id"),
                "source_type": query.get("source_type"),
                "site": query.get("site"),
                "department": query.get("department"),
                "text_preview": query.get("text_preview"),
            },
            "detected_pattern": {
                "risk_theme_id": theme.get("risk_theme_id"),
                "risk_theme_name": theme.get("risk_theme_name"),
                "classification_method": theme.get("classification_method"),
                "theme_profile_summary": (result.get("theme_profile", {}) or {}).get("theme_summary"),
            },
            "injury_similarity_evidence": {
                "selected_index": selected_injury_index,
                "selection_rule": "Use severe_injuries when meaningful severe-injury evidence is found; otherwise use all_injuries as fallback.",
                "evidence_type": injury_evidence.get("evidence_type"),
                "similarity_band": injury_evidence.get("similarity_band"),
                "message": injury_evidence.get("message"),
                "records": injury_records,
            },
            "leading_event_evidence": leading,
            "corrective_actions_prevention_evidence": prevention,
            "risk_factors_and_possible_control_gaps": factor_candidates[:12],
            "recommended_prevention_action_candidates": action_candidates[: int(getattr(self.settings, "llm_max_action_candidates", 8))],
            "missing_information_to_collect": missing[: int(getattr(self.settings, "llm_max_missing_info_prompts", 8))],
            "evidence_ids_by_section": evidence_ids_by_section,
        }

    def _build_structured_response_plan(self, evidence: dict) -> dict:
        injury = evidence.get("injury_similarity_evidence", {}) or {}
        leading = evidence.get("leading_event_evidence", {}) or {}
        prevention = evidence.get("corrective_actions_prevention_evidence", {}) or {}
        return {
            "note": "This plan maps user-facing response sections to the indexes/data used to support them.",
            "sections": [
                {
                    "section": "1. Detected pattern",
                    "output_field": "detected_pattern",
                    "indexes_used": [],
                    "data_sources": ["theme_kmeans.joblib", "safety_theme_profiles.pkl"],
                    "evidence_ids": [],
                },
                {
                    "section": "2. Injury similarity evidence",
                    "output_field": "injury_similarity_evidence",
                    "indexes_used": [injury.get("selected_index")],
                    "data_sources": ["FAISS/BM25 purpose-specific injury index"],
                    "evidence_ids": self._evidence_ids(injury.get("records", []) or []),
                },
                {
                    "section": "3. Leading-event evidence: hazards, near misses, audit observations",
                    "output_field": "leading_event_evidence",
                    "indexes_used": [k for k, v in leading.items() if v],
                    "data_sources": ["hazard_identifications", "near_misses", "audit_observations", "safe/unsafe action and condition indexes"],
                    "evidence_ids": [eid for records in leading.values() for eid in self._evidence_ids(records or [])],
                },
                {
                    "section": "4. Risk factors and possible control gaps",
                    "output_field": "risk_factors_and_possible_control_gaps",
                    "indexes_used": ["query_text", "retrieved leading-event evidence", "retrieved injury/corrective-action evidence"],
                    "data_sources": ["local keyphrase extraction over query and retrieved evidence"],
                    "evidence_ids": [],
                },
                {
                    "section": "5. Corrective actions / prevention evidence",
                    "output_field": "corrective_actions_prevention_evidence",
                    "indexes_used": [k for k, v in prevention.items() if v],
                    "data_sources": ["corrective_actions", "open_corrective_actions", "overdue_corrective_actions"],
                    "evidence_ids": [eid for records in prevention.values() for eid in self._evidence_ids(records or [])],
                },
                {
                    "section": "6. Missing information to collect",
                    "output_field": "missing_information_to_collect",
                    "indexes_used": [],
                    "data_sources": ["rule-based missing-information prompt", "detected theme", "retrieval confidence"],
                    "evidence_ids": [],
                },
                {
                    "section": "7. Evidence IDs",
                    "output_field": "evidence_ids_by_section",
                    "indexes_used": ["all indexes that returned user-facing evidence"],
                    "data_sources": ["structured evidence summary"],
                    "evidence_ids": [eid for ids in (evidence.get("evidence_ids_by_section", {}) or {}).values() for eid in (ids or [])],
                },
            ],
        }

    def _build_user_facing_final_response(self, result: dict) -> dict:
        structured = result.get("structured_evidence_summary", {}) or {}
        return {
            "event_id": result.get("query", {}).get("event_id"),
            "source_type": result.get("query", {}).get("source_type"),
            "risk_theme_id": structured.get("detected_pattern", {}).get("risk_theme_id"),
            "risk_theme_name": structured.get("detected_pattern", {}).get("risk_theme_name"),
            "response_text": result.get("llm_final_response", {}).get("response_text"),
            "response_status": result.get("llm_final_response", {}).get("status"),
            "llm_model_name": result.get("llm_final_response", {}).get("model_name"),
            "evidence_ids_by_section": structured.get("evidence_ids_by_section", {}),
            "note": result.get("llm_final_response", {}).get("note"),
        }

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
                "Generated from retrieved historical evidence only. The agent identifies similar historical patterns, "
                "potential risk factors, and prevention considerations. It does not guarantee that an injury will "
                "or will not occur. Review by an EHS professional is required."
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
        """Return quality-gated retrieval evidence for normal business use.

        Raw candidates are available through _retrieve_raw and are saved in
        raw_retrieval_debug for validation. This public helper keeps backward
        compatibility for any older code path that still calls _retrieve.
        """
        raw = self._retrieve_raw(index_name, query_text, query_vector, top_k, exclude_event_id=exclude_event_id)
        return self._filter_meaningful_matches(raw, index_name, top_k=top_k)

    def _retrieve_raw(
        self,
        index_name: str,
        query_text: str,
        query_vector: np.ndarray | None,
        top_k: int,
        exclude_event_id: str | None = None,
    ) -> pd.DataFrame:
        """Return raw candidates annotated with quality diagnostics.

        FAISS/BM25 always return nearest records when an index is non-empty. This
        method deliberately returns a larger candidate pool, annotates it with
        generic relevance signals, and lets _filter_meaningful_matches decide what
        is safe to use in structured evidence and the final LLM response.
        """
        search_k = self._quality_candidate_k(top_k)
        if self.retrieval_mode == "faiss":
            raw = self._faiss_search(index_name, query_vector, search_k, exclude_event_id=exclude_event_id)
        elif self.retrieval_mode == "bm25":
            raw = self._bm25_search(index_name, query_text, search_k, exclude_event_id=exclude_event_id)
        else:
            raw = self._hybrid_search(index_name, query_text, query_vector, search_k, exclude_event_id=exclude_event_id)
        return self._annotate_retrieval_quality(raw, index_name=index_name, query_text=query_text)

    def _quality_candidate_k(self, top_k: int) -> int:
        multiplier = max(1, int(getattr(self.settings, "retrieval_quality_candidate_multiplier", 3)))
        cap = max(1, int(getattr(self.settings, "retrieval_quality_max_raw_candidates", 50)))
        return max(1, min(cap, int(top_k) * multiplier))

    @staticmethod
    def _band_rank(band: str | None) -> int:
        order = {"no_match": 0, "weak_match": 1, "possible_match": 2, "strong_match": 3}
        return order.get(str(band or "no_match"), 0)

    def _minimum_return_band(self, index_name: str) -> str:
        thresholds = self._quality_thresholds(index_name)
        return str(thresholds.get("return_min_band") or getattr(self.settings, "retrieval_quality_return_min_band", "possible_match"))

    def _quality_thresholds(self, index_name: str) -> dict[str, float | str]:
        if hasattr(self.settings, "route_quality_thresholds"):
            return dict(self.settings.route_quality_thresholds(index_name))
        return {
            "weak_faiss": 0.50, "possible_faiss": 0.56, "strong_faiss": 0.66,
            "weak_bm25": 2.0, "possible_bm25": 8.0, "strong_bm25": 20.0,
            "weak_hybrid": 0.012, "possible_hybrid": 0.022, "strong_hybrid": 0.032,
            "return_min_band": "possible_match",
        }

    def _score_band(self, score: object, weak: float, possible: float, strong: float) -> str:
        try:
            value = float(score)
        except Exception:
            return "no_match"
        if not np.isfinite(value):
            return "no_match"
        if value >= strong:
            return "strong_match"
        if value >= possible:
            return "possible_match"
        if value >= weak:
            return "weak_match"
        return "no_match"

    def _quality_terms(self, text: object) -> set[str]:
        """Extract generic comparison terms for relevance gating.

        This is intentionally not a hazard keyword taxonomy. It only checks whether
        the query and retrieved evidence share meaningful lexical anchors. For CJK
        text, character bigrams are added so Chinese/Japanese/Korean reports still
        have a non-empty overlap signal.
        """
        value = clean_text_value(text).lower()
        if not value:
            return set()
        stopwords = {
            "the", "and", "for", "with", "this", "that", "from", "were", "was", "are", "not", "has", "have",
            "had", "will", "can", "could", "should", "would", "there", "their", "they", "them", "into", "onto",
            "near", "area", "employee", "employees", "worker", "workers", "task", "work", "working", "observed",
            "found", "needs", "need", "required", "issue", "safety", "incident", "hazard", "risk", "unsafe", "safe",
        }
        tokens = set()
        for token in re.findall(r"[a-z0-9][a-z0-9_\-/]{2,}", value):
            norm = token.strip("_-/")
            if len(norm) >= 3 and norm not in stopwords:
                tokens.add(norm)
        cjk_chars = re.findall(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", value)
        if cjk_chars:
            tokens.update(cjk_chars)
            tokens.update("".join(cjk_chars[i : i + 2]) for i in range(max(0, len(cjk_chars) - 1)))
        return tokens

    def _combined_matched_text(self, row: pd.Series) -> str:
        parts = [
            row.get("matched_title"),
            row.get("matched_description"),
            row.get("matched_retrieval_text"),
            row.get("matched_category"),
            row.get("matched_audit_type"),
            row.get("matched_risk_theme_name"),
        ]
        return " ".join(clean_text_value(x) for x in parts if clean_text_value(x))

    def _annotate_retrieval_quality(self, matches: pd.DataFrame, index_name: str, query_text: str) -> pd.DataFrame:
        """Annotate raw candidates with strict business-evidence quality signals.

        This method intentionally separates "nearest neighbor" from "usable
        evidence". FAISS/BM25/HNSW always return something if an index is not
        empty. A candidate is accepted only when it is a strong route-specific
        match and shares enough concrete lexical/risk anchors with the query.
        The raw candidates remain available in outputs/tests for debugging.
        """
        if matches is None or matches.empty:
            return pd.DataFrame()
        out = matches.copy()
        thresholds = self._quality_thresholds(index_name)
        query_terms = self._quality_terms(query_text)
        query_term_count = len(query_terms)
        min_words = int(getattr(self.settings, "retrieval_quality_min_evidence_text_words", 3))
        min_overlap_terms = int(thresholds.get("min_overlap_terms", getattr(self.settings, "retrieval_quality_min_overlap_terms", 1)))
        min_overlap_ratio = float(thresholds.get("min_overlap_ratio", getattr(self.settings, "retrieval_quality_min_overlap_ratio", 0.08)))
        require_overlap = bool(thresholds.get("require_overlap", getattr(self.settings, "retrieval_quality_require_overlap", True)))
        require_filter = bool(getattr(self.settings, "enable_retrieval_quality_filter", True))
        require_strong = bool(getattr(self.settings, "retrieval_quality_require_strong_match", True))
        min_band = self._minimum_return_band(index_name)
        if require_strong:
            min_band = "strong_match"
        min_band_rank = self._band_rank(min_band)

        quality_rows: list[dict[str, Any]] = []
        for _, row in out.iterrows():
            evidence_text = self._combined_matched_text(row)
            evidence_terms = self._quality_terms(evidence_text)
            shared = query_terms.intersection(evidence_terms) if query_terms and evidence_terms else set()
            word_count = len(clean_text_value(evidence_text).split())
            overlap_count = len(shared)
            overlap_ratio = float(overlap_count / max(1, query_term_count))
            lexical_ok = (overlap_count >= min_overlap_terms) or (overlap_ratio >= min_overlap_ratio)
            if query_term_count == 0 and not require_overlap:
                lexical_ok = True

            faiss_band = self._score_band(
                row.get("faiss_score"),
                float(thresholds.get("weak_faiss", 0.52)),
                float(thresholds.get("possible_faiss", 0.62)),
                float(thresholds.get("strong_faiss", 0.72)),
            )
            bm25_band = self._score_band(
                row.get("bm25_score"),
                float(thresholds.get("weak_bm25", 8.0)),
                float(thresholds.get("possible_bm25", 24.0)),
                float(thresholds.get("strong_bm25", 45.0)),
            )
            hybrid_band = self._score_band(
                row.get("hybrid_score"),
                float(thresholds.get("weak_hybrid", 0.014)),
                float(thresholds.get("possible_hybrid", 0.026)),
                float(thresholds.get("strong_hybrid", 0.036)),
            )
            band_candidates = [faiss_band, bm25_band, hybrid_band]
            strongest_band = max(band_candidates, key=self._band_rank)
            rank_ok = self._band_rank(strongest_band) >= min_band_rank
            has_faiss = pd.notna(row.get("faiss_score")) if "faiss_score" in row.index else False
            has_bm25 = pd.notna(row.get("bm25_score")) if "bm25_score" in row.index else False
            retriever_agreement = bool(has_faiss and has_bm25)
            strong_semantic_ok = self._band_rank(faiss_band) >= self._band_rank("strong_match")
            strong_keyword_ok = self._band_rank(bm25_band) >= self._band_rank("strong_match")
            strong_hybrid_ok = self._band_rank(hybrid_band) >= self._band_rank("strong_match")
            possible_semantic_ok = self._band_rank(faiss_band) >= self._band_rank("possible_match")
            possible_keyword_ok = self._band_rank(bm25_band) >= self._band_rank("possible_match")

            text_ok = word_count >= min_words or query_term_count > 0
            cross_encoder_score = None
            cross_encoder_pass = None

            if not require_filter:
                accepted = True
                reason = "quality_filter_disabled"
            elif not text_ok:
                accepted = False
                reason = "suppressed_short_or_empty_evidence_text"
            elif not rank_ok:
                accepted = False
                reason = f"suppressed_below_{min_band}"
            elif require_overlap and not lexical_ok:
                accepted = False
                reason = "suppressed_no_shared_risk_anchors"
            else:
                # Strict default acceptance rule:
                # Only strong route-specific matches can become evidence. A strong
                # FAISS match still needs lexical/risk-anchor overlap, because a
                # small route index can make an unrelated record the closest
                # neighbor. Hybrid agreement is useful only when both retrievers
                # contribute at least possible evidence.
                if strong_semantic_ok and lexical_ok:
                    accepted = True
                    reason = "accepted_strong_semantic_similarity_with_risk_anchor_overlap"
                elif strong_keyword_ok and lexical_ok:
                    accepted = True
                    reason = "accepted_strong_keyword_similarity_with_risk_anchor_overlap"
                elif strong_hybrid_ok and lexical_ok and retriever_agreement and (possible_semantic_ok or possible_keyword_ok):
                    accepted = True
                    reason = "accepted_strong_hybrid_agreement_with_risk_anchor_overlap"
                else:
                    accepted = False
                    reason = "suppressed_not_a_strong_relevant_match"

            if accepted and self.cross_encoder_guard is not None:
                try:
                    cross_encoder_score = float(self.cross_encoder_guard.predict([(query_text, evidence_text)])[0])
                    cross_encoder_pass = bool(cross_encoder_score >= float(getattr(self.settings, "cross_encoder_min_score", 0.25)))
                    if not cross_encoder_pass:
                        accepted = False
                        reason = "suppressed_by_cross_encoder_relevance_guard"
                except Exception as exc:
                    cross_encoder_score = None
                    cross_encoder_pass = None
                    # Do not fail the agent because an optional guard failed.
                    reason = f"{reason}; cross_encoder_guard_error={type(exc).__name__}"

            quality_rows.append({
                "retrieval_quality_pass": bool(accepted),
                "retrieval_quality_reason": reason,
                "retrieval_quality_band": strongest_band,
                "faiss_quality_band": faiss_band,
                "bm25_quality_band": bm25_band,
                "hybrid_quality_band": hybrid_band,
                "lexical_overlap_count": int(overlap_count),
                "lexical_overlap_ratio": float(overlap_ratio),
                "shared_quality_terms": sorted(shared)[:20],
                "query_term_count": int(query_term_count),
                "evidence_term_count": int(len(evidence_terms)),
                "retriever_agreement": bool(retriever_agreement),
                "min_return_band_required": min_band,
                "min_overlap_terms_required": int(min_overlap_terms),
                "min_overlap_ratio_required": float(min_overlap_ratio),
                "cross_encoder_score": cross_encoder_score,
                "cross_encoder_pass": cross_encoder_pass,
            })
        quality_df = pd.DataFrame(quality_rows, index=out.index)
        out = pd.concat([out, quality_df], axis=1)
        return out

    def _filter_meaningful_matches(self, matches: pd.DataFrame, index_name: str, top_k: int | None = None) -> pd.DataFrame:
        if matches is None or matches.empty:
            return pd.DataFrame()
        if not bool(getattr(self.settings, "enable_retrieval_quality_filter", True)):
            filtered = matches.copy()
        elif "retrieval_quality_pass" in matches.columns:
            filtered = matches[matches["retrieval_quality_pass"].fillna(False).astype(bool)].copy()
        else:
            filtered = matches.copy()
        if filtered.empty:
            return pd.DataFrame()
        # Keep the ranking order from the retriever, but compact ranks after filtering.
        if "rank" in filtered.columns:
            filtered = filtered.sort_values("rank", ascending=True)
        if top_k is not None:
            filtered = filtered.head(int(top_k))
        filtered = filtered.reset_index(drop=True)
        filtered["rank"] = range(1, len(filtered) + 1)
        return filtered


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


    @staticmethod
    def _has_meaningful_severe_evidence(severe_matches: pd.DataFrame, severe_band: str | None) -> bool:
        """Return True when severe-injury evidence should be shown in the final response.

        FAISS/BM25 always returns nearest neighbors if an index has records, so we
        do not treat every returned row as meaningful. For FAISS/hybrid search,
        the severity band must be at least low/medium/high. For BM25-only mode,
        the method can still return keyword evidence, but the final response will
        describe it as historical similarity rather than prediction.
        """
        if severe_matches is None or severe_matches.empty:
            return False
        band = str(severe_band or "").strip().lower()
        return band in {"low", "medium", "high", "bm25_retrieved_keyword_match", "retrieved_match"}

    @staticmethod
    def _filter_by_source_role(matches: pd.DataFrame, roles: list[str]) -> pd.DataFrame:
        """Return retrieved rows with matched_source_role in the requested role list."""
        if matches is None or matches.empty or "matched_source_role" not in matches.columns:
            return pd.DataFrame()
        role_set = {str(r) for r in roles}
        return matches[matches["matched_source_role"].astype(str).isin(role_set)].copy()

    def _select_injury_evidence_for_response(
        self,
        severe_matches: pd.DataFrame,
        injury_matches: pd.DataFrame,
        severe_band: str,
    ) -> dict:
        """Choose which injury evidence is shown in the business-readable final response.

        If meaningful severe-injury evidence is available, show only a few severe
        injury cases. If severe evidence is not available, show a few normal
        injury cases instead. This keeps the final response focused and prevents
        a long raw dump of all injury matches.
        """
        max_cases = int(getattr(self.settings, "llm_max_evidence_records_per_section", 5))
        severe_found = (
            severe_matches is not None
            and not severe_matches.empty
            and str(severe_band or "").lower() not in {"", "no_match"}
        )
        if severe_found:
            selected = severe_matches.head(max_cases).copy()
            return {
                "evidence_type": "severe_injury",
                "message": "Severe-injury similarity was found. Showing a small set of the closest severe-injury cases.",
                "similarity_band": severe_band,
                "matches": self._records_for_json(selected),
            }

        normal = pd.DataFrame()
        if injury_matches is not None and not injury_matches.empty:
            normal = injury_matches.copy()
            if "matched_severe_actual" in normal.columns:
                severe_flag = normal["matched_severe_actual"].fillna(False).astype(str).str.lower().isin({"true", "1", "yes", "y"})
                normal = normal.loc[~severe_flag].copy()
        return {
            "evidence_type": "normal_injury",
            "message": "No meaningful severe-injury similarity met the response threshold. Showing a small set of closest non-severe injury cases when available.",
            "similarity_band": severe_band or "no_match",
            "matches": self._records_for_json(normal.head(max_cases)),
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
    """Create one business-readable row per analyzed query for CSV exports.

    Raw retrieval records are intentionally excluded from this CSV. Technical
    scores/ranks/row IDs are saved separately under outputs/tests.
    """
    theme = result.get("risk_pattern_classification", {})
    runtime = result.get("runtime", {})
    structured = result.get("structured_evidence_summary", {}) or {}
    user_response = result.get("user_facing_final_response", {}) or {}
    evidence_ids = structured.get("evidence_ids_by_section", {}) or {}
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
        "injury_evidence_index_used": structured.get("injury_similarity_evidence", {}).get("selected_index"),
        "injury_similarity_band": structured.get("injury_similarity_evidence", {}).get("similarity_band"),
        "risk_factors_and_possible_control_gaps": compress_json_field(structured.get("risk_factors_and_possible_control_gaps", [])),
        "recommended_prevention_action_candidates": compress_json_field(structured.get("recommended_prevention_action_candidates", [])),
        "missing_information_to_collect": compress_json_field(structured.get("missing_information_to_collect", [])),
        "evidence_ids_by_section": compress_json_field(evidence_ids),
        "llm_response_status": result.get("llm_final_response", {}).get("status"),
        "llm_model_name": result.get("llm_final_response", {}).get("model_name"),
        "user_facing_response_text": user_response.get("response_text"),
        "llm_response_error": result.get("llm_final_response", {}).get("error"),
    }


def build_analysis_test_outputs(result: dict) -> dict:
    """Split one analysis result into raw, structured, and user-facing test layers."""
    return {
        "raw_retrieval_debug": result.get("raw_retrieval_debug", {}),
        "structured_evidence_summary": result.get("structured_evidence_summary", {}),
        "structured_response_plan": result.get("structured_response_plan", {}),
        "user_facing_final_response": result.get("user_facing_final_response", {}),
    }


def save_analysis_test_outputs(result: dict, tests_root: Path, run_name: str = "single_event", query_id: str | None = None) -> dict:
    """Save one analysis result into separate step-by-step files under outputs/tests.

    Files created:
      - raw_retrieval_debug.json
      - structured_evidence_summary.json
      - structured_response_plan.json
      - user_facing_final_response.json
      - user_facing_final_response.txt
    """
    safe_query = clean_text_value(query_id or result.get("query", {}).get("event_id") or "query").replace("/", "_").replace("\\", "_")
    out_dir = ensure_dir(Path(tests_root) / run_name / safe_query)
    outputs = build_analysis_test_outputs(result)
    paths = {}
    for name, payload in outputs.items():
        path = out_dir / f"{name}.json"
        save_json(payload, path)
        paths[name] = str(path)
    response_text = clean_text_value(outputs.get("user_facing_final_response", {}).get("response_text"))
    txt_path = out_dir / "user_facing_final_response.txt"
    txt_path.write_text(response_text, encoding="utf-8")
    paths["user_facing_final_response_text"] = str(txt_path)
    save_json({"output_dir": str(out_dir), "files": paths}, out_dir / "test_output_manifest.json")
    return {"output_dir": str(out_dir), "files": paths}
