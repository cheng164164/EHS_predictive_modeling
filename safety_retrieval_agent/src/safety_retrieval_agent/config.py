"""Configuration for the Safety Retrieval Agent MVP.

All runnable scripts use this file for defaults, so each script can be executed
without command-line arguments, for example:

    python scripts/00_build_unified_text_events.py
    python scripts/00_prepare_knowledge_base.py
    python scripts/01_build_faiss_indexes.py
    python scripts/02_run_mvp_recommendations.py
    python scripts/predict_single_event.py
    python scripts/run_end_to_end.py

You can edit this file directly or override selected paths/settings with
environment variables.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


PROJECT_NAME = "safety retrieval agent"


def _default_project_root() -> Path:
    # config.py lives at project_root/src/safety_retrieval_agent/config.py
    return Path(__file__).resolve().parents[2]


def _default_output_dir() -> Path:
    return _default_project_root() / "outputs" / PROJECT_NAME


def _path_from_env(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return Path(value) if value else default


def _optional_path_from_env(name: str, default: str | None = None) -> Path | None:
    value = os.getenv(name, default)
    if value is None or str(value).strip() == "":
        return None
    return Path(value)


def _optional_int_from_env(name: str, default: int | None = None) -> int | None:
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        return default
    return int(value)


def _bool_from_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "t"}


def _sequence_from_env(name: str, default: Sequence[str]) -> tuple[str, ...]:
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        return tuple(default)
    return tuple(part.strip() for part in str(value).split(",") if part.strip())


@dataclass(frozen=True)
class Settings:
    """Runtime settings for unified-data build, local retrieval indexing, and MVP analysis."""

    project_root: Path = _default_project_root()
    output_dir: Path = _path_from_env("SAFETY_RETRIEVAL_OUTPUT_DIR", _default_output_dir())

    # ------------------------------------------------------------------
    # Raw source data and unified-event build
    # ------------------------------------------------------------------
    # Put Velocity/Accelerate export CSVs here if safety_text_event.csv.gz is not
    # already available. Expected file names by default:
    # INCIDENT_VIEW.csv, INCIDENTINJURY_VIEW.csv, AUDIT_VIEW.csv, TASK_VIEW.csv,
    # LOCATION_VIEW.csv, LISTITEM_VIEW.csv.
    raw_data_dir: Path = _path_from_env("SAFETY_RETRIEVAL_RAW_DATA_DIR", _default_project_root() / "data" / "raw")
    incident_file_name: str = os.getenv("SAFETY_RETRIEVAL_INCIDENT_FILE", "INCIDENT_VIEW.csv")
    injury_file_name: str = os.getenv("SAFETY_RETRIEVAL_INJURY_FILE", "INCIDENTINJURY_VIEW.csv")
    audit_file_name: str = os.getenv("SAFETY_RETRIEVAL_AUDIT_FILE", "AUDIT_VIEW.csv")
    task_file_name: str = os.getenv("SAFETY_RETRIEVAL_TASK_FILE", "TASK_VIEW.csv")
    location_file_name: str = os.getenv("SAFETY_RETRIEVAL_LOCATION_FILE", "LOCATION_VIEW.csv")
    listitem_file_name: str = os.getenv("SAFETY_RETRIEVAL_LISTITEM_FILE", "LISTITEM_VIEW.csv")

    # Main unified event file consumed by the knowledge-base prep step. If you
    # already have a unified file, set this path to it. If not, run
    # 00_build_unified_text_events.py and it will write to this path.
    input_event_file: Path = _path_from_env(
        "SAFETY_RETRIEVAL_INPUT_FILE",
        _default_output_dir() / "data" / "safety_text_event.csv.gz",
    )
    unified_sample_size: int | None = _optional_int_from_env("SAFETY_RETRIEVAL_UNIFIED_SAMPLE_SIZE", None)
    drop_empty_unified_text: bool = _bool_from_env("SAFETY_RETRIEVAL_DROP_EMPTY_UNIFIED_TEXT", False)
    force_rebuild_unified_event_file: bool = _bool_from_env("SAFETY_RETRIEVAL_FORCE_REBUILD_UNIFIED", False)
    run_unified_builder_if_missing: bool = _bool_from_env("SAFETY_RETRIEVAL_RUN_UNIFIED_IF_MISSING", True)

    # ------------------------------------------------------------------
    # Date filter requested by the project owner
    # ------------------------------------------------------------------
    min_event_date: str = os.getenv("SAFETY_RETRIEVAL_MIN_EVENT_DATE", "2016-01-01")
    max_event_date: str = os.getenv("SAFETY_RETRIEVAL_MAX_EVENT_DATE", "2026-12-31")

    # ------------------------------------------------------------------
    # Embedding model and FAISS settings
    # ------------------------------------------------------------------
    # Default model: BGE-M3 is open-source/free to use under its model license and
    # is strong for multilingual semantic retrieval. Auto backend now defaults to
    # sentence-transformers for stability. FlagEmbedding remains optional, but some
    # FlagEmbedding/transformers version combinations can fail while loading BGE-M3.
    # Primary embedding model. BGE-M3 is the first choice for this project.
    embedding_model_name: str = os.getenv("SAFETY_RETRIEVAL_EMBEDDING_MODEL", "BAAI/bge-m3")

    # Fallback embedding model. If BGE-M3 fails to load or encode in the current
    # environment, the pipeline automatically switches to this model and saves
    # the actual model used in embedding_model_metadata.json.
    embedding_fallback_model_name: str = os.getenv(
        "SAFETY_RETRIEVAL_EMBEDDING_FALLBACK_MODEL",
        "Qwen/Qwen3-Embedding-0.6B",
    )
    enable_embedding_fallback: bool = _bool_from_env("SAFETY_RETRIEVAL_ENABLE_EMBEDDING_FALLBACK", True)
    embedding_fallback_backend: str = os.getenv("SAFETY_RETRIEVAL_EMBEDDING_FALLBACK_BACKEND", "sentence_transformers")

    # Query-time scripts force the same model recorded in FAISS metadata. This
    # prevents invalid searches caused by building the index with Qwen fallback
    # but querying later with BGE-M3, or vice versa.

    # Recommended backend for both BAAI/bge-m3 and Qwen/Qwen3-Embedding-0.6B.
    # It keeps the model local/free and avoids the FlagEmbedding dtype issue seen
    # with some transformers versions.
    embedding_backend: str = os.getenv("SAFETY_RETRIEVAL_EMBEDDING_BACKEND", "sentence_transformers")
    trust_remote_code: bool = _bool_from_env("SAFETY_RETRIEVAL_TRUST_REMOTE_CODE", True)
    embedding_device: str | None = os.getenv("SAFETY_RETRIEVAL_EMBEDDING_DEVICE") or None
    query_instruction: str = os.getenv(
        "SAFETY_RETRIEVAL_QUERY_INSTRUCTION",
        "Represent this sentence for searching relevant safety records: ",
    )
    embedding_batch_size: int = int(os.getenv("SAFETY_RETRIEVAL_EMBEDDING_BATCH_SIZE", "8"))
    embedding_max_length: int = int(os.getenv("SAFETY_RETRIEVAL_EMBEDDING_MAX_LENGTH", "1024"))
    use_fp16: bool = _bool_from_env("SAFETY_RETRIEVAL_USE_FP16", False)
    show_progress: bool = _bool_from_env("SAFETY_RETRIEVAL_SHOW_PROGRESS", True)

    # Retrieval mode used by the agent at query time.
    # Allowed values:
    # - "faiss": semantic vector search only
    # - "bm25": keyword/BM25 search only
    # - "hybrid": FAISS + BM25 merged with reciprocal-rank fusion
    retrieval_mode: str = os.getenv("SAFETY_RETRIEVAL_MODE", "hybrid")

    # FAISS settings. HNSW is the default because it is fast for interactive local
    # retrieval. Use "flat" for exact search during evaluation/debugging.
    faiss_index_type: str = os.getenv("SAFETY_RETRIEVAL_FAISS_INDEX_TYPE", "hnsw")
    hnsw_m: int = int(os.getenv("SAFETY_RETRIEVAL_HNSW_M", "32"))
    hnsw_ef_construction: int = int(os.getenv("SAFETY_RETRIEVAL_HNSW_EF_CONSTRUCTION", "200"))
    hnsw_ef_search: int = int(os.getenv("SAFETY_RETRIEVAL_HNSW_EF_SEARCH", "96"))
    build_reuse_embeddings: bool = _bool_from_env("SAFETY_RETRIEVAL_REUSE_EMBEDDINGS", False)

    # BM25 keyword retrieval settings. The BM25 index is built locally from the
    # same safety knowledge base and is used when retrieval_mode is "bm25" or
    # "hybrid".
    bm25_k1: float = float(os.getenv("SAFETY_RETRIEVAL_BM25_K1", "1.5"))
    bm25_b: float = float(os.getenv("SAFETY_RETRIEVAL_BM25_B", "0.75"))
    bm25_min_df: int = int(os.getenv("SAFETY_RETRIEVAL_BM25_MIN_DF", "2"))
    bm25_max_df: float = float(os.getenv("SAFETY_RETRIEVAL_BM25_MAX_DF", "0.98"))
    bm25_max_features: int = int(os.getenv("SAFETY_RETRIEVAL_BM25_MAX_FEATURES", "250000"))
    bm25_ngram_range: tuple[int, int] = (1, int(os.getenv("SAFETY_RETRIEVAL_BM25_MAX_NGRAM", "2")))

    # Hybrid retrieval candidate pool and fusion settings. Candidate K controls
    # how many records are pulled from each retriever before fusion. Final top-K
    # is still controlled by top_k_* settings below.
    faiss_candidate_k: int = int(os.getenv("SAFETY_RETRIEVAL_FAISS_CANDIDATE_K", "75"))
    bm25_candidate_k: int = int(os.getenv("SAFETY_RETRIEVAL_BM25_CANDIDATE_K", "75"))
    hybrid_rrf_k: int = int(os.getenv("SAFETY_RETRIEVAL_HYBRID_RRF_K", "60"))

    # Records can be capped for a fast smoke test. Leave as None for full build.
    max_records: int | None = _optional_int_from_env("SAFETY_RETRIEVAL_MAX_RECORDS", None)

    # ------------------------------------------------------------------
    # Embedding/retrieval scope
    # ------------------------------------------------------------------
    # Only these source_role values are embedded and indexed. This avoids spending
    # embedding time on generic incident/audit/inspection rows that are not part of
    # the MVP retrieval use case. Edit SAFETY_RETRIEVAL_EMBEDDING_SOURCE_ROLES as
    # a comma-separated environment variable if you want a different scope.
    embedding_source_roles: Sequence[str] = _sequence_from_env(
        "SAFETY_RETRIEVAL_EMBEDDING_SOURCE_ROLES",
        (
            "hazard_identification",
            "near_miss",
            "unsafe_observation",
            "safe_observation",
            "injury",
            "severe_injury",
            "corrective_action",
            "open_corrective_action",
            "overdue_corrective_action",
        ),
    )

    # Generic audit_observation rows can still be useful, but many have empty
    # descriptions. Keep them only when they have a non-empty description by
    # default. This is separate from unsafe/safe observations, which are already
    # included by source_role above.
    include_generic_audit_observations: bool = _bool_from_env(
        "SAFETY_RETRIEVAL_INCLUDE_GENERIC_AUDIT_OBSERVATIONS", True
    )
    require_generic_audit_description: bool = _bool_from_env(
        "SAFETY_RETRIEVAL_REQUIRE_GENERIC_AUDIT_DESCRIPTION", True
    )
    generic_audit_description_min_chars: int = int(
        os.getenv("SAFETY_RETRIEVAL_GENERIC_AUDIT_DESCRIPTION_MIN_CHARS", "1")
    )

    # ------------------------------------------------------------------
    # Theme discovery and profiles
    # ------------------------------------------------------------------
    # Existing theme columns are reused when present; otherwise MiniBatchKMeans
    # over embeddings is used.
    n_themes: int = int(os.getenv("SAFETY_RETRIEVAL_N_THEMES", "80"))
    theme_sample_size_for_labels: int = int(os.getenv("SAFETY_RETRIEVAL_THEME_LABEL_SAMPLE", "250000"))
    theme_min_cluster_size: int = int(os.getenv("SAFETY_RETRIEVAL_THEME_MIN_CLUSTER_SIZE", "10"))
    theme_top_terms: int = int(os.getenv("SAFETY_RETRIEVAL_THEME_TOP_TERMS", "6"))
    theme_representative_events: int = int(os.getenv("SAFETY_RETRIEVAL_THEME_REPRESENTATIVE_EVENTS", "8"))

    # ------------------------------------------------------------------
    # Retrieval defaults for MVP1 outputs
    # ------------------------------------------------------------------
    top_k_severe_injuries: int = int(os.getenv("SAFETY_RETRIEVAL_TOP_K_SEVERE", "5"))
    top_k_similar_events: int = int(os.getenv("SAFETY_RETRIEVAL_TOP_K_EVENTS", "15"))
    top_k_corrective_actions: int = int(os.getenv("SAFETY_RETRIEVAL_TOP_K_ACTIONS", "8"))
    top_k_safe_practices: int = int(os.getenv("SAFETY_RETRIEVAL_TOP_K_SAFE", "5"))

    # Operational score bands for cosine similarity over normalized embeddings.
    high_similarity_threshold: float = float(os.getenv("SAFETY_RETRIEVAL_HIGH_SIM", "0.75"))
    medium_similarity_threshold: float = float(os.getenv("SAFETY_RETRIEVAL_MEDIUM_SIM", "0.60"))
    low_similarity_threshold: float = float(os.getenv("SAFETY_RETRIEVAL_LOW_SIM", "0.45"))

    # ------------------------------------------------------------------
    # Local/free LLM response generation
    # ------------------------------------------------------------------
    # This layer runs AFTER retrieval. It does not rebuild embeddings or indexes.
    # It summarizes retrieved evidence into a readable final response.
    enable_llm_response: bool = _bool_from_env("SAFETY_RETRIEVAL_ENABLE_LLM_RESPONSE", True)

    # Small, free, easy-to-download default instruct model for local response
    # generation. For better quality, change to Qwen/Qwen2.5-1.5B-Instruct or a
    # larger local instruct model if your machine has enough memory.
    llm_model_name: str = os.getenv("SAFETY_RETRIEVAL_LLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    llm_backend: str = os.getenv("SAFETY_RETRIEVAL_LLM_BACKEND", "transformers")
    llm_trust_remote_code: bool = _bool_from_env("SAFETY_RETRIEVAL_LLM_TRUST_REMOTE_CODE", True)
    llm_device_map: str = os.getenv("SAFETY_RETRIEVAL_LLM_DEVICE_MAP", "auto")
    llm_torch_dtype: str = os.getenv("SAFETY_RETRIEVAL_LLM_TORCH_DTYPE", "auto")

    # Generation settings. Deterministic by default for repeatable safety outputs.
    llm_max_input_tokens: int = int(os.getenv("SAFETY_RETRIEVAL_LLM_MAX_INPUT_TOKENS", "4096"))
    llm_max_new_tokens: int = int(os.getenv("SAFETY_RETRIEVAL_LLM_MAX_NEW_TOKENS", "700"))
    llm_temperature: float = float(os.getenv("SAFETY_RETRIEVAL_LLM_TEMPERATURE", "0.2"))
    llm_do_sample: bool = _bool_from_env("SAFETY_RETRIEVAL_LLM_DO_SAMPLE", False)

    # Prompt compaction limits so the LLM receives concise evidence rather than
    # hundreds of raw records.
    llm_max_evidence_records_per_section: int = int(os.getenv("SAFETY_RETRIEVAL_LLM_MAX_EVIDENCE_PER_SECTION", "5"))
    llm_max_action_candidates: int = int(os.getenv("SAFETY_RETRIEVAL_LLM_MAX_ACTION_CANDIDATES", "8"))
    llm_max_missing_info_prompts: int = int(os.getenv("SAFETY_RETRIEVAL_LLM_MAX_MISSING_PROMPTS", "8"))

    # If the local model cannot load in a given environment, keep returning a
    # deterministic structured summary and expose the error in llm_final_response.
    llm_allow_heuristic_fallback: bool = _bool_from_env("SAFETY_RETRIEVAL_LLM_ALLOW_HEURISTIC_FALLBACK", True)

    # Text quality filters.
    min_text_chars: int = int(os.getenv("SAFETY_RETRIEVAL_MIN_TEXT_CHARS", "20"))
    min_text_words: int = int(os.getenv("SAFETY_RETRIEVAL_MIN_TEXT_WORDS", "3"))
    max_retrieval_text_chars: int = int(os.getenv("SAFETY_RETRIEVAL_MAX_RETRIEVAL_TEXT_CHARS", "4000"))

    # Knowledge-base CSV export. The pickle is the primary local pipeline input;
    # full CSV export can be slow for the full historical corpus.
    prepare_save_full_csv: bool = _bool_from_env("SAFETY_RETRIEVAL_SAVE_FULL_KB_CSV", False)

    # Batch recommendation examples. Use recent non-injury records by default.
    recommendation_sample_size: int = int(os.getenv("SAFETY_RETRIEVAL_RECOMMENDATION_SAMPLE_SIZE", "100"))
    recommendation_query_file: Path | None = _optional_path_from_env("SAFETY_RETRIEVAL_QUERY_FILE", None)
    recommendation_source_role: str | None = os.getenv("SAFETY_RETRIEVAL_RECOMMENDATION_SOURCE_ROLE") or None
    recommendation_recent: bool = _bool_from_env("SAFETY_RETRIEVAL_RECOMMENDATION_RECENT", False)

    # Optional fixed examples for scripts/02_run_mvp_recommendations.py.
    # Set use_configured_test_queries=True to run these manual examples instead
    # of sampling records from the knowledge base. Edit these directly when you
    # want repeatable local tests without maintaining a separate query CSV file.
    use_configured_test_queries: bool = _bool_from_env("SAFETY_RETRIEVAL_USE_CONFIGURED_TEST_QUERIES", False)
    configured_test_queries: Sequence[dict] = (
        {
            "query_id": "test_001",
            "query_text": "Forklift reversed near a loading dock and almost struck a pedestrian walking through the area.",
            "site": "Example Site",
            "department": "Warehouse",
            "source_type": "near_miss",
        },
        {
            "query_id": "test_002",
            "query_text": "An extension cord was stretched across a walkway creating a trip hazard for employees.",
            "site": "Example Site",
            "department": "Maintenance",
            "source_type": "hazard_identification",
        },
    )

    random_seed: int = int(os.getenv("SAFETY_RETRIEVAL_RANDOM_SEED", "42"))

    # Defaults for scripts/predict_single_event.py so it also runs without args.
    single_event_text: str = os.getenv(
        "SAFETY_RETRIEVAL_SINGLE_EVENT_TEXT",
        "Forklift reversed near a loading dock and almost struck a pedestrian walking through the area.",
    )
    single_event_site: str | None = os.getenv("SAFETY_RETRIEVAL_SINGLE_EVENT_SITE") or None
    single_event_department: str | None = os.getenv("SAFETY_RETRIEVAL_SINGLE_EVENT_DEPARTMENT") or "Warehouse"
    single_event_source_type: str | None = os.getenv("SAFETY_RETRIEVAL_SINGLE_EVENT_SOURCE_TYPE") or "near_miss"
    single_event_id: str | None = os.getenv("SAFETY_RETRIEVAL_SINGLE_EVENT_ID") or "manual_query_001"

    # Source-type groupings. These are source roles, not hazard keyword taxonomies.
    injury_source_types: Sequence[str] = ("incident",)
    leading_source_types: Sequence[str] = ("hazard_identification", "near_miss", "audit")
    corrective_action_source_types: Sequence[str] = ("task",)

    def raw_incident_path(self) -> Path:
        return self.raw_data_dir / self.incident_file_name

    def raw_injury_path(self) -> Path:
        return self.raw_data_dir / self.injury_file_name

    def raw_audit_path(self) -> Path:
        return self.raw_data_dir / self.audit_file_name

    def raw_task_path(self) -> Path:
        return self.raw_data_dir / self.task_file_name

    def raw_location_path(self) -> Path:
        return self.raw_data_dir / self.location_file_name

    def raw_listitem_path(self) -> Path:
        return self.raw_data_dir / self.listitem_file_name

    def unified_event_path(self) -> Path:
        return self.input_event_file

    def location_hierarchy_path(self) -> Path:
        return self.output_dir / "data" / "location_hierarchy.csv"

    def knowledge_base_path(self) -> Path:
        return self.output_dir / "data" / "safety_knowledge_base.pkl"

    def knowledge_base_csv_path(self) -> Path:
        return self.output_dir / "data" / "safety_knowledge_base.csv.gz"

    def embedding_scope_path(self) -> Path:
        return self.output_dir / "data" / "safety_embedding_scope.pkl"

    def embedding_scope_csv_path(self) -> Path:
        return self.output_dir / "data" / "safety_embedding_scope.csv.gz"

    def embedding_scope_summary_path(self) -> Path:
        return self.output_dir / "data" / "embedding_scope_summary.json"

    def embedding_scope_counts_path(self) -> Path:
        return self.output_dir / "data" / "embedding_scope_counts_by_role.csv"

    def enriched_knowledge_base_path(self) -> Path:
        return self.output_dir / "data" / "safety_knowledge_base_with_themes.pkl"

    def theme_profiles_path(self) -> Path:
        return self.output_dir / "data" / "safety_theme_profiles.pkl"

    def indexes_dir(self) -> Path:
        return self.output_dir / "faiss_indexes"

    def bm25_dir(self) -> Path:
        return self.output_dir / "bm25_indexes"

    def embeddings_dir(self) -> Path:
        return self.output_dir / "embeddings"

    def models_dir(self) -> Path:
        return self.output_dir / "models"

    def recommendations_dir(self) -> Path:
        return self.output_dir / "recommendations"

    def logs_dir(self) -> Path:
        return self.output_dir / "logs"


def get_settings() -> Settings:
    return Settings()
