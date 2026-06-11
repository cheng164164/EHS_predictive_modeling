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

import json
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


def _default_aml_artifact_output_uri() -> str:
    # Stable AML datastore path used by long-running jobs for resumable chunks and
    # managed build artifacts. Keep this space-free to avoid URI/mount quirks.
    return (
        "azureml://datastores/workspaceblobstore/paths/safety-retrieval-agent/"
        "managed-batch-artifacts/"
    )


def _default_aml_runtime_artifact_uri() -> str:
    # Full datastore URI for interactive/runtime artifact reads with azureml-fsspec.
    # The short azureml://datastores/... URI works well inside command jobs, but
    # interactive scripts generally need the fully qualified URI.
    return (
        "azureml://subscriptions/7f07baf7-8bba-4b88-b300-74ba5b15f52d/"
        "resourcegroups/EHS-Safety/workspaces/ehs-safety-aml/"
        "datastores/workspaceblobstore/paths/safety-retrieval-agent/"
        "managed-batch-artifacts/"
    )


def _uri_join(base: str, *parts: object) -> str:
    text = str(base).rstrip("/")
    clean = [str(p).strip("/") for p in parts if str(p).strip("/")]
    return "/".join([text, *clean]) if clean else text


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


def _embedding_model_default() -> str:
    """Return the default embedding model from a simple model switch.

    SAFETY_RETRIEVAL_EMBEDDING_MODEL still overrides this directly.
    SAFETY_RETRIEVAL_EMBEDDING_MODEL_CHOICE can be one of:
      - bge_m3 / bge: BAAI/bge-m3, default, strongest multilingual local retrieval option here
      - basic_sentence_transformer / basic / minilm: sentence-transformers/all-MiniLM-L6-v2
      - mpnet: sentence-transformers/all-mpnet-base-v2
    """
    choice = os.getenv("SAFETY_RETRIEVAL_EMBEDDING_MODEL_CHOICE", "bge_m3").strip().lower().replace("-", "_")
    presets = {
        "bge": "BAAI/bge-m3",
        "bge_m3": "BAAI/bge-m3",
        "basic": "sentence-transformers/all-MiniLM-L6-v2",
        "basic_sentence_transformer": "sentence-transformers/all-MiniLM-L6-v2",
        "minilm": "sentence-transformers/all-MiniLM-L6-v2",
        "all_minilm_l6_v2": "sentence-transformers/all-MiniLM-L6-v2",
        "mpnet": "sentence-transformers/all-mpnet-base-v2",
        "all_mpnet_base_v2": "sentence-transformers/all-mpnet-base-v2",
    }
    return presets.get(choice, choice or "BAAI/bge-m3")


def _query_instruction_default() -> str:
    """Return a query instruction appropriate for the selected embedding model.

    BGE-style retrieval models benefit from a retrieval instruction. Basic
    SentenceTransformer models generally do not need one, so keep it blank for
    those presets unless SAFETY_RETRIEVAL_QUERY_INSTRUCTION is explicitly set.
    """
    if "SAFETY_RETRIEVAL_QUERY_INSTRUCTION" in os.environ:
        return os.environ["SAFETY_RETRIEVAL_QUERY_INSTRUCTION"]
    choice = os.getenv("SAFETY_RETRIEVAL_EMBEDDING_MODEL_CHOICE", "bge_m3").strip().lower().replace("-", "_")
    model = os.getenv("SAFETY_RETRIEVAL_EMBEDDING_MODEL", _embedding_model_default()).lower()
    if choice in {"basic", "basic_sentence_transformer", "minilm", "all_minilm_l6_v2", "mpnet", "all_mpnet_base_v2"}:
        return ""
    if "all-minilm" in model or "all-mpnet" in model or "sentence-transformers/" in model:
        return ""
    return "Represent this sentence for searching relevant safety records: "


@dataclass(frozen=True)
class Settings:
    """Runtime settings for unified-data build, local retrieval indexing, and MVP analysis."""

    project_root: Path = _default_project_root()
    output_dir: Path = _path_from_env("SAFETY_RETRIEVAL_OUTPUT_DIR", _default_output_dir())

    # Test/debug outputs are intentionally separated from production recommendation
    # outputs. These files make it easier to inspect the retrieval pipeline step by
    # step without mixing raw technical evidence with user-facing responses.
    test_output_dir: Path = _path_from_env(
        "SAFETY_RETRIEVAL_TEST_OUTPUT_DIR",
        _default_project_root() / "outputs" / "tests",
    )

    # ------------------------------------------------------------------
    # Runtime artifact access for prediction/recommendation scripts
    # ------------------------------------------------------------------
    # Build jobs write large artifacts to the AML datastore. For interactive
    # testing, keep those artifacts in the datastore and read them directly via
    # azureml-fsspec instead of downloading them into the VS Code workspace.
    # Allowed values: "azureml"/"datastore" or "local".
    artifact_read_mode: str = os.getenv("SAFETY_RETRIEVAL_ARTIFACT_READ_MODE", "auto")

    # Fully-qualified datastore URI that contains data/, faiss_indexes/,
    # bm25_indexes/, models/, and embeddings/. This should match the current
    # workspaceblobstore path shown in Azure ML Studio.
    artifact_azureml_uri: str = os.getenv(
        "SAFETY_RETRIEVAL_ARTIFACT_AZUREML_URI",
        _default_aml_runtime_artifact_uri(),
    )

    # Small recommendation outputs are still written locally so you can review
    # JSON/CSV responses in the VS Code workspace. Heavy retrieval artifacts stay
    # in the datastore.
    local_recommendation_outputs: bool = _bool_from_env("SAFETY_RETRIEVAL_LOCAL_RECOMMENDATION_OUTPUTS", True)

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
    embedding_model_choice: str = os.getenv("SAFETY_RETRIEVAL_EMBEDDING_MODEL_CHOICE", "bge_m3")
    embedding_model_name: str = os.getenv("SAFETY_RETRIEVAL_EMBEDDING_MODEL", _embedding_model_default())

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
    query_instruction: str = _query_instruction_default()
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

    # Purpose-specific retrieval index names. The broad all_events index is no
    # longer built because it mixes injuries, audits, hazards, and tasks and makes
    # response interpretation noisier. Runtime agent retrieval uses these clear
    # indexes by purpose/source role.
    retrieval_index_names: Sequence[str] = _sequence_from_env(
        "SAFETY_RETRIEVAL_INDEX_NAMES",
        (
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
        ),
    )

    # Records can be capped for a fast smoke test. Leave as None for full build.
    max_records: int | None = _optional_int_from_env("SAFETY_RETRIEVAL_MAX_RECORDS", None)


    # ------------------------------------------------------------------
    # Azure ML batch job settings for resumable CPU embedding/index builds
    # ------------------------------------------------------------------
    # These settings are used only by the Azure ML submission scripts and the
    # split 01a/01b batch scripts. They do not affect existing 00/01/02 scripts.
    aml_subscription_id: str = os.getenv("AZURE_SUBSCRIPTION_ID", "7f07baf7-8bba-4b88-b300-74ba5b15f52d")
    aml_resource_group: str = os.getenv("AZURE_RESOURCE_GROUP", "EHS-Safety")
    aml_workspace_name: str = os.getenv("AZUREML_WORKSPACE_NAME", "ehs-safety-aml")
    aml_compute_name: str = os.getenv("SAFETY_RETRIEVAL_AML_COMPUTE", "Tan-dev-cluster")
    aml_experiment_name: str = os.getenv("SAFETY_RETRIEVAL_AML_EXPERIMENT", "safety-retrieval-agent")

    # Environment registration. The submit scripts create/update this Azure ML
    # environment from environments/safety_retrieval_agent_cpu.yml.
    aml_environment_name: str = os.getenv("SAFETY_RETRIEVAL_AML_ENV_NAME", "safety-retrieval-agent-cpu")
    aml_environment_version: str = os.getenv("SAFETY_RETRIEVAL_AML_ENV_VERSION", "1")
    aml_base_image: str = os.getenv(
        "SAFETY_RETRIEVAL_AML_BASE_IMAGE",
        "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
    )
    aml_register_environment: bool = _bool_from_env("SAFETY_RETRIEVAL_AML_REGISTER_ENV", True)

    # Stable datastore artifact path used by long, resumable Azure ML jobs.
    # Embedding chunks and managed job artifacts are written here first. This path
    # is intentionally space-free and separate from the Azure ML Studio Notebooks
    # file-explorer path, so failed/retried jobs can resume from previous chunks.
    aml_artifact_output_uri: str = os.getenv(
        "SAFETY_RETRIEVAL_AML_ARTIFACT_URI",
        _default_aml_artifact_output_uri(),
    )

    # Backward-compatible alias used by the older two-step submit scripts. By
    # default, it points at the managed artifact path.
    aml_output_uri: str = os.getenv(
        "SAFETY_RETRIEVAL_AML_OUTPUT_URI",
        os.getenv("SAFETY_RETRIEVAL_AML_ARTIFACT_URI", _default_aml_artifact_output_uri()),
    )

    # Optional prepared input folder. By default, the submit script uploads/mounts
    # the current local output_dir as a read-only input so the job can read
    # safety_knowledge_base.pkl without needing to upload it as source code.
    aml_use_local_output_dir_as_input: bool = _bool_from_env("SAFETY_RETRIEVAL_AML_USE_LOCAL_OUTPUT_AS_INPUT", True)
    aml_prepared_output_uri: str | None = os.getenv("SAFETY_RETRIEVAL_AML_PREPARED_OUTPUT_URI") or None
    prepared_output_dir: Path | None = _optional_path_from_env("SAFETY_RETRIEVAL_PREPARED_OUTPUT_DIR", None)
    aml_input_mode: str = os.getenv("SAFETY_RETRIEVAL_AML_INPUT_MODE", "ro_mount")
    aml_output_mode: str = os.getenv("SAFETY_RETRIEVAL_AML_OUTPUT_MODE", "rw_mount")
    # For the second job, optionally download the existing stable output folder
    # first and copy it into the writable output mount before building indexes.
    # This is safer if your AML output mount does not expose preexisting files.
    aml_index_download_existing_output_first: bool = _bool_from_env(
        "SAFETY_RETRIEVAL_AML_INDEX_DOWNLOAD_EXISTING_OUTPUT_FIRST", True
    )

    # CPU/threading. For your Standard_DS4_v2 cluster, default cores per node is 8.
    # At runtime the scripts also inspect os.cpu_count() and set PyTorch/FAISS
    # threads accordingly. cpu_thread_count=0 means use all CPUs visible to the job.
    aml_cpu_cores_per_node: int = int(os.getenv("SAFETY_RETRIEVAL_AML_CPU_CORES", "8"))
    use_all_available_cpus: bool = _bool_from_env("SAFETY_RETRIEVAL_USE_ALL_CPUS", True)
    cpu_thread_count: int = int(os.getenv("SAFETY_RETRIEVAL_CPU_THREAD_COUNT", "0"))
    torch_interop_thread_count: int = int(os.getenv("SAFETY_RETRIEVAL_TORCH_INTEROP_THREADS", "2"))
    force_cpu_embedding: bool = _bool_from_env("SAFETY_RETRIEVAL_FORCE_CPU_EMBEDDING", True)

    # Resumable embedding-chunk settings. Chunks are used only as checkpoints;
    # 01b merges them and builds normal FAISS/BM25 indexes for search.
    enable_embedding_chunking: bool = _bool_from_env("SAFETY_RETRIEVAL_ENABLE_EMBEDDING_CHUNKING", True)
    embedding_chunk_size: int = int(os.getenv("SAFETY_RETRIEVAL_EMBEDDING_CHUNK_SIZE", "5000"))
    skip_existing_embedding_chunks: bool = _bool_from_env("SAFETY_RETRIEVAL_SKIP_EXISTING_CHUNKS", True)
    force_rebuild_embeddings: bool = _bool_from_env("SAFETY_RETRIEVAL_FORCE_REBUILD_EMBEDDINGS", False)
    embedding_chunk_start_index: int = int(os.getenv("SAFETY_RETRIEVAL_CHUNK_START_INDEX", "0"))
    embedding_chunk_end_index: int | None = _optional_int_from_env("SAFETY_RETRIEVAL_CHUNK_END_INDEX", None)
    fail_if_embedding_chunks_incomplete: bool = _bool_from_env("SAFETY_RETRIEVAL_FAIL_IF_CHUNKS_INCOMPLETE", True)
    merge_embedding_chunks_after_generation: bool = _bool_from_env("SAFETY_RETRIEVAL_MERGE_CHUNKS_AFTER_GENERATION", False)

    # Azure ML job display names. You can change these without changing script code.
    aml_embedding_job_display_name: str = os.getenv("SAFETY_RETRIEVAL_AML_EMBED_JOB_NAME", "sra-generate-embedding-chunks")
    aml_index_job_display_name: str = os.getenv("SAFETY_RETRIEVAL_AML_INDEX_JOB_NAME", "sra-build-faiss-bm25-indexes")

    # Full Azure ML batch job. This single command job runs resumable
    # embedding-chunk generation and FAISS/BM25 index creation. Large artifacts
    # remain in the configured Azure ML datastore and runtime scripts read them
    # directly.
    aml_full_job_display_name: str = os.getenv("SAFETY_RETRIEVAL_AML_FULL_JOB_NAME", "sra-full-embedding-index")
    aml_full_copy_existing_output_first: bool = _bool_from_env("SAFETY_RETRIEVAL_AML_FULL_COPY_EXISTING_OUTPUT_FIRST", False)

    # Optional true filesystem path for a workspace-mounted folder, retained only
    # for backward-compatible diagnostics. Runtime artifact reads should use
    # artifact_read_mode="azureml" instead of copying large files locally.
    workspace_local_output_dir: Path | None = _optional_path_from_env(
        "SAFETY_RETRIEVAL_WORKSPACE_LOCAL_OUTPUT_DIR",
        "/home/azureuser/cloudfiles/code/Users/tan.cheng/EHS_predictive_modeling/safety_retrieval_agent/outputs/safety retrieval agent",
    )

    # ------------------------------------------------------------------
    # Local runtime artifact cache for faster interactive testing
    # ------------------------------------------------------------------
    # Azure ML build jobs keep large artifacts in workspaceblobstore. For faster
    # local/interactive testing, scripts/01c_download_runtime_artifacts.py copies
    # the runtime artifacts needed by the agent into these local folders. The
    # default artifact_read_mode="auto" uses these local artifacts when present;
    # otherwise it falls back to the Azure ML datastore URI.
    local_runtime_indexes_dir_name: str = os.getenv("SAFETY_RETRIEVAL_LOCAL_INDEXES_DIR_NAME", "indexes")
    local_runtime_cache_overwrite: bool = _bool_from_env("SAFETY_RETRIEVAL_LOCAL_CACHE_OVERWRITE", True)
    aml_download_job_display_name: str = os.getenv("SAFETY_RETRIEVAL_AML_DOWNLOAD_JOB_NAME", "sra-download-runtime-artifacts")

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
        os.getenv("SAFETY_RETRIEVAL_GENERIC_AUDIT_DESCRIPTION_MIN_CHARS", "20")
    )
    generic_audit_description_min_words: int = int(
        os.getenv("SAFETY_RETRIEVAL_GENERIC_AUDIT_DESCRIPTION_MIN_WORDS", "3")
    )

    # ------------------------------------------------------------------
    # Theme discovery and profiles
    # ------------------------------------------------------------------
    # Existing theme columns are reused when present; otherwise MiniBatchKMeans
    # over embeddings is used.
    n_themes: int = int(os.getenv("SAFETY_RETRIEVAL_N_THEMES", "80"))
    theme_sample_size_for_labels: int = int(os.getenv("SAFETY_RETRIEVAL_THEME_LABEL_SAMPLE", "250000"))
    theme_min_cluster_size: int = int(os.getenv("SAFETY_RETRIEVAL_THEME_MIN_CLUSTER_SIZE", "10"))
    theme_top_terms: int = int(os.getenv("SAFETY_RETRIEVAL_THEME_TOP_TERMS", "8"))
    theme_representative_events: int = int(os.getenv("SAFETY_RETRIEVAL_THEME_REPRESENTATIVE_EVENTS", "8"))

    # ------------------------------------------------------------------
    # Retrieval defaults for MVP1 outputs
    # ------------------------------------------------------------------
    top_k_severe_injuries: int = int(os.getenv("SAFETY_RETRIEVAL_TOP_K_SEVERE", "3"))
    top_k_similar_events: int = int(os.getenv("SAFETY_RETRIEVAL_TOP_K_EVENTS", "10"))
    top_k_corrective_actions: int = int(os.getenv("SAFETY_RETRIEVAL_TOP_K_ACTIONS", "5"))
    top_k_safe_practices: int = int(os.getenv("SAFETY_RETRIEVAL_TOP_K_SAFE", "3"))

    # Operational score bands for cosine similarity over normalized embeddings.
    high_similarity_threshold: float = float(os.getenv("SAFETY_RETRIEVAL_HIGH_SIM", "0.75"))
    medium_similarity_threshold: float = float(os.getenv("SAFETY_RETRIEVAL_MEDIUM_SIM", "0.60"))
    low_similarity_threshold: float = float(os.getenv("SAFETY_RETRIEVAL_LOW_SIM", "0.45"))

    # ------------------------------------------------------------------
    # Retrieval quality gating
    # ------------------------------------------------------------------
    # FAISS/BM25 will always return nearest records if an index is non-empty.
    # These settings control whether those raw candidates are strong enough to be
    # used in structured evidence and in the final LLM response. Raw rejected
    # candidates are still saved under outputs/tests for debugging.
    enable_retrieval_quality_filter: bool = _bool_from_env("SAFETY_RETRIEVAL_ENABLE_QUALITY_FILTER", True)
    retrieval_quality_return_min_band: str = os.getenv("SAFETY_RETRIEVAL_QUALITY_RETURN_MIN_BAND", "strong_match")
    retrieval_quality_candidate_multiplier: int = int(os.getenv("SAFETY_RETRIEVAL_QUALITY_CANDIDATE_MULTIPLIER", "3"))
    retrieval_quality_max_raw_candidates: int = int(os.getenv("SAFETY_RETRIEVAL_QUALITY_MAX_RAW_CANDIDATES", "60"))
    retrieval_quality_min_evidence_text_words: int = int(os.getenv("SAFETY_RETRIEVAL_QUALITY_MIN_EVIDENCE_WORDS", "3"))
    retrieval_quality_min_overlap_terms: int = int(os.getenv("SAFETY_RETRIEVAL_QUALITY_MIN_OVERLAP_TERMS", "1"))
    retrieval_quality_min_overlap_ratio: float = float(os.getenv("SAFETY_RETRIEVAL_QUALITY_MIN_OVERLAP_RATIO", "0.08"))

    # Strict business-evidence gate. When True, only strong route-specific
    # matches can be accepted into structured evidence and the final LLM prompt.
    # Raw candidates are still saved for debugging.
    retrieval_quality_require_strong_match: bool = _bool_from_env("SAFETY_RETRIEVAL_QUALITY_REQUIRE_STRONG", True)

    # Require the query and evidence to share concrete lexical/risk anchors.
    # This prevents a close-but-wrong nearest neighbor from being treated as
    # evidence just because it is the closest record in a small index.
    retrieval_quality_require_overlap: bool = _bool_from_env("SAFETY_RETRIEVAL_QUALITY_REQUIRE_OVERLAP", True)
    retrieval_quality_min_critical_overlap_terms: int = int(os.getenv("SAFETY_RETRIEVAL_QUALITY_MIN_CRITICAL_OVERLAP_TERMS", "2"))
    retrieval_quality_min_critical_overlap_ratio: float = float(os.getenv("SAFETY_RETRIEVAL_QUALITY_MIN_CRITICAL_OVERLAP_RATIO", "0.12"))

    # Optional second-stage open-source cross-encoder relevance check. Disabled
    # by default because it adds latency and downloads another model. It can be
    # enabled later for difficult cases after the strong score/overlap gate.
    enable_cross_encoder_relevance_check: bool = _bool_from_env("SAFETY_RETRIEVAL_ENABLE_CROSS_ENCODER_GUARD", False)
    cross_encoder_model_name: str = os.getenv("SAFETY_RETRIEVAL_CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    cross_encoder_min_score: float = float(os.getenv("SAFETY_RETRIEVAL_CROSS_ENCODER_MIN_SCORE", "0.25"))

    # Optional JSON override for route thresholds. Example:
    # SAFETY_RETRIEVAL_ROUTE_QUALITY_OVERRIDES_JSON='{"severe_injuries": {"possible_faiss": 0.65}}'
    # Each route can override weak_faiss, possible_faiss, strong_faiss,
    # weak_bm25, possible_bm25, strong_bm25, weak_hybrid, possible_hybrid,
    # strong_hybrid, return_min_band, min_overlap_terms, and min_overlap_ratio.
    route_quality_overrides_json: str = os.getenv("SAFETY_RETRIEVAL_ROUTE_QUALITY_OVERRIDES_JSON", "")

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
    recommendation_sample_size: int = int(os.getenv("SAFETY_RETRIEVAL_RECOMMENDATION_SAMPLE_SIZE", "10"))
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

    def local_runtime_root(self) -> Path:
        return self.output_dir

    def local_runtime_indexes_root(self) -> Path:
        return self.output_dir / self.local_runtime_indexes_dir_name

    def local_runtime_faiss_indexes_dir(self) -> Path:
        return self.local_runtime_indexes_root() / "faiss_indexes"

    def local_runtime_bm25_indexes_dir(self) -> Path:
        return self.local_runtime_indexes_root() / "bm25_indexes"

    def local_runtime_models_dir(self) -> Path:
        return self.output_dir / "models"

    def local_runtime_data_dir(self) -> Path:
        return self.output_dir / "data"

    def local_runtime_artifacts_available(self) -> bool:
        """Return True when the local cache has enough files for agent runtime."""
        return (
            self.local_runtime_faiss_indexes_dir().exists()
            and self.local_runtime_bm25_indexes_dir().exists()
            and self.local_runtime_models_dir().exists()
            and self.local_runtime_data_dir().exists()
        )

    def use_local_runtime_artifacts(self) -> bool:
        mode = str(self.artifact_read_mode or "auto").strip().lower()
        if mode in {"local", "local_cache", "cache", "workspace"}:
            return True
        if mode in {"auto", "prefer_local"}:
            return self.local_runtime_artifacts_available()
        return False

    def use_azureml_artifacts(self) -> bool:
        mode = str(self.artifact_read_mode or "auto").strip().lower()
        if mode in {"azureml", "datastore", "remote", "azureml_datastore"}:
            return True
        if mode in {"auto", "prefer_local"}:
            return not self.local_runtime_artifacts_available()
        return False

    def artifact_root(self) -> str | Path:
        """Return the root used by runtime agent reads.

        In auto/local mode, the agent uses locally downloaded runtime artifacts
        when they are present. Otherwise it reads from the configured Azure ML
        datastore URI through azureml-fsspec.
        """
        if self.use_local_runtime_artifacts():
            return self.local_runtime_root()
        return str(self.artifact_azureml_uri).rstrip("/")

    def artifact_path(self, *parts: object) -> str | Path:
        root = self.artifact_root()
        if isinstance(root, Path):
            out = root
            for part in parts:
                out = out / str(part)
            return out
        return _uri_join(str(root), *parts)

    def artifact_data_path(self, filename: str) -> str | Path:
        if self.use_local_runtime_artifacts():
            return self.local_runtime_data_dir() / filename
        return self.artifact_path("data", filename)

    def artifact_indexes_dir(self) -> str | Path:
        if self.use_local_runtime_artifacts():
            return self.local_runtime_faiss_indexes_dir()
        return self.artifact_path("faiss_indexes")

    def artifact_bm25_dir(self) -> str | Path:
        if self.use_local_runtime_artifacts():
            return self.local_runtime_bm25_indexes_dir()
        return self.artifact_path("bm25_indexes")

    def artifact_models_dir(self) -> str | Path:
        if self.use_local_runtime_artifacts():
            return self.local_runtime_models_dir()
        return self.artifact_path("models")

    def artifact_embeddings_dir(self) -> str | Path:
        return self.artifact_path("embeddings")

    def artifact_theme_profiles_path(self) -> str | Path:
        return self.artifact_data_path("safety_theme_profiles.pkl")

    def artifact_enriched_knowledge_base_path(self) -> str | Path:
        return self.artifact_data_path("safety_knowledge_base_with_themes.pkl")

    def artifact_knowledge_base_path(self) -> str | Path:
        return self.artifact_data_path("safety_knowledge_base.pkl")

    def route_quality_thresholds(self, index_name: str) -> dict:
        """Return tunable quality-gating thresholds for a retrieval route.

        Defaults are intentionally conservative. FAISS/BM25/HNSW always return
        nearest records, but only strong, route-appropriate matches should become
        business evidence. Tune with SAFETY_RETRIEVAL_ROUTE_QUALITY_OVERRIDES_JSON
        when validation shows a route is too strict or too permissive.
        """
        route = str(index_name or "").strip().lower()
        default_return_band = "strong_match" if self.retrieval_quality_require_strong_match else self.retrieval_quality_return_min_band
        base = {
            "weak_faiss": 0.52,
            "possible_faiss": 0.62,
            "strong_faiss": 0.72,
            "weak_bm25": 8.0,
            "possible_bm25": 24.0,
            "strong_bm25": 45.0,
            "weak_hybrid": 0.014,
            "possible_hybrid": 0.026,
            "strong_hybrid": 0.036,
            "return_min_band": default_return_band,
            "min_overlap_terms": self.retrieval_quality_min_overlap_terms,
            "min_overlap_ratio": self.retrieval_quality_min_overlap_ratio,
            "require_overlap": self.retrieval_quality_require_overlap,
        }
        route_defaults = {
            # Injury evidence must be very conservative. If there is no truly
            # comparable injury mechanism, the correct response is "no strong
            # injury similarity found".
            "severe_injuries": {
                "weak_faiss": 0.60,
                "possible_faiss": 0.68,
                "strong_faiss": 0.76,
                "weak_bm25": 15.0,
                "possible_bm25": 45.0,
                "strong_bm25": 80.0,
                "weak_hybrid": 0.018,
                "possible_hybrid": 0.030,
                "strong_hybrid": 0.040,
                "return_min_band": "strong_match",
                "min_overlap_terms": self.retrieval_quality_min_critical_overlap_terms,
                "min_overlap_ratio": self.retrieval_quality_min_critical_overlap_ratio,
                "require_overlap": True,
            },
            "all_injuries": {
                "weak_faiss": 0.58,
                "possible_faiss": 0.66,
                "strong_faiss": 0.74,
                "weak_bm25": 12.0,
                "possible_bm25": 35.0,
                "strong_bm25": 65.0,
                "weak_hybrid": 0.017,
                "possible_hybrid": 0.029,
                "strong_hybrid": 0.039,
                "return_min_band": "strong_match",
                "min_overlap_terms": self.retrieval_quality_min_critical_overlap_terms,
                "min_overlap_ratio": self.retrieval_quality_min_critical_overlap_ratio,
                "require_overlap": True,
            },
            # Full corrective-action library can be somewhat broader than open/
            # overdue actions, but still requires strong relevance.
            "corrective_actions": {
                "weak_faiss": 0.56,
                "possible_faiss": 0.64,
                "strong_faiss": 0.72,
                "weak_bm25": 10.0,
                "possible_bm25": 30.0,
                "strong_bm25": 60.0,
                "weak_hybrid": 0.016,
                "possible_hybrid": 0.028,
                "strong_hybrid": 0.038,
                "return_min_band": "strong_match",
                "min_overlap_terms": self.retrieval_quality_min_critical_overlap_terms,
                "min_overlap_ratio": self.retrieval_quality_min_critical_overlap_ratio,
                "require_overlap": True,
            },
            "open_corrective_actions": {
                "weak_faiss": 0.60,
                "possible_faiss": 0.68,
                "strong_faiss": 0.76,
                "weak_bm25": 15.0,
                "possible_bm25": 45.0,
                "strong_bm25": 80.0,
                "weak_hybrid": 0.018,
                "possible_hybrid": 0.030,
                "strong_hybrid": 0.040,
                "return_min_band": "strong_match",
                "min_overlap_terms": self.retrieval_quality_min_critical_overlap_terms,
                "min_overlap_ratio": self.retrieval_quality_min_critical_overlap_ratio,
                "require_overlap": True,
            },
            "overdue_corrective_actions": {
                "weak_faiss": 0.60,
                "possible_faiss": 0.68,
                "strong_faiss": 0.76,
                "weak_bm25": 15.0,
                "possible_bm25": 45.0,
                "strong_bm25": 80.0,
                "weak_hybrid": 0.018,
                "possible_hybrid": 0.030,
                "strong_hybrid": 0.040,
                "return_min_band": "strong_match",
                "min_overlap_terms": self.retrieval_quality_min_critical_overlap_terms,
                "min_overlap_ratio": self.retrieval_quality_min_critical_overlap_ratio,
                "require_overlap": True,
            },
        }
        leading_defaults = {
            "weak_faiss": 0.52,
            "possible_faiss": 0.60,
            "strong_faiss": 0.68,
            "weak_bm25": 8.0,
            "possible_bm25": 22.0,
            "strong_bm25": 42.0,
            "weak_hybrid": 0.014,
            "possible_hybrid": 0.026,
            "strong_hybrid": 0.036,
            "return_min_band": "strong_match",
            "min_overlap_terms": self.retrieval_quality_min_overlap_terms,
            "min_overlap_ratio": self.retrieval_quality_min_overlap_ratio,
            "require_overlap": True,
        }
        if route in {
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
        }:
            base.update(leading_defaults)
        if route in route_defaults:
            base.update(route_defaults[route])
        if self.route_quality_overrides_json.strip():
            try:
                overrides = json.loads(self.route_quality_overrides_json)
                if isinstance(overrides, dict):
                    common = overrides.get("default") if isinstance(overrides.get("default"), dict) else None
                    route_override = overrides.get(route) if isinstance(overrides.get(route), dict) else None
                    if common:
                        base.update(common)
                    if route_override:
                        base.update(route_override)
            except Exception:
                # Keep config robust; invalid override JSON should not prevent the
                # agent from running. The bad value is still visible in config.
                pass
        return base

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

    def tests_dir(self) -> Path:
        return self.test_output_dir

    def logs_dir(self) -> Path:
        return self.output_dir / "logs"


    def azureml_conda_env_path(self) -> Path:
        return self.project_root / "environments" / "safety_retrieval_agent_cpu.yml"

    def azureml_requirements_submit_path(self) -> Path:
        return self.project_root / "requirements_azureml_submit.txt"

    def prepared_base_dir(self) -> Path:
        """Input folder that contains previously prepared data, if mounted.

        In Azure ML jobs this is typically ${{inputs.prepared_output_dir}} and
        contains data/safety_knowledge_base.pkl from the interactive/local prep
        step. Outside Azure ML this falls back to output_dir.
        """
        return self.prepared_output_dir or self.output_dir

    def prepared_knowledge_base_path(self) -> Path:
        return self.prepared_base_dir() / "data" / "safety_knowledge_base.pkl"

    def prepared_embedding_scope_path(self) -> Path:
        return self.prepared_base_dir() / "data" / "safety_embedding_scope.pkl"

    def embedding_chunks_dir(self) -> Path:
        return self.embeddings_dir() / "chunks"

    def embedding_chunks_manifest_path(self) -> Path:
        return self.embeddings_dir() / "embedding_chunks_manifest.csv"

    def embedding_chunk_run_summary_path(self) -> Path:
        return self.embeddings_dir() / "embedding_chunk_run_summary.json"


def get_settings() -> Settings:
    return Settings()
