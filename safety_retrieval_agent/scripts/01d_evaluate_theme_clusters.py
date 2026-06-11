#!/usr/bin/env python
"""Evaluate MiniBatchKMeans theme-cluster counts and generate K=80 theme descriptions.

This script is intentionally standalone and does not require changes to artifact_io.py.
It reads embeddings from the Azure ML datastore even when the local runtime cache is used
for FAISS/BM25 indexes.

Outputs:
  outputs/safety retrieval agent/theme_evaluation/
    cluster_metrics.csv
    cluster_evaluation_summary.json
    elbow_curve.png
    silhouette_curve.png
    largest_cluster_share_by_k.png
    cluster_size_distribution_k*.csv/png
    pca_cluster_metrics_k80.csv
    pca_explained_variance.csv
    pca_silhouette_by_components.png
    pca_variance_explained_curve.png
    theme_descriptions_k80_pca100.csv/json
    theme_tfidf_phrases_k80_pca100.csv
    theme_description_examples_k80_pca100.csv
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Project bootstrap
# ---------------------------------------------------------------------
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from safety_retrieval_agent.config import get_settings  # noqa: E402


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _is_uri(path: Any) -> bool:
    text = str(path)
    return "://" in text


def _uri_join(base: str, *parts: object) -> str:
    text = str(base).rstrip("/")
    clean = [str(p).strip("/") for p in parts if str(p).strip("/")]
    return "/".join([text, *clean]) if clean else text


def _exists_any(path: str | Path) -> bool:
    """Path existence check for local paths and Azure ML/fsspec URIs."""
    if not _is_uri(path):
        return Path(path).exists()
    try:
        import fsspec
        fs, fs_path = fsspec.core.url_to_fs(str(path))
        return fs.exists(fs_path)
    except Exception:
        return False


def _load_numpy_any(path: str | Path) -> np.ndarray:
    """Load .npy from local path or Azure ML datastore URI."""
    if not _is_uri(path):
        return np.load(Path(path), allow_pickle=False)
    try:
        import fsspec
        with fsspec.open(str(path), "rb") as f:
            return np.load(f, allow_pickle=False)
    except Exception as exc:
        raise RuntimeError(
            "Failed to read numpy array from Azure ML datastore URI. "
            "Install azureml-fsspec and authenticate to the workspace if needed. "
            f"Path: {path}\nError: {exc}"
        ) from exc


def _read_pickle_any(path: str | Path) -> Any:
    if not _is_uri(path):
        return pd.read_pickle(Path(path))
    try:
        import fsspec
        with fsspec.open(str(path), "rb") as f:
            return pd.read_pickle(f)
    except Exception as exc:
        raise RuntimeError(f"Failed to read pickle from {path}: {exc}") from exc


def _read_json_any(path: str | Path) -> dict:
    if not _is_uri(path):
        with open(Path(path), "r", encoding="utf-8") as f:
            return json.load(f)
    import fsspec
    with fsspec.open(str(path), "rt", encoding="utf-8") as f:
        return json.load(f)


def _candidate_artifact_roots(settings) -> list[str | Path]:
    """Return roots to search, forcing Azure ML root before local cache.

    In artifact_read_mode='auto', settings.artifact_root() may point to the local
    runtime cache when indexes/models have been downloaded. That local cache usually
    does not include event_embeddings.npy. Therefore this script explicitly searches
    the datastore URI first.
    """
    roots: list[str | Path] = []

    # Fully qualified datastore URI used by runtime scripts.
    uri = getattr(settings, "artifact_azureml_uri", None)
    if uri:
        roots.append(str(uri).rstrip("/"))

    # Short Azure ML datastore URI used by batch jobs.
    uri = getattr(settings, "aml_artifact_output_uri", None)
    if uri:
        roots.append(str(uri).rstrip("/"))

    # Current artifact root as fallback.
    if hasattr(settings, "artifact_root"):
        try:
            roots.append(settings.artifact_root())
        except Exception:
            pass

    # Traditional local output folder as final fallback.
    roots.append(settings.output_dir)

    # Deduplicate preserving order.
    out: list[str | Path] = []
    seen = set()
    for r in roots:
        key = str(r).rstrip("/")
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def _embedding_matrix_candidates(settings) -> list[str | Path]:
    candidates: list[str | Path] = []
    for root in _candidate_artifact_roots(settings):
        if _is_uri(root):
            candidates.append(_uri_join(str(root), "embeddings", "event_embeddings.npy"))
            # Defensive fallback for the typo some notes used earlier.
            candidates.append(_uri_join(str(root), "emebddings", "event_embeddings.npy"))
        else:
            candidates.append(Path(root) / "embeddings" / "event_embeddings.npy")
            candidates.append(Path(root) / "emebddings" / "event_embeddings.npy")
    return candidates


def _load_embeddings(settings) -> tuple[np.ndarray, dict]:
    tried = []
    for path in _embedding_matrix_candidates(settings):
        tried.append(str(path))
        if _exists_any(path):
            print(f"[01d] Loading embeddings from: {path}", flush=True)
            arr = _load_numpy_any(path)
            return arr.astype("float32", copy=False), {
                "source": "event_embeddings.npy",
                "path": str(path),
                "shape": list(arr.shape),
            }
    msg = "\n  - ".join(tried)
    raise FileNotFoundError(
        "Could not find embeddings/event_embeddings.npy. The script searched:\n"
        f"  - {msg}\n\n"
        "Set SAFETY_RETRIEVAL_ARTIFACT_AZUREML_URI to the folder containing "
        "managed-batch-artifacts, or confirm that the datastore has "
        "safety-retrieval-agent/managed-batch-artifacts/embeddings/event_embeddings.npy."
    )


def _knowledge_base_candidates(settings) -> list[str | Path]:
    names = [
        ("data", "safety_embedding_scope.pkl"),
        ("data", "safety_knowledge_base_with_themes.pkl"),
        ("data", "safety_knowledge_base.pkl"),
    ]
    candidates: list[str | Path] = []
    for root in _candidate_artifact_roots(settings):
        for parts in names:
            if _is_uri(root):
                candidates.append(_uri_join(str(root), *parts))
            else:
                p = Path(root)
                for part in parts:
                    p = p / part
                candidates.append(p)
    return candidates


def _load_text_metadata(settings, n_rows: int) -> tuple[pd.DataFrame | None, dict]:
    for path in _knowledge_base_candidates(settings):
        if _exists_any(path):
            print(f"[01d] Loading text metadata from: {path}", flush=True)
            df = _read_pickle_any(path).reset_index(drop=True)
            if len(df) != n_rows:
                print(
                    f"[01d] Warning: metadata rows={len(df):,} != embedding rows={n_rows:,}. "
                    "Theme descriptions will use the first aligned rows only.",
                    flush=True,
                )
            return df, {"source": str(path), "rows": int(len(df))}
    print("[01d] No metadata table found. Cluster metrics will run, but theme descriptions will be skipped.", flush=True)
    return None, {"source": None, "rows": 0}


def _sample_embeddings(embeddings: np.ndarray, sample_size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    n = embeddings.shape[0]
    sample_size = max(2, min(int(sample_size), n))
    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(n, size=sample_size, replace=False) if sample_size < n else np.arange(n)
    sample = embeddings[sample_idx].astype("float32", copy=False)
    return sample, sample_idx.astype("int64")


def _parse_k_values() -> list[int]:
    text = os.getenv("SAFETY_RETRIEVAL_CLUSTER_EVAL_K_VALUES", "50,60,70,80,100,120,150")
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _cluster_eval_sample_size() -> int:
    return int(os.getenv("SAFETY_RETRIEVAL_CLUSTER_EVAL_SAMPLE_SIZE", "20000"))


def _theme_description_k(settings) -> int:
    return int(os.getenv("SAFETY_RETRIEVAL_THEME_DESCRIPTION_K", str(getattr(settings, "n_themes", 80))))


def _theme_description_pca_components() -> int | None:
    value = os.getenv("SAFETY_RETRIEVAL_THEME_DESCRIPTION_PCA_COMPONENTS", "100").strip().lower()
    if value in {"", "none", "null", "0"}:
        return None
    return int(value)


def _parse_pca_components() -> list[int | None]:
    """PCA component grid for the K=80 PCA study.

    Use "none" or "0" to include the no-PCA baseline. Default includes the
    baseline plus common reduced dimensions. Whitening is intentionally not used.
    """
    text = os.getenv("SAFETY_RETRIEVAL_CLUSTER_EVAL_PCA_COMPONENTS", "none,50,100,150,200")
    out: list[int | None] = []
    for part in text.split(","):
        val = part.strip().lower()
        if not val:
            continue
        if val in {"none", "null", "0"}:
            out.append(None)
        else:
            out.append(int(val))
    if not out:
        out = [None, 100]
    # Deduplicate preserving order.
    seen = set()
    unique: list[int | None] = []
    for item in out:
        key = "none" if item is None else str(item)
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def _pca_label(n_components: int | None) -> str:
    return "none" if n_components is None else f"pca{int(n_components)}"


def _reduce_with_pca(sample: np.ndarray, n_components: int | None, seed: int):
    """Return transformed sample, fitted PCA model, and explained variance info."""
    if n_components is None:
        return sample.astype("float32", copy=False), None, {
            "pca_components": None,
            "explained_variance_ratio_sum": None,
            "explained_variance_ratio_cumulative": None,
        }
    from sklearn.decomposition import PCA

    n_components = int(min(n_components, sample.shape[0] - 1, sample.shape[1]))
    if n_components <= 0:
        raise ValueError("PCA n_components must be positive after clipping.")
    print(f"[01d] Fitting PCA n_components={n_components}", flush=True)
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=seed, whiten=False)
    reduced = pca.fit_transform(sample).astype("float32", copy=False)
    # Normalize reduced vectors before cosine silhouette / KMeans so scale is stable.
    reduced = _normalise_vectors(reduced)
    info = {
        "pca_components": int(n_components),
        "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
        "explained_variance_ratio_cumulative": [float(x) for x in np.cumsum(pca.explained_variance_ratio_)],
    }
    return reduced, pca, info


def _evaluate_pca_at_fixed_k(sample: np.ndarray, k: int, pca_components: list[int | None], seed: int, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[int | None, Any]]:
    """Evaluate PCA dimensions at a fixed number of themes.

    Saves explained variance separately so you can review how much of the original
    embedding variance is retained by each PCA setting.
    """
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import silhouette_score

    rows = []
    var_rows = []
    pca_models: dict[int | None, Any] = {}
    for n_comp in pca_components:
        label = _pca_label(n_comp)
        reduced, pca, pca_info = _reduce_with_pca(sample, n_comp, seed)
        pca_models[n_comp] = pca
        print(f"[01d] PCA study: fitting K={k} on {label} vectors shape={reduced.shape}", flush=True)
        model = MiniBatchKMeans(
            n_clusters=int(k),
            random_state=seed,
            batch_size=int(os.getenv("SAFETY_RETRIEVAL_CLUSTER_EVAL_BATCH_SIZE", "4096")),
            n_init=int(os.getenv("SAFETY_RETRIEVAL_CLUSTER_EVAL_N_INIT", "3")),
            max_iter=int(os.getenv("SAFETY_RETRIEVAL_CLUSTER_EVAL_MAX_ITER", "100")),
            reassignment_ratio=0.01,
        )
        labels = model.fit_predict(reduced)
        counts = pd.Series(labels).value_counts().sort_index()
        try:
            sil = float(silhouette_score(reduced, labels, metric="cosine"))
        except Exception as exc:
            print(f"[01d] PCA silhouette failed for {label}, K={k}: {exc}", flush=True)
            sil = float("nan")
        rows.append({
            "k": int(k),
            "pca_components": "none" if n_comp is None else int(n_comp),
            "pca_label": label,
            "embedding_dim_after_pca": int(reduced.shape[1]),
            "inertia": float(model.inertia_),
            "silhouette_cosine": sil,
            "explained_variance_ratio_sum": pca_info["explained_variance_ratio_sum"],
            "largest_cluster_size": int(counts.max()),
            "largest_cluster_share": float(counts.max() / len(labels)),
            "smallest_cluster_size": int(counts.min()),
            "median_cluster_size": float(counts.median()),
            "mean_cluster_size": float(counts.mean()),
            "tiny_clusters_lt_20": int((counts < 20).sum()),
            "tiny_clusters_lt_50": int((counts < 50).sum()),
            "sample_size": int(len(sample)),
        })
        if pca is not None:
            cumulative = np.cumsum(pca.explained_variance_ratio_)
            for comp_idx, (ratio, cum) in enumerate(zip(pca.explained_variance_ratio_, cumulative), start=1):
                var_rows.append({
                    "pca_components_setting": int(n_comp),
                    "component_index": int(comp_idx),
                    "explained_variance_ratio": float(ratio),
                    "cumulative_explained_variance_ratio": float(cum),
                })
    metrics = pd.DataFrame(rows)
    var_df = pd.DataFrame(var_rows)
    metrics.to_csv(out_dir / f"pca_cluster_metrics_k{k}.csv", index=False)
    var_df.to_csv(out_dir / "pca_explained_variance.csv", index=False)
    return metrics, var_df, pca_models


def _plot_pca_metrics(pca_metrics: pd.DataFrame, pca_var: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    if pca_metrics is not None and not pca_metrics.empty:
        x = [str(v) for v in pca_metrics["pca_components"]]
        plt.figure(figsize=(10, 6))
        plt.plot(x, pca_metrics["silhouette_cosine"], marker="o")
        plt.title("PCA Study at Fixed K: Silhouette by PCA Components")
        plt.xlabel("PCA components")
        plt.ylabel("Cosine silhouette score")
        plt.grid(True, alpha=0.35)
        plt.tight_layout()
        plt.savefig(out_dir / "pca_silhouette_by_components.png", dpi=160)
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(x, pca_metrics["explained_variance_ratio_sum"], marker="o")
        plt.title("PCA Study: Total Explained Variance by Components")
        plt.xlabel("PCA components")
        plt.ylabel("Total explained variance ratio")
        plt.grid(True, alpha=0.35)
        plt.tight_layout()
        plt.savefig(out_dir / "pca_total_explained_variance_by_components.png", dpi=160)
        plt.close()

    if pca_var is not None and not pca_var.empty:
        plt.figure(figsize=(10, 6))
        for setting, grp in pca_var.groupby("pca_components_setting"):
            plt.plot(grp["component_index"], grp["cumulative_explained_variance_ratio"], label=f"PCA {setting}")
        plt.title("PCA Cumulative Explained Variance")
        plt.xlabel("Component index")
        plt.ylabel("Cumulative explained variance ratio")
        plt.grid(True, alpha=0.35)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "pca_variance_explained_curve.png", dpi=160)
        plt.close()


def _normalise_vectors(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype="float32")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def _evaluate_k_values(sample: np.ndarray, k_values: list[int], seed: int, out_dir: Path) -> pd.DataFrame:
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import silhouette_score

    rows = []
    for k in k_values:
        print(f"[01d] Fitting MiniBatchKMeans K={k}", flush=True)
        model = MiniBatchKMeans(
            n_clusters=int(k),
            random_state=seed,
            batch_size=int(os.getenv("SAFETY_RETRIEVAL_CLUSTER_EVAL_BATCH_SIZE", "4096")),
            n_init=int(os.getenv("SAFETY_RETRIEVAL_CLUSTER_EVAL_N_INIT", "3")),
            max_iter=int(os.getenv("SAFETY_RETRIEVAL_CLUSTER_EVAL_MAX_ITER", "100")),
            reassignment_ratio=0.01,
        )
        labels = model.fit_predict(sample)
        counts = pd.Series(labels).value_counts().sort_index()
        try:
            sil = float(silhouette_score(sample, labels, metric="cosine"))
        except Exception as exc:
            print(f"[01d] Silhouette failed for K={k}: {exc}", flush=True)
            sil = float("nan")
        dist_path = out_dir / f"cluster_size_distribution_k{k}.csv"
        counts.rename_axis("cluster_id").reset_index(name="cluster_size").to_csv(dist_path, index=False)
        rows.append({
            "k": int(k),
            "inertia": float(model.inertia_),
            "silhouette_cosine": sil,
            "largest_cluster_size": int(counts.max()),
            "largest_cluster_share": float(counts.max() / len(labels)),
            "smallest_cluster_size": int(counts.min()),
            "median_cluster_size": float(counts.median()),
            "mean_cluster_size": float(counts.mean()),
            "tiny_clusters_lt_20": int((counts < 20).sum()),
            "tiny_clusters_lt_50": int((counts < 50).sum()),
            "sample_size": int(len(sample)),
        })
    return pd.DataFrame(rows)


def _plot_metrics(metrics: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    plots = [
        ("elbow_curve.png", "inertia", "Elbow Curve: K vs. MiniBatchKMeans Inertia", "Inertia, lower is better"),
        ("silhouette_curve.png", "silhouette_cosine", "Silhouette Score by K", "Cosine silhouette score"),
        ("largest_cluster_share_by_k.png", "largest_cluster_share", "Largest Cluster Share by K", "Largest cluster share"),
    ]
    for filename, ycol, title, ylabel in plots:
        plt.figure(figsize=(10, 6))
        plt.plot(metrics["k"], metrics[ycol], marker="o")
        plt.title(title)
        plt.xlabel("Number of clusters (K)")
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.35)
        plt.tight_layout()
        plt.savefig(out_dir / filename, dpi=160)
        plt.close()


def _text_for_row(row: pd.Series) -> str:
    parts = []
    for col in ["title", "description", "clean_text", "retrieval_text", "summary"]:
        if col in row.index:
            val = str(row.get(col) or "").strip()
            if val and val.lower() != "nan":
                parts.append(val)
    text = " | ".join(parts)
    return text[:5000]


def _english_spanish_stopwords() -> set[str]:
    """Return English + Spanish stopwords for TF-IDF theme phrases.

    This list is used only for cluster/theme description extraction. It is not a
    hazard taxonomy and it does not affect FAISS/BM25 retrieval. The goal is to
    remove common function words so multi-word TF-IDF phrases are easier to
    review, especially for English/Spanish audit and incident text.
    """
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    # Compact Spanish stopword list adapted for unaccented matching because the
    # vectorizer uses strip_accents="unicode". Keep this local to avoid adding
    # nltk/spacy dependencies.
    spanish = {
        "a", "al", "algo", "algunas", "algunos", "ante", "antes", "aquel", "aquella", "aquellas",
        "aquello", "aquellos", "aqui", "arriba", "asi", "atras", "aun", "aunque", "bajo", "bastante",
        "bien", "cada", "casi", "como", "con", "contra", "cual", "cuales", "cualquier", "cuando",
        "cuanto", "cuantos", "de", "debe", "deben", "debido", "del", "desde", "donde", "dos", "durante",
        "el", "ella", "ellas", "ellos", "en", "entre", "era", "erais", "eramos", "eran", "eras", "eres",
        "es", "esa", "esas", "ese", "eso", "esos", "esta", "estaba", "estaban", "estado", "estais",
        "estamos", "estan", "estar", "estara", "estas", "este", "esto", "estos", "estoy", "fue", "fueron",
        "fui", "fuimos", "ha", "habia", "habian", "hace", "hacen", "hacer", "hacia", "han", "hasta",
        "hay", "la", "las", "le", "les", "lo", "los", "mas", "me", "mi", "mis", "mismo", "mucha",
        "muchas", "mucho", "muchos", "muy", "nada", "ni", "no", "nos", "nosotros", "nuestra",
        "nuestras", "nuestro", "nuestros", "o", "otra", "otras", "otro", "otros", "para", "pero", "poco",
        "por", "porque", "que", "quien", "quienes", "se", "sea", "sean", "segun", "ser", "si", "sido",
        "siempre", "sin", "sobre", "solo", "son", "su", "sus", "tambien", "tanto", "te", "tenia",
        "tienen", "todo", "todos", "tu", "tus", "un", "una", "unas", "uno", "unos", "usted", "ustedes",
        "va", "van", "varias", "varios", "y", "ya", "yo",
    }
    # Safety-record boilerplate words that are common in both languages and make
    # phrase labels less useful. These are generic corpus stopwords, not hazard
    # categories. Override with SAFETY_RETRIEVAL_THEME_EXTRA_STOPWORDS if needed.
    corpus_generic = {
        "incident", "hazard", "safety", "task", "audit", "observation", "observed", "reported",
        "employee", "employees", "worker", "workers", "area", "needs", "need", "required", "requires",
        "please", "found", "noted", "issue", "items", "item", "work", "working", "condition", "conditions",
        "action", "actions", "corrective", "inspection", "inspeccion", "seguridad", "trabajo", "trabajador",
        "trabajadores", "empleado", "empleados", "observacion", "accion", "acciones", "correctiva", "correctivo",
    }
    extra = os.getenv("SAFETY_RETRIEVAL_THEME_EXTRA_STOPWORDS", "")
    extra_words = {w.strip().lower() for w in extra.split(",") if w.strip()}
    return {str(w).lower() for w in ENGLISH_STOP_WORDS}.union(spanish).union(corpus_generic).union(extra_words)


def _phrase_vectorizer():
    from sklearn.feature_extraction.text import TfidfVectorizer

    # Multi-word phrases only. English and Spanish stopwords are removed before
    # n-gram creation so phrases like "de la", "in the", or "para el" do not
    # dominate the theme labels.
    return TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words=sorted(_english_spanish_stopwords()),
        ngram_range=(3, 5),
        min_df=int(os.getenv("SAFETY_RETRIEVAL_THEME_PHRASE_MIN_DF", "2")),
        max_df=float(os.getenv("SAFETY_RETRIEVAL_THEME_PHRASE_MAX_DF", "0.85")),
        max_features=int(os.getenv("SAFETY_RETRIEVAL_THEME_PHRASE_MAX_FEATURES", "80000")),
        token_pattern=r"(?u)\b\w[\w\-/]{1,}\b",
    )


def _phrase_is_useful(phrase: str) -> bool:
    """Reject phrases that are mostly stopwords/numbers or too short to interpret."""
    stopwords = _english_spanish_stopwords()
    tokens = [t for t in re.findall(r"(?u)\b\w[\w\-/]{1,}\b", str(phrase).lower())]
    if len(tokens) < 2:
        return False
    content = [t for t in tokens if t not in stopwords and not t.replace("/", "").replace("-", "").isdigit()]
    return len(content) >= 2


def _generate_theme_descriptions(
    sample: np.ndarray,
    sample_idx: np.ndarray,
    metadata: pd.DataFrame | None,
    k: int,
    seed: int,
    out_dir: Path,
    pca_components: int | None = 100,
) -> dict:
    from sklearn.cluster import MiniBatchKMeans

    if metadata is None or metadata.empty:
        return {"generated": False, "reason": "metadata_not_available"}

    pca_label = _pca_label(pca_components)
    print(f"[01d] Generating theme phrase/description files for K={k}, PCA={pca_label}", flush=True)
    clustering_sample, _pca, pca_info = _reduce_with_pca(sample, pca_components, seed)
    model = MiniBatchKMeans(
        n_clusters=int(k),
        random_state=seed,
        batch_size=int(os.getenv("SAFETY_RETRIEVAL_CLUSTER_EVAL_BATCH_SIZE", "4096")),
        n_init=int(os.getenv("SAFETY_RETRIEVAL_CLUSTER_EVAL_N_INIT", "3")),
        max_iter=int(os.getenv("SAFETY_RETRIEVAL_CLUSTER_EVAL_MAX_ITER", "100")),
        reassignment_ratio=0.01,
    )
    labels = model.fit_predict(clustering_sample)
    meta = metadata.iloc[np.minimum(sample_idx, len(metadata) - 1)].copy().reset_index(drop=True)
    meta["cluster_id"] = labels
    meta["embedding_row_id"] = sample_idx
    texts = meta.apply(_text_for_row, axis=1).fillna("").astype(str).tolist()

    vectorizer = _phrase_vectorizer()
    try:
        X = vectorizer.fit_transform(texts)
        feature_names = np.asarray(vectorizer.get_feature_names_out())
    except Exception as exc:
        return {"generated": False, "reason": f"tfidf_failed: {exc}"}

    phrase_rows = []
    desc_rows = []
    example_rows = []
    top_phrases = int(os.getenv("SAFETY_RETRIEVAL_THEME_TOP_PHRASES", "12"))
    top_examples = int(os.getenv("SAFETY_RETRIEVAL_THEME_DESCRIPTION_EXAMPLES", "8"))

    for cluster_id in sorted(np.unique(labels)):
        mask = labels == cluster_id
        row_positions = np.flatnonzero(mask)
        cluster_size = int(mask.sum())
        if cluster_size == 0:
            continue
        centroid = model.cluster_centers_[cluster_id].reshape(1, -1)
        vecs = clustering_sample[mask]
        # cosine similarity to centroid for examples
        v_norm = _normalise_vectors(vecs)
        c_norm = _normalise_vectors(centroid)
        sims = (v_norm @ c_norm.T).ravel()
        order = np.argsort(-sims)[:top_examples]

        cluster_tfidf = X[mask].mean(axis=0)
        arr = np.asarray(cluster_tfidf).ravel()
        top_idx = np.argsort(-arr)[: top_phrases * 3]
        phrases = []
        seen_lower = set()
        for idx in top_idx:
            score = float(arr[idx])
            if score <= 0:
                continue
            phrase = str(feature_names[idx]).strip()
            # Keep interpretable multi-word phrases only and avoid redundant duplicates.
            if not _phrase_is_useful(phrase):
                continue
            low = phrase.lower()
            if low in seen_lower:
                continue
            seen_lower.add(low)
            phrases.append((phrase, score))
            if len(phrases) >= top_phrases:
                break

        theme_id = f"RT{int(cluster_id) + 1:04d}"
        phrase_texts = [p for p, _ in phrases]
        short_label = " / ".join(phrase_texts[:3]) if phrase_texts else theme_id
        description = (
            f"Theme characterized by recurring phrases such as: {', '.join(phrase_texts[:6])}."
            if phrase_texts else
            "Theme description unavailable because no stable multi-word TF-IDF phrases were found."
        )
        desc_rows.append({
            "risk_theme_id": theme_id,
            "cluster_id": int(cluster_id),
            "cluster_size_in_sample": cluster_size,
            "pca_components": "none" if pca_components is None else int(pca_components),
            "pca_explained_variance_ratio_sum": pca_info.get("explained_variance_ratio_sum"),
            "theme_short_label_from_phrases": short_label,
            "theme_description_from_tfidf_phrases": description,
            "top_tfidf_phrases": " | ".join(phrase_texts),
        })
        for rank, (phrase, score) in enumerate(phrases, start=1):
            phrase_rows.append({
                "risk_theme_id": theme_id,
                "cluster_id": int(cluster_id),
                "phrase_rank": rank,
                "tfidf_phrase": phrase,
                "mean_tfidf_score": score,
                "cluster_size_in_sample": cluster_size,
            })
        for ex_rank, local_pos in enumerate(order, start=1):
            meta_row = meta.iloc[row_positions[local_pos]]
            example_rows.append({
                "risk_theme_id": theme_id,
                "cluster_id": int(cluster_id),
                "example_rank": ex_rank,
                "centroid_cosine_similarity": float(sims[local_pos]),
                "embedding_row_id": int(meta_row.get("embedding_row_id")),
                "event_id": meta_row.get("event_id"),
                "source_type": meta_row.get("source_type"),
                "source_role": meta_row.get("source_role"),
                "title": str(meta_row.get("title") or "")[:300],
                "description_preview": str(meta_row.get("description") or meta_row.get("retrieval_text") or "")[:500],
            })

    desc_df = pd.DataFrame(desc_rows)
    phrase_df = pd.DataFrame(phrase_rows)
    examples_df = pd.DataFrame(example_rows)
    suffix = f"k{k}_{pca_label}"
    desc_path = out_dir / f"theme_descriptions_{suffix}.csv"
    desc_json_path = out_dir / f"theme_descriptions_{suffix}.json"
    phrase_path = out_dir / f"theme_tfidf_phrases_{suffix}.csv"
    examples_path = out_dir / f"theme_description_examples_{suffix}.csv"
    desc_df.to_csv(desc_path, index=False)
    phrase_df.to_csv(phrase_path, index=False)
    examples_df.to_csv(examples_path, index=False)
    desc_json_path.write_text(json.dumps(desc_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "generated": True,
        "k": int(k),
        "pca_components": "none" if pca_components is None else int(pca_components),
        "pca_explained_variance_ratio_sum": pca_info.get("explained_variance_ratio_sum"),
        "theme_descriptions_csv": str(desc_path),
        "theme_descriptions_json": str(desc_json_path),
        "theme_tfidf_phrases_csv": str(phrase_path),
        "theme_examples_csv": str(examples_path),
        "n_themes": int(len(desc_df)),
    }


def main() -> None:
    settings = get_settings()
    out_dir = _ensure_dir(settings.output_dir / "theme_evaluation")
    seed = int(os.getenv("SAFETY_RETRIEVAL_CLUSTER_EVAL_RANDOM_SEED", str(getattr(settings, "random_seed", 42))))
    k_values = _parse_k_values()
    desc_k = _theme_description_k(settings)
    desc_pca = _theme_description_pca_components()
    pca_values = _parse_pca_components()
    if desc_pca not in pca_values:
        pca_values.append(desc_pca)
    if desc_k not in k_values:
        k_values.append(desc_k)
        k_values = sorted(set(k_values))

    embeddings, source_info = _load_embeddings(settings)
    print(f"[01d] Embedding matrix shape: {embeddings.shape}", flush=True)
    sample, sample_idx = _sample_embeddings(embeddings, _cluster_eval_sample_size(), seed)
    print(f"[01d] Cluster evaluation sample shape: {sample.shape}", flush=True)

    metrics = _evaluate_k_values(sample, k_values, seed, out_dir)
    metrics_path = out_dir / "cluster_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    _plot_metrics(metrics, out_dir)

    pca_metrics, pca_var, _pca_models = _evaluate_pca_at_fixed_k(sample, desc_k, pca_values, seed, out_dir)
    pca_metrics_path = out_dir / f"pca_cluster_metrics_k{desc_k}.csv"
    pca_var_path = out_dir / "pca_explained_variance.csv"
    _plot_pca_metrics(pca_metrics, pca_var, out_dir)

    metadata, metadata_info = _load_text_metadata(settings, n_rows=embeddings.shape[0])
    desc_info = _generate_theme_descriptions(sample, sample_idx, metadata, desc_k, seed, out_dir, pca_components=desc_pca)

    best_sil = None
    best_k = None
    if "silhouette_cosine" in metrics.columns and metrics["silhouette_cosine"].notna().any():
        best_row = metrics.loc[metrics["silhouette_cosine"].idxmax()]
        best_sil = float(best_row["silhouette_cosine"])
        best_k = int(best_row["k"])

    summary = {
        "embedding_source": source_info,
        "metadata_source": metadata_info,
        "output_dir": str(out_dir),
        "k_values": k_values,
        "pca_study_fixed_k": int(desc_k),
        "pca_study_components": ["none" if x is None else int(x) for x in pca_values],
        "theme_description_k": int(desc_k),
        "theme_description_pca_components": "none" if desc_pca is None else int(desc_pca),
        "sample_size": int(len(sample)),
        "embedding_rows": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "best_k_by_silhouette": best_k,
        "best_silhouette_cosine": best_sil,
        "theme_description_generation": desc_info,
        "outputs": {
            "cluster_metrics_csv": str(metrics_path),
            "pca_cluster_metrics_csv": str(pca_metrics_path),
            "pca_explained_variance_csv": str(pca_var_path),
            "pca_silhouette_by_components_png": str(out_dir / "pca_silhouette_by_components.png"),
            "pca_variance_explained_curve_png": str(out_dir / "pca_variance_explained_curve.png"),
            "elbow_curve_png": str(out_dir / "elbow_curve.png"),
            "silhouette_curve_png": str(out_dir / "silhouette_curve.png"),
            "largest_cluster_share_png": str(out_dir / "largest_cluster_share_by_k.png"),
        },
    }
    summary_path = out_dir / "cluster_evaluation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[01d] Saved metrics: {metrics_path}", flush=True)
    print(f"[01d] Saved summary: {summary_path}", flush=True)
    if desc_info.get("generated"):
        print(f"[01d] Saved theme descriptions: {desc_info.get('theme_descriptions_csv')}", flush=True)


if __name__ == "__main__":
    main()
