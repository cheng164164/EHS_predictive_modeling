#!/usr/bin/env python3
"""Cluster each source family separately and assign events to source-specific themes.

Supports HDBSCAN for discovery and MiniBatchKMeans for full assignment/fallback.
Outputs are saved under outputs/audits_tasks_modeling/04_theme_clusters.
"""
from __future__ import annotations

try:
    import config as cfg
    from theme_utils import ProgressLogger, cosine_similarity_matrix, ensure_dir, read_csv, save_json, write_csv
except ImportError:  # pragma: no cover
    from . import config as cfg
    from .theme_utils import ProgressLogger, cosine_similarity_matrix, ensure_dir, read_csv, save_json, write_csv

import traceback
from pathlib import Path

import numpy as np
import pandas as pd


def _normalize(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def _family_prefix(family: str) -> str:
    return {
        cfg.FAMILY_INCIDENT_HAZARD: "IH",
        getattr(cfg, "FAMILY_AUDIT_RISK", "audit_risk"): "AR",
        getattr(cfg, "FAMILY_AUDIT_POSITIVE", "audit_positive"): "AP",
        cfg.FAMILY_TASK_ACTION: "TA",
    }.get(family, family[:2].upper())


def _cluster_label_to_theme_id(family: str, label: int) -> str:
    prefix = _family_prefix(family)
    if int(label) < 0:
        return f"{prefix}_UNASSIGNED"
    return f"{prefix}_{int(label):04d}"


def _reduce_embeddings(emb: np.ndarray, family: str, log: ProgressLogger) -> tuple[np.ndarray, dict]:
    if not bool(getattr(cfg, "USE_DIMENSION_REDUCTION", True)):
        return emb.astype(np.float32), {"reduction_method": "none", "shape": list(emb.shape)}
    method = str(getattr(cfg, "REDUCTION_METHOD", "umap")).lower()
    if method == "umap":
        try:
            import umap
            reducer = umap.UMAP(
                n_neighbors=int(cfg.UMAP_N_NEIGHBORS),
                n_components=int(cfg.UMAP_N_COMPONENTS),
                min_dist=float(cfg.UMAP_MIN_DIST),
                metric=str(cfg.UMAP_METRIC),
                random_state=int(cfg.RANDOM_STATE),
                low_memory=True,
            )
            log.log(f"UMAP reducing {family}: {emb.shape} -> n_components={cfg.UMAP_N_COMPONENTS}")
            reduced = reducer.fit_transform(emb).astype(np.float32)
            return reduced, {"reduction_method": "umap", "shape": list(reduced.shape)}
        except Exception as exc:
            log.log(f"UMAP failed for {family}; falling back to PCA. reason={exc!r}")
    from sklearn.decomposition import PCA
    n_components = min(int(cfg.PCA_N_COMPONENTS), emb.shape[1], max(2, emb.shape[0] - 1))
    reducer = PCA(n_components=n_components, random_state=int(cfg.RANDOM_STATE))
    log.log(f"PCA reducing {family}: {emb.shape} -> n_components={n_components}")
    reduced = reducer.fit_transform(emb).astype(np.float32)
    return reduced, {"reduction_method": "pca", "shape": list(reduced.shape), "n_components": n_components}


def _choose_fit_indices(meta: pd.DataFrame, family: str, log: ProgressLogger) -> np.ndarray:
    n = len(meta)
    max_fit = int(cfg.CLUSTER_FIT_MAX_RECORDS_BY_FAMILY.get(family, 0) or 0)
    if max_fit <= 0 or n <= max_fit:
        return np.arange(n)
    rng = np.random.default_rng(int(cfg.RANDOM_STATE))
    high = meta.index[meta.get("review_priority", pd.Series(0, index=meta.index)).fillna(0).astype(float) >= 3.0].to_numpy()
    remaining_n = max(max_fit - len(high), 0)
    rest = np.setdiff1d(np.arange(n), high, assume_unique=False)
    if remaining_n > 0 and len(rest) > 0:
        sample_rest = rng.choice(rest, size=min(remaining_n, len(rest)), replace=False)
        idx = np.unique(np.concatenate([high, sample_rest]))
    else:
        idx = high[:max_fit]
    if len(idx) > max_fit:
        idx = rng.choice(idx, size=max_fit, replace=False)
    log.log(f"using fit sample for {family}: {len(idx):,}/{n:,} records")
    return np.sort(idx)


def _run_hdbscan(reduced_fit: np.ndarray, family: str, log: ProgressLogger) -> tuple[np.ndarray, np.ndarray | None, dict]:
    try:
        import hdbscan
    except Exception as exc:
        raise ImportError("hdbscan is not installed") from exc

    min_cluster_size = int(cfg.HDBSCAN_MIN_CLUSTER_SIZE_BY_FAMILY.get(family, 50))
    min_samples = int(cfg.HDBSCAN_MIN_SAMPLES_BY_FAMILY.get(family, 10))
    log.log(f"HDBSCAN clustering {family}; min_cluster_size={min_cluster_size}; min_samples={min_samples}")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method=str(cfg.HDBSCAN_CLUSTER_SELECTION_METHOD),
        prediction_data=False,
    )
    labels = clusterer.fit_predict(reduced_fit)
    probs = getattr(clusterer, "probabilities_", None)
    info = {
        "cluster_method_used": "hdbscan",
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples,
        "n_fit_records": int(len(reduced_fit)),
        "n_clusters_fit": int(len(set(labels)) - (1 if -1 in labels else 0)),
        "noise_records_fit": int((labels == -1).sum()),
    }
    return labels.astype(int), probs, info


def _run_kmeans(emb: np.ndarray, family: str, log: ProgressLogger) -> tuple[np.ndarray, np.ndarray, dict]:
    from sklearn.cluster import MiniBatchKMeans

    n_clusters = int(cfg.KMEANS_N_CLUSTERS_BY_FAMILY.get(family, 80))
    n_clusters = max(2, min(n_clusters, len(emb)))
    log.log(f"MiniBatchKMeans clustering {family}; n_clusters={n_clusters}")
    km = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=int(cfg.RANDOM_STATE),
        batch_size=int(cfg.KMEANS_BATCH_SIZE),
        max_iter=int(cfg.KMEANS_MAX_ITER),
        n_init="auto",
    )
    labels = km.fit_predict(emb)
    centroids = _normalize(km.cluster_centers_.astype(np.float32))
    sims = cosine_similarity_matrix(_normalize(emb), centroids)
    conf = sims[np.arange(len(emb)), labels]
    info = {
        "cluster_method_used": "minibatch_kmeans",
        "n_clusters": n_clusters,
        "n_fit_records": int(len(emb)),
        "noise_records_fit": 0,
    }
    return labels.astype(int), conf.astype(float), info


def _compute_centroids(emb: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, list[int]]:
    valid_labels = sorted([int(x) for x in set(labels.tolist()) if int(x) >= 0])
    centroids = []
    for lab in valid_labels:
        idx = np.where(labels == lab)[0]
        c = emb[idx].mean(axis=0)
        centroids.append(c)
    if len(centroids) == 0:
        return np.empty((0, emb.shape[1]), dtype=np.float32), []
    return _normalize(np.vstack(centroids).astype(np.float32)), valid_labels


def _assign_nearest(emb: np.ndarray, centroids: np.ndarray, centroid_labels: list[int]) -> tuple[np.ndarray, np.ndarray]:
    if len(centroids) == 0:
        return np.full(len(emb), -1, dtype=int), np.zeros(len(emb), dtype=float)
    sims = cosine_similarity_matrix(_normalize(emb), _normalize(centroids))
    best_pos = sims.argmax(axis=1)
    best_sim = sims[np.arange(len(emb)), best_pos]
    best_label = np.array([centroid_labels[p] for p in best_pos], dtype=int)
    return best_label, best_sim.astype(float)


def cluster_family(family: str, log: ProgressLogger) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, dict]:
    meta_file = cfg.EMBEDDING_META_FILE_BY_FAMILY[family]
    emb_file = cfg.EMBEDDING_FILE_BY_FAMILY[family]
    if not meta_file.exists() or not emb_file.exists():
        raise FileNotFoundError(f"Missing embeddings for {family}. Run 02_generate_theme_embeddings.py first.")
    meta = read_csv(meta_file)
    emb = np.load(emb_file)
    emb = _normalize(emb)
    if len(meta) != len(emb):
        raise ValueError(f"Metadata/embedding row mismatch for {family}: meta={len(meta)}, emb={len(emb)}")
    if len(meta) == 0:
        return meta, np.empty((0, 0), dtype=np.float32), pd.DataFrame(), {"family": family, "row_count": 0}

    method = str(cfg.CLUSTER_METHOD_BY_FAMILY.get(family, "hdbscan")).lower()
    family_info = {"family": family, "row_count": int(len(meta)), "configured_method": method}

    if method == "minibatch_kmeans":
        labels, confidence, method_info = _run_kmeans(emb, family, log)
        assignment_type = np.array(["forced_kmeans_cluster"] * len(labels), dtype=object)
        centroids, centroid_labels = _compute_centroids(emb, labels)
    else:
        fit_idx = _choose_fit_indices(meta, family, log)
        reduced, reduction_info = _reduce_embeddings(emb[fit_idx], family, log)
        try:
            fit_labels, fit_probs, method_info = _run_hdbscan(reduced, family, log)
        except Exception as exc:
            log.log(f"HDBSCAN failed for {family}; falling back to MiniBatchKMeans. reason={exc!r}")
            err = cfg.THEME_CLUSTER_DIR / f"hdbscan_error_{family}.txt"
            err.write_text(traceback.format_exc(), encoding="utf-8")
            labels, confidence, method_info = _run_kmeans(emb, family, log)
            method_info["fallback_reason"] = repr(exc)
            method_info["error_traceback_path"] = str(err)
            assignment_type = np.array(["forced_kmeans_cluster"] * len(labels), dtype=object)
            centroids, centroid_labels = _compute_centroids(emb, labels)
            family_info.update({"reduction": reduction_info})
        else:
            # Build centroids from high-confidence fit clusters in original embedding space.
            fit_full_labels = np.full(len(meta), -1, dtype=int)
            fit_full_labels[fit_idx] = fit_labels
            centroids, centroid_labels = _compute_centroids(emb[fit_idx], fit_labels)
            labels = np.full(len(meta), -1, dtype=int)
            confidence = np.zeros(len(meta), dtype=float)
            assignment_type = np.array(["unassigned_noise"] * len(meta), dtype=object)

            # Strong assignments only for HDBSCAN fit points with non-noise labels.
            labels[fit_idx] = fit_labels
            if fit_probs is not None:
                confidence[fit_idx] = fit_probs
            else:
                confidence[fit_idx] = np.where(fit_labels >= 0, 1.0, 0.0)
            assignment_type[(labels >= 0)] = "strong_cluster"

            # Assign all non-strong records to nearest centroid if similar enough.
            if bool(cfg.ASSIGN_HDBSCAN_NOISE_TO_NEAREST_THEME) and len(centroids) > 0:
                nearest_labels, nearest_sims = _assign_nearest(emb, centroids, centroid_labels)
                weak_mask = labels < 0
                ok = weak_mask & (nearest_sims >= float(cfg.NEAREST_THEME_MIN_COSINE_SIMILARITY))
                labels[ok] = nearest_labels[ok]
                confidence[ok] = nearest_sims[ok]
                assignment_type[ok] = "weak_nearest_theme"
                confidence[weak_mask & ~ok] = nearest_sims[weak_mask & ~ok]

            family_info.update({"reduction": reduction_info})

    theme_ids = [_cluster_label_to_theme_id(family, int(x)) for x in labels]
    out = meta.copy()
    out["cluster_label"] = labels.astype(int)
    out["theme_id"] = theme_ids
    out["assignment_type"] = assignment_type
    out["theme_confidence"] = confidence.astype(float)
    out["source_family"] = family

    centroid_rows = []
    for pos, lab in enumerate(centroid_labels):
        theme_id = _cluster_label_to_theme_id(family, lab)
        centroid_rows.append({
            "source_family": family,
            "cluster_label": int(lab),
            "theme_id": theme_id,
            "centroid_row": int(pos),
        })
    centroid_meta = pd.DataFrame(centroid_rows)
    family_info.update(method_info)
    family_info.update({
        "n_clusters_final": int(out.loc[out["cluster_label"] >= 0, "cluster_label"].nunique()),
        "assigned_records": int((out["cluster_label"] >= 0).sum()),
        "unassigned_records": int((out["cluster_label"] < 0).sum()),
        "assignment_type_counts": out["assignment_type"].value_counts(dropna=False).to_dict(),
    })
    return out, centroids, centroid_meta, family_info


def main() -> None:
    log = ProgressLogger("03_cluster_by_family")
    ensure_dir(cfg.THEME_CLUSTER_DIR)

    all_assignments = []
    all_centroids = []
    all_centroid_meta = []
    summary = []
    same_dim = True
    dim = None

    for family in cfg.SOURCE_FAMILIES:
        log.log(f"clustering family={family}")
        assignments, centroids, centroid_meta, info = cluster_family(family, log)
        write_csv(assignments, cfg.ASSIGNMENT_FILE_BY_FAMILY[family])
        family_centroid_file = cfg.THEME_CLUSTER_DIR / f"theme_centroids_{family}.npy"
        np.save(family_centroid_file, centroids)
        if len(centroid_meta) > 0:
            centroid_meta["centroid_file"] = str(family_centroid_file)
        all_assignments.append(assignments)
        if len(centroids) > 0:
            if dim is None:
                dim = centroids.shape[1]
            elif dim != centroids.shape[1]:
                same_dim = False
            all_centroids.append(centroids)
            all_centroid_meta.append(centroid_meta)
        summary.append(info)

    combined = pd.concat(all_assignments, ignore_index=True) if all_assignments else pd.DataFrame()
    write_csv(combined, cfg.THEME_ASSIGNMENTS_FILE)

    if all_centroid_meta:
        centroid_meta_all = pd.concat(all_centroid_meta, ignore_index=True)
        write_csv(centroid_meta_all, cfg.THEME_CENTROID_META_FILE)
        if same_dim and all_centroids:
            np.save(cfg.THEME_CENTROID_FILE, np.vstack(all_centroids).astype(np.float32))
        else:
            log.log("family centroid dimensions differ; combined centroid npy not written")

    save_json({"families": summary}, cfg.CLUSTER_SUMMARY_FILE)
    log.done("clustering complete")


if __name__ == "__main__":
    main()
