#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import config as cfg

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

from utils import ensure_dir, load_table, normalize_embeddings, save_csv, save_json


def _rng(seed):
    return np.random.default_rng(seed)


def make_theme_discovery_mask(events: pd.DataFrame, exclude_tasks: bool) -> np.ndarray:
    """Return candidate rows for theme discovery.

    Step 3 intentionally discovers themes on candidates only. Step 4 assigns every
    record, including tasks, to the discovered centroids.
    """
    text = events.get("clean_text", pd.Series([""] * len(events))).fillna("").astype(str)
    mask = text.str.len() > 0
    if exclude_tasks and "source_type" in events.columns:
        mask &= events["source_type"].ne("task")
    return mask.to_numpy()


def representative_sample_indices(
    events: pd.DataFrame,
    candidate_idx: np.ndarray,
    sample_size: int | None,
    strategy: str,
    random_state: int | None,
) -> np.ndarray:
    """Sample representative rows for UMAP/HDBSCAN.

    Default strategy is stratified to avoid a sample dominated by the largest source
    or category. Strata use available columns only, so the function is robust across
    slightly different exports.
    """
    if sample_size is None or sample_size <= 0 or len(candidate_idx) <= sample_size:
        return np.sort(candidate_idx)

    rng = _rng(random_state)
    strategy = (strategy or "stratified").lower()
    if strategy not in {"stratified", "random"}:
        strategy = "stratified"

    if strategy == "random":
        return np.sort(rng.choice(candidate_idx, size=sample_size, replace=False))

    candidates = events.iloc[candidate_idx].copy()
    strata_cols = [c for c in ["source_type", "incident_category_name", "consequence_potential"] if c in candidates.columns]
    if not strata_cols:
        return np.sort(rng.choice(candidate_idx, size=sample_size, replace=False))

    strata = candidates[strata_cols].fillna("unknown").astype(str).agg("|".join, axis=1)
    counts = strata.value_counts()
    # Allocate proportionally, with a small minimum so minority source/category groups remain visible.
    min_per_stratum = max(1, min(250, sample_size // max(len(counts), 1)))
    selected = []
    remaining_budget = sample_size

    # First pass: proportional quotas capped by stratum size.
    quotas = {}
    for key, count in counts.items():
        proportional = int(round(sample_size * count / len(candidates)))
        quotas[key] = min(int(count), max(min_per_stratum, proportional))

    # If quotas exceed sample size, scale down but keep at least one per stratum where possible.
    total_quota = sum(quotas.values())
    if total_quota > sample_size:
        scale = sample_size / total_quota
        quotas = {k: max(1, int(v * scale)) for k, v in quotas.items()}

    for key, quota in quotas.items():
        if remaining_budget <= 0:
            break
        group_positions = np.where(strata.to_numpy() == key)[0]
        take = min(len(group_positions), quota, remaining_budget)
        chosen_positions = rng.choice(group_positions, size=take, replace=False)
        selected.extend(candidate_idx[chosen_positions].tolist())
        remaining_budget -= take

    # Fill any remaining budget randomly from rows not already selected.
    if remaining_budget > 0:
        selected_set = set(selected)
        leftover = np.array([i for i in candidate_idx if i not in selected_set], dtype=int)
        if len(leftover) > 0:
            take = min(remaining_budget, len(leftover))
            selected.extend(rng.choice(leftover, size=take, replace=False).tolist())

    return np.sort(np.array(selected[:sample_size], dtype=int))


def fit_umap(x: np.ndarray, args):
    """Always reduce embeddings with UMAP before clustering."""
    try:
        import umap
    except Exception as exc:
        raise ImportError(
            "UMAP is required for Step 3 because clustering is configured to always run "
            "after UMAP dimensionality reduction. Install umap-learn or switch environments."
        ) from exc

    reducer = umap.UMAP(
        n_neighbors=args.umap_neighbors,
        n_components=args.umap_components,
        min_dist=args.umap_min_dist,
        metric="cosine",
        random_state=args.umap_random_state,
        n_jobs=args.umap_n_jobs,
    )
    x_low = reducer.fit_transform(x)
    return x_low, reducer


def cluster_hdbscan(x_low: np.ndarray, args, reducer):
    try:
        import hdbscan
    except Exception as exc:
        raise ImportError(
            "hdbscan is required when CLUSTER_ALGORITHM='hdbscan'. "
            "Install hdbscan or set CLUSTER_ALGORITHM='kmeans' in config.py."
        ) from exc

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        cluster_selection_epsilon=args.cluster_selection_epsilon,
        prediction_data=True,
    )
    labels = clusterer.fit_predict(x_low)
    strengths = getattr(clusterer, "probabilities_", np.ones(len(labels)))
    return labels, strengths, {
        "reducer": reducer,
        "clusterer": clusterer,
        "method": "sample_umap_hdbscan",
    }


def cluster_kmeans(x_low: np.ndarray, args, reducer):
    n_clusters = min(args.kmeans_clusters, max(2, int(len(x_low) / max(args.min_cluster_size, 1))))
    n_clusters = max(2, n_clusters)
    model = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=cfg.RANDOM_STATE,
        batch_size=4096,
        n_init="auto",
    )
    labels = model.fit_predict(x_low)
    distances = np.linalg.norm(x_low - model.cluster_centers_[labels], axis=1)
    strengths = 1.0 / (1.0 + distances)
    return labels, strengths, {
        "reducer": reducer,
        "clusterer": model,
        "method": "sample_umap_kmeans",
    }


def cluster_after_umap(x: np.ndarray, args):
    x_low, reducer = fit_umap(x, args)
    algorithm = (args.cluster_algorithm or "hdbscan").lower()
    if algorithm == "hdbscan":
        return cluster_hdbscan(x_low, args, reducer)
    if algorithm == "kmeans":
        return cluster_kmeans(x_low, args, reducer)
    raise ValueError(f"Unsupported cluster algorithm: {args.cluster_algorithm}. Use 'hdbscan' or 'kmeans'.")


def top_terms(docs, n=8):
    docs = [str(t) for t in docs if str(t).strip()]
    if len(docs) == 0:
        return []
    try:
        vec = TfidfVectorizer(max_features=3000, stop_words="english", ngram_range=(1, 2), min_df=2)
        mat = vec.fit_transform(docs)
        means = np.asarray(mat.mean(axis=0)).ravel()
        names = np.array(vec.get_feature_names_out())
        return names[np.argsort(-means)[:n]].tolist()
    except Exception:
        return []


def mode_pipe_values(series: pd.Series, n=3):
    c = Counter()
    for value in series.fillna("").astype(str):
        for token in value.replace(";", "|").split("|"):
            token = token.strip()
            if token and token.lower() not in {"nan", "none", "unknown"}:
                c[token] += 1
    return [k for k, _ in c.most_common(n)]


def readable_name(row):
    hazard = row.get("top_hazard_tags", "")
    control = row.get("top_control_failure_tags", "")
    terms = row.get("top_terms", "")
    first_hazard = hazard.split("|")[0] if hazard else ""
    first_control = control.split("|")[0] if control else ""
    if first_hazard and first_control:
        return f"{first_hazard.replace('_', ' ')} / {first_control.replace('_', ' ')}"
    if first_hazard:
        return first_hazard.replace("_", " ")
    if first_control:
        return first_control.replace("_", " ")
    if terms:
        return terms.split("|")[0].replace("_", " ")
    return f"Risk theme {row['risk_theme_id']}"


def main():
    parser = argparse.ArgumentParser(description="Discover readable risk themes from a representative sample of safety text embeddings.")
    parser.add_argument("--events", default=cfg.TAGGED_EVENTS_PATH)
    parser.add_argument("--embeddings", default=cfg.TEXT_EMBEDDINGS_PATH)
    parser.add_argument("--output-dir", default=cfg.STEP_03_DIR)
    parser.add_argument("--exclude-tasks", action="store_true", default=cfg.EXCLUDE_TASKS_FROM_THEME_DISCOVERY, help="Discover risk themes from non-task records, then assign tasks later in Step 4.")
    parser.add_argument("--include-tasks", dest="exclude_tasks", action="store_false")
    parser.add_argument("--sample-size", type=int, default=cfg.THEME_DISCOVERY_SAMPLE_SIZE, help="Representative sample size for clustering only. Step 4 still assigns all records.")
    parser.add_argument("--sample-strategy", default=cfg.THEME_DISCOVERY_SAMPLE_STRATEGY, choices=["stratified", "random"])
    parser.add_argument("--sample-random-state", type=int, default=cfg.THEME_DISCOVERY_RANDOM_STATE)
    parser.add_argument("--cluster-algorithm", default=cfg.CLUSTER_ALGORITHM, choices=["hdbscan", "kmeans"], help="Clustering algorithm to run after UMAP dimensionality reduction.")
    parser.add_argument("--min-cluster-size", type=int, default=cfg.MIN_CLUSTER_SIZE)
    parser.add_argument("--min-samples", type=int, default=cfg.MIN_SAMPLES)
    parser.add_argument("--cluster-selection-epsilon", type=float, default=cfg.CLUSTER_SELECTION_EPSILON, help="HDBSCAN cluster_selection_epsilon. Increase to merge nearby clusters; ignored for KMeans.")
    parser.add_argument("--umap-neighbors", type=int, default=cfg.UMAP_NEIGHBORS)
    parser.add_argument("--umap-components", type=int, default=cfg.UMAP_COMPONENTS)
    parser.add_argument("--umap-min-dist", type=float, default=cfg.UMAP_MIN_DIST)
    parser.add_argument("--umap-random-state", type=int, default=cfg.UMAP_RANDOM_STATE)
    parser.add_argument("--umap-n-jobs", type=int, default=cfg.UMAP_N_JOBS)
    parser.add_argument("--kmeans-clusters", type=int, default=cfg.KMEANS_CLUSTERS)
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    model_dir = ensure_dir(output_dir / "models")
    events = load_table(args.events)
    x = normalize_embeddings(np.load(args.embeddings).astype(np.float32))
    if len(events) != x.shape[0]:
        raise ValueError(f"Row mismatch: events={len(events)} embeddings={x.shape[0]}")

    candidate_mask = make_theme_discovery_mask(events, args.exclude_tasks)
    candidate_idx = np.where(candidate_mask)[0]
    sample_idx = representative_sample_indices(
        events=events,
        candidate_idx=candidate_idx,
        sample_size=args.sample_size,
        strategy=args.sample_strategy,
        random_state=args.sample_random_state,
    )
    x_cluster = x[sample_idx]
    clustered_events = events.iloc[sample_idx].reset_index(drop=True)

    print(
        f"Discovering risk themes on sample rows={len(sample_idx):,} "
        f"from candidate rows={len(candidate_idx):,}; total rows={len(events):,}. "
        "Step 4 will assign all rows to these themes."
    )

    labels, strengths, model_bundle = cluster_after_umap(x_cluster, args)
    method = model_bundle.get("method")

    membership = pd.DataFrame({
        "event_id": clustered_events["event_id"].values,
        "row_index": sample_idx,
        "cluster_label": labels,
        "membership_strength": strengths,
        "outlier_flag": labels == -1,
        "sampled_for_theme_discovery": True,
    })

    valid_clusters = sorted([int(c) for c in np.unique(labels) if int(c) != -1])
    theme_rows = []
    centroids = []
    for theme_no, lab in enumerate(valid_clusters, start=1):
        cluster_mask = labels == lab
        ev = clustered_events.loc[cluster_mask]
        emb = x_cluster[cluster_mask]
        centroid = normalize_embeddings(emb.mean(axis=0, keepdims=True))[0]
        centroids.append(centroid)
        terms = top_terms(ev["clean_text"].fillna("").astype(str).tolist(), n=8)
        hazard = mode_pipe_values(ev.get("hazard_tags", pd.Series(dtype=str)), n=5)
        control = mode_pipe_values(ev.get("control_failure_tags", pd.Series(dtype=str)), n=5)
        consequence = ev.get("consequence_potential", pd.Series(dtype=str)).value_counts().head(3).index.tolist()
        source_mix = ev.get("source_type", pd.Series(dtype=str)).value_counts().head(5).to_dict()
        examples = ev.sort_values("text_length", ascending=False).head(3)["event_id"].tolist() if "text_length" in ev.columns else ev.head(3)["event_id"].tolist()
        row = {
            "risk_theme_id": f"RT{theme_no:04d}",
            "cluster_label": int(lab),
            "cluster_size_in_discovery_sample": int(cluster_mask.sum()),
            "mean_membership_strength": float(np.mean(strengths[cluster_mask])),
            "top_hazard_tags": "|".join(hazard),
            "top_control_failure_tags": "|".join(control),
            "top_consequence_potential": "|".join([str(v) for v in consequence]),
            "top_terms": "|".join(terms),
            "source_type_mix_in_sample": str(source_mix),
            "example_event_ids": "|".join(map(str, examples)),
        }
        row["risk_theme_name"] = readable_name(row)
        theme_rows.append(row)

    theme_library = pd.DataFrame(theme_rows)
    if len(centroids) > 0:
        centroids = normalize_embeddings(np.vstack(centroids).astype(np.float32))
    else:
        centroids = np.zeros((0, x.shape[1]), dtype=np.float32)

    if not theme_library.empty:
        membership = membership.merge(theme_library[["risk_theme_id", "cluster_label", "risk_theme_name"]], on="cluster_label", how="left")
    else:
        membership["risk_theme_id"] = None
        membership["risk_theme_name"] = None
    membership.loc[membership["cluster_label"].eq(-1), "risk_theme_id"] = "OUTLIER"
    membership.loc[membership["cluster_label"].eq(-1), "risk_theme_name"] = "Outlier / needs review"

    save_csv(membership, output_dir / "discovered_theme_memberships.csv.gz")
    save_csv(theme_library, output_dir / "risk_theme_library.csv")
    np.save(output_dir / "risk_theme_centroids.npy", centroids.astype(np.float32))
    dump(model_bundle, model_dir / "risk_theme_discovery_model.joblib")

    sil = None
    try:
        if len(np.unique(labels)) > 1 and len(x_cluster) > 100:
            sample_n = min(10000, len(x_cluster))
            rng = _rng(cfg.RANDOM_STATE)
            sample = rng.choice(np.arange(len(x_cluster)), size=sample_n, replace=False)
            sil = float(silhouette_score(x_cluster[sample], labels[sample], metric="cosine"))
    except Exception:
        sil = None

    source_counts = clustered_events.get("source_type", pd.Series(dtype=str)).value_counts().to_dict()
    summary = {
        "method": method,
        "total_rows_available_for_step4_assignment": int(len(events)),
        "candidate_rows_for_theme_discovery": int(len(candidate_idx)),
        "sampled_rows_for_theme_discovery": int(len(x_cluster)),
        "sample_strategy": args.sample_strategy,
        "exclude_tasks_from_theme_discovery": bool(args.exclude_tasks),
        "sample_source_type_counts": source_counts,
        "cluster_algorithm": args.cluster_algorithm,
        "cluster_selection_epsilon": float(args.cluster_selection_epsilon),
        "umap_neighbors": int(args.umap_neighbors),
        "umap_components": int(args.umap_components),
        "umap_random_state": args.umap_random_state,
        "umap_n_jobs": int(args.umap_n_jobs),
        "cluster_count": int(len(valid_clusters)),
        "outlier_count_in_discovery_sample": int((labels == -1).sum()),
        "silhouette_cosine_sample": sil,
        "theme_library_path": str(output_dir / "risk_theme_library.csv"),
        "centroids_path": str(output_dir / "risk_theme_centroids.npy"),
        "note": "Step 3 discovers themes on a representative sample only. Step 4 assigns every record to the nearest discovered theme centroid by cosine similarity.",
    }
    save_json(summary, output_dir / "03_risk_theme_discovery_summary.json")
    print(summary)


if __name__ == "__main__":
    main()
