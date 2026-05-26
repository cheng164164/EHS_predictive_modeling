"""Train Sentence Embeddings + HDBSCAN recurring safety pattern clusters.

Run directly from the project root:

    python src/pattern_learning_unsupervised/train_pattern_clusters_hdbscan.py

Required upstream input:

    outputs/processed/pattern_learning_records.csv

Main downstream output consumed by supervised classification:

    outputs/modeling/hdbscan_patterns/final/pattern_learning_clustered_records.csv
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Allow direct execution from project root without installing the package.
if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from pattern_learning_unsupervised import config
from pattern_learning_unsupervised.pattern_hdbscan_utils import (
    approximate_hdbscan_predict,
    attach_cluster_labels,
    attach_theme_ids,
    attach_themes_to_cluster_summary,
    build_cluster_monthly_trend,
    build_cluster_site_summary,
    build_cluster_summary,
    build_theme_monthly_trend,
    build_theme_site_summary,
    build_theme_summary,
    clean_pattern_records,
    compact_business_columns,
    compute_clustering_metrics,
    compute_cluster_centroids,
    compute_theme_top_terms,
    compute_top_terms,
    encode_texts,
    fit_cluster_theme_model,
    fit_hdbscan,
    fit_umap,
    load_sentence_model,
    make_train_test_split,
    metrics_to_frame,
    optionally_sample_records,
    plot_cluster_size_distribution,
    plot_membership_strength,
    plot_outlier_rate_by_month,
    plot_top_clusters_by_severe,
    read_table,
    representative_records,
    save_model_artifacts,
    transform_umap,
    write_json,
    write_table,
)


def build_run_config() -> dict:
    """Collect all tunable values from config.py into a saved manifest."""
    return {
        "input_path": str(config.PATTERN_LEARNING_RECORDS_PATH),
        "clustered_output_path": str(config.CLUSTERED_PATTERN_RECORDS_PATH),
        "text_col": config.TEXT_COL,
        "id_col": config.ID_COL,
        "embedding_model": config.EMBEDDING_MODEL,
        "device": config.DEVICE,
        "batch_size": int(config.BATCH_SIZE),
        "min_words": int(config.MIN_WORDS),
        "max_records": config.MAX_RECORDS,
        "split_mode": config.SPLIT_MODE,
        "test_size": float(config.TEST_SIZE),
        "random_state": int(config.RANDOM_STATE),
        "use_umap": bool(config.USE_UMAP),
        "umap": {
            "n_neighbors": int(config.UMAP_N_NEIGHBORS),
            "n_components": int(config.UMAP_N_COMPONENTS),
            "min_dist": float(config.UMAP_MIN_DIST),
            "metric": config.UMAP_METRIC,
        },
        "hdbscan": {
            "min_cluster_size": int(config.MIN_CLUSTER_SIZE),
            "min_samples": int(config.MIN_SAMPLES),
            "metric": config.HDBSCAN_METRIC,
            "cluster_selection_method": config.CLUSTER_SELECTION_METHOD,
            "cluster_selection_epsilon": float(config.CLUSTER_SELECTION_EPSILON),
            "core_dist_n_jobs": -1,
        },
        "theme": {
            "enabled": bool(getattr(config, "ENABLE_THEMES", False)),
            "method": getattr(config, "THEME_METHOD", "agglomerative"),
            "centroid_space": getattr(config, "THEME_CENTROID_SPACE", "embedding"),
            "distance_threshold": float(getattr(config, "THEME_DISTANCE_THRESHOLD", 0.25)),
            "n_clusters": getattr(config, "THEME_N_CLUSTERS", None),
            "metric": getattr(config, "THEME_METRIC", "cosine"),
            "linkage": getattr(config, "THEME_LINKAGE", "average"),
            "use_membership_weights": bool(getattr(config, "THEME_USE_MEMBERSHIP_WEIGHTS", True)),
            "min_records_per_cluster": int(getattr(config, "THEME_MIN_RECORDS_PER_CLUSTER", 1)),
            "top_terms_n": int(getattr(config, "THEME_TOP_TERMS_N", 12)),
            "top_clusters_n": int(getattr(config, "THEME_TOP_CLUSTERS_N", 8)),
        },
        "metric_sample_size": int(config.METRIC_SAMPLE_SIZE),
        "fit_final": bool(config.FIT_FINAL),
    }


def fit_reduce_cluster(
    embeddings: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    run_config: dict,
) -> dict:
    """Fit train-split UMAP/HDBSCAN and score holdout records."""
    train_embeddings = embeddings[train_idx]
    test_embeddings = embeddings[test_idx]

    if run_config["use_umap"]:
        umap_model, train_vectors = fit_umap(train_embeddings, run_config["umap"], run_config["random_state"])
        test_vectors = transform_umap(umap_model, test_embeddings)
    else:
        umap_model = None
        train_vectors = train_embeddings
        test_vectors = test_embeddings

    hdbscan_model, train_labels, train_strengths = fit_hdbscan(train_vectors, run_config["hdbscan"])

    try:
        test_labels, test_strengths = approximate_hdbscan_predict(hdbscan_model, test_vectors)
    except Exception as exc:
        print(f"WARNING: approximate_predict failed. Test records will be marked as outliers. Error: {exc}")
        test_labels = np.full(len(test_idx), -1, dtype=int)
        test_strengths = np.zeros(len(test_idx), dtype=np.float32)

    return {
        "umap_model": umap_model,
        "hdbscan_model": hdbscan_model,
        "train_vectors": train_vectors,
        "test_vectors": test_vectors,
        "train_labels": train_labels,
        "test_labels": test_labels,
        "train_strengths": train_strengths,
        "test_strengths": test_strengths,
    }


def fit_final_model(embeddings: np.ndarray, run_config: dict) -> dict:
    """Fit final UMAP/HDBSCAN artifacts on all eligible pattern-learning records."""
    if run_config["use_umap"]:
        umap_model, vectors = fit_umap(embeddings, run_config["umap"], run_config["random_state"])
    else:
        umap_model = None
        vectors = embeddings
    hdbscan_model, labels, strengths = fit_hdbscan(vectors, run_config["hdbscan"])
    return {
        "umap_model": umap_model,
        "hdbscan_model": hdbscan_model,
        "vectors": vectors,
        "labels": labels,
        "strengths": strengths,
    }


def save_sentence_model_if_requested(model_name: str, output_dir: Path, device: str) -> None:
    """Optionally save a local copy of the SentenceTransformer model."""
    sentence_dir = output_dir / "artifacts" / "sentence_model"
    if sentence_dir.exists():
        shutil.rmtree(sentence_dir)
    sentence_dir.parent.mkdir(parents=True, exist_ok=True)
    model = load_sentence_model(model_name, device=device)
    model.save(str(sentence_dir))


def choose_theme_source_vectors(
    embeddings_in_row_order: np.ndarray,
    clustering_vectors_in_row_order: np.ndarray,
    run_config: dict,
) -> np.ndarray:
    """Choose the vector space used to group detailed clusters into themes."""
    theme_config = run_config.get("theme", {})
    space = str(theme_config.get("centroid_space", "embedding")).lower()
    if space == "umap":
        return clustering_vectors_in_row_order
    if space != "embedding":
        raise ValueError("THEME_CENTROID_SPACE must be either 'embedding' or 'umap'")
    return embeddings_in_row_order


def add_theme_layer(
    scored_df: pd.DataFrame,
    theme_source_vectors: np.ndarray,
    labels: np.ndarray,
    strengths: np.ndarray,
    cluster_summary: pd.DataFrame,
    run_config: dict,
    split_name: str,
) -> dict:
    """Attach a generic second-level theme layer above HDBSCAN clusters.

    The theme model is fit from cluster centroid vectors. It does not use
    hand-selected keywords, result-specific stop words, severe outcome labels,
    site names, or future information. Theme text labels are computed only after
    the grouping exists, for interpretability and reporting.
    """
    theme_config = dict(run_config.get("theme", {}))
    enabled = bool(theme_config.get("enabled", False))
    if not enabled:
        return {
            "scored": attach_cluster_labels(scored_df, cluster_summary),
            "cluster_summary": cluster_summary,
            "theme_summary": pd.DataFrame(),
            "cluster_theme_map": pd.DataFrame(),
            "theme_model_bundle": None,
            "theme_info": {"enabled": False, "split_name": split_name},
        }

    cluster_ids, centroids, centroid_summary = compute_cluster_centroids(
        theme_source_vectors,
        labels,
        membership_strength=strengths,
        use_membership_weights=bool(theme_config.get("use_membership_weights", True)),
        min_records_per_cluster=int(theme_config.get("min_records_per_cluster", 1)),
    )
    theme_model_bundle, cluster_theme_map, theme_info = fit_cluster_theme_model(
        cluster_ids=cluster_ids,
        centroids=centroids,
        cluster_summary=cluster_summary,
        config=theme_config,
        random_state=int(run_config.get("random_state", 42)),
    )
    if not cluster_theme_map.empty and not centroid_summary.empty:
        cluster_theme_map = cluster_theme_map.merge(centroid_summary, on="cluster_id", how="left")
    if not cluster_theme_map.empty:
        cluster_theme_map["theme_split_name"] = split_name

    # First attach only cluster labels, then attach numeric theme IDs.
    temp_scored = attach_cluster_labels(scored_df, cluster_summary)
    temp_scored = attach_theme_ids(temp_scored, cluster_theme_map)

    # Theme labels are generated after fitting and do not influence grouping.
    theme_top_terms = compute_theme_top_terms(
        temp_scored,
        text_col="model_text",
        top_n=int(theme_config.get("top_terms_n", 12)),
    )
    theme_summary = build_theme_summary(
        temp_scored,
        theme_top_terms=theme_top_terms,
        top_clusters_n=int(theme_config.get("top_clusters_n", 8)),
    )
    cluster_summary = attach_themes_to_cluster_summary(
        cluster_summary,
        cluster_theme_map=cluster_theme_map,
        theme_summary=theme_summary,
    )
    scored_with_labels = attach_cluster_labels(scored_df, cluster_summary)
    theme_info = dict(theme_info)
    theme_info["enabled"] = True
    theme_info["split_name"] = split_name

    return {
        "scored": scored_with_labels,
        "cluster_summary": cluster_summary,
        "theme_summary": theme_summary,
        "cluster_theme_map": cluster_theme_map,
        "theme_model_bundle": theme_model_bundle,
        "theme_info": theme_info,
    }


def main() -> None:
    start_time = time.time()
    output_dir = Path(config.HDBSCAN_OUTPUT_DIR).resolve()
    validation_dir = output_dir / "validation"
    final_dir = output_dir / "final"
    plots_dir = output_dir / "plots"
    artifact_dir = output_dir / "artifacts"
    for folder in [validation_dir, final_dir, plots_dir, artifact_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    run_config = build_run_config()
    write_json(run_config, output_dir / "run_config.json")

    input_path = Path(config.PATTERN_LEARNING_RECORDS_PATH)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing pattern learning input file: {input_path}\n"
            "Run the data-preparation pipeline first: python src/run_data_prep.py"
        )

    print("Loading prepared pattern-learning records", flush=True)
    print(f"Input: {input_path}", flush=True)
    raw_df = read_table(input_path)
    print(f"Raw shape: {raw_df.shape[0]:,} rows x {raw_df.shape[1]:,} columns", flush=True)

    print("Cleaning pattern records before embedding", flush=True)
    clean_df, rejected_df, data_profile = clean_pattern_records(
        raw_df,
        text_col=config.TEXT_COL,
        min_words=int(config.MIN_WORDS),
        id_col=config.ID_COL,
    )
    if rejected_df is not None and len(rejected_df) > 0:
        write_table(rejected_df, validation_dir / "rejected_records.csv")
    write_json(data_profile, validation_dir / "data_profile.json")
    print(f"Eligible rows after cleaning: {len(clean_df):,}", flush=True)

    if config.MAX_RECORDS:
        clean_df = optionally_sample_records(clean_df, int(config.MAX_RECORDS), int(config.RANDOM_STATE))
        print(f"Rows after optional sampling: {len(clean_df):,}", flush=True)

    clean_df = clean_df.reset_index(drop=True)
    write_table(compact_business_columns(clean_df), validation_dir / "cleaned_modeling_records_sample.csv")

    print("Creating validation split", flush=True)
    train_idx, test_idx, split_info = make_train_test_split(
        clean_df,
        split_mode=config.SPLIT_MODE,
        test_size=float(config.TEST_SIZE),
        random_state=int(config.RANDOM_STATE),
    )
    write_json(split_info, validation_dir / "split_info.json")
    print(json.dumps(split_info, indent=2, default=str), flush=True)

    print("Generating sentence embeddings", flush=True)
    embeddings = encode_texts(
        clean_df["model_text"].astype(str).tolist(),
        model_name_or_path=config.EMBEDDING_MODEL,
        batch_size=int(config.BATCH_SIZE),
        device=config.DEVICE,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    print(f"Embedding matrix shape: {embeddings.shape}", flush=True)
    if config.SAVE_EMBEDDINGS:
        np.save(artifact_dir / "all_sentence_embeddings.npy", embeddings)
        clean_df[["record_id", "row_uid", "model_text_hash"]].to_csv(artifact_dir / "embedding_row_map.csv", index=False)

    print("Training validation model on train split", flush=True)
    dev = fit_reduce_cluster(embeddings, train_idx, test_idx, run_config)

    train_df = clean_df.iloc[train_idx].copy().reset_index(drop=True)
    train_df["split"] = "train"
    train_df["cluster_id"] = dev["train_labels"]
    train_df["membership_strength"] = dev["train_strengths"]
    train_df["is_outlier"] = train_df["cluster_id"].eq(-1)

    test_df = clean_df.iloc[test_idx].copy().reset_index(drop=True)
    test_df["split"] = "test"
    test_df["cluster_id"] = dev["test_labels"]
    test_df["membership_strength"] = dev["test_strengths"]
    test_df["is_outlier"] = test_df["cluster_id"].eq(-1)

    print("Computing validation metrics", flush=True)
    metrics = [
        compute_clustering_metrics(
            "train",
            dev["train_vectors"],
            dev["train_labels"],
            dev["train_strengths"],
            max_metric_points=int(config.METRIC_SAMPLE_SIZE),
            random_state=int(config.RANDOM_STATE),
        ),
        compute_clustering_metrics(
            "test_approx_predict",
            dev["test_vectors"],
            dev["test_labels"],
            dev["test_strengths"],
            max_metric_points=int(config.METRIC_SAMPLE_SIZE),
            random_state=int(config.RANDOM_STATE),
        ),
    ]
    metrics_df = metrics_to_frame(metrics)
    metrics_df.to_csv(validation_dir / "validation_metrics.csv", index=False)
    write_json(metrics_df.to_dict(orient="records"), validation_dir / "validation_metrics.json")
    print(metrics_df.to_string(index=False), flush=True)

    validation_scored = pd.concat([train_df, test_df], ignore_index=True)
    validation_top_terms = compute_top_terms(train_df, dev["train_labels"], text_col="model_text")
    validation_cluster_summary = build_cluster_summary(
        train_df,
        dev["train_labels"],
        dev["train_strengths"],
        validation_top_terms,
    )

    print("Building validation theme layer", flush=True)
    validation_train_embeddings = embeddings[train_idx]
    validation_theme_source_vectors = choose_theme_source_vectors(
        embeddings_in_row_order=validation_train_embeddings,
        clustering_vectors_in_row_order=dev["train_vectors"],
        run_config=run_config,
    )
    validation_theme = add_theme_layer(
        scored_df=validation_scored,
        theme_source_vectors=validation_theme_source_vectors,
        labels=dev["train_labels"],
        strengths=dev["train_strengths"],
        cluster_summary=validation_cluster_summary,
        run_config=run_config,
        split_name="validation_train",
    )
    validation_scored = validation_theme["scored"]
    validation_cluster_summary = validation_theme["cluster_summary"]
    validation_theme_summary = validation_theme["theme_summary"]
    validation_cluster_theme_map = validation_theme["cluster_theme_map"]
    write_json(validation_theme["theme_info"], validation_dir / "validation_theme_info.json")

    write_table(compact_business_columns(validation_scored), validation_dir / "validation_scored_records.csv")
    write_table(validation_cluster_summary, validation_dir / "validation_cluster_summary.csv")
    if not validation_theme_summary.empty:
        write_table(validation_theme_summary, validation_dir / "validation_theme_summary.csv")
    if not validation_cluster_theme_map.empty:
        write_table(validation_cluster_theme_map, validation_dir / "validation_cluster_theme_map.csv")
    reps = representative_records(train_df, dev["train_labels"], dev["train_strengths"], top_k=5)
    reps = attach_cluster_labels(reps, validation_cluster_summary) if not reps.empty else reps
    write_table(reps, validation_dir / "validation_representative_records.csv")
    plot_membership_strength(dev["test_strengths"], plots_dir / "test_membership_strength.png", "Validation Test Membership Strength")

    if config.FIT_FINAL:
        print("Fitting final model on all cleaned records", flush=True)
        final = fit_final_model(embeddings, run_config)
        final_df = clean_df.copy()
        final_df["split"] = "final_all_records"
        final_df["cluster_id"] = final["labels"]
        final_df["membership_strength"] = final["strengths"]
        final_df["is_outlier"] = final_df["cluster_id"].eq(-1)
        final_vectors = final["vectors"]
        final_embeddings_in_order = embeddings
        final_labels = final["labels"]
        final_strengths = final["strengths"]
        final_umap_model = final["umap_model"]
        final_hdbscan_model = final["hdbscan_model"]
    else:
        print("Using train-fitted validation model as final artifact", flush=True)
        # Keep the final dataframe in the same row order as the stacked vectors.
        # Do not reuse validation_scored here because it already has label columns
        # attached, which can create duplicate columns when final labels are merged.
        final_df = pd.concat([train_df, test_df], ignore_index=True)
        final_vectors = np.vstack([dev["train_vectors"], dev["test_vectors"]])
        final_embeddings_in_order = np.vstack([embeddings[train_idx], embeddings[test_idx]])
        final_labels = final_df["cluster_id"].to_numpy(dtype=int)
        final_strengths = final_df["membership_strength"].to_numpy(dtype=float)
        final_umap_model = dev["umap_model"]
        final_hdbscan_model = dev["hdbscan_model"]

    print("Creating final model outputs", flush=True)
    final_metrics = compute_clustering_metrics(
        "final_all_records",
        final_vectors,
        final_labels,
        final_strengths,
        max_metric_points=int(config.METRIC_SAMPLE_SIZE),
        random_state=int(config.RANDOM_STATE),
    )
    final_metrics_df = metrics_to_frame([final_metrics])
    final_metrics_df.to_csv(final_dir / "final_model_metrics.csv", index=False)

    final_top_terms = compute_top_terms(final_df, final_labels, text_col="model_text")
    cluster_summary = build_cluster_summary(final_df, final_labels, final_strengths, final_top_terms)

    print("Building final theme layer", flush=True)
    final_theme_source_vectors = choose_theme_source_vectors(
        embeddings_in_row_order=final_embeddings_in_order,
        clustering_vectors_in_row_order=final_vectors,
        run_config=run_config,
    )
    final_theme = add_theme_layer(
        scored_df=final_df,
        theme_source_vectors=final_theme_source_vectors,
        labels=final_labels,
        strengths=final_strengths,
        cluster_summary=cluster_summary,
        run_config=run_config,
        split_name="final_all_records",
    )
    final_scored = final_theme["scored"]
    cluster_summary = final_theme["cluster_summary"]
    theme_summary = final_theme["theme_summary"]
    cluster_theme_map = final_theme["cluster_theme_map"]
    theme_model_bundle = final_theme["theme_model_bundle"]
    theme_info = final_theme["theme_info"]
    write_json(theme_info, final_dir / "theme_info.json")

    clustered_records = compact_business_columns(final_scored)
    cluster_site_summary = build_cluster_site_summary(final_scored)
    cluster_monthly_trend = build_cluster_monthly_trend(final_scored)
    theme_site_summary = build_theme_site_summary(final_scored)
    theme_monthly_trend = build_theme_monthly_trend(final_scored)
    final_reps = representative_records(final_df, final_labels, final_strengths, top_k=5)
    final_reps = attach_cluster_labels(final_reps, cluster_summary) if not final_reps.empty else final_reps

    write_table(clustered_records, Path(config.CLUSTERED_PATTERN_RECORDS_PATH))
    write_table(cluster_summary, final_dir / "cluster_summary.csv")
    write_table(cluster_site_summary, final_dir / "cluster_site_summary.csv")
    write_table(cluster_monthly_trend, final_dir / "cluster_monthly_trend.csv")
    if not theme_summary.empty:
        write_table(theme_summary, final_dir / "theme_summary.csv")
    if not cluster_theme_map.empty:
        write_table(cluster_theme_map, final_dir / "cluster_theme_map.csv")
    if not theme_site_summary.empty:
        write_table(theme_site_summary, final_dir / "theme_site_summary.csv")
    if not theme_monthly_trend.empty:
        write_table(theme_monthly_trend, final_dir / "theme_monthly_trend.csv")
    write_table(final_reps, final_dir / "representative_records.csv")

    plot_cluster_size_distribution(cluster_summary, plots_dir / "cluster_size_distribution.png")
    plot_membership_strength(final_strengths, plots_dir / "final_membership_strength.png", "Final Model Membership Strength")
    plot_outlier_rate_by_month(final_scored, plots_dir / "outlier_rate_by_month.png")
    plot_top_clusters_by_severe(cluster_summary, plots_dir / "top_clusters_by_historical_severe_actual.png")

    print("Saving model artifacts", flush=True)
    save_model_artifacts(
        output_dir=output_dir,
        sentence_model_name=config.EMBEDDING_MODEL,
        umap_model=final_umap_model,
        hdbscan_model=final_hdbscan_model,
        config=run_config,
        cluster_summary=cluster_summary,
        theme_summary=theme_summary,
        cluster_theme_map=cluster_theme_map,
        theme_model_bundle=theme_model_bundle,
    )
    if config.SAVE_SENTENCE_MODEL:
        print("Saving local copy of SentenceTransformer model", flush=True)
        save_sentence_model_if_requested(config.EMBEDDING_MODEL, output_dir, config.DEVICE)

    manifest = {
        "runtime_seconds": round(time.time() - start_time, 2),
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "raw_rows": int(len(raw_df)),
        "clean_rows": int(len(clean_df)),
        "validation_split": split_info,
        "data_profile": data_profile,
        "final_metrics": final_metrics_df.to_dict(orient="records"),
        "cluster_count_excluding_outliers": int(cluster_summary[cluster_summary["cluster_id"] != -1].shape[0]),
        "theme_count_excluding_outliers": int(theme_summary[theme_summary["theme_id"] != -1].shape[0]) if not theme_summary.empty and "theme_id" in theme_summary.columns else 0,
        "outlier_records": int((final_labels == -1).sum()),
        "theme_info": theme_info,
        "output_files": {
            "clustered_records": str(config.CLUSTERED_PATTERN_RECORDS_PATH),
            "cluster_summary": str(final_dir / "cluster_summary.csv"),
            "cluster_site_summary": str(final_dir / "cluster_site_summary.csv"),
            "cluster_monthly_trend": str(final_dir / "cluster_monthly_trend.csv"),
            "theme_summary": str(final_dir / "theme_summary.csv"),
            "cluster_theme_map": str(final_dir / "cluster_theme_map.csv"),
            "theme_site_summary": str(final_dir / "theme_site_summary.csv"),
            "theme_monthly_trend": str(final_dir / "theme_monthly_trend.csv"),
            "validation_metrics": str(validation_dir / "validation_metrics.csv"),
            "artifacts": str(artifact_dir),
        },
    }
    write_json(manifest, output_dir / "training_manifest.json")

    print("Done", flush=True)
    print(f"Final clustered records: {config.CLUSTERED_PATTERN_RECORDS_PATH}", flush=True)
    print(f"Cluster summary: {final_dir / 'cluster_summary.csv'}", flush=True)
    print(f"Validation metrics: {validation_dir / 'validation_metrics.csv'}", flush=True)
    print(f"Artifacts: {artifact_dir}", flush=True)


if __name__ == "__main__":
    main()
