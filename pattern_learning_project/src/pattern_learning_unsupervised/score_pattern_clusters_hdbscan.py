"""Score prepared near-miss/hazard records with the trained pattern model.

Run directly from the project root:

    python src/pattern_learning_unsupervised/score_pattern_clusters_hdbscan.py

By default, this reads config.SCORE_INPUT_FILE and writes
config.SCORE_OUTPUT_FILE. Edit config.py to score a different file.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from pattern_learning_unsupervised import config
from pattern_learning_unsupervised.pattern_hdbscan_utils import (
    approximate_hdbscan_predict,
    attach_cluster_labels,
    clean_pattern_records,
    compact_business_columns,
    encode_texts,
    load_model_artifacts,
    read_table,
    transform_umap,
    write_json,
    write_table,
)


def main() -> None:
    artifact_dir = Path(config.HDBSCAN_OUTPUT_DIR) / "artifacts"
    input_file = Path(config.SCORE_INPUT_FILE)
    output_file = Path(config.SCORE_OUTPUT_FILE)
    rejected_file = Path(config.SCORE_REJECTED_OUTPUT_FILE)

    artifacts = load_model_artifacts(artifact_dir)
    model_config = artifacts.get("config", {})
    text_col = model_config.get("text_col", config.TEXT_COL)
    id_col = model_config.get("id_col", config.ID_COL)
    min_words = int(model_config.get("min_words", config.MIN_WORDS))

    if not input_file.exists():
        raise FileNotFoundError(f"Missing scoring input file: {input_file}")

    print(f"Loading records: {input_file}", flush=True)
    raw_df = read_table(input_file)
    clean_df, rejected_df, profile = clean_pattern_records(
        raw_df,
        text_col=text_col,
        min_words=min_words,
        id_col=id_col,
    )
    if rejected_df is not None and len(rejected_df) > 0:
        write_table(rejected_df, rejected_file)

    if clean_df.empty:
        raise ValueError("No eligible records remain after cleaning. Check text fields and MIN_WORDS in config.py.")

    print(f"Scoring {len(clean_df):,} eligible records", flush=True)
    embeddings = encode_texts(
        clean_df["model_text"].astype(str).tolist(),
        model_name_or_path=artifacts["sentence_model_name"],
        batch_size=int(config.BATCH_SIZE),
        device=config.DEVICE,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    if artifacts["umap_model"] is not None:
        vectors = transform_umap(artifacts["umap_model"], embeddings)
    else:
        vectors = embeddings

    labels, strengths = approximate_hdbscan_predict(artifacts["hdbscan_model"], vectors)
    clean_df["cluster_id"] = labels.astype(int)
    clean_df["membership_strength"] = strengths.astype(float)
    clean_df["is_outlier"] = clean_df["cluster_id"].eq(-1)
    clean_df["split"] = "scored"

    scored = attach_cluster_labels(clean_df, artifacts["cluster_summary"])
    scored_out = compact_business_columns(scored)
    write_table(scored_out, output_file)
    write_json(profile, output_file.with_suffix(".data_profile.json"))

    print("Done", flush=True)
    print(f"Output: {output_file}", flush=True)
    print(f"Assigned clusters: {int((labels != -1).sum()):,}", flush=True)
    print(f"Outliers / new pattern candidates: {int((labels == -1).sum()):,}", flush=True)
    if "theme_id" in scored_out.columns:
        assigned_themes = scored_out.loc[scored_out["theme_id"].fillna(-1).astype(int).ne(-1), "theme_id"].nunique()
        print(f"Assigned themes represented in scored records: {int(assigned_themes):,}", flush=True)


if __name__ == "__main__":
    main()
