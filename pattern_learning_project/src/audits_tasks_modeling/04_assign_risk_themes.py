#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import config as cfg

import numpy as np
import pandas as pd

from utils import ensure_dir, load_table, normalize_embeddings, save_csv, save_json


def assign_in_batches(x, centroids, batch_size: int):
    labels = []
    scores = []
    for start in range(0, x.shape[0], batch_size):
        xb = x[start:start + batch_size]
        sim = xb @ centroids.T
        best = np.argmax(sim, axis=1)
        score = sim[np.arange(sim.shape[0]), best]
        labels.append(best)
        scores.append(score)
    return np.concatenate(labels), np.concatenate(scores)


def main():
    parser = argparse.ArgumentParser(description="Assign every safety text event to the closest discovered risk theme centroid.")
    parser.add_argument("--events", default=cfg.TAGGED_EVENTS_PATH)
    parser.add_argument("--embeddings", default=cfg.TEXT_EMBEDDINGS_PATH)
    parser.add_argument("--theme-library", default=cfg.THEME_LIBRARY_PATH)
    parser.add_argument("--centroids", default=cfg.THEME_CENTROIDS_PATH)
    parser.add_argument("--output-dir", default=cfg.STEP_04_DIR)
    parser.add_argument("--similarity-threshold", type=float, default=cfg.THEME_SIMILARITY_THRESHOLD)
    parser.add_argument("--batch-size", type=int, default=cfg.ASSIGNMENT_BATCH_SIZE)
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    events = load_table(args.events)
    x = normalize_embeddings(np.load(args.embeddings))
    centroids = normalize_embeddings(np.load(args.centroids))
    themes = load_table(args.theme_library)
    if len(events) != x.shape[0]:
        raise ValueError(f"Row mismatch: events={len(events)} embeddings={x.shape[0]}")
    if centroids.shape[0] != len(themes):
        raise ValueError(f"Centroid/theme mismatch: centroids={centroids.shape[0]} themes={len(themes)}")
    if centroids.shape[0] == 0:
        raise ValueError("No risk theme centroids found. Run 03_discover_risk_themes.py first.")

    best_idx, score = assign_in_batches(x, centroids, args.batch_size)
    events["risk_theme_id"] = themes.iloc[best_idx]["risk_theme_id"].to_numpy()
    events["risk_theme_name"] = themes.iloc[best_idx]["risk_theme_name"].to_numpy()
    events["theme_similarity_score"] = score.astype(float)
    events["needs_theme_review"] = events["theme_similarity_score"] < args.similarity_threshold
    events.loc[events["needs_theme_review"], "risk_theme_id"] = "UNASSIGNED"
    events.loc[events["needs_theme_review"], "risk_theme_name"] = "Unassigned / needs review"

    output_path = output_dir / "safety_text_event_themed.csv.gz"
    assignments_path = output_dir / "risk_theme_assignments.csv.gz"
    save_csv(events, output_path)
    save_csv(events[["event_id", "source_type", "source_id", "event_date", "site", "department", "risk_theme_id", "risk_theme_name", "theme_similarity_score", "needs_theme_review"]], assignments_path)
    summary = {
        "output_path": str(output_path),
        "row_count": int(len(events)),
        "assigned_count": int((~events["needs_theme_review"]).sum()),
        "needs_review_count": int(events["needs_theme_review"].sum()),
        "top_themes": events["risk_theme_name"].value_counts().head(20).to_dict(),
    }
    save_json(summary, output_dir / "04_theme_assignment_summary.json")
    print(summary)


if __name__ == "__main__":
    main()
