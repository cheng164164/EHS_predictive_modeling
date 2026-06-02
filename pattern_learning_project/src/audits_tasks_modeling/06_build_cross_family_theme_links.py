#!/usr/bin/env python3
"""Build candidate links between incident, audit, and task themes.

These are NOT causal links. They are review candidates based on:
  1) semantic similarity between theme documents
  2) co-occurrence in the same location-period
"""
from __future__ import annotations

try:
    import config as cfg
    from theme_utils import ProgressLogger, compact_text, ensure_dir, read_csv, save_json, write_csv
except ImportError:  # pragma: no cover
    from . import config as cfg
    from .theme_utils import ProgressLogger, compact_text, ensure_dir, read_csv, save_json, write_csv

from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


FAMILY_PAIRS = [
    ("incident_hazard", "audit_risk"),
    ("incident_hazard", "audit_positive"),
    ("incident_hazard", "task_action"),
    ("audit_risk", "audit_positive"),
    ("audit_risk", "task_action"),
    ("audit_positive", "task_action"),
]


def _theme_doc(row: pd.Series) -> str:
    parts = [
        row.get("theme_name", ""),
        row.get("theme_description", ""),
        row.get("top_terms", ""),
        row.get("event_kind_mix", ""),
        row.get("category_mix", ""),
        row.get("representative_texts", ""),
    ]
    return compact_text(" | ".join([str(x) for x in parts if str(x).strip()]), 6000)


def _build_theme_period_sets(profile: pd.DataFrame) -> dict[str, set[str]]:
    sets: dict[str, set[str]] = {}
    for theme_id, g in profile.groupby("theme_id", dropna=False):
        keys = (g["location_id"].astype(str) + "::" + g["period"].astype(str)).tolist()
        sets[str(theme_id)] = set(keys)
    return sets


def _top_locations_for_pair(profile: pd.DataFrame, theme_a: str, theme_b: str, max_items: int = 8) -> str:
    a = profile[profile["theme_id"].astype(str).eq(str(theme_a))][["location_id", "location_path", "period"]].copy()
    b = profile[profile["theme_id"].astype(str).eq(str(theme_b))][["location_id", "location_path", "period"]].copy()
    m = a.merge(b, on=["location_id", "period"], suffixes=("_a", "_b"))
    if m.empty:
        return ""
    loc = m["location_path_a"].fillna(m["location_path_b"]).astype(str).value_counts().head(max_items)
    return "; ".join([f"{idx}={int(v)}" for idx, v in loc.items()])


def main() -> None:
    log = ProgressLogger("06_build_cross_family_theme_links")
    ensure_dir(cfg.THEME_LINK_DIR)

    if not bool(getattr(cfg, "ENABLE_CROSS_FAMILY_LINKS", True)):
        log.log("ENABLE_CROSS_FAMILY_LINKS=False; skipping")
        return
    if not cfg.THEME_CATALOG_FILE.exists():
        raise FileNotFoundError(f"Missing theme catalog: {cfg.THEME_CATALOG_FILE}. Run 04_label_theme_clusters.py first.")
    if not cfg.LOCATION_THEME_PERIOD_FILE.exists():
        raise FileNotFoundError(f"Missing location/theme profile: {cfg.LOCATION_THEME_PERIOD_FILE}. Run 05_build_location_theme_period_profiles.py first.")

    catalog = read_csv(cfg.THEME_CATALOG_FILE)
    profile = read_csv(cfg.LOCATION_THEME_PERIOD_FILE)
    # Exclude unassigned/generic noise from automated link candidates.
    catalog = catalog[~catalog["theme_id"].astype(str).str.contains("UNASSIGNED", case=False, na=False)].copy()
    catalog = catalog[catalog["theme_size"].fillna(0).astype(float) >= int(cfg.MIN_THEME_SIZE_FOR_CATALOG)].copy()
    if catalog.empty:
        raise ValueError("No catalog themes available for linking.")

    log.log(f"building theme documents for {len(catalog):,} themes")
    catalog["theme_doc"] = catalog.apply(_theme_doc, axis=1)
    vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=1, stop_words="english")
    x = vectorizer.fit_transform(catalog["theme_doc"].fillna(""))
    theme_to_idx = {str(t): i for i, t in enumerate(catalog["theme_id"].astype(str))}
    period_sets = _build_theme_period_sets(profile)

    rows = []
    for family_a, family_b in FAMILY_PAIRS:
        a = catalog[catalog["source_family"].astype(str).eq(family_a)].copy()
        b = catalog[catalog["source_family"].astype(str).eq(family_b)].copy()
        if a.empty or b.empty:
            continue
        log.log(f"linking {family_a} -> {family_b}; themes={len(a):,} x {len(b):,}")
        a_idx = [theme_to_idx[str(t)] for t in a["theme_id"].astype(str)]
        b_idx = [theme_to_idx[str(t)] for t in b["theme_id"].astype(str)]
        sim = cosine_similarity(x[a_idx], x[b_idx])
        for i, row_a in enumerate(a.itertuples(index=False)):
            sims = sim[i]
            # Preselect by semantic similarity, then keep top N.
            candidates = np.where(sims >= float(cfg.LINK_MIN_COSINE_SIMILARITY))[0]
            if len(candidates) == 0:
                candidates = np.argsort(-sims)[: int(cfg.LINK_MAX_PAIRS_PER_SOURCE_THEME)]
            else:
                candidates = candidates[np.argsort(-sims[candidates])[: int(cfg.LINK_MAX_PAIRS_PER_SOURCE_THEME)]]
            set_a = period_sets.get(str(row_a.theme_id), set())
            for j in candidates:
                row_b = b.iloc[int(j)]
                set_b = period_sets.get(str(row_b["theme_id"]), set())
                co = len(set_a & set_b)
                if co < int(cfg.LINK_MIN_COOCCURRENCE_COUNT) and float(sims[j]) < float(cfg.LINK_MIN_COSINE_SIMILARITY):
                    continue
                co_norm = min(co / 10.0, 1.0)
                score = float(cfg.LINK_SCORE_SIMILARITY_WEIGHT) * float(sims[j]) + float(cfg.LINK_SCORE_COOCCURRENCE_WEIGHT) * co_norm
                rows.append({
                    "source_theme_id": str(row_a.theme_id),
                    "target_theme_id": str(row_b["theme_id"]),
                    "source_family": family_a,
                    "target_family": family_b,
                    "source_theme_name": str(row_a.theme_name),
                    "target_theme_name": str(row_b["theme_name"]),
                    "semantic_similarity": float(sims[j]),
                    "location_period_cooccurrence_count": int(co),
                    "link_score": score,
                    "link_strength": "strong" if score >= 0.70 and co >= 2 else "possible" if score >= 0.50 else "weak",
                    "source_theme_size": int(row_a.theme_size),
                    "target_theme_size": int(row_b["theme_size"]),
                    "source_top_terms": str(row_a.top_terms),
                    "target_top_terms": str(row_b["top_terms"]),
                    "representative_cooccurring_locations": _top_locations_for_pair(profile, str(row_a.theme_id), str(row_b["theme_id"]), 8),
                })

    links = pd.DataFrame(rows)
    if not links.empty:
        links = links.sort_values(["link_score", "location_period_cooccurrence_count", "semantic_similarity"], ascending=[False, False, False])
    write_csv(links, cfg.CROSS_FAMILY_LINKS_FILE)

    # A second review table: location-period rows where multiple source families are active.
    log.log("building location-period cross-family candidate table")
    active = profile[~profile["theme_id"].astype(str).str.contains("UNASSIGNED", case=False, na=False)].copy()
    rows2 = []
    for keys, g in active.groupby(["location_id", "location_path", "period"], dropna=False):
        families = set(g["source_family"].astype(str))
        if len(families) < 2:
            continue
        rows2.append({
            "location_id": keys[0],
            "location_path": keys[1],
            "period": keys[2],
            "source_families_active": ";".join(sorted(families)),
            "theme_count": int(g["theme_id"].nunique()),
            "event_count": int(g["event_count"].sum()),
            "review_score": float(g["theme_review_score"].sum()),
            "incident_hazard_themes": " || ".join(g[g["source_family"].eq("incident_hazard")].sort_values("event_count", ascending=False).head(5).apply(lambda r: f"{r['theme_id']}: {r['theme_name']} ({int(r['event_count'])})", axis=1)),
            "audit_risk_themes": " || ".join(g[g["source_family"].eq("audit_risk")].sort_values("event_count", ascending=False).head(5).apply(lambda r: f"{r['theme_id']}: {r['theme_name']} ({int(r['event_count'])})", axis=1)),
            "audit_positive_themes": " || ".join(g[g["source_family"].eq("audit_positive")].sort_values("event_count", ascending=False).head(5).apply(lambda r: f"{r['theme_id']}: {r['theme_name']} ({int(r['event_count'])})", axis=1)),
            "task_action_themes": " || ".join(g[g["source_family"].eq("task_action")].sort_values("event_count", ascending=False).head(5).apply(lambda r: f"{r['theme_id']}: {r['theme_name']} ({int(r['event_count'])})", axis=1)),
        })
    cross = pd.DataFrame(rows2)
    if not cross.empty:
        cross = cross.sort_values(["review_score", "event_count"], ascending=[False, False])
    write_csv(cross, cfg.LOCATION_DOMAIN_CANDIDATE_FILE)

    save_json({
        "cross_family_links_file": str(cfg.CROSS_FAMILY_LINKS_FILE),
        "location_period_candidates_file": str(cfg.LOCATION_DOMAIN_CANDIDATE_FILE),
        "link_rows": int(len(links)),
        "location_period_candidate_rows": int(len(cross)),
        "note": "Links are candidates for review, not causal proof.",
    }, cfg.THEME_LINK_DIR / "cross_family_link_summary.json")
    log.done("cross-family candidate linking complete")


if __name__ == "__main__":
    main()
