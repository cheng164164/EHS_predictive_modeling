#!/usr/bin/env python3
"""Profile the unified safety event CSV.

Runs without command-line arguments. All settings are in config.py.
"""
from __future__ import annotations

import csv
from collections import Counter, defaultdict

# Allow direct execution.
try:
    from . import config
    from .utils import ProgressLogger, ensure_dir, location_leaf, open_text_csv, parse_dt, write_json
except ImportError:  # pragma: no cover
    import config  # type: ignore
    from utils import ProgressLogger, ensure_dir, location_leaf, open_text_csv, parse_dt, write_json  # type: ignore

import pandas as pd


def counter_to_csv(counter: Counter, path, key_name: str) -> None:
    df = pd.DataFrame([{key_name: k, "count": v} for k, v in counter.most_common()])
    df.to_csv(path, index=False)


def main() -> None:
    log = ProgressLogger("00_profile_events")
    outdir = ensure_dir(config.PROFILE_DIR)

    if not config.INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {config.INPUT_FILE}")

    counters = {
        "source_type": Counter(),
        "source_subtype": Counter(),
        "category": Counter(),
        "status": Counter(),
        "audit_type": Counter(),
        "task_source_module": Counter(),
        "detected_language": Counter(),
        "year_source_type": Counter(),
    }
    location_stats = defaultdict(lambda: Counter())
    total_rows = 0
    rows_with_date = 0
    rows_with_text = 0
    min_date = None
    max_date = None

    with open_text_csv(config.INPUT_FILE) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        for row in reader:
            total_rows += 1
            source_type = (row.get("source_type") or "").strip() or "(blank)"
            counters["source_type"][source_type] += 1
            counters["source_subtype"][(row.get("source_subtype") or "").strip() or "(blank)"] += 1
            counters["category"][(row.get("category") or "").strip() or "(blank)"] += 1
            counters["status"][(row.get("status") or "").strip() or "(blank)"] += 1
            counters["audit_type"][(row.get("audit_type") or "").strip() or "(blank)"] += 1
            counters["task_source_module"][(row.get("task_source_module") or "").strip() or "(blank)"] += 1
            counters["detected_language"][(row.get("detected_language") or "").strip() or "(blank)"] += 1

            dt = parse_dt(row.get("event_date"))
            if dt:
                rows_with_date += 1
                min_date = dt if min_date is None or dt < min_date else min_date
                max_date = dt if max_date is None or dt > max_date else max_date
                counters["year_source_type"][(str(dt.year), source_type)] += 1

            if (row.get("clean_text") or "").strip():
                rows_with_text += 1

            location_id = str(row.get("location_id") or "").strip()
            location_path = row.get("location_path") or ""
            loc_key = (location_id, location_leaf(location_path), location_path)
            location_stats[loc_key]["event_count"] += 1
            if (row.get("clean_text") or "").strip():
                location_stats[loc_key]["text_event_count"] += 1
            location_stats[loc_key][f"source_type_{source_type}"] += 1

            if total_rows % config.PROGRESS_EVERY_ROWS == 0:
                log.log(f"scanned {total_rows:,} rows; locations={len(location_stats):,}")

    counter_to_csv(counters["source_type"], outdir / "counts_by_source_type.csv", "source_type")
    counter_to_csv(counters["source_subtype"], outdir / "counts_by_source_subtype.csv", "source_subtype")
    counter_to_csv(counters["category"], outdir / "counts_by_category.csv", "category")
    counter_to_csv(counters["status"], outdir / "counts_by_status.csv", "status")
    counter_to_csv(counters["audit_type"], outdir / "counts_by_audit_type.csv", "audit_type")
    counter_to_csv(counters["task_source_module"], outdir / "counts_by_task_source_module.csv", "task_source_module")
    counter_to_csv(counters["detected_language"], outdir / "counts_by_detected_language.csv", "detected_language")

    ys_rows = [
        {"year": year, "source_type": source, "count": count}
        for (year, source), count in counters["year_source_type"].items()
    ]
    pd.DataFrame(ys_rows).sort_values(["year", "source_type"]).to_csv(outdir / "counts_by_year_source_type.csv", index=False)

    loc_rows = []
    for (location_id, location_label, location_path), stats in location_stats.items():
        row = {
            "location_id": location_id,
            "location_label": location_label,
            "location_path": location_path,
        }
        row.update(stats)
        loc_rows.append(row)
    pd.DataFrame(loc_rows).sort_values("event_count", ascending=False).to_csv(outdir / "location_profile.csv", index=False)

    overall = {
        "input_file": str(config.INPUT_FILE),
        "total_rows": total_rows,
        "columns": fieldnames,
        "row_count_with_date": rows_with_date,
        "row_count_with_clean_text": rows_with_text,
        "min_event_date": min_date.isoformat(sep=" ") if min_date else None,
        "max_event_date": max_date.isoformat(sep=" ") if max_date else None,
        "distinct_locations": len(location_stats),
    }
    write_json(outdir / "profile_overall.json", overall)

    log.done(f"wrote profile CSVs to {outdir}; total_rows={total_rows:,}; locations={len(location_stats):,}")


if __name__ == "__main__":
    main()
