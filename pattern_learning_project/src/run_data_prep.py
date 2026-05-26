"""Run the Pattern Learning data preparation and EDA pipeline.

Default run, no arguments needed:
    python src/run_data_prep.py

Optional override example:
    python src/run_data_prep.py --input-dir /mnt/data --output-dir outputs --reference-date 2026-05-20
"""

from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path

import pandas as pd

from eda import create_eda_outputs
from features import (
    make_site_department_month_features,
    prepare_audits,
    prepare_incidents,
    prepare_injury_agg,
    prepare_incident_injury_all_records,
    prepare_pattern_learning_records,
    prepare_tasks,
)
from locations import build_location_hierarchy
from lookups import prepare_listitems
from utils import read_csv_safely, write_dataframe


RAW_FILE_NAMES = {
    "incident": "INCIDENT_VIEW.csv",
    "injury": "INCIDENTINJURY_VIEW.csv",
    "listitem": "LISTITEM_VIEW.csv",
    "location": "LOCATION_VIEW.csv",
    "task": "TASK_VIEW.csv",
    "audit": "AUDIT_VIEW.csv",
}


def get_project_root() -> Path:
    """Return the project root based on this file location."""
    return Path(__file__).resolve().parents[1]


def find_default_input_dir(project_root: Path) -> Path:
    """Find the most likely raw-data folder so the pipeline can run without arguments.

    Search order:
    1. PATTERN_LEARNING_INPUT_DIR environment variable
    2. <project_root>/data/raw
    3. current working directory / data / raw
    4. current working directory
    5. /mnt/data, useful when running in this ChatGPT sandbox
    """
    env_dir = os.getenv("PATTERN_LEARNING_INPUT_DIR")
    candidates = []
    if env_dir:
        candidates.append(Path(env_dir))
    candidates.extend([
        project_root / "data" / "raw",
        Path.cwd() / "data" / "raw",
        Path.cwd(),
        Path("/mnt/data"),
    ])

    required_files = set(RAW_FILE_NAMES.values())
    for candidate in candidates:
        if candidate.exists() and required_files.issubset({p.name for p in candidate.glob("*.csv")}):
            return candidate

    checked = "\n  - ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        "Could not find the required raw CSV files automatically. "
        "Place the six *_VIEW.csv files in data/raw, run from the folder containing them, "
        "set PATTERN_LEARNING_INPUT_DIR, or pass --input-dir.\n\nChecked:\n  - " + checked
    )


def get_default_output_dir(project_root: Path) -> Path:
    """Default outputs folder, overridable by environment variable."""
    return Path(os.getenv("PATTERN_LEARNING_OUTPUT_DIR", project_root / "outputs"))


def parse_args() -> argparse.Namespace:
    project_root = get_project_root()
    default_output_dir = get_default_output_dir(project_root)

    parser = argparse.ArgumentParser(description="Prepare EHS data for near-miss/hazard pattern learning.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Optional folder containing the raw *_VIEW.csv files. If omitted, the script auto-detects data/raw, the current folder, or /mnt/data.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help=f"Folder for processed data and EDA outputs. Default: {default_output_dir}",
    )
    parser.add_argument(
        "--reference-date",
        type=str,
        default=None,
        help="Reference date for overdue/current snapshot features, e.g. 2026-05-20. Defaults to current UTC date.",
    )
    parser.add_argument(
        "--include-inactive",
        action="store_true",
        help="Include inactive/archived incidents in pattern_learning_records and site-month features. Default excludes them.",
    )
    parser.add_argument(
        "--formats",
        type=str,
        default="csv",
        help="Comma-separated output formats for processed tables: csv, parquet, or csv,parquet.",
    )
    parser.add_argument(
        "--skip-large-processed",
        dest="skip_large_processed",
        action="store_true",
        help="Only write the focused modeling tables and EDA outputs, not the large enriched all-source tables. This is the default.",
    )
    parser.add_argument(
        "--write-large-processed",
        dest="skip_large_processed",
        action="store_false",
        help="Also write the large incident_enriched, task_enriched, and audit_enriched tables. This can require more memory and disk space.",
    )
    parser.set_defaults(skip_large_processed=True)

    args = parser.parse_args()
    args.project_root = project_root
    if args.input_dir is None:
        args.input_dir = find_default_input_dir(project_root)
    return args


def parse_formats(formats: str) -> list[str]:
    out = [f.strip().lower() for f in formats.split(",") if f.strip()]
    allowed = {"csv", "parquet"}
    bad = sorted(set(out) - allowed)
    if bad:
        raise ValueError(f"Unsupported format(s): {bad}. Allowed: {sorted(allowed)}")
    return out or ["csv"]




def keep_existing(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return a compact copy with only columns that exist."""
    return df[[c for c in columns if c in df.columns]].copy()


def compact_incident_for_eda(df: pd.DataFrame) -> pd.DataFrame:
    return keep_existing(df, [
        "incident_id", "incident_date", "incident_month", "incident_category_name", "incident_status_name",
        "location_id", "site_name_filled", "department_name_filled", "business_unit_name_filled",
        "country_name_filled", "region_name_filled", "is_active_record", "injury_count",
        "lost_time_any", "restricted_time_any", "fatality_any", "emergency_room_any", "inpatient_any",
        "severe_actual", "text_early_word_count", "has_location_match", "incident_date_missing",
        "incident_date_after_reference", "incident_date_before_2000", "report_lag_days"
    ])


def compact_task_for_eda(df: pd.DataFrame) -> pd.DataFrame:
    return keep_existing(df, [
        "task_id", "task_category_name", "task_status_name", "source_type_name", "location_id",
        "site_name_filled", "department_name_filled", "business_unit_name_filled", "country_name_filled",
        "region_name_filled", "task_event_month", "is_active_record", "is_open", "is_overdue",
        "is_closed", "days_open", "days_overdue"
    ])


def compact_audit_for_eda(df: pd.DataFrame) -> pd.DataFrame:
    return keep_existing(df, [
        "audit_id", "audit_category_name", "audit_type_name", "audit_status_name", "scheduled_location_id",
        "scheduled_site_name_filled", "scheduled_department_name_filled", "scheduled_business_unit_name_filled",
        "scheduled_country_name_filled", "scheduled_region_name_filled", "audit_event_month", "is_active_record",
        "is_observation", "is_inspection", "is_risk_assessment", "is_unsafe_act", "is_unsafe_condition",
        "is_safe_act", "is_safe_condition"
    ])

def read_required(input_dir: Path, key: str) -> pd.DataFrame:
    file_name = RAW_FILE_NAMES[key]
    path = input_dir / file_name
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    print(f"Reading {path}", flush=True)
    df = read_csv_safely(path)
    print(f"  shape={df.shape}", flush=True)
    return df


def main() -> None:
    args = parse_args()
    print(f"Input directory: {args.input_dir}", flush=True)
    print(f"Output directory: {args.output_dir}", flush=True)
    print(f"Skip large processed tables: {args.skip_large_processed}", flush=True)

    output_dir: Path = args.output_dir
    processed_dir = output_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    if args.reference_date:
        reference_date = pd.Timestamp(args.reference_date, tz="UTC")
    else:
        reference_date = pd.Timestamp.utcnow().normalize()
    active_only = not args.include_inactive
    formats = parse_formats(args.formats)
    raw_shapes: dict[str, tuple[int, int]] = {}

    print("Preparing lookups and location hierarchy", flush=True)
    listitem_raw = read_required(args.input_dir, "listitem")
    raw_shapes["listitem"] = listitem_raw.shape
    listitems = prepare_listitems(listitem_raw)
    del listitem_raw

    location_raw = read_required(args.input_dir, "location")
    raw_shapes["location"] = location_raw.shape
    location_hierarchy = build_location_hierarchy(location_raw, listitems)
    del location_raw
    gc.collect()

    print("Preparing injury aggregation", flush=True)
    injury_raw = read_required(args.input_dir, "injury")
    raw_shapes["injury"] = injury_raw.shape
    injury_agg = prepare_injury_agg(injury_raw)
    del injury_raw
    gc.collect()

    print("Preparing incident enrichment", flush=True)
    incident_raw = read_required(args.input_dir, "incident")
    raw_shapes["incident"] = incident_raw.shape
    incident_enriched = prepare_incidents(
        incident_raw,
        listitems=listitems,
        location_hierarchy=location_hierarchy,
        injury_agg=injury_agg,
        reference_date=reference_date,
    )
    del incident_raw
    gc.collect()

    print("Preparing all incident-injury records before pattern filters", flush=True)
    incident_injury_all_records = prepare_incident_injury_all_records(incident_enriched)

    print("Preparing pattern-learning incident records", flush=True)
    pattern_records = prepare_pattern_learning_records(incident_enriched, active_only=active_only)

    if args.skip_large_processed:
        incident_enriched = compact_incident_for_eda(incident_enriched)
        gc.collect()

    print("Preparing task enrichment", flush=True)
    task_raw = read_required(args.input_dir, "task")
    raw_shapes["task"] = task_raw.shape
    task_enriched = prepare_tasks(task_raw, listitems, location_hierarchy, reference_date=reference_date)
    del task_raw
    if args.skip_large_processed:
        task_enriched = compact_task_for_eda(task_enriched)
    gc.collect()

    print("Preparing audit enrichment", flush=True)
    audit_raw = read_required(args.input_dir, "audit")
    raw_shapes["audit"] = audit_raw.shape
    audit_enriched = prepare_audits(audit_raw, listitems, location_hierarchy)
    del audit_raw
    if args.skip_large_processed:
        audit_enriched = compact_audit_for_eda(audit_enriched)
    gc.collect()

    print("Preparing joined site/department/month features", flush=True)
    site_month_features = make_site_department_month_features(
        incident_enriched=incident_enriched,
        task_enriched=task_enriched,
        audit_enriched=audit_enriched,
        active_only=active_only,
    )

    print("Writing processed outputs", flush=True)
    write_dataframe(listitems, processed_dir / "listitem_lookup", formats)
    write_dataframe(location_hierarchy, processed_dir / "location_hierarchy", formats)
    write_dataframe(injury_agg, processed_dir / "injury_agg", formats)
    write_dataframe(incident_injury_all_records, processed_dir / "incident_injury_all_records", formats)
    write_dataframe(pattern_records, processed_dir / "pattern_learning_records", formats)
    write_dataframe(site_month_features, processed_dir / "site_department_month_features", formats)

    if not args.skip_large_processed:
        write_dataframe(incident_enriched, processed_dir / "incident_enriched", formats)
        write_dataframe(task_enriched, processed_dir / "task_enriched", formats)
        write_dataframe(audit_enriched, processed_dir / "audit_enriched", formats)

    print("Creating EDA tables, plots, and summary", flush=True)
    create_eda_outputs(
        raw_shapes=raw_shapes,
        incident_enriched=incident_enriched,
        pattern_records=pattern_records,
        injury_agg=injury_agg,
        task_enriched=task_enriched,
        audit_enriched=audit_enriched,
        location_hierarchy=location_hierarchy,
        site_month_features=site_month_features,
        output_dir=output_dir,
    )

    print("Done", flush=True)
    print(f"Pattern records: {len(pattern_records):,}", flush=True)
    print(f"Site/month feature rows: {len(site_month_features):,}", flush=True)
    print(f"EDA summary: {output_dir / 'eda' / 'eda_summary.md'}", flush=True)


if __name__ == "__main__":
    main()
