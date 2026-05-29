#!/usr/bin/env python
"""Run steps 0 through 8 using settings from config.py.

No arguments are required:

    python run_end_to_end.py

Optional arguments are available only for convenience:

    python run_end_to_end.py --start-step 3 --end-step 8
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import config as cfg


STEPS = [
    (0, "build unified text event table", "00_build_unified_text_events.py"),
    (1, "generate text embeddings", "01_generate_embeddings.py"),
    (2, "extract safety tags", "02_extract_safety_tags.py"),
    (3, "discover risk themes", "03_discover_risk_themes.py"),
    (4, "assign risk themes", "04_assign_risk_themes.py"),
    (5, "build risk-state dataset", "05_build_risk_state_dataset.py"),
    (6, "train risk-burden model", "06_train_risk_burden_model.py"),
    (7, "train elevated-risk classifier", "07_train_elevated_risk_classifier.py"),
    (8, "explain risk drivers", "08_explain_risk_drivers.py"),
]


REQUIRED_INPUTS = [
    cfg.AUDIT_VIEW_PATH,
    cfg.INCIDENT_VIEW_PATH,
    cfg.INCIDENTINJURY_VIEW_PATH,
    cfg.LISTITEM_VIEW_PATH,
    cfg.LOCATION_VIEW_PATH,
    cfg.TASK_VIEW_PATH,
]


def check_inputs() -> None:
    missing = [p for p in REQUIRED_INPUTS if not Path(p).exists()]
    if missing:
        formatted = "\n".join(f"  - {p}" for p in missing)
        raise FileNotFoundError(
            "Missing required input CSV files. Update DATA_DIR in config.py or set SAFETY_DATA_DIR.\n" + formatted
        )


def run_step(step_no: int, label: str, script_name: str) -> None:
    script_path = cfg.SRC_DIR / script_name
    print("\n" + "=" * 88)
    print(f"Step {step_no}: {label}")
    print(f"Script: {script_path}")
    print("=" * 88)
    started = time.time()
    subprocess.run([sys.executable, str(script_path)], cwd=str(cfg.SRC_DIR), check=True)
    elapsed = time.time() - started
    print(f"Finished step {step_no} in {elapsed / 60:.2f} minutes.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run audits/tasks modeling steps 0-8 from config.py.")
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--end-step", type=int, default=8)
    args = parser.parse_args()

    cfg.print_config_summary()
    check_inputs()
    for path in cfg.all_step_dirs():
        path.mkdir(parents=True, exist_ok=True)

    selected = [s for s in STEPS if args.start_step <= s[0] <= args.end_step]
    if not selected:
        raise ValueError("No steps selected. Use --start-step and --end-step between 0 and 8.")

    for step_no, label, script_name in selected:
        run_step(step_no, label, script_name)

    print("\nPipeline complete.")
    print(f"Outputs saved under: {cfg.OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
