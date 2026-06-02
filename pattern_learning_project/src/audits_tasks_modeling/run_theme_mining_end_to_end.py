#!/usr/bin/env python3
"""Run source-aware safety theme mining end-to-end.

By default this uses the existing unified dataset from Step 00. To rebuild Step
00 from raw source CSVs, set RUN_STEP_00_IN_END_TO_END=True in config.py.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

try:
    import config as cfg
    from theme_utils import ProgressLogger
except ImportError:  # pragma: no cover
    from . import config as cfg
    from .theme_utils import ProgressLogger


SCRIPT_ORDER = [
    "01_prepare_theme_text.py",
    "02_generate_theme_embeddings.py",
    "03_cluster_by_family.py",
    "04_label_theme_clusters.py",
    "05_build_location_theme_period_profiles.py",
    "06_build_cross_family_theme_links.py",
]


def _run_script(script_path: Path) -> None:
    spec = importlib.util.spec_from_file_location(script_path.stem, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "main"):
        raise RuntimeError(f"Script {script_path} does not define main()")
    module.main()


def main() -> None:
    log = ProgressLogger("run_theme_mining_end_to_end")
    script_dir = Path(__file__).resolve().parent

    if bool(getattr(cfg, "RUN_STEP_00_IN_END_TO_END", False)):
        log.log("RUN_STEP_00_IN_END_TO_END=True; running 00_build_unified_text_events.py")
        _run_script(script_dir / "00_build_unified_text_events.py")
    else:
        log.log(f"using existing unified event file: {cfg.UNIFIED_EVENTS_FILE}")
        if not cfg.UNIFIED_EVENTS_FILE.exists():
            raise FileNotFoundError(
                f"Unified event file not found: {cfg.UNIFIED_EVENTS_FILE}. "
                "Either run 00_build_unified_text_events.py first or set RUN_STEP_00_IN_END_TO_END=True."
            )

    for script_name in SCRIPT_ORDER:
        if script_name == "06_build_cross_family_theme_links.py" and not bool(getattr(cfg, "ENABLE_CROSS_FAMILY_LINKS", True)):
            log.log("skipping 06_build_cross_family_theme_links.py because ENABLE_CROSS_FAMILY_LINKS=False")
            continue
        log.log(f"running {script_name}")
        _run_script(script_dir / script_name)
    log.done("all theme-mining steps complete")


if __name__ == "__main__":
    main()
