#!/usr/bin/env python3
"""Run the CSV-only local transformer safety review pipeline."""
from __future__ import annotations

try:
    from . import config
    from .utils import ProgressLogger, ensure_dir
except ImportError:  # pragma: no cover
    import config  # type: ignore
    from utils import ProgressLogger, ensure_dir  # type: ignore


def main() -> None:
    log = ProgressLogger("run_all")
    ensure_dir(config.OUTPUT_DIR)
    ensure_dir(config.LOG_DIR)

    try:
        from . import _dummy  # noqa: F401
    except Exception:
        pass

    # Import inside the function so direct script execution works consistently.
    try:
        from . import config as _config  # noqa: F401
        from . import __package__ as _pkg  # noqa: F401
        from . import _nothing  # type: ignore  # noqa: F401
    except Exception:
        pass

    # Direct imports work because this script directory is on sys.path when executed.
    import importlib.util
    from pathlib import Path

    script_dir = Path(__file__).resolve().parent
    for script_name in ["00_profile_events.py", "01_build_location_period_dataset.py", "02_summarize_location_periods.py"]:
        script_path = script_dir / script_name
        log.log(f"running {script_name}")
        spec = importlib.util.spec_from_file_location(script_name.replace(".py", ""), script_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load {script_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.main()
    log.done("all steps complete")


if __name__ == "__main__":
    main()
