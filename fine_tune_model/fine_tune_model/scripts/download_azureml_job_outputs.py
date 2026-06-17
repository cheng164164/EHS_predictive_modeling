"""Download Azure ML job outputs into the local VS Code project folder.

Examples:
    python scripts/download_azureml_job_outputs.py --job-name safety-ft-qwen-prepare-20260616_021104
    python scripts/download_azureml_job_outputs.py --name-contains train
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from utils import load_config  # noqa: E402
from output_sync import download_job_outputs, latest_job_name_from_cli  # noqa: E402


def required(d: dict, key: str) -> str:
    value = d.get(key)
    if value is None or str(value).strip() == "":
        raise ValueError(f"Missing azureml.{key} in configs/config.yaml")
    return str(value).strip()


def main() -> None:
    ap = argparse.ArgumentParser(description="Download Azure ML job outputs to local fine_tune_model/outputs.")
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "config.yaml"))
    ap.add_argument("--job-name", default=None, help="Azure ML job name. If omitted, the latest job is used.")
    ap.add_argument("--name-contains", default=None, help="Optional filter when --job-name is omitted, such as prepare/train.")
    ap.add_argument("--download-root", default=str(PROJECT_ROOT / "outputs" / "azureml_downloads"))
    ap.add_argument("--local-output-root", default=str(PROJECT_ROOT / "outputs"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    aml = cfg.get("azureml", {})
    subscription_id = required(aml, "subscription_id")
    resource_group = required(aml, "resource_group")
    workspace_name = required(aml, "workspace_name")

    job_name = args.job_name or latest_job_name_from_cli(
        subscription_id=subscription_id,
        resource_group=resource_group,
        workspace_name=workspace_name,
        name_contains=args.name_contains,
    )
    print(f"Downloading Azure ML job: {job_name}")

    download_job_outputs(
        job_name=job_name,
        subscription_id=subscription_id,
        resource_group=resource_group,
        workspace_name=workspace_name,
        download_root=Path(args.download_root),
        local_output_root=Path(args.local_output_root),
        overwrite_download=True,
    )


if __name__ == "__main__":
    main()
