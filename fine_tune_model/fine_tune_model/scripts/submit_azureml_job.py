"""Submit project stages to Azure ML using configs/config.yaml.

Default with no args runs data preparation + light LLM labeling:
    python scripts/submit_azureml_job.py

Other useful stages:
    python scripts/submit_azureml_job.py --stage train
    python scripts/submit_azureml_job.py --stage all
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from azure.ai.ml import MLClient, command
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
import sys
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from utils import load_config  # noqa: E402
from output_sync import download_job_outputs  # noqa: E402


def _get_required(d: Dict[str, Any], key: str) -> str:
    value = d.get(key)
    if value is None or str(value).strip() == "":
        raise ValueError(f"Missing azureml.{key} in configs/config.yaml.")
    return str(value).strip()


def stage_command(stage: str) -> str:
    """Build the command run inside Azure ML.

    The scripts write to project-relative outputs/ folders. The final find/ls
    block is just diagnostic; Azure ML captures the outputs/ folder and the
    local submit script downloads it after the run completes.
    """
    commands = {
        "prepare": "python src/01_prepare_data.py --config configs/config.yaml && python src/02_light_llm_label.py --config configs/config.yaml",
        "train": "python src/03_train_qlora.py --config configs/config.yaml && python src/04_evaluate.py --config configs/config.yaml",
        "all": "python src/01_prepare_data.py --config configs/config.yaml && python src/02_light_llm_label.py --config configs/config.yaml && python src/03_train_qlora.py --config configs/config.yaml && python src/04_evaluate.py --config configs/config.yaml",
        "01": "python src/01_prepare_data.py --config configs/config.yaml",
        "02": "python src/02_light_llm_label.py --config configs/config.yaml",
        "03": "python src/03_train_qlora.py --config configs/config.yaml",
        "04": "python src/04_evaluate.py --config configs/config.yaml",
    }
    if stage not in commands:
        raise ValueError(f"Unknown stage {stage}. Use one of: {', '.join(commands)}")
    diagnostic = "echo 'Output files created:' && find outputs -maxdepth 3 -type f | sort || true"
    return commands[stage] + " && " + diagnostic


def build_job(cfg: Dict[str, Any], stage: str):
    aml = cfg.get("azureml", {})
    compute = _get_required(aml, "compute")
    experiment_name = aml.get("experiment_name", "safety_fine_tune_qwen")
    env_cfg = aml.get("environment", {})
    image = env_cfg.get("image", "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04")
    conda_file = PROJECT_ROOT / env_cfg.get("conda_file", "azureml/conda.yml")
    if not conda_file.exists():
        raise FileNotFoundError(f"Conda file not found: {conda_file}")

    input_csv = PROJECT_ROOT / str(cfg.get("paths", {}).get("input_csv", "data/safety_text_event.csv.gz"))
    if stage in {"prepare", "all", "01", "02"} and not input_csv.exists():
        raise FileNotFoundError(
            f"Input CSV not found inside the project folder: {input_csv}\n"
            "Put safety_text_event.csv.gz under fine_tune_model/data/ or update paths.input_csv."
        )

    display_stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    display_name = f"{aml.get('job_name_prefix', 'safety-ft-qwen')}-{stage}-{display_stamp}"

    return command(
        code=str(PROJECT_ROOT),
        command=stage_command(stage),
        environment=Environment(image=image, conda_file=str(conda_file)),
        compute=compute,
        experiment_name=experiment_name,
        display_name=display_name,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Submit safety fine-tuning stages to Azure ML.")
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "config.yaml"))
    ap.add_argument("--stage", default="prepare", choices=["prepare", "train", "all", "01", "02", "03", "04"])
    ap.add_argument("--no-stream", action="store_true", help="Submit the job but do not stream logs.")
    ap.add_argument("--no-download", action="store_true", help="Do not download outputs after the streamed job finishes.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    aml = cfg.get("azureml", {})
    subscription_id = _get_required(aml, "subscription_id")
    resource_group = _get_required(aml, "resource_group")
    workspace_name = _get_required(aml, "workspace_name")

    print("Azure ML workspace:")
    print(f"  subscription_id: {subscription_id}")
    print(f"  resource_group:   {resource_group}")
    print(f"  workspace_name:  {workspace_name}")
    print(f"  compute:         {aml.get('compute')}")
    print(f"  stage:           {args.stage}")
    print()

    ml_client = MLClient(DefaultAzureCredential(), subscription_id, resource_group, workspace_name)
    job = build_job(cfg, args.stage)
    returned = ml_client.jobs.create_or_update(job)

    print("Submitted Azure ML job:")
    print(f"  name:       {returned.name}")
    print(f"  studio_url: {returned.studio_url}")
    print()

    streamed_to_completion = False
    if not args.no_stream:
        print("Streaming job logs. Ctrl+C stops local streaming but does not cancel the Azure ML job.")
        try:
            ml_client.jobs.stream(returned.name)
            streamed_to_completion = True
        except KeyboardInterrupt:
            print("\nStopped local log streaming. The Azure ML job will continue running.")

    if streamed_to_completion and not args.no_download:
        print("\nJob stream finished. Downloading Azure ML outputs back to the local VS Code project...")
        download_job_outputs(
            job_name=returned.name,
            subscription_id=subscription_id,
            resource_group=resource_group,
            workspace_name=workspace_name,
            download_root=PROJECT_ROOT / "outputs" / "azureml_downloads",
            local_output_root=PROJECT_ROOT / "outputs",
            overwrite_download=True,
        )
    elif args.no_stream and not args.no_download:
        print("\nOutputs were not downloaded because --no-stream was used. After the job completes, run:")
        print(f"  python scripts/download_azureml_job_outputs.py --job-name {returned.name}")


if __name__ == "__main__":
    main()
