"""Utilities for syncing Azure ML job outputs back into the local project folder.

This module is intentionally CLI based because `az ml job download` is the most
stable way to retrieve the same files shown in Azure ML Studio's Outputs + logs.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_OUTPUT_ROOT = PROJECT_ROOT / "outputs"
DOWNLOAD_ROOT = LOCAL_OUTPUT_ROOT / "azureml_downloads"

OUTPUT_SUBDIRS = ("prepared", "model", "eval")
LOG_SUBDIRS = ("user_logs", "system_logs")


def run_cmd(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def remove_if_exists(path: Path) -> None:
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def copy_tree_merge(src: Path, dst: Path) -> int:
    """Copy all files from src to dst, overwriting same-name files.

    Returns the number of copied files.
    """
    if not src.exists() or not src.is_dir():
        return 0
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for item in src.rglob("*"):
        rel = item.relative_to(src)
        target = dst / rel
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)
            n += 1
    return n


def find_output_dirs(download_dir: Path, subdir_name: str) -> list[Path]:
    """Find folders such as outputs/prepared, outputs/model, outputs/eval.

    Azure ML download layouts can differ by SDK/CLI version. This searches for
    directories with the expected name whose parent is an outputs-like folder,
    then falls back to any directory with the expected name.
    """
    candidates: list[Path] = []
    for p in download_dir.rglob(subdir_name):
        if not p.is_dir():
            continue
        parent_names = {part.lower() for part in p.parts}
        if p.parent.name.lower() == "outputs" or "outputs" in parent_names:
            candidates.append(p)
    if candidates:
        return sorted(set(candidates), key=lambda x: len(str(x)))

    # Fallback: avoid copying from the local destination tree if user points
    # download_dir at PROJECT_ROOT by mistake.
    fallback = [p for p in download_dir.rglob(subdir_name) if p.is_dir() and LOCAL_OUTPUT_ROOT not in p.parents]
    return sorted(set(fallback), key=lambda x: len(str(x)))


def sync_downloaded_outputs(download_dir: Path, local_output_root: Path = LOCAL_OUTPUT_ROOT) -> dict[str, int]:
    """Copy prepared/model/eval and logs from downloaded Azure ML job files."""
    local_output_root.mkdir(parents=True, exist_ok=True)
    copied: dict[str, int] = {}

    for subdir in OUTPUT_SUBDIRS:
        n_total = 0
        dirs = find_output_dirs(download_dir, subdir)
        if not dirs:
            copied[subdir] = 0
            continue
        for src in dirs:
            n_total += copy_tree_merge(src, local_output_root / subdir)
            print(f"Copied {subdir}: {src} -> {local_output_root / subdir}")
        copied[subdir] = n_total

    # Keep logs under outputs/job_logs/<job_download_folder>/...
    logs_root = local_output_root / "job_logs" / download_dir.name
    for log_dir_name in LOG_SUBDIRS:
        for src in [p for p in download_dir.rglob(log_dir_name) if p.is_dir()]:
            dst = logs_root / log_dir_name
            copied[f"logs_{log_dir_name}"] = copied.get(f"logs_{log_dir_name}", 0) + copy_tree_merge(src, dst)
            print(f"Copied logs: {src} -> {dst}")

    return copied


def download_job_outputs(
    *,
    job_name: str,
    subscription_id: str,
    resource_group: str,
    workspace_name: str,
    download_root: Path = DOWNLOAD_ROOT,
    local_output_root: Path = LOCAL_OUTPUT_ROOT,
    overwrite_download: bool = True,
) -> Path:
    """Download an Azure ML job and sync outputs into the local project."""
    download_root.mkdir(parents=True, exist_ok=True)
    download_dir = download_root / job_name
    if overwrite_download:
        remove_if_exists(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "az", "ml", "job", "download",
        "--name", job_name,
        "--resource-group", resource_group,
        "--workspace-name", workspace_name,
        "--subscription", subscription_id,
        "--download-path", str(download_dir),
    ]
    run_cmd(cmd)

    copied = sync_downloaded_outputs(download_dir, local_output_root)
    print("\nSync summary:")
    for k, v in copied.items():
        print(f"  {k}: {v} files")
    print(f"\nFull Azure ML job download kept at: {download_dir}")
    print(f"Local project outputs synced under: {local_output_root}")
    return download_dir


def latest_job_name_from_cli(subscription_id: str, resource_group: str, workspace_name: str, name_contains: Optional[str] = None) -> str:
    """Return latest job name using az ml job list.

    This is a lightweight fallback for manual downloads when job name is not given.
    """
    import json

    query = "[].{name:name,creation_context:creation_context,display_name:display_name,status:status}"
    cmd = [
        "az", "ml", "job", "list",
        "--resource-group", resource_group,
        "--workspace-name", workspace_name,
        "--subscription", subscription_id,
        "--query", query,
        "-o", "json",
    ]
    raw = subprocess.check_output(cmd, text=True)
    jobs = json.loads(raw)
    if name_contains:
        jobs = [j for j in jobs if name_contains.lower() in str(j.get("name", "") + " " + j.get("display_name", "")).lower()]
    if not jobs:
        raise RuntimeError("No Azure ML jobs found for the requested filter.")

    # CLI usually returns newest first, but sort defensively by creation time when available.
    def created(j):
        cc = j.get("creation_context") or {}
        return str(cc.get("created_at") or cc.get("creation_time") or "")

    jobs = sorted(jobs, key=created, reverse=True)
    return jobs[0]["name"]
