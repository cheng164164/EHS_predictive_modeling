"""Download/cache runtime artifacts from Azure ML datastore to local workspace.

The build jobs keep large artifacts in workspaceblobstore. For interactive
agent testing in VS Code, repeated reads through azureml:// can be slow. This
module downloads only the runtime artifacts needed for prediction/recommendation
into the local project output folder:

    outputs/safety retrieval agent/indexes/faiss_indexes/
    outputs/safety retrieval agent/indexes/bm25_indexes/
    outputs/safety retrieval agent/models/
    outputs/safety retrieval agent/data/

Embedding chunk files and event_embeddings.npy are intentionally not downloaded
by default because the agent does not need them at query time.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from .artifact_io import is_remote_uri
from .config import Settings
from .utils import ensure_dir


def _require_fsspec():
    try:
        import fsspec
        try:
            import azureml.fsspec  # noqa: F401
        except Exception:
            pass
        return fsspec
    except ImportError as exc:  # pragma: no cover - dependency guidance only
        raise ImportError(
            "Downloading from azureml:// datastore URIs requires fsspec and azureml-fsspec.\n"
            "Install in the VS Code/Azure ML notebook environment with:\n"
            "    pip install azureml-fsspec fsspec\n"
        ) from exc


def _walk_remote_files(fs, path: str) -> list[dict[str, Any]]:
    """Return file entries under a remote path using fs.ls recursively."""
    out: list[dict[str, Any]] = []
    try:
        entries = fs.ls(path, detail=True)
    except FileNotFoundError:
        return out
    except Exception:
        # Some filesystems do not raise FileNotFoundError exactly.
        if not fs.exists(path):
            return out
        entries = fs.ls(path, detail=True)

    for entry in entries:
        if isinstance(entry, str):
            name = entry
            info = {"name": name, "type": "file" if not name.endswith("/") else "directory"}
        else:
            info = dict(entry)
            name = str(info.get("name") or info.get("Key") or info.get("path") or "")
        if not name:
            continue
        typ = str(info.get("type") or info.get("kind") or "").lower()
        is_dir = typ in {"directory", "dir"} or name.endswith("/")
        if is_dir:
            out.extend(_walk_remote_files(fs, name.rstrip("/")))
        else:
            out.append(info)
    return out


def _copy_remote_tree(
    fs,
    remote_root: str,
    remote_subdir: str,
    local_dest: Path,
    *,
    overwrite: bool = True,
    skip_zero_byte_markers: bool = True,
) -> dict[str, Any]:
    """Copy one subfolder from the remote artifact root to a local folder."""
    remote_dir = "/".join([remote_root.rstrip("/"), remote_subdir.strip("/")])
    local_dest = ensure_dir(local_dest)
    started = time.time()
    entries = _walk_remote_files(fs, remote_dir)
    copied = 0
    skipped = 0
    failed = 0
    bytes_copied = 0
    errors: list[dict[str, str]] = []

    for info in entries:
        remote_file = str(info.get("name") or info.get("Key") or info.get("path") or "")
        if not remote_file:
            skipped += 1
            continue
        # Skip folder-marker blobs such as .../data with size 0.
        size = int(info.get("size") or info.get("ContentLength") or 0)
        rel = remote_file[len(remote_dir.rstrip("/")):].lstrip("/") if remote_file.startswith(remote_dir.rstrip("/")) else Path(remote_file).name
        if not rel or rel in {".", ".."}:
            skipped += 1
            continue
        if skip_zero_byte_markers and size == 0 and "/" not in rel and "." not in Path(rel).name:
            skipped += 1
            continue
        local_file = local_dest / rel
        if local_file.exists() and not overwrite:
            skipped += 1
            continue
        try:
            ensure_dir(local_file.parent)
            with fs.open(remote_file, "rb") as src, open(local_file, "wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
            copied += 1
            try:
                bytes_copied += local_file.stat().st_size
            except OSError:
                pass
        except Exception as exc:  # pragma: no cover - depends on remote IO
            failed += 1
            errors.append({"remote_file": remote_file, "local_file": str(local_file), "error": repr(exc)})

    return {
        "remote_subdir": remote_subdir,
        "remote_dir": remote_dir,
        "local_dest": str(local_dest),
        "remote_file_count": len(entries),
        "copied_file_count": copied,
        "skipped_file_count": skipped,
        "failed_file_count": failed,
        "bytes_copied": bytes_copied,
        "elapsed_seconds": round(time.time() - started, 3),
        "errors": errors[:20],
    }


def _copy_local_tree(src: Path, dest: Path, *, overwrite: bool = True) -> dict[str, Any]:
    started = time.time()
    src = Path(src)
    dest = ensure_dir(Path(dest))
    copied = skipped = failed = bytes_copied = 0
    errors: list[dict[str, str]] = []
    if not src.exists():
        return {"source": str(src), "local_dest": str(dest), "copied_file_count": 0, "missing_source": True}
    for file in src.rglob("*"):
        if not file.is_file():
            continue
        rel = file.relative_to(src)
        out = dest / rel
        if out.exists() and not overwrite:
            skipped += 1
            continue
        try:
            ensure_dir(out.parent)
            shutil.copy2(file, out)
            copied += 1
            bytes_copied += out.stat().st_size
        except Exception as exc:
            failed += 1
            errors.append({"source_file": str(file), "local_file": str(out), "error": repr(exc)})
    return {
        "source": str(src),
        "local_dest": str(dest),
        "copied_file_count": copied,
        "skipped_file_count": skipped,
        "failed_file_count": failed,
        "bytes_copied": bytes_copied,
        "elapsed_seconds": round(time.time() - started, 3),
        "errors": errors[:20],
    }


def local_runtime_artifact_dirs(settings: Settings) -> dict[str, Path]:
    """Return local runtime-cache destinations used by the agent."""
    return {
        "data": settings.output_dir / "data",
        "faiss_indexes": settings.output_dir / "indexes" / "faiss_indexes",
        "bm25_indexes": settings.output_dir / "indexes" / "bm25_indexes",
        "models": settings.output_dir / "models",
    }


def download_runtime_artifacts(settings: Settings) -> dict[str, Any]:
    """Download runtime artifacts from artifact_azureml_uri to local workspace.

    Downloads data/, faiss_indexes/, bm25_indexes/, and models/. It does not
    download embeddings/chunks or event_embeddings.npy by default.
    """
    remote_root = str(settings.artifact_azureml_uri).rstrip("/")
    overwrite = bool(getattr(settings, "local_runtime_cache_overwrite", True))
    started = time.time()
    local_dirs = local_runtime_artifact_dirs(settings)
    for path in local_dirs.values():
        ensure_dir(path)

    print(f"[Download] Source artifact root: {remote_root}", flush=True)
    print(f"[Download] Local output root: {settings.output_dir}", flush=True)
    print(f"[Download] Local FAISS indexes: {local_dirs['faiss_indexes']}", flush=True)
    print(f"[Download] Local BM25 indexes: {local_dirs['bm25_indexes']}", flush=True)
    print(f"[Download] Local models: {local_dirs['models']}", flush=True)
    print(f"[Download] Local data: {local_dirs['data']}", flush=True)

    results: list[dict[str, Any]] = []
    if is_remote_uri(remote_root):
        fsspec = _require_fsspec()
        fs, inner_root = fsspec.core.url_to_fs(remote_root)
        inner_root = inner_root.rstrip("/")
        mappings = [
            ("data", local_dirs["data"]),
            ("faiss_indexes", local_dirs["faiss_indexes"]),
            ("bm25_indexes", local_dirs["bm25_indexes"]),
            ("models", local_dirs["models"]),
        ]
        for remote_subdir, local_dest in mappings:
            print(f"[Download] Copying {remote_subdir}/ -> {local_dest}", flush=True)
            results.append(_copy_remote_tree(fs, inner_root, remote_subdir, local_dest, overwrite=overwrite))
    else:
        local_root = Path(remote_root)
        mappings = [
            (local_root / "data", local_dirs["data"]),
            (local_root / "faiss_indexes", local_dirs["faiss_indexes"]),
            (local_root / "bm25_indexes", local_dirs["bm25_indexes"]),
            (local_root / "models", local_dirs["models"]),
        ]
        for source, local_dest in mappings:
            print(f"[Download] Copying {source} -> {local_dest}", flush=True)
            results.append(_copy_local_tree(source, local_dest, overwrite=overwrite))

    summary = {
        "source_artifact_root": remote_root,
        "local_output_root": str(settings.output_dir),
        "local_runtime_dirs": {k: str(v) for k, v in local_dirs.items()},
        "artifact_read_mode_recommendation": "local",
        "elapsed_seconds": round(time.time() - started, 3),
        "copy_results": results,
        "note": (
            "Runtime artifacts were downloaded for faster interactive queries. "
            "Set SAFETY_RETRIEVAL_ARTIFACT_READ_MODE=local, or leave mode=auto if configured, "
            "so predict_single_event.py and 02_run_mvp_recommendations.py use these local files."
        ),
    }
    summary_path = ensure_dir(settings.output_dir / "logs") / "runtime_artifact_download_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[Download] Saved summary: {summary_path}", flush=True)
    return summary
