"""Artifact I/O helpers for local paths and Azure ML datastore URIs.

Prediction/recommendation scripts need to load large FAISS/BM25/metadata files.
Those files may live either in the local project output folder or only in an
Azure ML datastore path such as:

    azureml://subscriptions/.../resourcegroups/.../workspaces/.../datastores/workspaceblobstore/paths/...

This module keeps runtime code independent of the storage location.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


def is_remote_uri(path: object) -> bool:
    text = str(path)
    return text.startswith("azureml://") or text.startswith("abfss://") or text.startswith("wasbs://") or text.startswith("https://")


def artifact_join(base: str | Path, *parts: object) -> str | Path:
    if is_remote_uri(base):
        text = str(base).rstrip("/")
        clean = [str(p).strip("/") for p in parts if str(p).strip("/")]
        return "/".join([text, *clean]) if clean else text
    out = Path(base)
    for part in parts:
        out = out / str(part)
    return out


def _require_fsspec():
    try:
        import fsspec  # noqa: F401
        # Register Azure ML datastore filesystem implementation when installed.
        try:
            import azureml.fsspec  # noqa: F401
        except Exception:
            pass
        return fsspec
    except ImportError as exc:  # pragma: no cover - dependency guidance only
        raise ImportError(
            "Reading artifacts from azureml:// datastore URIs requires fsspec and azureml-fsspec.\n"
            "Install them in the environment where you run predict_single_event.py / 02_run_mvp_recommendations.py:\n"
            "    pip install azureml-fsspec fsspec\n"
        ) from exc


def artifact_exists(path: str | Path) -> bool:
    if is_remote_uri(path):
        fsspec = _require_fsspec()
        try:
            fs, inner = fsspec.core.url_to_fs(str(path))
            return bool(fs.exists(inner))
        except Exception:
            try:
                with fsspec.open(str(path), "rb"):
                    return True
            except Exception:
                return False
    return Path(path).exists()


def read_bytes(path: str | Path) -> bytes:
    if is_remote_uri(path):
        fsspec = _require_fsspec()
        with fsspec.open(str(path), "rb") as f:
            return f.read()
    return Path(path).read_bytes()


def open_binary(path: str | Path):
    if is_remote_uri(path):
        fsspec = _require_fsspec()
        return fsspec.open(str(path), "rb")
    return open(Path(path), "rb")


def read_json(path: str | Path) -> dict[str, Any]:
    data = read_bytes(path)
    return json.loads(data.decode("utf-8"))


def read_pickle(path: str | Path):
    if is_remote_uri(path):
        # Use BytesIO so pandas can seek reliably even if the remote file object
        # does not fully support random access.
        return pd.read_pickle(io.BytesIO(read_bytes(path)))
    return pd.read_pickle(Path(path))


def load_numpy(path: str | Path, allow_pickle: bool = False):
    if is_remote_uri(path):
        return np.load(io.BytesIO(read_bytes(path)), allow_pickle=allow_pickle)
    return np.load(Path(path), allow_pickle=allow_pickle)


def load_joblib(path: str | Path):
    if is_remote_uri(path):
        return joblib.load(io.BytesIO(read_bytes(path)))
    return joblib.load(Path(path))


def load_faiss_index(path: str | Path):
    import faiss

    if is_remote_uri(path):
        data = read_bytes(path)
        buffer = np.frombuffer(data, dtype="uint8")
        return faiss.deserialize_index(buffer)
    return faiss.read_index(str(Path(path)))


def describe_artifact_root(root: str | Path) -> str:
    return str(root)
