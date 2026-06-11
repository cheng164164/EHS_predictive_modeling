"""Runtime helpers for CPU-thread configuration and environment diagnostics."""
from __future__ import annotations

import os
import platform
from typing import Any

from .config import Settings


def resolve_cpu_threads(settings: Settings) -> int:
    available = os.cpu_count() or 1
    requested = int(getattr(settings, "cpu_thread_count", 0) or 0)
    if bool(getattr(settings, "use_all_available_cpus", True)) or requested <= 0:
        return int(available)
    return max(1, min(int(requested), int(available)))


def configure_cpu_runtime(settings: Settings, include_faiss: bool = False) -> dict[str, Any]:
    """Configure common CPU parallelism knobs and return diagnostics.

    This should be called before loading PyTorch/SentenceTransformer/FAISS models.
    Environment variables are set with defaults only if they are not already set by
    Azure ML job submission.
    """
    threads = resolve_cpu_threads(settings)
    interop = max(1, int(getattr(settings, "torch_interop_thread_count", 2) or 1))
    interop = min(interop, max(1, threads))

    for name in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ.setdefault(name, str(threads))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    diagnostics: dict[str, Any] = {
        "platform": platform.platform(),
        "os_cpu_count": os.cpu_count(),
        "configured_cpu_threads": threads,
        "configured_torch_interop_threads": interop,
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
        "TOKENIZERS_PARALLELISM": os.environ.get("TOKENIZERS_PARALLELISM"),
    }

    try:
        import torch

        torch.set_num_threads(threads)
        try:
            torch.set_num_interop_threads(interop)
        except RuntimeError:
            # PyTorch can raise if interop threads were already initialized.
            pass
        diagnostics.update(
            {
                "torch_version": torch.__version__,
                "torch_num_threads": torch.get_num_threads(),
                "torch_num_interop_threads": torch.get_num_interop_threads(),
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            }
        )
        if bool(getattr(settings, "force_cpu_embedding", True)):
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
            diagnostics["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    except Exception as exc:  # pragma: no cover - diagnostic only
        diagnostics["torch_configuration_error"] = repr(exc)

    if include_faiss:
        try:
            import faiss

            faiss.omp_set_num_threads(threads)
            diagnostics["faiss_max_threads"] = int(faiss.omp_get_max_threads())
        except Exception as exc:  # pragma: no cover - diagnostic only
            diagnostics["faiss_configuration_error"] = repr(exc)

    print("[Runtime] CPU/thread diagnostics:", diagnostics, flush=True)
    return diagnostics
