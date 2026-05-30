from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
import platform
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Allow running as a direct script import path.
try:
    from . import config
    from .utils import SUMMARY_FIELDS, compact_text, date_list, keyword_themes, ensure_dir
except ImportError:  # pragma: no cover
    import config  # type: ignore
    from utils import SUMMARY_FIELDS, compact_text, date_list, keyword_themes, ensure_dir  # type: ignore


FIELD_CONFIG = {
    "unsafe_conditions_summary": {
        "label": "unsafe conditions and unsafe acts",
        "event_types": ["audit_unsafe_condition", "audit_unsafe_act", "hazard_identification", "near_miss"],
        "empty": "No unsafe condition, unsafe act, hazard, or near-miss evidence was sampled for this location-period.",
    },
    "serious_injury_summary": {
        "label": "serious injuries first",
        "event_types": ["serious_injury"],
        "empty": "No serious injury records were sampled for this location-period.",
    },
    "normal_injury_summary": {
        "label": "normal injuries",
        "event_types": ["normal_injury"],
        "empty": "No normal injury records were sampled for this location-period.",
    },
    "near_miss_summary": {
        "label": "near misses",
        "event_types": ["near_miss"],
        "empty": "No near-miss records were sampled for this location-period.",
    },
    "hazards_summary": {
        "label": "hazard identifications",
        "event_types": ["hazard_identification"],
        "empty": "No hazard-identification records were sampled for this location-period.",
    },
    "audits_summary": {
        "label": "audits, observations, inspections, and unsafe condition findings",
        "event_types": ["audit_unsafe_condition", "audit_unsafe_act", "audit_other"],
        "empty": "No audit or observation records were sampled for this location-period.",
    },
    "actions_summary": {
        "label": "corrective actions, open tasks, overdue tasks, and completed tasks",
        "event_types": ["task_overdue", "task_open", "task_completed_or_closed", "task_other"],
        "empty": "No task or corrective-action records were sampled for this location-period.",
    },
}


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _version(package_name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(package_name)
    except Exception:
        return None


def _module_found(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False


def collect_transformer_environment(load_torch: bool = True) -> Dict[str, Any]:
    """Collect lightweight diagnostics without loading the transformer model."""
    report: Dict[str, Any] = {
        "created_at": _now_iso(),
        "python_executable": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "current_working_directory": os.getcwd(),
        "config_file": str(Path(config.__file__).resolve()) if hasattr(config, "__file__") else "unknown",
        "project_root": str(getattr(config, "PROJECT_ROOT", "")),
        "input_file": str(getattr(config, "INPUT_FILE", "")),
        "input_file_exists": bool(Path(getattr(config, "INPUT_FILE", "")).exists()),
        "output_dir": str(getattr(config, "OUTPUT_DIR", "")),
        "summarizer_backend": getattr(config, "SUMMARIZER_BACKEND", None),
        "allow_extractive_fallback": getattr(config, "ALLOW_EXTRACTIVE_FALLBACK", None),
        "transformer_model_name": getattr(config, "TRANSFORMER_MODEL_NAME", None),
        "transformer_task": getattr(config, "TRANSFORMER_TASK", None),
        "transformer_device": getattr(config, "TRANSFORMER_DEVICE", None),
        "transformer_local_files_only": getattr(config, "TRANSFORMER_LOCAL_FILES_ONLY", None),
        "hf_home": os.environ.get("HF_HOME"),
        "transformers_cache": os.environ.get("TRANSFORMERS_CACHE"),
        "huggingface_hub_cache": os.environ.get("HUGGINGFACE_HUB_CACHE"),
        "http_proxy_set": bool(os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")),
        "https_proxy_set": bool(os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")),
        "modules_found": {
            "transformers": _module_found("transformers"),
            "torch": _module_found("torch"),
            "huggingface_hub": _module_found("huggingface_hub"),
            "tokenizers": _module_found("tokenizers"),
            "safetensors": _module_found("safetensors"),
            "sentencepiece": _module_found("sentencepiece"),
            "accelerate": _module_found("accelerate"),
        },
        "package_versions": {
            "transformers": _version("transformers"),
            "torch": _version("torch"),
            "huggingface_hub": _version("huggingface-hub"),
            "tokenizers": _version("tokenizers"),
            "safetensors": _version("safetensors"),
            "sentencepiece": _version("sentencepiece"),
            "accelerate": _version("accelerate"),
        },
    }

    if load_torch and report["modules_found"].get("torch"):
        try:
            import torch

            report["torch"] = {
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
                "cuda_device_name_0": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            }
        except Exception as exc:
            report["torch_error"] = repr(exc)

    return report


def write_transformer_diagnostics(report: Dict[str, Any], filename: str = "transformer_diagnostics.json") -> Path:
    ensure_dir(config.LOG_DIR)
    path = Path(config.LOG_DIR) / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    return path


def print_transformer_diagnostics(report: Dict[str, Any]) -> None:
    print("----- transformer environment diagnostics -----", flush=True)
    print(f"python: {report.get('python_executable')}", flush=True)
    print(f"config: {report.get('config_file')}", flush=True)
    print(f"input_file_exists: {report.get('input_file_exists')} -> {report.get('input_file')}", flush=True)
    print(f"backend: {report.get('summarizer_backend')}", flush=True)
    print(f"model: {report.get('transformer_model_name')}", flush=True)
    print(f"device: {report.get('transformer_device')}", flush=True)
    print(f"local_files_only: {report.get('transformer_local_files_only')}", flush=True)
    print(f"modules_found: {report.get('modules_found')}", flush=True)
    print(f"package_versions: {report.get('package_versions')}", flush=True)
    if "torch" in report:
        print(f"torch: {report.get('torch')}", flush=True)
    print("-----------------------------------------------", flush=True)


class BaseSummarizer:
    model_used = "base"

    def summarize(self, fact: pd.Series, group_examples: pd.DataFrame) -> Dict[str, str]:
        raise NotImplementedError


class ExtractiveSummarizer(BaseSummarizer):
    """Conservative no-model fallback.

    This is useful for testing and for locked-down machines. It does not infer
    causation; it simply preserves representative evidence in compact form.
    """

    model_used = "fallback_extractive_no_llm"

    def summarize(self, fact: pd.Series, group_examples: pd.DataFrame) -> Dict[str, str]:
        all_text = "\n".join(group_examples.get("event_detail", pd.Series(dtype=str)).astype(str).tolist())
        out: Dict[str, str] = {}
        for field, spec in FIELD_CONFIG.items():
            subset = group_examples[group_examples["review_event_type"].isin(spec["event_types"])] if not group_examples.empty else pd.DataFrame()
            if subset.empty:
                out[field] = spec["empty"]
            else:
                details = subset.get("event_detail", pd.Series(dtype=str)).astype(str).head(6).tolist()
                out[field] = compact_text(" ; ".join(details), 1600)
        out["recurring_themes"] = keyword_themes(all_text) or "No dominant terms found in sampled evidence."
        out["dates_to_review"] = date_list(all_text)
        out["data_gaps_or_cautions"] = (
            "Extractive fallback summary only. Counts and sampled evidence are deterministic, "
            "but summaries should be reviewed against source rows before drawing conclusions."
        )
        out["summary_model_used"] = self.model_used
        return out


class LocalTransformerSummarizer(BaseSummarizer):
    """Free local Hugging Face transformer summarizer.

    This implementation intentionally avoids transformers.pipeline for generation.
    Some package/version combinations accidentally forward tokenizer_kwargs into
    model.generate(), which raises:
        ValueError: model_kwargs are not used by the model: ['tokenizer_kwargs']

    Loading tokenizer + model directly and calling model.generate() gives us more
    control and avoids that pipeline keyword-routing issue.
    """

    def __init__(self):
        self.model_name = config.TRANSFORMER_MODEL_NAME
        self.model_used = f"local_transformers_manual_generate:{self.model_name}"

        print(f"Loading local transformer summarizer with manual generate: {self.model_name}", flush=True)
        print("This may take time on the first run if Hugging Face model files must be downloaded.", flush=True)

        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except Exception as exc:
            raise RuntimeError(
                "Required packages for local transformer summarization are missing or cannot be imported. "
                "Install requirements.txt, activate the correct Python environment, or set "
                "SUMMARIZER_BACKEND='extractive' in config.py."
            ) from exc

        self.torch = torch

        try:
            local_only = bool(getattr(config, "TRANSFORMER_LOCAL_FILES_ONLY", False))
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                local_files_only=local_only,
            )
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                local_files_only=local_only,
            )

            requested_device = int(getattr(config, "TRANSFORMER_DEVICE", -1))
            if requested_device >= 0 and torch.cuda.is_available():
                self.device = torch.device(f"cuda:{requested_device}")
            else:
                self.device = torch.device("cpu")

            self.model.to(self.device)
            self.model.eval()

            raw_max_len = getattr(self.tokenizer, "model_max_length", 1024)
            # Some tokenizers use huge sentinel values. Keep summarization inputs bounded.
            if not isinstance(raw_max_len, int) or raw_max_len > 4096:
                raw_max_len = 1024
            self.max_input_tokens = min(int(raw_max_len), 1024)

            print(
                f"Successfully loaded transformer model: {self.model_used}; "
                f"device={self.device}; max_input_tokens={self.max_input_tokens}",
                flush=True,
            )
        except Exception as exc:
            raise RuntimeError(
                "Transformer model/tokenizer initialization failed. Common causes: missing torch, "
                "model download blocked, invalid model name, no cached model files when "
                "TRANSFORMER_LOCAL_FILES_ONLY=True, package version mismatch, or insufficient memory."
            ) from exc

    def _prepare_input(self, fact: pd.Series, field: str, subset: pd.DataFrame) -> str:
        spec = FIELD_CONFIG[field]
        location = str(fact.get("location_label", ""))
        period = str(fact.get("period", ""))
        header = (
            f"Summarize EHS records for location {location}, period {period}. "
            f"Focus on {spec['label']}. Keep important dates and event IDs. "
            "Do not infer causation between audits, actions, and injuries. Records: "
        )
        details = subset.get("event_detail", pd.Series(dtype=str)).astype(str).tolist()
        text = header + " ".join(details)
        return compact_text(text, config.TRANSFORMER_MAX_INPUT_CHARS)

    def _run_batch(self, texts: List[str]) -> List[str]:
        if not texts:
            return []

        summaries: List[str] = []
        batch_size = max(1, int(getattr(config, "TRANSFORMER_BATCH_SIZE", 1)))
        max_new_tokens = int(getattr(config, "TRANSFORMER_MAX_SUMMARY_TOKENS", 150))
        min_new_tokens = int(getattr(config, "TRANSFORMER_MIN_SUMMARY_TOKENS", 25))

        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]
            try:
                encoded = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_input_tokens,
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}

                with self.torch.no_grad():
                    output_ids = self.model.generate(
                        **encoded,
                        max_new_tokens=max_new_tokens,
                        min_new_tokens=min_new_tokens,
                        num_beams=4,
                        do_sample=False,
                        early_stopping=True,
                    )

                decoded = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                summaries.extend([compact_text(str(x).strip(), 1600) for x in decoded])
            except Exception as exc:
                # Surface the exact generation error, because this is usually where
                # package incompatibilities or input-size problems appear.
                raise RuntimeError(f"Transformer generation failed for batch starting at {start}: {exc}") from exc

        return summaries

    def summarize(self, fact: pd.Series, group_examples: pd.DataFrame) -> Dict[str, str]:
        all_text = "\n".join(group_examples.get("event_detail", pd.Series(dtype=str)).astype(str).tolist())
        out: Dict[str, str] = {}
        fields_to_run: List[str] = []
        texts_to_run: List[str] = []

        fallback = ExtractiveSummarizer()
        fallback_summary: Optional[Dict[str, str]] = None

        for field, spec in FIELD_CONFIG.items():
            subset = group_examples[group_examples["review_event_type"].isin(spec["event_types"])] if not group_examples.empty else pd.DataFrame()
            if subset.empty:
                out[field] = spec["empty"]
            else:
                fields_to_run.append(field)
                texts_to_run.append(self._prepare_input(fact, field, subset))

        generated = self._run_batch(texts_to_run)
        for field, summary in zip(fields_to_run, generated):
            if summary:
                out[field] = summary
            else:
                if fallback_summary is None:
                    fallback_summary = fallback.summarize(fact, group_examples)
                out[field] = fallback_summary.get(field, "")

        out["recurring_themes"] = keyword_themes(all_text) or "No dominant terms found in sampled evidence."
        out["dates_to_review"] = date_list(all_text)
        out["data_gaps_or_cautions"] = (
            "Local transformer summary from sampled records only. Do not treat as causal evidence; "
            "review source records when making decisions."
        )
        out["summary_model_used"] = self.model_used
        return out

def get_summarizer() -> BaseSummarizer:
    backend = str(config.SUMMARIZER_BACKEND or "transformers").lower().strip()

    if getattr(config, "PRINT_TRANSFORMER_DIAGNOSTICS", True):
        report = collect_transformer_environment(load_torch=True)
        print_transformer_diagnostics(report)
        path = write_transformer_diagnostics(report, "transformer_startup_diagnostics.json")
        print(f"Wrote transformer startup diagnostics: {path}", flush=True)

    if backend == "extractive":
        print("SUMMARIZER_BACKEND='extractive', so the transformer model will not be loaded.", flush=True)
        return ExtractiveSummarizer()

    if backend == "transformers":
        try:
            return LocalTransformerSummarizer()
        except Exception as exc:
            error_report = collect_transformer_environment(load_torch=True)
            error_report["error_type"] = type(exc).__name__
            error_report["error_message"] = str(exc)
            error_report["traceback"] = traceback.format_exc()
            path = write_transformer_diagnostics(error_report, "transformer_initialization_error.json")
            txt_path = Path(config.LOG_DIR) / "transformer_initialization_error.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(error_report["traceback"])

            print("WARNING: local transformer summarizer could not be initialized.", flush=True)
            print(f"Reason type: {type(exc).__name__}", flush=True)
            print(f"Reason message: {exc}", flush=True)
            print(f"Wrote transformer error diagnostics: {path}", flush=True)
            print(f"Wrote transformer traceback: {txt_path}", flush=True)

            if config.ALLOW_EXTRACTIVE_FALLBACK:
                print("ALLOW_EXTRACTIVE_FALLBACK=True, so using extractive summaries.", flush=True)
                return ExtractiveSummarizer()
            raise

    raise ValueError("SUMMARIZER_BACKEND must be either 'transformers' or 'extractive'.")
