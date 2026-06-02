from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
import platform
import re
import sys
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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
        "label": "unsafe conditions, unsafe acts, hazards, and near misses",
        "event_types": ["audit_unsafe_condition", "audit_unsafe_act", "hazard_identification", "near_miss"],
        "fact_counts": ["unsafe_condition_audit_count", "unsafe_act_audit_count", "hazard_identification_count", "near_miss_count"],
        "empty": "No sampled unsafe condition, unsafe act, hazard, or near-miss evidence for this location-period.",
    },
    "serious_injury_summary": {
        "label": "serious injuries",
        "event_types": ["serious_injury"],
        "fact_counts": ["serious_injury_count"],
        "empty": "No sampled serious injury records for this location-period.",
    },
    "normal_injury_summary": {
        "label": "normal injuries",
        "event_types": ["normal_injury"],
        "fact_counts": ["normal_injury_count"],
        "empty": "No sampled normal injury records for this location-period.",
    },
    "near_miss_summary": {
        "label": "near misses",
        "event_types": ["near_miss"],
        "fact_counts": ["near_miss_count"],
        "empty": "No sampled near-miss records for this location-period.",
    },
    "hazards_summary": {
        "label": "hazard identifications",
        "event_types": ["hazard_identification"],
        "fact_counts": ["hazard_identification_count"],
        "empty": "No sampled hazard-identification records for this location-period.",
    },
    "audits_summary": {
        "label": "audits, observations, inspections, and unsafe findings",
        "event_types": ["audit_unsafe_condition", "audit_unsafe_act", "audit_other"],
        "fact_counts": ["audit_count", "unsafe_condition_audit_count", "unsafe_act_audit_count"],
        "empty": "No sampled audit or observation records for this location-period.",
    },
    "actions_summary": {
        "label": "corrective actions, open tasks, overdue tasks, and completed tasks",
        "event_types": ["task_overdue", "task_open", "task_completed_or_closed", "task_other"],
        "fact_counts": ["task_count", "open_action_count", "overdue_action_count", "completed_or_closed_task_count"],
        "empty": "No sampled task or corrective-action records for this location-period.",
    },
}

BAD_GENERATION_PATTERNS = [
    "summarize ehs records",
    "do not infer",
    "focus on",
    "records:",
    "location-period",
    "keep important dates",
]

EXTRA_STOPWORDS = set(
    "title description immediateaction immediate action offpremiseslocation premiseslocation "
    "source subtype category status record records summarize summary location period ehs "
    "closed review investigation pending open completed complete corrective action actions "
    "employee employees found observed observation audit incident hazard hazards near miss "
    "unsafe safe condition act normal serious injury injuries date dates event events "
    "first second third shift area department plant site task tasks were taken using from over under then then the and with into while when where what this that there they them their his her our your was are been being had has have will would should could also after before during only report".split()
)


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
    print(f"output_dir: {report.get('output_dir')}", flush=True)
    print(f"backend: {report.get('summarizer_backend')}", flush=True)
    print(f"model: {report.get('transformer_model_name')}", flush=True)
    print(f"device: {report.get('transformer_device')}", flush=True)
    print(f"local_files_only: {report.get('transformer_local_files_only')}", flush=True)
    print(f"modules_found: {report.get('modules_found')}", flush=True)
    print(f"package_versions: {report.get('package_versions')}", flush=True)
    if "torch" in report:
        print(f"torch: {report.get('torch')}", flush=True)
    print("-----------------------------------------------", flush=True)


def clean_evidence_text(value: object, max_chars: int = 450) -> str:
    """Remove noisy field labels and normalize source evidence text."""
    text = "" if value is None else str(value)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""

    # Drop prompt or instruction residue if present from older runs.
    text = re.sub(r"(?i)Summarize EHS records.*?Records:\s*", "", text)
    text = re.sub(r"(?i)Do not infer causation.*?Records:\s*", "", text)

    # Normalize common source field labels.
    replacements = [
        (r"(?i)\btitle\s*:\s*", ""),
        (r"(?i)\btask\s*:\s*", ""),
        (r"(?i)\breport only\s*:\s*", ""),
        (r"(?i)\bdescription\s*:\s*", ""),
        (r"(?i)\bimmediate\s*action\s*:\s*", "Action taken: "),
        (r"(?i)\bimmediateaction\s*:\s*", "Action taken: "),
        (r"(?i)\boffpremiseslocation\s*:\s*[^|;]+", ""),
        (r"(?i)\botherprocess\s*:\s*", "process: "),
        (r"(?i)\botheractivity\s*:\s*", "activity: "),
        (r"(?i)\bactivityduringincident\s*:\s*", "activity: "),
    ]
    for pattern, repl in replacements:
        text = re.sub(pattern, repl, text)

    text = re.sub(r"\s*\|\s*", " | ", text)
    text = re.sub(r"\s*;\s*", "; ", text)
    text = re.sub(r"\s+", " ", text).strip(" -|;.")
    return compact_text(text, max_chars)


def _fact_counts(fact: pd.Series, fields: List[str]) -> str:
    parts = []
    for field in fields:
        if field in fact.index:
            try:
                value = int(float(fact.get(field, 0) or 0))
            except Exception:
                value = 0
            label = field.replace("_count", "").replace("_", " ")
            parts.append(f"{label}={value}")
    return "; ".join(parts)


def _sample_counts(subset: pd.DataFrame) -> str:
    if subset.empty or "review_event_type" not in subset.columns:
        return "sampled=0"
    counts = subset["review_event_type"].astype(str).value_counts().to_dict()
    return "; ".join(f"{k}={v}" for k, v in counts.items())


def _text_for_row(row: pd.Series, max_chars: int = 450) -> str:
    raw = row.get("clean_text", "")
    text = clean_evidence_text(raw, max_chars=max_chars)
    if not text:
        text = clean_evidence_text(row.get("event_detail", ""), max_chars=max_chars)
    return text


def _representative_records(subset: pd.DataFrame, max_events: int) -> List[str]:
    if subset.empty:
        return []
    sort_cols = [c for c in ["review_priority", "event_dt"] if c in subset.columns]
    work = subset.sort_values(sort_cols) if sort_cols else subset.copy()
    seen = set()
    records: List[str] = []
    for _, row in work.iterrows():
        date = str(row.get("event_dt", ""))[:10]
        event_id = str(row.get("event_id", "")).strip()
        rtype = str(row.get("review_event_type", "")).replace("_", " ")
        status = str(row.get("status", "")).strip()
        due = str(row.get("due_dt", "")).strip()
        completion = str(row.get("completion_dt", "")).strip()
        text = _text_for_row(row, max_chars=420)
        if not text:
            continue
        norm = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()[:160]
        if norm in seen:
            continue
        seen.add(norm)
        meta = []
        if date:
            meta.append(date)
        if event_id:
            meta.append(event_id)
        if rtype:
            meta.append(rtype)
        if status:
            meta.append(f"status={status}")
        if due and due.lower() != "nan":
            meta.append(f"due={due[:10]}")
        if completion and completion.lower() != "nan":
            meta.append(f"completed={completion[:10]}")
        records.append(f"{' | '.join(meta)} - {text}")
        if len(records) >= max_events:
            break
    return records


def _top_terms_from_examples(subset: pd.DataFrame, max_terms: int = 8) -> str:
    if subset.empty:
        return ""
    texts = []
    for _, row in subset.iterrows():
        texts.append(_text_for_row(row, max_chars=800))
    text = " ".join(texts).lower()
    words = re.findall(r"[a-z][a-z\-]{3,}", text)
    counts = Counter(w for w in words if w not in EXTRA_STOPWORDS and not w.isdigit())
    return ", ".join(w for w, _ in counts.most_common(max_terms))


def _dedupe_sentences(text: str, max_chars: int = 900) -> str:
    text = clean_evidence_text(text, max_chars=5000)
    parts = re.split(r"(?<=[.!?])\s+|\s*;\s*", text)
    out = []
    seen = set()
    for p in parts:
        p = p.strip(" -|;.")
        if len(p) < 12:
            continue
        norm = re.sub(r"[^a-z0-9]+", " ", p.lower()).strip()[:120]
        if norm in seen:
            continue
        seen.add(norm)
        out.append(p)
        if len(". ".join(out)) >= max_chars:
            break
    if not out:
        return compact_text(text, max_chars)
    return compact_text(". ".join(out) + ".", max_chars)


def _looks_bad_generation(text: str) -> bool:
    lo = str(text or "").lower().strip()
    if len(lo) < 30:
        return True
    if any(p in lo for p in BAD_GENERATION_PATTERNS):
        return True
    # Reject obvious field-label echoes.
    if lo.count("title:") + lo.count("description:") + lo.count("immediateaction") >= 2:
        return True
    # Reject repeated loops.
    words = lo.split()
    if len(words) >= 25:
        unique_ratio = len(set(words)) / max(1, len(words))
        if unique_ratio < 0.45:
            return True
    return False


class BaseSummarizer:
    model_used = "base"

    def summarize(self, fact: pd.Series, group_examples: pd.DataFrame) -> Dict[str, str]:
        raise NotImplementedError


class StructuredEvidenceSummarizer(BaseSummarizer):
    """Review-grade deterministic summary from sampled records.

    This is intentionally not a language model. It is designed to be more useful
    for EHS review than a generic news summarizer: it keeps counts, dates, event
    IDs, status, and representative evidence without inventing relationships.
    """

    model_used = "structured_evidence_no_llm"

    def _section_summary(self, fact: pd.Series, subset: pd.DataFrame, field: str) -> str:
        spec = FIELD_CONFIG[field]
        if subset.empty:
            return spec["empty"]
        fact_line = _fact_counts(fact, spec.get("fact_counts", []))
        sample_line = _sample_counts(subset)
        terms = _top_terms_from_examples(subset)
        records = _representative_records(
            subset,
            max_events=int(getattr(config, "STRUCTURED_SUMMARY_MAX_EVENTS_PER_SECTION", 5)),
        )
        pieces = [f"Counts: {fact_line}. Sample used: {sample_line}."]
        if terms:
            pieces.append(f"Repeated terms/themes in sampled evidence: {terms}.")
        if records:
            pieces.append("Representative records: " + " || ".join(records))
        return compact_text(" ".join(pieces), int(getattr(config, "STRUCTURED_SUMMARY_MAX_CHARS", 1600)))

    def summarize(self, fact: pd.Series, group_examples: pd.DataFrame) -> Dict[str, str]:
        out: Dict[str, str] = {}
        all_clean_text = []
        for field, spec in FIELD_CONFIG.items():
            subset = group_examples[group_examples["review_event_type"].isin(spec["event_types"])] if not group_examples.empty else pd.DataFrame()
            out[field] = self._section_summary(fact, subset, field)
            if not subset.empty:
                all_clean_text.extend(_text_for_row(row, max_chars=800) for _, row in subset.iterrows())

        all_text = "\n".join(all_clean_text)
        out["recurring_themes"] = _top_terms_from_examples(group_examples, max_terms=12) or keyword_themes(all_text) or "No dominant terms found in sampled evidence."
        if not group_examples.empty and "event_dt" in group_examples.columns:
            dates = sorted(set(str(x)[:10] for x in group_examples["event_dt"].dropna().astype(str) if str(x).strip()))
            out["dates_to_review"] = ", ".join(dates[:25]) if dates else "No dates found in sampled evidence."
        else:
            out["dates_to_review"] = "No sampled evidence dates available for this location-period."
        out["data_gaps_or_cautions"] = (
            "Structured summary from sampled records. It preserves evidence but does not prove causal links between "
            "hazards, audits, actions, near misses, and injuries. Review source records before decisions."
        )
        out["summary_model_used"] = self.model_used
        return out


# Backward compatible alias. The old extractive fallback is now replaced by the
# better structured evidence summary.
class ExtractiveSummarizer(StructuredEvidenceSummarizer):
    model_used = "structured_evidence_no_llm"


class LocalTransformerSummarizer(BaseSummarizer):
    """Free local Hugging Face seq2seq summarizer using manual generate."""

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
                "SUMMARIZER_BACKEND='structured' in config.py."
            ) from exc

        self.torch = torch

        try:
            local_only = bool(getattr(config, "TRANSFORMER_LOCAL_FILES_ONLY", False))
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=local_only)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, local_files_only=local_only)

            requested_device = int(getattr(config, "TRANSFORMER_DEVICE", -1))
            if requested_device >= 0 and torch.cuda.is_available():
                self.device = torch.device(f"cuda:{requested_device}")
            else:
                self.device = torch.device("cpu")

            self.model.to(self.device)
            self.model.eval()

            raw_max_len = getattr(self.tokenizer, "model_max_length", 512)
            if not isinstance(raw_max_len, int) or raw_max_len > 4096:
                raw_max_len = 512
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
        records = _representative_records(subset, max_events=6)
        evidence = "\n".join(f"- {r}" for r in records)
        prompt = (
            "Write a concise EHS review summary from the evidence only. "
            "Use 2 to 4 short bullets. Mention repeated hazards, unsafe conditions, injuries, near misses, "
            "or task status when present. Do not invent causes. Do not repeat these instructions.\n"
            f"Location: {location}\nPeriod: {period}\nTopic: {spec['label']}\nEvidence:\n{evidence}\nSummary:"
        )
        return compact_text(prompt, int(getattr(config, "TRANSFORMER_MAX_INPUT_CHARS", 2800)))

    def _run_batch(self, texts: List[str]) -> List[str]:
        if not texts:
            return []

        summaries: List[str] = []
        batch_size = max(1, int(getattr(config, "TRANSFORMER_BATCH_SIZE", 1)))
        max_new_tokens = int(getattr(config, "TRANSFORMER_MAX_SUMMARY_TOKENS", 140))
        min_new_tokens = int(getattr(config, "TRANSFORMER_MIN_SUMMARY_TOKENS", 20))

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
                cleaned = []
                for x in decoded:
                    summary = _dedupe_sentences(str(x).strip(), max_chars=900)
                    cleaned.append(summary)
                summaries.extend(cleaned)
            except Exception as exc:
                raise RuntimeError(f"Transformer generation failed for batch starting at {start}: {exc}") from exc

        return summaries

    def summarize(self, fact: pd.Series, group_examples: pd.DataFrame) -> Dict[str, str]:
        structured = StructuredEvidenceSummarizer().summarize(fact, group_examples)
        out = dict(structured)
        fields_to_run: List[str] = []
        texts_to_run: List[str] = []

        for field, spec in FIELD_CONFIG.items():
            subset = group_examples[group_examples["review_event_type"].isin(spec["event_types"])] if not group_examples.empty else pd.DataFrame()
            if subset.empty:
                continue
            fields_to_run.append(field)
            texts_to_run.append(self._prepare_input(fact, field, subset))

        generated = self._run_batch(texts_to_run)
        used_fields = []
        for field, summary in zip(fields_to_run, generated):
            if summary and not _looks_bad_generation(summary):
                # Keep dates/IDs available by appending representative evidence from the structured summary.
                out[field] = compact_text(summary + " Evidence: " + out.get(field, ""), int(getattr(config, "STRUCTURED_SUMMARY_MAX_CHARS", 1600)))
                used_fields.append(field)

        out["data_gaps_or_cautions"] = (
            "Hybrid transformer/evidence summary from sampled records only. Transformer text was rejected when it "
            "looked generic or repeated the prompt. Do not treat this as causal proof."
        )
        out["summary_model_used"] = self.model_used + (f"; accepted_fields={','.join(used_fields)}" if used_fields else "; structured_only_after_quality_check")
        return out


class HybridSummarizer(BaseSummarizer):
    """Structured summaries for all rows, optional transformer polish for top groups."""

    def __init__(self):
        self.structured = StructuredEvidenceSummarizer()
        self.transformer: Optional[LocalTransformerSummarizer] = None
        self.group_index = 0
        self.max_transformer_groups = int(getattr(config, "HYBRID_TRANSFORMER_MAX_GROUPS", 0) or 0)
        self.transformer_fields = set(getattr(config, "HYBRID_TRANSFORMER_FIELDS", set()) or set())
        self.model_used = f"hybrid_structured_plus_optional_transformer:{config.TRANSFORMER_MODEL_NAME}"

        if self.max_transformer_groups != -1:
            try:
                self.transformer = LocalTransformerSummarizer()
            except Exception as exc:
                if not getattr(config, "ALLOW_EXTRACTIVE_FALLBACK", True):
                    raise
                print("WARNING: hybrid transformer polish unavailable; using structured summaries only.", flush=True)
                print(f"Reason: {exc}", flush=True)
                self.transformer = None
                self.model_used = "hybrid_structured_only_transformer_unavailable"

    def summarize(self, fact: pd.Series, group_examples: pd.DataFrame) -> Dict[str, str]:
        self.group_index += 1
        base = self.structured.summarize(fact, group_examples)
        if self.transformer is None or group_examples.empty:
            base["summary_model_used"] = self.model_used
            return base

        # Limit transformer polishing to top groups by the sorted order passed from 02.
        if self.max_transformer_groups > 0 and self.group_index > self.max_transformer_groups:
            base["summary_model_used"] = f"{self.model_used}; transformer_skipped_after_top_{self.max_transformer_groups}"
            return base

        fields_to_run: List[str] = []
        texts_to_run: List[str] = []
        for field, spec in FIELD_CONFIG.items():
            if self.transformer_fields and field not in self.transformer_fields:
                continue
            subset = group_examples[group_examples["review_event_type"].isin(spec["event_types"])]
            if subset.empty:
                continue
            fields_to_run.append(field)
            texts_to_run.append(self.transformer._prepare_input(fact, field, subset))

        if not texts_to_run:
            base["summary_model_used"] = f"{self.model_used}; no_transformer_fields_with_evidence"
            return base

        try:
            generated = self.transformer._run_batch(texts_to_run)
        except Exception as exc:
            base["summary_model_used"] = f"{self.model_used}; transformer_generation_failed"
            base["data_gaps_or_cautions"] += f" Transformer generation failed for this row and structured summary was kept: {exc}"
            return base

        accepted = []
        for field, summary in zip(fields_to_run, generated):
            if summary and not _looks_bad_generation(summary):
                base[field] = compact_text(summary + " Evidence: " + base.get(field, ""), int(getattr(config, "STRUCTURED_SUMMARY_MAX_CHARS", 1600)))
                accepted.append(field)

        base["summary_model_used"] = f"{self.model_used}; transformer_accepted={','.join(accepted) if accepted else 'none'}"
        return base


def get_summarizer() -> BaseSummarizer:
    backend = str(config.SUMMARIZER_BACKEND or "hybrid").lower().strip()

    if backend in {"hybrid", "transformers"} and getattr(config, "PRINT_TRANSFORMER_DIAGNOSTICS", True):
        report = collect_transformer_environment(load_torch=True)
        print_transformer_diagnostics(report)
        path = write_transformer_diagnostics(report, "transformer_startup_diagnostics.json")
        print(f"Wrote transformer startup diagnostics: {path}", flush=True)

    if backend in {"structured", "extractive"}:
        print(f"SUMMARIZER_BACKEND='{backend}', so no transformer model will be loaded.", flush=True)
        return StructuredEvidenceSummarizer()

    if backend == "hybrid":
        return HybridSummarizer()

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
            ensure_dir(config.LOG_DIR)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(error_report["traceback"])

            print("WARNING: local transformer summarizer could not be initialized.", flush=True)
            print(f"Reason type: {type(exc).__name__}", flush=True)
            print(f"Reason message: {exc}", flush=True)
            print(f"Wrote transformer error diagnostics: {path}", flush=True)
            print(f"Wrote transformer traceback: {txt_path}", flush=True)

            if config.ALLOW_EXTRACTIVE_FALLBACK:
                print("ALLOW_EXTRACTIVE_FALLBACK=True, so using structured evidence summaries.", flush=True)
                return StructuredEvidenceSummarizer()
            raise

    raise ValueError("SUMMARIZER_BACKEND must be one of: hybrid, structured, transformers, extractive.")
