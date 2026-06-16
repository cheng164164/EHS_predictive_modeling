"""Generate higher-quality weak labels with an instruction LLM.

Key behavior:
- Uses a GPU model by default when CUDA is available, with CPU fallback.
- Scans the training JSONL and only labels records with meaningful text.
- Skips low-information records instead of forcing hallucinated labels.
- Writes raw/merged outputs plus skipped/rejected-label audit files.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from utils import cfg_path, load_config, read_jsonl, write_jsonl


CONFIDENCE_ORDER = {"low": 0, "medium": 1, "high": 2}
BAD_LABEL_STRINGS = {"", "0", "none", "n/a", "na", "null", "unknown", "distinct hazard_tags"}
LOW_INFO_PATTERNS = [
    r"^\s*(test|testing|n/?a|none|null|no issue|no issues)\s*$",
    r"^\s*(inspection|audit|task|title|description)\s*$",
    r"^\s*(completed|complete|done|ok|okay)\s*$",
]
SAFETY_TERMS = {
    "injury", "injured", "hurt", "pain", "cut", "burn", "fall", "fell", "slip", "trip", "strike", "struck",
    "pinch", "caught", "crush", "forklift", "truck", "vehicle", "crane", "hoist", "lift", "pedestrian",
    "electrical", "energized", "panel", "wire", "cord", "chemical", "spill", "leak", "ppe", "glove", "glasses",
    "hazard", "unsafe", "near miss", "near-miss", "blocked", "walkway", "housekeeping", "guard", "machine",
    "ladder", "height", "welding", "fire", "smoke", "pressure", "hydraulic", "lockout", "loto", "barricade",
}


def extract_user_record_text(rec: Dict[str, Any]) -> str:
    try:
        return str(rec.get("messages", [])[1].get("content", ""))
    except Exception:
        return ""


def compact_for_prompt(text: str, max_chars: int) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text or "")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()[:max_chars]


def is_meaningful_record(rec: Dict[str, Any], llm_cfg: Dict[str, Any]) -> Tuple[bool, str]:
    text = extract_user_record_text(rec)
    plain = re.sub(r"[^A-Za-zÀ-ÿ0-9\s/\-]", " ", text or " ")
    plain = re.sub(r"\s+", " ", plain).strip().lower()
    min_chars = int(llm_cfg.get("min_meaningful_text_chars", 140))
    min_words = int(llm_cfg.get("min_alpha_words", 12))

    if len(plain) < min_chars:
        return False, f"text too short ({len(plain)} chars < {min_chars})"
    for pat in LOW_INFO_PATTERNS:
        if re.search(pat, plain, flags=re.I):
            return False, "low-information/test-like text"
    words = re.findall(r"[a-zÀ-ÿ]{3,}", plain)
    if len(words) < min_words:
        return False, f"too few alpha words ({len(words)} < {min_words})"

    # The record can still be meaningful even without obvious safety terms, but if it lacks
    # any likely safety vocabulary and is mostly metadata, skip it to avoid bad synthetic labels.
    has_safety_term = any(term in plain for term in SAFETY_TERMS)
    metadata_noise = sum(plain.count(x) for x in ["source type", "event id", "description", "title", "status", "category"])
    if not has_safety_term and metadata_noise >= 4:
        return False, "mostly metadata with no clear safety content"
    return True, ""


def select_meaningful_records(records: List[Dict[str, Any]], max_records: int, max_scan: int, llm_cfg: Dict[str, Any]):
    selected: List[Tuple[int, Dict[str, Any]]] = []
    skipped: List[Dict[str, Any]] = []
    scan_count = min(max_scan, len(records)) if max_scan > 0 else len(records)
    for idx, rec in enumerate(records[:scan_count]):
        ok, reason = is_meaningful_record(rec, llm_cfg)
        if ok:
            selected.append((idx, rec))
            if len(selected) >= max_records:
                break
        else:
            skipped.append({
                "record_index": idx,
                "event_id": rec.get("metadata", {}).get("event_id"),
                "source_type": rec.get("metadata", {}).get("source_type"),
                "reason": reason,
            })
    return selected, skipped, scan_count


def resolve_model_name(cfg_section: Dict[str, Any], use_cuda: bool) -> str:
    model_name = str(cfg_section.get("model_name") or "Qwen/Qwen2.5-1.5B-Instruct")
    if not use_cuda and bool(cfg_section.get("auto_fallback_to_cpu_model", True)):
        return str(cfg_section.get("cpu_fallback_model") or model_name)
    return model_name


def load_generation_model(model_name: str, use_cuda: bool, trust_remote_code: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    is_encoder_decoder = bool(getattr(cfg, "is_encoder_decoder", False))

    if is_encoder_decoder:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch.float16 if use_cuda else torch.float32,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch.float16 if use_cuda else torch.float32,
            device_map="auto" if use_cuda else None,
        )
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

    torch_device = torch.device("cuda:0" if use_cuda else "cpu")
    if not use_cuda or is_encoder_decoder:
        model.to(torch_device)
    model.eval()
    return tokenizer, model, is_encoder_decoder, torch_device


def build_prompt(record_text: str, tokenizer: Any | None = None) -> str:
    instruction = (
        "You are creating weak labels for safety instruction-tuning data.\n"
        "Return valid JSON only. Do not include markdown. Do not classify PSIF.\n"
        "Use only facts supported by the record. If the record is unclear, use unknown/empty fields and confidence=low.\n"
        "Do not invent injuries, locations, equipment, root causes, or controls that are not mentioned.\n\n"
        "Return this exact schema:\n"
        "{\"risk_pattern\": string, \"hazard_tags\": [strings], \"control_failure_tags\": [strings], "
        "\"potential_consequence\": string, \"recommended_actions\": [strings], "
        "\"evidence_phrases\": [exact short phrases from record], \"limitations\": string, "
        "\"confidence\": \"low|medium|high\"}\n\n"
        "Examples:\n"
        "Record: Inspection completed. No specific issue described.\n"
        "JSON: {\"risk_pattern\": \"unknown\", \"hazard_tags\": [], \"control_failure_tags\": [], "
        "\"potential_consequence\": \"unknown\", \"recommended_actions\": [], \"evidence_phrases\": [], "
        "\"limitations\": \"No specific hazard or control failure is described.\", \"confidence\": \"low\"}\n\n"
        "Record: Broken pallet was found blocking the pedestrian walkway.\n"
        "JSON: {\"risk_pattern\": \"housekeeping / walking-working surface\", "
        "\"hazard_tags\": [\"broken pallet\", \"blocked walkway\"], "
        "\"control_failure_tags\": [\"housekeeping gap\", \"walkway access control gap\"], "
        "\"potential_consequence\": \"slip, trip, fall, or struck-by exposure\", "
        "\"recommended_actions\": [\"Remove the broken pallet from the walkway.\", \"Inspect the area for similar obstructions.\"], "
        "\"evidence_phrases\": [\"Broken pallet\", \"blocking the pedestrian walkway\"], "
        "\"limitations\": \"The record does not state whether the condition was recurring.\", \"confidence\": \"medium\"}\n\n"
        "Safety record:\n"
        f"{record_text}\n\nJSON:"
    )
    return instruction


def generate_batch(tokenizer, model, prompts: List[str], max_new_tokens: int, batch_size: int, torch_device, is_encoder_decoder: bool) -> List[str]:
    outputs: List[str] = []
    for _, sub_prompts in batch_iter(prompts, batch_size):
        encoded = tokenizer(sub_prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        encoded = {k: v.to(torch_device) for k, v in encoded.items()}
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        if is_encoder_decoder:
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        else:
            input_len = encoded["input_ids"].shape[1]
            decoded = tokenizer.batch_decode(generated[:, input_len:], skip_special_tokens=True)
        outputs.extend([d.strip() for d in decoded])
    return outputs


def try_parse_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return {}
    return {}


def normalize_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip() and str(x).strip().lower() not in BAD_LABEL_STRINGS]
    if isinstance(value, str):
        parts = re.split(r"[,;|]", value.strip())
        return [p.strip() for p in parts if p.strip() and p.strip().lower() not in BAD_LABEL_STRINGS]
    return []


def normalize_label(label: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(label, dict) or not label:
        return {}
    out: Dict[str, Any] = {}
    risk = str(label.get("risk_pattern", "unknown") or "unknown").strip()
    out["risk_pattern"] = "unknown" if risk.lower() in BAD_LABEL_STRINGS else risk
    out["hazard_tags"] = normalize_list(label.get("hazard_tags"))
    out["control_failure_tags"] = normalize_list(label.get("control_failure_tags"))
    pc = str(label.get("potential_consequence", "unknown") or "unknown").strip()
    out["potential_consequence"] = "unknown" if pc.lower() in BAD_LABEL_STRINGS else pc
    out["recommended_actions"] = normalize_list(label.get("recommended_actions"))
    out["evidence_phrases"] = normalize_list(label.get("evidence_phrases"))
    out["limitations"] = str(label.get("limitations", "") or "").strip()
    conf = str(label.get("confidence", "low") or "low").strip().lower()
    out["confidence"] = conf if conf in CONFIDENCE_ORDER else "low"

    # Conservative validation: useful labels should have at least hazard tags or evidence.
    if out["risk_pattern"] == "unknown" and not out["hazard_tags"]:
        out["control_failure_tags"] = []
        out["potential_consequence"] = "unknown"
        out["recommended_actions"] = []
        out["confidence"] = "low"
        out["limitations"] = out["limitations"] or "The record does not provide enough detail to identify a specific hazard or control failure."
    if out["confidence"] in {"medium", "high"} and not (out["hazard_tags"] or out["evidence_phrases"]):
        out["confidence"] = "low"
        out["limitations"] = out["limitations"] or "Generated label lacked supporting evidence phrases or hazard tags."
    return out


def confidence_allowed(label: Dict[str, Any], min_confidence: str) -> bool:
    min_confidence = str(min_confidence or "medium").lower()
    return CONFIDENCE_ORDER.get(label.get("confidence", "low"), 0) >= CONFIDENCE_ORDER.get(min_confidence, 1)


def merge_label_into_output(rec: Dict[str, Any], label: Dict[str, Any], min_confidence: str = "medium") -> Dict[str, Any]:
    if not label or not confidence_allowed(label, min_confidence):
        return rec
    try:
        out = json.loads(rec["messages"][2]["content"])
    except Exception:
        return rec
    for k in ["risk_pattern", "hazard_tags", "control_failure_tags", "potential_consequence", "recommended_actions"]:
        v = label.get(k)
        if v and v != "unknown":
            out[k] = v
    if label.get("evidence_phrases"):
        out["evidence_phrases"] = label["evidence_phrases"]
    if label.get("limitations"):
        out["limitations"] = label["limitations"]
    rec["messages"][2]["content"] = json.dumps(out, ensure_ascii=False, indent=2)
    return rec


def batch_iter(items: List[Any], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield i, items[i : i + batch_size]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--input-file", default="safety_instruction_train.jsonl")
    args = ap.parse_args()

    cfg = load_config(args.config)
    llm_cfg = cfg["labeling"]["light_llm"]
    prepared = cfg_path(cfg, "paths.prepared_dir")
    in_path = prepared / args.input_file

    if not llm_cfg.get("enabled", False):
        print("Light LLM labeling is disabled in config. Set labeling.light_llm.enabled=true to run.", flush=True)
        return

    records = read_jsonl(in_path)
    use_cuda = torch.cuda.is_available()
    model_name = resolve_model_name(llm_cfg, use_cuda)
    batch_size = int(llm_cfg.get("batch_size_gpu" if use_cuda else "batch_size_cpu", 1))
    batch_size = max(1, batch_size)
    max_records = min(int(llm_cfg.get("max_records", 200)), len(records))
    max_scan = int(llm_cfg.get("max_candidate_scan", len(records)) or len(records))
    progress_interval = max(1, int(llm_cfg.get("progress_interval", 25)))
    merge_enabled = bool(llm_cfg.get("merge_into_assistant_output", False))
    min_confidence = str(llm_cfg.get("merge_min_confidence", "medium"))
    max_prompt_chars = int(llm_cfg.get("max_prompt_chars", 1800))

    selected, skipped, scan_count = select_meaningful_records(records, max_records, max_scan, llm_cfg)
    out = list(records)

    print("Starting light LLM labeling", flush=True)
    print(f"  input_file: {in_path}", flush=True)
    print(f"  model_name: {model_name}", flush=True)
    print(f"  cuda_available: {use_cuda}", flush=True)
    print(f"  device: {'cuda:0' if use_cuda else 'cpu'}", flush=True)
    print(f"  meaningful_records_to_label: {len(selected)} requested={max_records} scanned={scan_count}", flush=True)
    print(f"  skipped_before_labeling: {len(skipped)}", flush=True)
    print(f"  batch_size: {batch_size}", flush=True)
    print(f"  merge_into_assistant_output: {merge_enabled}", flush=True)
    print(f"  merge_min_confidence: {min_confidence}", flush=True)

    tokenizer, model, is_encoder_decoder, torch_device = load_generation_model(
        model_name,
        use_cuda,
        trust_remote_code=bool(llm_cfg.get("trust_remote_code", False)),
    )
    max_new_tokens = int(llm_cfg.get("max_new_tokens", 384))

    start = time.time()
    parsed_count = merged_count = rejected_count = 0
    low_count = medium_count = high_count = 0
    rejected: List[Dict[str, Any]] = []

    for batch_start, batch_pairs in batch_iter(selected, batch_size):
        prompts = []
        indices = []
        for idx, rec in batch_pairs:
            text = compact_for_prompt(extract_user_record_text(rec), max_prompt_chars)
            prompts.append(build_prompt(text, tokenizer))
            indices.append(idx)

        results = generate_batch(tokenizer, model, prompts, max_new_tokens, batch_size, torch_device, is_encoder_decoder)

        for rec_idx, raw in zip(indices, results):
            rec = out[rec_idx]
            parsed = normalize_label(try_parse_json(raw))
            if parsed:
                parsed_count += 1
            else:
                rejected_count += 1
                rejected.append({
                    "record_index": rec_idx,
                    "event_id": rec.get("metadata", {}).get("event_id"),
                    "source_type": rec.get("metadata", {}).get("source_type"),
                    "reason": "could not parse valid JSON",
                    "raw_label": raw,
                })
                parsed = {"confidence": "low"}

            conf = parsed.get("confidence", "low")
            if conf == "high":
                high_count += 1
            elif conf == "medium":
                medium_count += 1
            else:
                low_count += 1
                if raw and raw.strip():
                    rejected.append({
                        "record_index": rec_idx,
                        "event_id": rec.get("metadata", {}).get("event_id"),
                        "source_type": rec.get("metadata", {}).get("source_type"),
                        "reason": "low confidence or insufficient evidence",
                        "parsed_label": parsed,
                        "raw_label": raw,
                    })

            rec.setdefault("metadata", {})["light_llm_raw_label"] = raw
            rec["metadata"]["light_llm_parsed_label"] = parsed
            rec["metadata"]["label_source"] = rec["metadata"].get("label_source", "") + "+light_llm"
            if merge_enabled and confidence_allowed(parsed, min_confidence):
                rec = merge_label_into_output(rec, parsed, min_confidence=min_confidence)
                rec["metadata"]["label_source"] += "_merged"
                merged_count += 1
            out[rec_idx] = rec

        completed = min(batch_start + len(batch_pairs), len(selected))
        if completed % progress_interval == 0 or completed == len(selected):
            elapsed = time.time() - start
            rate = completed / elapsed if elapsed > 0 else 0.0
            remaining = len(selected) - completed
            eta_min = (remaining / rate / 60.0) if rate > 0 else 0.0
            print(
                f"Progress: {completed}/{len(selected)} labeled ({completed / max(1, len(selected)):.1%}); "
                f"parsed={parsed_count}; rejected={rejected_count}; "
                f"confidence low/medium/high={low_count}/{medium_count}/{high_count}; "
                f"merged={merged_count}; elapsed={elapsed/60.0:.1f} min; ETA={eta_min:.1f} min",
                flush=True,
            )

    suffix = "merged" if merge_enabled else "raw"
    configured_key = "output_merged_file" if merge_enabled else "output_raw_file"
    out_path = prepared / llm_cfg.get(configured_key, f"safety_instruction_train_with_light_llm_{suffix}.jsonl")
    write_jsonl(out, out_path)

    skipped_path = prepared / llm_cfg.get("skipped_file", "light_llm_skipped_records.jsonl")
    rejected_path = prepared / llm_cfg.get("rejected_file", "light_llm_rejected_labels.jsonl")
    write_jsonl(skipped, skipped_path)
    write_jsonl(rejected, rejected_path)

    elapsed = time.time() - start
    summary = {
        "input_file": str(in_path),
        "output_file": str(out_path),
        "skipped_file": str(skipped_path),
        "rejected_file": str(rejected_path),
        "model_name": model_name,
        "device": "cuda:0" if use_cuda else "cpu",
        "records_total": len(records),
        "records_scanned": scan_count,
        "meaningful_records_selected": len(selected),
        "skipped_before_labeling": len(skipped),
        "parsed_count": parsed_count,
        "rejected_count": rejected_count,
        "confidence_counts": {"low": low_count, "medium": medium_count, "high": high_count},
        "merge_into_assistant_output": merge_enabled,
        "merged_count": merged_count,
        "elapsed_minutes": round(elapsed / 60.0, 2),
    }
    summary_path = prepared / "light_llm_label_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Wrote {out_path}", flush=True)
    print(f"Wrote {summary_path}", flush=True)
    print(f"Wrote {skipped_path}", flush=True)
    print(f"Wrote {rejected_path}", flush=True)
    print("Light LLM labeling finished", flush=True)


if __name__ == "__main__":
    main()
