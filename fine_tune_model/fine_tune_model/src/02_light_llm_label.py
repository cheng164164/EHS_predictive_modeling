"""Generate higher-quality weak labels with an instruction LLM.

Key behavior:
- Uses a GPU model by default when CUDA is available, with CPU fallback.
- Scans the training JSONL and only labels records with meaningful text.
- Skips low-information records instead of forcing hallucinated labels.
- Keeps all selected safety-related records, including low-confidence and JSON-fallback labels.
- Only rejects records before labeling when text is too short or not meaningful.
- Writes raw/merged outputs plus included/skipped/rejected-label audit files.
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
    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    is_encoder_decoder = bool(getattr(cfg, "is_encoder_decoder", False))

    # Qwen/Llama/Mistral are decoder-only models. For batched generation,
    # left padding avoids the repeated warning about right-padding and gives
    # more reliable generation.
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if not is_encoder_decoder:
        tokenizer.padding_side = "left"

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
        "{\"risk_pattern\": string, \"risk_pattern_description\": string, \"additional_patterns\": [strings], "
        "\"hazard_tags\": [strings], \"control_failure_tags\": [strings], "
        "\"potential_consequence\": string, \"recommended_actions\": [strings], "
        "\"evidence_phrases\": [exact short phrases from record], \"limitations\": string, "
        "\"confidence\": \"low|medium|high\"}\n\n"
        "Examples:\n"
        "Record: Inspection completed. No specific issue described.\n"
        "JSON: {\"risk_pattern\": \"unknown\", \"risk_pattern_description\": \"No specific risk pattern can be inferred from the available text.\", "
        "\"additional_patterns\": [], \"hazard_tags\": [], \"control_failure_tags\": [], "
        "\"potential_consequence\": \"unknown\", \"recommended_actions\": [], \"evidence_phrases\": [], "
        "\"limitations\": \"No specific hazard or control failure is described.\", \"confidence\": \"low\"}\n\n"
        "Record: Broken pallet was found blocking the pedestrian walkway.\n"
        "JSON: {\"risk_pattern\": \"housekeeping / walking-working surface\", "
        "\"risk_pattern_description\": \"The record describes an obstruction in a pedestrian walkway, indicating housekeeping and walking-working-surface exposure.\", "
        "\"additional_patterns\": [\"blocked walkway / pedestrian access obstruction\"], "
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


def extract_balanced_json(text: str) -> str:
    """Return the first balanced JSON-like object found in text, or an empty string."""
    text = (text or "").strip()
    start = text.find("{")
    if start < 0:
        return ""
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return ""


def repair_json_text(text: str) -> str:
    """Repair only common LLM formatting issues; do not invent field values."""
    text = (text or "").strip()
    candidate = extract_balanced_json(text) or text
    candidate = candidate.strip()
    if not candidate.startswith("{") and ('"risk_pattern"' in candidate or '"hazard_tags"' in candidate):
        candidate = "{" + candidate
    if candidate.startswith("{") and not candidate.endswith("}"):
        candidate = candidate + "}"
    candidate = candidate.replace(";\n", ",\n")
    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
    # Normalize common accidental key casing.
    replacements = {
        '"Recommended_actions"': '"recommended_actions"',
        '"recommended action"': '"recommended_actions"',
        '"hazard tag"': '"hazard_tags"',
        '"hazard pattern"': '"risk_pattern"',
        '"risk pattern"': '"risk_pattern"',
    }
    for a, b in replacements.items():
        candidate = candidate.replace(a, b)
    return candidate


def try_parse_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {}
    for candidate in [text, extract_balanced_json(text), repair_json_text(text)]:
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            continue
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
    desc = str(label.get("risk_pattern_description", "") or "").strip()
    out["risk_pattern_description"] = "" if desc.lower() in BAD_LABEL_STRINGS else desc
    out["additional_patterns"] = dedupe_list(normalize_list(label.get("additional_patterns")), max_items=8)
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




def dedupe_list(items: List[str], max_items: int = 6) -> List[str]:
    seen = set()
    out = []
    for item in items:
        item = re.sub(r"\s+", " ", str(item or "").strip())
        key = item.lower()
        if not item or key in seen or key in BAD_LABEL_STRINGS:
            continue
        seen.add(key)
        out.append(item)
        if len(out) >= max_items:
            break
    return out


def compact_safety_description_from_text(record_text: str, raw_label: str = "") -> str:
    """Build a useful free-text description when the model did not return valid JSON.

    This keeps safety-related records instead of discarding them. It does not try
    to claim a precise risk taxonomy; it preserves the model's useful natural
    language if available, otherwise uses the cleaned record text.
    """
    raw = re.sub(r"\s+", " ", (raw_label or "").strip())
    # Avoid keeping schema-repetition noise as the summary.
    if raw and raw.lower().count("recommended_actions") <= 1 and len(raw) >= 30:
        return raw[:700]
    cleaned = re.sub(r"\s+", " ", (record_text or "").strip())
    return cleaned[:700]


def load_existing_assistant_output(rec: Dict[str, Any]) -> Dict[str, Any]:
    try:
        obj = json.loads(rec.get("messages", [])[2].get("content", "{}"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def fallback_label_from_record(rec: Dict[str, Any], raw_label: str, record_text: str) -> Dict[str, Any]:
    """Create a schema-compatible low-confidence label when JSON parsing fails."""
    base = load_existing_assistant_output(rec)
    summary = compact_safety_description_from_text(record_text, raw_label)
    hazard_tags = normalize_list(base.get("hazard_tags"))
    control_tags = normalize_list(base.get("control_failure_tags"))
    recommended_actions = normalize_list(base.get("recommended_actions"))
    evidence = normalize_list(base.get("evidence_phrases"))
    if not evidence:
        # Use a short exact-ish phrase from the record as weak evidence.
        cleaned = re.sub(r"\s+", " ", (record_text or "").strip())
        evidence = [cleaned[:180]] if cleaned else []
    label = {
        "risk_pattern": str(base.get("risk_pattern") or "unknown"),
        "risk_pattern_description": str(base.get("risk_pattern_description") or "Fallback label retained the existing weak risk pattern because the LLM output was not valid JSON."),
        "additional_patterns": dedupe_list(normalize_list(base.get("additional_patterns")), max_items=8),
        "hazard_tags": dedupe_list(hazard_tags),
        "control_failure_tags": dedupe_list(control_tags),
        "potential_consequence": str(base.get("potential_consequence") or "unknown"),
        "recommended_actions": dedupe_list(recommended_actions),
        "evidence_phrases": dedupe_list(evidence, max_items=3),
        "limitations": "LLM output was not valid JSON, so this label keeps the existing weak taxonomy label and stores the LLM output as free-text description.",
        "confidence": "low",
        "llm_free_text_description": summary,
        "parse_status": "fallback_from_invalid_json",
    }
    return normalize_label(label) | {"llm_free_text_description": summary, "parse_status": "fallback_from_invalid_json"}

def confidence_allowed(label: Dict[str, Any], min_confidence: str) -> bool:
    min_confidence = str(min_confidence or "medium").lower()
    return CONFIDENCE_ORDER.get(label.get("confidence", "low"), 0) >= CONFIDENCE_ORDER.get(min_confidence, 1)


def clean_pattern_value(value: Any) -> str:
    value = re.sub(r"\s+", " ", str(value or "").strip())
    return "" if value.lower() in BAD_LABEL_STRINGS else value


def merge_pattern_lists(*values: Any, max_items: int = 10) -> List[str]:
    patterns: List[str] = []
    for value in values:
        if isinstance(value, list):
            patterns.extend(value)
        else:
            patterns.append(value)
    return dedupe_list([clean_pattern_value(x) for x in patterns], max_items=max_items)


def build_risk_pattern_description(primary: str, all_patterns: List[str], llm_description: str = "") -> str:
    if llm_description:
        return llm_description
    if not all_patterns:
        return "No specific risk pattern was identified from the available record text."
    if len(all_patterns) == 1:
        return f"Primary risk pattern retained from the weak label/light LLM merge: {all_patterns[0]}."
    related = "; ".join(all_patterns[1:])
    return (
        f"Primary risk pattern retained for training: {primary}. "
        f"The light LLM also detected related or more specific pattern(s): {related}. "
        "These patterns are kept together as weak training signals rather than forcing the event into only the predefined taxonomy."
    )


def merge_label_into_output(rec: Dict[str, Any], label: Dict[str, Any], min_confidence: str = "low") -> Dict[str, Any]:
    """Merge light-LLM labels without overwriting the original weak risk pattern.

    The original risk_pattern is kept as the primary pattern when it is present.
    The LLM-detected pattern is preserved in additional_patterns when it differs,
    so training data can learn both predefined taxonomy patterns and newly
    discovered/free-form patterns.
    """
    if not label:
        return rec
    try:
        out = json.loads(rec["messages"][2]["content"])
    except Exception:
        return rec

    original_pattern = clean_pattern_value(out.get("risk_pattern"))
    llm_pattern = clean_pattern_value(label.get("risk_pattern"))
    existing_additional = normalize_list(out.get("additional_patterns"))
    llm_additional = normalize_list(label.get("additional_patterns"))

    all_patterns = merge_pattern_lists(original_pattern, existing_additional, llm_pattern, llm_additional, max_items=10)
    primary_pattern = original_pattern or (all_patterns[0] if all_patterns else "unknown")
    additional_patterns = [p for p in all_patterns if p.lower() != primary_pattern.lower()]

    out["risk_pattern"] = primary_pattern
    out["additional_patterns"] = additional_patterns
    out["risk_pattern_description"] = build_risk_pattern_description(
        primary=primary_pattern,
        all_patterns=[primary_pattern] + additional_patterns if primary_pattern != "unknown" else additional_patterns,
        llm_description=str(label.get("risk_pattern_description", "") or "").strip(),
    )

    # Let the light LLM enrich supporting fields, but do not replace the primary risk_pattern.
    for k in ["hazard_tags", "control_failure_tags", "potential_consequence", "recommended_actions"]:
        v = label.get(k)
        if v and v != "unknown":
            out[k] = v
    if label.get("evidence_phrases"):
        out["evidence_phrases"] = label["evidence_phrases"]
    if label.get("limitations"):
        out["limitations"] = label["limitations"]
    if label.get("llm_free_text_description"):
        out["llm_free_text_description"] = label["llm_free_text_description"]

    rec.setdefault("metadata", {})["risk_pattern_original"] = original_pattern
    rec["metadata"]["risk_pattern_light_llm"] = llm_pattern
    rec["metadata"]["risk_pattern_all"] = all_patterns
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
    parsed_count = merged_count = 0
    parse_fallback_count = 0
    low_count = medium_count = high_count = 0
    included: List[Dict[str, Any]] = []
    parse_fallbacks: List[Dict[str, Any]] = []
    # Rejected now means not useful enough to send to the LLM, not low-confidence output.
    rejected: List[Dict[str, Any]] = list(skipped)

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
            record_text = compact_for_prompt(extract_user_record_text(rec), max_prompt_chars)
            parsed_raw = try_parse_json(raw)
            parsed = normalize_label(parsed_raw)
            parse_status = "parsed_json" if parsed else "fallback_from_invalid_json"
            if parsed:
                parsed_count += 1
            else:
                parse_fallback_count += 1
                parsed = fallback_label_from_record(rec, raw, record_text)
                parse_fallbacks.append({
                    "record_index": rec_idx,
                    "event_id": rec.get("metadata", {}).get("event_id"),
                    "source_type": rec.get("metadata", {}).get("source_type"),
                    "reason": "invalid_json_but_kept_with_fallback_label",
                    "fallback_label": parsed,
                    "raw_label": raw,
                })

            # Confidence is no longer a rejection condition. Keep low/medium/high as audit metadata only.
            conf = parsed.get("confidence", "low")
            if conf == "high":
                high_count += 1
            elif conf == "medium":
                medium_count += 1
            else:
                low_count += 1

            rec.setdefault("metadata", {})["light_llm_raw_label"] = raw
            rec["metadata"]["light_llm_parsed_label"] = parsed
            rec["metadata"]["light_llm_parse_status"] = parse_status
            rec["metadata"]["label_source"] = rec["metadata"].get("label_source", "") + "+light_llm"
            if merge_enabled:
                rec = merge_label_into_output(rec, parsed, min_confidence=min_confidence)
                rec["metadata"]["label_source"] += "_merged"
                merged_count += 1
            out[rec_idx] = rec
            included.append({
                "record_index": rec_idx,
                "event_id": rec.get("metadata", {}).get("event_id"),
                "source_type": rec.get("metadata", {}).get("source_type"),
                "parse_status": parse_status,
                "confidence": parsed.get("confidence", "low"),
                "parsed_label": parsed,
                "raw_label": raw,
            })

        completed = min(batch_start + len(batch_pairs), len(selected))
        if completed % progress_interval == 0 or completed == len(selected):
            elapsed = time.time() - start
            rate = completed / elapsed if elapsed > 0 else 0.0
            remaining = len(selected) - completed
            eta_min = (remaining / rate / 60.0) if rate > 0 else 0.0
            print(
                f"Progress: {completed}/{len(selected)} labeled ({completed / max(1, len(selected)):.1%}); "
                f"parsed={parsed_count}; parse_fallback={parse_fallback_count}; rejected_prelabel={len(rejected)}; "
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
    included_path = prepared / llm_cfg.get("included_file", "light_llm_included_labels.jsonl")
    fallback_path = prepared / llm_cfg.get("parse_fallback_file", "light_llm_parse_fallback_labels.jsonl")
    write_jsonl(skipped, skipped_path)
    write_jsonl(rejected, rejected_path)
    write_jsonl(included, included_path)
    write_jsonl(parse_fallbacks, fallback_path)

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
        "parse_fallback_count": parse_fallback_count,
        "included_count": len(included),
        "rejected_count": len(rejected),
        "rejection_policy": "Only records skipped before labeling for too-short/non-meaningful text are rejected. Low confidence and invalid JSON outputs are kept with audit metadata/fallback labels.",
        "included_file": str(included_path),
        "parse_fallback_file": str(fallback_path),
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
    print(f"Wrote {included_path}", flush=True)
    print(f"Wrote {fallback_path}", flush=True)
    print("Light LLM labeling finished", flush=True)


if __name__ == "__main__":
    main()
