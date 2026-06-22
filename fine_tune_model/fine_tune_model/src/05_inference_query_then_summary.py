"""
05_inference.py

Config-driven stakeholder demo inference.

Core flow:
1. Build or load test cases from config/sample data.
2. Send only the configured query prompt to the fine-tuned model.
3. Save the model's raw structured output exactly as generated.
4. Parse JSON only for recording/reporting if possible; no hard-coded schema is enforced.
5. Send the structured output/raw output to the LLM again to generate a stakeholder-friendly summary.
6. Save JSONL, CSV, summary JSON, and Markdown report outputs.

Run with no arguments:
    python src/05_inference.py

Optional config override is supported but not required:
    python src/05_inference.py --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import cfg_path, ensure_dir, load_config, read_jsonl

try:
    from src_compat import format_messages_for_eval
except Exception:
    def format_messages_for_eval(messages: List[Dict[str, str]]) -> str:
        parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            parts.append(f"{role.upper()}: {content}")
        parts.append("ASSISTANT:")
        return "\n\n".join(parts)


SYNTHETIC_CASES = [
    {
        "case_id": "synthetic_001",
        "source_type": "near_miss",
        "site": "Demo Site",
        "department": "Warehouse",
        "description": "A forklift reversed out of a trailer while a pedestrian was walking through the loading dock aisle. The operator stopped after a coworker shouted. No contact occurred, but the pedestrian route was not clearly separated from forklift traffic.",
    },
    {
        "case_id": "synthetic_002",
        "source_type": "hazard_identification",
        "site": "Demo Site",
        "department": "Maintenance",
        "description": "An extension cord was found stretched across a wet work area near a portable welder. The plug connection was loose and the cable jacket was damaged. Employees were still using the equipment.",
    },
    {
        "case_id": "synthetic_003",
        "source_type": "near_miss",
        "site": "Demo Site",
        "department": "Operations",
        "description": "During a lift, a custom lifting bracket shifted and the suspended part dropped several inches onto the table. No one was injured, but employees were standing close to the load path.",
    },
    {
        "case_id": "synthetic_004",
        "source_type": "incident",
        "site": "Demo Site",
        "department": "Production",
        "description": "An employee cut their finger while opening a box with a utility knife. The blade slipped when the cardboard strap released. The employee was wearing gloves, but the task method did not control hand placement.",
    },
    {
        "case_id": "synthetic_005",
        "source_type": "hazard_identification",
        "site": "Demo Site",
        "department": "Housekeeping",
        "description": "Oil and water were observed on the floor near the parts washer. Several employees walked through the area before barricades were placed. No slip occurred, but traction was poor.",
    },
    {
        "case_id": "synthetic_006",
        "source_type": "audit",
        "site": "Demo Site",
        "department": "EHS",
        "description": "Audit found that lockout tags were available but the written energy isolation procedure was not posted at the machine. Operators were uncertain which disconnect controlled the conveyor motor.",
    },
    {
        "case_id": "synthetic_007",
        "source_type": "task",
        "site": "Demo Site",
        "department": "Facilities",
        "description": "Corrective action task opened to repair a missing guardrail section on an elevated platform. Temporary caution tape was installed, but employees continue to access the platform for daily checks.",
    },
    {
        "case_id": "synthetic_008",
        "source_type": "near_miss",
        "site": "Demo Site",
        "department": "Shipping",
        "description": "A pallet stacked above shoulder height leaned when moved by pallet jack. The top box fell to the floor near an employee. No injury occurred. The load was not wrapped and weight was unevenly distributed.",
    },
    {
        "case_id": "synthetic_009",
        "source_type": "hazard_identification",
        "site": "Demo Site",
        "department": "Laboratory",
        "description": "A chemical container was stored without a readable label and the secondary containment tray had residue. Employees were unsure whether the material was corrosive or solvent based.",
    },
    {
        "case_id": "synthetic_010",
        "source_type": "audit",
        "site": "Demo Site",
        "department": "Production",
        "description": "Machine guarding inspection found an interlock bypassed with tape during troubleshooting. The machine was still available for use and rotating parts were accessible when the panel was open.",
    },
]


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_text(value: Any, default: str = "unknown") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def read_jsonl_local(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        rows.append(obj)
                except Exception:
                    continue
    return rows


def write_jsonl_local(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort JSON extraction. This does not enforce any schema."""
    if not text:
        return None
    raw = text.strip()
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        try:
            obj = json.loads(fence.group(1))
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass

    start = raw.find("{")
    if start < 0:
        return None

    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(raw)):
        ch = raw[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = raw[start : i + 1]
                    try:
                        obj = json.loads(candidate)
                        return obj if isinstance(obj, dict) else None
                    except Exception:
                        return None
    return None


def json_compact(value: Any, max_chars: int = 30000) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, indent=2)
    except Exception:
        text = str(value)
    if len(text) > max_chars:
        return text[:max_chars] + "\n... [truncated]"
    return text


def render_template(template: str, case: Dict[str, Any], model_output: str = "", parsed_output: Optional[Dict[str, Any]] = None) -> str:
    parsed_text = json_compact(parsed_output) if parsed_output else ""
    values = {
        "case_id": safe_text(case.get("case_id"), "unknown"),
        "source_type": safe_text(case.get("source_type"), "unknown"),
        "site": safe_text(case.get("site"), "unknown"),
        "department": safe_text(case.get("department"), "unknown"),
        "description": safe_text(case.get("description"), ""),
        "query": safe_text(case.get("description"), ""),
        "model_output": model_output or "",
        "structured_output": parsed_text or model_output or "",
        "parsed_output": parsed_text,
    }
    try:
        return template.format(**values)
    except KeyError as e:
        raise KeyError(f"Prompt template uses unknown placeholder {e}. Supported placeholders: {sorted(values.keys())}")


def extract_field_from_prompt(user_text: str, field_name: str, default: str = "unknown") -> str:
    """Extract one metadata field from the prepared user prompt.

    Prepared records store site/department inside the user message created by
    01_prepare_data.py. Older prepared JSONL files do not include those fields
    in metadata, so inference sampling must recover them from the prompt.
    """
    if not user_text:
        return default
    pattern = rf"(?im)^\s*{re.escape(field_name)}\s*:\s*(.*?)\s*$"
    match = re.search(pattern, user_text)
    if not match:
        return default
    return safe_text(match.group(1), default)


def get_case_from_prepared_row(row: Dict[str, Any], fallback_id: str) -> Optional[Dict[str, Any]]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    messages = row.get("messages") if isinstance(row.get("messages"), list) else []
    user_text = ""
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "user":
            user_text = safe_text(m.get("content"), "")
            break
    if not user_text:
        return None

    # Try to recover the actual description from the prepared user prompt.
    desc_match = re.search(r"(?im)^\s*description\s*:\s*(.*)", user_text, flags=re.DOTALL)
    description = desc_match.group(1).strip() if desc_match else user_text
    description = re.sub(r"\n+\s*Return .*", "", description, flags=re.IGNORECASE | re.DOTALL).strip()
    if len(description) < 20:
        description = user_text

    # Backward-compatible metadata recovery:
    # - Newer prepared rows may have these fields directly in metadata.
    # - Older prepared rows usually have them only in the user prompt.
    case_id = safe_text(
        metadata.get("event_id")
        or metadata.get("record_id")
        or extract_field_from_prompt(user_text, "event_id", ""),
        fallback_id,
    )
    source_type = safe_text(
        metadata.get("source_type") or extract_field_from_prompt(user_text, "source_type", ""),
        "unknown",
    )
    site = safe_text(
        metadata.get("site")
        or metadata.get("location")
        or extract_field_from_prompt(user_text, "site", ""),
        "unknown",
    )
    department = safe_text(
        metadata.get("department")
        or metadata.get("dept")
        or extract_field_from_prompt(user_text, "department", ""),
        "unknown",
    )

    return {
        "case_id": case_id,
        "source_type": source_type,
        "site": site,
        "department": department,
        "description": description,
        "sample_source": "prepared",
    }


def sample_from_prepared(cfg: Dict[str, Any], n: int, seed: int) -> List[Dict[str, Any]]:
    prepared_dir = cfg_path(cfg, "paths.prepared_dir")
    candidates: List[Dict[str, Any]] = []
    for fname in ["safety_instruction_test.jsonl", "safety_instruction_val.jsonl", "safety_instruction_train.jsonl"]:
        path = prepared_dir / fname
        if not path.exists():
            continue
        rows = read_jsonl_local(path)
        for idx, row in enumerate(rows):
            case = get_case_from_prepared_row(row, f"prepared_{idx:05d}")
            if case:
                candidates.append(case)
        if len(candidates) >= n:
            break
    rnd = random.Random(seed)
    rnd.shuffle(candidates)
    return candidates[:n]


def sample_from_original_csv(cfg: Dict[str, Any], n: int, seed: int) -> List[Dict[str, Any]]:
    path = cfg_path(cfg, "paths.input_csv")
    if not path.exists():
        return []

    try:
        import pandas as pd
    except Exception:
        return []

    try:
        df = pd.read_csv(path, compression="infer", nrows=200000)
    except Exception:
        try:
            opener = gzip.open if path.suffix == ".gz" else open
            with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
                df = pd.read_csv(f, nrows=200000)
        except Exception:
            return []

    text_cols = [c for c in df.columns if any(x in c.lower() for x in ["description", "text", "summary", "title", "event"])]
    if not text_cols:
        return []

    def first_existing(row: Any, names: List[str], default: str = "unknown") -> str:
        for name in names:
            for col in df.columns:
                if col.lower() == name.lower() and safe_text(row.get(col), ""):
                    return safe_text(row.get(col), default)
        return default

    cases: List[Dict[str, Any]] = []
    for idx, row in df.iterrows():
        parts = [safe_text(row.get(c), "") for c in text_cols]
        desc = " | ".join([p for p in parts if p and p != "unknown"])
        if len(desc) < 80:
            continue
        source_type = first_existing(row, ["source_type", "source", "record_type"], "unknown")
        site = first_existing(row, ["site", "location", "facility"], "unknown")
        dept = first_existing(row, ["department", "dept", "area"], "unknown")
        event_id = first_existing(row, ["event_id", "id", "record_id"], f"csv_{idx:05d}")
        cases.append({
            "case_id": event_id,
            "source_type": source_type,
            "site": site,
            "department": dept,
            "description": desc[:4000],
            "sample_source": "original_csv",
        })

    rnd = random.Random(seed)
    rnd.shuffle(cases)
    return cases[:n]


def read_cases_file(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"queries_file not found: {path}")
    suffix = path.suffix.lower()
    cases: List[Dict[str, Any]] = []

    if suffix == ".txt":
        with open(path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                text = line.strip()
                if text:
                    cases.append({
                        "case_id": f"manual_txt_{idx + 1:03d}",
                        "source_type": "unknown",
                        "site": "unknown",
                        "department": "unknown",
                        "description": text,
                        "sample_source": "queries_file",
                    })
        return cases

    if suffix == ".jsonl":
        rows = read_jsonl_local(path)
        for idx, row in enumerate(rows):
            desc = safe_text(row.get("description") or row.get("text") or row.get("query"), "")
            if desc:
                cases.append({
                    "case_id": safe_text(row.get("case_id") or row.get("event_id"), f"manual_jsonl_{idx + 1:03d}"),
                    "source_type": safe_text(row.get("source_type"), "unknown"),
                    "site": safe_text(row.get("site"), "unknown"),
                    "department": safe_text(row.get("department"), "unknown"),
                    "description": desc,
                    "sample_source": "queries_file",
                })
        return cases

    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data if isinstance(data, list) else data.get("cases", []) if isinstance(data, dict) else []
        for idx, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            desc = safe_text(row.get("description") or row.get("text") or row.get("query"), "")
            if desc:
                cases.append({
                    "case_id": safe_text(row.get("case_id") or row.get("event_id"), f"manual_json_{idx + 1:03d}"),
                    "source_type": safe_text(row.get("source_type"), "unknown"),
                    "site": safe_text(row.get("site"), "unknown"),
                    "department": safe_text(row.get("department"), "unknown"),
                    "description": desc,
                    "sample_source": "queries_file",
                })
        return cases

    if suffix == ".csv":
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                desc = safe_text(row.get("description") or row.get("text") or row.get("query"), "")
                if desc:
                    cases.append({
                        "case_id": safe_text(row.get("case_id") or row.get("event_id"), f"manual_csv_{idx + 1:03d}"),
                        "source_type": safe_text(row.get("source_type"), "unknown"),
                        "site": safe_text(row.get("site"), "unknown"),
                        "department": safe_text(row.get("department"), "unknown"),
                        "description": desc,
                        "sample_source": "queries_file",
                    })
        return cases

    raise ValueError(f"Unsupported queries_file type: {path}")


def build_cases(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    inf = cfg.get("inference", {}) or {}
    mode = str(inf.get("mode", "auto")).lower()
    n = int(inf.get("auto_cases", 10))
    seed = int(inf.get("seed", cfg.get("seed", 42)))

    if mode == "single":
        case = dict(inf.get("single_case") or {})
        if not safe_text(case.get("description"), ""):
            raise ValueError("inference.mode is single, but inference.single_case.description is empty.")
        case.setdefault("case_id", "manual_single_001")
        case.setdefault("source_type", "unknown")
        case.setdefault("site", "unknown")
        case.setdefault("department", "unknown")
        case["sample_source"] = "single_case"
        return [case]

    if mode == "manual":
        rows = inf.get("manual_cases") or []
        cases: List[Dict[str, Any]] = []
        for idx, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            desc = safe_text(row.get("description") or row.get("text") or row.get("query"), "")
            if not desc:
                continue
            cases.append({
                "case_id": safe_text(row.get("case_id"), f"manual_{idx + 1:03d}"),
                "source_type": safe_text(row.get("source_type"), "unknown"),
                "site": safe_text(row.get("site"), "unknown"),
                "department": safe_text(row.get("department"), "unknown"),
                "description": desc,
                "sample_source": "manual_cases",
            })
        if not cases:
            raise ValueError("inference.mode is manual, but no valid inference.manual_cases were found.")
        return cases

    if mode == "queries_file":
        qf = inf.get("queries_file")
        if not qf:
            raise ValueError("inference.mode is queries_file, but inference.queries_file is empty.")
        path = Path(qf)
        if not path.is_absolute():
            path = cfg_path(cfg, "paths.output_dir").parent / path
        return read_cases_file(path)

    if mode != "auto":
        raise ValueError(f"Unsupported inference.mode: {mode}")

    sample_source = str(inf.get("sample_source", "auto")).lower()
    cases: List[Dict[str, Any]] = []
    if sample_source in {"auto", "prepared"}:
        cases = sample_from_prepared(cfg, n, seed)
    if len(cases) < n and sample_source in {"auto", "original", "csv", "original_csv"}:
        cases.extend(sample_from_original_csv(cfg, n - len(cases), seed + 1))
    if len(cases) < n and sample_source in {"auto", "synthetic"}:
        rnd = random.Random(seed)
        synth = [dict(c) for c in SYNTHETIC_CASES]
        rnd.shuffle(synth)
        cases.extend(synth[: n - len(cases)])
    if not cases:
        raise RuntimeError("No inference cases were available. Check prepared data, input_csv, or manual_cases.")
    return cases[:n]


def load_model_and_tokenizer(cfg: Dict[str, Any]):
    train_cfg = cfg.get("training", {}) or {}
    model_dir = cfg_path(cfg, "paths.model_dir")
    adapter_name = train_cfg.get("output_model_name", "safety-risk-qwen-lora")
    adapter_path = model_dir / adapter_name
    base_model = train_cfg.get("base_model", "Qwen/Qwen2.5-1.5B-Instruct")
    trust_remote_code = bool(train_cfg.get("trust_remote_code", False))

    device = "cuda" if torch.cuda.is_available() and not bool(train_cfg.get("force_cpu", False)) else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    tokenizer_source = str(adapter_path) if adapter_path.exists() else base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
        device_map=None,
    )
    if adapter_path.exists():
        model = PeftModel.from_pretrained(base, str(adapter_path))
        adapter_loaded = True
    else:
        model = base
        adapter_loaded = False
    model.to(device)
    model.eval()
    return model, tokenizer, device, str(adapter_path), adapter_loaded


def generate_text(
    model: Any,
    tokenizer: Any,
    device: str,
    messages: List[Dict[str, str]],
    max_input_tokens: int,
    max_new_tokens: int,
    temperature: float,
) -> str:
    prompt = format_messages_for_eval(messages)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
    ).to(device)
    do_sample = temperature > 0
    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)
    new_ids = output_ids[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def output_dir_for_run(cfg: Dict[str, Any]) -> Path:
    inf = cfg.get("inference", {}) or {}
    base = inf.get("output_dir")
    if base:
        out_base = Path(base)
        if not out_base.is_absolute():
            out_base = cfg_path(cfg, "paths.output_dir").parent / out_base
    else:
        out_base = cfg_path(cfg, "paths.eval_dir")
    run_name = inf.get("run_name") or f"stakeholder_demo_{now_stamp()}"
    return ensure_dir(out_base / run_name)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_id",
        "source_type",
        "site",
        "department",
        "sample_source",
        "description",
        "model_output_json_valid",
        "model_output_raw",
        "model_output_parsed_json",
        "stakeholder_summary",
    ]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "case_id": row.get("case_id"),
                "source_type": row.get("source_type"),
                "site": row.get("site"),
                "department": row.get("department"),
                "sample_source": row.get("sample_source"),
                "description": row.get("description"),
                "model_output_json_valid": row.get("model_output_json_valid"),
                "model_output_raw": row.get("model_output_raw"),
                "model_output_parsed_json": json_compact(row.get("model_output_parsed_json")) if row.get("model_output_parsed_json") else "",
                "stakeholder_summary": row.get("stakeholder_summary"),
            })


def write_markdown(path: Path, rows: List[Dict[str, Any]], metadata: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("# Stakeholder Demo Inference Results")
    lines.append("")
    lines.append("## Run Metadata")
    lines.append("")
    for k, v in metadata.items():
        lines.append(f"- **{k}:** {v}")
    lines.append("")
    lines.append("## Cases")
    for idx, row in enumerate(rows, start=1):
        lines.append("")
        lines.append(f"### Case {idx}: {row.get('case_id')}")
        lines.append("")
        lines.append(f"- **Source type:** {row.get('source_type')}")
        lines.append(f"- **Site:** {row.get('site')}")
        lines.append(f"- **Department:** {row.get('department')}")
        lines.append(f"- **JSON valid:** {row.get('model_output_json_valid')}")
        lines.append("")
        lines.append("**Input query / safety event**")
        lines.append("")
        lines.append(row.get("description", ""))
        lines.append("")
        lines.append("**Model structured output, raw**")
        lines.append("")
        lines.append("```json")
        raw = row.get("model_output_raw", "")
        parsed = row.get("model_output_parsed_json")
        lines.append(json_compact(parsed) if parsed else raw)
        lines.append("```")
        lines.append("")
        lines.append("**Stakeholder-friendly summary**")
        lines.append("")
        lines.append(row.get("stakeholder_summary", ""))
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--config", default="configs/config.yaml", help="Optional. Defaults to configs/config.yaml.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    inf = cfg.get("inference", {}) or {}

    cases = build_cases(cfg)
    out_dir = output_dir_for_run(cfg)

    print(f"Loaded {len(cases)} inference case(s).")
    print(f"Output directory: {out_dir}")

    model, tokenizer, device, adapter_path, adapter_loaded = load_model_and_tokenizer(cfg)
    print(f"Device: {device}")
    print(f"Adapter path: {adapter_path}")
    print(f"Adapter loaded: {adapter_loaded}")

    # These prompts should be set in config.yaml. Fallbacks are intentionally minimal and do not define a schema.
    model_prompt_template = inf.get("model_prompt_template") or (
        "Analyze the following safety event.\n\n"
        "source_type: {source_type}\n"
        "site: {site}\n"
        "department: {department}\n"
        "description: {description}\n"
    )
    model_system_prompt = inf.get("model_system_prompt") or "You are a safety risk analysis assistant."

    summary_prompt_template = inf.get("summary_prompt_template") or (
        "Create a concise stakeholder-friendly summary from the model output below. "
        "Focus on the main risk, why it matters, and practical prevention actions. "
        "Do not add facts that are not supported by the input.\n\n"
        "Original safety event:\n{description}\n\n"
        "Model output:\n{structured_output}\n"
    )
    summary_system_prompt = inf.get("summary_system_prompt") or "You summarize safety analysis results for business stakeholders."

    max_input_tokens = int(inf.get("max_input_tokens", cfg.get("training", {}).get("max_seq_length", 1536)))
    max_new_tokens = int(inf.get("max_new_tokens", cfg.get("evaluation", {}).get("max_new_tokens", 800)))
    temperature = float(inf.get("temperature", cfg.get("evaluation", {}).get("temperature", 0.1)))
    generate_summary = bool(inf.get("generate_stakeholder_summary", True))
    summary_max_new_tokens = int(inf.get("summary_max_new_tokens", 300))
    summary_temperature = float(inf.get("summary_temperature", 0.1))

    results: List[Dict[str, Any]] = []
    for idx, case in enumerate(cases, start=1):
        print(f"[{idx}/{len(cases)}] Running case_id={case.get('case_id')}", flush=True)

        model_user_prompt = render_template(model_prompt_template, case)
        model_messages = [
            {"role": "system", "content": model_system_prompt},
            {"role": "user", "content": model_user_prompt},
        ]
        raw_output = generate_text(
            model=model,
            tokenizer=tokenizer,
            device=device,
            messages=model_messages,
            max_input_tokens=max_input_tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        parsed_output = extract_json_object(raw_output)

        stakeholder_summary = ""
        if generate_summary:
            summary_user_prompt = render_template(
                summary_prompt_template,
                case,
                model_output=raw_output,
                parsed_output=parsed_output,
            )
            summary_messages = [
                {"role": "system", "content": summary_system_prompt},
                {"role": "user", "content": summary_user_prompt},
            ]
            stakeholder_summary = generate_text(
                model=model,
                tokenizer=tokenizer,
                device=device,
                messages=summary_messages,
                max_input_tokens=max_input_tokens,
                max_new_tokens=summary_max_new_tokens,
                temperature=summary_temperature,
            )

        results.append({
            "case_id": case.get("case_id"),
            "source_type": case.get("source_type"),
            "site": case.get("site"),
            "department": case.get("department"),
            "sample_source": case.get("sample_source"),
            "description": case.get("description"),
            "model_prompt": model_user_prompt,
            "summary_prompt": summary_user_prompt if generate_summary else "",
            "model_output_raw": raw_output,
            "model_output_parsed_json": parsed_output,
            "model_output_json_valid": parsed_output is not None,
            "stakeholder_summary": stakeholder_summary,
        })

    metadata = {
        "run_timestamp": now_stamp(),
        "n_cases": len(cases),
        "json_valid_count": sum(1 for r in results if r.get("model_output_json_valid")),
        "json_valid_rate": round(sum(1 for r in results if r.get("model_output_json_valid")) / max(len(results), 1), 4),
        "base_model": cfg.get("training", {}).get("base_model"),
        "adapter_path": adapter_path,
        "adapter_loaded": adapter_loaded,
        "max_new_tokens": max_new_tokens,
        "summary_max_new_tokens": summary_max_new_tokens,
    }

    if bool(inf.get("write_jsonl", True)):
        write_jsonl_local(out_dir / "stakeholder_demo_results.jsonl", results)
    if bool(inf.get("write_csv", True)):
        write_csv(out_dir / "stakeholder_demo_results.csv", results)
    if bool(inf.get("write_summary_json", True)):
        (out_dir / "stakeholder_demo_summary.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    if bool(inf.get("write_markdown_report", True)):
        write_markdown(out_dir / "stakeholder_demo_report.md", results, metadata)

    print("Done.")
    print(f"JSON valid: {metadata['json_valid_count']}/{metadata['n_cases']} ({metadata['json_valid_rate']})")
    print(f"Results: {out_dir}")


if __name__ == "__main__":
    main()
