import argparse
import json
import re
from pathlib import Path
from typing import Dict, Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import cfg_path, ensure_dir, load_config, read_jsonl, write_jsonl
from src_compat import format_messages_for_eval

REQUIRED_KEYS = [
    'event_summary', 'risk_pattern', 'hazard_tags', 'control_failure_tags',
    'potential_consequence', 'risk_level', 'recommended_actions',
    'escalation_recommended', 'recommended_review_group', 'evidence_phrases', 'limitations'
]


def extract_json(text: str) -> Dict[str, Any] | None:
    # Try to find the first JSON object in generated text.
    m = re.search(r'\{.*\}', text, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def load_model(cfg):
    train_cfg = cfg['training']
    model_path = cfg_path(cfg, 'paths.model_dir') / train_cfg.get('output_model_name', 'safety-risk-lora-demo')
    base_model = train_cfg['base_model']
    device = 'cuda' if torch.cuda.is_available() and not train_cfg.get('force_cpu', False) else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(model_path if model_path.exists() else base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16 if device == 'cuda' else torch.float32)
    if model_path.exists():
        model = PeftModel.from_pretrained(base, str(model_path))
    else:
        print('WARNING: fine-tuned adapter not found. Evaluating base model.')
        model = base
    model.to(device)
    model.eval()
    return model, tokenizer, device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/config.yaml')
    args = ap.parse_args()
    cfg = load_config(args.config)
    eval_dir = ensure_dir(cfg_path(cfg, 'paths.eval_dir'))
    test_records = read_jsonl(Path(cfg['paths']['prepared_dir']) / 'safety_instruction_test.jsonl')[: int(cfg['evaluation'].get('max_test_samples', 50))]
    model, tokenizer, device = load_model(cfg)
    predictions = []
    valid = 0
    schema_ok = 0
    for rec in test_records:
        prompt = format_messages_for_eval(rec['messages'][:2])
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=int(cfg['training'].get('max_seq_length', 768))).to(device)
        with torch.no_grad():
            ids = model.generate(
                **inputs,
                max_new_tokens=int(cfg['evaluation'].get('max_new_tokens', 300)),
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen = tokenizer.decode(ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        parsed = extract_json(gen)
        is_valid = parsed is not None
        has_schema = bool(is_valid and all(k in parsed for k in REQUIRED_KEYS))
        valid += int(is_valid)
        schema_ok += int(has_schema)
        predictions.append({
            'metadata': rec.get('metadata', {}),
            'expected': rec['messages'][2]['content'],
            'generated': gen,
            'json_valid': is_valid,
            'schema_ok': has_schema,
            'parsed': parsed,
        })
    report = {
        'n_tested': len(test_records),
        'json_valid_count': valid,
        'json_valid_rate': valid / max(1, len(test_records)),
        'schema_ok_count': schema_ok,
        'schema_ok_rate': schema_ok / max(1, len(test_records)),
    }
    write_jsonl(predictions, eval_dir / 'model_predictions.jsonl')
    with open(eval_dir / 'evaluation_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
