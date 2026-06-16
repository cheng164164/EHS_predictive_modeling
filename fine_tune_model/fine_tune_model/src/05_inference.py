import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import cfg_path, load_config
from src_compat import format_messages_for_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/config.yaml')
    ap.add_argument('--text', required=True)
    ap.add_argument('--source_type', default='near_miss')
    ap.add_argument('--site', default='unknown')
    ap.add_argument('--department', default='unknown')
    args = ap.parse_args()
    cfg = load_config(args.config)
    train_cfg = cfg['training']
    model_path = cfg_path(cfg, 'paths.model_dir') / train_cfg.get('output_model_name', 'safety-risk-lora-demo')
    base_model = train_cfg['base_model']
    device = 'cuda' if torch.cuda.is_available() and not train_cfg.get('force_cpu', False) else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(model_path if model_path.exists() else base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16 if device == 'cuda' else torch.float32)
    model = PeftModel.from_pretrained(base, str(model_path)) if model_path.exists() else base
    model.to(device)
    model.eval()
    user_prompt = f'''Analyze the safety record and return only valid JSON.

Safety record:
source_type: {args.source_type}
site: {args.site}
department: {args.department}
description: {args.text}
'''
    messages = [
        {'role': 'system', 'content': 'You are a safety risk analysis assistant. Return concise, grounded, valid JSON.'},
        {'role': 'user', 'content': user_prompt},
    ]
    prompt = format_messages_for_eval(messages)
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=int(train_cfg.get('max_seq_length', 768))).to(device)
    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=300, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    print(tokenizer.decode(ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True))

if __name__ == '__main__':
    main()
