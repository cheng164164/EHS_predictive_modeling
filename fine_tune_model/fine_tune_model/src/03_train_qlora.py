"""Train a LoRA/QLoRA adapter for safety risk instruction generation.

Default GPU path uses Qwen/Qwen2.5-1.5B-Instruct with 4-bit QLoRA.
If CUDA is unavailable, the script automatically falls back to a smaller CPU model
unless training.force_cpu=false and auto fallback is disabled.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments

from utils import cfg_path, ensure_dir, load_config, read_jsonl, set_seed

try:
    from transformers import BitsAndBytesConfig
except Exception:  # pragma: no cover
    BitsAndBytesConfig = None


def resolve_training_model(train_cfg: Dict[str, Any], use_cuda: bool) -> str:
    base_model = str(train_cfg.get("base_model") or "Qwen/Qwen2.5-1.5B-Instruct")
    if not use_cuda and bool(train_cfg.get("auto_fallback_to_cpu_model", True)):
        return str(train_cfg.get("cpu_fallback_model") or base_model)
    return base_model


def format_messages(messages: List[Dict[str, str]], tokenizer: Any | None = None) -> str:
    """Prefer model chat template when available; otherwise use a stable text format."""
    clean_messages = [{"role": m.get("role", "user"), "content": str(m.get("content", ""))} for m in messages]
    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(clean_messages, tokenize=False, add_generation_prompt=False)
        except Exception:
            pass
    parts = []
    for m in clean_messages:
        role = m["role"].upper()
        parts.append(f"### {role}:\n{m['content']}")
    parts.append("### END")
    return "\n\n".join(parts)


def tokenize_dataset(records: List[Dict[str, Any]], tokenizer: Any, max_seq_length: int) -> Dataset:
    texts = [format_messages(r["messages"], tokenizer) for r in records]
    ds = Dataset.from_dict({"text": texts})

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_seq_length, padding=False)

    return ds.map(tok, batched=True, remove_columns=["text"])


def infer_lora_targets(model) -> List[str]:
    names = []
    for name, module in model.named_modules():
        cls = module.__class__.__name__.lower()
        if "linear" in cls or "conv1d" in cls:
            leaf = name.split(".")[-1]
            if leaf not in names:
                names.append(leaf)
    preferred_order = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "c_attn", "c_proj"]
    preferred = [x for x in preferred_order if x in names]
    return preferred or names[-4:]


def build_training_args(model_dir: Path, train_cfg: Dict[str, Any], use_cuda: bool) -> TrainingArguments:
    requested_bf16 = bool(train_cfg.get("bf16", False))
    requested_fp16 = bool(train_cfg.get("fp16", True))

    bf16_supported = False
    if use_cuda:
        try:
            bf16_supported = bool(torch.cuda.is_bf16_supported())
        except Exception:
            bf16_supported = False

    # Only enable bf16 if explicitly requested and supported.
    bf16 = bool(use_cuda and requested_bf16 and bf16_supported)

    # Enable fp16 on CUDA unless bf16 is being used.
    fp16 = bool(use_cuda and requested_fp16 and not bf16)

    common = dict(
        output_dir=str(model_dir),
        num_train_epochs=float(train_cfg.get("num_train_epochs", 1)),
        per_device_train_batch_size=int(train_cfg.get("per_device_train_batch_size", 1)),
        per_device_eval_batch_size=int(train_cfg.get("per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 8)),
        learning_rate=float(train_cfg.get("learning_rate", 2e-4)),
        logging_steps=int(train_cfg.get("logging_steps", 50)),
        eval_steps=int(train_cfg.get("eval_steps", 200)),
        save_steps=int(train_cfg.get("save_steps", 200)),
        save_total_limit=2,
        report_to=[],
        fp16=fp16,
        bf16=bf16,
        remove_unused_columns=False,
        gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", True)) and use_cuda,
    )

    print(f"Mixed precision: fp16={fp16}, bf16={bf16}, bf16_supported={bf16_supported}")

    try:
        return TrainingArguments(eval_strategy="steps", **common)
    except TypeError:
        return TrainingArguments(evaluation_strategy="steps", **common)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    train_cfg = cfg["training"]
    prepared = cfg_path(cfg, "paths.prepared_dir")
    model_dir = ensure_dir(cfg_path(cfg, "paths.model_dir"))

    force_cpu = bool(train_cfg.get("force_cpu", False))
    cuda_available = torch.cuda.is_available()
    use_cuda = cuda_available and not force_cpu
    model_name = resolve_training_model(train_cfg, use_cuda)
    use_4bit_requested = bool(train_cfg.get("use_4bit", True))
    use_4bit = bool(use_4bit_requested and use_cuda and BitsAndBytesConfig is not None)

    print("Starting training", flush=True)
    print(f"  cuda_available: {cuda_available}", flush=True)
    if cuda_available:
        print(f"  gpu_name: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"  force_cpu: {force_cpu}", flush=True)
    print(f"  model_name: {model_name}", flush=True)
    print(f"  use_4bit: {use_4bit}", flush=True)

    if use_4bit_requested and not use_4bit:
        print("WARNING: 4-bit QLoRA requested but not available. Running standard LoRA without 4-bit quantization.", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        trust_remote_code=bool(train_cfg.get("trust_remote_code", False)),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    quant_cfg = None
    if use_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_cfg,
        device_map="auto" if use_cuda else None,
        torch_dtype=torch.float16 if use_cuda else torch.float32,
        trust_remote_code=bool(train_cfg.get("trust_remote_code", False)),
    )

    if bool(train_cfg.get("gradient_checkpointing", True)) and use_cuda:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    targets = infer_lora_targets(model)
    print("LoRA target modules:", targets, flush=True)
    lora_cfg = LoraConfig(
        r=int(train_cfg["lora"].get("r", 16)),
        lora_alpha=int(train_cfg["lora"].get("alpha", 32)),
        lora_dropout=float(train_cfg["lora"].get("dropout", 0.05)),
        target_modules=targets,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    train_file = str(train_cfg.get("train_file", "safety_instruction_train.jsonl"))
    val_file = str(train_cfg.get("val_file", "safety_instruction_val.jsonl"))
    train_path = prepared / train_file
    val_path = prepared / val_file
    print(f"  train_file: {train_path}", flush=True)
    print(f"  val_file:   {val_path}", flush=True)

    train_records = read_jsonl(train_path)
    val_records = read_jsonl(val_path)

    max_train_samples = train_cfg.get("max_train_samples", None)
    max_eval_samples = train_cfg.get("max_eval_samples", None)

    if max_train_samples is not None:
        train_records = train_records[: int(max_train_samples)]

    if max_eval_samples is not None:
        val_records = val_records[: int(max_eval_samples)]
    print(f"  train_records_used: {len(train_records)}", flush=True)
    print(f"  val_records_used: {len(val_records)}", flush=True)

    train_ds = tokenize_dataset(train_records, tokenizer, int(train_cfg.get("max_seq_length", 1536)))
    val_ds = tokenize_dataset(val_records, tokenizer, int(train_cfg.get("max_seq_length", 1536)))

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    args_tr = build_training_args(model_dir, train_cfg, use_cuda)
    trainer_kwargs = dict(
        model=model,
        args=args_tr,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    try:
        trainer = Trainer(
            **trainer_kwargs,
            processing_class=tokenizer,
        )
    except TypeError:
        trainer = Trainer(
            **trainer_kwargs,
            tokenizer=tokenizer,
        )


    trainer.train()

    adapter_dir = model_dir / train_cfg.get("output_model_name", "safety-risk-qwen-lora")
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    summary = {
        "base_model_configured": train_cfg.get("base_model"),
        "model_used": model_name,
        "cpu_fallback_model": train_cfg.get("cpu_fallback_model"),
        "cuda_available": cuda_available,
        "use_cuda": use_cuda,
        "use_4bit": use_4bit,
        "train_file": train_file,
        "val_file": val_file,
        "train_records_used": len(train_records),
        "val_records_used": len(val_records),
        "lora_targets": targets,
        "adapter_dir": str(adapter_dir),
    }
    with open(model_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("Training complete.", flush=True)
    print(f"Saved adapter to {adapter_dir}", flush=True)


if __name__ == "__main__":
    main()
