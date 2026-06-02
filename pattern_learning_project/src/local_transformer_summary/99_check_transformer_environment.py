#!/usr/bin/env python3
"""Check whether the local Hugging Face transformer summarizer can load.

Runs without command-line arguments. All settings are in config.py.
This script does not summarize the dataset. It only checks environment,
package availability, model loading, and a tiny manual-generate test.
"""
from __future__ import annotations

import traceback

try:
    from . import config
    from .transformer_summarizer import collect_transformer_environment, print_transformer_diagnostics, write_transformer_diagnostics
    from .utils import ensure_dir
except ImportError:  # pragma: no cover
    import config  # type: ignore
    from transformer_summarizer import collect_transformer_environment, print_transformer_diagnostics, write_transformer_diagnostics  # type: ignore
    from utils import ensure_dir  # type: ignore


def main() -> None:
    ensure_dir(config.LOG_DIR)
    report = collect_transformer_environment(load_torch=True)
    report["load_test_started"] = True
    print_transformer_diagnostics(report)

    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        print(f"Attempting to load model: {config.TRANSFORMER_MODEL_NAME}", flush=True)
        local_only = bool(getattr(config, "TRANSFORMER_LOCAL_FILES_ONLY", False))
        tokenizer = AutoTokenizer.from_pretrained(config.TRANSFORMER_MODEL_NAME, local_files_only=local_only)
        model = AutoModelForSeq2SeqLM.from_pretrained(config.TRANSFORMER_MODEL_NAME, local_files_only=local_only)

        requested_device = int(getattr(config, "TRANSFORMER_DEVICE", -1))
        if requested_device >= 0 and torch.cuda.is_available():
            device = torch.device(f"cuda:{requested_device}")
        else:
            device = torch.device("cpu")
        model.to(device)
        model.eval()

        print(f"Model loaded successfully on device={device}. Running tiny manual-generate test...", flush=True)
        prompt = (
            "Write a concise EHS review summary from the evidence only. "
            "Evidence: 2025-01-05 incident_1 - Forklift near miss occurred near a pedestrian walkway. "
            "2025-01-06 task_1 - Corrective action opened to improve traffic separation. Summary:"
        )
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            output_ids = model.generate(
                **encoded,
                max_new_tokens=80,
                min_new_tokens=10,
                num_beams=4,
                do_sample=False,
                early_stopping=True,
            )
        summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        report["load_test_success"] = True
        report["manual_generate_test"] = True
        report["test_summary_result"] = summary
        print("Test summary result:", summary, flush=True)

    except Exception as exc:
        report["load_test_success"] = False
        report["manual_generate_test"] = False
        report["error_type"] = type(exc).__name__
        report["error_message"] = str(exc)
        report["traceback"] = traceback.format_exc()
        print("Model load/generate test failed.", flush=True)
        print(f"Error type: {type(exc).__name__}", flush=True)
        print(f"Error message: {exc}", flush=True)
        print("Traceback:", flush=True)
        print(report["traceback"], flush=True)

    path = write_transformer_diagnostics(report, "transformer_environment_check.json")
    print(f"Wrote diagnostics: {path}", flush=True)


if __name__ == "__main__":
    main()
