#!/usr/bin/env python3
"""Check whether the local Hugging Face transformer summarizer can load.

Runs without command-line arguments. All settings are in config.py.
This script does not summarize the dataset. It only checks environment,
package availability, model loading, and a tiny test summary.
"""
from __future__ import annotations

import json
import traceback
from pathlib import Path

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
        from transformers import pipeline

        print(f"Attempting to load model: {config.TRANSFORMER_MODEL_NAME}", flush=True)
        # Compatibility fix:
        # Do not pass local_files_only=False through model_kwargs/tokenizer_kwargs.
        # With some transformers versions this creates a duplicate local_files_only
        # argument inside AutoConfig.from_pretrained().
        if bool(getattr(config, "TRANSFORMER_LOCAL_FILES_ONLY", False)):
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            print("TRANSFORMER_LOCAL_FILES_ONLY=True; loading from local cache only.", flush=True)
            tokenizer = AutoTokenizer.from_pretrained(config.TRANSFORMER_MODEL_NAME, local_files_only=True)
            model = AutoModelForSeq2SeqLM.from_pretrained(config.TRANSFORMER_MODEL_NAME, local_files_only=True)
            pipe = pipeline(
                task=config.TRANSFORMER_TASK,
                model=model,
                tokenizer=tokenizer,
                device=config.TRANSFORMER_DEVICE,
            )
        else:
            pipe = pipeline(
                task=config.TRANSFORMER_TASK,
                model=config.TRANSFORMER_MODEL_NAME,
                tokenizer=config.TRANSFORMER_MODEL_NAME,
                device=config.TRANSFORMER_DEVICE,
            )
        print("Model loaded successfully. Running tiny test summary...", flush=True)
        result = pipe(
            "A forklift near miss occurred near a pedestrian walkway. A corrective action was opened to improve traffic separation.",
            max_length=60,
            min_length=10,
            do_sample=False,
            truncation=True,
        )
        report["load_test_success"] = True
        report["test_summary_result"] = result
        print("Test summary result:", result, flush=True)

    except Exception as exc:
        report["load_test_success"] = False
        report["error_type"] = type(exc).__name__
        report["error_message"] = str(exc)
        report["traceback"] = traceback.format_exc()
        print("Model load test failed.", flush=True)
        print(f"Error type: {type(exc).__name__}", flush=True)
        print(f"Error message: {exc}", flush=True)
        print("Traceback:", flush=True)
        print(report["traceback"], flush=True)

    path = write_transformer_diagnostics(report, "transformer_environment_check.json")
    print(f"Wrote diagnostics: {path}", flush=True)


if __name__ == "__main__":
    main()
