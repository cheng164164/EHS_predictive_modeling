# Safety Fine-Tuning Project

This version is configured for the new Azure ML GPU compute cluster `tan-dev-gpu`.

## Default model choices

- Light LLM label generation: `Qwen/Qwen2.5-1.5B-Instruct`
- QLoRA training: `Qwen/Qwen2.5-1.5B-Instruct`
- CPU fallback for light labeling: `google/flan-t5-base`
- CPU fallback for training: `Qwen/Qwen2.5-0.5B-Instruct`

You can override these in `configs/config.yaml`:

```yaml
labeling:
  light_llm:
    model_name: Qwen/Qwen2.5-1.5B-Instruct
    cpu_fallback_model: google/flan-t5-base
    max_records: 800

training:
  base_model: Qwen/Qwen2.5-1.5B-Instruct
  cpu_fallback_model: Qwen/Qwen2.5-0.5B-Instruct
```

## Light LLM label generation

`src/02_light_llm_label.py` now scans the train file and labels only meaningful records. It skips records that are too short, mostly metadata, test-like, or lack useful text.

Main controls:

```yaml
labeling:
  light_llm:
    max_records: 800              # number of meaningful records to label
    max_candidate_scan: 5000      # how many training records to scan to find meaningful ones
    min_meaningful_text_chars: 140
    min_alpha_words: 12
    merge_into_assistant_output: false
```

Outputs:

```text
outputs/prepared/safety_instruction_train_with_light_llm_raw.jsonl
outputs/prepared/light_llm_label_summary.json
outputs/prepared/light_llm_skipped_records.jsonl
outputs/prepared/light_llm_rejected_labels.jsonl
```

Do not set `merge_into_assistant_output: true` until you have reviewed the generated label quality.

## Submit Azure ML jobs

Default stage runs data preparation + light LLM labeling:

```bash
python scripts/submit_azureml_job.py
```

Training + evaluation:

```bash
python scripts/submit_azureml_job.py --stage train
```

Full pipeline:

```bash
python scripts/submit_azureml_job.py --stage all
```

Single stages:

```bash
python scripts/submit_azureml_job.py --stage 01
python scripts/submit_azureml_job.py --stage 02
python scripts/submit_azureml_job.py --stage 03
python scripts/submit_azureml_job.py --stage 04
```

## Training

The training script uses LoRA/QLoRA. With CUDA and `use_4bit: true`, it uses 4-bit QLoRA via bitsandbytes. Without CUDA, it falls back to standard LoRA on the configured CPU fallback model.

Outputs are saved under:

```text
outputs/model/safety-risk-qwen-lora/
outputs/model/training_summary.json
```
