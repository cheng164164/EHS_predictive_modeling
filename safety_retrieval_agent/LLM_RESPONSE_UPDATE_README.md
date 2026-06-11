# LLM response layer update

This patch adds a local/free LLM response layer **after retrieval**. It does not modify these scripts:

- `scripts/00_build_unified_text_events.py`
- `scripts/00_prepare_knowledge_base.py`
- `scripts/01_build_faiss_indexes.py`
- `scripts/02_run_mvp_recommendations.py`

## Modified files

- `src/safety_retrieval_agent/config.py`
- `src/safety_retrieval_agent/agent.py`
- `requirements.txt`

## Added files

- `src/safety_retrieval_agent/local_llm.py`
- `LLM_RESPONSE_UPDATE_README.md`

## Default LLM

The default local LLM is:

```python
llm_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
```

It is loaded with Hugging Face `transformers`. The model is used only to organize the final response based on retrieved evidence. It does not rebuild embeddings or indexes.

## Main config controls

```python
enable_llm_response = True
llm_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
llm_max_input_tokens = 4096
llm_max_new_tokens = 700
llm_do_sample = False
llm_allow_heuristic_fallback = True
```

## Output

`SafetyRetrievalAgent.analyze_event(...)` now includes:

```json
"llm_final_response": {
  "status": "generated",
  "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
  "response_text": "...",
  "error": null,
  "prompt_chars": 12345
}
```

If the local model cannot load, the output status becomes `fallback_heuristic_after_llm_error` when `llm_allow_heuristic_fallback=True`.
