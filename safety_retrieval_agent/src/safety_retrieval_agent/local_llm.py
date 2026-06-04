"""Local/free LLM response generation for the Safety Retrieval Agent.

This module runs after retrieval. It does not change embeddings, FAISS indexes,
BM25 indexes, or query matching. Its only job is to turn the structured retrieval
result into a concise, readable response that remains grounded in returned
historical evidence.

Default local model:
    Qwen/Qwen2.5-0.5B-Instruct

The model is small enough for simple local tests and can be swapped in config.py.
For better response quality, set llm_model_name to a larger instruct model such
as Qwen/Qwen2.5-1.5B-Instruct if your machine can support it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import Settings
from .utils import clean_text_value, preview


@dataclass
class LLMGenerationResult:
    """Structured return object for local LLM generation."""

    status: str
    model_name: str | None
    response_text: str
    error: str | None = None
    prompt_chars: int | None = None


class LocalLLMResponder:
    """Generate a final human-readable response from retrieval evidence.

    The class loads the local model lazily on first generation so importing the
    agent is still lightweight. It uses Hugging Face Transformers only. No cloud
    service is required.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model_name = str(getattr(settings, "llm_model_name", "Qwen/Qwen2.5-0.5B-Instruct"))
        self.tokenizer = None
        self.model = None
        self.load_error: str | None = None

    def generate(self, analysis: dict[str, Any]) -> LLMGenerationResult:
        """Generate the final response text.

        If the local model cannot be loaded or generation fails, the method can
        return a deterministic fallback summary when llm_allow_heuristic_fallback
        is enabled. This keeps batch scripts from crashing unexpectedly while
        making the LLM failure visible in the output JSON.
        """
        if not bool(getattr(self.settings, "enable_llm_response", True)):
            return LLMGenerationResult(
                status="disabled",
                model_name=None,
                response_text=self._heuristic_response(analysis),
                error=None,
                prompt_chars=0,
            )

        prompt = self._build_prompt(analysis)
        try:
            self._ensure_model_loaded()
            response = self._generate_text(prompt)
            response = clean_text_value(response)
            if not response:
                raise RuntimeError("The local LLM returned an empty response.")
            return LLMGenerationResult(
                status="generated",
                model_name=self.model_name,
                response_text=response,
                error=None,
                prompt_chars=len(prompt),
            )
        except Exception as exc:  # pragma: no cover - environment/model dependent
            error = repr(exc)
            if bool(getattr(self.settings, "llm_allow_heuristic_fallback", True)):
                return LLMGenerationResult(
                    status="fallback_heuristic_after_llm_error",
                    model_name=self.model_name,
                    response_text=self._heuristic_response(analysis),
                    error=error,
                    prompt_chars=len(prompt),
                )
            raise

    def _ensure_model_loaded(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Local LLM response generation requires torch and transformers. "
                "Install/update dependencies with: pip install -r requirements.txt"
            ) from exc

        tokenizer_kwargs = {}
        if bool(getattr(self.settings, "llm_trust_remote_code", True)):
            tokenizer_kwargs["trust_remote_code"] = True
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, **tokenizer_kwargs)

        model_kwargs: dict[str, Any] = {}
        if bool(getattr(self.settings, "llm_trust_remote_code", True)):
            model_kwargs["trust_remote_code"] = True

        dtype_setting = str(getattr(self.settings, "llm_torch_dtype", "auto") or "auto").strip().lower()
        if dtype_setting == "auto":
            model_kwargs["torch_dtype"] = "auto"
        elif dtype_setting in {"float16", "fp16"}:
            model_kwargs["torch_dtype"] = torch.float16
        elif dtype_setting in {"bfloat16", "bf16"}:
            model_kwargs["torch_dtype"] = torch.bfloat16
        elif dtype_setting in {"float32", "fp32"}:
            model_kwargs["torch_dtype"] = torch.float32

        device_map = str(getattr(self.settings, "llm_device_map", "auto") or "auto").strip()
        if device_map:
            model_kwargs["device_map"] = device_map

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        self.model.eval()

    def _generate_text(self, prompt: str) -> str:
        import torch

        max_input_tokens = int(getattr(self.settings, "llm_max_input_tokens", 4096))
        max_new_tokens = int(getattr(self.settings, "llm_max_new_tokens", 700))
        temperature = float(getattr(self.settings, "llm_temperature", 0.2))
        do_sample = bool(getattr(self.settings, "llm_do_sample", False))

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an EHS safety prevention assistant. Use only the provided historical evidence. "
                    "Do not invent incident details, corrective actions, sites, or event IDs. "
                    "Be concise, practical, and clear. If evidence is weak or missing, say so."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        if hasattr(self.tokenizer, "apply_chat_template"):
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                truncation=True,
                max_length=max_input_tokens,
            )
            input_ids = input_ids.to(self.model.device)
            attention_mask = torch.ones_like(input_ids)
        else:
            full_prompt = (
                "System: You are an EHS safety prevention assistant. Use only the provided historical evidence.\n\n"
                f"User:\n{prompt}\n\nAssistant:"
            )
            encoded = self.tokenizer(
                full_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_input_tokens,
            )
            input_ids = encoded["input_ids"].to(self.model.device)
            attention_mask = encoded.get("attention_mask")
            attention_mask = attention_mask.to(self.model.device) if attention_mask is not None else None

        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else eos_token_id

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": pad_token_id,
            "eos_token_id": eos_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                **generation_kwargs,
            )
        new_tokens = output_ids[0, input_ids.shape[-1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def _build_prompt(self, analysis: dict[str, Any]) -> str:
        max_evidence = int(getattr(self.settings, "llm_max_evidence_records_per_section", 5))
        query = analysis.get("query", {}) or {}
        theme = analysis.get("risk_pattern_classification", {}) or {}
        severe = analysis.get("historical_severe_injury_similarity", {}) or {}
        injuries = analysis.get("historical_injury_similarity", {}) or {}
        recall = analysis.get("similar_historical_event_recall", {}) or {}
        actions = analysis.get("recommended_prevention_actions", []) or []
        risk_factors = analysis.get("risk_factor_extraction", []) or []
        missing = analysis.get("missing_information_prompt", []) or []

        parts: list[str] = []
        parts.append("Create a final response for a safety form-entry assistant.")
        parts.append("The response should be readable, coherent, concise, and evidence-backed.")
        parts.append("Use these exact section headings:")
        parts.append(
            "1. Detected pattern\n"
            "2. Why this may matter\n"
            "3. Similar historical evidence\n"
            "4. Risk factors and possible control gaps\n"
            "5. Suggested prevention actions\n"
            "6. Missing information to collect\n"
            "7. Evidence IDs"
        )
        parts.append("Do not claim an injury will happen. Say 'historically similar' rather than 'predicted'.")
        parts.append("Do not call historical actions proven effective unless the evidence explicitly says that.")
        parts.append("Keep the full response under about 450 words. Use short bullets.")

        parts.append("\nNEW REPORT")
        parts.append(f"Event ID: {query.get('event_id') or 'manual/new report'}")
        parts.append(f"Source type: {query.get('source_type') or 'not provided'}")
        parts.append(f"Site: {query.get('site') or 'not provided'}")
        parts.append(f"Department: {query.get('department') or 'not provided'}")
        parts.append(f"Description: {query.get('text_preview') or ''}")

        parts.append("\nDETECTED THEME")
        parts.append(f"Theme ID: {theme.get('risk_theme_id') or 'unknown'}")
        parts.append(f"Theme name: {theme.get('risk_theme_name') or 'Unknown theme'}")
        parts.append(f"Classification method: {theme.get('classification_method') or 'not available'}")

        parts.append("\nSEVERE INJURY SIMILARITY")
        parts.append(f"Similarity band: {severe.get('similarity_band') or 'no_match'}")
        parts.append(f"Top score: {severe.get('top_score')}")
        parts.append(f"Top FAISS cosine score: {severe.get('top_faiss_cosine_score')}")
        parts.append(self._format_match_section("Severe injury matches", severe.get("matches", []), max_evidence))
        parts.append(self._format_match_section("All injury matches", injuries.get("matches", []), max_evidence))
        parts.append(self._format_match_section("Similar historical events", recall.get("matches", []), max_evidence))

        parts.append("\nEXTRACTED RISK FACTOR CANDIDATES")
        if risk_factors:
            for item in risk_factors[:12]:
                if isinstance(item, dict):
                    parts.append(f"- {item.get('risk_factor')}")
                else:
                    parts.append(f"- {item}")
        else:
            parts.append("- No risk-factor phrases were extracted by the local heuristic.")

        parts.append("\nHISTORICAL ACTION / SAFE PRACTICE CANDIDATES")
        if actions:
            for item in actions[: int(getattr(self.settings, "llm_max_action_candidates", 8))]:
                if isinstance(item, dict):
                    parts.append(
                        f"- Evidence {item.get('supporting_event_id')}: {preview(item.get('recommendation'), 260)}"
                    )
                else:
                    parts.append(f"- {preview(item, 260)}")
        else:
            parts.append("- No related corrective-action or safe-practice records were returned.")

        parts.append("\nMISSING INFORMATION CANDIDATES FROM RULE CHECK")
        if missing:
            for item in missing[: int(getattr(self.settings, "llm_max_missing_info_prompts", 8))]:
                if isinstance(item, dict):
                    parts.append(f"- {item.get('missing_area')}: {item.get('prompt')}")
                else:
                    parts.append(f"- {item}")
        else:
            parts.append("- The rule-based check did not flag obvious missing information.")

        return "\n".join(parts)

    def _format_match_section(self, title: str, matches: list[dict[str, Any]], max_evidence: int) -> str:
        lines = [f"\n{title.upper()}"]
        if not matches:
            lines.append("- None returned.")
            return "\n".join(lines)
        for item in matches[:max_evidence]:
            event_id = item.get("event_id") or item.get("source_id") or "unknown_id"
            role = item.get("source_role") or item.get("source_type") or "unknown_role"
            site = item.get("site") or "unknown site"
            title_text = clean_text_value(item.get("title") or item.get("description") or item.get("retrieval_text") or "")
            score_bits = []
            for key in ["faiss_score", "bm25_score", "hybrid_score", "similarity_score"]:
                value = item.get(key)
                if value is not None:
                    try:
                        score_bits.append(f"{key}={float(value):.4f}")
                    except Exception:
                        pass
            score_text = "; ".join(score_bits[:3])
            lines.append(f"- {event_id} | {role} | {site} | {score_text} | {preview(title_text, 220)}")
        return "\n".join(lines)

    def _heuristic_response(self, analysis: dict[str, Any]) -> str:
        """Deterministic fallback response when LLM generation is unavailable."""
        query = analysis.get("query", {}) or {}
        theme = analysis.get("risk_pattern_classification", {}) or {}
        severe = analysis.get("historical_severe_injury_similarity", {}) or {}
        recall = analysis.get("similar_historical_event_recall", {}) or {}
        actions = analysis.get("recommended_prevention_actions", []) or []
        missing = analysis.get("missing_information_prompt", []) or []

        lines = []
        lines.append("1. Detected pattern")
        lines.append(f"- {theme.get('risk_theme_name') or 'Unknown theme'}")
        lines.append("\n2. Why this may matter")
        lines.append(
            f"- The report was compared with historical safety records. Severe-injury similarity band: {severe.get('similarity_band') or 'no_match'}."
        )
        lines.append("\n3. Similar historical evidence")
        severe_matches = severe.get("matches", []) or []
        event_matches = recall.get("matches", []) or []
        lines.append(f"- Severe injury matches returned: {len(severe_matches)}")
        lines.append(f"- Similar historical events returned: {len(event_matches)}")
        lines.append("\n4. Risk factors and possible control gaps")
        for item in (analysis.get("risk_factor_extraction", []) or [])[:6]:
            if isinstance(item, dict):
                lines.append(f"- {item.get('risk_factor')}")
        lines.append("\n5. Suggested prevention actions")
        if actions:
            for item in actions[:5]:
                if isinstance(item, dict):
                    lines.append(f"- {item.get('recommendation')} (evidence: {item.get('supporting_event_id')})")
        else:
            lines.append("- No related historical corrective-action evidence was returned.")
        lines.append("\n6. Missing information to collect")
        if missing:
            for item in missing[:6]:
                if isinstance(item, dict):
                    lines.append(f"- {item.get('prompt')}")
        else:
            lines.append("- No obvious missing information was flagged by the rule-based check.")
        lines.append("\n7. Evidence IDs")
        ids = []
        for section in [severe_matches, event_matches]:
            for item in section[:5]:
                ids.append(str(item.get("event_id") or ""))
        ids = [x for x in ids if x]
        lines.append("- " + (", ".join(ids) if ids else "No evidence IDs returned."))
        return "\n".join(lines)
