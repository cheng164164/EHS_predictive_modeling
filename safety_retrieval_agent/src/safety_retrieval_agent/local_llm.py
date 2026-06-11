"""Local/free LLM response generation for the Safety Retrieval Agent.

This module runs after retrieval. It does not change embeddings, FAISS indexes,
BM25 indexes, or query matching. Its job is to turn the structured retrieval
result into a concise, business-readable response grounded in returned evidence.

Default local model:
    Qwen/Qwen2.5-0.5B-Instruct
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
            response = self._postprocess_response(response)
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

        system_text = (
            "You are an EHS safety prevention assistant. Use only the provided historical evidence. "
            "Do not invent incident details, corrective actions, sites, or event IDs. "
            "Do not claim that an injury will or will not occur. "
            "Write concise, practical business language for EHS users."
        )
        messages = [
            {"role": "system", "content": system_text},
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
            full_prompt = f"System: {system_text}\n\nUser:\n{prompt}\n\nAssistant:"
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
            output_ids = self.model.generate(input_ids, attention_mask=attention_mask, **generation_kwargs)
        new_tokens = output_ids[0, input_ids.shape[-1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def _build_prompt(self, analysis: dict[str, Any]) -> str:
        """Build a compact evidence package for the local LLM.

        The prompt intentionally avoids raw score dumps. Scores are reduced to
        a simple similarity band and short evidence descriptions so the final
        response is business-readable.
        """
        structured = analysis.get("structured_evidence_summary")
        if isinstance(structured, dict) and structured:
            return self._build_prompt_from_structured(structured)

        max_evidence = int(getattr(self.settings, "llm_max_evidence_records_per_section", 5))
        query = analysis.get("query", {}) or {}
        theme = analysis.get("risk_pattern_classification", {}) or {}
        severe = analysis.get("historical_severe_injury_similarity", {}) or {}
        injury_evidence = analysis.get("injury_evidence_for_response", {}) or {}
        leading = analysis.get("leading_event_evidence", {}) or {}
        recall = analysis.get("similar_historical_event_recall", {}) or {}
        action_recall = analysis.get("corrective_action_recall", {}) or {}
        actions = analysis.get("recommended_prevention_actions", []) or []
        risk_factors = analysis.get("risk_factor_extraction", []) or []
        missing = analysis.get("missing_information_prompt", []) or []

        parts: list[str] = []
        parts.append("Create the final response for a safety form-entry assistant.")
        parts.append("Use these exact section headings, in this exact order, and write complete sentences:")
        parts.append(
            "1. Detected pattern\n"
            "2. Injury similarity evidence\n"
            "3. Leading-event evidence: hazards, near misses, audit observations\n"
            "4. Risk factors and possible control gaps\n"
            "5. Corrective actions / prevention evidence\n"
            "6. Missing information to collect\n"
            "7. Evidence IDs"
        )
        parts.append("The final response must be business-readable for EHS users, not a raw retrieval dump.")
        parts.append("Do not list every retrieved record. Summarize the patterns and cite only a few supporting event IDs.")
        parts.append("Do not include raw row IDs, raw scores, FAISS/BM25 terms, ranks, pipe-delimited tables, or technical retrieval details.")
        parts.append("Do not use markdown heading symbols such as ### or ####. Plain numbered headings and short bullets are acceptable.")
        parts.append("Do not say the system predicts an injury. Say records are historically similar or that similar historical evidence was found.")
        parts.append("If severe injury evidence is available, summarize only those severe cases. If not, summarize the normal injury examples if provided.")

        parts.append("\nNEW REPORT")
        parts.append(f"Event ID: {query.get('event_id') or 'manual/new report'}")
        parts.append(f"Source type: {query.get('source_type') or 'not provided'}")
        parts.append(f"Site: {query.get('site') or 'not provided'}")
        parts.append(f"Department: {query.get('department') or 'not provided'}")
        parts.append(f"Description: {query.get('text_preview') or ''}")

        parts.append("\nDETECTED PATTERN EVIDENCE")
        parts.append(f"Theme ID: {theme.get('risk_theme_id') or 'unknown'}")
        parts.append(f"Theme name: {theme.get('risk_theme_name') or 'Unknown theme'}")
        parts.append(f"Classification method: {theme.get('classification_method') or 'not available'}")

        parts.append("\nINJURY EVIDENCE FOR RESPONSE")
        parts.append(f"Evidence type selected: {injury_evidence.get('evidence_type') or 'not_available'}")
        parts.append(f"Similarity band: {injury_evidence.get('similarity_band') or severe.get('similarity_band') or 'no_match'}")
        parts.append(f"Selection note: {injury_evidence.get('message') or 'No injury evidence selection note.'}")
        parts.append(self._format_evidence_section("Selected injury evidence", injury_evidence.get("matches", []) or [], max_evidence))

        parts.append("\nLEADING EVENT EVIDENCE")
        hazards = leading.get("hazard_identification_matches", []) or []
        near_misses = leading.get("near_miss_matches", []) or []
        audits = leading.get("audit_observation_matches", []) or []
        other_audits = leading.get("other_audit_observation_matches", []) or []
        unsafe_actions = leading.get("unsafe_action_matches", []) or []
        unsafe_conditions = leading.get("unsafe_condition_matches", []) or []
        safe_actions = leading.get("safe_action_matches", []) or []
        safe_conditions = leading.get("safe_condition_matches", []) or []
        if not (hazards or near_misses or audits or other_audits or unsafe_actions or unsafe_conditions or safe_actions or safe_conditions):
            mixed = recall.get("matches", []) or []
            parts.append(self._format_evidence_section("Mixed leading-event matches", mixed, max_evidence))
        else:
            parts.append(self._format_evidence_section("Hazard identification matches", hazards, max_evidence))
            parts.append(self._format_evidence_section("Near-miss matches", near_misses, max_evidence))
            parts.append(self._format_evidence_section("Audit observation matches", audits, max_evidence))
            parts.append(self._format_evidence_section("Unsafe action matches", unsafe_actions, max_evidence))
            parts.append(self._format_evidence_section("Unsafe condition matches", unsafe_conditions, max_evidence))
            parts.append(self._format_evidence_section("Safe action matches", safe_actions, max_evidence))
            parts.append(self._format_evidence_section("Safe condition matches", safe_conditions, max_evidence))
            parts.append(self._format_evidence_section("Other audit observation matches", other_audits, max_evidence))

        parts.append("\nRISK FACTOR CANDIDATES")
        if risk_factors:
            for item in risk_factors[:12]:
                if isinstance(item, dict):
                    factor = item.get("risk_factor") or item.get("phrase") or item.get("text")
                    if factor:
                        parts.append(f"- {factor}")
                else:
                    parts.append(f"- {item}")
        else:
            parts.append("- No risk-factor phrases were extracted by the local heuristic.")

        parts.append("\nCORRECTIVE ACTION / PREVENTION EVIDENCE")
        parts.append(self._format_evidence_section("Retrieved corrective-action records", action_recall.get("matches", []) or [], max_evidence))
        if actions:
            parts.append("Suggested action candidates derived from retrieved evidence:")
            for item in actions[: int(getattr(self.settings, "llm_max_action_candidates", 8))]:
                if isinstance(item, dict):
                    rec = clean_text_value(item.get("recommendation") or "")
                    evidence_id = item.get("supporting_event_id") or item.get("event_id") or "unknown evidence"
                    parts.append(f"- {preview(rec, 180)} (evidence: {evidence_id})")
                else:
                    parts.append(f"- {preview(item, 180)}")
        else:
            parts.append("No action candidates were extracted from corrective-action evidence.")

        parts.append("\nMISSING INFORMATION CANDIDATES")
        if missing:
            for item in missing[: int(getattr(self.settings, "llm_max_missing_info_prompts", 8))]:
                if isinstance(item, dict):
                    area = item.get("missing_area") or item.get("area") or "detail"
                    prompt = item.get("prompt") or item.get("question") or ""
                    parts.append(f"- {area}: {prompt}")
                else:
                    parts.append(f"- {item}")
        else:
            parts.append("- The rule-based check did not flag obvious missing information.")

        return "\n".join(parts)


    def _build_prompt_from_structured(self, evidence: dict[str, Any]) -> str:
        """Build the LLM prompt from cleaned structured evidence only.

        This path intentionally excludes raw retrieval fields such as FAISS/BM25
        scores, ranks, row IDs, retrieval method, and numeric site codes.
        """
        max_evidence = int(getattr(self.settings, "llm_max_evidence_records_per_section", 5))
        query = evidence.get("query", {}) or {}
        pattern = evidence.get("detected_pattern", {}) or {}
        injury = evidence.get("injury_similarity_evidence", {}) or {}
        leading = evidence.get("leading_event_evidence", {}) or {}
        prevention = evidence.get("corrective_actions_prevention_evidence", {}) or {}
        risk_factors = evidence.get("risk_factors_and_possible_control_gaps", []) or []
        actions = evidence.get("recommended_prevention_action_candidates", []) or []
        missing = evidence.get("missing_information_to_collect", []) or []
        evidence_ids = evidence.get("evidence_ids_by_section", {}) or {}

        parts: list[str] = []
        parts.append("Create the final user-facing response for an EHS safety form-entry assistant.")
        parts.append("Use these exact section headings in this exact order:")
        parts.append(
            "1. Detected pattern\n"
            "2. Injury similarity evidence\n"
            "3. Leading-event evidence: hazards, near misses, audit observations\n"
            "4. Risk factors and possible control gaps\n"
            "5. Corrective actions / prevention evidence\n"
            "6. Missing information to collect\n"
            "7. Evidence IDs"
        )
        parts.append("Write in complete, concise business sentences for EHS users.")
        parts.append("The evidence below has already been quality-gated. Use only the records listed in the structured evidence sections.")
        parts.append("If an evidence section says no records were returned, say that no strong historical evidence was found for that section; do not fill the gap with generic advice.")
        parts.append("Do not include technical retrieval details, raw scores, raw ranks, BM25, FAISS, hybrid score, row ID, or numeric site codes.")
        parts.append("Do not claim the system predicts an injury. State that records are historically similar when evidence is available.")
        parts.append("Do not list every record. Summarize the pattern and cite only a few supporting evidence IDs.")
        parts.append("Do not use markdown heading symbols such as ### or ####.")

        parts.append("\nNEW REPORT")
        parts.append(f"Event ID: {query.get('event_id') or 'manual/new report'}")
        parts.append(f"Source type: {query.get('source_type') or 'not provided'}")
        parts.append(f"Site: {query.get('site') or 'not provided'}")
        parts.append(f"Department: {query.get('department') or 'not provided'}")
        parts.append(f"Description: {query.get('text_preview') or ''}")

        parts.append("\nDETECTED PATTERN")
        parts.append(f"Theme ID: {pattern.get('risk_theme_id') or 'unknown'}")
        parts.append(f"Theme name: {pattern.get('risk_theme_name') or 'Unknown theme'}")
        if pattern.get("theme_profile_summary"):
            parts.append(f"Theme profile summary: {preview(pattern.get('theme_profile_summary'), 400)}")

        parts.append("\nINJURY SIMILARITY EVIDENCE")
        parts.append(f"Selected index: {injury.get('selected_index') or 'not available'}")
        parts.append(f"Evidence type: {injury.get('evidence_type') or 'not available'}")
        parts.append(f"Similarity band: {injury.get('similarity_band') or 'not available'}")
        parts.append(f"Selection note: {injury.get('message') or 'not available'}")
        parts.append(self._format_structured_records("Selected injury evidence", injury.get("records", []) or [], max_evidence))

        parts.append("\nLEADING EVENT EVIDENCE")
        parts.append(self._format_structured_records("Hazard identification evidence", leading.get("hazard_identifications", []) or [], max_evidence))
        parts.append(self._format_structured_records("Near-miss evidence", leading.get("near_misses", []) or [], max_evidence))
        parts.append(self._format_structured_records("Audit observation evidence", leading.get("audit_observations", []) or [], max_evidence))
        parts.append(self._format_structured_records("Unsafe action evidence", leading.get("unsafe_actions", []) or [], max_evidence))
        parts.append(self._format_structured_records("Unsafe condition evidence", leading.get("unsafe_conditions", []) or [], max_evidence))
        parts.append(self._format_structured_records("Safe action evidence", leading.get("safe_actions", []) or [], max_evidence))
        parts.append(self._format_structured_records("Safe condition evidence", leading.get("safe_conditions", []) or [], max_evidence))
        parts.append(self._format_structured_records("Other audit observation evidence", leading.get("other_audit_observations", []) or [], max_evidence))

        parts.append("\nRISK FACTOR AND CONTROL GAP CANDIDATES")
        if risk_factors:
            for factor in risk_factors[:12]:
                parts.append(f"- {preview(factor, 180)}")
        else:
            parts.append("- No risk-factor phrases were extracted by the local heuristic.")

        parts.append("\nCORRECTIVE ACTION / PREVENTION EVIDENCE")
        parts.append(self._format_structured_records("Historical corrective-action evidence", prevention.get("corrective_actions", []) or [], max_evidence))
        parts.append(self._format_structured_records("Open corrective-action evidence", prevention.get("open_corrective_actions", []) or [], max_evidence))
        parts.append(self._format_structured_records("Overdue corrective-action evidence", prevention.get("overdue_corrective_actions", []) or [], max_evidence))
        if actions:
            parts.append("Action candidates derived from retrieved evidence:")
            for item in actions[: int(getattr(self.settings, "llm_max_action_candidates", 8))]:
                action = preview(item.get("suggested_action"), 220)
                evid = item.get("supporting_evidence_id") or "supporting evidence"
                parts.append(f"- {action} (evidence: {evid})")
        else:
            parts.append("No action candidates were extracted from corrective-action evidence.")

        parts.append("\nMISSING INFORMATION CANDIDATES")
        if missing:
            for item in missing[: int(getattr(self.settings, "llm_max_missing_info_prompts", 8))]:
                parts.append(f"- {item.get('question')}")
        else:
            parts.append("- No obvious missing information was flagged by the rule-based check.")

        parts.append("\nEVIDENCE IDS BY SECTION")
        for key, ids in evidence_ids.items():
            if ids:
                parts.append(f"- {key}: {', '.join(map(str, ids[:8]))}")
        return "\n".join(parts)

    def _format_structured_records(self, title: str, records: list[dict[str, Any]], max_evidence: int) -> str:
        lines = [title + ":"]
        if not records:
            lines.append("- No records were returned for this evidence type.")
            return "\n".join(lines)
        for item in records[:max_evidence]:
            event_id = item.get("evidence_id") or "unknown evidence ID"
            role = self._friendly_role_name(item.get("source_role") or item.get("source_type") or "historical record")
            site = clean_text_value(item.get("site_label") or "")
            title_text = clean_text_value(item.get("title") or "")
            summary = clean_text_value(item.get("summary") or "")
            summary_text = title_text or summary
            where = f" at {site}" if site and not self._is_numeric_like(site) else ""
            if summary_text:
                lines.append(f"- {event_id}: This {role}{where} describes {preview(summary_text, 180)}")
            else:
                lines.append(f"- {event_id}: This {role}{where} was retrieved as supporting evidence.")
        return "\n".join(lines)

    @staticmethod
    def _is_numeric_like(value: object) -> bool:
        text = clean_text_value(value)
        if not text:
            return False
        try:
            float(text)
            return True
        except Exception:
            return False

    def _format_evidence_section(self, title: str, matches: list[dict[str, Any]], max_evidence: int) -> str:
        """Format retrieved evidence for the LLM without raw score/rank fields."""
        lines = [title + ":"]
        if not matches:
            lines.append("- No records were returned for this evidence type.")
            return "\n".join(lines)
        for item in matches[:max_evidence]:
            event_id = item.get("event_id") or item.get("source_id") or "unknown evidence ID"
            role = self._friendly_role_name(item.get("source_role") or item.get("source_type") or "historical record")
            site = clean_text_value(item.get("site") or "")
            title_text = clean_text_value(item.get("title") or "")
            description = clean_text_value(item.get("description") or item.get("retrieval_text") or "")
            summary_text = title_text or description
            where = f" at {site}" if site and site.lower() not in {"unknown", "unknown site"} else ""
            if summary_text:
                lines.append(f"- {event_id}: This {role}{where} describes {preview(summary_text, 180)}")
            else:
                lines.append(f"- {event_id}: This {role}{where} was retrieved as supporting evidence.")
        return "\n".join(lines)

    @staticmethod
    def _friendly_role_name(value: object) -> str:
        role = clean_text_value(value).replace("_", " ").strip().lower()
        mapping = {
            "hazard identification": "hazard identification record",
            "near miss": "near-miss record",
            "audit observation": "audit observation",
            "corrective action": "corrective-action record",
            "open corrective action": "open corrective-action record",
            "overdue corrective action": "overdue corrective-action record",
            "severe injury": "severe-injury record",
            "injury": "injury record",
            "safe observation": "safe-observation record",
            "unsafe observation": "unsafe-observation record",
            "safe action": "safe-action observation",
            "unsafe action": "unsafe-action observation",
            "safe condition": "safe-condition observation",
            "unsafe condition": "unsafe-condition observation",
        }
        return mapping.get(role, role or "historical record")

    def _postprocess_response(self, response: object) -> str:
        """Clean LLM output so it reads like a business response, not raw retrieval text."""
        text = clean_text_value(response)
        if not text:
            return ""
        cleaned_lines: list[str] = []
        blocked_fragments = (
            "top score",
            "top faiss",
            "faiss cosine",
            "bm25 score",
            "hybrid score",
            "similarity_score",
            "matched_row_id",
            "row id",
            "retrieval_method",
            "faiss rank",
            "bm25 rank",
            "bm25",
            "faiss",
            "hybrid",
            "retrieval method",
            "numeric site",
        )
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            # Remove markdown heading marks and excessive emphasis markers. Keep bullets and numbering.
            while line.startswith("#"):
                line = line[1:].strip()
            line = line.replace("####", "").replace("###", "").replace("**", "").strip()
            lower = line.lower()
            if any(fragment in lower for fragment in blocked_fragments):
                continue
            # Remove raw pipe-delimited retrieval rows if the model copied evidence lines verbatim.
            if line.count("|") >= 2:
                continue
            cleaned_lines.append(line)
        text = "\n".join(cleaned_lines)
        return clean_text_value(text)

    def _heuristic_response(self, analysis: dict[str, Any]) -> str:
        """Deterministic fallback response when LLM generation is unavailable."""
        query = analysis.get("query", {}) or {}
        theme = analysis.get("risk_pattern_classification", {}) or {}
        injury_evidence = analysis.get("injury_evidence_for_response", {}) or {}
        leading = analysis.get("leading_event_evidence", {}) or {}
        action_recall = analysis.get("corrective_action_recall", {}) or {}
        actions = analysis.get("recommended_prevention_actions", []) or []
        missing = analysis.get("missing_information_prompt", []) or []
        risk_factors = analysis.get("risk_factor_extraction", []) or []

        lines: list[str] = []
        lines.append("1. Detected pattern")
        lines.append(f"- {theme.get('risk_theme_name') or 'Unknown theme'}")
        lines.append(f"- Report reviewed: {query.get('text_preview') or ''}")

        lines.append("\n2. Injury similarity evidence")
        evidence_type = injury_evidence.get("evidence_type") or "not_available"
        selected = injury_evidence.get("matches", []) or []
        if evidence_type == "severe_injury" and selected:
            lines.append("- Similar severe-injury cases were found. Closest examples:")
        elif selected:
            lines.append("- No meaningful severe-injury evidence was selected; closest non-severe injury examples:")
        else:
            lines.append("- No injury examples were selected for the final response.")
        for item in selected[:5]:
            lines.append(f"  - {item.get('event_id')}: {preview(item.get('title') or item.get('description') or item.get('retrieval_text'), 160)}")

        lines.append("\n3. Leading-event evidence: hazards, near misses, audit observations")
        hazards = leading.get("hazard_identification_matches", []) or []
        near_misses = leading.get("near_miss_matches", []) or []
        audits = leading.get("audit_observation_matches", []) or []
        unsafe_actions = leading.get("unsafe_action_matches", []) or []
        unsafe_conditions = leading.get("unsafe_condition_matches", []) or []
        safe_actions = leading.get("safe_action_matches", []) or []
        safe_conditions = leading.get("safe_condition_matches", []) or []
        lines.append(f"- Hazard identification matches returned: {len(hazards)}")
        lines.append(f"- Near-miss matches returned: {len(near_misses)}")
        lines.append(f"- Audit observation matches returned: {len(audits)}")
        lines.append(f"- Unsafe action/condition matches returned: {len(unsafe_actions) + len(unsafe_conditions)}")
        lines.append(f"- Safe action/condition matches returned: {len(safe_actions) + len(safe_conditions)}")

        lines.append("\n4. Risk factors and possible control gaps")
        if risk_factors:
            for item in risk_factors[:8]:
                if isinstance(item, dict):
                    val = item.get("risk_factor") or item.get("phrase") or item.get("text")
                    if val:
                        lines.append(f"- {val}")
                else:
                    lines.append(f"- {item}")
        else:
            lines.append("- No risk-factor candidates were extracted.")

        lines.append("\n5. Corrective actions / prevention evidence")
        action_matches = action_recall.get("matches", []) or []
        if actions:
            for item in actions[:5]:
                if isinstance(item, dict):
                    lines.append(f"- {item.get('recommendation')} (evidence: {item.get('supporting_event_id')})")
        elif action_matches:
            for item in action_matches[:5]:
                lines.append(f"- Historical action record {item.get('event_id')}: {preview(item.get('title') or item.get('description') or item.get('retrieval_text'), 160)}")
        else:
            lines.append("- No related corrective-action evidence was returned.")

        lines.append("\n6. Missing information to collect")
        if missing:
            for item in missing[:6]:
                if isinstance(item, dict):
                    lines.append(f"- {item.get('prompt')}")
                else:
                    lines.append(f"- {item}")
        else:
            lines.append("- No obvious missing information was flagged by the rule-based check.")

        lines.append("\n7. Evidence IDs")
        ids: list[str] = []
        for section in [selected, hazards, near_misses, audits, unsafe_actions, unsafe_conditions, safe_actions, safe_conditions, action_matches]:
            for item in section[:3]:
                event_id = item.get("event_id")
                if event_id:
                    ids.append(str(event_id))
        seen = []
        for event_id in ids:
            if event_id not in seen:
                seen.append(event_id)
        lines.append("- " + (", ".join(seen[:20]) if seen else "No evidence IDs returned."))
        return "\n".join(lines)
