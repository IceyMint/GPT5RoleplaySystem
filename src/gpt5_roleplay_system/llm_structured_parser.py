from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger("gpt5_roleplay_llm")


class StructuredParser:
    def __init__(self) -> None:
        self._structured_parse_fallback_keys: set[str] = set()

    def request_structured(
        self,
        client: Any,
        model_class: Any,
        kwargs: dict[str, Any],
        *,
        request_type: str,
        trace_label: str = "",
    ) -> Any | None:
        if trace_label and hasattr(client, "_clear_reasoning_trace"):
            client._clear_reasoning_trace(trace_label)

        parse_key = self._structured_parse_key(kwargs)
        if parse_key in self._structured_parse_fallback_keys:
            raw_text = self._request_structured_raw_create(client, kwargs, request_type="structured.cached_create", trace_label=trace_label)
            if raw_text:
                try:
                    cleaned = self.clean_json(raw_text)
                    return model_class.model_validate_json(cleaned)
                except Exception as final_exc:
                    logger.warning(
                        "Robust parse failed after cached create fallback: %s. Raw:\n%s",
                        final_exc,
                        raw_text,
                    )
            return None

        try:
            completion = client._client.chat.completions.parse(**kwargs)
            if hasattr(client, "_record_cache_usage"):
                client._record_cache_usage("structured.parse", completion)
            if trace_label and hasattr(client, "_record_reasoning_trace"):
                client._record_reasoning_trace(trace_label, "structured.parse", kwargs, completion)
            if not completion.choices:
                return None
            message = completion.choices[0].message
            if getattr(message, "refusal", None):
                return None
            parsed = getattr(message, "parsed", None)
            if parsed is not None:
                return parsed
            raw_text = self._message_content_text(message)
            if raw_text:
                try:
                    cleaned = self.clean_json(raw_text)
                    return model_class.model_validate_json(cleaned)
                except Exception as fallback_exc:
                    logger.warning(
                        "Structured parse returned unparsed content and cleanup failed: %s. Raw:\n%s",
                        fallback_exc,
                        raw_text,
                    )
            return None
        except Exception as exc:
            if self._is_no_endpoint_not_found(exc):
                self._structured_parse_fallback_keys.add(parse_key)
            raw_text = None
            completion = getattr(exc, "completion", None)
            if completion is not None and hasattr(client, "_record_cache_usage"):
                client._record_cache_usage("structured.parse_error", completion)
                if trace_label and hasattr(client, "_record_reasoning_trace"):
                    client._record_reasoning_trace(trace_label, "structured.parse_error", kwargs, completion)
            if completion and completion.choices:
                raw_text = self._message_content_text(completion.choices[0].message)

            if not raw_text:
                raw_text = self._request_structured_raw_create(client, kwargs, request_type="structured.debug_create", trace_label=trace_label)

            if raw_text:
                try:
                    cleaned = self.clean_json(raw_text)
                    return model_class.model_validate_json(cleaned)
                except Exception as final_exc:
                    logger.warning("Robust parse failed even after cleanup: %s. Raw:\n%s", final_exc, raw_text)

            logger.warning("Structured request failed (%s): %s", exc.__class__.__name__, exc)
            return None

    def _request_structured_raw_create(
        self,
        client: Any,
        kwargs: dict[str, Any],
        *,
        request_type: str,
        trace_label: str = "",
    ) -> str | None:
        try:
            debug_kwargs = {k: v for k, v in kwargs.items() if k != "response_format"}
            debug_completion = client._client.chat.completions.create(**debug_kwargs)
            if hasattr(client, "_record_cache_usage"):
                client._record_cache_usage(request_type, debug_completion)
            if trace_label and hasattr(client, "_record_reasoning_trace"):
                client._record_reasoning_trace(trace_label, request_type, debug_kwargs, debug_completion)
            if debug_completion.choices:
                return self._message_content_text(debug_completion.choices[0].message)
        except Exception as debug_exc:
            logger.error("Failed to fetch raw text for cleanup: %s", debug_exc)
        return None

    @staticmethod
    def _structured_parse_key(kwargs: dict[str, Any]) -> str:
        model = str(kwargs.get("model", "") or "").strip()
        provider_payload: dict[str, Any] = {}
        extra_body = kwargs.get("extra_body")
        if isinstance(extra_body, dict):
            provider_raw = extra_body.get("provider")
            if isinstance(provider_raw, dict):
                provider_payload = {
                    "order": [str(item).strip() for item in provider_raw.get("order", []) if str(item).strip()],
                    "allow_fallbacks": provider_raw.get("allow_fallbacks"),
                }
        payload = {"model": model, "provider": provider_payload}
        return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _is_no_endpoint_not_found(exc: Exception) -> bool:
        status_code = getattr(exc, "status_code", None)
        body = getattr(exc, "body", None)
        if isinstance(body, dict):
            error = body.get("error")
            if isinstance(error, dict):
                message = str(error.get("message", "") or "").casefold()
                if "no endpoints found" in message:
                    return True
        text = str(exc).casefold()
        if "no endpoints found" in text and "404" in text:
            return True
        return status_code == 404 and "no endpoints found" in text

    @staticmethod
    def _message_content_text(message: Any) -> str:
        return StructuredParser._coerce_text_content(getattr(message, "content", None)).strip()

    @staticmethod
    def _coerce_text_content(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float, bool)):
            return str(value)
        if isinstance(value, list):
            parts = [StructuredParser._coerce_text_content(item) for item in value]
            return "\n".join(part for part in parts if part).strip()
        if isinstance(value, dict):
            for key in ("text", "content", "value"):
                candidate = StructuredParser._coerce_text_content(value.get(key))
                if candidate:
                    return candidate
            return ""
        for attr in ("text", "content", "value"):
            candidate_value = getattr(value, attr, None)
            if candidate_value is value:
                continue
            candidate = StructuredParser._coerce_text_content(candidate_value)
            if candidate:
                return candidate
        return ""

    def clean_json(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r"```(?:json)?\s*(.*?)\s*```", r"\1", text, flags=re.DOTALL)
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]
        text = re.sub(r",\s*([\]\}])", r"\1", text)
        text = re.sub(r"\[\s*,\s*", "[", text)
        text = re.sub(r",\s*,+", ",", text)
        return text.strip()
