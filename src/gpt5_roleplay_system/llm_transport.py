from __future__ import annotations

from typing import Any


class OpenRouterTransport:
    def __init__(
        self,
        client: Any,
        reasoning: str,
        provider_order: list[str] | None,
        provider_allow_fallbacks: bool | None,
        facts_provider_order: list[str] | None,
        facts_provider_allow_fallbacks: bool | None,
    ) -> None:
        self._client = client
        self._reasoning = str(reasoning or "")
        self._provider_order = [str(item).strip() for item in (provider_order or []) if str(item).strip()]
        self._provider_allow_fallbacks = (
            bool(provider_allow_fallbacks) if provider_allow_fallbacks is not None else None
        )
        self._facts_provider_order = (
            [str(item).strip() for item in facts_provider_order if str(item).strip()]
            if facts_provider_order is not None
            else None
        )
        self._facts_provider_allow_fallbacks = (
            bool(facts_provider_allow_fallbacks) if facts_provider_allow_fallbacks is not None else None
        )

    def request_text(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
        include_reasoning: bool | None,
        include_provider: bool,
        provider_for_facts: bool = False,
        trace_label: str = "",
    ) -> str:
        kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
        }
        reasoning_enabled = bool(self._reasoning) if include_reasoning is None else bool(include_reasoning and self._reasoning)
        self.apply_extra_body(
            kwargs,
            include_reasoning=reasoning_enabled,
            include_provider=include_provider,
            provider_for_facts=provider_for_facts,
        )
        completion = self._client.chat.completions.create(**kwargs)
        if not completion.choices:
            return ""
        message = completion.choices[0].message
        return self._message_content_text(message)

    def apply_extra_body(
        self,
        kwargs: dict[str, Any],
        *,
        include_reasoning: bool,
        include_provider: bool,
        provider_for_facts: bool = False,
    ) -> None:
        extra_body: dict[str, Any] = dict(kwargs.get("extra_body") or {})
        if include_reasoning and self._reasoning:
            extra_body["include_reasoning"] = True
            extra_body["reasoning_effort"] = str(self._reasoning)
        if include_provider:
            provider_payload = self._provider_payload(for_facts=provider_for_facts)
            if provider_payload:
                extra_body["provider"] = provider_payload
        if extra_body:
            kwargs["extra_body"] = extra_body

    def _provider_payload(self, *, for_facts: bool = False) -> dict[str, Any] | None:
        order = [str(item).strip() for item in self._provider_order if str(item).strip()]
        allow_fallbacks = self._provider_allow_fallbacks
        if for_facts:
            if self._facts_provider_order is not None:
                order = [str(item).strip() for item in self._facts_provider_order if str(item).strip()]
            if self._facts_provider_allow_fallbacks is not None:
                allow_fallbacks = bool(self._facts_provider_allow_fallbacks)
        if not order and allow_fallbacks is None:
            return None
        payload: dict[str, Any] = {}
        if order:
            payload["order"] = order
        if allow_fallbacks is not None:
            payload["allow_fallbacks"] = bool(allow_fallbacks)
        return payload

    @staticmethod
    def _coerce_text_content(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float, bool)):
            return str(value)
        if isinstance(value, list):
            parts = [OpenRouterTransport._coerce_text_content(item) for item in value]
            return "\n".join(part for part in parts if part).strip()
        if isinstance(value, dict):
            for key in ("text", "content", "value"):
                candidate = OpenRouterTransport._coerce_text_content(value.get(key))
                if candidate:
                    return candidate
            return ""
        for attr in ("text", "content", "value"):
            candidate_value = getattr(value, attr, None)
            if candidate_value is value:
                continue
            candidate = OpenRouterTransport._coerce_text_content(candidate_value)
            if candidate:
                return candidate
        return ""

    @classmethod
    def _message_content_text(cls, message: Any) -> str:
        return cls._coerce_text_content(getattr(message, "content", None)).strip()
