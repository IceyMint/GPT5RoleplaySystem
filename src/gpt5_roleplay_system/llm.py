from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time
from datetime import datetime
from dataclasses import dataclass
from .time_utils import format_pacific_time
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None

try:
    from pydantic import AliasChoices, BaseModel, Field, field_validator
except ImportError:  # pragma: no cover - optional dependency
    BaseModel = None
    Field = None
    AliasChoices = None
    field_validator = None

from .models import Action, CommandType, ConversationContext, EnvironmentSnapshot, InboundChat, Participant
from .name_utils import split_display_and_username
from .llm_prompts import PromptManager
from .llm_response_mapper import ResponseMapper
from .llm_structured_parser import StructuredParser
from .llm_transport import OpenRouterTransport

logger = logging.getLogger("gpt5_roleplay_llm")
_UUID_LIKE_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    flags=re.IGNORECASE,
)
_MOST_RECENT_CONFIRMED_RE = re.compile(
    r"Most recent confirmed\s*\(as of\s+([^)]+)\)\s*:",
    flags=re.IGNORECASE,
)


def _field_as_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, dict):
            return dumped
    as_dict = getattr(value, "dict", None)
    if callable(as_dict):
        dumped = as_dict()
        if isinstance(dumped, dict):
            return dumped
    data: Dict[str, Any] = {}
    for key in (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "prompt_tokens_details",
        "completion_tokens_details",
        "input_tokens_details",
        "output_tokens_details",
        "reasoning_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
        "cache_discount",
    ):
        if hasattr(value, key):
            data[key] = getattr(value, key)
    return data


def _field_get(value: Any, key: str, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _to_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _coerce_text_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        parts = [_coerce_text_content(item) for item in value]
        return "\n".join(part for part in parts if part).strip()
    if isinstance(value, dict):
        for key in ("text", "content", "value"):
            candidate = _coerce_text_content(value.get(key))
            if candidate:
                return candidate
        return ""
    for attr in ("text", "content", "value"):
        candidate_value = getattr(value, attr, None)
        if candidate_value is value:
            continue
        candidate = _coerce_text_content(candidate_value)
        if candidate:
            return candidate
    return ""


def _parse_summary_as_of_timestamp(summary: str) -> float:
    if not summary:
        return 0.0
    matches = list(_MOST_RECENT_CONFIRMED_RE.finditer(str(summary)))
    if not matches:
        return 0.0
    raw_value = str(matches[-1].group(1) or "").strip()
    if not raw_value:
        return 0.0
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        pass
    try:
        parsed = datetime.fromisoformat(raw_value)
    except ValueError:
        return 0.0
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=ZoneInfo("America/Los_Angeles"))
    return float(parsed.timestamp())


def _message_content_text(message: Any) -> str:
    return _coerce_text_content(getattr(message, "content", None)).strip()


def _truncate_text(value: str, limit: int = 4000) -> str:
    text = value.strip()
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return f"{text[:limit]}... (truncated {omitted} chars)"


def _json_safe(value: Any, depth: int = 0) -> Any:
    if depth >= 6:
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item, depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item, depth + 1) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _json_safe(model_dump(), depth + 1)
    as_dict = getattr(value, "dict", None)
    if callable(as_dict):
        return _json_safe(as_dict(), depth + 1)
    attrs = getattr(value, "__dict__", None)
    if isinstance(attrs, dict) and attrs:
        filtered = {str(key): item for key, item in attrs.items() if not str(key).startswith("_")}
        if filtered:
            return _json_safe(filtered, depth + 1)
    return str(value)


def _extract_reasoning_tokens(completion: Any) -> int:
    usage_map = _field_as_dict(_field_get(completion, "usage"))
    if not usage_map:
        return 0
    completion_details = _field_as_dict(_field_get(usage_map, "completion_tokens_details", {}))
    output_details = _field_as_dict(_field_get(usage_map, "output_tokens_details", {}))
    return max(
        _to_int(_field_get(completion_details, "reasoning_tokens", 0)),
        _to_int(_field_get(output_details, "reasoning_tokens", 0)),
        _to_int(_field_get(usage_map, "reasoning_tokens", 0)),
    )


@dataclass
class PromptCacheUsageSample:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cached_read_tokens: int
    cache_write_tokens: int
    cache_discount: float

    @property
    def uncached_prompt_tokens(self) -> int:
        return max(self.prompt_tokens - self.cached_read_tokens, 0)


def _extract_prompt_cache_usage(completion: Any) -> PromptCacheUsageSample | None:
    usage = _field_get(completion, "usage")
    usage_map = _field_as_dict(usage)
    if not usage_map:
        return None

    prompt_details = _field_get(usage_map, "prompt_tokens_details", {})
    prompt_details_map = _field_as_dict(prompt_details)
    input_details = _field_get(usage_map, "input_tokens_details", {})
    input_details_map = _field_as_dict(input_details)

    prompt_tokens = _to_int(_field_get(usage_map, "prompt_tokens", 0))
    completion_tokens = _to_int(_field_get(usage_map, "completion_tokens", 0))
    total_tokens = _to_int(_field_get(usage_map, "total_tokens", prompt_tokens + completion_tokens))

    cached_read_tokens = max(
        _to_int(_field_get(prompt_details_map, "cached_tokens", 0)),
        _to_int(_field_get(input_details_map, "cached_tokens", 0)),
        _to_int(_field_get(usage_map, "cache_read_input_tokens", 0)),
    )
    cache_write_tokens = max(
        _to_int(_field_get(prompt_details_map, "cache_write_tokens", 0)),
        _to_int(_field_get(input_details_map, "cache_creation_tokens", 0)),
        _to_int(_field_get(usage_map, "cache_creation_input_tokens", 0)),
    )
    cache_discount = _to_float(_field_get(usage_map, "cache_discount", _field_get(completion, "cache_discount", 0.0)))

    return PromptCacheUsageSample(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cached_read_tokens=cached_read_tokens,
        cache_write_tokens=cache_write_tokens,
        cache_discount=cache_discount,
    )


class PromptCacheStats:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._requests_total = 0
        self._requests_with_usage = 0
        self._cache_hit_requests = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._total_tokens = 0
        self._cached_read_tokens = 0
        self._cache_write_tokens = 0
        self._cache_discount_total = 0.0
        self._request_type_counts: Dict[str, int] = {}

    def record(self, request_type: str, sample: PromptCacheUsageSample | None) -> None:
        with self._lock:
            self._requests_total += 1
            self._request_type_counts[request_type] = self._request_type_counts.get(request_type, 0) + 1
            if sample is None:
                return
            self._requests_with_usage += 1
            if sample.cached_read_tokens > 0:
                self._cache_hit_requests += 1
            self._prompt_tokens += sample.prompt_tokens
            self._completion_tokens += sample.completion_tokens
            self._total_tokens += sample.total_tokens
            self._cached_read_tokens += sample.cached_read_tokens
            self._cache_write_tokens += sample.cache_write_tokens
            self._cache_discount_total += sample.cache_discount

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            prompt_tokens = self._prompt_tokens
            cached_read_tokens = self._cached_read_tokens
            uncached_prompt_tokens = max(prompt_tokens - cached_read_tokens, 0)
            cache_hit_rate = (cached_read_tokens / prompt_tokens) if prompt_tokens else 0.0
            request_hit_rate = (
                self._cache_hit_requests / self._requests_with_usage if self._requests_with_usage else 0.0
            )
            return {
                "requests_total": self._requests_total,
                "requests_with_usage": self._requests_with_usage,
                "cache_hit_requests": self._cache_hit_requests,
                "request_hit_rate": request_hit_rate,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": self._completion_tokens,
                "total_tokens": self._total_tokens,
                "cached_read_tokens": cached_read_tokens,
                "cache_write_tokens": self._cache_write_tokens,
                "uncached_prompt_tokens": uncached_prompt_tokens,
                "prompt_cache_hit_rate": cache_hit_rate,
                "cache_discount_total": self._cache_discount_total,
                "request_type_counts": dict(self._request_type_counts),
            }


_PROMPT_CACHE_STATS = PromptCacheStats()


def prompt_cache_stats_snapshot() -> Dict[str, Any]:
    return _PROMPT_CACHE_STATS.snapshot()


def log_prompt_cache_summary() -> None:
    stats = prompt_cache_stats_snapshot()
    if stats["requests_total"] == 0:
        logger.info("Prompt cache summary: no LLM requests were made.")
        return
    logger.info(
        "Prompt cache summary: requests=%d with_usage=%d cache_hit_requests=%d request_hit_rate=%.2f%% "
        "prompt_tokens=%d cached_read=%d cache_write=%d uncached_prompt=%d prompt_hit_rate=%.2f%% "
        "completion_tokens=%d total_tokens=%d cache_discount_total=%.6f request_types=%s",
        stats["requests_total"],
        stats["requests_with_usage"],
        stats["cache_hit_requests"],
        stats["request_hit_rate"] * 100.0,
        stats["prompt_tokens"],
        stats["cached_read_tokens"],
        stats["cache_write_tokens"],
        stats["uncached_prompt_tokens"],
        stats["prompt_cache_hit_rate"] * 100.0,
        stats["completion_tokens"],
        stats["total_tokens"],
        stats["cache_discount_total"],
        stats["request_type_counts"],
    )


@dataclass
class ExtractedFact:
    user_id: str
    name: str
    facts: List[str]


@dataclass
class LLMResponse:
    text: str
    actions: List[Action]


@dataclass
class ParticipantHint:
    user_id: str
    name: str


@dataclass
class LLMResponseBundle:
    text: str
    actions: List[Action]
    facts: List[ExtractedFact]
    participant_hints: List[ParticipantHint]
    summary: str | None = None
    mood: str | None = None
    status: str | None = None
    autonomy_decision: str | None = None
    next_delay_seconds: float | None = None


@dataclass
class LLMStateUpdate:
    facts: List[ExtractedFact]
    participant_hints: List[ParticipantHint]
    summary_update: str | None = None
    mood: str | None = None
    status: str | None = None
    autonomy_decision: str | None = None
    next_delay_seconds: float | None = None


if BaseModel is not None:

    class StructuredAction(BaseModel):
        # Accept both "type" (preferred) and "command" (common model variant).
        type: CommandType = Field(validation_alias=AliasChoices("type", "command", "action"))
        content: str = ""
        x: float = 0.0
        y: float = 0.0
        z: float = 0.0
        target_uuid: str = ""
        parameters: Dict[str, Any] = Field(default_factory=dict)

    class StructuredFact(BaseModel):
        user_id: str
        name: str = Field(
            default="",
            validation_alias=AliasChoices("name", "display_name", "full_name", "username"),
        )
        facts: List[str] = Field(default_factory=list, validation_alias=AliasChoices("facts", "fact"))

        @field_validator("facts", mode="before")
        @classmethod
        def _normalize_facts(cls, value: Any) -> List[str]:
            if value is None:
                return []
            if isinstance(value, str):
                cleaned = value.strip()
                return [cleaned] if cleaned else []
            if isinstance(value, list):
                normalized: List[str] = []
                for item in value:
                    text = str(item or "").strip()
                    if text:
                        normalized.append(text)
                return normalized
            text = str(value).strip()
            return [text] if text else []

    class StructuredParticipantHint(BaseModel):
        user_id: str = ""
        name: str = ""

    class StructuredBundle(BaseModel):
        thought_process: Optional[str] = Field(default=None, description="Internal reasoning only. Discarded after parsing.")
        actions: List[StructuredAction] = Field(default_factory=list)
        participant_hints: List[StructuredParticipantHint] = Field(default_factory=list)
        facts: List[StructuredFact] = Field(default_factory=list)
        summary_update: Optional[str] = None
        mood: Optional[str] = None
        status: Optional[str] = None
        autonomy_decision: Optional[str] = Field(
            default=None,
            validation_alias=AliasChoices("autonomy_decision", "decision"),
        )
        next_delay_seconds: Optional[float] = None

        @field_validator("participant_hints", mode="before")
        @classmethod
        def _normalize_participant_hints(cls, value: Any) -> Any:
            return [] if value is None else value

    class StructuredStateUpdate(BaseModel):
        participant_hints: List[StructuredParticipantHint] = Field(default_factory=list)
        facts: List[StructuredFact] = Field(default_factory=list)
        summary_update: Optional[str] = None
        mood: Optional[str] = None
        status: Optional[str] = None
        autonomy_decision: Optional[str] = Field(
            default=None,
            validation_alias=AliasChoices("autonomy_decision", "decision"),
        )
        next_delay_seconds: Optional[float] = None

        @field_validator("participant_hints", mode="before")
        @classmethod
        def _normalize_participant_hints(cls, value: Any) -> Any:
            return [] if value is None else value

    class StructuredFactsOnly(BaseModel):
        facts: List[StructuredFact] = Field(default_factory=list)

else:  # pragma: no cover - optional dependency
    StructuredAction = None
    StructuredFact = None
    StructuredParticipantHint = None
    StructuredBundle = None
    StructuredStateUpdate = None
    StructuredFactsOnly = None


class LLMClient:
    async def is_addressed_to_me(
        self,
        chat: InboundChat,
        persona: str,
        environment: EnvironmentSnapshot | None = None,
        participants: List[Participant] | None = None,
        context: ConversationContext | None = None,
    ) -> bool:
        raise NotImplementedError

    async def generate_response(self, chat: InboundChat, context: ConversationContext) -> LLMResponse:
        raise NotImplementedError

    async def extract_facts(self, chat: InboundChat, context: ConversationContext) -> List[ExtractedFact]:
        raise NotImplementedError

    async def summarize_overflow(self, summary: str, messages: List[InboundChat]) -> str:
        raise NotImplementedError

    async def summarize_episode(self, messages: List[InboundChat]) -> str:
        raise NotImplementedError

    async def generate_bundle(
        self,
        chat: InboundChat,
        context: ConversationContext,
        overflow: List[InboundChat] | None = None,
        incoming_batch: List[Dict[str, Any]] | None = None,
    ) -> LLMResponseBundle:
        response = await self.generate_response(chat, context)
        facts = await self.extract_facts(chat, context)
        summary = None
        if overflow:
            summary = await self.summarize_overflow(context.summary, overflow)
        return LLMResponseBundle(
            text=response.text,
            actions=response.actions,
            facts=facts,
            participant_hints=[],
            summary=summary,
        )

    async def generate_state_update(
        self,
        chat: InboundChat,
        context: ConversationContext,
        overflow: List[InboundChat] | None = None,
        incoming_batch: List[Dict[str, Any]] | None = None,
    ) -> LLMStateUpdate:
        return LLMStateUpdate(
            facts=[],
            participant_hints=[],
            summary_update=None,
            mood=None,
            status=None,
            autonomy_decision=None,
            next_delay_seconds=None,
        )

    async def generate_autonomous_bundle(
        self,
        context: ConversationContext,
        activity: Dict[str, Any],
    ) -> LLMResponseBundle:
        return LLMResponseBundle(text="", actions=[], facts=[], participant_hints=[], summary=None)

    def extract_facts_from_evidence_sync(
        self,
        context: ConversationContext,
        evidence_messages: List[InboundChat],
        participants: List[Participant],
    ) -> List[ExtractedFact]:
        return []

    def consume_reasoning_trace(self, label: str) -> Dict[str, Any] | None:
        return None


class RuleBasedLLMClient(LLMClient):
    async def is_addressed_to_me(
        self,
        chat: InboundChat,
        persona: str,
        environment: EnvironmentSnapshot | None = None,
        participants: List[Participant] | None = None,
        context: ConversationContext | None = None,
    ) -> bool:
        text = chat.text.lower().strip()
        if not text:
            return False

        persona_norm = _normalize_name(persona)
        if persona_norm and _name_in_text(persona_norm, text):
            return True
        if text.startswith("@"):
            return persona_norm in text
        if any(token in text for token in ("everyone", "anyone", "all")):
            return True

        other_names = _collect_other_names(persona_norm, environment, participants, chat.sender_name)
        if other_names:
            mentioned = {name for name in other_names if _name_in_text(name, text)}
            if mentioned:
                return False

        return True

    async def generate_response(self, chat: InboundChat, context: ConversationContext) -> LLMResponse:
        # Silent fallback: avoid immersion-breaking canned acknowledgements.
        return LLMResponse(text="", actions=[])

    async def extract_facts(self, chat: InboundChat, context: ConversationContext) -> List[ExtractedFact]:
        return []

    async def summarize_overflow(self, summary: str, messages: List[InboundChat]) -> str:
        return summary

    async def summarize_episode(self, messages: List[InboundChat]) -> str:
        return ""

    async def generate_bundle(
        self,
        chat: InboundChat,
        context: ConversationContext,
        overflow: List[InboundChat] | None = None,
        incoming_batch: List[Dict[str, Any]] | None = None,
    ) -> LLMResponseBundle:
        response = await self.generate_response(chat, context)
        return LLMResponseBundle(
            text=response.text,
            actions=response.actions,
            facts=[],
            participant_hints=[],
            summary=None,
        )

    async def generate_autonomous_bundle(
        self,
        context: ConversationContext,
        activity: Dict[str, Any],
    ) -> LLMResponseBundle:
        return LLMResponseBundle(text="", actions=[], facts=[], participant_hints=[], summary=None)

    async def generate_autonomous_bundle(
        self,
        context: ConversationContext,
        activity: Dict[str, Any],
    ) -> LLMResponseBundle:
        seconds = float(activity.get("seconds_since_activity", 0.0) or 0.0)
        recent_window = float(activity.get("recent_activity_window_seconds", 45.0) or 45.0)
        if seconds < recent_window:
            return LLMResponseBundle(text="", actions=[], facts=[], participant_hints=[], summary=None)
        if not context.environment.agents:
            return LLMResponseBundle(text="", actions=[], facts=[], participant_hints=[], summary=None)
        action = Action(
            command_type=CommandType.EMOTE,
            content="looks around quietly.",
            parameters={"content": "looks around quietly."},
        )
        return LLMResponseBundle(text="", actions=[action], facts=[], participant_hints=[], summary=None)


class EchoLLMClient(LLMClient):
    async def is_addressed_to_me(
        self,
        chat: InboundChat,
        persona: str,
        environment: EnvironmentSnapshot | None = None,
        participants: List[Participant] | None = None,
        context: ConversationContext | None = None,
    ) -> bool:
        return True

    async def generate_response(self, chat: InboundChat, context: ConversationContext) -> LLMResponse:
        response_text = f"Echo: {chat.text}"
        action = Action(command_type=CommandType.CHAT, content=response_text, parameters={"content": response_text})
        return LLMResponse(text=response_text, actions=[action])

    async def extract_facts(self, chat: InboundChat, context: ConversationContext) -> List[ExtractedFact]:
        return []

    async def summarize_overflow(self, summary: str, messages: List[InboundChat]) -> str:
        return summary

    async def summarize_episode(self, messages: List[InboundChat]) -> str:
        return ""

    async def generate_bundle(
        self,
        chat: InboundChat,
        context: ConversationContext,
        overflow: List[InboundChat] | None = None,
        incoming_batch: List[Dict[str, Any]] | None = None,
    ) -> LLMResponseBundle:
        response = await self.generate_response(chat, context)
        return LLMResponseBundle(
            text=response.text,
            actions=response.actions,
            facts=[],
            participant_hints=[],
            summary=None,
        )


class OpenRouterLLMClient(LLMClient):
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        bundle_model: str = "",
        summary_model: str = "",
        facts_model: str = "",
        address_model: str | None = None,
        max_tokens: int = 500,
        temperature: float = 0.6,
        timeout_seconds: float = 30.0,
        facts_in_bundle: bool = True,
        fallback: Optional[LLMClient] = None,
        reasoning: str = "",
        provider_order: List[str] | None = None,
        provider_allow_fallbacks: bool | None = None,
        facts_provider_order: List[str] | None = None,
        facts_provider_allow_fallbacks: bool | None = None,
    ) -> None:
        if OpenAI is None:
            raise RuntimeError("openai package is not installed")
        if BaseModel is None or StructuredBundle is None:
            raise RuntimeError("pydantic package is not installed")
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._bundle_model = bundle_model or model
        self._summary_model = summary_model or model
        self._facts_model = facts_model or model
        self._address_model = address_model or model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._timeout = timeout_seconds
        self._facts_in_bundle = bool(facts_in_bundle)
        self._reasoning = reasoning
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
        # Cache model/provider combinations where structured parse is known to fail with
        # "No endpoints found" so we can skip the failing parse call on future requests.
        self._structured_parse_fallback_keys: set[str] = set()
        self._fallback = fallback or RuleBasedLLMClient()
        actual_base_url = self._base_url if self._base_url else None
        self._client = OpenAI(api_key=self._api_key, base_url=actual_base_url, timeout=self._timeout)
        self._reasoning_traces: Dict[str, Dict[str, Any]] = {}
        self._prompt_manager = PromptManager()
        self._structured_parser = StructuredParser()
        self._transport = OpenRouterTransport(
            client=self._client,
            reasoning=self._reasoning,
            provider_order=self._provider_order,
            provider_allow_fallbacks=self._provider_allow_fallbacks,
            facts_provider_order=self._facts_provider_order,
            facts_provider_allow_fallbacks=self._facts_provider_allow_fallbacks,
        )
        self._response_mapper = ResponseMapper()

    def _reasoning_trace_store(self) -> Dict[str, Dict[str, Any]]:
        traces = getattr(self, "_reasoning_traces", None)
        if isinstance(traces, dict):
            return traces
        traces = {}
        self._reasoning_traces = traces
        return traces

    def _clear_reasoning_trace(self, label: str) -> None:
        if not label:
            return
        self._reasoning_trace_store().pop(label, None)

    def consume_reasoning_trace(self, label: str) -> Dict[str, Any] | None:
        if not label:
            return None
        payload = self._reasoning_trace_store().pop(label, None)
        return payload if isinstance(payload, dict) else None

    def _record_reasoning_trace(
        self,
        label: str,
        request_type: str,
        kwargs: Dict[str, Any],
        completion: Any,
    ) -> None:
        if not label:
            return

        usage_map = _field_as_dict(_field_get(completion, "usage"))
        prompt_tokens = _to_int(_field_get(usage_map, "prompt_tokens", 0))
        completion_tokens = _to_int(_field_get(usage_map, "completion_tokens", 0))
        total_tokens = _to_int(_field_get(usage_map, "total_tokens", prompt_tokens + completion_tokens))
        reasoning_tokens = _extract_reasoning_tokens(completion)

        message = None
        choices = _field_get(completion, "choices")
        if isinstance(choices, list) and choices:
            message = _field_get(choices[0], "message")

        reasoning_text = ""
        reasoning_details_payload: Any = None
        if message is not None:
            reasoning_value = _field_get(message, "reasoning")
            reasoning_details = _field_get(message, "reasoning_details")
            reasoning_text = _truncate_text(_coerce_text_content(reasoning_value))
            if not reasoning_text:
                reasoning_text = _truncate_text(_coerce_text_content(reasoning_details))
            reasoning_details_payload = _json_safe(reasoning_details)

        extra_body = kwargs.get("extra_body")
        extra_body_map = extra_body if isinstance(extra_body, dict) else {}
        include_reasoning = bool(_field_get(extra_body_map, "include_reasoning", False))
        reasoning_effort = str(_field_get(extra_body_map, "reasoning_effort", "") or "").strip()
        provider_payload = _json_safe(_field_get(extra_body_map, "provider")) if "provider" in extra_body_map else None

        payload: Dict[str, Any] = {
            "label": label,
            "request_type": request_type,
            "model": str(kwargs.get("model", "") or ""),
            "include_reasoning": include_reasoning,
            "reasoning_effort": reasoning_effort or None,
            "reasoning_tokens": reasoning_tokens,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "reasoning_text": reasoning_text,
            "reasoning_details": reasoning_details_payload,
            "has_reasoning": bool(reasoning_text or reasoning_details_payload or reasoning_tokens > 0),
        }
        if provider_payload is not None:
            payload["provider"] = provider_payload

        self._reasoning_trace_store()[label] = payload

    def _record_cache_usage(self, request_type: str, completion: Any) -> None:
        sample = _extract_prompt_cache_usage(completion)
        _PROMPT_CACHE_STATS.record(request_type, sample)
        if sample is None:
            logger.info("LLM usage (%s): usage metadata unavailable", request_type)
            return
        logger.info(
            "LLM usage (%s): prompt=%d completion=%d total=%d cached_read=%d cache_write=%d "
            "uncached_prompt=%d prompt_hit_rate=%.2f%% cache_discount=%.6f",
            request_type,
            sample.prompt_tokens,
            sample.completion_tokens,
            sample.total_tokens,
            sample.cached_read_tokens,
            sample.cache_write_tokens,
            sample.uncached_prompt_tokens,
            ((sample.cached_read_tokens / sample.prompt_tokens) * 100.0) if sample.prompt_tokens else 0.0,
            sample.cache_discount,
        )

    def _provider_payload(self, *, for_facts: bool = False) -> Dict[str, Any] | None:
        order = [str(item).strip() for item in getattr(self, "_provider_order", []) if str(item).strip()]
        allow_fallbacks = getattr(self, "_provider_allow_fallbacks", None)
        if for_facts:
            facts_order = getattr(self, "_facts_provider_order", None)
            facts_allow_fallbacks = getattr(self, "_facts_provider_allow_fallbacks", None)
            if facts_order is not None:
                order = [str(item).strip() for item in facts_order if str(item).strip()]
            if facts_allow_fallbacks is not None:
                allow_fallbacks = bool(facts_allow_fallbacks)
        if not order and allow_fallbacks is None:
            return None
        payload: Dict[str, Any] = {}
        if order:
            payload["order"] = order
        if allow_fallbacks is not None:
            payload["allow_fallbacks"] = bool(allow_fallbacks)
        return payload

    def _apply_extra_body(
        self,
        kwargs: Dict[str, Any],
        *,
        include_reasoning: bool,
        include_provider: bool,
        provider_for_facts: bool = False,
    ) -> None:
        extra_body: Dict[str, Any] = dict(kwargs.get("extra_body") or {})
        if include_reasoning and getattr(self, "_reasoning", ""):
            extra_body["include_reasoning"] = True
            extra_body["reasoning_effort"] = str(getattr(self, "_reasoning"))
        if include_provider:
            provider_payload = self._provider_payload(for_facts=provider_for_facts)
            if provider_payload:
                extra_body["provider"] = provider_payload
        if extra_body:
            kwargs["extra_body"] = extra_body

    async def is_addressed_to_me(
        self,
        chat: InboundChat,
        persona: str,
        environment: EnvironmentSnapshot | None = None,
        participants: List[Participant] | None = None,
        context: ConversationContext | None = None,
    ) -> bool:
        if not self._api_key:
            return await self._fallback.is_addressed_to_me(chat, persona, environment, participants, context)
        system_prompt = (
            "You are a fast classifier. Decide if the message is addressed to the AI persona. "
            "Consider nicknames, phonetic spellings, typos, and direct mentions. "
            "Use conversation context, recent messages, and spatial proximity to judge intent. "
            "If the incoming message only acknowledges the AI's previous line and adds no new request, question, or topic, treat it as not requiring a reply. "
            "Coordinates (x, y, z) are in meters. "
            "Reply with only 'true' or 'false'."
        )
        user_prompt = self._format_address_check(chat, persona, environment, participants, context)
        response = self._request_text_with_model(
            model=self._address_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=5,
            temperature=0.0,
        )
        return _parse_bool_response(response)

    async def generate_response(self, chat: InboundChat, context: ConversationContext) -> LLMResponse:
        bundle = await self.generate_bundle(chat, context, None)
        return LLMResponse(text=bundle.text, actions=bundle.actions)

    async def extract_facts(self, chat: InboundChat, context: ConversationContext) -> List[ExtractedFact]:
        bundle = await self.generate_bundle(chat, context, None)
        return bundle.facts

    async def summarize_overflow(self, summary: str, messages: List[InboundChat]) -> str:
        if not self._api_key:
            return await self._fallback.summarize_overflow(summary, messages)

        ordered_messages = sorted(
            enumerate(messages),
            key=lambda item: (
                float(item[1].timestamp or 0.0) <= 0.0,
                float(item[1].timestamp or 0.0),
                int(item[0]),
            ),
        )

        summary_as_of_ts = _parse_summary_as_of_timestamp(summary)
        if summary_as_of_ts > 0.0:
            filtered_messages = [
                (idx, msg)
                for idx, msg in ordered_messages
                if float(msg.timestamp or 0.0) <= 0.0 or float(msg.timestamp or 0.0) > summary_as_of_ts
            ]
            if not filtered_messages:
                logger.info(
                    "Skipping summary update: all candidate messages are at/before existing summary as-of (%s).",
                    format_pacific_time(summary_as_of_ts),
                )
                return summary
            ordered_messages = filtered_messages

        ordered_chats = [msg for _, msg in ordered_messages]
        message_timestamps = [float(msg.timestamp or 0.0) for msg in ordered_chats if float(msg.timestamp or 0.0) > 0.0]
        content = {
            "existing_summary": summary,
            "existing_summary_is_historical": True,
            "new_messages_time_range": {
                "start": format_pacific_time(min(message_timestamps)) if message_timestamps else "0",
                "end": format_pacific_time(max(message_timestamps)) if message_timestamps else "0",
            },
            "messages": [
                {
                    "timestamp": format_pacific_time(float(msg.timestamp or 0.0)),
                    "timestamp_unix": float(msg.timestamp or 0.0),
                    "sender": msg.sender_name,
                    "sender_id": msg.sender_id,
                    "text": msg.text,
                }
                for msg in ordered_chats
            ],
        }
        prompt = (
            "Update the continuity summary by combining 'existing_summary' (older history) and 'messages' (new events). "
            "Use message timestamps to keep strict chronology and temporal language. "
            "If newer events supersede earlier states, keep the earlier event as past context and reflect the latest current state. "
            "Use temporal phrasing such as earlier/later/now when it improves clarity. "
            "Write a short-story style summary in about 4-8 sentences, clear about who did what and why. "
            "Preserve unresolved questions, promises, and emotional shifts so later responses can continue naturally. "
            "End with exactly one line in this format: "
            "'Most recent confirmed (as of <timestamp>): <state>'. "
            "Use the latest timestamp from 'messages' when available, otherwise use 'new_messages_time_range.end'. "
            "Do not invent events."
        )
        payload = await asyncio.to_thread(
            self._request_text_with_model,
            self._summary_model,
            prompt,
            json.dumps(content, ensure_ascii=True),
            self._max_tokens,
            self._temperature,
        )
        return payload.strip() if payload else summary

    async def summarize_episode(self, messages: List[InboundChat]) -> str:
        if not self._api_key:
            return await self._fallback.summarize_episode(messages)

        ordered_messages = sorted(
            enumerate(messages),
            key=lambda item: (
                float(item[1].timestamp or 0.0) <= 0.0,
                float(item[1].timestamp or 0.0),
                int(item[0]),
            ),
        )
        ordered_chats = [msg for _, msg in ordered_messages]
        message_timestamps = [float(msg.timestamp or 0.0) for msg in ordered_chats if float(msg.timestamp or 0.0) > 0.0]
        content = {
            "episode_time_range": {
                "start": format_pacific_time(min(message_timestamps)) if message_timestamps else "0",
                "end": format_pacific_time(max(message_timestamps)) if message_timestamps else "0",
            },
            "messages": [
                {
                    "timestamp": format_pacific_time(float(msg.timestamp or 0.0)),
                    "timestamp_unix": float(msg.timestamp or 0.0),
                    "sender": msg.sender_name,
                    "sender_id": msg.sender_id,
                    "text": msg.text,
                }
                for msg in ordered_chats
            ],
        }
        prompt = (
            "Create an episodic memory summary for long-term retrieval from the provided message sequence. "
            "Keep strict chronology and focus on concrete events, key participants, important state changes, "
            "commitments/promises, outcomes, and unresolved threads. "
            "Keep it factual, compact (about 5-10 sentences), and specific with names. "
            "Do not invent events."
        )
        payload = await asyncio.to_thread(
            self._request_text_with_model,
            self._summary_model,
            prompt,
            json.dumps(content, ensure_ascii=True),
            self._max_tokens,
            self._temperature,
        )
        return payload.strip() if payload else ""

    async def generate_bundle(
        self,
        chat: InboundChat,
        context: ConversationContext,
        overflow: List[InboundChat] | None = None,
        incoming_batch: List[Dict[str, Any]] | None = None,
    ) -> LLMResponseBundle:
        if not self._api_key:
            return await self._fallback.generate_bundle(chat, context, overflow, incoming_batch)
        parsed = await asyncio.to_thread(self._request_bundle, chat, context, overflow, incoming_batch)
        if parsed is None:
            return await self._fallback.generate_bundle(chat, context, overflow, incoming_batch)
        bundle = _bundle_from_structured(parsed, mode="chat")
        if not self._facts_in_bundle:
            bundle.facts = []
            return bundle
        facts_only = await asyncio.to_thread(self._request_facts_from_chat, chat, context)
        if facts_only is not None:
            bundle.facts = _facts_from_structured(facts_only)
        return bundle

    async def generate_state_update(
        self,
        chat: InboundChat,
        context: ConversationContext,
        overflow: List[InboundChat] | None = None,
        incoming_batch: List[Dict[str, Any]] | None = None,
    ) -> LLMStateUpdate:
        if not self._api_key:
            return await self._fallback.generate_state_update(chat, context, overflow, incoming_batch)
        parsed = await asyncio.to_thread(self._request_state_update, chat, context, overflow, incoming_batch)
        if parsed is None:
            return await self._fallback.generate_state_update(chat, context, overflow, incoming_batch)
        update = _state_update_from_structured(parsed)
        if not self._facts_in_bundle:
            update.facts = []
        return update

    async def generate_autonomous_bundle(
        self,
        context: ConversationContext,
        activity: Dict[str, Any],
    ) -> LLMResponseBundle:
        if not self._api_key:
            return await self._fallback.generate_autonomous_bundle(context, activity)
        parsed = await asyncio.to_thread(self._request_autonomous_bundle, context, activity)
        if parsed is None:
            return await self._fallback.generate_autonomous_bundle(context, activity)
        return _bundle_from_structured(parsed, mode="autonomous")

    def _system_prompt(self) -> str:
        return (
            "# IDENTITY & VOICE\n"
            "You are the roleplay persona declared below in this system instruction. Your primary goal is to provide immersive, "
            "in-character responses. Treat the provided persona name and persona instructions as your absolute identity.\n\n"
            "# BEHAVIORAL GUIDELINES\n"
            "- Respond to the conversation context and environment naturally.\n"
            "- Grounding: Only interact with objects (TOUCH, SIT) or mention people explicitly listed in the 'environment' or 'participants' fields. If an object is not in the list, it does not exist for you.\n"
            "- Current time is provided in Pacific Time (America/Los_Angeles).\n"
            "- 'last_message_received_at' and 'last_ai_response_at' indicate the timing of the most recent interactions.\n"
            "- Spatial context (coordinates) is provided in meters. Use this to judge proximity.\n"
            "- Consider yourself 'at' or 'inside' an object/location if you are within 1.0 meter of its coordinates.\n"
            "- Keep strict persona voice: preserve declared age, language level, and speaking style at all times.\n"
            "- If a message only closes the previous exchange and adds no new intent, return an empty actions array.\n"
            "- If you have nothing meaningful to add, you may return no actions.\n"
            "- All internal monologue and persona planning must go in 'thought_process'. All outward behavior (speech/emotes) must go in 'actions'.\n"
            "- Use 'CHAT' for dialogue and 'EMOTE' for physical descriptions or internal states expressed outwardly.\n"
            "- For complex maneuvers (e.g., walking while talking), emit multiple actions in a single response.\n"
            "- When 'incoming_batch' is provided, prioritize the 'latest_text' but use earlier messages for context or corrections.\n\n"
            "# TECHNICAL CONSTRAINTS\n"
            "- OUTPUT SCHEMA: You must strictly adhere to the provided JSON schema.\n"
            "- ACTION TYPES: Only use [CHAT, EMOTE, MOVE, TOUCH, SIT, STAND, FACE_TARGET].\n"
            "- ACTION KEYS: Every action item MUST use the key 'type'. Never use 'command' or 'action' as keys.\n"
            "- PARAMETERS: Do not place command types inside the 'parameters' dictionary.\n"
            "- DO NOT mix multiple commands into one action item.\n"
            "- CHAT CONTENT: Dialogue only. Do not include action narration or *asterisk* emote markup in CHAT.\n"
            "- EMOTE CONTENT: Emote/action narration only. Do not include spoken dialogue in EMOTE.\n"
            "- EMOTE STYLE: For EMOTE content, write plain text and do not wrap with surrounding asterisks.\n"
            "- ACTION CONTENT: Do not include '*' characters anywhere in action content.\n"
            "- If both speech and action are needed, emit separate actions (one CHAT, one EMOTE).\n\n"
            "# MEMORY & KNOWLEDGE\n"
            "- Use 'related_experiences' to inform your behavior based on past events.\n"
            "- Treat 'previously' as historical recap, not immediate present state.\n"
            "- For current reality, prioritize 'incoming', 'recent_messages', 'environment', and explicit timestamps.\n"
            "- If 'summary_meta.range_age_seconds' is high (for example >1800), treat state claims in 'previously' as stale unless recent evidence confirms them.\n"
            "- Suggest 'participant_hints' for new or important individuals mentioned in the chat.\n"
            "- Optional scheduler override: you may set 'autonomy_decision' ([act, wait, sleep]) and "
            "'next_delay_seconds' to adjust future autonomous cadence after this interaction."
        ) + "\n\n# IMPORTANT: RESPONSE FORMAT\n- You must respond ONLY with a valid JSON object matching the schema.\n- DO NOT include any preamble, conversational filler, or markdown formatting (like ```json) outside the JSON object."

    def _system_prompt_for_context(self, context: ConversationContext) -> str:
        persona = (context.persona or "").strip()
        instructions = (context.persona_instructions or "").strip()
        lines = [
            self._system_prompt(),
            "",
            f"Persona name: {persona}.",
            "You are this persona.",
        ]
        if instructions:
            lines.append("Persona instructions:")
            lines.append(instructions)
        return "\n".join(line for line in lines if line is not None).strip()

    def _facts_system_prompt(self) -> str:
        return (
            "You are a precision fact extractor. Analyze the provided chat evidence to identify **Durable Person Facts**.\n\n"
            "# DEFINITION: DURABLE FACT\n"
            "A durable fact is information that is likely to remain true over time.\n"
            "- YES: Names, relationships, professions, long-term preferences, physical traits, core history.\n"
            "- NO: Temporary states (is hungry, is tired), fleeting locations (is at the bar), or current actions (is walking).\n\n"
            "# GUIDELINES\n"
            "- Only extract facts explicitly stated in 'evidence_messages' or 'incoming'.\n"
            "- Do not derive facts from 'previously' or 'related_experiences'.\n"
            "- Use 'people_facts' to avoid duplicates; if a fact already exists or is a close paraphrase, omit it.\n"
            "- Only include facts that are likely to remain true for weeks or months.\n"
            "- If no new durable facts are found, return an empty facts list."
        ) + "\n\n# IMPORTANT: RESPONSE FORMAT\n- You must respond ONLY with a valid JSON object matching the schema.\n- DO NOT include any preamble, conversational filler, or markdown formatting outside the JSON object."

    def _state_system_prompt(self) -> str:
        return (
            "# ROLE\n"
            "You are the roleplay persona described in the input, but you MUST NOT generate any dialogue or actions.\n\n"
            "# TASK\n"
            "- Update mood and status based on the latest interaction.\n"
            "- Optionally provide participant_hints for notable individuals.\n"
            "- Optionally extract durable person facts (names, relationships, long-term preferences).\n"
            "- Optional scheduler override: set 'autonomy_decision' ([act, wait, sleep]) and "
            "'next_delay_seconds' to influence future autonomous cadence.\n\n"
            "# CONSTRAINTS\n"
            "- DO NOT output chat text.\n"
            "- DO NOT output actions.\n"
            "- Return ONLY fields defined in the JSON schema.\n\n"
            "# IMPORTANT: RESPONSE FORMAT\n"
            "- Respond ONLY with a valid JSON object matching the schema.\n"
            "- No preamble, no markdown."
        )

    def _state_system_prompt_for_context(self, context: ConversationContext) -> str:
        persona = (context.persona or "").strip()
        instructions = (context.persona_instructions or "").strip()
        lines = [
            self._state_system_prompt(),
            "",
            f"Persona name: {persona}.",
            "You are this persona.",
        ]
        if instructions:
            lines.append("Persona instructions:")
            lines.append(instructions)
        return "\n".join(line for line in lines if line is not None).strip()

    def _autonomous_system_prompt(self) -> str:
        return (
            "# IDENTITY & VOICE\n"
            "You are the roleplay persona declared below in this system instruction, deciding whether to act autonomously. "
            "Treat the provided persona name and persona instructions as your identity and voice. You are this persona.\n\n"
            "# BEHAVIORAL GUIDELINES\n"
            "- Only act when it makes sense given recent activity, environment, and relationships.\n"
            "- Grounding: Only interact with objects (TOUCH, SIT) or mention people explicitly listed in the 'environment' or 'participants' fields. If an object is not in the list, it does not exist for you.\n"
            "- Current time is provided in Pacific Time (America/Los_Angeles).\n"
            "- 'last_message_received_at' and 'last_ai_response_at' indicate the timing of the most recent interactions.\n"
            "- Spatial context (coordinates) is provided in meters. Use this to judge proximity.\n"
            "- Consider yourself 'at' or 'inside' an object/location if you are within 1.0 meter of its coordinates.\n"
            "- It is acceptable to return no actions if no action is appropriate.\n"
            "- Treat 'previously' as historical recap; do not assume it is still true without recent confirmation.\n"
            "- Prioritize 'activity', 'recent_messages', and 'environment' for what is true right now.\n"
            "- If 'summary_meta.range_age_seconds' is high (for example >1800), avoid acting on 'previously' alone.\n"
            "- All internal monologue and persona planning must go in 'thought_process'. All outward behavior (speech/emotes) must go in 'actions'.\n"
            "- Never output internal monologue, private reasoning, or narration about waiting.\n"
            "- When you 'CHAT', speak outwardly to nearby people or the environment.\n"
            "- Do not refer to yourself in the third person.\n"
            "\n"
            "# TECHNICAL CONSTRAINTS\n"
            "- OUTPUT SCHEMA: You must strictly adhere to the provided JSON schema.\n"
            "- ACTION TYPES: Only use [CHAT, EMOTE, MOVE, TOUCH, SIT, STAND, FACE_TARGET].\n"
            "- ACTION KEYS: Every action item MUST use the key 'type'. Never use 'command' or 'action' as keys.\n"
            "- PARAMETERS: Do not place command types inside the 'parameters' dictionary.\n"
            "- CHAT CONTENT: Dialogue only. Do not include action narration or *asterisk* emote markup in CHAT.\n"
            "- EMOTE CONTENT: Emote/action narration only. Do not include spoken dialogue in EMOTE.\n"
            "- EMOTE STYLE: For EMOTE content, write plain text and do not wrap with surrounding asterisks.\n"
            "- ACTION CONTENT: Do not include '*' characters anywhere in action content.\n"
            "- If both speech and action are needed, emit separate actions (one CHAT, one EMOTE).\n"
            "- Include 'autonomy_decision' as one of [act, wait, sleep].\n"
            "- 'act': emit one or more actions.\n"
            "- 'wait': emit no actions and choose a suitable 'next_delay_seconds'.\n"
            "- 'sleep': emit no actions and choose a longer 'next_delay_seconds'.\n"
            "- Include 'next_delay_seconds' whenever possible so the scheduler can pick a natural next check.\n"
            "- Include mood (short label) and status (brief current activity)."
        ) + "\n\n# IMPORTANT: RESPONSE FORMAT\n- You must respond ONLY with a valid JSON object matching the schema.\n- DO NOT include any preamble, conversational filler, or markdown formatting outside the JSON object."

    def _autonomous_system_prompt_for_context(self, context: ConversationContext) -> str:
        persona = (context.persona or "").strip()
        instructions = (context.persona_instructions or "").strip()
        lines = [
            self._autonomous_system_prompt(),
            "",
            f"Persona name: {persona}.",
            "You are this persona.",
        ]
        if instructions:
            lines.append("Persona instructions:")
            lines.append(instructions)
        return "\n".join(line for line in lines if line is not None).strip()

    def _structured_parse_cache(self) -> set[str]:
        cache = getattr(self, "_structured_parse_fallback_keys", None)
        if cache is None:
            cache = set()
            self._structured_parse_fallback_keys = cache
        return cache

    @staticmethod
    def _structured_parse_key(kwargs: Dict[str, Any]) -> str:
        model = str(kwargs.get("model", "") or "").strip()
        provider_payload: Dict[str, Any] = {}
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

    def _request_structured_raw_create(
        self,
        kwargs: Dict[str, Any],
        *,
        request_type: str,
        trace_label: str = "",
    ) -> str | None:
        try:
            debug_kwargs = {k: v for k, v in kwargs.items() if k != "response_format"}
            debug_completion = self._client.chat.completions.create(**debug_kwargs)
            self._record_cache_usage(request_type, debug_completion)
            if trace_label:
                self._record_reasoning_trace(trace_label, request_type, debug_kwargs, debug_completion)
            if debug_completion.choices:
                return _message_content_text(debug_completion.choices[0].message)
        except Exception as debug_exc:
            logger.error("Failed to fetch raw text for cleanup: %s", debug_exc)
        return None

    def _request_structured(
        self,
        model_class: Any,
        kwargs: Dict[str, Any],
        *,
        trace_label: str = "",
    ) -> Optional[Any]:
        """Execute a structured parse request with fallback cleanup for malformed JSON."""
        if trace_label:
            self._clear_reasoning_trace(trace_label)
        parse_cache = self._structured_parse_cache()
        parse_key = self._structured_parse_key(kwargs)
        if parse_key in parse_cache:
            raw_text = self._request_structured_raw_create(
                kwargs,
                request_type="structured.cached_create",
                trace_label=trace_label,
            )
            if raw_text:
                try:
                    cleaned = _clean_json(raw_text)
                    return model_class.model_validate_json(cleaned)
                except Exception as final_exc:
                    logger.warning(
                        "Robust parse failed after cached create fallback: %s. Raw:\n%s",
                        final_exc,
                        raw_text,
                    )
            return None
        try:
            completion = self._client.chat.completions.parse(**kwargs)
            self._record_cache_usage("structured.parse", completion)
            if trace_label:
                self._record_reasoning_trace(trace_label, "structured.parse", kwargs, completion)
            if not completion.choices:
                return None
            message = completion.choices[0].message
            if getattr(message, "refusal", None):
                return None
            parsed = getattr(message, "parsed", None)
            if parsed is not None:
                return parsed
            raw_text = _message_content_text(message)
            if raw_text:
                try:
                    cleaned = _clean_json(raw_text)
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
                parse_cache.add(parse_key)
            raw_text = None
            # Check for truncation or content filter in API error
            completion = getattr(exc, "completion", None)
            if completion is not None:
                self._record_cache_usage("structured.parse_error", completion)
                if trace_label:
                    self._record_reasoning_trace(trace_label, "structured.parse_error", kwargs, completion)
            if completion and completion.choices:
                raw_text = _message_content_text(completion.choices[0].message)

            if not raw_text:
                # If parse() failed due to ValidationError, it won't have the completion object.
                # We fetch the raw text by performing a standard create() call without parsing.
                raw_text = self._request_structured_raw_create(
                    kwargs,
                    request_type="structured.debug_create",
                    trace_label=trace_label,
                )

            if raw_text:
                try:
                    cleaned = _clean_json(raw_text)
                    return model_class.model_validate_json(cleaned)
                except Exception as final_exc:
                    logger.warning("Robust parse failed even after cleanup: %s. Raw:\n%s", final_exc, raw_text)

            logger.warning("Structured request failed (%s): %s", exc.__class__.__name__, exc)
            return None

    def _request_structured_with_trace(
        self,
        model_class: Any,
        kwargs: Dict[str, Any],
        *,
        trace_label: str,
    ) -> Optional[Any]:
        request_fn = getattr(self, "_request_structured")
        try:
            return request_fn(model_class, kwargs, trace_label=trace_label)
        except TypeError as exc:
            # Backward-compatible for tests/subclasses that override _request_structured
            # without the trace_label keyword.
            if "trace_label" not in str(exc):
                raise
            return request_fn(model_class, kwargs)

    def _request_bundle(
        self,
        chat: InboundChat,
        context: ConversationContext,
        overflow: List[InboundChat] | None,
        incoming_batch: List[Dict[str, Any]] | None,
    ) -> Optional[StructuredBundle]:
        kwargs = {
            "model": self._bundle_model,
            "messages": [
                {"role": "system", "content": self._system_prompt_for_context(context)},
                {"role": "user", "content": self._format_context(chat, context, overflow, incoming_batch)},
            ],
            "response_format": StructuredBundle,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
        }
        self._apply_extra_body(kwargs, include_reasoning=True, include_provider=True)
        return self._request_structured_with_trace(StructuredBundle, kwargs, trace_label="bundle")

    def _request_state_update(
        self,
        chat: InboundChat,
        context: ConversationContext,
        overflow: List[InboundChat] | None,
        incoming_batch: List[Dict[str, Any]] | None,
    ) -> Optional[StructuredStateUpdate]:
        kwargs = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": self._state_system_prompt_for_context(context)},
                {"role": "user", "content": self._format_context(chat, context, overflow, incoming_batch)},
            ],
            "response_format": StructuredStateUpdate,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
        }
        self._apply_extra_body(kwargs, include_reasoning=True, include_provider=True)
        return self._request_structured_with_trace(StructuredStateUpdate, kwargs, trace_label="state")

    def _request_autonomous_bundle(
        self,
        context: ConversationContext,
        activity: Dict[str, Any],
    ) -> Optional[StructuredBundle]:
        kwargs = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": self._autonomous_system_prompt_for_context(context)},
                {"role": "user", "content": self._format_autonomous_context(context, activity)},
            ],
            "response_format": StructuredBundle,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
        }
        self._apply_extra_body(kwargs, include_reasoning=True, include_provider=True)
        return self._request_structured_with_trace(StructuredBundle, kwargs, trace_label="autonomy")

    def _request_facts_from_chat(self, chat: InboundChat, context: ConversationContext) -> Optional[StructuredFactsOnly]:
        evidence_messages: List[InboundChat] = list(context.recent_messages[-12:])
        evidence_messages.append(chat)
        return self._request_facts_from_messages(
            evidence_messages=evidence_messages,
            participants=context.participants,
            persona=context.persona,
            user_id=context.user_id,
            summary=context.summary,
            summary_meta=context.summary_meta,
            related_experiences=context.related_experiences,
            people_facts=context.people_facts,
        )

    def _request_facts_from_messages(
        self,
        evidence_messages: List[InboundChat],
        participants: List[Participant],
        persona: str,
        user_id: str,
        summary: str = "",
        summary_meta: Dict[str, Any] | None = None,
        related_experiences: List[Dict[str, Any]] | None = None,
        people_facts: Dict[str, Any] | None = None,
    ) -> Optional[StructuredFactsOnly]:
        if StructuredFactsOnly is None:
            return None
        if not evidence_messages:
            return None
        kwargs = {
            "model": self._facts_model,
            "messages": [
                {"role": "system", "content": self._facts_system_prompt()},
                {
                    "role": "user",
                    "content": self._format_facts_context_from_messages(
                        evidence_messages=evidence_messages,
                        participants=participants,
                        persona=persona,
                        user_id=user_id,
                        summary=summary,
                        summary_meta=summary_meta or {},
                        related_experiences=related_experiences or [],
                        people_facts=people_facts or {},
                    ),
                },
            ],
            "response_format": StructuredFactsOnly,
            "max_tokens": max(self._max_tokens, 500),
            "temperature": 0.0,
        }
        self._apply_extra_body(kwargs, include_reasoning=False, include_provider=True, provider_for_facts=True)
        return self._request_structured_with_trace(StructuredFactsOnly, kwargs, trace_label="facts")

    def extract_facts_from_evidence_sync(
        self,
        context: ConversationContext,
        evidence_messages: List[InboundChat],
        participants: List[Participant],
    ) -> List[ExtractedFact]:
        parsed = self._request_facts_from_messages(
            evidence_messages=evidence_messages,
            participants=participants,
            persona=context.persona,
            user_id=context.user_id,
            summary="",
            summary_meta={},
            related_experiences=[],
            people_facts=context.people_facts,
        )
        return _facts_from_structured(parsed) if parsed is not None else []

    def _request_text(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        include_reasoning: bool | None = None,
    ) -> str:
        kwargs = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": int(max_tokens if max_tokens is not None else self._max_tokens),
            "temperature": float(temperature if temperature is not None else self._temperature),
        }
        reasoning_enabled = (
            bool(self._reasoning) if include_reasoning is None else bool(include_reasoning and self._reasoning)
        )
        self._apply_extra_body(kwargs, include_reasoning=reasoning_enabled, include_provider=True)
        self._clear_reasoning_trace("text")
        completion = self._client.chat.completions.create(**kwargs)
        self._record_cache_usage("text.create", completion)
        self._record_reasoning_trace("text", "text.create", kwargs, completion)
        if not completion.choices:
            return ""
        message = completion.choices[0].message
        return _message_content_text(message)

    def _request_text_with_model(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
        *,
        include_reasoning: bool | None = None,
        include_provider: bool = False,
    ) -> str:
        kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        reasoning_enabled = (
            bool(self._reasoning) if include_reasoning is None else bool(include_reasoning and self._reasoning)
        )
        self._apply_extra_body(
            kwargs,
            include_reasoning=reasoning_enabled,
            include_provider=bool(include_provider),
        )
        self._clear_reasoning_trace("text_with_model")
        completion = self._client.chat.completions.create(**kwargs)
        self._record_cache_usage("text_with_model.create", completion)
        self._record_reasoning_trace("text_with_model", "text_with_model.create", kwargs, completion)
        if not completion.choices:
            return ""
        message = completion.choices[0].message
        return _message_content_text(message)

    @staticmethod
    def _serialize_payload(payload: Dict[str, Any]) -> str:
        # Compact deterministic JSON reduces token count and keeps prefix matching stable.
        return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))

    @classmethod
    def _canonicalize_for_prompt(cls, value: Any) -> Any:
        if isinstance(value, dict):
            canonical: Dict[Any, Any] = {}
            for key in sorted(value.keys(), key=lambda item: str(item)):
                canonical[key] = cls._canonicalize_for_prompt(value[key])
            return canonical
        if isinstance(value, list):
            return [cls._canonicalize_for_prompt(item) for item in value]
        return value

    @classmethod
    def _stable_participants_payload(cls, participants: List[Participant]) -> List[Dict[str, Any]]:
        payload = [cls._participant_payload(participant) for participant in participants]
        return sorted(
            payload,
            key=lambda item: (
                str(item.get("user_id") or ""),
                str(item.get("username") or ""),
                str(item.get("display_name") or ""),
            ),
        )

    @classmethod
    def _stable_entity_list(cls, entries: List[Any]) -> List[Any]:
        canonical = [cls._canonicalize_for_prompt(entry) for entry in entries]

        def entity_sort_key(item: Any) -> str:
            if isinstance(item, dict):
                stable_json = json.dumps(
                    item,
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                    default=str,
                )
                return "|".join(
                    [
                        str(item.get("uuid") or item.get("target_uuid") or ""),
                        str(item.get("name") or ""),
                        str(item.get("display_name") or ""),
                        stable_json,
                    ]
                )
            return json.dumps(
                item,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
                default=str,
            )

        return sorted(canonical, key=entity_sort_key)

    def _format_address_check(
        self,
        chat: InboundChat,
        persona: str,
        environment: EnvironmentSnapshot | None,
        participants: List[Participant] | None,
        context: ConversationContext | None,
    ) -> str:
        env_block = {
            "location": environment.location if environment else "",
            "is_sitting": environment.is_sitting if environment else False,
            "nearby_agents": [],
            "nearby_objects": [],
            "avatar_position": environment.avatar_position if environment else "",
        }
        if environment:
            for agent in environment.agents[:6]:
                name = str(agent.get("name", ""))
                if name:
                    env_block["nearby_agents"].append(name)
            for obj in environment.objects[:4]:
                name = str(obj.get("name", ""))
                if name:
                    env_block["nearby_objects"].append(name)
            env_block["nearby_agents"].sort(key=str.casefold)
            env_block["nearby_objects"].sort(key=str.casefold)

        participant_names: List[str] = []
        participant_details: List[Dict[str, Any]] = []
        if participants:
            participant_details = self._stable_participants_payload(participants)
            for detail in participant_details:
                username = str(detail.get("username") or detail.get("name") or "").strip()
                if username:
                    participant_names.append(username)

        recent_messages: List[Dict[str, Any]] = []
        summary = ""
        recent_timestamps: List[float] = []
        if context:
            summary = context.summary
            for msg in context.recent_messages[-6:]:
                recent_messages.append(self._chat_payload(msg))
                recent_timestamps.append(float(msg.timestamp or 0.0))

        recent_time_range = {
            "start": format_pacific_time(min(recent_timestamps)) if recent_timestamps else "0",
            "end": format_pacific_time(max(recent_timestamps)) if recent_timestamps else "0",
        }

        payload = {
            "persona": persona,
            "participants": participant_names[:8],
            "participants_detail": participant_details[:8],
            "environment": env_block,
            "previously": summary,
            "summary_meta": self._canonicalize_for_prompt(context.summary_meta if context else {}),
            "agent_state": self._canonicalize_for_prompt(context.agent_state if context else {}),
            "recent_time_range": recent_time_range,
            "recent_messages": recent_messages,
            "sender_name": chat.sender_name,
            "sender_id": chat.sender_id,
            "incoming": self._chat_payload(chat),
            "now_timestamp": format_pacific_time(),
        }
        return self._serialize_payload(payload)

    def _format_context(
        self,
        chat: InboundChat,
        context: ConversationContext,
        overflow: List[InboundChat] | None,
        incoming_batch: List[Dict[str, Any]] | None,
    ) -> str:
        recent_timestamps = [float(m.timestamp or 0.0) for m in context.recent_messages]
        recent_time_range = {
            "start": format_pacific_time(min(recent_timestamps)) if recent_timestamps else "0",
            "end": format_pacific_time(max(recent_timestamps)) if recent_timestamps else "0",
        }
        participants_payload = self._stable_participants_payload(context.participants)
        now = time.time()
        payload = {
            "user_id": context.user_id,
            "participants": participants_payload,
            "environment": {
                "location": context.environment.location,
                "is_sitting": context.environment.is_sitting,
                "objects": self._stable_entity_list(context.environment.objects),
                "agents": self._stable_entity_list(context.environment.agents),
                "avatar_position": context.environment.avatar_position,
            },
            "people_facts": self._canonicalize_for_prompt(context.people_facts),
            "previously": context.summary,
            "summary_meta": self._canonicalize_for_prompt(context.summary_meta),
            "agent_state": self._canonicalize_for_prompt(context.agent_state),
            "related_experiences": self._canonicalize_for_prompt(context.related_experiences),
            "recent_time_range": recent_time_range,
            "recent_messages": [self._chat_payload(m) for m in context.recent_messages],
            "overflow_messages": [self._chat_payload(m) for m in (overflow or [])],
            "incoming_batch": self._canonicalize_for_prompt(incoming_batch or []),
            "incoming": self._chat_payload(chat),
            "now_timestamp": format_pacific_time(now),
        }
        return self._serialize_payload(payload)

    def _format_facts_context_from_messages(
        self,
        evidence_messages: List[InboundChat],
        participants: List[Participant],
        persona: str,
        user_id: str,
        summary: str,
        summary_meta: Dict[str, Any],
        related_experiences: List[Dict[str, Any]],
        people_facts: Dict[str, Any],
    ) -> str:
        trimmed_evidence = list(evidence_messages[-24:])
        incoming = trimmed_evidence[-1]
        evidence_payload: List[Dict[str, Any]] = []
        for message in trimmed_evidence[:-1]:
            evidence_payload.append(self._chat_payload(message))
        filtered_participants, filtered_people_facts = self._filter_facts_entities(
            evidence_messages=trimmed_evidence,
            participants=participants,
            people_facts=people_facts,
        )
        payload = {
            "persona": persona,
            "user_id": user_id,
            "participants": self._stable_participants_payload(filtered_participants),
            "people_facts": self._canonicalize_for_prompt(filtered_people_facts),
            "evidence_messages": evidence_payload,
            "incoming": self._chat_payload(incoming),
            "now_timestamp": format_pacific_time(),
        }
        return self._serialize_payload(payload)

    @classmethod
    def _filter_facts_entities(
        cls,
        evidence_messages: List[InboundChat],
        participants: List[Participant],
        people_facts: Dict[str, Any],
    ) -> tuple[List[Participant], Dict[str, Dict[str, Any]]]:
        text_blob = " ".join(str(message.text or "") for message in evidence_messages)
        sender_ids = {str(message.sender_id or "").strip() for message in evidence_messages if str(message.sender_id or "").strip()}
        sender_names = {
            _normalize_name(str(message.sender_name or ""))
            for message in evidence_messages
            if str(message.sender_name or "").strip()
        }
        sender_names.discard("")

        filtered_people_facts: Dict[str, Dict[str, Any]] = {}
        relevant_user_ids = set(sender_ids)
        for raw_user_id, raw_profile in (people_facts or {}).items():
            user_id = str(raw_user_id or "").strip()
            profile = raw_profile if isinstance(raw_profile, dict) else {}
            include = bool(user_id and user_id in sender_ids)
            if not include:
                include = cls._profile_mentioned_in_text(profile, text_blob)
            if not include:
                continue
            relevant_user_ids.add(user_id)
            filtered_people_facts[user_id] = cls._minimal_people_fact_profile(profile)

        filtered_participants: List[Participant] = []
        seen_participants: set[str] = set()
        for participant in participants or []:
            user_id = str(participant.user_id or "").strip()
            name = str(participant.name or "").strip()
            include = bool(user_id and user_id in relevant_user_ids)
            if not include and not user_id:
                normalized_name = _normalize_name(name)
                include = bool(normalized_name and normalized_name in sender_names)
            if not include:
                include = cls._name_mentioned_in_text(name, text_blob)
            if not include:
                continue
            key = user_id or _normalize_name(name)
            if not key or key in seen_participants:
                continue
            seen_participants.add(key)
            filtered_participants.append(Participant(user_id=user_id, name=name))

        return filtered_participants, filtered_people_facts

    @staticmethod
    def _minimal_people_fact_profile(profile: Dict[str, Any], max_facts: int = 6) -> Dict[str, Any]:
        name = str(
            profile.get("name")
            or profile.get("username")
            or profile.get("display_name")
            or profile.get("full_name")
            or ""
        ).strip()
        facts_raw = profile.get("facts", [])
        facts: List[str] = []
        if isinstance(facts_raw, list):
            for item in facts_raw:
                value = str(item or "").strip()
                if not value:
                    continue
                facts.append(value)
                if len(facts) >= max_facts:
                    break
        return {"name": name, "facts": facts}

    @classmethod
    def _profile_mentioned_in_text(cls, profile: Dict[str, Any], text: str) -> bool:
        for key in ("name", "username", "display_name", "full_name"):
            candidate = str(profile.get(key, "") or "").strip()
            if cls._name_mentioned_in_text(candidate, text):
                return True
        return False

    @staticmethod
    def _name_mentioned_in_text(name: str, text: str) -> bool:
        raw_name = str(name or "").strip()
        if not raw_name or _UUID_LIKE_RE.match(raw_name):
            return False
        normalized_text = _normalize_name(text)
        if not normalized_text:
            return False
        display_name, username = split_display_and_username(raw_name)
        for candidate in (raw_name, display_name, username):
            normalized_candidate = _normalize_name(str(candidate or ""))
            if len(normalized_candidate) < 3:
                continue
            if _UUID_LIKE_RE.match(normalized_candidate):
                continue
            if _name_in_text(normalized_candidate, normalized_text):
                return True
        return False

    def _format_autonomous_context(self, context: ConversationContext, activity: Dict[str, Any]) -> str:
        recent_timestamps = [float(m.timestamp or 0.0) for m in context.recent_messages]
        recent_time_range = {
            "start": format_pacific_time(min(recent_timestamps)) if recent_timestamps else "0",
            "end": format_pacific_time(max(recent_timestamps)) if recent_timestamps else "0",
        }
        participants_payload = self._stable_participants_payload(context.participants)
        now = time.time()
        payload = {
            "mode": "autonomous",
            "user_id": context.user_id,
            "participants": participants_payload,
            "environment": {
                "location": context.environment.location,
                "is_sitting": context.environment.is_sitting,
                "objects": self._stable_entity_list(context.environment.objects),
                "agents": self._stable_entity_list(context.environment.agents),
                "avatar_position": context.environment.avatar_position,
            },
            "people_facts": self._canonicalize_for_prompt(context.people_facts),
            "previously": context.summary,
            "summary_meta": self._canonicalize_for_prompt(context.summary_meta),
            "agent_state": self._canonicalize_for_prompt(context.agent_state),
            "related_experiences": self._canonicalize_for_prompt(context.related_experiences),
            "recent_time_range": recent_time_range,
            "recent_messages": [self._chat_payload(m) for m in context.recent_messages],
            "activity": {
                "seconds_since_activity": activity.get("seconds_since_activity"),
                "last_message_received_at": format_pacific_time(activity.get("last_inbound_ts")),
                "last_ai_response_at": format_pacific_time(activity.get("last_response_ts")),
                "mood": activity.get("mood"),
                "status": activity.get("status"),
            },
            "now_timestamp": format_pacific_time(now),
        }
        return self._serialize_payload(payload)

    @staticmethod
    def _chat_payload(chat: InboundChat) -> Dict[str, Any]:
        raw = chat.raw if isinstance(chat.raw, dict) else {}
        sender_id = str(chat.sender_id or "")
        full_name = (
            str(raw.get("_sender_full_name") or "")
            or str(raw.get("from_name") or raw.get("sender_name") or "")
            or str(chat.sender_name or sender_id)
        )
        display_name, username = split_display_and_username(full_name)
        sender_username = str(raw.get("_sender_username") or "") or username or chat.sender_name or sender_id
        sender_display = str(raw.get("_sender_display_name") or "") or display_name or chat.sender_name or sender_username
        sender_value = sender_username or sender_display or sender_id
        payload = {
            "sender": sender_value,
            "sender_id": sender_id,
            "text": chat.text,
            "timestamp": format_pacific_time(float(chat.timestamp or 0.0)),
        }
        if sender_username and sender_username != sender_value:
            payload["sender_username"] = sender_username
        if sender_display and sender_display not in {sender_value, sender_username}:
            payload["sender_display_name"] = sender_display
        if full_name and full_name not in {sender_value, sender_username, sender_display}:
            payload["sender_full_name"] = full_name
        return payload

    @staticmethod
    def _participant_payload(participant: Participant) -> Dict[str, Any]:
        full_name = str(participant.name or participant.user_id or "")
        display_name, username = split_display_and_username(full_name)
        username_value = username or participant.name or participant.user_id
        display_value = display_name or participant.name or username_value
        payload = {
            "user_id": participant.user_id,
            "name": username_value,
        }
        if username_value and username_value != payload["name"]:
            payload["username"] = username_value
        if display_value and display_value != payload["name"]:
            payload["display_name"] = display_value
        if full_name and full_name not in {payload["name"], display_value}:
            payload["full_name"] = full_name
        return payload


def _bundle_from_structured(parsed: StructuredBundle, mode: str = "chat") -> LLMResponseBundle:
    actions: List[Action] = []
    for action in getattr(parsed, "actions", []) or []:
        actions.extend(_actions_from_structured(action))
    text = _first_chat_text(actions)

    facts = _facts_from_structured(parsed)

    hints = []
    for hint in getattr(parsed, "participant_hints", []) or []:
        hints.append(
            ParticipantHint(
                user_id=getattr(hint, "user_id", ""),
                name=getattr(hint, "name", ""),
            )
        )

    summary_update = getattr(parsed, "summary_update", None)
    summary = summary_update.strip() if isinstance(summary_update, str) and summary_update.strip() else None
    mood = getattr(parsed, "mood", None)
    status = getattr(parsed, "status", None)
    raw_decision = getattr(parsed, "autonomy_decision", None)
    if raw_decision is None:
        raw_decision = getattr(parsed, "decision", None)
    decision = _normalize_autonomy_decision(raw_decision)
    next_delay_seconds = _optional_positive_float(getattr(parsed, "next_delay_seconds", None))
    if mode == "autonomous":
        if actions:
            decision = "act"
        elif decision is None:
            decision = "wait"

    return LLMResponseBundle(
        text=text,
        actions=actions,
        facts=facts,
        participant_hints=hints,
        summary=summary,
        mood=mood.strip() if isinstance(mood, str) and mood.strip() else None,
        status=status.strip() if isinstance(status, str) and status.strip() else None,
        autonomy_decision=decision,
        next_delay_seconds=next_delay_seconds,
    )


def _state_update_from_structured(parsed: StructuredStateUpdate) -> LLMStateUpdate:
    facts = _facts_from_structured(parsed)
    hints = []
    for hint in getattr(parsed, "participant_hints", []) or []:
        hints.append(
            ParticipantHint(
                user_id=getattr(hint, "user_id", ""),
                name=getattr(hint, "name", ""),
            )
        )
    summary_update = getattr(parsed, "summary_update", None)
    summary = summary_update.strip() if isinstance(summary_update, str) and summary_update.strip() else None
    mood = getattr(parsed, "mood", None)
    status = getattr(parsed, "status", None)
    raw_decision = getattr(parsed, "autonomy_decision", None)
    if raw_decision is None:
        raw_decision = getattr(parsed, "decision", None)
    decision = _normalize_autonomy_decision(raw_decision)
    next_delay_seconds = _optional_positive_float(getattr(parsed, "next_delay_seconds", None))
    return LLMStateUpdate(
        facts=facts,
        participant_hints=hints,
        summary_update=summary,
        mood=mood.strip() if isinstance(mood, str) and mood.strip() else None,
        status=status.strip() if isinstance(status, str) and status.strip() else None,
        autonomy_decision=decision,
        next_delay_seconds=next_delay_seconds,
    )


def _facts_from_structured(parsed: Any) -> List[ExtractedFact]:
    extracted: List[ExtractedFact] = []
    for fact in getattr(parsed, "facts", []) or []:
        extracted.append(
            ExtractedFact(
                user_id=getattr(fact, "user_id", ""),
                name=getattr(fact, "name", ""),
                facts=list(getattr(fact, "facts", []) or []),
            )
        )
    return extracted


def _strip_emote_wrapping_asterisks(content: str) -> str:
    cleaned = content.strip()
    while len(cleaned) >= 2 and cleaned.startswith("*") and cleaned.endswith("*"):
        cleaned = cleaned[1:-1].strip()
    return cleaned


def _normalize_text_action_content(command_type: CommandType, content: str) -> str:
    text = str(content or "")
    if command_type != CommandType.EMOTE:
        return text
    return _strip_emote_wrapping_asterisks(text)


def _actions_from_structured(action: StructuredAction) -> List[Action]:
    primary = _command_from_structured(action)
    if primary is None:
        return []
    param_type_raw = str(primary.parameters.get("type", "") or "").upper()
    if not param_type_raw or param_type_raw == primary.command_type.value:
        primary.parameters.pop("type", None)
        return [primary]
    primary.parameters.pop("type", None)
    try:
        param_command = CommandType(param_type_raw)
    except ValueError:
        return [primary]
    # If the model mixed command types, emit both so we do not drop intent.
    secondary_params = dict(primary.parameters)
    secondary_content = _normalize_text_action_content(param_command, primary.content)
    secondary = Action(
        command_type=param_command,
        content=secondary_content,
        x=primary.x,
        y=primary.y,
        z=primary.z,
        target_uuid=primary.target_uuid,
        parameters=secondary_params,
    )
    if param_command in {CommandType.CHAT, CommandType.EMOTE} and secondary.content:
        secondary.parameters["content"] = secondary.content
    return [secondary, primary]


def _command_from_structured(action: StructuredAction) -> Optional[Action]:
    action_type = getattr(action, "type", "")
    if isinstance(action_type, CommandType):
        command_type = action_type
    else:
        raw_type = str(action_type).upper()
        try:
            command_type = CommandType(raw_type)
        except ValueError:
            return None
    parameters_raw = getattr(action, "parameters", {})
    parameters = dict(parameters_raw) if isinstance(parameters_raw, dict) else {}
    content = str(getattr(action, "content", ""))
    if command_type in {CommandType.CHAT, CommandType.EMOTE} and not content:
        fallback_content = parameters.get("content", parameters.get("text"))
        if fallback_content is not None:
            content = str(fallback_content)
    if command_type in {CommandType.CHAT, CommandType.EMOTE} and content:
        content = _normalize_text_action_content(command_type, content)
        parameters["content"] = content
    if command_type in {CommandType.MOVE, CommandType.FACE_TARGET}:
        parameters.setdefault("x", str(getattr(action, "x", 0.0)))
        parameters.setdefault("y", str(getattr(action, "y", 0.0)))
        parameters.setdefault("z", str(getattr(action, "z", 0.0)))
    return Action(
        command_type=command_type,
        content=content,
        x=float(getattr(action, "x", 0.0) or 0.0),
        y=float(getattr(action, "y", 0.0) or 0.0),
        z=float(getattr(action, "z", 0.0) or 0.0),
        target_uuid=str(getattr(action, "target_uuid", "")),
        parameters=parameters,
    )


def _first_chat_text(actions: List[Action]) -> str:
    for action in actions:
        if action.command_type == CommandType.CHAT and action.content:
            return action.content
    return ""


def _normalize_name(name: str) -> str:
    if not name:
        return ""
    base = name.split("(", 1)[0].strip().lower()
    return " ".join(base.split())


def _name_in_text(name: str, text: str) -> bool:
    if not name:
        return False
    if name in text:
        return True
    tokens = [tok for tok in name.split() if len(tok) >= 3]
    for token in tokens:
        if token in text:
            return True
    return False


def _clean_json(text: str) -> str:
    """Attempt to repair common LLM JSON malformations."""
    if not text:
        return ""

    # 1. Remove markdown code blocks if present
    text = re.sub(r"```(?:json)?\s*(.*?)\s*```", r"\1", text, flags=re.DOTALL)

    # 2. Basic preamble/postamble removal: find first { and last }
    # Using a slightly more robust approach for nested structures
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]

    # 3. Remove trailing commas in objects and arrays: e.g., {"a": 1,} -> {"a": 1}
    text = re.sub(r",\s*([\]\}])", r"\1", text)

    # 4. Remove empty or leading commas in arrays: e.g., [ , {"type":...}] -> [{"type":...}]
    text = re.sub(r"\[\s*,\s*", "[", text)

    # 5. Remove multiple consecutive commas
    text = re.sub(r",\s*,+", ",", text)

    return text.strip()


def _collect_other_names(
    persona_norm: str,
    environment: EnvironmentSnapshot | None,
    participants: List[Participant] | None,
    sender_name: str,
) -> List[str]:
    names: List[str] = []

    def add_name(value: str) -> None:
        norm = _normalize_name(value)
        if not norm or norm == persona_norm:
            return
        if norm == _normalize_name(sender_name):
            return
        if len(norm) < 3:
            return
        if not any(ch.isalpha() for ch in norm):
            return
        names.append(norm)

    if participants:
        for participant in participants:
            add_name(participant.name)
    if environment:
        for agent in environment.agents:
            add_name(str(agent.get("name", "")))
    deduped = []
    seen = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        deduped.append(name)
    return deduped


def _parse_bool_response(response: str) -> bool:
    cleaned = (response or "").strip().lower()
    if "true" in cleaned:
        return True
    if "false" in cleaned:
        return False
    return True


def _normalize_autonomy_decision(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    decision = value.strip().lower()
    if decision in {"act", "wait", "sleep"}:
        return decision
    return None


def _optional_positive_float(value: Any) -> float | None:
    try:
        candidate = float(value)
    except (TypeError, ValueError):
        return None
    if candidate <= 0.0:
        return None
    return candidate
