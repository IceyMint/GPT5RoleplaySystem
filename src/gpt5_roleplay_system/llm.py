from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time
from dataclasses import dataclass
from .time_utils import format_pacific_time
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None

try:
    from pydantic import AliasChoices, BaseModel, Field
except ImportError:  # pragma: no cover - optional dependency
    BaseModel = None
    Field = None
    AliasChoices = None

from .models import Action, CommandType, ConversationContext, EnvironmentSnapshot, InboundChat, Participant
from .name_utils import split_display_and_username

logger = logging.getLogger("gpt5_roleplay_llm")


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
    for key in ("prompt_tokens", "completion_tokens", "total_tokens", "prompt_tokens_details", "cache_discount"):
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
        name: str
        facts: List[str] = Field(default_factory=list)

    class StructuredParticipantHint(BaseModel):
        user_id: str = ""
        name: str = ""

    class StructuredBundle(BaseModel):
        text: str = ""
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

    async def summarize(self, summary: str, messages: List[InboundChat]) -> str:
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
            summary = await self.summarize(context.summary, overflow)
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

    async def summarize(self, summary: str, messages: List[InboundChat]) -> str:
        return summary

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

    async def summarize(self, summary: str, messages: List[InboundChat]) -> str:
        return summary

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
        address_model: str | None = None,
        max_tokens: int = 500,
        temperature: float = 0.6,
        timeout_seconds: float = 30.0,
        facts_in_bundle: bool = True,
        fallback: Optional[LLMClient] = None,
        reasoning: str = "",
    ) -> None:
        if OpenAI is None:
            raise RuntimeError("openai package is not installed")
        if BaseModel is None or StructuredBundle is None:
            raise RuntimeError("pydantic package is not installed")
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._bundle_model = bundle_model or model
        self._address_model = address_model or model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._timeout = timeout_seconds
        self._facts_in_bundle = bool(facts_in_bundle)
        self._reasoning = reasoning
        self._fallback = fallback or RuleBasedLLMClient()
        actual_base_url = self._base_url if self._base_url else None
        self._client = OpenAI(api_key=self._api_key, base_url=actual_base_url, timeout=self._timeout)

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

    async def summarize(self, summary: str, messages: List[InboundChat]) -> str:
        if not self._api_key:
            return await self._fallback.summarize(summary, messages)
        content = {
            "existing_summary": summary,
            "messages": [
                {"sender": msg.sender_name, "text": msg.text} for msg in messages
            ],
        }
        prompt = "Summarize the conversation for long-term memory, concise and factual."
        payload = await asyncio.to_thread(self._request_text, prompt, json.dumps(content, ensure_ascii=True))
        return payload.strip() if payload else summary

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
            "You are the roleplay persona described in the input. Your primary goal is to provide immersive, "
            "in-character responses. Treat the 'persona' and 'persona_instructions' fields as your absolute identity.\n\n"
            "# BEHAVIORAL GUIDELINES\n"
            "- Respond to the conversation context and environment naturally.\n"
            "- Grounding: Only interact with objects (TOUCH, SIT) or mention people explicitly listed in the 'environment' or 'participants' fields. If an object is not in the list, it does not exist for you.\n"
            "- Current time is provided in Pacific Time (America/Los_Angeles).\n"
            "- 'last_message_received_at' and 'last_ai_response_at' indicate the timing of the most recent interactions.\n"
            "- Spatial context (coordinates) is provided in meters. Use this to judge proximity.\n"
            "- Consider yourself 'at' or 'inside' an object/location if you are within 1.0 meter of its coordinates.\n"
            "- If you have nothing meaningful to add, you may return empty text and no actions.\n"
            "- Use 'CHAT' for dialogue and 'EMOTE' for physical descriptions or internal states expressed outwardly.\n"
            "- For complex maneuvers (e.g., walking while talking), emit multiple actions in a single response.\n"
            "- When 'incoming_batch' is provided, prioritize the 'latest_text' but use earlier messages for context or corrections.\n\n"
            "# TECHNICAL CONSTRAINTS\n"
            "- OUTPUT SCHEMA: You must strictly adhere to the provided JSON schema.\n"
            "- ACTION TYPES: Only use [CHAT, EMOTE, MOVE, TOUCH, SIT, STAND].\n"
            "- ACTION KEYS: Every action item MUST use the key 'type'. Never use 'command' or 'action' as keys.\n"
            "- PARAMETERS: Do not place command types inside the 'parameters' dictionary.\n"
            "- DO NOT mix multiple commands into one action item.\n\n"
            "# MEMORY & KNOWLEDGE\n"
            "- Use 'related_experiences' to inform your behavior based on past events.\n"
            "- Update 'summary_update' if 'overflow_messages' are present to compress older context.\n"
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
            "- Do not derive facts from 'summary' or 'related_experiences'.\n"
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
            "- If 'overflow_messages' are present, provide a concise 'summary_update' to compress older context.\n"
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
            "You are the roleplay persona described in the input, deciding whether to act autonomously. "
            "Treat the 'persona' field as your identity and voice. You are this persona.\n\n"
            "# BEHAVIORAL GUIDELINES\n"
            "- Only act when it makes sense given recent activity, environment, and relationships.\n"
            "- Grounding: Only interact with objects (TOUCH, SIT) or mention people explicitly listed in the 'environment' or 'participants' fields. If an object is not in the list, it does not exist for you.\n"
            "- Current time is provided in Pacific Time (America/Los_Angeles).\n"
            "- 'last_message_received_at' and 'last_ai_response_at' indicate the timing of the most recent interactions.\n"
            "- Spatial context (coordinates) is provided in meters. Use this to judge proximity.\n"
            "- Consider yourself 'at' or 'inside' an object/location if you are within 1.0 meter of its coordinates.\n"
            "- It is acceptable to return no actions and empty text if no action is appropriate.\n"
            "- Never output internal monologue, private reasoning, or narration about waiting.\n"
            "- When you 'CHAT', speak outwardly to nearby people or the environment.\n"
            "- Do not refer to yourself in the third person.\n"
            "\n"
            "# TECHNICAL CONSTRAINTS\n"
            "- OUTPUT SCHEMA: You must strictly adhere to the provided JSON schema.\n"
            "- ACTION TYPES: Only use [CHAT, EMOTE, MOVE, TOUCH, SIT, STAND].\n"
            "- ACTION KEYS: Every action item MUST use the key 'type'. Never use 'command' or 'action' as keys.\n"
            "- PARAMETERS: Do not place command types inside the 'parameters' dictionary.\n"
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

    def _request_structured(self, model_class: Any, kwargs: Dict[str, Any]) -> Optional[Any]:
        """Execute a structured parse request with fallback cleanup for malformed JSON."""
        try:
            completion = self._client.chat.completions.parse(**kwargs)
            self._record_cache_usage("structured.parse", completion)
            if not completion.choices:
                return None
            message = completion.choices[0].message
            if getattr(message, "refusal", None):
                return None
            return getattr(message, "parsed", None)
        except Exception as exc:
            raw_text = None
            # Check for truncation or content filter in API error
            completion = getattr(exc, "completion", None)
            if completion is not None:
                self._record_cache_usage("structured.parse_error", completion)
            if completion and completion.choices:
                raw_text = completion.choices[0].message.content

            if not raw_text:
                # If parse() failed due to ValidationError, it won't have the completion object.
                # We fetch the raw text by performing a standard create() call without parsing.
                try:
                    debug_kwargs = {k: v for k, v in kwargs.items() if k != "response_format"}
                    debug_completion = self._client.chat.completions.create(**debug_kwargs)
                    self._record_cache_usage("structured.debug_create", debug_completion)
                    raw_text = debug_completion.choices[0].message.content
                except Exception as debug_exc:
                    logger.error("Failed to fetch raw text for cleanup: %s", debug_exc)

            if raw_text:
                try:
                    cleaned = _clean_json(raw_text)
                    return model_class.model_validate_json(cleaned)
                except Exception as final_exc:
                    logger.warning("Robust parse failed even after cleanup: %s. Raw:\n%s", final_exc, raw_text)
            
            logger.warning("Structured request failed (%s): %s", exc.__class__.__name__, exc)
            return None

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
        if self._reasoning:
            kwargs["extra_body"] = {
                "include_reasoning": True,
                "reasoning_effort": self._reasoning,
        }
        return self._request_structured(StructuredBundle, kwargs)

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
        if self._reasoning:
            kwargs["extra_body"] = {
                "include_reasoning": True,
                "reasoning_effort": self._reasoning,
            }
        return self._request_structured(StructuredStateUpdate, kwargs)

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
        if self._reasoning:
            kwargs["extra_body"] = {
                "include_reasoning": True,
                "reasoning_effort": self._reasoning,
            }
        return self._request_structured(StructuredBundle, kwargs)

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
            "model": self._model,
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
        if self._reasoning:
            kwargs["extra_body"] = {
                "include_reasoning": True,
                "reasoning_effort": self._reasoning,
            }
        return self._request_structured(StructuredFactsOnly, kwargs)

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
        reasoning_enabled = bool(self._reasoning) if include_reasoning is None else bool(include_reasoning and self._reasoning)
        if reasoning_enabled:
            kwargs["extra_body"] = {
                "include_reasoning": True,
                "reasoning_effort": self._reasoning,
            }
        completion = self._client.chat.completions.create(**kwargs)
        self._record_cache_usage("text.create", completion)
        if not completion.choices:
            return ""
        message = completion.choices[0].message
        return message.content or ""

    def _request_text_with_model(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
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
        if self._reasoning:
            kwargs["extra_body"] = {
                "include_reasoning": True,
                "reasoning_effort": self._reasoning,
            }
        completion = self._client.chat.completions.create(**kwargs)
        self._record_cache_usage("text_with_model.create", completion)
        if not completion.choices:
            return ""
        message = completion.choices[0].message
        return message.content or ""

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
            "summary": summary,
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
            "persona": context.persona,
            "user_id": context.user_id,
            "persona_instructions": context.persona_instructions,
            "participants": participants_payload,
            "environment": {
                "location": context.environment.location,
                "is_sitting": context.environment.is_sitting,
                "objects": self._stable_entity_list(context.environment.objects),
                "agents": self._stable_entity_list(context.environment.agents),
                "avatar_position": context.environment.avatar_position,
            },
            "people_facts": self._canonicalize_for_prompt(context.people_facts),
            "summary": context.summary,
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
        payload = {
            "persona": persona,
            "user_id": user_id,
            "participants": self._stable_participants_payload(participants),
            "people_facts": self._canonicalize_for_prompt(people_facts),
            "evidence_messages": evidence_payload,
            "incoming": self._chat_payload(incoming),
            "now_timestamp": format_pacific_time(),
        }
        return self._serialize_payload(payload)

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
            "persona": context.persona,
            "user_id": context.user_id,
            "persona_instructions": context.persona_instructions,
            "participants": participants_payload,
            "environment": {
                "location": context.environment.location,
                "is_sitting": context.environment.is_sitting,
                "objects": self._stable_entity_list(context.environment.objects),
                "agents": self._stable_entity_list(context.environment.agents),
                "avatar_position": context.environment.avatar_position,
            },
            "people_facts": self._canonicalize_for_prompt(context.people_facts),
            "summary": context.summary,
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
    text = getattr(parsed, "text", "") or ""
    actions: List[Action] = []
    for action in getattr(parsed, "actions", []) or []:
        actions.extend(_actions_from_structured(action))
    if mode != "autonomous" and not actions and text:
        actions = [Action(command_type=CommandType.CHAT, content=text, parameters={"content": text})]
    if not text and actions:
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
    secondary = Action(
        command_type=param_command,
        content=primary.content,
        x=primary.x,
        y=primary.y,
        z=primary.z,
        target_uuid=primary.target_uuid,
        parameters=secondary_params,
    )
    if param_command in {CommandType.CHAT, CommandType.EMOTE} and secondary.content:
        secondary.parameters.setdefault("content", secondary.content)
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
    if command_type in {CommandType.CHAT, CommandType.EMOTE} and content:
        parameters.setdefault("content", content)
    if command_type in {CommandType.MOVE}:
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
    stopwords = {"the", "and", "you", "hey", "hi"}
    names: List[str] = []

    def add_name(value: str) -> None:
        norm = _normalize_name(value)
        if not norm or norm == persona_norm:
            return
        if norm == _normalize_name(sender_name):
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
        if name in stopwords or name in seen:
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
