from __future__ import annotations

import asyncio
import json
import logging
import re
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


@dataclass
class LLMStateUpdate:
    facts: List[ExtractedFact]
    participant_hints: List[ParticipantHint]
    summary_update: str | None = None
    mood: str | None = None
    status: str | None = None


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

    class StructuredStateUpdate(BaseModel):
        participant_hints: List[StructuredParticipantHint] = Field(default_factory=list)
        facts: List[StructuredFact] = Field(default_factory=list)
        summary_update: Optional[str] = None
        mood: Optional[str] = None
        status: Optional[str] = None

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
        response_text = f"{chat.sender_name}, I hear you."
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
        self._address_model = address_model or model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._timeout = timeout_seconds
        self._facts_in_bundle = bool(facts_in_bundle)
        self._reasoning = reasoning
        self._fallback = fallback or RuleBasedLLMClient()
        actual_base_url = self._base_url if self._base_url else None
        self._client = OpenAI(api_key=self._api_key, base_url=actual_base_url, timeout=self._timeout)

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
            facts_only = await asyncio.to_thread(self._request_facts_from_chat, chat, context)
            if facts_only is not None:
                update.facts = _facts_from_structured(facts_only)
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
            "- Suggest 'participant_hints' for new or important individuals mentioned in the chat."
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
            "- Optionally extract durable person facts (names, relationships, long-term preferences).\n\n"
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
            "- Do not refer to yourself in the third person.\n\n"
            "# TECHNICAL CONSTRAINTS\n"
            "- OUTPUT SCHEMA: You must strictly adhere to the provided JSON schema.\n"
            "- ACTION TYPES: Only use [CHAT, EMOTE, MOVE, TOUCH, SIT, STAND].\n"
            "- ACTION KEYS: Every action item MUST use the key 'type'. Never use 'command' or 'action' as keys.\n"
            "- PARAMETERS: Do not place command types inside the 'parameters' dictionary.\n"
            "- Include mood (short label) and status (brief current activity) when you take action."
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
            if completion and completion.choices:
                raw_text = completion.choices[0].message.content

            if not raw_text:
                # If parse() failed due to ValidationError, it won't have the completion object.
                # We fetch the raw text by performing a standard create() call without parsing.
                try:
                    debug_kwargs = {k: v for k, v in kwargs.items() if k != "response_format"}
                    debug_completion = self._client.chat.completions.create(**debug_kwargs)
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
            "model": self._model,
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
        )
        return _facts_from_structured(parsed) if parsed is not None else []

    def _request_text(self, system_prompt: str, user_prompt: str) -> str:
        kwargs = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
        }
        if self._reasoning:
            kwargs["extra_body"] = {
                "include_reasoning": True,
                "reasoning_effort": self._reasoning,
            }
        completion = self._client.chat.completions.create(**kwargs)
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
        if not completion.choices:
            return ""
        message = completion.choices[0].message
        return message.content or ""

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
            "avatar_position": environment.avatar_position if environment else "",
            "is_sitting": environment.is_sitting if environment else False,
            "nearby_agents": [],
            "nearby_objects": [],
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

        participant_names: List[str] = []
        participant_details: List[Dict[str, Any]] = []
        if participants:
            for participant in participants:
                detail = self._participant_payload(participant)
                participant_details.append(detail)
                if detail["username"]:
                    participant_names.append(detail["username"])

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
            "now_timestamp": format_pacific_time(),
            "persona": persona,
            "sender_name": chat.sender_name,
            "sender_id": chat.sender_id,
            "incoming": self._chat_payload(chat),
            "participants": participant_names[:8],
            "participants_detail": participant_details[:8],
            "environment": env_block,
            "summary": summary,
            "summary_meta": context.summary_meta if context else {},
            "agent_state": context.agent_state if context else {},
            "recent_time_range": recent_time_range,
            "recent_messages": recent_messages,
        }
        return json.dumps(payload, ensure_ascii=True)

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
        now = time.time()
        payload = {
            "now_timestamp": format_pacific_time(now),
            "current_time_iso": format_pacific_time(now),
            "persona": context.persona,
            "user_id": context.user_id,
            "participants": [self._participant_payload(p) for p in context.participants],
            "environment": {
                "location": context.environment.location,
                "avatar_position": context.environment.avatar_position,
                "is_sitting": context.environment.is_sitting,
                "agents": context.environment.agents,
                "objects": context.environment.objects,
            },
            "people_facts": context.people_facts,
            "summary": context.summary,
            "summary_meta": context.summary_meta,
            "agent_state": context.agent_state,
            "persona_instructions": context.persona_instructions,
            "recent_time_range": recent_time_range,
            "recent_messages": [
                self._chat_payload(m) for m in context.recent_messages
            ],
            "related_experiences": context.related_experiences,
            "incoming": self._chat_payload(chat),
            "overflow_messages": [
                self._chat_payload(m) for m in (overflow or [])
            ],
            "incoming_batch": incoming_batch or [],
        }
        return json.dumps(payload, ensure_ascii=True)

    def _format_facts_context_from_messages(
        self,
        evidence_messages: List[InboundChat],
        participants: List[Participant],
        persona: str,
        user_id: str,
        summary: str,
        summary_meta: Dict[str, Any],
        related_experiences: List[Dict[str, Any]],
    ) -> str:
        evidence_payload: List[Dict[str, Any]] = []
        for message in evidence_messages[-24:]:
            evidence_payload.append(self._chat_payload(message))
        incoming = evidence_messages[-1]
        payload = {
            "now_timestamp": format_pacific_time(),
            "persona": persona,
            "user_id": user_id,
            "participants": [self._participant_payload(p) for p in participants],
            "incoming": self._chat_payload(incoming),
            "evidence_messages": evidence_payload,
            # Included for transparency, but the system prompt explicitly forbids using them for facts.
            "summary": summary,
            "summary_meta": summary_meta,
            "related_experiences": related_experiences,
            "persona_instructions": "",
        }
        return json.dumps(payload, ensure_ascii=True)

    def _format_autonomous_context(self, context: ConversationContext, activity: Dict[str, Any]) -> str:
        recent_timestamps = [float(m.timestamp or 0.0) for m in context.recent_messages]
        recent_time_range = {
            "start": format_pacific_time(min(recent_timestamps)) if recent_timestamps else "0",
            "end": format_pacific_time(max(recent_timestamps)) if recent_timestamps else "0",
        }
        now = time.time()
        payload = {
            "mode": "autonomous",
            "now_timestamp": format_pacific_time(now),
            "current_time_iso": format_pacific_time(now),
            "persona": context.persona,
            "user_id": context.user_id,
            "activity": {
                "seconds_since_activity": activity.get("seconds_since_activity"),
                "last_message_received_at": format_pacific_time(activity.get("last_inbound_ts")),
                "last_ai_response_at": format_pacific_time(activity.get("last_response_ts")),
                "mood": activity.get("mood"),
                "status": activity.get("status"),
            },
            "participants": [self._participant_payload(p) for p in context.participants],
            "environment": {
                "location": context.environment.location,
                "avatar_position": context.environment.avatar_position,
                "is_sitting": context.environment.is_sitting,
                "agents": context.environment.agents,
                "objects": context.environment.objects,
            },
            "people_facts": context.people_facts,
            "summary": context.summary,
            "summary_meta": context.summary_meta,
            "agent_state": context.agent_state,
            "persona_instructions": context.persona_instructions,
            "recent_time_range": recent_time_range,
            "recent_messages": [
                self._chat_payload(m) for m in context.recent_messages
            ],
            "related_experiences": context.related_experiences,
        }
        return json.dumps(payload, ensure_ascii=True)

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
        return {
            "sender": sender_value,
            "sender_id": sender_id,
            "sender_username": sender_username,
            "sender_display_name": sender_display,
            "sender_full_name": full_name,
            "text": chat.text,
            "timestamp": format_pacific_time(float(chat.timestamp or 0.0)),
        }

    @staticmethod
    def _participant_payload(participant: Participant) -> Dict[str, Any]:
        full_name = str(participant.name or participant.user_id or "")
        display_name, username = split_display_and_username(full_name)
        username_value = username or participant.name or participant.user_id
        display_value = display_name or participant.name or username_value
        return {
            "user_id": participant.user_id,
            "name": username_value,
            "username": username_value,
            "display_name": display_value,
            "full_name": full_name,
        }


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

    return LLMResponseBundle(
        text=text,
        actions=actions,
        facts=facts,
        participant_hints=hints,
        summary=summary,
        mood=mood.strip() if isinstance(mood, str) and mood.strip() else None,
        status=status.strip() if isinstance(status, str) and status.strip() else None,
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
    return LLMStateUpdate(
        facts=facts,
        participant_hints=hints,
        summary_update=summary,
        mood=mood.strip() if isinstance(mood, str) and mood.strip() else None,
        status=status.strip() if isinstance(status, str) and status.strip() else None,
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
    return False


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
