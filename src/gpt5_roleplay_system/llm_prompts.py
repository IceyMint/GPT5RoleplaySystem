from __future__ import annotations

import json
import re
import time
from typing import Any

from .models import ConversationContext, EnvironmentSnapshot, InboundChat, Participant
from .name_utils import split_display_and_username
from .time_utils import format_pacific_time

_UUID_LIKE_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    flags=re.IGNORECASE,
)


class PromptManager:
    def system_prompt_for_context(self, context: ConversationContext) -> str:
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

    def state_system_prompt_for_context(self, context: ConversationContext) -> str:
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

    def autonomous_system_prompt_for_context(self, context: ConversationContext) -> str:
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

    def facts_system_prompt(self) -> str:
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

    def format_address_check(
        self,
        chat: InboundChat,
        persona: str,
        environment: EnvironmentSnapshot | None,
        participants: list[Participant] | None,
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

        participant_names: list[str] = []
        participant_details: list[dict[str, Any]] = []
        if participants:
            participant_details = self._stable_participants_payload(participants)
            for detail in participant_details:
                username = str(detail.get("username") or detail.get("name") or "").strip()
                if username:
                    participant_names.append(username)

        recent_messages: list[dict[str, Any]] = []
        summary = ""
        recent_timestamps: list[float] = []
        if context:
            summary = context.summary
            for msg in context.recent_messages[-6:]:
                recent_messages.append(self.chat_payload(msg))
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
            "incoming": self.chat_payload(chat),
            "now_timestamp": format_pacific_time(),
        }
        return self._serialize_payload(payload)

    def format_context(
        self,
        chat: InboundChat,
        context: ConversationContext,
        overflow: list[InboundChat] | None,
        incoming_batch: list[dict[str, Any]] | None,
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
            "summary": context.summary,
            "summary_meta": self._canonicalize_for_prompt(context.summary_meta),
            "agent_state": self._canonicalize_for_prompt(context.agent_state),
            "related_experiences": self._canonicalize_for_prompt(context.related_experiences),
            "recent_time_range": recent_time_range,
            "recent_messages": [self.chat_payload(m) for m in context.recent_messages],
            "overflow_messages": [self.chat_payload(m) for m in (overflow or [])],
            "incoming_batch": self._canonicalize_for_prompt(incoming_batch or []),
            "incoming": self.chat_payload(chat),
            "now_timestamp": format_pacific_time(now),
        }
        return self._serialize_payload(payload)

    def format_facts_context_from_messages(
        self,
        evidence_messages: list[InboundChat],
        participants: list[Participant],
        persona: str,
        user_id: str,
        summary: str,
        summary_meta: dict[str, Any],
        related_experiences: list[dict[str, Any]],
        people_facts: dict[str, Any],
    ) -> str:
        trimmed_evidence = list(evidence_messages[-24:])
        incoming = trimmed_evidence[-1]
        evidence_payload: list[dict[str, Any]] = []
        for message in trimmed_evidence[:-1]:
            evidence_payload.append(self.chat_payload(message))
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
            "incoming": self.chat_payload(incoming),
            "now_timestamp": format_pacific_time(),
        }
        return self._serialize_payload(payload)

    def format_autonomous_context(
        self,
        context: ConversationContext,
        activity: dict[str, Any],
    ) -> str:
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
            "summary": context.summary,
            "summary_meta": self._canonicalize_for_prompt(context.summary_meta),
            "agent_state": self._canonicalize_for_prompt(context.agent_state),
            "related_experiences": self._canonicalize_for_prompt(context.related_experiences),
            "recent_time_range": recent_time_range,
            "recent_messages": [self.chat_payload(m) for m in context.recent_messages],
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
    def chat_payload(chat: InboundChat) -> dict[str, Any]:
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
    def participant_payload(participant: Participant) -> dict[str, Any]:
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

    @staticmethod
    def _serialize_payload(payload: dict[str, Any]) -> str:
        return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))

    @classmethod
    def _canonicalize_for_prompt(cls, value: Any) -> Any:
        if isinstance(value, dict):
            canonical: dict[Any, Any] = {}
            for key in sorted(value.keys(), key=lambda item: str(item)):
                canonical[key] = cls._canonicalize_for_prompt(value[key])
            return canonical
        if isinstance(value, list):
            return [cls._canonicalize_for_prompt(item) for item in value]
        return value

    @classmethod
    def _stable_participants_payload(cls, participants: list[Participant]) -> list[dict[str, Any]]:
        payload = [cls.participant_payload(participant) for participant in participants]
        return sorted(
            payload,
            key=lambda item: (
                str(item.get("user_id") or ""),
                str(item.get("username") or ""),
                str(item.get("display_name") or ""),
            ),
        )

    @classmethod
    def _stable_entity_list(cls, entries: list[Any]) -> list[Any]:
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

    @classmethod
    def _filter_facts_entities(
        cls,
        evidence_messages: list[InboundChat],
        participants: list[Participant],
        people_facts: dict[str, Any],
    ) -> tuple[list[Participant], dict[str, dict[str, Any]]]:
        text_blob = " ".join(str(message.text or "") for message in evidence_messages)
        sender_ids = {str(message.sender_id or "").strip() for message in evidence_messages if str(message.sender_id or "").strip()}
        sender_names = {
            cls._normalize_name(str(message.sender_name or ""))
            for message in evidence_messages
            if str(message.sender_name or "").strip()
        }
        sender_names.discard("")

        filtered_people_facts: dict[str, dict[str, Any]] = {}
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

        filtered_participants: list[Participant] = []
        seen_participants: set[str] = set()
        for participant in participants or []:
            user_id = str(participant.user_id or "").strip()
            name = str(participant.name or "").strip()
            include = bool(user_id and user_id in relevant_user_ids)
            if not include and not user_id:
                normalized_name = cls._normalize_name(name)
                include = bool(normalized_name and normalized_name in sender_names)
            if not include:
                include = cls._name_mentioned_in_text(name, text_blob)
            if not include:
                continue
            key = user_id or cls._normalize_name(name)
            if not key or key in seen_participants:
                continue
            seen_participants.add(key)
            filtered_participants.append(Participant(user_id=user_id, name=name))

        return filtered_participants, filtered_people_facts

    @staticmethod
    def _minimal_people_fact_profile(profile: dict[str, Any], max_facts: int = 6) -> dict[str, Any]:
        name = str(
            profile.get("name")
            or profile.get("username")
            or profile.get("display_name")
            or profile.get("full_name")
            or ""
        ).strip()
        facts_raw = profile.get("facts", [])
        facts: list[str] = []
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
    def _profile_mentioned_in_text(cls, profile: dict[str, Any], text: str) -> bool:
        for key in ("name", "username", "display_name", "full_name"):
            candidate = str(profile.get(key, "") or "").strip()
            if cls._name_mentioned_in_text(candidate, text):
                return True
        return False

    @staticmethod
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

    @staticmethod
    def _normalize_name(name: str) -> str:
        if not name:
            return ""
        base = name.split("(", 1)[0].strip().lower()
        return " ".join(base.split())

    @classmethod
    def _name_mentioned_in_text(cls, name: str, text: str) -> bool:
        raw_name = str(name or "").strip()
        if not raw_name or _UUID_LIKE_RE.match(raw_name):
            return False
        normalized_text = cls._normalize_name(text)
        if not normalized_text:
            return False
        display_name, username = split_display_and_username(raw_name)
        for candidate in (raw_name, display_name, username):
            normalized_candidate = cls._normalize_name(str(candidate or ""))
            if len(normalized_candidate) < 3:
                continue
            if _UUID_LIKE_RE.match(normalized_candidate):
                continue
            if cls._name_in_text(normalized_candidate, normalized_text):
                return True
        return False

    @staticmethod
    def _system_prompt() -> str:
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
            "- If you have nothing meaningful to add, you may return empty text and no actions.\n"
            "- Use 'CHAT' for dialogue and 'EMOTE' for physical descriptions or internal states expressed outwardly.\n"
            "- For complex maneuvers (e.g., walking while talking), emit multiple actions in a single response.\n"
            "- When 'incoming_batch' is provided, prioritize the 'latest_text' but use earlier messages for context or corrections.\n\n"
            "# TECHNICAL CONSTRAINTS\n"
            "- OUTPUT SCHEMA: You must strictly adhere to the provided JSON schema.\n"
            "- ACTION TYPES: Only use [CHAT, EMOTE, MOVE, TOUCH, SIT, STAND, FACE_TARGET].\n"
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

    @staticmethod
    def _state_system_prompt() -> str:
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

    @staticmethod
    def _autonomous_system_prompt() -> str:
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
            "- It is acceptable to return no actions and empty text if no action is appropriate.\n"
            "- Never output internal monologue, private reasoning, or narration about waiting.\n"
            "- When you 'CHAT', speak outwardly to nearby people or the environment.\n"
            "- Do not refer to yourself in the third person.\n"
            "\n"
            "# TECHNICAL CONSTRAINTS\n"
            "- OUTPUT SCHEMA: You must strictly adhere to the provided JSON schema.\n"
            "- ACTION TYPES: Only use [CHAT, EMOTE, MOVE, TOUCH, SIT, STAND, FACE_TARGET].\n"
            "- ACTION KEYS: Every action item MUST use the key 'type'. Never use 'command' or 'action' as keys.\n"
            "- PARAMETERS: Do not place command types inside the 'parameters' dictionary.\n"
            "- Include 'autonomy_decision' as one of [act, wait, sleep].\n"
            "- 'act': emit one or more actions.\n"
            "- 'wait': emit no actions and choose a suitable 'next_delay_seconds'.\n"
            "- 'sleep': emit no actions and choose a longer 'next_delay_seconds'.\n"
            "- Include 'next_delay_seconds' whenever possible so the scheduler can pick a natural next check.\n"
            "- Include mood (short label) and status (brief current activity)."
        ) + "\n\n# IMPORTANT: RESPONSE FORMAT\n- You must respond ONLY with a valid JSON object matching the schema.\n- DO NOT include any preamble, conversational filler, or markdown formatting outside the JSON object."
