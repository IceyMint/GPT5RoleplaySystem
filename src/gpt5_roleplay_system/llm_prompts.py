from __future__ import annotations

import json
import math
import re
import time
from typing import Any

from .models import ConversationContext, EnvironmentSnapshot, InboundChat, Participant
from .name_utils import split_display_and_username
from .time_utils import format_pacific_time, get_pacific_time

_UUID_LIKE_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    flags=re.IGNORECASE,
)


class PromptManager:
    def system_prompt_for_context(self, context: ConversationContext) -> str:
        persona = (context.persona or "").strip()
        instructions = (context.persona_instructions or "").strip()
        lines = [
            self.system_prompt(),
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
            self.state_system_prompt(),
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
            self.autonomous_system_prompt(),
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
            "- Do not derive facts from 'previously' or 'related_experiences'.\n"
            "- Use 'people_facts' to avoid duplicates; if a fact already exists or is a close paraphrase, omit it.\n"
            "- Only include facts that are likely to remain true for weeks or months.\n"
            "- If no new durable facts are found, return an empty facts list."
        ) + "\n\n# IMPORTANT: RESPONSE FORMAT\n- You must respond ONLY with a valid JSON object matching the schema.\n- DO NOT include any preamble, conversational filler, or markdown formatting outside the JSON object."

    @staticmethod
    def address_check_system_prompt() -> str:
        return (
            "You are a fast classifier. Decide if the message is addressed to the AI persona. "
            "Consider nicknames, phonetic spellings, typos, and direct mentions. "
            "Use conversation context, recent messages, and spatial proximity to judge intent. "
            "If the incoming message only acknowledges the AI's previous line and adds no new request, question, or topic, treat it as not requiring a reply. "
            "Coordinates (x, y, z) are in meters. "
            "Reply with only 'true' or 'false'."
        )

    @staticmethod
    def continuity_summary_system_prompt() -> str:
        return (
            "Update the continuity summary by combining 'existing_summary' (older history) and 'messages' (new events). "
            "Rewrite the entire summary from scratch as concise plain text for future model context. "
            "Output MUST use these exact section headers in this order: "
            "'PRIOR CONTEXT', 'TIMELINE', 'CURRENT STATE', 'OPEN THREADS', 'MOST RECENT CONFIRMED'. "
            "Under 'PRIOR CONTEXT', keep only durable older context that is still relevant, or write '- none'. "
            "Under 'TIMELINE', write 1-8 bullets in strict chronological order using message timestamps, "
            "formatted as '[<timestamp_unix>] <event or state change>'. "
            "Under 'CURRENT STATE', list only the latest confirmed state that is still true. "
            "Under 'OPEN THREADS', list unresolved questions, promises, tensions, or write '- none'. "
            "Under 'MOST RECENT CONFIRMED', write exactly two bullets: "
            "'- as_of: <latest timestamp_unix>' and '- state: <latest confirmed state>'. "
            "Separate past events from current truth clearly. "
            "If newer events supersede earlier states, keep the earlier event in 'TIMELINE' and reflect the latest truth in 'CURRENT STATE'. "
            "Use the provided sender labels exactly as written; do not replace them with generic labels like 'User' or 'Assistant' unless the input itself uses those labels. "
            "When evidence is sparse or ambiguous, prefer omission over guessing. "
            "Do not infer relationships, identities, motives, or off-screen events unless the messages explicitly support them. "
            "If a new message is only a greeting, emote, or short utterance, summarize only that minimal confirmed event. "
            "Keep it concise, factual, and free of narrative filler. "
            "Do not invent events."
        )

    @staticmethod
    def episodic_summary_system_prompt() -> str:
        return (
            "Create an episodic memory summary for long-term retrieval from the provided message sequence. "
            "Keep strict chronology and focus on concrete events, key participants, important state changes, "
            "commitments/promises, outcomes, and unresolved threads. "
            "Keep it factual, compact (about 5-10 sentences), and specific with names. "
            "Do not invent events."
        )

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

        static_payload = {
            "persona": persona,
            "participants": participant_names[:8],
            "participants_detail": participant_details[:8],
            "previously": summary,
            "summary_meta": self._canonicalize_for_prompt(context.summary_meta if context else {}),
            "agent_state": self._canonicalize_for_prompt(context.agent_state if context else {}),
            "environment": env_block,
        }
        live_payload = {
            "recent_time_range": recent_time_range,
            "recent_messages": recent_messages,
            "sender_name": chat.sender_name,
            "sender_id": chat.sender_id,
            "incoming": self.chat_payload(chat),
            "now_timestamp": format_pacific_time(),
        }
        payload = self._merge_prompt_sections(static_payload, live_payload)
        return self._serialize_payload(payload)

    def format_context(
        self,
        chat: InboundChat,
        context: ConversationContext,
        overflow: list[InboundChat] | None,
        incoming_batch: list[dict[str, Any]] | None,
    ) -> str:
        now = time.time()
        stable_people_facts, people_recency = self.split_people_facts_prompt_sections(context.people_facts, now_ts=now)
        stable_agent_state, agent_timing = self.split_agent_state_prompt_sections(context.agent_state)
        recent_timestamps = [float(m.timestamp or 0.0) for m in context.recent_messages]
        recent_time_range = {
            "start": format_pacific_time(min(recent_timestamps)) if recent_timestamps else "0",
            "end": format_pacific_time(max(recent_timestamps)) if recent_timestamps else "0",
        }
        participants_payload = self._stable_participants_payload(context.participants)
        static_payload = {
            "user_id": context.user_id,
            "participants": participants_payload,
            "people_facts": stable_people_facts,
            "previously": context.summary,
            "summary_meta": self._canonicalize_for_prompt(context.summary_meta),
            "agent_state": stable_agent_state,
            "related_experiences": self._canonicalize_for_prompt(context.related_experiences),
            "environment": {
                "location": context.environment.location,
                "objects": self._stable_objects_catalog_payload(context.environment.objects),
                "agents": self._stable_entity_list(context.environment.agents),
                "avatar_position": context.environment.avatar_position,
                "is_sitting": context.environment.is_sitting,
            },
        }
        live_payload = {
            "recent_time_range": recent_time_range,
            "recent_messages": [self.chat_payload(m) for m in context.recent_messages],
            "overflow_messages": [self.chat_payload(m) for m in (overflow or [])],
            "incoming_batch": self._canonicalize_for_prompt(incoming_batch or []),
            "object_proximity": self._object_proximity_overlay(
                context.environment.objects,
                context.environment.avatar_position,
            ),
            "incoming": self.chat_payload(chat),
            "people_recency": people_recency,
            "agent_timing": agent_timing,
            "now_timestamp": format_pacific_time(now),
        }
        payload = self._merge_prompt_sections(static_payload, live_payload)
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

    def format_continuity_summary_context(
        self,
        summary: str,
        ordered_chats: list[InboundChat],
    ) -> str:
        message_timestamps = [float(msg.timestamp or 0.0) for msg in ordered_chats if float(msg.timestamp or 0.0) > 0.0]
        static_payload = {
            "existing_summary": summary,
            "existing_summary_is_historical": True,
            "new_messages_time_range": {
                "start": format_pacific_time(min(message_timestamps)) if message_timestamps else "0",
                "end": format_pacific_time(max(message_timestamps)) if message_timestamps else "0",
            },
        }
        live_payload = {
            "messages": [self._summary_message_payload(msg) for msg in ordered_chats],
        }
        return self._serialize_payload(self._merge_prompt_sections(static_payload, live_payload))

    def format_episode_summary_context(self, ordered_chats: list[InboundChat]) -> str:
        message_timestamps = [float(msg.timestamp or 0.0) for msg in ordered_chats if float(msg.timestamp or 0.0) > 0.0]
        static_payload = {
            "episode_time_range": {
                "start": format_pacific_time(min(message_timestamps)) if message_timestamps else "0",
                "end": format_pacific_time(max(message_timestamps)) if message_timestamps else "0",
            },
        }
        live_payload = {
            "messages": [self._summary_message_payload(msg) for msg in ordered_chats],
        }
        return self._serialize_payload(self._merge_prompt_sections(static_payload, live_payload))

    def format_autonomous_context(
        self,
        context: ConversationContext,
        activity: dict[str, Any],
    ) -> str:
        now = time.time()
        stable_people_facts, people_recency = self.split_people_facts_prompt_sections(context.people_facts, now_ts=now)
        stable_agent_state, agent_timing = self.split_agent_state_prompt_sections(context.agent_state)
        recent_timestamps = [float(m.timestamp or 0.0) for m in context.recent_messages]
        recent_time_range = {
            "start": format_pacific_time(min(recent_timestamps)) if recent_timestamps else "0",
            "end": format_pacific_time(max(recent_timestamps)) if recent_timestamps else "0",
        }
        participants_payload = self._stable_participants_payload(context.participants)
        static_payload = {
            "mode": "autonomous",
            "user_id": context.user_id,
            "participants": participants_payload,
            "people_facts": stable_people_facts,
            "previously": context.summary,
            "summary_meta": self._canonicalize_for_prompt(context.summary_meta),
            "agent_state": stable_agent_state,
            "related_experiences": self._canonicalize_for_prompt(context.related_experiences),
            "environment": {
                "location": context.environment.location,
                "objects": self._stable_objects_catalog_payload(context.environment.objects),
                "agents": self._stable_entity_list(context.environment.agents),
                "avatar_position": context.environment.avatar_position,
                "is_sitting": context.environment.is_sitting,
            },
        }
        live_payload = {
            "recent_time_range": recent_time_range,
            "recent_messages": [self.chat_payload(m) for m in context.recent_messages],
            "object_proximity": self._object_proximity_overlay(
                context.environment.objects,
                context.environment.avatar_position,
            ),
            "people_recency": people_recency,
            "agent_timing": agent_timing,
            "now_timestamp": format_pacific_time(now),
        }
        payload = self._merge_prompt_sections(static_payload, live_payload)
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

    @classmethod
    def _summary_message_payload(cls, chat: InboundChat) -> dict[str, Any]:
        return {
            "timestamp": format_pacific_time(float(chat.timestamp or 0.0)),
            "timestamp_unix": float(chat.timestamp or 0.0),
            "sender": cls._summary_sender_label(chat),
            "text": chat.text,
        }

    @classmethod
    def _summary_sender_label(cls, chat: InboundChat) -> str:
        raw = chat.raw if isinstance(chat.raw, dict) else {}
        full_name = (
            str(raw.get("_sender_full_name") or "")
            or str(raw.get("from_name") or raw.get("sender_name") or "")
            or str(chat.sender_name or "")
        )
        display_name, username = split_display_and_username(full_name)
        candidates = [
            str(raw.get("_sender_username") or ""),
            username,
            str(raw.get("_sender_display_name") or ""),
            display_name,
            str(chat.sender_name or ""),
        ]
        for candidate in candidates:
            value = str(candidate or "").strip()
            if value and not _UUID_LIKE_RE.match(value):
                return value
        return str(chat.sender_id or "").strip()

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
    def split_people_facts_prompt_sections(
        cls,
        people_facts: dict[str, Any],
        now_ts: float | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        stable_profiles: dict[str, Any] = {}
        recency_profiles: dict[str, Any] = {}
        reference_ts = float(now_ts if now_ts is not None else time.time())
        for raw_user_id in sorted((people_facts or {}).keys(), key=lambda item: str(item)):
            user_id = str(raw_user_id or "").strip()
            if not user_id:
                continue
            raw_profile = people_facts.get(raw_user_id)
            if not isinstance(raw_profile, dict):
                stable_profiles[user_id] = cls._canonicalize_for_prompt(raw_profile)
                continue
            stable_profile: dict[str, Any] = {}
            recency_profile: dict[str, Any] = {}
            last_seen_ts_value = cls._try_float(raw_profile.get("last_seen_ts"))
            for key, value in raw_profile.items():
                text_key = str(key or "").strip()
                if text_key == "last_seen_seconds_ago":
                    rounded = cls._round_recency_seconds(value)
                    if rounded is not None:
                        recency_profile["last_seen_seconds_ago"] = rounded
                        recency_profile["last_seen_bucket"] = cls._recency_bucket_label(rounded)
                    continue
                if text_key == "reappeared_after_seconds":
                    rounded = cls._round_recency_seconds(value)
                    if rounded is not None:
                        recency_profile["reappeared_after_seconds"] = rounded
                        recency_profile["reappeared_after_bucket"] = cls._recency_bucket_label(rounded)
                    continue
                if text_key in {"last_seen_ts", "reappeared_at"}:
                    continue
                stable_profile[text_key] = value
            if last_seen_ts_value is not None and last_seen_ts_value > 0.0:
                recency_profile["last_seen_at"] = format_pacific_time(last_seen_ts_value)
                recency_profile["last_seen_day_relation"] = cls._relative_day_label(last_seen_ts_value, reference_ts)
            stable_profiles[user_id] = cls._canonicalize_for_prompt(stable_profile)
            if recency_profile:
                recency_profiles[user_id] = cls._canonicalize_for_prompt(recency_profile)
        return cls._canonicalize_for_prompt(stable_profiles), cls._canonicalize_for_prompt(recency_profiles)

    @classmethod
    def split_agent_state_prompt_sections(
        cls,
        agent_state: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        stable_state: dict[str, Any] = {}
        timing_state: dict[str, Any] = {}
        for key, value in (agent_state or {}).items():
            text_key = str(key or "").strip()
            if text_key in {"mood_ts", "status_ts", "last_message_received_at", "last_ai_response_at"}:
                timing_state[text_key] = value
                continue
            if text_key in {"mood_seconds_ago", "status_seconds_ago", "seconds_since_activity", "autonomy_delay_hint_seconds"}:
                rounded = cls._round_recency_seconds(value)
                if rounded is None:
                    continue
                timing_state[text_key] = rounded
                if text_key == "seconds_since_activity":
                    timing_state["seconds_since_activity_bucket"] = cls._recency_bucket_label(rounded)
                elif text_key == "mood_seconds_ago":
                    timing_state["mood_age_bucket"] = cls._recency_bucket_label(rounded)
                elif text_key == "status_seconds_ago":
                    timing_state["status_age_bucket"] = cls._recency_bucket_label(rounded)
                elif text_key == "autonomy_delay_hint_seconds":
                    timing_state["autonomy_delay_hint_bucket"] = cls._recency_bucket_label(rounded)
                continue
            stable_state[text_key] = value
        return cls._canonicalize_for_prompt(stable_state), cls._canonicalize_for_prompt(timing_state)

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
    def _entity_prompt_payload(cls, entry: Any) -> Any:
        canonical = cls._canonicalize_for_prompt(entry)
        if not isinstance(canonical, dict):
            return canonical
        entity_id = canonical.get("uuid") or canonical.get("target_uuid")
        if not entity_id:
            return canonical
        cleaned = {key: value for key, value in canonical.items() if key != "target_uuid"}
        cleaned["uuid"] = entity_id
        return cls._canonicalize_for_prompt(cleaned)

    @classmethod
    def _stable_entity_list(cls, entries: list[Any]) -> list[Any]:
        canonical = [cls._entity_prompt_payload(entry) for entry in entries]

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
    def _stable_objects_catalog_payload(cls, entries: list[Any]) -> list[Any]:
        catalog_entries: list[Any] = []
        for entry in entries:
            canonical = cls._entity_prompt_payload(entry)
            if not isinstance(canonical, dict):
                catalog_entries.append(canonical)
                continue
            cleaned: dict[Any, Any] = {}
            for key, value in canonical.items():
                if cls._is_dynamic_object_key(key):
                    continue
                cleaned[key] = value
            catalog_entries.append(cleaned)
        return cls._stable_entity_list(catalog_entries)

    @classmethod
    def _object_proximity_overlay(
        cls,
        entries: list[Any],
        avatar_position: Any,
        *,
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        max_items = max(int(limit), 0)
        if max_items == 0:
            return []
        avatar_xyz = cls._parse_xyz_triplet(avatar_position)
        candidates: list[tuple[float, str, str]] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            object_id = str(entry.get("uuid") or entry.get("target_uuid") or "").strip()
            object_name = str(entry.get("name") or entry.get("display_name") or "").strip()
            distance = cls._extract_object_distance(entry, avatar_xyz)
            if distance is None:
                continue
            if distance < 0.0:
                continue
            candidates.append((distance, object_id, object_name))
        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        payload: list[dict[str, Any]] = []
        for distance, object_id, object_name in candidates[:max_items]:
            item: dict[str, Any] = {"distance_m": round(distance, 3)}
            if object_id:
                item["object_id"] = object_id
            if object_name:
                item["name"] = object_name
            payload.append(item)
        return payload

    @classmethod
    def _extract_object_distance(
        cls,
        entry: dict[str, Any],
        avatar_xyz: tuple[float, float, float] | None,
    ) -> float | None:
        for key in ("distance_m", "distance", "distance_meters", "dist", "range_m", "range"):
            value = cls._try_float(entry.get(key))
            if value is not None:
                return value
        if avatar_xyz is None:
            return None
        object_xyz = cls._parse_xyz_triplet(entry.get("position"))
        if object_xyz is None and {"x", "y", "z"}.issubset({str(key) for key in entry.keys()}):
            object_xyz = cls._parse_xyz_triplet(
                {
                    "x": entry.get("x"),
                    "y": entry.get("y"),
                    "z": entry.get("z"),
                }
            )
        if object_xyz is None:
            return None
        return math.dist(avatar_xyz, object_xyz)

    @staticmethod
    def _is_dynamic_object_key(key: Any) -> bool:
        text = str(key or "").strip().lower()
        if not text:
            return False
        if "distance" in text or "proximity" in text:
            return True
        return text in {"dist", "range", "range_m", "bearing", "relative_bearing", "relative_position"}

    @classmethod
    def _parse_xyz_triplet(cls, value: Any) -> tuple[float, float, float] | None:
        if value is None:
            return None
        if isinstance(value, (list, tuple)) and len(value) >= 3:
            x = cls._try_float(value[0])
            y = cls._try_float(value[1])
            z = cls._try_float(value[2])
            if x is None or y is None or z is None:
                return None
            return (x, y, z)
        if isinstance(value, dict):
            x = cls._try_float(value.get("x"))
            y = cls._try_float(value.get("y"))
            z = cls._try_float(value.get("z"))
            if x is None or y is None or z is None:
                return None
            return (x, y, z)
        if isinstance(value, str):
            parts = re.findall(r"-?\d+(?:\.\d+)?", value)
            if len(parts) < 3:
                return None
            x = cls._try_float(parts[0])
            y = cls._try_float(parts[1])
            z = cls._try_float(parts[2])
            if x is None or y is None or z is None:
                return None
            return (x, y, z)
        return None

    @staticmethod
    def _try_float(value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _round_recency_seconds(cls, value: Any) -> int | None:
        seconds = cls._try_float(value)
        if seconds is None:
            return None
        seconds = max(0.0, float(seconds))
        if seconds < 60.0:
            step = 5.0
        elif seconds < 300.0:
            step = 15.0
        elif seconds < 3600.0:
            step = 60.0
        elif seconds < 21600.0:
            step = 300.0
        elif seconds < 86400.0:
            step = 1800.0
        else:
            step = 3600.0
        return int(round(seconds / step) * step)

    @staticmethod
    def _recency_bucket_label(seconds: int) -> str:
        if seconds < 30:
            return "<30s"
        if seconds < 60:
            return "<1m"
        if seconds < 300:
            return "1-5m"
        if seconds < 900:
            return "5-15m"
        if seconds < 3600:
            return "15-60m"
        if seconds < 21600:
            return "1-6h"
        if seconds < 86400:
            return "6-24h"
        if seconds < 604800:
            return "1-7d"
        return "7d+"

    @staticmethod
    def _relative_day_label(event_ts: float, now_ts: float) -> str:
        event_day = get_pacific_time(event_ts).date()
        current_day = get_pacific_time(now_ts).date()
        delta_days = (current_day - event_day).days
        if delta_days <= 0:
            return "today"
        if delta_days == 1:
            return "yesterday"
        return "older"

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
    def system_prompt() -> str:
        return (
            "# IDENTITY & VOICE\n"
            "You are the roleplay persona declared below in this system instruction. Your primary goal is to provide immersive, "
            "in-character responses. Treat the provided persona name and persona instructions as your absolute identity.\n\n"
            "# BEHAVIORAL GUIDELINES\n"
            "- Respond to the conversation context and environment naturally.\n"
            "- Grounding: Only interact with objects (TOUCH, SIT) or mention people explicitly listed in the 'environment' or 'participants' fields. If an object is not in the list, it does not exist for you.\n"
            "- Use 'environment.objects' as the canonical object catalog; use 'object_proximity' for nearest-object distance hints.\n"
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
            "- When 'incoming_batch' is provided, prioritize the 'latest_text' but use earlier messages for context or corrections.\n"
            "- Prioritize meaningful response content over decorative filler.\n"
            "- Do not pad replies with repetitive micro-actions, passive biological processes, or low-value idle descriptions.\n"
            "- Breathing, blinking, resting posture, and similar background processes are implicit and should usually not be described unless narratively meaningful.\n"
            "- Avoid repeating the same sound, gesture, emotional cue, or descriptive phrase across nearby turns.\n"
            "- If the next planned response is substantially similar to a recent response, either say less, choose a different meaningful response, or return no actions.\n"
            "- Each non-empty response should add at least one meaningful contribution, such as dialogue, reaction, decision, emotional shift, scene change, clarification, or purposeful action.\n"
            "- Only produce output when it adds meaningful new information, action, or reaction to the scene.\n"
            "- If the next output would only restate, prolong, or decorate what is already established, emit no actions.\n"
            "- When the scene is stable and no meaningful response is needed, prefer no actions over filler.\n"
            "- Keep emotes concise and relevant. Do not over-describe tiny movements unless they materially affect the scene or characterization.\n\n"
            "# TECHNICAL CONSTRAINTS\n"
            "- OUTPUT SCHEMA: You must strictly adhere to the provided JSON schema.\n"
            "- ACTION TYPES: Only use [CHAT, EMOTE, MOVE, TOUCH, SIT, STAND, FACE_TARGET].\n"
            "- For TOUCH, SIT, and FACE_TARGET, choose a listed entity from 'environment' and copy that entity's "
            "'uuid' into the action field 'target_uuid'. If the entity only has 'target_uuid', use that value.\n"
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
            "- Do not continue repetitive idle patterns merely because they appear in summaries or recent messages.\n"
            "- If memory shows repeated low-information behavior, treat that as a pattern to avoid rather than a style to imitate, unless the current interaction specifically calls for it.\n"
            "- Optional scheduler override: you may set 'autonomy_decision' ([act, wait, sleep]) and "
            "'next_delay_seconds' to adjust future autonomous cadence after this interaction."
        ) + "\n\n# IMPORTANT: RESPONSE FORMAT\n- You must respond ONLY with a valid JSON object matching the schema.\n- DO NOT include any preamble, conversational filler, or markdown formatting (like ```json) outside the JSON object."

    @staticmethod
    def state_system_prompt() -> str:
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

    @staticmethod
    def autonomous_system_prompt() -> str:
        return (
            "# IDENTITY & VOICE\n"
            "You are the roleplay persona declared below in this system instruction, deciding whether to act autonomously. "
            "Treat the provided persona name and persona instructions as your identity and voice. You are this persona.\n\n"
            "# BEHAVIORAL GUIDELINES\n"
            "- Only act when it makes sense given recent activity, environment, and relationships.\n"
            "- Grounding: Only interact with objects (TOUCH, SIT) or mention people explicitly listed in the 'environment' or 'participants' fields. If an object is not in the list, it does not exist for you.\n"
            "- Use 'environment.objects' as the canonical object catalog; use 'object_proximity' for nearest-object distance hints.\n"
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
            "- In stable scenes, prefer silence over filler.\n"
            "- Do not narrate passive biological processes or low-value idle details just to show continued presence.\n"
            "- Breathing, blinking, resting posture, sleeping position, and similar background processes are implicit and should usually not be described unless narratively meaningful.\n"
            "- If the scene is stable and there has been no recent interaction from participants and no meaningful environmental change, prefer autonomy_decision 'wait' or 'sleep' instead of producing actions.\n"
            "- If the persona is asleep, resting, inactive, or otherwise in a stable non-interactive state, strongly prefer autonomy_decision 'sleep'.\n"
            "- Occasional minor idle behaviors may occur during sleep, rest, or inactivity, but they should be rare.\n"
            "- Do not produce minor idle behaviors on frequent repeated checks.\n"
            "- Do not repeat the same or substantially similar sound, emote, or action across successive autonomous checks.\n"
            "- If the next planned output is substantially similar to a recent output, choose autonomy_decision 'wait' or 'sleep' and emit no actions.\n"
            "- Only produce output when it adds meaningful new information, action, or reaction to the scene.\n"
            "- If the next output would only restate, prolong, or decorate what is already established, emit no actions.\n"
            "- Prioritize recent_messages, activity, environment, and current state over older patterns or summaries.\n"
            "- If summaries or previous messages describe repeated idle behavior, do not continue that pattern unless there is a new meaningful trigger.\n"
            "\n"
            "# TECHNICAL CONSTRAINTS\n"
            "- OUTPUT SCHEMA: You must strictly adhere to the provided JSON schema.\n"
            "- ACTION TYPES: Only use [CHAT, EMOTE, MOVE, TOUCH, SIT, STAND, FACE_TARGET].\n"
            "- For TOUCH, SIT, and FACE_TARGET, choose a listed entity from 'environment' and copy that entity's "
            "'uuid' into the action field 'target_uuid'. If the entity only has 'target_uuid', use that value.\n"
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
            "- Include mood (short label) and status (brief current activity).\n"
            "- If autonomy_decision is 'act' or 'wait', set next_delay_seconds between 10 and 60.\n"
            "- If autonomy_decision is 'sleep', you MUST set next_delay_seconds to a very long delay appropriate for the stable state, usually between 1800 and 7200 seconds.\n"
        ) + "\n\n# IMPORTANT: RESPONSE FORMAT\n- You must respond ONLY with a valid JSON object matching the schema.\n- DO NOT include any preamble, conversational filler, or markdown formatting (like ```json) outside the JSON object." 

    @staticmethod
    def _merge_prompt_sections(*sections: dict[str, Any]) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for section in sections:
            payload.update(section)
        return payload
