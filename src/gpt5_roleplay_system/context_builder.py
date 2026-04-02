from __future__ import annotations

import asyncio
import time
from difflib import SequenceMatcher
from typing import Any, Dict, Protocol

from .memory import ConversationMemory, ExperienceRecord, ExperienceStore, MemoryItem, TokenSimilaritySearch
from .models import ConversationContext, EnvironmentSnapshot, InboundChat, Participant
from .name_utils import (
    extract_display_name,
    extract_username,
    name_matches,
    normalize_display_name,
    normalize_for_match,
    split_display_and_username,
)
from .payload_contract import canonical_identity_key, normalize_participants
from .neo4j_store import KnowledgeStore
from .observability import Tracer
from .pipeline_state import PipelineRuntimeState
from .time_utils import format_pacific_time


class ExperienceIndexProtocol(Protocol):
    def is_enabled(self) -> bool:
        ...

    async def search(self, query: str, persona_id: str, top_k: int = 3) -> list[ExperienceRecord]:
        ...


class ContextBuilder:
    def __init__(
        self,
        state: PipelineRuntimeState,
        knowledge_store: KnowledgeStore,
        memory: ConversationMemory,
        experience_store: ExperienceStore,
        tracer: Tracer,
        experience_vector_index: ExperienceIndexProtocol | None = None,
        experience_top_k: int = 3,
        experience_score_min: float = 0.78,
        experience_score_delta: float = 0.03,
        near_duplicate_collapse_enabled: bool = True,
        near_duplicate_similarity: float = 0.9,
        routine_summary_enabled: bool = False,
        routine_summary_limit: int = 2,
        routine_summary_min_count: int = 2,
        max_environment_participants: int = 10,
        persona_profiles: dict[str, str] | None = None,
        owner: Any | None = None,
    ) -> None:
        self._state = state
        self._knowledge_store = knowledge_store
        self._memory = memory
        self._experience_store = experience_store
        self._tracer = tracer
        self._experience_vector_index = experience_vector_index
        self._similarity = TokenSimilaritySearch()
        self._experience_top_k = int(experience_top_k)
        self._experience_score_min = float(experience_score_min)
        self._experience_score_delta = float(experience_score_delta)
        self._near_duplicate_collapse_enabled = bool(near_duplicate_collapse_enabled)
        self._near_duplicate_similarity = min(1.0, max(0.0, float(near_duplicate_similarity)))
        self._routine_summary_enabled = bool(routine_summary_enabled)
        self._routine_summary_limit = max(0, int(routine_summary_limit))
        self._routine_summary_min_count = max(2, int(routine_summary_min_count))
        self._routine_candidate_limit = max(self._experience_top_k, self._experience_top_k * 4)
        self._max_environment_participants = max_environment_participants
        self._persona_profiles = {str(k).casefold(): str(v) for k, v in (persona_profiles or {}).items() if k and v}
        self._last_seen_cache: dict[str, float] = {}
        self._owner = owner

    def upsert_persona_profile(self, persona: str, instructions: str) -> None:
        key = str(persona or "").strip().casefold()
        value = str(instructions or "").strip()
        if not key or not value:
            return
        self._persona_profiles[key] = value

    def update_environment(self, data: dict[str, Any]) -> None:
        agents = self._normalize_entities(data.get("agents", []))
        objects = self._normalize_entities(data.get("objects", []))
        raw_is_sitting = data.get("is_sitting", False)
        if isinstance(raw_is_sitting, str):
            posture_value = raw_is_sitting.strip().lower() in {"1", "true", "yes", "y", "t"}
        else:
            posture_value = bool(raw_is_sitting)
        update_ts = float(data.get("timestamp", time.time()) or time.time())
        for agent in agents:
            if not isinstance(agent, dict):
                continue
            user_id = str(agent.get("uuid") or agent.get("target_uuid") or "")
            name = str(agent.get("name") or "")
            self._record_display_name(user_id, name)
        self._update_last_seen_cache_from_agents(agents, update_ts)
        self._schedule_update_last_seen_from_agents(agents, update_ts)
        self._state.last_environment_update_ts = max(float(self._state.last_environment_update_ts or 0.0), update_ts)
        if update_ts >= float(self._state.last_posture_update_ts or 0.0):
            self._state.last_posture_update_ts = update_ts
            self._state.posture_known = True
            self._state.posture_is_sitting = posture_value
        is_sitting_last_known = self._state.posture_is_sitting if self._state.posture_known else False
        self._state.environment = EnvironmentSnapshot(
            agents=agents,
            objects=objects,
            location=data.get("location", ""),
            avatar_position=data.get("avatar_position", ""),
            is_sitting=is_sitting_last_known,
        )
        self._tracer.log_event("environment_update", {"agents": len(self._state.environment.agents)})

    def build_chat(self, data: dict[str, Any]) -> InboundChat:
        if self._owner is not None and hasattr(self._owner, "_build_chat"):
            return self._owner._build_chat(data)
        sender_id_raw = data.get("from_id", data.get("sender_id", ""))
        sender_id = str(sender_id_raw or "")
        full_name_raw = data.get("from_name", data.get("sender_name", ""))
        full_name = str(full_name_raw or "")
        display_name, username = split_display_and_username(full_name)
        sender_username = (username or display_name or full_name or sender_id).strip()
        if sender_id and full_name:
            self._record_display_name(sender_id, full_name)
        raw_payload = dict(data)
        raw_payload["_sender_full_name"] = full_name
        raw_payload["_sender_display_name"] = display_name
        raw_payload["_sender_username"] = sender_username
        return InboundChat(
            text=data.get("text", ""),
            sender_id=sender_id,
            sender_name=sender_username,
            timestamp=float(data.get("timestamp", time.time())),
            raw=raw_payload,
        )

    def merge_participants(self, chats: list[InboundChat]) -> list[Participant]:
        if self._owner is not None and hasattr(self._owner, "_merge_participants"):
            return self._owner._merge_participants(chats)
        merged: dict[str, Participant] = {}
        for chat in chats:
            for participant in self._resolve_participants(chat):
                key = participant.user_id or f"name:{self._name_key(participant.name)}"
                merged[key] = participant
        return self._normalize_participant_records(list(merged.values()))

    def participants_from_messages(
        self,
        messages: list[InboundChat],
        base_participants: list[Participant],
    ) -> list[Participant]:
        if self._owner is not None and hasattr(self._owner, "_participants_from_messages"):
            return self._owner._participants_from_messages(messages, base_participants)
        merged: dict[str, Participant] = {}
        for participant in base_participants:
            if self._is_self_participant(participant.user_id, participant.name):
                continue
            canonical_name = self._canonical_name(participant.user_id, participant.name)
            key = self._participant_key(participant.user_id, canonical_name)
            merged[key] = Participant(user_id=participant.user_id, name=canonical_name)
        for message in messages:
            if self._is_self_message(message):
                continue
            if not message.sender_id and not message.sender_name:
                continue
            canonical_name = self._canonical_name(message.sender_id, message.sender_name)
            participant = Participant(user_id=message.sender_id, name=canonical_name)
            if self._is_self_participant(participant.user_id, participant.name):
                continue
            key = self._participant_key(participant.user_id, participant.name)
            merged[key] = participant
        return self._normalize_participant_records(list(merged.values()))

    def participants_for_autonomy(self) -> list[Participant]:
        if self._owner is not None and hasattr(self._owner, "_participants_for_autonomy"):
            return self._owner._participants_for_autonomy()
        merged: dict[str, Participant] = {}

        def add_participant(user_id: str, name: str) -> None:
            if self._is_self_participant(user_id, name):
                return
            if not user_id and not name:
                return
            self._record_display_name(user_id, name)
            canonical_name = self._canonical_name(user_id, name)
            key = user_id or f"name:{self._name_key(canonical_name)}"
            merged[key] = Participant(user_id=user_id, name=canonical_name)

        for hint in self._state.participant_hints:
            add_participant(hint.user_id, hint.name)
        for agent in self._state.environment.agents[: self._max_environment_participants]:
            user_id = str(agent.get("uuid") or agent.get("target_uuid") or "")
            name = str(agent.get("name") or "")
            add_participant(user_id, name)
        return self._normalize_participant_records(list(merged.values()))

    async def build_context(
        self,
        chat: InboundChat | None,
        participants: list[Participant],
        query_text: str | None = None,
        *,
        agent_state: dict[str, Any],
    ) -> ConversationContext:
        if self._owner is not None and hasattr(self._owner, "_build_context"):
            return await self._owner._build_context(chat, participants, query_text=query_text)
        query = (query_text or (chat.text if chat else "")).strip()
        recent = self._memory.recent()
        people_facts = self._people_facts_for_participants(participants)
        persona_experiences = self._experiences_for_persona(self._experience_store.all())
        related = self._similarity.search(query, persona_experiences, top_k=self._experience_top_k)
        related = self._collapse_near_duplicate_experiences(related)
        now_ts = time.time()
        summary_meta_raw = self._memory.summary_meta()
        summary_meta: Dict[str, Any] = dict(summary_meta_raw)
        last_updated_ts = float(summary_meta_raw.get("last_updated_ts", 0.0) or 0.0)
        range_end_ts = float(summary_meta_raw.get("range_end_ts", 0.0) or 0.0)
        if last_updated_ts > 0.0:
            summary_meta["age_seconds"] = max(0.0, now_ts - last_updated_ts)
        if range_end_ts > 0.0:
            summary_meta["range_age_seconds"] = max(0.0, now_ts - range_end_ts)
        return ConversationContext(
            persona=self._state.persona,
            user_id=self._state.user_id,
            environment=self._state.environment,
            participants=participants,
            people_facts=people_facts,
            recent_messages=self._memory_items_to_chats(recent),
            summary=self._memory.summary(),
            related_experiences=[{"text": item.text, "metadata": item.metadata} for item in related],
            summary_meta=summary_meta,
            agent_state=agent_state,
            persona_instructions=self._persona_instructions(),
        )

    def build_facts_context(
        self,
        participants: list[Participant],
        evidence_messages: list[InboundChat],
        *,
        agent_state: dict[str, Any],
    ) -> ConversationContext:
        if self._owner is not None and hasattr(self._owner, "_facts_context"):
            return self._owner._facts_context(participants, evidence_messages)
        evidence = list(evidence_messages)[-24:]
        return ConversationContext(
            persona=self._state.persona,
            user_id=self._state.user_id,
            environment=self._state.environment,
            participants=participants,
            people_facts=self._people_facts_for_participants(participants),
            recent_messages=evidence,
            summary="",
            related_experiences=[],
            summary_meta={},
            agent_state=agent_state,
        )

    def collapse_batch(self, chats: list[InboundChat]) -> list[dict[str, Any]]:
        if self._owner is not None and hasattr(self._owner, "_collapse_batch"):
            return self._owner._collapse_batch(chats)
        groups: dict[str, dict[str, Any]] = {}
        for chat in chats:
            key = chat.sender_id or chat.sender_name
            if not key:
                key = f"anon:{len(groups)}"
            full_name = self._full_name_for(chat.sender_id, chat.sender_name)
            display_name, username = split_display_and_username(full_name)
            sender_username = username or chat.sender_name or chat.sender_id
            sender_display = display_name or chat.sender_name or sender_username
            entry = groups.get(key)
            if entry is None:
                entry = {
                    "sender_id": chat.sender_id,
                    "sender_name": sender_username,
                    "sender_username": sender_username,
                    "sender_display_name": sender_display,
                    "sender_full_name": full_name,
                    "texts": [],
                    "timestamps": [],
                    "latest_text": "",
                    "first_timestamp": chat.timestamp,
                    "last_timestamp": chat.timestamp,
                }
                groups[key] = entry
            entry["texts"].append(chat.text)
            entry["timestamps"].append(chat.timestamp)
            entry["latest_text"] = chat.text
            entry["last_timestamp"] = chat.timestamp
        return list(groups.values())

    def participant_payload(self, participant: Participant) -> dict[str, Any]:
        full_name = self._full_name_for(participant.user_id, participant.name)
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

    def chat_payload(self, chat: Any) -> dict[str, Any]:
        raw_value = getattr(chat, "raw", {})
        raw = raw_value if isinstance(raw_value, dict) else {}
        sender_id = str(getattr(chat, "sender_id", "") or "")
        sender_name = str(getattr(chat, "sender_name", "") or "")
        text = str(getattr(chat, "text", "") or "")
        timestamp = float(getattr(chat, "timestamp", 0.0) or 0.0)
        if sender_id == "ai":
            persona = str(self._state.persona or "")
            sender_id_out = str(self._state.user_id or "")
            return {
                "sender": persona,
                "sender_id": sender_id_out,
                "sender_username": persona,
                "sender_display_name": persona,
                "sender_full_name": persona,
                "text": text,
                "timestamp": format_pacific_time(timestamp),
            }
        full_name = (
            str(raw.get("_sender_full_name") or "")
            or str(raw.get("from_name") or raw.get("sender_name") or "")
            or self._full_name_for(sender_id, sender_name)
        )
        display_name, username = split_display_and_username(full_name)
        sender_username = str(raw.get("_sender_username") or "") or username or sender_name or sender_id
        sender_display = str(raw.get("_sender_display_name") or "") or display_name or sender_name or sender_username
        sender_value = sender_username or sender_display or sender_id
        return {
            "sender": sender_value,
            "sender_id": sender_id,
            "sender_username": sender_username,
            "sender_display_name": sender_display,
            "sender_full_name": full_name,
            "text": text,
            "timestamp": format_pacific_time(timestamp),
        }

    def environment_payload(self, object_limit: int = 25) -> dict[str, Any]:
        return {
            "location": self._state.environment.location,
            "avatar_position": self._state.environment.avatar_position,
            "is_sitting": self._state.environment.is_sitting,
            "agents": self._state.environment.agents[: self._max_environment_participants],
            "objects": self._state.environment.objects[:object_limit],
        }

    def snapshot_state(self) -> dict[str, Any]:
        return {
            "environment": {
                "agents": list(self._state.environment.agents),
                "objects": list(self._state.environment.objects),
                "location": self._state.environment.location,
                "avatar_position": self._state.environment.avatar_position,
                "is_sitting": self._state.environment.is_sitting,
            },
            "participant_hints": [
                {"user_id": participant.user_id, "name": participant.name}
                for participant in self._state.participant_hints
            ],
            "display_names_by_id": dict(self._state.display_names_by_id),
            "last_seen_cache": dict(self._last_seen_cache),
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        env = state.get("environment", {})
        if isinstance(env, dict):
            self._state.environment = EnvironmentSnapshot(
                agents=list(env.get("agents", []) or []),
                objects=list(env.get("objects", []) or []),
                location=str(env.get("location", "") or ""),
                avatar_position=str(env.get("avatar_position", "") or ""),
                is_sitting=bool(env.get("is_sitting", False)),
            )
        hints_raw = state.get("participant_hints", [])
        hints: list[Participant] = []
        if isinstance(hints_raw, list):
            for item in hints_raw:
                if not isinstance(item, dict):
                    continue
                hints.append(
                    Participant(
                        user_id=str(item.get("user_id", "") or ""),
                        name=str(item.get("name", "") or ""),
                    )
                )
        self._state.participant_hints = hints
        names = state.get("display_names_by_id", {})
        if isinstance(names, dict):
            self._state.display_names_by_id = {str(k): str(v) for k, v in names.items() if str(k)}
        cache = state.get("last_seen_cache", {})
        if isinstance(cache, dict):
            self._last_seen_cache = {str(k): float(v or 0.0) for k, v in cache.items() if str(k)}

    def _resolve_participants(self, chat: InboundChat) -> list[Participant]:
        participants: dict[str, Participant] = {}

        def add_participant(user_id: str, name: str) -> None:
            if not user_id and not name:
                return
            if self._is_self_participant(user_id, name):
                return
            self._record_display_name(user_id, name)
            canonical_name = self._canonical_name(user_id, name)
            key = user_id or f"name:{self._name_key(canonical_name)}"
            participants[key] = Participant(user_id=user_id, name=canonical_name)

        for hint in self._state.participant_hints:
            add_participant(hint.user_id, hint.name)

        if chat.sender_id or chat.sender_name:
            add_participant(chat.sender_id, chat.sender_name)
        explicit = chat.raw.get("participants", [])
        if isinstance(explicit, list):
            for entry in explicit:
                if not isinstance(entry, dict):
                    continue
                user_id = entry.get("user_id") or entry.get("uuid") or entry.get("target_uuid", "")
                name = entry.get("name", "")
                if user_id or name:
                    if user_id == chat.sender_id:
                        continue
                    self._record_display_name(str(user_id or ""), str(name or ""))
                    add_participant(user_id, name)
        text_lower = chat.text.lower()
        for agent in self._state.environment.agents[: self._max_environment_participants]:
            user_id = agent.get("uuid") or agent.get("target_uuid", "")
            name = agent.get("name", "")
            if user_id == chat.sender_id:
                continue
            self._record_display_name(str(user_id or ""), str(name or ""))
            display_name = str(name or "")
            if display_name and display_name.lower() in text_lower:
                add_participant(user_id, display_name)
        return list(participants.values())

    def _memory_items_to_chats(self, items: list[MemoryItem]) -> list[InboundChat]:
        chats: list[InboundChat] = []
        for item in items:
            sender_id = str(item.sender_id or "")
            is_ai_marker = sender_id == "ai"
            sender_id_out = (self._state.user_id or "") if is_ai_marker else sender_id
            sender_name_out = self._state.persona if is_ai_marker else (sender_id or str(item.sender_name or ""))
            raw_payload: Dict[str, Any] = {}
            if is_ai_marker:
                raw_payload["_sender_full_name"] = self._state.persona
                raw_payload["_sender_display_name"] = self._state.persona
                raw_payload["_sender_username"] = self._state.persona
            chats.append(
                InboundChat(
                    text=str(item.text or ""),
                    sender_id=sender_id_out,
                    sender_name=sender_name_out,
                    timestamp=float(item.timestamp or 0.0),
                    raw=raw_payload,
                )
            )
        return chats

    def _people_facts_for_participants(self, participants: list[Participant]) -> dict[str, dict[str, Any]]:
        user_ids = [p.user_id for p in participants if p.user_id]
        if not user_ids:
            return {}
        people = self._knowledge_store.fetch_people(user_ids)
        now_ts = time.time()
        people_facts: dict[str, dict[str, Any]] = {}
        for user_id, profile in people.items():
            facts = self._dedupe_preserve_order(profile.facts)
            last_seen_ts = float(getattr(profile, "last_seen_ts", 0.0) or 0.0)
            last_seen_ts = max(last_seen_ts, float(self._last_seen_cache.get(user_id, 0.0) or 0.0))
            last_seen_seconds_ago = (now_ts - last_seen_ts) if last_seen_ts > 0 else None
            full_name = self._full_name_for(user_id, getattr(profile, "name", ""))
            display_name, username = split_display_and_username(full_name)
            username_value = username or extract_username(getattr(profile, "name", "")) or user_id
            display_value = (
                display_name
                or extract_display_name(getattr(profile, "name", ""))
                or getattr(profile, "name", "")
                or username_value
            )
            entry: Dict[str, Any] = {
                "name": username_value,
                "username": username_value,
                "display_name": display_value,
                "full_name": full_name or display_value,
                "facts": facts,
                "relationships": profile.relationships,
                "last_seen_ts": last_seen_ts,
                "last_seen_seconds_ago": last_seen_seconds_ago,
            }
            people_facts[user_id] = entry
        return people_facts

    def _record_display_name(self, user_id: str, full_name: str) -> None:
        if not user_id or not full_name:
            return
        if self._is_self_participant(user_id, full_name):
            return
        prior = self._state.display_names_by_id.get(user_id, "")
        if not prior or len(full_name) >= len(prior):
            self._state.display_names_by_id[user_id] = full_name

    def _full_name_for(self, user_id: str, fallback_name: str = "") -> str:
        if user_id and user_id in self._state.display_names_by_id:
            return self._state.display_names_by_id[user_id]
        return fallback_name or user_id

    def _update_last_seen_cache_from_agents(self, agents: list[dict[str, Any]], timestamp: float) -> None:
        ts = float(timestamp or 0.0)
        if ts <= 0:
            return
        for agent in agents:
            if not isinstance(agent, dict):
                continue
            user_id = str(agent.get("uuid") or agent.get("target_uuid") or "")
            name = str(agent.get("name") or "")
            if not user_id or self._is_self_participant(user_id, name):
                continue
            prior = float(self._last_seen_cache.get(user_id, 0.0) or 0.0)
            if ts > prior:
                self._last_seen_cache[user_id] = ts

    def _update_last_seen_from_agents(self, agents: list[dict[str, Any]], timestamp: float) -> None:
        ts = float(timestamp or 0.0)
        if ts <= 0:
            return
        for agent in agents:
            if not isinstance(agent, dict):
                continue
            user_id = str(agent.get("uuid") or agent.get("target_uuid") or "")
            name = str(agent.get("name") or "")
            if not user_id or self._is_self_participant(user_id, name):
                continue
            try:
                self._knowledge_store.update_last_seen(user_id, name, ts)
            except Exception:
                continue

    def _schedule_update_last_seen_from_agents(self, agents: list[dict[str, Any]], timestamp: float) -> None:
        if not agents:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._update_last_seen_from_agents(agents, timestamp)
            return
        loop.create_task(asyncio.to_thread(self._update_last_seen_from_agents, agents, timestamp))

    def _participant_key(self, user_id: str, name: str) -> str:
        key = canonical_identity_key(user_id, name)
        if key:
            return key
        return f"name:{self._name_key(name)}"

    def _normalize_participant_records(self, participants: list[Participant]) -> list[Participant]:
        raw = [self.participant_payload(participant) for participant in participants]
        normalized_payload, _ = normalize_participants(raw)
        normalized: list[Participant] = []
        for entry in normalized_payload:
            user_id = str(entry.get("user_id", "") or "")
            name = str(entry.get("name", "") or "")
            full_name = str(entry.get("full_name", "") or "")
            if user_id and full_name:
                self._record_display_name(user_id, full_name)
            normalized.append(Participant(user_id=user_id, name=name))
        return normalized

    def _is_self_participant(self, user_id: str, name: str) -> bool:
        if user_id and self._state.user_id and user_id == self._state.user_id:
            return True
        if not name:
            return False
        return name_matches(name, self._state.persona)

    def _is_self_message(self, chat: InboundChat) -> bool:
        sender_id = str(chat.sender_id or "")
        if sender_id and self._state.user_id and sender_id == self._state.user_id:
            return True
        sender_full_name = self._chat_full_name(chat)
        if not sender_full_name:
            return False
        return name_matches(sender_full_name, self._state.persona)

    def _chat_full_name(self, chat: InboundChat) -> str:
        raw = chat.raw if isinstance(chat.raw, dict) else {}
        sender_id = str(chat.sender_id or "")
        return (
            str(raw.get("_sender_full_name") or "")
            or str(raw.get("from_name") or raw.get("sender_name") or "")
            or self._full_name_for(sender_id, chat.sender_name)
        )

    @staticmethod
    def _normalize_name(name: str) -> str:
        return normalize_display_name(name)

    @staticmethod
    def _normalize_name_for_match(name: str) -> str:
        return normalize_for_match(name)

    def _name_key(self, name: str) -> str:
        match_key = self._normalize_name_for_match(name)
        if match_key:
            return match_key
        display_key = self._normalize_name(name)
        if display_key:
            return display_key
        return (name or "").casefold()

    @staticmethod
    def _canonical_name(user_id: str, name: str) -> str:
        if name:
            username = extract_username(name)
            if username:
                return username
            display = extract_display_name(name)
            if display:
                return display
        return user_id or ""

    def _persona_instructions(self) -> str:
        if not self._persona_profiles:
            return ""
        key = str(self._state.persona or "").casefold()
        return str(self._persona_profiles.get(key, "") or "")

    @staticmethod
    def _dedupe_preserve_order(items: list[str]) -> list[str]:
        seen: set[str] = set()
        deduped: list[str] = []
        for item in items:
            key = ContextBuilder._normalize_fact_key(item)
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    @staticmethod
    def _normalize_fact_key(text: str) -> str:
        if not text:
            return ""
        normalized = "".join(ch if ch.isalnum() else " " for ch in str(text).casefold())
        return " ".join(normalized.split())

    def _experiences_for_persona(self, experiences: list[ExperienceRecord]) -> list[ExperienceRecord]:
        if not self._state.persona:
            return experiences
        filtered: list[ExperienceRecord] = []
        for item in experiences:
            metadata = item.metadata if isinstance(item.metadata, dict) else {}
            persona_id = str(metadata.get("persona_id", "") or "")
            if persona_id and persona_id != self._state.persona:
                continue
            filtered.append(item)
        return filtered

    def _collapse_near_duplicate_experiences(self, experiences: list[ExperienceRecord]) -> list[ExperienceRecord]:
        if not self._near_duplicate_collapse_enabled:
            return experiences
        if len(experiences) < 2:
            return experiences
        threshold = self._near_duplicate_similarity
        if threshold <= 0.0:
            return experiences

        clusters: list[dict[str, Any]] = []
        for item in experiences:
            normalized_text = self._normalize_experience_text(item.text)
            best_idx = -1
            best_score = 0.0
            for idx, cluster in enumerate(clusters):
                score = self._near_duplicate_similarity_score(
                    normalized_text,
                    str(cluster.get("anchor_text", "")),
                )
                if score > best_score:
                    best_score = score
                    best_idx = idx
            if best_idx >= 0 and best_score >= threshold:
                clusters[best_idx]["items"].append(item)
                continue
            clusters.append({"anchor_text": normalized_text, "items": [item]})

        if all(len(cluster.get("items", [])) < 2 for cluster in clusters):
            return experiences

        collapsed: list[ExperienceRecord] = []
        for cluster in clusters:
            items = cluster.get("items", [])
            if not items:
                continue
            representative = items[0]
            if len(items) == 1:
                collapsed.append(representative)
                continue
            metadata = representative.metadata if isinstance(representative.metadata, dict) else {}
            merged_metadata: dict[str, Any] = dict(metadata)
            merged_metadata["near_duplicate_count"] = len(items)
            collapsed.append(ExperienceRecord(text=representative.text, metadata=merged_metadata))
        return collapsed

    @staticmethod
    def _normalize_experience_text(text: str) -> str:
        clean = " ".join(str(text or "").split())
        return clean.casefold()

    @staticmethod
    def _near_duplicate_similarity_score(left: str, right: str) -> float:
        if not left or not right:
            return 0.0
        if left == right:
            return 1.0
        sequence = SequenceMatcher(a=left, b=right).ratio()
        left_tokens = set(left.split())
        right_tokens = set(right.split())
        token_score = ContextBuilder._token_jaccard(left_tokens, right_tokens)
        return max(sequence, token_score)

    @staticmethod
    def _token_jaccard(left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        union = left | right
        if not union:
            return 0.0
        return len(left & right) / len(union)

    @staticmethod
    def _normalize_entities(entries: list[Any]) -> list[Any]:
        normalized: list[Any] = []
        for entry in entries:
            if not isinstance(entry, dict):
                normalized.append(entry)
                continue
            uuid_value = entry.get("uuid") or entry.get("target_uuid")
            if not uuid_value:
                normalized.append(entry)
                continue
            merged = dict(entry)
            merged.setdefault("uuid", uuid_value)
            merged.setdefault("target_uuid", uuid_value)
            normalized.append(merged)
        return normalized
