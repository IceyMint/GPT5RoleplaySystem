from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Protocol, Set

from .config import EpisodeConfig, FactsConfig
from .llm import ExtractedFact, LLMClient, LLMResponseBundle, ParticipantHint
from .memory import (
    ConversationMemory,
    ExperienceRecord,
    ExperienceStore,
    MemoryItem,
    RollingBuffer,
    SimpleMemoryCompressor,
    TokenSimilaritySearch,
)
from .models import Action, CommandType, ConversationContext, EnvironmentSnapshot, InboundChat, Participant
from .time_utils import format_pacific_time
from .name_utils import (
    extract_display_name,
    extract_username,
    name_matches,
    normalize_display_name,
    normalize_for_match,
    split_display_and_username,
)
from .neo4j_store import KnowledgeStore
from .observability import Tracer

logger = logging.getLogger("gpt5_roleplay_pipeline")


class ExperienceIndexProtocol(Protocol):
    def is_enabled(self) -> bool:
        ...

    async def add_record_async(self, record: ExperienceRecord, persona_id: str) -> None:
        ...

    async def search(self, query: str, persona_id: str, top_k: int = 3) -> List[ExperienceRecord]:
        ...


class MessagePipeline:
    def __init__(
        self,
        persona: str,
        user_id: str,
        llm: LLMClient,
        knowledge_store: KnowledgeStore,
        memory: ConversationMemory,
        rolling_buffer: RollingBuffer,
        experience_store: ExperienceStore,
        tracer: Tracer,
        experience_vector_index: ExperienceIndexProtocol | None = None,
        summary_strategy: str = "simple",
        experience_top_k: int = 3,
        experience_score_min: float = 0.78,
        experience_score_delta: float = 0.03,
        max_environment_participants: int = 10,
        facts_config: FactsConfig | None = None,
        episode_config: EpisodeConfig | None = None,
        persona_profiles: Dict[str, str] | None = None,
    ) -> None:
        self._persona = persona
        self._user_id = user_id
        self._llm = llm
        self._knowledge_store = knowledge_store
        self._memory = memory
        self._rolling_buffer = rolling_buffer
        self._experience_store = experience_store
        self._experience_vector_index = experience_vector_index
        self._similarity = TokenSimilaritySearch()
        self._episode_compressor = SimpleMemoryCompressor()
        self._environment = EnvironmentSnapshot()
        self._tracer = tracer
        self._summary_strategy = summary_strategy
        self._experience_top_k = experience_top_k
        self._experience_score_min = experience_score_min
        self._experience_score_delta = experience_score_delta
        self._max_environment_participants = max_environment_participants
        self._participant_hints: List[Participant] = []
        self._persona_profiles = {str(k).casefold(): str(v) for k, v in (persona_profiles or {}).items() if k and v}
        # Tracks viewer-provided full names by UUID so we can pass display+username to the LLM.
        self._display_names_by_id: Dict[str, str] = {}
        facts = facts_config or FactsConfig()
        self._facts_enabled = bool(facts.enabled)
        mode = str(facts.mode or "periodic").strip().lower()
        self._facts_mode = mode if mode in {"periodic", "per_message"} else "periodic"
        self._facts_interval_seconds = max(1.0, float(facts.interval_seconds))
        self._facts_evidence_max_messages = max(4, int(facts.evidence_max_messages))
        self._facts_last_sweep_ts = 0.0
        self._facts_cursor_ts = 0.0
        self._facts_pending_messages: List[InboundChat] = []
        self._facts_pending_participants: Dict[str, Participant] = {}
        self._facts_task: asyncio.Task | None = None
        self._last_seen_cache: Dict[str, float] = {}
        episode = episode_config or EpisodeConfig()
        self._episode_enabled = bool(episode.enabled)
        self._episode_min_messages = max(1, int(episode.min_messages))
        self._episode_max_messages = max(self._episode_min_messages, int(episode.max_messages))
        self._episode_inactivity_seconds = max(0.0, float(episode.inactivity_seconds))
        self._episode_forced_interval_seconds = max(0.0, float(episode.forced_interval_seconds))
        self._episode_overlap_messages = max(0, int(episode.overlap_messages))
        self._last_episode_ts = 0.0
        self._last_episode_size = 0
        self._episode_task: asyncio.Task | None = None
        now = time.time()
        self._last_inbound_ts = now
        self._last_response_ts = now
        self._current_mood = "neutral"
        self._current_mood_ts = now
        self._current_status = "idle"
        self._current_status_ts = now
        self._mood_source = "init"
        self._status_source = "init"
        logger.info(
            "Pipeline LLM client: %s (persona=%s, user_id=%s)",
            type(self._llm).__name__,
            self._persona,
            self._user_id,
        )

    def snapshot_state(self) -> Dict[str, Any]:
        return {
            "rolling_buffer": self._rolling_buffer.snapshot(),
            "memory": self._memory.snapshot(),
            "episode": {
                "last_episode_ts": self._last_episode_ts,
                "last_episode_size": self._last_episode_size,
            },
            "status": {
                "mood": self._current_mood,
                "mood_ts": self._current_mood_ts,
                "mood_source": self._mood_source,
                "status": self._current_status,
                "status_ts": self._current_status_ts,
                "status_source": self._status_source,
            },
        }

    def restore_state(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        rolling_items = state.get("rolling_buffer", [])
        if isinstance(rolling_items, list):
            self._rolling_buffer.restore(rolling_items)
        memory_state = state.get("memory", {})
        if isinstance(memory_state, dict):
            self._memory.restore(memory_state)
        episode_state = state.get("episode", {})
        if isinstance(episode_state, dict):
            self._last_episode_ts = float(episode_state.get("last_episode_ts", 0.0) or 0.0)
            self._last_episode_size = int(episode_state.get("last_episode_size", 0) or 0)
        status_state = state.get("status", {})
        if isinstance(status_state, dict):
            mood = status_state.get("mood", self._current_mood)
            status = status_state.get("status", self._current_status)
            self._current_mood = str(mood or self._current_mood)
            self._current_status = str(status or self._current_status)
            self._current_mood_ts = float(status_state.get("mood_ts", self._current_mood_ts) or self._current_mood_ts)
            self._current_status_ts = float(
                status_state.get("status_ts", self._current_status_ts) or self._current_status_ts
            )
            self._mood_source = str(status_state.get("mood_source", self._mood_source) or self._mood_source)
            self._status_source = str(
                status_state.get("status_source", self._status_source) or self._status_source
            )
        # Prevent immediate inactivity-triggered episodes after restoring.
        latest_ts = self._latest_timestamp()
        if latest_ts > 0:
            self._last_inbound_ts = latest_ts
            self._last_response_ts = latest_ts

    def update_environment(self, data: Dict[str, Any]) -> None:
        agents = self._normalize_entities(data.get("agents", []))
        objects = self._normalize_entities(data.get("objects", []))
        update_ts = float(data.get("timestamp", time.time()) or time.time())
        for agent in agents:
            if not isinstance(agent, dict):
                continue
            user_id = str(agent.get("uuid") or agent.get("target_uuid") or "")
            name = str(agent.get("name") or "")
            self._record_display_name(user_id, name)
        self._update_last_seen_cache_from_agents(agents, update_ts)
        self._schedule_update_last_seen_from_agents(agents, update_ts)
        self._environment = EnvironmentSnapshot(
            agents=agents,
            objects=objects,
            location=data.get("location", ""),
            avatar_position=data.get("avatar_position", ""),
        )
        self._tracer.log_event("environment_update", {"agents": len(self._environment.agents)})

    def set_persona(self, persona: str) -> None:
        self._persona = persona

    def set_user_id(self, user_id: str) -> None:
        self._user_id = user_id

    async def process_chat(self, data: Dict[str, Any]) -> List[Any]:
        return await self.process_chat_batch([data])

    async def process_chat_batch(self, batch: List[Dict[str, Any]]) -> List[Any]:
        chats = [self._build_chat(item) for item in batch]
        non_self_chats: List[InboundChat] = []
        for chat in chats:
            if self._is_self_message(chat):
                self._handle_self_message(chat)
            else:
                non_self_chats.append(chat)

        if not non_self_chats:
            return []

        self._last_inbound_ts = time.time()

        for chat in non_self_chats:
            self._rolling_buffer.add_user_message(chat)
            self._memory.add_message(chat)

        overflow = self._memory.drain_overflow()
        overflow_chats = self._memory_items_to_chats(overflow)
        overflow_timestamps = [float(item.timestamp or 0.0) for item in overflow]

        participants = self._merge_participants(non_self_chats)
        self._maybe_schedule_fact_sweep(non_self_chats, overflow_chats, participants)
        primary_chat = non_self_chats[-1]
        context = await self._build_context(primary_chat, participants)

        addressed = False
        for chat in non_self_chats:
            self._tracer.log_event(
                "llm_address_check",
                {
                    "sender": chat.sender_id or chat.sender_name,
                    "text": chat.text,
                    "persona": self._persona,
                    "participants": [(p.user_id or p.name) for p in participants[:10]],
                    "location": self._environment.location,
                },
            )
            is_addressed = await self._llm.is_addressed_to_me(
                chat,
                self._persona,
                self._environment,
                participants,
                context,
            )
            self._tracer.log_event(
                "llm_address_result",
                {"sender": chat.sender_id or chat.sender_name, "addressed": bool(is_addressed)},
            )
            if is_addressed:
                addressed = True
                break
        if not addressed:
            return []

        incoming_batch = self._collapse_batch(non_self_chats)
        self._tracer.log_event(
            "llm_prompt_bundle",
            self._bundle_prompt_payload(primary_chat, context, overflow_chats, incoming_batch),
        )
        bundle = await self._llm.generate_bundle(primary_chat, context, overflow_chats, incoming_batch)
        self._tracer.log_event("llm_response_bundle", self._bundle_response_payload(bundle))
        if bundle.text:
            self._rolling_buffer.add_ai_message(bundle.text, self._persona)
            self._memory.add_ai_message(bundle.text, self._persona)
        if bundle.summary:
            self._memory.apply_summary(bundle.summary, timestamps=overflow_timestamps)
        elif overflow:
            if self._summary_strategy == "llm":
                summary_text = await self._llm.summarize(self._memory.summary(), overflow_chats)
                if summary_text:
                    self._memory.apply_summary(summary_text, timestamps=overflow_timestamps)
                else:
                    self._memory.compress_overflow(overflow)
            else:
                self._memory.compress_overflow(overflow)
        if self._facts_enabled and self._facts_mode == "per_message":
            self._schedule_store_facts(bundle.facts, participants)
        self._update_participant_hints(bundle.participant_hints)
        self._update_mood_and_status(bundle, source="chat")
        if bundle.actions or bundle.text:
            self._last_response_ts = time.time()
        self._tracer.log_event("response", {"actions": len(bundle.actions), "batch": len(chats)})
        self._schedule_episode_check()
        return bundle.actions

    def _build_chat(self, data: Dict[str, Any]) -> InboundChat:
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
            # Internally we prefer stable usernames over UUIDs or stylized display names.
            sender_name=sender_username,
            timestamp=float(data.get("timestamp", time.time())),
            raw=raw_payload,
        )

    def _resolve_participants(self, chat: InboundChat) -> List[Participant]:
        participants: Dict[str, Participant] = {}

        def add_participant(user_id: str, name: str) -> None:
            if not user_id and not name:
                return
            if self._is_self_participant(user_id, name):
                return
            self._record_display_name(user_id, name)
            canonical_name = self._canonical_name(user_id, name)
            key = user_id or f"name:{self._name_key(canonical_name)}"
            participants[key] = Participant(user_id=user_id, name=canonical_name)

        for hint in self._participant_hints:
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
        for agent in self._environment.agents[: self._max_environment_participants]:
            user_id = agent.get("uuid") or agent.get("target_uuid", "")
            name = agent.get("name", "")
            if user_id == chat.sender_id:
                continue
            self._record_display_name(str(user_id or ""), str(name or ""))
            display_name = str(name or "")
            if display_name and display_name.lower() in text_lower:
                add_participant(user_id, display_name)
        return list(participants.values())

    def _merge_participants(self, chats: List[InboundChat]) -> List[Participant]:
        merged: Dict[str, Participant] = {}
        for chat in chats:
            for participant in self._resolve_participants(chat):
                key = participant.user_id or f"name:{self._name_key(participant.name)}"
                merged[key] = participant
        return list(merged.values())

    async def _build_context(
        self,
        chat: InboundChat | None,
        participants: List[Participant],
        query_text: str | None = None,
    ) -> ConversationContext:
        user_ids = [participant.user_id for participant in participants if participant.user_id]
        people = await asyncio.to_thread(self._knowledge_store.fetch_people, user_ids)
        unresolved_names = []
        for participant in participants:
            if participant.user_id and participant.user_id in people:
                continue
            # Only attempt name-based resolution when no user_id is available.
            if not participant.user_id and participant.name:
                unresolved_names.append(participant.name)
        if unresolved_names:
            extras = await asyncio.to_thread(
                self._knowledge_store.fetch_people_by_name,
                list(dict.fromkeys(unresolved_names)),
            )
            for user_id, profile in extras.items():
                if user_id not in people:
                    people[user_id] = profile
        match_metadata: Dict[str, Dict[str, str]] = {}
        remaining_names = []
        known_names = {profile.name.lower() for profile in people.values() if profile.name}
        for name in unresolved_names:
            if name.lower() in known_names:
                continue
            remaining_names.append(name)
        for name in remaining_names:
            tokens = self._partial_name_tokens(name)
            if not tokens:
                continue
            matches = await asyncio.to_thread(
                self._knowledge_store.fetch_people_by_partial_name,
                tokens,
            )
            if len(matches) != 1:
                continue
            user_id, profile = next(iter(matches.items()))
            if user_id not in people:
                people[user_id] = profile
            match_metadata[user_id] = {
                "match_type": "partial_name",
                "matched_query": name,
            }
        recent = self._memory.recent()
        query = (query_text or (chat.text if chat else "")).strip()
        related = self._similarity.search(
            query,
            self._experience_store.all(),
            top_k=self._experience_top_k,
        )
        if self._experience_vector_index and self._experience_vector_index.is_enabled():
            semantic_related = await self._experience_vector_index.search(
                query, persona_id=self._persona, top_k=self._experience_top_k
            )
            semantic_related = self._gate_semantic_experiences(semantic_related)
            related = self._merge_related_experiences(semantic_related, related)
        now_ts = time.time()
        summary_meta_raw = self._memory.summary_meta()
        summary_meta: Dict[str, Any] = dict(summary_meta_raw)
        last_updated_ts = float(summary_meta_raw.get("last_updated_ts", 0.0) or 0.0)
        range_end_ts = float(summary_meta_raw.get("range_end_ts", 0.0) or 0.0)
        if last_updated_ts > 0.0:
            summary_meta["age_seconds"] = max(0.0, now_ts - last_updated_ts)
        if range_end_ts > 0.0:
            summary_meta["range_age_seconds"] = max(0.0, now_ts - range_end_ts)
        people_facts: Dict[str, Dict[str, Any]] = {}
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
            metadata = match_metadata.get(user_id)
            if metadata:
                entry.update(metadata)
            people_facts[user_id] = entry
        agent_state = self._agent_state(now_ts)
        persona_instructions = self._persona_instructions()
        return ConversationContext(
            persona=self._persona,
            user_id=self._user_id,
            environment=self._environment,
            participants=participants,
            people_facts=people_facts,
            recent_messages=self._memory_items_to_chats(recent),
            summary=self._memory.summary(),
            related_experiences=[{"text": item.text, "metadata": item.metadata} for item in related],
            summary_meta=summary_meta,
            agent_state=agent_state,
            persona_instructions=persona_instructions,
        )

    def _collapse_batch(self, chats: List[InboundChat]) -> List[Dict[str, Any]]:
        groups: Dict[str, Dict[str, Any]] = {}
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

    def _handle_self_message(self, chat: InboundChat) -> None:
        text = chat.text.strip()
        if not text:
            return
        if self._is_duplicate_ai_text(text):
            return
        self._rolling_buffer.add_ai_message(text, self._persona)
        self._memory.add_ai_message(text, self._persona)

    def _is_duplicate_ai_text(self, text: str) -> bool:
        if not text:
            return True
        recent = self._rolling_buffer.items()[-5:]
        for item in recent:
            if item.sender_id != "ai":
                continue
            if item.sender_name not in {self._user_id, self._persona}:
                continue
            if item.text == text:
                return True
        return False

    def _is_self_message(self, chat: InboundChat) -> bool:
        sender_id = str(chat.sender_id or "")
        if sender_id and self._user_id and sender_id == self._user_id:
            return True
        sender_full_name = self._chat_full_name(chat)
        if not sender_full_name:
            return False
        for candidate in self._self_name_candidates(chat):
            if name_matches(sender_full_name, candidate):
                return True
        return False

    def _chat_full_name(self, chat: InboundChat) -> str:
        raw = chat.raw if isinstance(chat.raw, dict) else {}
        sender_id = str(chat.sender_id or "")
        return (
            str(raw.get("_sender_full_name") or "")
            or str(raw.get("from_name") or raw.get("sender_name") or "")
            or self._full_name_for(sender_id, chat.sender_name)
        )

    def _self_name_candidates(self, chat: InboundChat) -> List[str]:
        raw = chat.raw if isinstance(chat.raw, dict) else {}
        candidates: List[str] = []

        def add(value: str) -> None:
            if not value:
                return
            candidates.append(value)
            username = extract_username(value)
            if username and username != value:
                candidates.append(username)

        add(self._persona)
        add(str(raw.get("logged_in_agent") or ""))
        add(str(raw.get("persona") or raw.get("ai_name") or ""))

        deduped: List[str] = []
        seen: Set[str] = set()
        for value in candidates:
            key = value.casefold()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(value)
        return deduped

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

    @staticmethod
    def _dedupe_preserve_order(items: List[str]) -> List[str]:
        seen: set[str] = set()
        deduped: List[str] = []
        for item in items:
            if not item or item in seen:
                continue
            seen.add(item)
            deduped.append(item)
        return deduped

    def _record_display_name(self, user_id: str, full_name: str) -> None:
        if not user_id or not full_name:
            return
        if self._is_self_participant(user_id, full_name):
            return
        prior = self._display_names_by_id.get(user_id, "")
        # Prefer richer full-name strings (e.g., "Display (username)").
        if not prior or len(full_name) >= len(prior):
            self._display_names_by_id[user_id] = full_name

    def _full_name_for(self, user_id: str, fallback_name: str = "") -> str:
        if user_id and user_id in self._display_names_by_id:
            return self._display_names_by_id[user_id]
        return fallback_name or user_id

    def _participant_payload(self, participant: Participant) -> Dict[str, Any]:
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

    def _update_last_seen_cache_from_agents(self, agents: List[Dict[str, Any]], timestamp: float) -> None:
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

    def _update_last_seen_from_agents(self, agents: List[Dict[str, Any]], timestamp: float) -> None:
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

    def _schedule_update_last_seen_from_agents(self, agents: List[Dict[str, Any]], timestamp: float) -> None:
        if not agents:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._update_last_seen_from_agents(agents, timestamp)
            return
        loop.create_task(asyncio.to_thread(self._update_last_seen_from_agents, agents, timestamp))

    def _memory_items_to_chats(self, items: List[MemoryItem]) -> List[InboundChat]:
        chats: List[InboundChat] = []
        for item in items:
            sender_id = str(item.sender_id or "")
            is_ai_marker = sender_id == "ai"
            sender_id_out = (self._user_id or "") if is_ai_marker else sender_id
            sender_name_out = self._persona if is_ai_marker else (sender_id or str(item.sender_name or ""))
            raw_payload: Dict[str, Any] = {}
            if is_ai_marker:
                raw_payload["_sender_full_name"] = self._persona
                raw_payload["_sender_display_name"] = self._persona
                raw_payload["_sender_username"] = self._persona
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

    def _chat_payload(self, chat: Any) -> Dict[str, Any]:
        raw_value = getattr(chat, "raw", {})
        raw = raw_value if isinstance(raw_value, dict) else {}
        sender_id = str(getattr(chat, "sender_id", "") or "")
        sender_name = str(getattr(chat, "sender_name", "") or "")
        text = str(getattr(chat, "text", "") or "")
        timestamp = float(getattr(chat, "timestamp", 0.0) or 0.0)
        if sender_id == "ai":
            persona = str(self._persona or "")
            sender_id_out = str(self._user_id or "")
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

    def _participant_key(self, user_id: str, name: str) -> str:
        if user_id:
            return f"id:{user_id}"
        return f"name:{self._name_key(name)}"

    def _participants_from_messages(
        self,
        messages: List[InboundChat],
        base_participants: List[Participant],
    ) -> List[Participant]:
        merged: Dict[str, Participant] = {}
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
            if message.sender_id and isinstance(message.raw, dict):
                full_name = str(message.raw.get("_sender_full_name") or message.raw.get("from_name") or "")
                self._record_display_name(message.sender_id, full_name)
            canonical_name = self._canonical_name(message.sender_id, message.sender_name)
            participant = Participant(user_id=message.sender_id, name=canonical_name)
            if self._is_self_participant(participant.user_id, participant.name):
                continue
            key = self._participant_key(participant.user_id, participant.name)
            merged[key] = participant
        return list(merged.values())

    def _facts_context(self, participants: List[Participant], evidence_messages: List[InboundChat]) -> ConversationContext:
        evidence = list(evidence_messages)[-self._facts_evidence_max_messages :]
        now_ts = time.time()
        return ConversationContext(
            persona=self._persona,
            user_id=self._user_id,
            environment=self._environment,
            participants=participants,
            people_facts={},
            recent_messages=evidence,
            summary="",
            related_experiences=[],
            summary_meta={},
            agent_state=self._agent_state(now_ts),
        )

    def _maybe_schedule_fact_sweep(
        self,
        recent_chats: List[InboundChat],
        overflow_chats: List[InboundChat],
        participants: List[Participant],
    ) -> None:
        if not self._facts_enabled:
            return
        if self._facts_mode != "periodic":
            return
        facts_participants = self._participants_from_messages(recent_chats + overflow_chats, participants)
        if overflow_chats:
            self._enqueue_fact_messages(overflow_chats, facts_participants)
        now = time.time()
        if (now - self._facts_last_sweep_ts) < self._facts_interval_seconds:
            return
        recent_items = self._memory.recent()
        new_items = [item for item in recent_items if float(item.timestamp or 0.0) > self._facts_cursor_ts]
        new_chats = self._memory_items_to_chats(new_items)
        if new_chats:
            self._enqueue_fact_messages(new_chats, facts_participants)
        self._facts_last_sweep_ts = now

    def _enqueue_fact_messages(self, messages: List[InboundChat], participants: List[Participant]) -> None:
        if not self._facts_enabled:
            return
        for message in messages:
            if self._is_self_message(message):
                continue
            self._facts_pending_messages.append(message)
        for participant in participants:
            if self._is_self_participant(participant.user_id, participant.name):
                continue
            key = self._participant_key(participant.user_id, participant.name)
            self._facts_pending_participants[key] = participant
        max_pending = max(self._facts_evidence_max_messages * 3, self._facts_evidence_max_messages)
        if len(self._facts_pending_messages) > max_pending:
            self._facts_pending_messages = self._facts_pending_messages[-max_pending:]
        if self._facts_task is None or self._facts_task.done():
            self._facts_task = asyncio.create_task(self._facts_worker())

    async def _facts_worker(self) -> None:
        while self._facts_pending_messages:
            pending_messages = self._facts_pending_messages[-self._facts_evidence_max_messages :]
            self._facts_pending_messages = []
            pending_participants = list(self._facts_pending_participants.values())
            participants = self._participants_from_messages(pending_messages, pending_participants)
            cursor_before = self._facts_cursor_ts
            sweep = await asyncio.to_thread(self._extract_and_store_facts, pending_messages, participants)
            max_ts = float(sweep.get("max_ts", 0.0) or 0.0)
            if max_ts > self._facts_cursor_ts:
                self._facts_cursor_ts = max_ts
            self._tracer.log_event(
                "facts_sweep",
                {
                    "mode": self._facts_mode,
                    "messages": int(sweep.get("messages", len(pending_messages))),
                    "participants": int(sweep.get("participants", len(participants))),
                    "facts_extracted": int(sweep.get("facts_extracted", 0)),
                    "fact_strings_extracted": int(sweep.get("fact_strings_extracted", 0)),
                    "fact_strings_stored": int(sweep.get("fact_strings_stored", 0)),
                    "people_updated": int(sweep.get("people_updated", 0)),
                    "cursor_before": cursor_before,
                    "cursor_after": self._facts_cursor_ts,
                    "max_ts": max_ts,
                },
            )
        self._facts_task = None
        if self._facts_pending_messages:
            self._facts_task = asyncio.create_task(self._facts_worker())

    def _extract_and_store_facts(self, messages: List[InboundChat], participants: List[Participant]) -> Dict[str, Any]:
        if not messages or not participants:
            return {
                "messages": len(messages),
                "participants": len(participants),
                "facts_extracted": 0,
                "fact_strings_extracted": 0,
                "fact_strings_stored": 0,
                "people_updated": 0,
                "max_ts": self._facts_cursor_ts,
            }
        context = self._facts_context(participants, messages)
        facts = self._llm.extract_facts_from_evidence_sync(context, messages, participants)
        fact_strings_extracted = sum(len(fact.facts) for fact in facts)
        stored_stats = {"fact_strings_stored": 0, "people_updated": 0}
        if facts:
            stored_stats = self._store_facts(facts, participants)
        timestamps = [float(message.timestamp or 0.0) for message in messages]
        return {
            "messages": len(messages),
            "participants": len(participants),
            "facts_extracted": len(facts),
            "fact_strings_extracted": fact_strings_extracted,
            "fact_strings_stored": int(stored_stats.get("fact_strings_stored", 0)),
            "people_updated": int(stored_stats.get("people_updated", 0)),
            "max_ts": max(timestamps) if timestamps else self._facts_cursor_ts,
        }

    def _store_facts(self, facts: List[ExtractedFact], participants: List[Participant]) -> Dict[str, int]:
        alias_to_id: Dict[str, str] = {}
        for participant in participants:
            if not participant.user_id:
                continue
            user_key = self._normalize_name_for_match(participant.user_id)
            if user_key:
                alias_to_id[user_key] = participant.user_id
            name_key = self._normalize_name_for_match(participant.name)
            if name_key:
                alias_to_id[name_key] = participant.user_id
        fact_strings_stored = 0
        people_updated = 0
        for fact in facts:
            user_id = fact.user_id
            if not user_id and fact.name:
                key = self._normalize_name_for_match(fact.name)
                user_id = alias_to_id.get(key, "") if key else ""
            if not user_id:
                continue
            profile = self._knowledge_store.fetch_people([user_id]).get(user_id)
            existing_facts = set(profile.facts) if profile else set()
            deduped_new_facts = self._dedupe_preserve_order(fact.facts)
            missing_facts = [item for item in deduped_new_facts if item not in existing_facts]
            name_changed = False
            if fact.name:
                fact_key = self._normalize_name_for_match(fact.name)
                user_key = self._normalize_name_for_match(user_id)
                if fact_key and user_key and fact_key != user_key:
                    name_changed = bool(not profile or fact.name != profile.name)
            if not missing_facts and not name_changed:
                continue
            # Only persist facts that are truly new to avoid repeated writes and DB bloat.
            name_to_store = profile.name if profile and profile.name else (fact.name or user_id)
            self._knowledge_store.upsert_person_facts(user_id, name_to_store, missing_facts)
            fact_strings_stored += len(missing_facts)
            people_updated += 1
        return {"fact_strings_stored": fact_strings_stored, "people_updated": people_updated}

    def _schedule_store_facts(self, facts: List[ExtractedFact], participants: List[Participant]) -> None:
        if not facts:
            return
        asyncio.create_task(asyncio.to_thread(self._store_facts, facts, participants))

    def _update_participant_hints(self, hints: List[ParticipantHint]) -> None:
        if not hints:
            return
        merged: Dict[str, Participant] = {}
        for hint in self._participant_hints:
            self._record_display_name(hint.user_id, hint.name)
            canonical_name = self._canonical_name(hint.user_id, hint.name)
            key = hint.user_id or f"name:{self._name_key(canonical_name)}"
            merged[key] = Participant(user_id=hint.user_id, name=canonical_name)
        for hint in hints:
            if self._is_self_participant(hint.user_id, hint.name):
                continue
            self._record_display_name(hint.user_id, hint.name)
            canonical_name = self._canonical_name(hint.user_id, hint.name)
            key = hint.user_id or f"name:{self._name_key(canonical_name)}"
            merged[key] = Participant(user_id=hint.user_id, name=canonical_name)
        self._participant_hints = list(merged.values())[: self._max_environment_participants]

    def _is_self_participant(self, user_id: str, name: str) -> bool:
        if user_id and self._user_id and user_id == self._user_id:
            return True
        if not name:
            return False
        return name_matches(name, self._persona)

    def seconds_since_activity(self, now: float | None = None) -> float:
        current = now or time.time()
        last_activity = max(self._last_inbound_ts, self._last_response_ts)
        return max(0.0, current - last_activity)

    def activity_snapshot(self, recent_activity_window_seconds: float) -> Dict[str, Any]:
        seconds = self.seconds_since_activity()
        return {
            "seconds_since_activity": seconds,
            "last_inbound_ts": self._last_inbound_ts,
            "last_response_ts": self._last_response_ts,
            "recent_messages": len(self._memory.recent()),
            "recent_activity_window_seconds": recent_activity_window_seconds,
            "mood": self._current_mood,
            "mood_ts": self._current_mood_ts,
            "mood_source": self._mood_source,
            "status": self._current_status,
            "status_ts": self._current_status_ts,
            "status_source": self._status_source,
        }

    def _agent_state(self, now_ts: float | None = None) -> Dict[str, Any]:
        current = now_ts or time.time()
        mood_ts = float(self._current_mood_ts or 0.0)
        status_ts = float(self._current_status_ts or 0.0)
        mood_seconds_ago = (current - mood_ts) if mood_ts > 0.0 else None
        status_seconds_ago = (current - status_ts) if status_ts > 0.0 else None
        return {
            "mood": self._current_mood,
            "mood_ts": format_pacific_time(mood_ts),
            "mood_seconds_ago": mood_seconds_ago,
            "mood_source": self._mood_source,
            "status": self._current_status,
            "status_ts": format_pacific_time(status_ts),
            "status_seconds_ago": status_seconds_ago,
            "status_source": self._status_source,
            "last_inbound_ts": self._last_inbound_ts,
            "last_response_ts": self._last_response_ts,
            "seconds_since_activity": self.seconds_since_activity(current),
        }

    def _update_mood_and_status(self, bundle: LLMResponseBundle, source: str) -> None:
        now = time.time()
        mood = self._clean_optional_text(bundle.mood)
        status = self._clean_optional_text(bundle.status)
        if mood:
            self._current_mood = mood
            self._current_mood_ts = now
            self._mood_source = source
        status_candidate = status or self._derive_status_from_actions(bundle)
        if status_candidate:
            self._current_status = status_candidate
            self._current_status_ts = now
            self._status_source = source

    @staticmethod
    def _clean_optional_text(value: Any) -> str:
        if not isinstance(value, str):
            return ""
        cleaned = value.strip()
        return cleaned if cleaned else ""

    def _derive_status_from_actions(self, bundle: LLMResponseBundle) -> str:
        if bundle.actions:
            return self._status_from_command(bundle.actions[0].command_type)
        if bundle.text:
            return "chatting"
        return ""

    @staticmethod
    def _status_from_command(command_type: CommandType) -> str:
        mapping = {
            CommandType.CHAT: "chatting",
            CommandType.EMOTE: "emoting",
            CommandType.MOVE: "moving",
            CommandType.TOUCH: "interacting",
            CommandType.SIT: "sitting",
            CommandType.STAND: "standing",
            # CommandType.LOOK_AT: "looking",
            # CommandType.WALK_TO: "walking",
            # CommandType.TURN_TO: "turning",
            # CommandType.GESTURE: "gesturing",
            # CommandType.FOLLOW: "following",
        }
        return mapping.get(command_type, "active")

    def _chat_texts_from_actions(self, actions: List[Action]) -> List[str]:
        texts: List[str] = []
        for action in actions:
            if action.command_type not in {CommandType.CHAT, CommandType.EMOTE}:
                continue
            content = (action.content or "").strip()
            if not content:
                continue
            texts.append(content)
        return texts

    def _filter_autonomous_actions(self, actions: List[Action], participants: List[Participant]) -> List[Action]:
        if not actions:
            return []
        # Use the persona/username (not the UUID) for self-reference filtering.
        persona_key = self._normalize_name_for_match(self._persona)
        participant_keys = {
            self._normalize_name_for_match(participant.user_id or participant.name)
            for participant in participants
            if participant.user_id or participant.name
        }
        participant_keys.discard("")
        wait_tokens = ("wait", "waiting", "waited")
        filtered: List[Action] = []
        for action in actions:
            if action.command_type not in {CommandType.CHAT, CommandType.EMOTE}:
                filtered.append(action)
                continue
            content = (action.content or "").strip()
            if not content:
                continue
            content_lower = content.casefold()
            content_key = self._normalize_name_for_match(content)
            mentions_persona = bool(persona_key and persona_key in content_key)
            mentions_wait = any(token in content_lower for token in wait_tokens)
            mentions_other = any(key and key in content_key for key in participant_keys)
            # Guard against autonomy narrating that the persona is waiting for itself.
            if mentions_persona and mentions_wait and not mentions_other:
                continue
            filtered.append(action)
        return filtered

    async def generate_autonomous_actions(self, recent_activity_window_seconds: float) -> List[Any]:
        activity = self.activity_snapshot(recent_activity_window_seconds)
        if activity["seconds_since_activity"] < recent_activity_window_seconds:
            return []
        participants = self._participants_for_autonomy()
        query_text = self._autonomy_query_text()
        context = await self._build_context(None, participants, query_text=query_text)
        self._tracer.log_event(
            "llm_prompt_autonomy",
            self._bundle_prompt_payload(None, context, None, None, activity=activity, mode="autonomous"),
        )
        bundle = await self._llm.generate_autonomous_bundle(context, activity)
        bundle.actions = self._filter_autonomous_actions(bundle.actions, participants)
        self._tracer.log_event("llm_response_autonomy", self._bundle_response_payload(bundle))
        chat_texts = self._chat_texts_from_actions(bundle.actions)
        for text in chat_texts:
            self._rolling_buffer.add_ai_message(text, self._persona)
            self._memory.add_ai_message(text, self._persona)
        if bundle.actions or chat_texts:
            self._last_response_ts = time.time()
        if self._facts_enabled and self._facts_mode == "per_message":
            self._schedule_store_facts(bundle.facts, participants)
        self._update_participant_hints(bundle.participant_hints)
        self._update_mood_and_status(bundle, source="autonomy")
        self._schedule_episode_check()
        return bundle.actions

    def _participants_for_autonomy(self) -> List[Participant]:
        merged: Dict[str, Participant] = {}

        def add_participant(user_id: str, name: str) -> None:
            if self._is_self_participant(user_id, name):
                return
            if not user_id and not name:
                return
            self._record_display_name(user_id, name)
            canonical_name = self._canonical_name(user_id, name)
            key = user_id or f"name:{self._name_key(canonical_name)}"
            merged[key] = Participant(user_id=user_id, name=canonical_name)

        for hint in self._participant_hints:
            add_participant(hint.user_id, hint.name)
        for agent in self._environment.agents[: self._max_environment_participants]:
            user_id = str(agent.get("uuid") or agent.get("target_uuid") or "")
            name = str(agent.get("name") or "")
            add_participant(user_id, name)
        return list(merged.values())

    def _autonomy_query_text(self) -> str:
        summary = self._memory.summary().strip()
        if summary:
            return summary
        recent = self._memory.recent()
        if recent:
            return recent[-1].text
        return self._environment.location or "autonomous"

    def _schedule_episode_check(self) -> None:
        if not self._episode_enabled:
            return
        if self._episode_task and not self._episode_task.done():
            return
        self._episode_task = asyncio.create_task(self._maybe_finalize_episode())

    async def _maybe_finalize_episode(self) -> None:
        items = self._rolling_buffer.items()
        count = len(items)
        if count < self._episode_min_messages:
            return
        now = time.time()
        last_activity = max(self._last_inbound_ts, self._last_response_ts)
        inactivity = max(0.0, now - last_activity)
        new_items = max(0, count - self._last_episode_size)

        trigger_reason = ""
        if count >= self._episode_max_messages:
            trigger_reason = "max_messages"
        elif self._episode_inactivity_seconds > 0 and inactivity >= self._episode_inactivity_seconds:
            trigger_reason = "inactivity"
        elif (
            self._episode_forced_interval_seconds > 0
            and (now - self._last_episode_ts) >= self._episode_forced_interval_seconds
        ):
            trigger_reason = "forced_interval"

        if not trigger_reason or new_items < self._episode_min_messages:
            return

        snapshot = list(items)
        summary = await self._summarize_episode(snapshot)
        if not summary:
            return

        metadata = self._episode_metadata(snapshot, trigger_reason)
        record = self._experience_store.add(summary, metadata, persona_id=self._persona)
        if self._experience_vector_index and self._experience_vector_index.is_enabled():
            asyncio.create_task(self._experience_vector_index.add_record_async(record, self._persona))

        overlap = min(self._episode_overlap_messages, count)
        self._rolling_buffer.trim_to_last(overlap)
        self._last_episode_ts = now
        self._last_episode_size = overlap
        self._tracer.log_event(
            "episode_summary",
            {"reason": trigger_reason, "messages": count, "overlap": overlap},
        )

    async def _summarize_episode(self, items: List[Any]) -> str:
        chats = [
            InboundChat(
                text=str(getattr(item, "text", "") or ""),
                sender_id=str(getattr(item, "sender_id", "") or ""),
                sender_name=str(getattr(item, "sender_name", "") or ""),
                timestamp=float(getattr(item, "timestamp", time.time()) or time.time()),
                raw={},
            )
            for item in items
        ]
        try:
            summary = await self._llm.summarize("", chats)
        except Exception:
            summary = ""
        if summary:
            return summary.strip()
        return self._episode_compressor.compress("", items)

    @staticmethod
    def _episode_metadata(items: List[Any], reason: str) -> Dict[str, Any]:
        timestamps = [float(getattr(item, "timestamp", 0.0) or 0.0) for item in items]
        sender_names: List[str] = []
        seen: Set[str] = set()
        for item in items:
            name = str(getattr(item, "sender_name", "") or "")
            if not name or name in seen:
                continue
            seen.add(name)
            sender_names.append(name)
        return {
            "source": "episode_summary",
            "reason": reason,
            "message_count": len(items),
            "timestamp_start": format_pacific_time(min(timestamps)) if timestamps else "0",
            "timestamp_end": format_pacific_time(max(timestamps)) if timestamps else "0",
            "sender_names": sender_names,
        }

    def _latest_timestamp(self) -> float:
        timestamps: List[float] = []
        for item in self._rolling_buffer.items():
            timestamps.append(float(getattr(item, "timestamp", 0.0) or 0.0))
        for item in self._memory.recent():
            timestamps.append(float(getattr(item, "timestamp", 0.0) or 0.0))
        return max(timestamps) if timestamps else 0.0

    def _environment_payload(self, object_limit: int = 20) -> Dict[str, Any]:
        return {
            "location": self._environment.location,
            "avatar_position": self._environment.avatar_position,
            "agents": self._environment.agents[: self._max_environment_participants],
            "objects": self._environment.objects[:object_limit],
        }

    def _bundle_prompt_payload(
        self,
        chat: InboundChat | None,
        context: ConversationContext,
        overflow: List[InboundChat] | None,
        incoming_batch: List[Dict[str, Any]] | None,
        *,
        activity: Dict[str, Any] | None = None,
        mode: str = "chat",
    ) -> Dict[str, Any]:
        recent_timestamps = [float(m.timestamp or 0.0) for m in context.recent_messages]
        recent_time_range = {
            "start": min(recent_timestamps) if recent_timestamps else 0.0,
            "end": max(recent_timestamps) if recent_timestamps else 0.0,
        }
        payload: Dict[str, Any] = {
            "mode": mode,
            "now_timestamp": format_pacific_time(),
            "persona": context.persona,
            "user_id": context.user_id,
            "participants": [self._participant_payload(p) for p in context.participants],
            "environment": self._environment_payload(),
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
            "incoming_batch": incoming_batch or [],
        }
        if chat is not None:
            payload["incoming"] = self._chat_payload(chat)
            incoming_id = str(chat.sender_id or "")
            payload["incoming_sender_id"] = incoming_id
            payload["incoming_sender_known"] = bool(incoming_id and incoming_id in context.people_facts)
        if overflow:
            payload["overflow_messages"] = [
                self._chat_payload(m) for m in overflow
            ]
        if activity is not None:
            payload["activity"] = activity
        return payload

    def _persona_instructions(self) -> str:
        if not self._persona_profiles:
            return ""
        key = str(self._persona or "").casefold()
        return str(self._persona_profiles.get(key, "") or "")

    @staticmethod
    def _bundle_response_payload(bundle: LLMResponseBundle) -> Dict[str, Any]:
        return {
            "text": bundle.text,
            "actions": [
                {
                    "type": action.command_type.value,
                    "content": action.content,
                    "target_uuid": action.target_uuid,
                    "parameters": action.parameters,
                }
                for action in bundle.actions
            ],
            "facts": [
                {"user_id": fact.user_id, "name": fact.name, "facts": fact.facts}
                for fact in bundle.facts
            ],
            "participant_hints": [
                {"user_id": hint.user_id, "name": hint.name}
                for hint in bundle.participant_hints
            ],
            "summary": bundle.summary,
            "mood": bundle.mood,
            "status": bundle.status,
        }

    @staticmethod
    def _partial_name_tokens(name: str) -> List[str]:
        cleaned = name.strip()
        if not cleaned:
            return []
        tokens = [token for token in cleaned.split() if len(token) >= 4]
        if len(cleaned) >= 4:
            tokens.append(cleaned)
        deduped = []
        seen = set()
        for token in tokens:
            lower = token.lower()
            if lower in seen:
                continue
            seen.add(lower)
            deduped.append(lower)
        return deduped

    @staticmethod
    def _merge_related_experiences(
        primary: List[Any],
        secondary: List[Any],
        limit: int = 6,
    ) -> List[Any]:
        merged: List[Any] = []
        seen = set()

        def add_item(item: Any) -> None:
            metadata = getattr(item, "metadata", {}) or {}
            experience_id = ""
            if isinstance(metadata, dict):
                experience_id = str(metadata.get("experience_id", "") or "")
            key = experience_id or (getattr(item, "text", ""), str(metadata))
            if key in seen:
                return
            seen.add(key)
            merged.append(item)

        for item in primary:
            add_item(item)
            if len(merged) >= limit:
                return merged
        for item in secondary:
            add_item(item)
            if len(merged) >= limit:
                break
        return merged

    def _gate_semantic_experiences(self, experiences: List[ExperienceRecord]) -> List[ExperienceRecord]:
        if not experiences:
            return experiences
        scored: List[tuple[float, ExperienceRecord]] = []
        for item in experiences:
            metadata = item.metadata if isinstance(item.metadata, dict) else {}
            score = metadata.get("score") if isinstance(metadata, dict) else None
            try:
                score_value = float(score)
            except (TypeError, ValueError):
                continue
            scored.append((score_value, item))
        if not scored:
            return experiences
        scored.sort(key=lambda pair: pair[0], reverse=True)
        top_score = scored[0][0]
        if top_score < self._experience_score_min:
            return []
        gated: List[ExperienceRecord] = []
        for score, item in scored:
            if score < self._experience_score_min:
                continue
            if (top_score - score) > self._experience_score_delta:
                continue
            gated.append(item)
            if len(gated) >= self._experience_top_k:
                break
        return gated

    @staticmethod
    def _normalize_entities(entries: List[Any]) -> List[Any]:
        normalized: List[Any] = []
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
