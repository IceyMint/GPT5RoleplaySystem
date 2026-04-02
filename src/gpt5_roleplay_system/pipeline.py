from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from difflib import SequenceMatcher
import logging
import time
import unicodedata
from typing import Any, Dict, List, Protocol, Set

from .config import EpisodeConfig, FactsConfig
from .llm import ExtractedFact, LLMClient, LLMResponseBundle, LLMStateUpdate, ParticipantHint
from .llm_prompts import PromptManager
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
from .autonomy_manager import AutonomyManager
from .context_builder import ContextBuilder
from .episode_manager import EpisodeManager
from .fact_manager import FactManager
from .pipeline_state import PipelineRuntimeState
from .name_utils import (
    extract_display_name,
    extract_username,
    name_matches,
    normalize_display_name,
    normalize_for_match,
    split_display_and_username,
)
from .payload_contract import (
    canonical_identity_key,
    is_ignored_user_id,
    is_placeholder_self_id,
    looks_like_uuid,
    normalize_participants,
)
from .neo4j_store import KnowledgeStore
from .observability import Tracer

logger = logging.getLogger("gpt5_roleplay_pipeline")
SUMMARY_BOUNDARY_WARN_EPSILON_SECONDS = 0.001
REAPPEARANCE_THRESHOLD_SECONDS = 300.0
REAPPEARANCE_SIGNAL_TTL_SECONDS = 180.0
MENTION_INDEX_REFRESH_INTERVAL_SECONDS = 120.0
MENTION_INDEX_MAX_SCAN_WORDS = 8


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
        near_duplicate_collapse_enabled: bool = True,
        near_duplicate_similarity: float = 0.9,
        routine_summary_enabled: bool = False,
        routine_summary_limit: int = 2,
        routine_summary_min_count: int = 2,
        max_environment_participants: int = 10,
        posture_stale_seconds: float = 6.0,
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
        self._near_duplicate_collapse_enabled = bool(near_duplicate_collapse_enabled)
        self._near_duplicate_similarity = min(1.0, max(0.0, float(near_duplicate_similarity)))
        self._routine_summary_enabled = bool(routine_summary_enabled)
        self._routine_summary_limit = max(0, int(routine_summary_limit))
        self._routine_summary_min_count = max(2, int(routine_summary_min_count))
        self._routine_candidate_limit = max(self._experience_top_k, self._experience_top_k * 4)
        self._max_environment_participants = max_environment_participants
        self._posture_stale_seconds = max(0.0, float(posture_stale_seconds))
        self._participant_hints: List[Participant] = []
        self._persona_profiles = {str(k).casefold(): str(v) for k, v in (persona_profiles or {}).items() if k and v}
        # Tracks viewer-provided full names by UUID so we can pass display+username to the LLM.
        self._display_names_by_id: Dict[str, str] = {}
        self._mention_runtime_names_by_id: Dict[str, Set[str]] = {}
        self._mention_index_by_phrase: Dict[str, Set[str]] = {}
        self._mention_index_names_by_id: Dict[str, Set[str]] = {}
        self._mention_index_max_phrase_words = 1
        self._mention_index_dirty = True
        self._mention_index_last_refresh_ts = 0.0
        self._llm_chat_enabled = True
        facts = facts_config or FactsConfig()
        self._facts_enabled = bool(facts.enabled)
        mode = str(facts.mode or "periodic").strip().lower()
        self._facts_mode = mode if mode in {"periodic", "per_message"} else "periodic"
        self._facts_interval_seconds = max(1.0, float(facts.interval_seconds))
        self._facts_evidence_max_messages = max(4, int(facts.evidence_max_messages))
        self._facts_min_pending_messages = max(1, int(facts.min_pending_messages))
        self._facts_max_pending_age_seconds = max(1.0, float(facts.max_pending_age_seconds))
        self._facts_flush_on_overflow = bool(facts.flush_on_overflow)
        self._facts_last_sweep_ts = time.time()
        self._facts_cursor_ts = 0.0
        self._facts_pending_messages: List[InboundChat] = []
        self._facts_pending_keys: Set[str] = set()
        self._facts_pending_since_ts = 0.0
        self._facts_pending_participants: Dict[str, Participant] = {}
        self._facts_task: asyncio.Task | None = None
        self._last_seen_cache: Dict[str, float] = {}
        self._reappearance_signals: Dict[str, Dict[str, float]] = {}
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
        self._autonomy_delay_hint_seconds: float | None = None
        self._autonomy_decision = "wait"
        self._mood_source = "init"
        self._status_source = "init"
        logger.info(
            "Pipeline LLM client: %s (persona=%s, user_id=%s)",
            type(self._llm).__name__,
            self._persona,
            self._user_id,
        )
        self._runtime_state = PipelineRuntimeState(
            persona=self._persona,
            user_id=self._user_id,
            posture_stale_seconds=self._posture_stale_seconds,
        )
        self._runtime_state.llm_chat_enabled = self._llm_chat_enabled
        self._runtime_state.environment = self._environment
        self._runtime_state.posture_known = False
        self._runtime_state.posture_is_sitting = self._environment.is_sitting
        self._runtime_state.participant_hints = self._participant_hints
        self._runtime_state.display_names_by_id = self._display_names_by_id
        self._autonomy_manager_component = AutonomyManager(self._runtime_state)
        self._context_builder_component = ContextBuilder(
            state=self._runtime_state,
            knowledge_store=self._knowledge_store,
            memory=self._memory,
            experience_store=self._experience_store,
            tracer=self._tracer,
            experience_vector_index=self._experience_vector_index,
            experience_top_k=self._experience_top_k,
            experience_score_min=self._experience_score_min,
            experience_score_delta=self._experience_score_delta,
            near_duplicate_collapse_enabled=self._near_duplicate_collapse_enabled,
            near_duplicate_similarity=self._near_duplicate_similarity,
            routine_summary_enabled=self._routine_summary_enabled,
            routine_summary_limit=self._routine_summary_limit,
            routine_summary_min_count=self._routine_summary_min_count,
            max_environment_participants=self._max_environment_participants,
            persona_profiles=self._persona_profiles,
            owner=self,
        )
        self._fact_manager_component = FactManager(
            state=self._runtime_state,
            llm=self._llm,
            knowledge_store=self._knowledge_store,
            tracer=self._tracer,
            facts_config=facts,
            context_builder=self._context_builder_component,
            autonomy_manager=self._autonomy_manager_component,
            owner=self,
        )
        self._episode_manager_component = EpisodeManager(
            state=self._runtime_state,
            llm=self._llm,
            rolling_buffer=self._rolling_buffer,
            experience_store=self._experience_store,
            tracer=self._tracer,
            episode_config=episode,
            experience_vector_index=self._experience_vector_index,
            compressor=self._episode_compressor,
        )

    def snapshot_state(self) -> Dict[str, Any]:
        return {
            "rolling_buffer": self._rolling_buffer.snapshot(),
            "memory": self._memory.snapshot(),
            "facts": self._snapshot_facts_state(),
            "identity": {
                "display_names_by_id": dict(self._display_names_by_id),
                "last_seen_cache": dict(self._last_seen_cache),
            },
            "environment": {
                "last_environment_update_ts": self._runtime_state.last_environment_update_ts,
                "last_posture_update_ts": self._runtime_state.last_posture_update_ts,
                "posture_known": self._runtime_state.posture_known,
                "posture_is_sitting": self._runtime_state.posture_is_sitting,
                "posture_stale_seconds": self._posture_stale_seconds,
            },
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
        identity_state = state.get("identity", {})
        identity_loaded = False
        if isinstance(identity_state, dict):
            names_raw = identity_state.get("display_names_by_id", {})
            if isinstance(names_raw, dict):
                self._display_names_by_id = {
                    str(k): cleaned_name
                    for k, v in names_raw.items()
                    if str(k)
                    and not self._is_ignored_user_id(str(k))
                    and (cleaned_name := self._clean_name_candidate(v))
                }
                identity_loaded = True
            cache_raw = identity_state.get("last_seen_cache", {})
            if isinstance(cache_raw, dict):
                self._last_seen_cache = {
                    str(k): float(v or 0.0)
                    for k, v in cache_raw.items()
                    if str(k) and not self._is_ignored_user_id(str(k))
                }
                identity_loaded = True
        if not identity_loaded:
            legacy_names = state.get("display_names_by_id", {})
            if isinstance(legacy_names, dict):
                self._display_names_by_id = {
                    str(k): cleaned_name
                    for k, v in legacy_names.items()
                    if str(k)
                    and not self._is_ignored_user_id(str(k))
                    and (cleaned_name := self._clean_name_candidate(v))
                }
            legacy_last_seen = state.get("last_seen_cache", {})
            if isinstance(legacy_last_seen, dict):
                self._last_seen_cache = {
                    str(k): float(v or 0.0)
                    for k, v in legacy_last_seen.items()
                    if str(k) and not self._is_ignored_user_id(str(k))
                }
        self._mention_runtime_names_by_id = {}
        self._mention_index_by_phrase = {}
        self._mention_index_names_by_id = {}
        self._mention_index_max_phrase_words = 1
        self._mention_index_dirty = True
        self._mention_index_last_refresh_ts = 0.0
        for user_id, full_name in self._display_names_by_id.items():
            self._remember_runtime_mention_name(user_id, full_name)
        facts_state = state.get("facts", {})
        if isinstance(facts_state, dict):
            self._restore_facts_state(facts_state)
        environment_state = state.get("environment", {})
        if isinstance(environment_state, dict):
            self._runtime_state.last_environment_update_ts = float(
                environment_state.get("last_environment_update_ts", self._runtime_state.last_environment_update_ts)
                or self._runtime_state.last_environment_update_ts
            )
            self._runtime_state.last_posture_update_ts = float(
                environment_state.get("last_posture_update_ts", self._runtime_state.last_posture_update_ts)
                or self._runtime_state.last_posture_update_ts
            )
            self._runtime_state.posture_known = bool(
                environment_state.get("posture_known", self._runtime_state.posture_known)
            )
            self._runtime_state.posture_is_sitting = bool(
                environment_state.get("posture_is_sitting", self._runtime_state.posture_is_sitting)
            )
            self._posture_stale_seconds = max(
                0.0,
                float(environment_state.get("posture_stale_seconds", self._posture_stale_seconds) or self._posture_stale_seconds),
            )
            self._runtime_state.posture_stale_seconds = self._posture_stale_seconds
            if self._runtime_state.posture_known:
                self._environment.is_sitting = self._runtime_state.posture_is_sitting
        # Backward-compatible recovery for snapshots saved before facts-state persistence.
        self._recover_pending_facts_from_memory()
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
        self._runtime_state.environment = self._environment
        self._runtime_state.participant_hints = self._participant_hints
        self._runtime_state.display_names_by_id = self._display_names_by_id
        self._runtime_state.persona = self._persona
        self._runtime_state.user_id = self._user_id
        self._runtime_state.llm_chat_enabled = self._llm_chat_enabled

    def _snapshot_facts_state(self) -> Dict[str, Any]:
        pending_messages = [self._serialize_inbound_chat_state(msg) for msg in self._facts_pending_messages]
        pending_participants = [
            {"user_id": participant.user_id, "name": participant.name}
            for participant in self._facts_pending_participants.values()
            if participant.user_id or participant.name
        ]
        return {
            "cursor_ts": float(self._facts_cursor_ts or 0.0),
            "last_sweep_ts": float(self._facts_last_sweep_ts or 0.0),
            "pending_since_ts": float(self._facts_pending_since_ts or 0.0),
            "pending_messages": pending_messages,
            "pending_participants": pending_participants,
        }

    def _restore_facts_state(self, state: Dict[str, Any]) -> None:
        self._facts_cursor_ts = float(state.get("cursor_ts", self._facts_cursor_ts) or self._facts_cursor_ts)
        self._facts_last_sweep_ts = float(state.get("last_sweep_ts", self._facts_last_sweep_ts) or self._facts_last_sweep_ts)
        self._facts_pending_messages = []
        self._facts_pending_keys = set()
        pending_raw = state.get("pending_messages", [])
        if isinstance(pending_raw, list):
            for item in pending_raw:
                chat = self._deserialize_inbound_chat_state(item)
                if chat is None or self._is_ignored_message(chat) or self._is_self_message(chat):
                    continue
                key = self._fact_message_key(chat)
                if key in self._facts_pending_keys:
                    continue
                self._facts_pending_messages.append(chat)
                self._facts_pending_keys.add(key)
        self._facts_pending_participants = {}
        pending_participants_raw = state.get("pending_participants", [])
        if isinstance(pending_participants_raw, list):
            for item in pending_participants_raw:
                if not isinstance(item, dict):
                    continue
                participant = Participant(
                    user_id=str(item.get("user_id", "") or ""),
                    name=str(item.get("name", "") or ""),
                )
                if self._is_ignored_user_id(participant.user_id) or self._is_self_participant(
                    participant.user_id, participant.name
                ):
                    continue
                key = self._participant_key(participant.user_id, participant.name)
                self._facts_pending_participants[key] = participant
        if self._facts_pending_messages:
            derived_participants = self._participants_from_messages(self._facts_pending_messages, [])
            for participant in derived_participants:
                if self._is_self_participant(participant.user_id, participant.name):
                    continue
                key = self._participant_key(participant.user_id, participant.name)
                self._facts_pending_participants[key] = participant
        pending_since = float(state.get("pending_since_ts", 0.0) or 0.0)
        self._facts_pending_since_ts = pending_since if self._facts_pending_messages else 0.0
        if self._facts_pending_messages and self._facts_pending_since_ts <= 0.0:
            self._facts_pending_since_ts = time.time()

    def _recover_pending_facts_from_memory(self) -> None:
        if not self._facts_enabled or self._facts_mode != "periodic":
            return
        recent_items = self._memory.recent()
        if not recent_items:
            return
        recent_chats = self._memory_items_to_chats(recent_items)
        if not recent_chats:
            return
        base_participants = list(self._facts_pending_participants.values())
        participants = self._participants_from_messages(recent_chats, base_participants)
        self._enqueue_fact_messages(recent_chats, participants)

    @staticmethod
    def _serialize_inbound_chat_state(chat: InboundChat) -> Dict[str, Any]:
        raw_payload = chat.raw if isinstance(chat.raw, dict) else {}
        return {
            "text": str(chat.text or ""),
            "sender_id": str(chat.sender_id or ""),
            "sender_name": str(chat.sender_name or ""),
            "timestamp": float(chat.timestamp or 0.0),
            "raw": dict(raw_payload),
        }

    @staticmethod
    def _deserialize_inbound_chat_state(raw: Any) -> InboundChat | None:
        if not isinstance(raw, dict):
            return None
        raw_payload = raw.get("raw", {})
        return InboundChat(
            text=str(raw.get("text", "") or ""),
            sender_id=str(raw.get("sender_id", "") or ""),
            sender_name=str(raw.get("sender_name", "") or ""),
            timestamp=float(raw.get("timestamp", 0.0) or 0.0),
            raw=raw_payload if isinstance(raw_payload, dict) else {},
        )

    def update_environment(self, data: Dict[str, Any]) -> None:
        agents = [
            agent
            for agent in self._normalize_entities(data.get("agents", []))
            if not (
                isinstance(agent, dict)
                and self._is_ignored_user_id(str(agent.get("uuid") or agent.get("target_uuid") or ""))
            )
        ]
        objects = self._normalize_entities(data.get("objects", []))
        raw_is_sitting = data.get("is_sitting", False)
        if isinstance(raw_is_sitting, str):
            posture_value = raw_is_sitting.strip().lower() in {"1", "true", "yes", "y", "t"}
        else:
            posture_value = bool(raw_is_sitting)
        update_ts = float(data.get("timestamp", time.time()) or time.time())
        prior_visible_ids = {
            str(agent.get("uuid") or agent.get("target_uuid") or "")
            for agent in self._environment.agents
            if isinstance(agent, dict)
        }
        for agent in agents:
            if not isinstance(agent, dict):
                continue
            user_id = str(agent.get("uuid") or agent.get("target_uuid") or "")
            name = str(agent.get("name") or "")
            self._record_display_name(user_id, name)
        self._update_reappearance_signals_from_agents(agents, update_ts, prior_visible_ids)
        self._update_last_seen_cache_from_agents(agents, update_ts)
        self._schedule_update_last_seen_from_agents(agents, update_ts)
        self._runtime_state.last_environment_update_ts = max(
            float(self._runtime_state.last_environment_update_ts or 0.0),
            update_ts,
        )
        if update_ts >= float(self._runtime_state.last_posture_update_ts or 0.0):
            self._runtime_state.last_posture_update_ts = update_ts
            self._runtime_state.posture_known = True
            self._runtime_state.posture_is_sitting = posture_value
        is_sitting_last_known = (
            self._runtime_state.posture_is_sitting
            if self._runtime_state.posture_known
            else bool(self._environment.is_sitting)
        )
        self._environment = EnvironmentSnapshot(
            agents=agents,
            objects=objects,
            location=data.get("location", ""),
            avatar_position=data.get("avatar_position", ""),
            is_sitting=is_sitting_last_known,
        )
        self._runtime_state.environment = self._environment
        self._tracer.log_event("environment_update", {"agents": len(self._environment.agents)})

    def set_persona(self, persona: str) -> None:
        self._persona = persona
        self._runtime_state.persona = persona
        self._mark_mention_index_dirty()

    def upsert_persona_profile(self, persona: str, instructions: str) -> None:
        key = str(persona or "").strip().casefold()
        value = str(instructions or "").strip()
        if not key or not value:
            return
        self._persona_profiles[key] = value
        self._context_builder_component.upsert_persona_profile(persona, value)

    def set_user_id(self, user_id: str) -> None:
        self._user_id = user_id
        self._runtime_state.user_id = user_id
        self._mark_mention_index_dirty()

    def set_llm_chat_enabled(self, enabled: bool) -> None:
        self._llm_chat_enabled = bool(enabled)
        self._runtime_state.llm_chat_enabled = self._llm_chat_enabled

    def llm_chat_enabled(self) -> bool:
        return self._llm_chat_enabled

    def user_id(self) -> str:
        return self._user_id

    async def process_chat(self, data: Dict[str, Any]) -> List[Any]:
        return await self.process_chat_batch([data])

    async def process_chat_batch(self, batch: List[Dict[str, Any]]) -> List[Any]:
        chats = [self._context_builder_component.build_chat(item) for item in batch]
        non_self_chats: List[InboundChat] = []
        for chat in chats:
            if self._is_ignored_message(chat):
                continue
            if self._is_self_message(chat):
                self._handle_self_message(chat)
            else:
                non_self_chats.append(chat)

        if not non_self_chats:
            return []

        self._last_inbound_ts = time.time()

        seed_participants = self._context_builder_component.merge_participants(non_self_chats)
        participants = self._normalize_participants(seed_participants)
        if not self._llm_chat_enabled:
            # Low-cost ingest mode: store chat into memory/experience buffers without running
            # addressed-to-me classification or bundle/state generation.
            for chat in non_self_chats:
                self._rolling_buffer.add_user_message(chat)
                self._memory.add_message(chat)
            self._fact_manager_component.maybe_schedule_periodic_sweep(non_self_chats, [], participants)
            # Ensure deferred compression doesn't accumulate unbounded while silent.
            self._memory.compress_overflow()
            self._enforce_summary_before_recent(reason="chat_disabled")
            self._tracer.log_event("response", {"actions": 0, "batch": len(chats), "chat_output_enabled": False})
            self._schedule_episode_check()
            return []

        overflow = self._filter_overflow_against_summary(self._memory.drain_overflow())
        overflow_chats = self._memory_items_to_chats(overflow)
        overflow_timestamps = [float(item.timestamp or 0.0) for item in overflow]
        recent_chats = self._memory_items_to_chats(self._memory.recent())
        participants = self._normalize_participants(
            self._participants_from_messages(
                recent_chats + overflow_chats + non_self_chats,
                participants,
            )
        )

        primary_chat = self._select_primary_incoming_chat(non_self_chats)
        context = await self._context_builder_component.build_context(
            primary_chat,
            participants,
            agent_state=self._agent_state(),
        )

        # Add messages to history AFTER building context to avoid duplication
        # of the current message in 'recent_messages' vs 'incoming'.
        for chat in non_self_chats:
            self._rolling_buffer.add_user_message(chat)
            self._memory.add_message(chat)

        self._fact_manager_component.maybe_schedule_periodic_sweep(non_self_chats, overflow_chats, participants)

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
            self._log_quality_fallbacks()
            self._tracer.log_event(
                "llm_address_result",
                {"sender": chat.sender_id or chat.sender_name, "addressed": bool(is_addressed)},
            )
            if is_addressed:
                addressed = True
                break
        if not addressed:
            if overflow:
                if self._summary_strategy == "llm":
                    summary_text = await self._llm.summarize_overflow(self._memory.summary(), overflow_chats)
                    self._log_quality_fallbacks()
                    if summary_text:
                        self._memory.apply_summary(summary_text, timestamps=overflow_timestamps)
                    else:
                        self._memory.requeue_overflow(overflow)
                else:
                    self._memory.compress_overflow(overflow)
            self._enforce_summary_before_recent(reason="not_addressed")
            self._schedule_episode_check()
            return []

        incoming_batch = self._context_builder_component.collapse_batch(non_self_chats)
        prompt_payload = self._bundle_prompt_payload(primary_chat, context, overflow_chats, incoming_batch)
        self._tracer.log_event("llm_prompt_bundle", prompt_payload)
        self._check_payload_contract(mode="chat", payload=prompt_payload)
        bundle = await self._llm.generate_bundle(primary_chat, context, overflow_chats, incoming_batch)
        self._log_quality_fallbacks()
        override_decision, override_delay = self._apply_autonomy_scheduler_override_from_incoming(
            bundle.autonomy_decision,
            bundle.next_delay_seconds,
        )
        if override_decision is not None or override_delay is not None:
            bundle.autonomy_decision = override_decision
            bundle.next_delay_seconds = override_delay
        self._tracer.log_event("llm_response_bundle", self._bundle_response_payload(bundle))
        self._log_reasoning_trace("bundle", "llm_reasoning_bundle")
        if bundle.text:
            self._rolling_buffer.add_ai_message(bundle.text, self._persona)
            self._memory.add_ai_message(bundle.text, self._persona)
        if overflow:
            if self._summary_strategy == "llm":
                summary_text = await self._llm.summarize_overflow(self._memory.summary(), overflow_chats)
                self._log_quality_fallbacks()
                if summary_text:
                    self._memory.apply_summary(summary_text, timestamps=overflow_timestamps)
                else:
                    self._memory.requeue_overflow(overflow)
            else:
                self._memory.compress_overflow(overflow)
        self._enforce_summary_before_recent(reason="after_response")
        if self._facts_enabled and self._facts_mode == "per_message":
            self._fact_manager_component.schedule_store_facts(bundle.facts, participants)
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
        full_name = self._best_available_sender_full_name(data, sender_id)
        display_name, username = split_display_and_username(full_name)
        sender_username = (
            self._clean_name_candidate(data.get("sender_username"))
            or username
            or self._clean_name_candidate(data.get("sender_name"))
            or display_name
            or full_name
            or sender_id
        ).strip()
        sender_display_name = self._clean_name_candidate(data.get("sender_display_name")) or display_name
        if sender_id and full_name:
            self._record_display_name(sender_id, full_name)
        raw_payload = dict(data)
        raw_payload["_sender_full_name"] = full_name
        raw_payload["_sender_display_name"] = sender_display_name
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
            if self._is_ignored_user_id(user_id):
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
                    if self._is_ignored_user_id(str(user_id or "")):
                        continue
                    if user_id == chat.sender_id:
                        continue
                    self._record_display_name(str(user_id or ""), str(name or ""))
                    add_participant(user_id, name)
        text_lower = chat.text.lower()
        for agent in self._environment.agents[: self._max_environment_participants]:
            user_id = agent.get("uuid") or agent.get("target_uuid", "")
            name = agent.get("name", "")
            if self._is_ignored_user_id(str(user_id or "")):
                continue
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
        return self._normalize_participants(list(merged.values()))

    async def _build_context(
        self,
        chat: InboundChat | None,
        participants: List[Participant],
        query_text: str | None = None,
    ) -> ConversationContext:
        query = (query_text or (chat.text if chat else "")).strip()
        user_ids = [participant.user_id for participant in participants if participant.user_id]
        people = await asyncio.to_thread(self._knowledge_store.fetch_people, user_ids)
        match_metadata: Dict[str, Dict[str, str]] = {}
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
        mention_matches = await asyncio.to_thread(self._resolve_text_mentions, query)
        mention_user_ids = [user_id for user_id, _matched_query in mention_matches]
        missing_mentioned_ids = [user_id for user_id in mention_user_ids if user_id and user_id not in people]
        if missing_mentioned_ids:
            mentioned_profiles = await asyncio.to_thread(
                self._knowledge_store.fetch_people,
                list(dict.fromkeys(missing_mentioned_ids)),
            )
            for user_id, profile in mentioned_profiles.items():
                if user_id not in people:
                    people[user_id] = profile
        for user_id, matched_query in mention_matches:
            if user_id not in people:
                continue
            match_metadata.setdefault(
                user_id,
                {
                    "match_type": "text_mention",
                    "matched_query": matched_query,
                },
            )
        recent = self._memory.recent()
        persona_experiences = self._experiences_for_persona(self._experience_store.all())
        candidate_limit = self._experience_top_k
        if self._routine_summary_enabled and self._routine_summary_limit > 0:
            candidate_limit = max(candidate_limit, self._routine_candidate_limit)
        lexical_candidates = self._similarity.search(
            query,
            persona_experiences,
            top_k=candidate_limit,
        )
        related = lexical_candidates[: self._experience_top_k]
        semantic_candidates: List[ExperienceRecord] = []
        if self._experience_vector_index and self._experience_vector_index.is_enabled():
            semantic_candidates = await self._experience_vector_index.search(
                query, persona_id=self._persona, top_k=candidate_limit
            )
            semantic_candidates = self._gate_semantic_experiences(semantic_candidates, limit=candidate_limit)
            related = self._merge_related_experiences(semantic_candidates[: self._experience_top_k], related)
        if self._routine_summary_enabled and self._routine_summary_limit > 0:
            routine_candidates = self._merge_related_experiences(
                semantic_candidates,
                lexical_candidates,
                limit=candidate_limit,
            )
            routine_summaries = self._build_routine_summaries(routine_candidates)
            if routine_summaries:
                related = self._merge_related_experiences(
                    related,
                    routine_summaries,
                    limit=6 + self._routine_summary_limit,
                )
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
            reappearance_signal = self._active_reappearance_signal(user_id, now_ts)
            if reappearance_signal is not None:
                entry["reappeared_after_seconds"] = reappearance_signal
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
        for index, chat in enumerate(chats):
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
                    "arrival_order": index,
                }
                groups[key] = entry
            entry["texts"].append(chat.text)
            entry["timestamps"].append(chat.timestamp)
            entry["latest_text"] = chat.text
            entry["last_timestamp"] = chat.timestamp
            entry["arrival_order"] = index
        collapsed = list(groups.values())
        collapsed.sort(
            key=lambda entry: (
                float(entry.get("last_timestamp", 0.0) or 0.0),
                int(entry.get("arrival_order", 0) or 0),
                str(canonical_identity_key(str(entry.get("sender_id", "") or ""), str(entry.get("sender_name", "") or ""))),
            )
        )
        return collapsed

    def _handle_self_message(self, chat: InboundChat) -> None:
        text = chat.text.strip()
        if not text:
            return
        self._maybe_promote_self_user_id(chat)
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

    def _maybe_promote_self_user_id(self, chat: InboundChat) -> None:
        sender_id = str(chat.sender_id or "").strip()
        if not sender_id or not looks_like_uuid(sender_id):
            return
        if looks_like_uuid(self._user_id):
            return
        if not is_placeholder_self_id(self._user_id):
            return
        self._user_id = sender_id
        self._runtime_state.user_id = sender_id

    def _self_sender_id_for_payload(self) -> str:
        user_id = str(self._user_id or "").strip()
        if looks_like_uuid(user_id):
            return user_id
        return user_id

    @staticmethod
    def _is_ignored_user_id(user_id: str) -> bool:
        return is_ignored_user_id(user_id)

    def _is_ignored_message(self, chat: InboundChat) -> bool:
        return self._is_ignored_user_id(str(chat.sender_id or ""))

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
            key = MessagePipeline._normalize_fact_key(item)
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    @staticmethod
    def _normalize_fact_key(text: str) -> str:
        if not text:
            return ""
        normalized = unicodedata.normalize("NFKC", text).casefold()
        cleaned = "".join(ch if ch.isalnum() else " " for ch in normalized)
        return " ".join(cleaned.split())

    @staticmethod
    def _fact_tokens(text: str, min_len: int = 3) -> set[str]:
        if not text:
            return set()
        normalized = MessagePipeline._normalize_fact_key(text)
        if not normalized:
            return set()
        tokens = {token for token in normalized.split() if len(token) >= min_len}
        return tokens

    @staticmethod
    def _fact_similarity(tokens_a: set[str], tokens_b: set[str]) -> float:
        if not tokens_a or not tokens_b:
            return 0.0
        intersection = tokens_a & tokens_b
        if not intersection:
            return 0.0
        union = tokens_a | tokens_b
        return len(intersection) / max(len(union), 1)

    @staticmethod
    def _is_near_duplicate(
        candidate_tokens: set[str],
        existing_tokens: List[set[str]],
        threshold: float,
    ) -> bool:
        if not candidate_tokens or len(candidate_tokens) < 2:
            return False
        for tokens in existing_tokens:
            if MessagePipeline._fact_similarity(candidate_tokens, tokens) >= threshold:
                return True
        return False

    def _people_facts_for_participants(self, participants: List[Participant]) -> Dict[str, Dict[str, Any]]:
        user_ids = [p.user_id for p in participants if p.user_id]
        if not user_ids:
            return {}
        people = self._knowledge_store.fetch_people(user_ids)
        now_ts = time.time()
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
            people_facts[user_id] = entry
        return people_facts

    def _record_display_name(self, user_id: str, full_name: str) -> None:
        normalized_user_id = str(user_id or "").strip()
        normalized_full_name = self._clean_name_candidate(full_name)
        if not normalized_user_id or not normalized_full_name:
            return
        if self._is_ignored_user_id(normalized_user_id):
            return
        if self._is_self_participant(normalized_user_id, normalized_full_name):
            return
        self._remember_runtime_mention_name(normalized_user_id, normalized_full_name)
        prior = self._display_names_by_id.get(normalized_user_id, "")
        # Prefer richer full-name strings (e.g., "Display (username)").
        if not prior or len(normalized_full_name) >= len(prior):
            self._display_names_by_id[normalized_user_id] = normalized_full_name
            self._runtime_state.display_names_by_id = self._display_names_by_id

    def _full_name_for(self, user_id: str, fallback_name: str = "") -> str:
        if user_id and user_id in self._display_names_by_id:
            return self._display_names_by_id[user_id]
        return fallback_name or user_id

    @staticmethod
    def _clean_name_candidate(value: Any) -> str:
        text = str(value or "").strip()
        if not text or looks_like_uuid(text):
            return ""
        return text

    def _best_available_sender_full_name(self, data: Dict[str, Any], sender_id: str) -> str:
        direct_full_name = (
            self._clean_name_candidate(data.get("from_name"))
            or self._clean_name_candidate(data.get("sender_full_name"))
        )
        if direct_full_name:
            return direct_full_name

        participants = data.get("participants", [])
        if sender_id and isinstance(participants, list):
            for entry in participants:
                if not isinstance(entry, dict):
                    continue
                entry_user_id = str(entry.get("user_id") or entry.get("uuid") or entry.get("target_uuid") or "")
                if entry_user_id != sender_id:
                    continue
                participant_full_name = (
                    self._clean_name_candidate(entry.get("full_name"))
                    or self._clean_name_candidate(entry.get("name"))
                )
                if participant_full_name:
                    return participant_full_name
                participant_display_name = self._clean_name_candidate(entry.get("display_name"))
                participant_username = self._clean_name_candidate(entry.get("username"))
                if participant_display_name and participant_username and not name_matches(
                    participant_display_name, participant_username
                ):
                    return f"{participant_display_name} ({participant_username})"
                participant_name = participant_display_name or participant_username
                if participant_name:
                    return participant_name

        cached_full_name = self._clean_name_candidate(self._display_names_by_id.get(sender_id, ""))
        if cached_full_name:
            return cached_full_name

        sender_display_name = self._clean_name_candidate(data.get("sender_display_name"))
        sender_username = (
            self._clean_name_candidate(data.get("sender_username"))
            or self._clean_name_candidate(data.get("sender_name"))
        )
        if sender_display_name and sender_username and not name_matches(sender_display_name, sender_username):
            return f"{sender_display_name} ({sender_username})"
        return sender_display_name or sender_username

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

    def _normalize_participants(self, participants: List[Participant]) -> List[Participant]:
        raw = [
            self._participant_payload(participant)
            for participant in participants
            if not self._is_ignored_user_id(participant.user_id)
        ]
        normalized_payload, _ = normalize_participants(raw)
        normalized: List[Participant] = []
        for entry in normalized_payload:
            user_id = str(entry.get("user_id", "") or "")
            name = str(entry.get("name", "") or "")
            full_name = str(entry.get("full_name", "") or "")
            if self._is_ignored_user_id(user_id):
                continue
            if user_id and full_name:
                self._record_display_name(user_id, full_name)
            normalized.append(Participant(user_id=user_id, name=name))
        return normalized

    @staticmethod
    def _select_primary_incoming_chat(chats: List[InboundChat]) -> InboundChat:
        if not chats:
            raise ValueError("at least one non-self chat is required")
        ranked = sorted(
            enumerate(chats),
            key=lambda item: (
                float(item[1].timestamp or 0.0),
                int(item[0]),
                str(canonical_identity_key(item[1].sender_id, item[1].sender_name)),
            ),
        )
        return ranked[-1][1]

    def _enforce_summary_before_recent(self, *, reason: str) -> None:
        recent = self._memory.recent()
        recent_timestamps = [float(item.timestamp or 0.0) for item in recent if float(item.timestamp or 0.0) > 0.0]
        if not recent_timestamps:
            return
        oldest_recent_ts = min(recent_timestamps)
        summary_meta = self._memory.summary_meta()
        summary_end_ts = float(summary_meta.get("range_end_ts", 0.0) or 0.0)
        if summary_end_ts <= 0.0:
            return
        delta_seconds = summary_end_ts - oldest_recent_ts
        if delta_seconds <= 0.0:
            return
        clamped = self._memory.clamp_summary_range_end(oldest_recent_ts)
        if not clamped:
            return
        if delta_seconds <= SUMMARY_BOUNDARY_WARN_EPSILON_SECONDS:
            return
        logger.warning(
            "Summary boundary violation repaired (%s): range_end_ts=%s > oldest_recent_ts=%s (delta=%.6fs)",
            reason,
            format_pacific_time(summary_end_ts),
            format_pacific_time(oldest_recent_ts),
            delta_seconds,
        )
        self._tracer.log_event(
            "summary_boundary_violation",
            {
                "reason": reason,
                "summary_range_end_before": format_pacific_time(summary_end_ts),
                "summary_range_end_after": format_pacific_time(oldest_recent_ts),
                "oldest_recent_ts": format_pacific_time(oldest_recent_ts),
                "delta_seconds": delta_seconds,
            },
        )

    def _check_payload_contract(self, *, mode: str, payload: Dict[str, Any]) -> None:
        from .payload_contract import normalize_and_validate_payload

        _normalized, warnings = normalize_and_validate_payload(payload)
        if not warnings:
            return
        counts: Dict[str, int] = {}
        for warning in warnings:
            category = str(warning.get("category", "") or "unknown")
            counts[category] = counts.get(category, 0) + 1
        self._tracer.log_event(
            "payload_contract_warning",
            {
                "mode": mode,
                "total": len(warnings),
                "counts": counts,
                "warnings": warnings[:20],
            },
        )

    def _update_last_seen_cache_from_agents(self, agents: List[Dict[str, Any]], timestamp: float) -> None:
        ts = float(timestamp or 0.0)
        if ts <= 0:
            return
        for agent in agents:
            if not isinstance(agent, dict):
                continue
            user_id = str(agent.get("uuid") or agent.get("target_uuid") or "")
            name = str(agent.get("name") or "")
            if not user_id or self._is_ignored_user_id(user_id) or self._is_self_participant(user_id, name):
                continue
            prior = float(self._last_seen_cache.get(user_id, 0.0) or 0.0)
            if ts > prior:
                self._last_seen_cache[user_id] = ts

    def _update_reappearance_signals_from_agents(
        self,
        agents: List[Dict[str, Any]],
        timestamp: float,
        prior_visible_ids: Set[str],
    ) -> None:
        ts = float(timestamp or 0.0)
        if ts <= 0.0:
            return
        current_visible_ids: Set[str] = set()
        for agent in agents:
            if not isinstance(agent, dict):
                continue
            user_id = str(agent.get("uuid") or agent.get("target_uuid") or "")
            name = str(agent.get("name") or "")
            if not user_id or self._is_ignored_user_id(user_id) or self._is_self_participant(user_id, name):
                continue
            current_visible_ids.add(user_id)
            if user_id in prior_visible_ids:
                continue
            prior_seen_ts = float(self._last_seen_cache.get(user_id, 0.0) or 0.0)
            if prior_seen_ts <= 0.0:
                continue
            gap_seconds = ts - prior_seen_ts
            if gap_seconds < REAPPEARANCE_THRESHOLD_SECONDS:
                continue
            self._reappearance_signals[user_id] = {
                "gap_seconds": gap_seconds,
                "seen_at_ts": ts,
                "expires_at_ts": ts + REAPPEARANCE_SIGNAL_TTL_SECONDS,
            }
        self._prune_reappearance_signals(current_visible_ids, ts)

    def _prune_reappearance_signals(self, visible_ids: Set[str], now_ts: float) -> None:
        for user_id, payload in list(self._reappearance_signals.items()):
            expires_at = float(payload.get("expires_at_ts", 0.0) or 0.0)
            if user_id not in visible_ids or (expires_at > 0.0 and now_ts > expires_at):
                self._reappearance_signals.pop(user_id, None)

    def _active_reappearance_signal(self, user_id: str, now_ts: float) -> float | None:
        payload = self._reappearance_signals.get(str(user_id or ""))
        if not isinstance(payload, dict):
            return None
        expires_at = float(payload.get("expires_at_ts", 0.0) or 0.0)
        if expires_at > 0.0 and now_ts > expires_at:
            self._reappearance_signals.pop(str(user_id or ""), None)
            return None
        gap_seconds = float(payload.get("gap_seconds", 0.0) or 0.0)
        return gap_seconds if gap_seconds > 0.0 else None

    def _update_last_seen_from_agents(self, agents: List[Dict[str, Any]], timestamp: float) -> None:
        ts = float(timestamp or 0.0)
        if ts <= 0:
            return
        wrote_updates = False
        for agent in agents:
            if not isinstance(agent, dict):
                continue
            user_id = str(agent.get("uuid") or agent.get("target_uuid") or "")
            name = str(agent.get("name") or "")
            if not user_id or self._is_ignored_user_id(user_id) or self._is_self_participant(user_id, name):
                continue
            try:
                self._knowledge_store.update_last_seen(user_id, name, ts)
                wrote_updates = True
            except Exception:
                continue
        if wrote_updates:
            self._mark_mention_index_dirty()

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
            if not is_ai_marker and self._is_ignored_user_id(sender_id):
                continue
            sender_id_out = self._self_sender_id_for_payload() if is_ai_marker else sender_id
            raw_payload: Dict[str, Any] = {}
            if is_ai_marker:
                sender_name_out = self._persona
                raw_payload["_sender_full_name"] = self._persona
                raw_payload["_sender_display_name"] = self._persona
                raw_payload["_sender_username"] = self._persona
            else:
                stored_sender_name = str(item.sender_name or "").strip()
                full_name = self._clean_name_candidate(self._full_name_for(sender_id, stored_sender_name))
                display_name, username = split_display_and_username(full_name)
                sender_name_out = username or self._clean_name_candidate(stored_sender_name) or display_name or full_name or sender_id
                if full_name:
                    raw_payload["_sender_full_name"] = full_name
                if display_name:
                    raw_payload["_sender_display_name"] = display_name
                if sender_name_out and not looks_like_uuid(sender_name_out):
                    raw_payload["_sender_username"] = sender_name_out
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

    def _filter_overflow_against_summary(self, overflow: List[MemoryItem]) -> List[MemoryItem]:
        if not overflow:
            return []
        summary_end_ts = float(self._memory.summary_meta().get("range_end_ts", 0.0) or 0.0)
        if summary_end_ts <= 0.0:
            return list(overflow)
        filtered = [
            item
            for item in overflow
            if float(item.timestamp or 0.0) <= 0.0 or float(item.timestamp or 0.0) > summary_end_ts
        ]
        skipped = len(overflow) - len(filtered)
        if skipped > 0:
            logger.info(
                "Skipping %d overflow messages already covered by summary through %s.",
                skipped,
                format_pacific_time(summary_end_ts),
            )
        return filtered

    def _chat_payload(self, chat: Any) -> Dict[str, Any]:
        raw_value = getattr(chat, "raw", {})
        raw = raw_value if isinstance(raw_value, dict) else {}
        sender_id = str(getattr(chat, "sender_id", "") or "")
        sender_name = str(getattr(chat, "sender_name", "") or "")
        text = str(getattr(chat, "text", "") or "")
        timestamp = float(getattr(chat, "timestamp", 0.0) or 0.0)
        if sender_id == "ai":
            persona = str(self._persona or "")
            sender_id_out = self._self_sender_id_for_payload()
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
        key = canonical_identity_key(user_id, name)
        if key:
            return key
        return f"name:{self._name_key(name)}"

    def _participants_from_messages(
        self,
        messages: List[InboundChat],
        base_participants: List[Participant],
    ) -> List[Participant]:
        merged: Dict[str, Participant] = {}
        for participant in base_participants:
            if self._is_ignored_user_id(participant.user_id):
                continue
            if self._is_self_participant(participant.user_id, participant.name):
                continue
            canonical_name = self._canonical_name(participant.user_id, participant.name)
            key = self._participant_key(participant.user_id, canonical_name)
            merged[key] = Participant(user_id=participant.user_id, name=canonical_name)
        for message in messages:
            if self._is_ignored_message(message) or self._is_self_message(message):
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
        return self._normalize_participants(list(merged.values()))

    def _facts_context(self, participants: List[Participant], evidence_messages: List[InboundChat]) -> ConversationContext:
        evidence = list(evidence_messages)[-self._facts_evidence_max_messages :]
        now_ts = time.time()
        return ConversationContext(
            persona=self._persona,
            user_id=self._user_id,
            environment=self._environment,
            participants=participants,
            people_facts=self._people_facts_for_participants(participants),
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
        # Queue only newly arrived chat messages. Overflow messages are usually already in
        # this queue from earlier turns; they should not force immediate extraction unless
        # explicitly configured.
        self._enqueue_fact_messages(recent_chats, facts_participants)
        self._maybe_start_facts_worker(overflow_present=bool(overflow_chats))

    def _enqueue_fact_messages(self, messages: List[InboundChat], participants: List[Participant]) -> None:
        if not self._facts_enabled:
            return
        now = time.time()
        appended = False
        for message in messages:
            if self._is_ignored_message(message) or self._is_self_message(message):
                continue
            timestamp = float(message.timestamp or 0.0)
            if timestamp > 0.0 and timestamp <= self._facts_cursor_ts:
                continue
            key = self._fact_message_key(message)
            if key in self._facts_pending_keys:
                continue
            self._facts_pending_messages.append(message)
            self._facts_pending_keys.add(key)
            appended = True
        if appended and self._facts_pending_since_ts <= 0.0:
            self._facts_pending_since_ts = now
        for participant in participants:
            if self._is_ignored_user_id(participant.user_id):
                continue
            if self._is_self_participant(participant.user_id, participant.name):
                continue
            key = self._participant_key(participant.user_id, participant.name)
            self._facts_pending_participants[key] = participant
        max_pending = max(self._facts_evidence_max_messages * 3, self._facts_evidence_max_messages)
        if len(self._facts_pending_messages) > max_pending:
            self._facts_pending_messages = self._facts_pending_messages[-max_pending:]
            self._facts_pending_keys = {self._fact_message_key(item) for item in self._facts_pending_messages}
            # Queue was trimmed to recent items; reset age to avoid immediate re-flush loops.
            self._facts_pending_since_ts = now if self._facts_pending_messages else 0.0

    def _fact_message_key(self, message: InboundChat) -> str:
        timestamp = float(message.timestamp or 0.0)
        sender_key = str(message.sender_id or "").strip()
        if not sender_key:
            sender_key = f"name:{self._name_key(str(message.sender_name or ''))}"
        return f"{sender_key}|{timestamp:.6f}|{(message.text or '').strip()}"

    def _facts_pending_age_seconds(self, now: float | None = None) -> float:
        if self._facts_pending_since_ts <= 0.0:
            return 0.0
        timestamp = time.time() if now is None else float(now)
        return max(0.0, timestamp - self._facts_pending_since_ts)

    def _fact_flush_reason(self, now: float, overflow_present: bool) -> str:
        pending_count = len(self._facts_pending_messages)
        if pending_count <= 0:
            return ""
        if overflow_present and self._facts_flush_on_overflow:
            return "overflow"
        if pending_count >= self._facts_min_pending_messages:
            return "min_pending"
        if self._facts_pending_age_seconds(now) >= self._facts_max_pending_age_seconds:
            return "max_age"
        if (now - self._facts_last_sweep_ts) >= self._facts_interval_seconds:
            return "interval"
        return ""

    def _maybe_start_facts_worker(self, overflow_present: bool = False) -> None:
        if not self._facts_enabled or self._facts_mode != "periodic":
            return
        if self._facts_task is not None and not self._facts_task.done():
            return
        now = time.time()
        flush_reason = self._fact_flush_reason(now, overflow_present)
        if not flush_reason:
            return
        pending_messages = list(self._facts_pending_messages)
        if not pending_messages:
            return
        pending_participants = list(self._facts_pending_participants.values())
        pending_before = len(pending_messages)
        pending_age_seconds = self._facts_pending_age_seconds(now)
        self._facts_pending_messages = []
        self._facts_pending_keys = set()
        self._facts_pending_since_ts = 0.0
        self._facts_last_sweep_ts = now
        self._facts_task = asyncio.create_task(
            self._facts_worker(
                pending_messages=pending_messages,
                pending_participants=pending_participants,
                flush_reason=flush_reason,
                pending_before=pending_before,
                pending_age_seconds=pending_age_seconds,
            )
        )

    async def _facts_worker(
        self,
        pending_messages: List[InboundChat],
        pending_participants: List[Participant],
        flush_reason: str,
        pending_before: int,
        pending_age_seconds: float,
    ) -> None:
        remaining = list(pending_messages)
        chunk_index = 0
        while remaining:
            chunk = remaining[: self._facts_evidence_max_messages]
            remaining = remaining[self._facts_evidence_max_messages :]
            participants = self._participants_from_messages(chunk, pending_participants)
            cursor_before = self._facts_cursor_ts
            sweep = await asyncio.to_thread(self._extract_and_store_facts, chunk, participants)
            max_ts = float(sweep.get("max_ts", 0.0) or 0.0)
            if max_ts > self._facts_cursor_ts:
                self._facts_cursor_ts = max_ts
            payload = {
                "mode": self._facts_mode,
                "flush_reason": flush_reason if chunk_index == 0 else "drain",
                "pending_before": pending_before if chunk_index == 0 else len(remaining) + len(chunk),
                "pending_after": len(remaining),
                "pending_age_seconds": pending_age_seconds if chunk_index == 0 else 0.0,
                "messages": int(sweep.get("messages", len(chunk))),
                "participants": int(sweep.get("participants", len(participants))),
                "facts_extracted": int(sweep.get("facts_extracted", 0)),
                "fact_strings_extracted": int(sweep.get("fact_strings_extracted", 0)),
                "fact_strings_stored": int(sweep.get("fact_strings_stored", 0)),
                "people_updated": int(sweep.get("people_updated", 0)),
                "cursor_before": cursor_before,
                "cursor_after": self._facts_cursor_ts,
                "max_ts": max_ts,
            }
            reasoning_trace = sweep.get("reasoning")
            if isinstance(reasoning_trace, dict) and reasoning_trace:
                payload["reasoning"] = reasoning_trace
            self._tracer.log_event("facts_sweep", payload)
            chunk_index += 1
        self._facts_task = None
        if self._facts_pending_messages:
            self._maybe_start_facts_worker()

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
        self._log_quality_fallbacks()
        reasoning_trace = self._llm.consume_reasoning_trace("facts")
        fact_strings_extracted = sum(len(fact.facts) for fact in facts)
        stored_stats = {"fact_strings_stored": 0, "people_updated": 0}
        if facts:
            stored_stats = self._store_facts(facts, participants)
        timestamps = [float(message.timestamp or 0.0) for message in messages]
        payload = {
            "messages": len(messages),
            "participants": len(participants),
            "facts_extracted": len(facts),
            "fact_strings_extracted": fact_strings_extracted,
            "fact_strings_stored": int(stored_stats.get("fact_strings_stored", 0)),
            "people_updated": int(stored_stats.get("people_updated", 0)),
            "max_ts": max(timestamps) if timestamps else self._facts_cursor_ts,
        }
        if isinstance(reasoning_trace, dict) and reasoning_trace:
            payload["reasoning"] = reasoning_trace
        return payload

    def _store_facts(self, facts: List[ExtractedFact], participants: List[Participant]) -> Dict[str, int]:
        alias_to_id: Dict[str, str] = {}
        similarity_threshold = 0.82
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
            existing_list = list(profile.facts) if profile else []
            existing_keys = {self._normalize_fact_key(item) for item in existing_list if self._normalize_fact_key(item)}
            existing_tokens = [self._fact_tokens(item) for item in existing_list]
            existing_tokens = [tokens for tokens in existing_tokens if tokens]
            deduped_new_facts = self._dedupe_preserve_order(fact.facts)
            missing_facts: List[str] = []
            for item in deduped_new_facts:
                key = self._normalize_fact_key(item)
                if not key or key in existing_keys:
                    continue
                tokens = self._fact_tokens(item)
                if self._is_near_duplicate(tokens, existing_tokens, similarity_threshold):
                    continue
                missing_facts.append(item)
                existing_keys.add(key)
                if tokens:
                    existing_tokens.append(tokens)
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
            self._mark_mention_index_dirty()
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
        self._participant_hints = self._normalize_participants(list(merged.values()))[: self._max_environment_participants]
        self._runtime_state.participant_hints = self._participant_hints

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
            "autonomy_decision": self._autonomy_decision,
            "autonomy_delay_hint_seconds": self._autonomy_delay_hint_seconds,
        }

    def consume_autonomy_delay_hint_seconds(self) -> float | None:
        hint = self._autonomy_delay_hint_seconds
        self._autonomy_delay_hint_seconds = None
        return hint

    def _effective_posture_is_sitting(self, now_ts: float | None = None) -> bool | None:
        if not self._runtime_state.posture_known:
            return None
        current = float(now_ts or time.time())
        last_update = float(self._runtime_state.last_posture_update_ts or 0.0)
        if last_update <= 0.0:
            return None
        if self._posture_stale_seconds > 0.0 and (current - last_update) > self._posture_stale_seconds:
            return None
        return bool(self._runtime_state.posture_is_sitting)

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
            "last_message_received_at": format_pacific_time(self._last_inbound_ts),
            "last_ai_response_at": format_pacific_time(self._last_response_ts),
            "seconds_since_activity": self.seconds_since_activity(current),
            "autonomy_decision": self._autonomy_decision,
            "autonomy_delay_hint_seconds": self._autonomy_delay_hint_seconds,
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
            CommandType.FACE_TARGET: "facing",
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
        effective_posture = self._effective_posture_is_sitting()
        participant_keys = {
            self._normalize_name_for_match(participant.user_id or participant.name)
            for participant in participants
            if participant.user_id or participant.name
        }
        participant_keys.discard("")
        wait_tokens = ("wait", "waiting", "waited")
        filtered: List[Action] = []
        for action in actions:
            if action.command_type == CommandType.SIT and effective_posture is True:
                continue
            if action.command_type == CommandType.STAND and effective_posture is False:
                continue
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
        if not self._llm_chat_enabled:
            # Even when chat output is disabled, we still want episodic summaries to flush based on
            # inactivity/max/forced-interval triggers. Otherwise long "time gaps" can accumulate
            # in Neo4j experience timelines until chat output is re-enabled.
            self._schedule_episode_check()
            return []
        activity = self.activity_snapshot(recent_activity_window_seconds)
        if activity["seconds_since_activity"] < recent_activity_window_seconds:
            return []
        participants = self._context_builder_component.participants_for_autonomy()
        query_text = self._autonomy_query_text()
        context = await self._context_builder_component.build_context(
            None,
            participants,
            query_text=query_text,
            agent_state=self._agent_state(),
        )
        prompt_payload = self._bundle_prompt_payload(None, context, None, None, activity=activity, mode="autonomous")
        self._tracer.log_event("llm_prompt_autonomy", prompt_payload)
        self._check_payload_contract(mode="autonomous", payload=prompt_payload)
        bundle = await self._llm.generate_autonomous_bundle(context, activity)
        self._log_quality_fallbacks()
        bundle.actions = self._filter_autonomous_actions(bundle.actions, participants)
        decision = self._normalize_autonomy_decision(bundle.autonomy_decision, bool(bundle.actions))
        delay_hint = self._sanitize_autonomy_delay_hint(bundle.next_delay_seconds)
        bundle.autonomy_decision = decision
        bundle.next_delay_seconds = delay_hint
        self._autonomy_decision = decision
        self._autonomy_delay_hint_seconds = delay_hint
        self._tracer.log_event("llm_response_autonomy", self._bundle_response_payload(bundle))
        self._log_reasoning_trace("autonomy", "llm_reasoning_autonomy")
        chat_texts = self._chat_texts_from_actions(bundle.actions)
        for text in chat_texts:
            self._rolling_buffer.add_ai_message(text, self._persona)
            self._memory.add_ai_message(text, self._persona)
        if bundle.actions or chat_texts:
            self._last_response_ts = time.time()
        if self._facts_enabled and self._facts_mode == "per_message":
            self._fact_manager_component.schedule_store_facts(bundle.facts, participants)
        self._update_participant_hints(bundle.participant_hints)
        self._update_mood_and_status(bundle, source="autonomy")
        self._schedule_episode_check()
        return bundle.actions

    @staticmethod
    def _sanitize_autonomy_delay_hint(value: Any) -> float | None:
        try:
            delay = float(value)
        except (TypeError, ValueError):
            return None
        if delay <= 0.0:
            return None
        return delay

    @staticmethod
    def _normalize_autonomy_decision(value: Any, has_actions: bool) -> str:
        if isinstance(value, str):
            decision = value.strip().lower()
            if decision in {"act", "wait", "sleep"}:
                if has_actions and decision != "act":
                    return "act"
                if decision == "act" and not has_actions:
                    return "wait"
                return decision
        return "act" if has_actions else "wait"

    @staticmethod
    def _normalize_autonomy_decision_value(value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        decision = value.strip().lower()
        if decision in {"act", "wait", "sleep"}:
            return decision
        return None

    def _apply_autonomy_scheduler_override_from_incoming(
        self,
        decision_value: Any,
        delay_value: Any,
    ) -> tuple[str | None, float | None]:
        decision = self._normalize_autonomy_decision_value(decision_value)
        delay_hint = self._sanitize_autonomy_delay_hint(delay_value)
        if decision is None and delay_hint is None:
            return None, None
        if decision is not None:
            self._autonomy_decision = decision
            # Decision-only overrides should clear stale delay hints.
            if delay_hint is None:
                self._autonomy_delay_hint_seconds = None
        if delay_hint is not None:
            self._autonomy_delay_hint_seconds = delay_hint
        return decision, delay_hint

    def _participants_for_autonomy(self) -> List[Participant]:
        merged: Dict[str, Participant] = {}

        def add_participant(user_id: str, name: str) -> None:
            if self._is_ignored_user_id(user_id):
                return
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
        return self._normalize_participants(list(merged.values()))

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
        chats: List[InboundChat] = []
        for item in items:
            sender_id = str(getattr(item, "sender_id", "") or "")
            if sender_id != "ai" and is_ignored_user_id(sender_id):
                continue
            chats.append(
                InboundChat(
                    text=str(getattr(item, "text", "") or ""),
                    sender_id=sender_id,
                    sender_name=str(getattr(item, "sender_name", "") or ""),
                    timestamp=float(getattr(item, "timestamp", time.time()) or time.time()),
                    raw={},
                )
            )
        if not chats:
            return ""
        try:
            summary = await self._llm.summarize_episode(chats)
            self._log_quality_fallbacks()
        except Exception:
            summary = ""
        if summary:
            return summary.strip()
        return self._episode_compressor.compress("", items)

    @staticmethod
    def _episode_metadata(items: List[Any], reason: str) -> Dict[str, Any]:
        filtered_items = [
            item
            for item in items
            if str(getattr(item, "sender_id", "") or "") == "ai"
            or not is_ignored_user_id(str(getattr(item, "sender_id", "") or ""))
        ]
        timestamps = [float(getattr(item, "timestamp", 0.0) or 0.0) for item in filtered_items]
        sender_names: List[str] = []
        seen: Set[str] = set()
        for item in filtered_items:
            name = str(getattr(item, "sender_name", "") or "")
            if not name or name in seen:
                continue
            seen.add(name)
            sender_names.append(name)
        
        start_ts = min(timestamps) if timestamps else 0.0
        return {
            "source": "episode_summary",
            "reason": reason,
            "message_count": len(items),
            "timestamp": start_ts,
            "timestamp_start": format_pacific_time(start_ts) if start_ts > 0 else "0",
            "timestamp_end": format_pacific_time(max(timestamps)) if timestamps else "0",
            "sender_names": sender_names,
        }

    def _latest_timestamp(self) -> float:
        timestamps: List[float] = []
        for item in self._rolling_buffer.items():
            sender_id = str(getattr(item, "sender_id", "") or "")
            if sender_id != "ai" and self._is_ignored_user_id(sender_id):
                continue
            timestamps.append(float(getattr(item, "timestamp", 0.0) or 0.0))
        for item in self._memory.recent():
            sender_id = str(getattr(item, "sender_id", "") or "")
            if sender_id != "ai" and self._is_ignored_user_id(sender_id):
                continue
            timestamps.append(float(getattr(item, "timestamp", 0.0) or 0.0))
        return max(timestamps) if timestamps else 0.0

    @staticmethod
    def _entity_prompt_payload(entry: Any) -> Any:
        if not isinstance(entry, dict):
            return entry
        cleaned = dict(entry)
        uuid_value = cleaned.get("uuid") or cleaned.get("target_uuid")
        cleaned.pop("target_uuid", None)
        if uuid_value:
            cleaned["uuid"] = uuid_value
        return cleaned

    def _environment_payload(self, object_limit: int = 25) -> Dict[str, Any]:
        return {
            "location": self._environment.location,
            "avatar_position": self._environment.avatar_position,
            "is_sitting": self._environment.is_sitting,
            "agents": [
                self._entity_prompt_payload(agent)
                for agent in self._environment.agents[: self._max_environment_participants]
            ],
            "objects": [
                self._entity_prompt_payload(obj)
                for obj in self._environment.objects[:object_limit]
            ],
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
        now_ts = time.time()
        participant_inputs = list(context.participants)
        evidence_messages = list(context.recent_messages)
        if overflow:
            evidence_messages.extend(overflow)
        if chat is not None:
            evidence_messages.append(chat)
        participants = self._participants_from_messages(evidence_messages, participant_inputs)
        participants_payload = [self._participant_payload(p) for p in participants]

        recent_timestamps = [float(m.timestamp or 0.0) for m in context.recent_messages]
        recent_start_ts = min(recent_timestamps) if recent_timestamps else 0.0
        summary_meta = self._summary_meta_for_payload(
            summary_meta=context.summary_meta,
            recent_start_ts=recent_start_ts,
            now_ts=now_ts,
        )
        recent_time_range = {
            "start": format_pacific_time(recent_start_ts) if recent_timestamps else "0",
            "end": format_pacific_time(max(recent_timestamps)) if recent_timestamps else "0",
        }
        ordered_incoming_batch = incoming_batch or []
        incoming_payload = None
        if ordered_incoming_batch:
            latest_batch_entry = ordered_incoming_batch[-1]
            incoming_payload = self._incoming_payload_from_batch_entry(latest_batch_entry)
        elif chat is not None:
            incoming_payload = self._chat_payload(chat)
        stable_people_facts, people_recency = PromptManager.split_people_facts_prompt_sections(
            context.people_facts,
            now_ts=now_ts,
        )
        stable_agent_state, agent_timing = PromptManager.split_agent_state_prompt_sections(context.agent_state)

        payload: Dict[str, Any] = {
            "mode": mode,
            "now_timestamp": format_pacific_time(now_ts),
            "persona": context.persona,
            "user_id": context.user_id,
            "participants": participants_payload,
            "environment": self._environment_payload(),
            "people_facts": stable_people_facts,
            "previously": context.summary,
            "summary_meta": summary_meta,
            "agent_state": stable_agent_state,
            "persona_instructions": context.persona_instructions,
            "recent_time_range": recent_time_range,
            "recent_messages": [
                self._chat_payload(m) for m in context.recent_messages
            ],
            "related_experiences": context.related_experiences,
            "incoming_batch": ordered_incoming_batch,
        }
        if incoming_payload is not None:
            payload["incoming"] = incoming_payload
            incoming_id = str(incoming_payload.get("sender_id", "") or "")
            payload["incoming_sender_id"] = incoming_id
            payload["incoming_sender_known"] = bool(incoming_id and incoming_id in context.people_facts)
        if people_recency:
            payload["people_recency"] = people_recency
        if agent_timing:
            payload["agent_timing"] = agent_timing
        if overflow:
            payload["overflow_messages"] = [
                self._chat_payload(m) for m in overflow
            ]
        return payload

    @staticmethod
    def _summary_meta_for_payload(
        *,
        summary_meta: Dict[str, Any],
        recent_start_ts: float,
        now_ts: float,
    ) -> Dict[str, Any]:
        meta = dict(summary_meta or {})
        range_start = float(meta.get("range_start_ts", 0.0) or 0.0)
        range_end = float(meta.get("range_end_ts", 0.0) or 0.0)
        if range_start > 0.0 and range_end > 0.0 and range_start > range_end:
            range_start, range_end = range_end, range_start
        if recent_start_ts > 0.0 and range_end > recent_start_ts:
            range_end = recent_start_ts
            if range_start > range_end and range_start > 0.0:
                range_start = range_end
        meta["range_start_ts"] = float(range_start or 0.0)
        meta["range_end_ts"] = float(range_end or 0.0)
        last_updated_ts = float(meta.get("last_updated_ts", 0.0) or 0.0)
        if last_updated_ts > 0.0:
            meta["age_seconds"] = max(0.0, now_ts - last_updated_ts)
        elif "age_seconds" in meta:
            meta.pop("age_seconds", None)
        if range_end > 0.0:
            meta["range_age_seconds"] = max(0.0, now_ts - range_end)
        elif "range_age_seconds" in meta:
            meta.pop("range_age_seconds", None)
        return meta

    @staticmethod
    def _incoming_payload_from_batch_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
        sender_id = str(entry.get("sender_id", "") or "")
        sender_username = str(entry.get("sender_username", "") or "") or str(entry.get("sender_name", "") or "")
        sender_display_name = str(entry.get("sender_display_name", "") or "") or sender_username
        sender_full_name = str(entry.get("sender_full_name", "") or "") or sender_display_name or sender_username
        sender_value = sender_username or sender_display_name or sender_id
        latest_text = str(entry.get("latest_text", "") or "")
        if not latest_text:
            texts = entry.get("texts", [])
            if isinstance(texts, list) and texts:
                latest_text = str(texts[-1] or "")
        timestamp = float(entry.get("last_timestamp", 0.0) or 0.0)
        return {
            "sender": sender_value,
            "sender_id": sender_id,
            "sender_username": sender_username,
            "sender_display_name": sender_display_name,
            "sender_full_name": sender_full_name,
            "text": latest_text,
            "timestamp": format_pacific_time(timestamp) if timestamp > 0.0 else "0",
        }

    def _persona_instructions(self) -> str:
        if not self._persona_profiles:
            return ""
        key = str(self._persona or "").casefold()
        return str(self._persona_profiles.get(key, "") or "")

    def _log_reasoning_trace(self, label: str, event_name: str) -> None:
        trace = self._llm.consume_reasoning_trace(label)
        if isinstance(trace, dict) and trace:
            self._tracer.log_event(event_name, trace)

    def _log_quality_fallbacks(self) -> None:
        events = self._llm.consume_quality_fallback_events()
        if not isinstance(events, list):
            return
        for payload in events:
            if isinstance(payload, dict) and payload:
                self._tracer.log_event("llm_quality_fallback", payload)

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
            "autonomy_decision": bundle.autonomy_decision,
            "next_delay_seconds": bundle.next_delay_seconds,
        }

    @staticmethod
    def _state_update_payload(update: LLMStateUpdate) -> Dict[str, Any]:
        return {
            "facts": [
                {"user_id": fact.user_id, "name": fact.name, "facts": fact.facts}
                for fact in update.facts
            ],
            "participant_hints": [
                {"user_id": hint.user_id, "name": hint.name}
                for hint in update.participant_hints
            ],
            "summary_update": update.summary_update,
            "mood": update.mood,
            "status": update.status,
            "autonomy_decision": update.autonomy_decision,
            "next_delay_seconds": update.next_delay_seconds,
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

    def _mark_mention_index_dirty(self) -> None:
        self._mention_index_dirty = True

    def _remember_runtime_mention_name(self, user_id: str, full_name: str) -> None:
        normalized_user_id = str(user_id or "").strip()
        if not normalized_user_id:
            return
        variants = self._mention_name_variants(full_name)
        if not variants:
            return
        runtime_names = self._mention_runtime_names_by_id.setdefault(normalized_user_id, set())
        runtime_names.update(variants)
        if self._mention_index_dirty:
            return
        indexed_names = self._mention_index_names_by_id.setdefault(normalized_user_id, set())
        for phrase in variants:
            phrase_matches = self._mention_index_by_phrase.setdefault(phrase, set())
            phrase_matches.add(normalized_user_id)
            indexed_names.add(phrase)
            self._mention_index_max_phrase_words = max(self._mention_index_max_phrase_words, len(phrase.split()))

    @staticmethod
    def _normalize_mention_phrase(text: str) -> str:
        normalized = unicodedata.normalize("NFKC", str(text or "")).casefold()
        cleaned = "".join(ch if ch.isalnum() else " " for ch in normalized)
        return " ".join(cleaned.split())

    @classmethod
    def _mention_name_variants(cls, name: str) -> List[str]:
        raw = str(name or "").strip()
        if not raw:
            return []
        display_name, username = split_display_and_username(raw)
        candidates: List[str] = [raw]
        if display_name:
            candidates.append(display_name)
        if username:
            candidates.append(username)
        for source in (display_name, username):
            normalized_source = cls._normalize_mention_phrase(source)
            if not normalized_source:
                continue
            source_tokens = normalized_source.split()
            if source_tokens:
                if len(source_tokens[0]) >= 3:
                    candidates.append(source_tokens[0])
                for token in source_tokens[1:]:
                    if len(token) >= 4:
                        candidates.append(token)
        deduped: List[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            normalized_candidate = cls._normalize_mention_phrase(candidate)
            if len(normalized_candidate) < 3:
                continue
            if not any(ch.isalpha() for ch in normalized_candidate):
                continue
            if normalized_candidate in seen:
                continue
            seen.add(normalized_candidate)
            deduped.append(normalized_candidate)
        return deduped

    def _refresh_mention_index_if_needed(self, now_ts: float | None = None) -> None:
        now_value = float(now_ts or time.time())
        if (
            not self._mention_index_dirty
            and (now_value - float(self._mention_index_last_refresh_ts or 0.0)) < MENTION_INDEX_REFRESH_INTERVAL_SECONDS
        ):
            return
        phrase_index: Dict[str, Set[str]] = {}
        names_by_id: Dict[str, Set[str]] = {}

        def add_phrase(user_id: str, phrase: str) -> None:
            if not user_id or not phrase:
                return
            if self._is_self_participant(user_id, ""):
                return
            users = phrase_index.setdefault(phrase, set())
            users.add(user_id)
            names_by_id.setdefault(user_id, set()).add(phrase)

        try:
            entries = self._knowledge_store.fetch_people_name_index()
        except Exception:
            entries = []
        for entry in entries:
            user_id = str(getattr(entry, "user_id", "") or "")
            names = getattr(entry, "names", [])
            if isinstance(entry, dict):
                user_id = str(entry.get("user_id", user_id) or "")
                names = entry.get("names", names)
            if not user_id:
                continue
            if not isinstance(names, list):
                continue
            for raw_name in names:
                for variant in self._mention_name_variants(str(raw_name or "")):
                    add_phrase(user_id, variant)
        for user_id, variants in self._mention_runtime_names_by_id.items():
            for variant in variants:
                add_phrase(user_id, variant)

        self._mention_index_by_phrase = phrase_index
        self._mention_index_names_by_id = names_by_id
        self._mention_index_max_phrase_words = max((len(phrase.split()) for phrase in phrase_index), default=1)
        self._mention_index_last_refresh_ts = now_value
        self._mention_index_dirty = False

    def _resolve_text_mentions(self, text: str) -> List[tuple[str, str]]:
        normalized_text = self._normalize_mention_phrase(text)
        if not normalized_text:
            return []
        self._refresh_mention_index_if_needed()
        if not self._mention_index_by_phrase:
            return []
        tokens = normalized_text.split()
        if not tokens:
            return []
        max_span = min(
            max(1, int(self._mention_index_max_phrase_words or 1)),
            MENTION_INDEX_MAX_SCAN_WORDS,
            len(tokens),
        )
        matches: List[tuple[str, str]] = []
        seen_user_ids: Set[str] = set()
        idx = 0
        while idx < len(tokens):
            matched = False
            span_limit = min(max_span, len(tokens) - idx)
            for span_len in range(span_limit, 0, -1):
                phrase = " ".join(tokens[idx : idx + span_len])
                user_ids = self._mention_index_by_phrase.get(phrase, set())
                if len(user_ids) != 1:
                    continue
                user_id = next(iter(user_ids))
                if user_id in seen_user_ids:
                    matched = True
                    idx += span_len
                    break
                seen_user_ids.add(user_id)
                matches.append((user_id, phrase))
                matched = True
                idx += span_len
                break
            if not matched:
                idx += 1
        return matches

    @staticmethod
    def _looks_like_uuid(value: str) -> bool:
        candidate = str(value or "").strip()
        if not candidate:
            return False
        parts = candidate.split("-")
        if len(parts) != 5:
            return False
        expected_lengths = (8, 4, 4, 4, 12)
        for part, expected in zip(parts, expected_lengths):
            if len(part) != expected:
                return False
            if any(ch not in "0123456789abcdefABCDEF" for ch in part):
                return False
        return True

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

    def _collapse_near_duplicate_experiences(
        self,
        experiences: List[ExperienceRecord],
    ) -> List[ExperienceRecord]:
        if not self._near_duplicate_collapse_enabled:
            return experiences
        if len(experiences) < 2:
            return experiences
        threshold = self._near_duplicate_similarity
        if threshold <= 0.0:
            return experiences

        clusters: List[Dict[str, Any]] = []
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

        collapsed: List[ExperienceRecord] = []
        for cluster in clusters:
            items = cluster.get("items", [])
            if not items:
                continue
            representative = items[0]
            if len(items) == 1:
                collapsed.append(representative)
                continue
            collapsed.append(self._with_near_duplicate_metadata(representative, items))
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
        token_score = MessagePipeline._token_jaccard(left_tokens, right_tokens)
        left_trigrams = MessagePipeline._char_trigrams(left)
        right_trigrams = MessagePipeline._char_trigrams(right)
        trigram_score = MessagePipeline._token_jaccard(left_trigrams, right_trigrams)
        return max(sequence, token_score, trigram_score)

    @staticmethod
    def _char_trigrams(text: str) -> Set[str]:
        if len(text) < 3:
            return {text} if text else set()
        return {text[idx : idx + 3] for idx in range(len(text) - 2)}

    def _with_near_duplicate_metadata(
        self,
        representative: ExperienceRecord,
        items: List[ExperienceRecord],
    ) -> ExperienceRecord:
        metadata = representative.metadata if isinstance(representative.metadata, dict) else {}
        merged_metadata: Dict[str, Any] = dict(metadata)
        merged_metadata["near_duplicate_count"] = len(items)

        timestamps = [self._experience_timestamp(item) for item in items]
        valid_timestamps = [ts for ts in timestamps if ts > 0.0]
        if valid_timestamps:
            first_seen_ts = min(valid_timestamps)
            last_seen_ts = max(valid_timestamps)
            merged_metadata["near_duplicate_first_seen"] = self._iso_date(first_seen_ts)
            merged_metadata["near_duplicate_last_seen"] = self._iso_date(last_seen_ts)

        return ExperienceRecord(text=representative.text, metadata=merged_metadata)

    @staticmethod
    def _iso_date(timestamp: float) -> str:
        return datetime.fromtimestamp(timestamp, timezone.utc).date().isoformat()

    def _gate_semantic_experiences(
        self,
        experiences: List[ExperienceRecord],
        limit: int | None = None,
    ) -> List[ExperienceRecord]:
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
        gate_limit = self._experience_top_k if limit is None else max(1, int(limit))
        for score, item in scored:
            if score < self._experience_score_min:
                continue
            if (top_score - score) > self._experience_score_delta:
                continue
            gated.append(item)
            if len(gated) >= gate_limit:
                break
        return gated

    def _experiences_for_persona(self, experiences: List[ExperienceRecord]) -> List[ExperienceRecord]:
        if not self._persona:
            return experiences
        filtered: List[ExperienceRecord] = []
        for item in experiences:
            metadata = item.metadata if isinstance(item.metadata, dict) else {}
            persona_id = str(metadata.get("persona_id", "") or "")
            if persona_id and persona_id != self._persona:
                continue
            filtered.append(item)
        return filtered

    def _build_routine_summaries(self, candidates: List[ExperienceRecord]) -> List[ExperienceRecord]:
        if not candidates:
            return []
        clusters: List[Dict[str, Any]] = []
        for item in candidates:
            tokens = self._routine_tokens(item.text)
            if len(tokens) < 4:
                continue
            best_idx = -1
            best_score = 0.0
            for idx, cluster in enumerate(clusters):
                score = self._token_jaccard(tokens, cluster["anchor_tokens"])
                if score > best_score:
                    best_idx = idx
                    best_score = score
            if best_idx >= 0 and best_score >= 0.60:
                cluster = clusters[best_idx]
                cluster["items"].append(item)
                overlap = cluster["anchor_tokens"] & tokens
                if len(overlap) >= 3:
                    cluster["anchor_tokens"] = overlap
            else:
                clusters.append({"anchor_tokens": set(tokens), "items": [item]})

        summaries: List[tuple[int, float, ExperienceRecord]] = []
        for cluster in clusters:
            items = cluster.get("items", [])
            if len(items) < self._routine_summary_min_count:
                continue
            latest = max(items, key=self._experience_timestamp)
            count = len(items)
            metadata_raw = latest.metadata if isinstance(latest.metadata, dict) else {}
            last_seen = self._format_last_seen(metadata_raw)
            phrase = self._routine_phrase(latest.text)
            prefix = "Often" if count >= 3 else "Repeatedly"
            text = f"{prefix} {phrase} ({count} similar experiences, last seen {last_seen})."
            metadata: Dict[str, Any] = {
                "source": "routine_summary",
                "routine_count": count,
                "last_seen": last_seen,
            }
            rep_id = str(metadata_raw.get("experience_id", "") or "")
            if rep_id:
                metadata["representative_experience_id"] = rep_id
            summaries.append(
                (
                    count,
                    self._experience_timestamp(latest),
                    ExperienceRecord(text=text, metadata=metadata),
                )
            )

        summaries.sort(key=lambda row: (row[0], row[1]), reverse=True)
        return [row[2] for row in summaries[: self._routine_summary_limit]]

    @staticmethod
    def _routine_tokens(text: str) -> Set[str]:
        if not text:
            return set()
        cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
        tokens: List[str] = []
        seen: Set[str] = set()
        for token in cleaned.split():
            if len(token) < 4:
                continue
            if token in seen:
                continue
            seen.add(token)
            tokens.append(token)
            if len(tokens) >= 24:
                break
        return set(tokens)

    @staticmethod
    def _token_jaccard(left: Set[str], right: Set[str]) -> float:
        if not left or not right:
            return 0.0
        union = left | right
        if not union:
            return 0.0
        return len(left & right) / len(union)

    @staticmethod
    def _routine_phrase(text: str) -> str:
        clean = " ".join(str(text or "").split())
        if not clean:
            return "similar events occurred"
        sentence = clean
        for punct in (".", "!", "?"):
            if punct in sentence:
                sentence = sentence.split(punct, 1)[0].strip()
                break
        words = sentence.split()
        if len(words) > 18:
            sentence = " ".join(words[:18]).rstrip(",;:")
        return sentence

    @staticmethod
    def _experience_timestamp(item: ExperienceRecord) -> float:
        metadata = item.metadata if isinstance(item.metadata, dict) else {}
        raw_ts = metadata.get("timestamp")
        try:
            ts = float(raw_ts)
            if ts > 0.0:
                return ts
        except (TypeError, ValueError):
            pass
        for key in ("timestamp_end", "timestamp_start"):
            value = str(metadata.get(key, "") or "").strip()
            if not value:
                continue
            try:
                dt = datetime.fromisoformat(value)
                return dt.replace(tzinfo=timezone.utc).timestamp()
            except ValueError:
                continue
        return 0.0

    @staticmethod
    def _format_last_seen(metadata: Dict[str, Any]) -> str:
        for key in ("timestamp_end", "timestamp_start"):
            value = str(metadata.get(key, "") or "").strip()
            if value:
                return value.split(" ", 1)[0]
        raw_ts = metadata.get("timestamp")
        try:
            ts = float(raw_ts)
            if ts > 0.0:
                return datetime.fromtimestamp(ts, timezone.utc).date().isoformat()
        except (TypeError, ValueError):
            pass
        return "unknown"

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
