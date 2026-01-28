from __future__ import annotations

import asyncio
import contextlib
from typing import Any, Dict, List

from .config import EpisodeConfig, FactsConfig
from .llm import LLMClient, RuleBasedLLMClient
from .memory import ConversationMemory, ExperienceStore, RollingBuffer, SimpleMemoryCompressor
from .models import Action
from .name_utils import extract_username
from .neo4j_store import InMemoryKnowledgeStore, KnowledgeStore
from .observability import NoOpTracer, Tracer
from .pipeline import ExperienceIndexProtocol, MessagePipeline
from .session_state import SessionStateStore


class SessionController:
    def __init__(
        self,
        persona: str,
        user_id: str,
        knowledge_store: KnowledgeStore | None = None,
        llm: LLMClient | None = None,
        tracer: Tracer | None = None,
        experience_vector_index: ExperienceIndexProtocol | None = None,
        max_recent_messages: int = 20,
        max_rolling_buffer: int = 30,
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
        self._knowledge_store = knowledge_store or InMemoryKnowledgeStore()
        self._llm = llm or RuleBasedLLMClient()
        self._tracer = tracer or NoOpTracer()
        self._rolling_buffer = RollingBuffer(max_items=max_rolling_buffer)
        self._memory = ConversationMemory(
            SimpleMemoryCompressor(),
            max_recent=max_recent_messages,
            defer_compression=True,
        )
        self._experience_store = ExperienceStore()
        self._episode_config = episode_config or EpisodeConfig()
        self._facts_config = facts_config or FactsConfig()
        self._pipeline = MessagePipeline(
            persona=self._persona,
            user_id=self._user_id,
            llm=self._llm,
            knowledge_store=self._knowledge_store,
            memory=self._memory,
            rolling_buffer=self._rolling_buffer,
            experience_store=self._experience_store,
            tracer=self._tracer,
            experience_vector_index=experience_vector_index,
            summary_strategy=summary_strategy,
            experience_top_k=experience_top_k,
            experience_score_min=experience_score_min,
            experience_score_delta=experience_score_delta,
            max_environment_participants=max_environment_participants,
            facts_config=self._facts_config,
            episode_config=self._episode_config,
            persona_profiles=persona_profiles or {},
        )
        self._state_store: SessionStateStore | None = None
        self._state_task: asyncio.Task | None = None
        self._reset_state_store()

    async def process_chat(self, data: Dict[str, Any]) -> List[Action]:
        self._maybe_update_persona_from_payloads([data])
        actions = await self._pipeline.process_chat(data)
        self._schedule_state_save()
        return actions

    async def process_chat_batch(self, batch: List[Dict[str, Any]]) -> List[Action]:
        self._maybe_update_persona_from_payloads(batch)
        actions = await self._pipeline.process_chat_batch(batch)
        self._schedule_state_save()
        return actions

    def update_environment(self, data: Dict[str, Any]) -> None:
        self._pipeline.update_environment(data)

    def set_persona(self, persona: str) -> None:
        self._persona = persona
        self._pipeline.set_persona(persona)
        self._reset_state_store()

    def set_user_id(self, user_id: str) -> None:
        self._user_id = user_id
        self._pipeline.set_user_id(user_id)
        self._reset_state_store()

    def knowledge_store(self) -> KnowledgeStore:
        return self._knowledge_store

    def memory_summary(self) -> str:
        return self._memory.summary()

    async def generate_autonomous_actions(self, recent_activity_window_seconds: float) -> List[Action]:
        return await self._pipeline.generate_autonomous_actions(recent_activity_window_seconds)

    def activity_snapshot(self, recent_activity_window_seconds: float) -> Dict[str, Any]:
        return self._pipeline.activity_snapshot(recent_activity_window_seconds)

    def persona(self) -> str:
        return self._persona

    def user_id(self) -> str:
        return self._user_id

    async def flush_state(self) -> None:
        if self._state_task and not self._state_task.done():
            with contextlib.suppress(asyncio.CancelledError):
                await self._state_task
        self._save_state_now()

    def _reset_state_store(self) -> None:
        self._schedule_state_save()
        if not self._episode_config.persist_state:
            self._state_store = None
            return
        self._state_store = SessionStateStore(
            state_dir=self._episode_config.state_dir,
            persona=self._persona,
            user_id=self._user_id,
        )
        state = self._state_store.load()
        if state:
            self._pipeline.restore_state(state)

    def _schedule_state_save(self) -> None:
        store = self._state_store
        if store is None:
            return
        if self._state_task and not self._state_task.done():
            return
        snapshot = self._pipeline.snapshot_state()
        self._state_task = asyncio.create_task(asyncio.to_thread(store.save, snapshot))

    def _save_state_now(self) -> None:
        store = self._state_store
        if store is None:
            return
        snapshot = self._pipeline.snapshot_state()
        store.save(snapshot)

    def _maybe_update_persona_from_payloads(self, payloads: List[Dict[str, Any]]) -> None:
        for payload in payloads:
            if not isinstance(payload, dict):
                continue
            logged_in_agent = payload.get("logged_in_agent")
            if isinstance(logged_in_agent, str) and logged_in_agent.strip():
                name = extract_username(logged_in_agent)
            else:
                candidate = payload.get("persona") or payload.get("ai_name")
                if not isinstance(candidate, str):
                    continue
                name = candidate.strip()
            if not name or name == self._persona:
                continue
            self.set_persona(name)
            break
