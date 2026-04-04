from __future__ import annotations

import asyncio
import time
from typing import Any

from .config import FactsConfig
from .llm import ExtractedFact, LLMClient
from .memory import MemoryItem
from .models import InboundChat, Participant
from .neo4j_store import KnowledgeStore
from .observability import Tracer
from .pipeline_state import PipelineRuntimeState


class FactManager:
    def __init__(
        self,
        state: PipelineRuntimeState,
        llm: LLMClient,
        knowledge_store: KnowledgeStore,
        tracer: Tracer,
        facts_config: FactsConfig,
        context_builder: Any,
        autonomy_manager: Any,
        owner: Any | None = None,
    ) -> None:
        self._state = state
        self._llm = llm
        self._knowledge_store = knowledge_store
        self._tracer = tracer
        self._context_builder = context_builder
        self._autonomy_manager = autonomy_manager
        self._owner = owner

        self._facts_enabled = bool(facts_config.enabled)
        mode = str(facts_config.mode or "periodic").strip().lower()
        self._facts_mode = mode if mode in {"periodic", "per_message"} else "periodic"
        self._facts_interval_seconds = max(1.0, float(facts_config.interval_seconds))
        self._facts_evidence_max_messages = max(4, int(facts_config.evidence_max_messages))
        self._facts_min_pending_messages = max(1, int(facts_config.min_pending_messages))
        self._facts_max_pending_age_seconds = max(1.0, float(facts_config.max_pending_age_seconds))
        self._facts_flush_on_overflow = bool(facts_config.flush_on_overflow)
        self._facts_last_sweep_ts = time.time()
        self._facts_cursor_ts = 0.0
        self._facts_pending_messages: list[InboundChat] = []
        self._facts_pending_keys: set[str] = set()
        self._facts_pending_since_ts = 0.0
        self._facts_pending_participants: dict[str, Participant] = {}
        self._facts_task: asyncio.Task | None = None

    def snapshot_state(self) -> dict[str, Any]:
        if self._owner is not None and hasattr(self._owner, "_snapshot_facts_state"):
            return self._owner._snapshot_facts_state()
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

    def restore_state(self, state: dict[str, Any]) -> None:
        if self._owner is not None and hasattr(self._owner, "_restore_facts_state"):
            self._owner._restore_facts_state(state)
            return
        if not isinstance(state, dict):
            return
        self._facts_cursor_ts = float(state.get("cursor_ts", self._facts_cursor_ts) or self._facts_cursor_ts)
        self._facts_last_sweep_ts = float(state.get("last_sweep_ts", self._facts_last_sweep_ts) or self._facts_last_sweep_ts)
        self._facts_pending_messages = []
        self._facts_pending_keys = set()
        pending_raw = state.get("pending_messages", [])
        if isinstance(pending_raw, list):
            for item in pending_raw:
                chat = self._deserialize_inbound_chat_state(item)
                if chat is None:
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
                key = self._participant_key(participant.user_id, participant.name)
                self._facts_pending_participants[key] = participant
        pending_since = float(state.get("pending_since_ts", 0.0) or 0.0)
        self._facts_pending_since_ts = pending_since if self._facts_pending_messages else 0.0
        if self._facts_pending_messages and self._facts_pending_since_ts <= 0.0:
            self._facts_pending_since_ts = time.time()

    def recover_pending_from_memory(self, recent_items: list[MemoryItem]) -> None:
        if self._owner is not None and hasattr(self._owner, "_recover_pending_facts_from_memory"):
            self._owner._recover_pending_facts_from_memory()
            return
        if not self._facts_enabled or self._facts_mode != "periodic":
            return
        chats = self._memory_items_to_chats(recent_items)
        if not chats:
            return
        base_participants = list(self._facts_pending_participants.values())
        participants = self._context_builder.participants_from_messages(chats, base_participants)
        self._enqueue_fact_messages(chats, participants)

    def maybe_schedule_periodic_sweep(
        self,
        recent_chats: list[InboundChat],
        overflow_chats: list[InboundChat],
        participants: list[Participant],
    ) -> None:
        if self._owner is not None and hasattr(self._owner, "_maybe_schedule_fact_sweep"):
            self._owner._maybe_schedule_fact_sweep(recent_chats, overflow_chats, participants)
            return
        if not self._facts_enabled or self._facts_mode != "periodic":
            return
        facts_participants = self._context_builder.participants_from_messages(recent_chats + overflow_chats, participants)
        self._enqueue_fact_messages(recent_chats, facts_participants)
        self._maybe_start_facts_worker(overflow_present=bool(overflow_chats))

    def schedule_store_facts(
        self,
        facts: list[ExtractedFact],
        participants: list[Participant],
    ) -> None:
        if self._owner is not None and hasattr(self._owner, "_schedule_store_facts"):
            self._owner._schedule_store_facts(facts, participants)
            return
        if not facts:
            return
        asyncio.create_task(asyncio.to_thread(self.store_facts, facts, participants))

    def store_facts(
        self,
        facts: list[ExtractedFact],
        participants: list[Participant],
    ) -> dict[str, int]:
        if self._owner is not None and hasattr(self._owner, "_store_facts"):
            return self._owner._store_facts(facts, participants)

        alias_to_id: dict[str, str] = {}
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
            existing_keys = {
                self._normalize_fact_key(item)
                for item in existing_list
                if self._normalize_fact_key(item)
            }
            deduped_new_facts = self._dedupe_preserve_order(fact.facts)
            missing_facts: list[str] = []
            for item in deduped_new_facts:
                key = self._normalize_fact_key(item)
                if not key or key in existing_keys:
                    continue
                missing_facts.append(item)
                existing_keys.add(key)
            name_changed = False
            if fact.name:
                fact_key = self._normalize_name_for_match(fact.name)
                user_key = self._normalize_name_for_match(user_id)
                if fact_key and user_key and fact_key != user_key:
                    name_changed = bool(not profile or fact.name != profile.name)
            if not missing_facts and not name_changed:
                continue
            name_to_store = profile.name if profile and profile.name else (fact.name or user_id)
            self._knowledge_store.upsert_person_facts(user_id, name_to_store, missing_facts)
            fact_strings_stored += len(missing_facts)
            people_updated += 1
        return {"fact_strings_stored": fact_strings_stored, "people_updated": people_updated}

    async def wait_for_idle(self) -> None:
        if self._owner is not None:
            task = getattr(self._owner, "_facts_task", None)
        else:
            task = self._facts_task
        if task is None:
            return
        try:
            await task
        except Exception:
            return

    def _enqueue_fact_messages(self, messages: list[InboundChat], participants: list[Participant]) -> None:
        now = time.time()
        appended = False
        for message in messages:
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
            key = self._participant_key(participant.user_id, participant.name)
            self._facts_pending_participants[key] = participant
        max_pending = max(self._facts_evidence_max_messages * 3, self._facts_evidence_max_messages)
        if len(self._facts_pending_messages) > max_pending:
            self._facts_pending_messages = self._facts_pending_messages[-max_pending:]
            self._facts_pending_keys = {self._fact_message_key(item) for item in self._facts_pending_messages}
            self._facts_pending_since_ts = now if self._facts_pending_messages else 0.0

    def _fact_message_key(self, message: InboundChat) -> str:
        timestamp = float(message.timestamp or 0.0)
        sender_key = str(message.sender_id or "").strip()
        if not sender_key:
            sender_key = f"name:{self._normalize_name_for_match(str(message.sender_name or ''))}"
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
        self._facts_pending_participants = {}
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
        pending_messages: list[InboundChat],
        pending_participants: list[Participant],
        flush_reason: str,
        pending_before: int,
        pending_age_seconds: float,
    ) -> None:
        remaining = list(pending_messages)
        chunk_index = 0
        while remaining:
            chunk = remaining[: self._facts_evidence_max_messages]
            remaining = remaining[self._facts_evidence_max_messages :]
            participants = self._context_builder.participants_from_messages(chunk, pending_participants)
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

    def _extract_and_store_facts(self, messages: list[InboundChat], participants: list[Participant]) -> dict[str, Any]:
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
        context = self._context_builder.build_facts_context(
            participants,
            messages,
            agent_state=self._autonomy_manager.agent_state(),
        )
        facts = self._llm.extract_facts_from_evidence_sync(context, messages, participants)
        reasoning_trace = self._llm.consume_reasoning_trace("facts")
        fact_strings_extracted = sum(len(fact.facts) for fact in facts)
        stored_stats = {"fact_strings_stored": 0, "people_updated": 0}
        if facts:
            stored_stats = self.store_facts(facts, participants)
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

    @staticmethod
    def _serialize_inbound_chat_state(chat: InboundChat) -> dict[str, Any]:
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

    @staticmethod
    def _memory_items_to_chats(items: list[MemoryItem]) -> list[InboundChat]:
        chats: list[InboundChat] = []
        for item in items:
            chats.append(
                InboundChat(
                    text=str(item.text or ""),
                    sender_id=str(item.sender_id or ""),
                    sender_name=str(item.sender_name or ""),
                    timestamp=float(item.timestamp or 0.0),
                    raw={},
                )
            )
        return chats

    @staticmethod
    def _participant_key(user_id: str, name: str) -> str:
        if user_id:
            return f"id:{user_id}"
        return f"name:{name.casefold()}"

    @staticmethod
    def _normalize_name_for_match(name: str) -> str:
        cleaned = " ".join(str(name or "").casefold().split())
        return cleaned

    @staticmethod
    def _normalize_fact_key(text: str) -> str:
        if not text:
            return ""
        normalized = "".join(ch if ch.isalnum() else " " for ch in str(text).casefold())
        return " ".join(normalized.split())

    @staticmethod
    def _dedupe_preserve_order(items: list[str]) -> list[str]:
        seen: set[str] = set()
        deduped: list[str] = []
        for item in items:
            key = FactManager._normalize_fact_key(item)
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped
