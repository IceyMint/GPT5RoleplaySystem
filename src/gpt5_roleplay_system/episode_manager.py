from __future__ import annotations

import asyncio
import time
from typing import Any, Protocol

from .config import EpisodeConfig
from .llm import LLMClient
from .memory import ExperienceRecord, ExperienceStore, MemoryCompressor, RollingBuffer, SimpleMemoryCompressor
from .models import InboundChat
from .pipeline_state import PipelineRuntimeState
from .time_utils import format_pacific_time
from .observability import Tracer


class ExperienceIndexProtocol(Protocol):
    def is_enabled(self) -> bool:
        ...

    async def add_record_async(self, record: ExperienceRecord, persona_id: str) -> None:
        ...


class EpisodeManager:
    def __init__(
        self,
        state: PipelineRuntimeState,
        llm: LLMClient,
        rolling_buffer: RollingBuffer,
        experience_store: ExperienceStore,
        tracer: Tracer,
        episode_config: EpisodeConfig,
        experience_vector_index: ExperienceIndexProtocol | None = None,
        compressor: MemoryCompressor | None = None,
    ) -> None:
        self._state = state
        self._llm = llm
        self._rolling_buffer = rolling_buffer
        self._experience_store = experience_store
        self._tracer = tracer
        self._experience_vector_index = experience_vector_index
        self._compressor = compressor or SimpleMemoryCompressor()

        self._episode_enabled = bool(episode_config.enabled)
        self._episode_min_messages = max(1, int(episode_config.min_messages))
        self._episode_max_messages = max(self._episode_min_messages, int(episode_config.max_messages))
        self._episode_inactivity_seconds = max(0.0, float(episode_config.inactivity_seconds))
        self._episode_forced_interval_seconds = max(0.0, float(episode_config.forced_interval_seconds))
        self._episode_overlap_messages = max(0, int(episode_config.overlap_messages))

        self._last_episode_ts = 0.0
        self._last_episode_size = 0
        self._episode_task: asyncio.Task | None = None

    def snapshot_state(self) -> dict[str, Any]:
        return {
            "last_episode_ts": self._last_episode_ts,
            "last_episode_size": self._last_episode_size,
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        self._last_episode_ts = float(state.get("last_episode_ts", 0.0) or 0.0)
        self._last_episode_size = int(state.get("last_episode_size", 0) or 0)

    def schedule_check(
        self,
        *,
        last_inbound_ts: float,
        last_response_ts: float,
        persona: str,
    ) -> None:
        if not self._episode_enabled:
            return
        if self._episode_task and not self._episode_task.done():
            return
        self._episode_task = asyncio.create_task(
            self.maybe_finalize(
                last_inbound_ts=last_inbound_ts,
                last_response_ts=last_response_ts,
                persona=persona,
            )
        )

    async def maybe_finalize(
        self,
        *,
        last_inbound_ts: float,
        last_response_ts: float,
        persona: str,
    ) -> bool:
        items = self._rolling_buffer.items()
        count = len(items)
        if count < self._episode_min_messages:
            return False

        now = time.time()
        last_activity = max(float(last_inbound_ts or 0.0), float(last_response_ts or 0.0))
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
            return False

        snapshot = list(items)
        summary = await self._summarize_episode(snapshot)
        if not summary:
            return False

        metadata = self._episode_metadata(snapshot, trigger_reason)
        record = self._experience_store.add(summary, metadata, persona_id=persona)
        if self._experience_vector_index and self._experience_vector_index.is_enabled():
            asyncio.create_task(self._experience_vector_index.add_record_async(record, persona))

        overlap = min(self._episode_overlap_messages, count)
        self._rolling_buffer.trim_to_last(overlap)
        self._last_episode_ts = now
        self._last_episode_size = overlap
        self._tracer.log_event(
            "episode_summary",
            {"reason": trigger_reason, "messages": count, "overlap": overlap},
        )
        return True

    async def wait_for_idle(self) -> None:
        task = self._episode_task
        if task is None:
            return
        try:
            await task
        except Exception:
            return

    async def _summarize_episode(self, items: list[Any]) -> str:
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
        import logging
        logger = logging.getLogger(__name__)
        try:
            summary = await self._llm.summarize_episode(chats)
            if summary:
                return summary.strip()
            return ""
        except Exception as e:
            logger.error(f"Error during episode summarization: {e}", exc_info=True)
            return ""

    @staticmethod
    def _episode_metadata(items: list[Any], reason: str) -> dict[str, Any]:
        timestamps = [float(getattr(item, "timestamp", 0.0) or 0.0) for item in items]
        sender_names: list[str] = []
        seen: set[str] = set()
        for item in items:
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
