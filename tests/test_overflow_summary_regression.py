"""Regression tests for overflow-summary coverage and overflow-queue eviction.

Covers two data-loss bugs:

* Overflow summarization that did not advance (empty, unchanged, or raising) used to
  still advance the summary timestamp watermark, after which
  ``_filter_overflow_against_summary`` discarded those messages as "already summarized".
* The overflow queue cap used to keep the NEWEST items, destroying the oldest
  un-summarized messages that sit directly after the summary watermark.
"""

from __future__ import annotations

import asyncio

from gpt5_roleplay_system.config import EpisodeConfig, FactsConfig
from gpt5_roleplay_system.llm import LLMClient, LLMResponse, LLMResponseBundle
from gpt5_roleplay_system.memory import (
    OVERFLOW_WARN_THRESHOLD,
    ConversationMemory,
    ExperienceStore,
    RollingBuffer,
    SimpleMemoryCompressor,
)
from gpt5_roleplay_system.models import Action, CommandType, InboundChat
from gpt5_roleplay_system.neo4j_store import InMemoryKnowledgeStore
from gpt5_roleplay_system.observability import NoOpTracer
from gpt5_roleplay_system.pipeline import MessagePipeline


class _OverflowSummaryLLM(LLMClient):
    """Minimal client whose summarize_overflow behaviour is injectable."""

    def __init__(self, behaviour: str) -> None:
        self._behaviour = behaviour
        self.calls = 0

    async def is_addressed_to_me(self, chat, persona, environment=None, participants=None, context=None) -> bool:
        return True

    async def generate_response(self, chat, context) -> LLMResponse:  # pragma: no cover - unused
        raise NotImplementedError

    async def extract_facts(self, chat, context):  # pragma: no cover - unused
        return []

    async def summarize_overflow(self, summary, messages) -> str:
        self.calls += 1
        if self._behaviour == "echo":
            return summary          # legacy failure shape: old summary echoed back
        if self._behaviour == "empty":
            return ""
        if self._behaviour == "raise":
            raise RuntimeError("provider 500")
        return "ADVANCED SUMMARY"

    async def generate_bundle(self, chat, context, overflow=None, incoming_batch=None) -> LLMResponseBundle:
        action = Action(command_type=CommandType.CHAT, content="ok", parameters={"content": "ok"})
        return LLMResponseBundle(text="ok", actions=[action], facts=[], participant_hints=[])


def _make_pipeline(llm, memory):
    return MessagePipeline(
        persona="TestPersona",
        user_id="ai-1",
        llm=llm,
        knowledge_store=InMemoryKnowledgeStore(),
        memory=memory,
        rolling_buffer=RollingBuffer(max_items=100),
        experience_store=ExperienceStore(),
        tracer=NoOpTracer(),
        summary_strategy="llm",
        facts_config=FactsConfig(enabled=False),
        episode_config=EpisodeConfig(enabled=False),
    )


def _chat(text, ts):
    return InboundChat(text=text, sender_id="user-1", sender_name="User", timestamp=ts, raw={})


def _run_turn_with_stalled_summary(behaviour):
    memory = ConversationMemory(SimpleMemoryCompressor(), max_recent=2, defer_compression=True)
    llm = _OverflowSummaryLLM(behaviour)
    pipeline = _make_pipeline(llm, memory)

    memory.apply_summary("PRIOR SUMMARY", timestamps=[1000.0])
    for i in range(6):
        memory.add_message(_chat(f"IMPORTANT-{i}", 2000.0 + i))

    asyncio.run(
        pipeline.process_chat_batch(
            [{"text": "hi", "from_id": "user-1", "from_name": "User", "timestamp": 3000.0}]
        )
    )
    return memory, llm


def test_echoed_summary_does_not_advance_watermark_or_drop_messages():
    memory, llm = _run_turn_with_stalled_summary("echo")

    assert llm.calls >= 1
    assert memory.summary() == "PRIOR SUMMARY"
    # Watermark must not move past the messages the summary never described.
    assert float(memory.summary_meta().get("range_end_ts", 0.0)) <= 1000.0

    alive = {item.text for item in memory.drain_overflow()} | {item.text for item in memory.recent()}
    assert {f"IMPORTANT-{i}" for i in range(6)} <= alive


def test_empty_summary_requeues_overflow():
    memory, _ = _run_turn_with_stalled_summary("empty")

    assert memory.summary() == "PRIOR SUMMARY"
    assert float(memory.summary_meta().get("range_end_ts", 0.0)) <= 1000.0
    alive = {item.text for item in memory.drain_overflow()} | {item.text for item in memory.recent()}
    assert {f"IMPORTANT-{i}" for i in range(6)} <= alive


def test_raising_summary_requeues_overflow_instead_of_propagating():
    memory, _ = _run_turn_with_stalled_summary("raise")

    assert memory.summary() == "PRIOR SUMMARY"
    alive = {item.text for item in memory.drain_overflow()} | {item.text for item in memory.recent()}
    assert {f"IMPORTANT-{i}" for i in range(6)} <= alive


def test_advancing_summary_still_applies_and_moves_watermark():
    memory, llm = _run_turn_with_stalled_summary("advance")

    assert llm.calls >= 1
    assert memory.summary() == "ADVANCED SUMMARY"
    # A genuine summary is allowed to claim coverage of the drained messages.
    assert float(memory.summary_meta().get("range_end_ts", 0.0)) > 2000.0


def test_overflow_is_never_truncated_and_retains_full_history():
    """Overflow must grow without bound; it is the only copy of un-summarized history."""
    memory = ConversationMemory(SimpleMemoryCompressor(), max_recent=2)
    total = OVERFLOW_WARN_THRESHOLD + 200
    for i in range(total):
        memory.add_message(_chat(f"m{i}", 1000.0 + i))

    drained = memory.drain_overflow()
    alive = [item.text for item in drained] + [item.text for item in memory.recent()]
    # Nothing may be discarded, and order must stay chronological.
    assert alive == [f"m{i}" for i in range(total)]


def test_overflow_grows_during_outage_and_drains_after_recovery():
    """An LLM outage must not cost history, and recovery must clear the backlog."""
    memory = ConversationMemory(SimpleMemoryCompressor(), max_recent=2, defer_compression=True)
    llm = _OverflowSummaryLLM("raise")
    pipeline = _make_pipeline(llm, memory)

    total = OVERFLOW_WARN_THRESHOLD + 120
    for i in range(total):
        memory.add_message(_chat(f"m{i}", 1000.0 + i))
        asyncio.run(
            pipeline.process_chat_batch(
                [{"text": f"in{i}", "from_id": "user-1", "from_name": "User", "timestamp": 5000.0 + i}]
            )
        )

    # Outage: every original message is still retained, none discarded.
    pending = {item.text for item in memory._overflow}
    assert {f"m{i}" for i in range(total)} <= pending
    assert len(memory._overflow) > OVERFLOW_WARN_THRESHOLD

    # Recovery: the provider starts working and the accumulated backlog is summarized.
    llm._behaviour = "advance"
    asyncio.run(
        pipeline.process_chat_batch(
            [{"text": "recovered", "from_id": "user-1", "from_name": "User", "timestamp": 9000.0}]
        )
    )
    assert memory.summary() == "ADVANCED SUMMARY"
    # The whole backlog is drained. (A couple of messages from this very turn may have
    # just overflowed again, which is normal steady-state behaviour.)
    remaining = {item.text for item in memory._overflow}
    assert not any(name.startswith("m") and name[1:].isdigit() for name in remaining)
    assert len(memory._overflow) < OVERFLOW_WARN_THRESHOLD


def test_overflow_retained_while_replies_disabled():
    """Low-cost ingest mode (llm_chat_enabled=False) must also retain everything."""
    memory = ConversationMemory(SimpleMemoryCompressor(), max_recent=2, defer_compression=True)
    pipeline = _make_pipeline(_OverflowSummaryLLM("raise"), memory)
    pipeline.set_llm_chat_enabled(False)

    total = OVERFLOW_WARN_THRESHOLD + 120
    for i in range(total):
        asyncio.run(
            pipeline.process_chat_batch(
                [{"text": f"msg{i}", "from_id": "user-1", "from_name": "User", "timestamp": 1000.0 + i}]
            )
        )

    alive = {item.text for item in memory._overflow} | {item.text for item in memory.recent()}
    assert {f"msg{i}" for i in range(total)} <= alive

