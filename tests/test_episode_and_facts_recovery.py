"""Regression tests for two background-work data-loss bugs.

D: episode summarization awaits an LLM call, then trimmed the *live* rolling buffer by
   position -- discarding messages that arrived during the await and were never
   summarized.

F: the fact sweep removed evidence from the pending queue *before* calling the LLM, so
   a provider error destroyed that evidence permanently.
"""

from __future__ import annotations

import asyncio

from gpt5_roleplay_system.config import EpisodeConfig, FactsConfig
from gpt5_roleplay_system.llm import ExtractedFact, LLMClient, LLMResponseBundle
from gpt5_roleplay_system.memory import (
    ConversationMemory,
    ExperienceStore,
    RollingBuffer,
    SimpleMemoryCompressor,
)
from gpt5_roleplay_system.models import Action, CommandType, InboundChat, Participant
from gpt5_roleplay_system.neo4j_store import InMemoryKnowledgeStore
from gpt5_roleplay_system.observability import NoOpTracer
from gpt5_roleplay_system.pipeline import MessagePipeline


def _chat(text, ts, sid="u1", name="User"):
    return InboundChat(text=text, sender_id=sid, sender_name=name, timestamp=ts, raw={})


class _SlowEpisodeLLM(LLMClient):
    """Summarization takes time, letting new messages land mid-flight."""

    def __init__(self):
        self.summarized: list[str] = []

    async def is_addressed_to_me(self, *a, **k):
        return True

    async def generate_response(self, chat, context):
        raise NotImplementedError

    async def extract_facts(self, chat, context):
        return []

    async def summarize_overflow(self, summary, messages):
        return "S"

    async def summarize_episode(self, messages):
        self.summarized = [m.text for m in messages]
        await asyncio.sleep(0.05)
        return "EPISODE SUMMARY"

    async def generate_bundle(self, chat, context, overflow=None, incoming_batch=None):
        action = Action(command_type=CommandType.CHAT, content="ok", parameters={"content": "ok"})
        return LLMResponseBundle(text="ok", actions=[action], facts=[], participant_hints=[])


class _FailingFactsLLM(_SlowEpisodeLLM):
    def __init__(self, fail_times=1):
        super().__init__()
        self.calls = 0
        self.fail_times = fail_times

    def extract_facts_from_evidence_sync(self, context, evidence_messages, participants):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("provider 500")
        return [ExtractedFact(user_id="u1", name="User", facts=["likes lakes"])]

    def consume_reasoning_trace(self, label):
        return None

    def consume_quality_fallback_events(self):
        return []


def _pipeline(llm, buf=None, facts=None, episode=None):
    return MessagePipeline(
        persona="P",
        user_id="ai-1",
        llm=llm,
        knowledge_store=InMemoryKnowledgeStore(),
        memory=ConversationMemory(SimpleMemoryCompressor(), max_recent=20, defer_compression=True),
        rolling_buffer=buf or RollingBuffer(max_items=100),
        experience_store=ExperienceStore(),
        tracer=NoOpTracer(),
        facts_config=facts or FactsConfig(enabled=False),
        episode_config=episode or EpisodeConfig(enabled=False),
    )


# ----------------------------------------------------------------- D
def test_messages_arriving_during_episode_summarization_are_not_lost():
    async def main():
        buf = RollingBuffer(max_items=100)
        llm = _SlowEpisodeLLM()
        pipe = _pipeline(
            llm,
            buf=buf,
            episode=EpisodeConfig(
                enabled=True, min_messages=2, max_messages=5,
                inactivity_seconds=0, forced_interval_seconds=0, overlap_messages=1,
            ),
        )
        for i in range(5):
            buf.add_user_message(_chat(f"old{i}", 1000.0 + i))

        task = asyncio.create_task(pipe._maybe_finalize_episode())
        await asyncio.sleep(0.01)          # episode is awaiting the LLM
        for i in range(3):                 # new traffic lands mid-summarization
            buf.add_user_message(_chat(f"NEW{i}", 2000.0 + i))
        await task

        summarized = set(llm.summarized)
        survived = {item.text for item in buf.items()}
        # Every new message must still be in the buffer: none were summarized.
        assert {"NEW0", "NEW1", "NEW2"} <= survived
        assert not ({"NEW0", "NEW1", "NEW2"} & summarized)
        # And the summarized batch was trimmed apart from the overlap.
        assert len(survived & {f"old{i}" for i in range(5)}) == 1

    asyncio.run(main())


def test_episode_watermark_matches_retained_overlap():
    async def main():
        buf = RollingBuffer(max_items=100)
        pipe = _pipeline(
            _SlowEpisodeLLM(),
            buf=buf,
            episode=EpisodeConfig(
                enabled=True, min_messages=2, max_messages=4,
                inactivity_seconds=0, forced_interval_seconds=0, overlap_messages=2,
            ),
        )
        for i in range(4):
            buf.add_user_message(_chat(f"m{i}", 1000.0 + i))
        await pipe._maybe_finalize_episode()
        # Watermark must equal what was actually retained, or the next episode's
        # "new_items" arithmetic is wrong.
        assert pipe._last_episode_size == len(buf.items())

    asyncio.run(main())


def test_normal_episode_still_trims_and_stores():
    async def main():
        buf = RollingBuffer(max_items=100)
        pipe = _pipeline(
            _SlowEpisodeLLM(),
            buf=buf,
            episode=EpisodeConfig(
                enabled=True, min_messages=2, max_messages=4,
                inactivity_seconds=0, forced_interval_seconds=0, overlap_messages=1,
            ),
        )
        for i in range(4):
            buf.add_user_message(_chat(f"m{i}", 1000.0 + i))
        await pipe._maybe_finalize_episode()
        assert len(pipe._experience_store.all()) == 1
        assert [i.text for i in buf.items()] == ["m3"]

    asyncio.run(main())


# ----------------------------------------------------------------- F
def test_failed_fact_sweep_requeues_evidence():
    async def main():
        llm = _FailingFactsLLM(fail_times=1)
        pipe = _pipeline(
            llm,
            facts=FactsConfig(enabled=True, mode="periodic", min_pending_messages=1),
        )
        pipe._enqueue_fact_messages(
            [_chat("I am allergic to peanuts", 1000.0)],
            [Participant(user_id="u1", name="User")],
        )
        assert len(pipe._facts_pending_messages) == 1

        pipe._maybe_start_facts_worker()
        await pipe._facts_task

        # Evidence must survive the provider error.
        texts = [m.text for m in pipe._facts_pending_messages]
        assert texts == ["I am allergic to peanuts"]
        assert pipe._facts_pending_participants  # participants preserved for the retry

    asyncio.run(main())


def test_requeued_evidence_is_retried_and_stored_after_recovery():
    async def main():
        llm = _FailingFactsLLM(fail_times=1)
        pipe = _pipeline(
            llm,
            facts=FactsConfig(enabled=True, mode="periodic", min_pending_messages=1),
        )
        pipe._enqueue_fact_messages(
            [_chat("I am allergic to peanuts", 1000.0)],
            [Participant(user_id="u1", name="User")],
        )
        pipe._maybe_start_facts_worker()
        await pipe._facts_task
        assert pipe._facts_pending_messages  # still queued after failure

        pipe._facts_retry_not_before_ts = 0.0   # skip the cooldown
        pipe._maybe_start_facts_worker()
        await pipe._facts_task

        assert pipe._facts_pending_messages == []
        profile = pipe._knowledge_store.fetch_people(["u1"]).get("u1")
        assert profile is not None and "likes lakes" in profile.facts

    asyncio.run(main())


def test_failed_sweep_sets_retry_cooldown():
    """A persistently failing provider must not trigger an LLM call per message."""
    async def main():
        llm = _FailingFactsLLM(fail_times=99)
        pipe = _pipeline(
            llm,
            facts=FactsConfig(enabled=True, mode="periodic", min_pending_messages=1),
        )
        pipe._enqueue_fact_messages(
            [_chat("evidence", 1000.0)], [Participant(user_id="u1", name="User")]
        )
        pipe._maybe_start_facts_worker()
        await pipe._facts_task
        assert pipe._facts_retry_not_before_ts > 0.0

        before = llm.calls
        for _ in range(10):
            pipe._maybe_start_facts_worker()
            if pipe._facts_task:
                await pipe._facts_task
        assert llm.calls == before  # cooldown suppressed every retry

    asyncio.run(main())


def test_successful_sweep_leaves_queue_empty():
    async def main():
        llm = _FailingFactsLLM(fail_times=0)
        pipe = _pipeline(
            llm,
            facts=FactsConfig(enabled=True, mode="periodic", min_pending_messages=1),
        )
        pipe._enqueue_fact_messages(
            [_chat("evidence", 1000.0)], [Participant(user_id="u1", name="User")]
        )
        pipe._maybe_start_facts_worker()
        await pipe._facts_task
        assert pipe._facts_pending_messages == []
        assert pipe._facts_task is None

    asyncio.run(main())
